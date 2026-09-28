"""
Pilot A - model/regression.py

loader.py가 만든 PilotABatch에서
E, PGV, z_wood, event_idx를 받아
잠재상태 4개(00, 10, 01, 11) 각각에 대해
피해 6채널의 기대피해 건수 mu를 계산한다.

PilotABatch 실제 데이터 형태:

필드                    실제 shape     dtype           의미
y                       [418, 6]       torch.float64   피해 6채널 관측값
E                       [418, 6]       torch.float64   피해 6채널 Exposure
pgv                     [418]          torch.float64   각 시정촌×이벤트 행의 PGV
z_wood                  [418]          torch.float64   표준화된 목조주택 비율
z_mtn                   [418]          torch.float64   표준화된 산지 비율 (이 브랜치에서는 쓰지 않음)
pi_ls                   [418]          torch.float64   LS prior 평균
pi_lq                   [418]          torch.float64   LQ prior 평균
event_idx               [418]          torch.int64     9개 지진 이벤트 중 어떤 이벤트인지
obs_mask                [418, 6]       torch.bool      각 피해 채널이 실제 관측됐는지
municipality_code       길이 418       tuple[str,...]  "01581" 같은 5자리 시정촌 ID


설계식:

    log(lambda_ice)
        = alpha_c
        + alpha_e
        + beta_c * log(PGV_ie)
        + delta_c * z_wood_i
        + gamma_c^LS * LS
        + gamma_c^LQ * LQ

    lambda_ice = exp(log(lambda_ice))
    mu_ice = E_ice * lambda_ice


입력 shape:

    E           [B, 6]
    pgv         [B]
    z_wood      [B]
    event_idx   [B]


출력 shape:

    log_lambda      [B, 4, 6]
    lambda_rate     [B, 4, 6]
    mu              [B, 4, 6]


실제 아쓰마초 한 행 예:

    E = [4838, 4838, 4838, 2121, 2121, 2121]
    pgv = 44.64
    event_idx = 5

이 한 행이 00/10/01/11 네 상태로 확장되므로
최종 mu의 한 행 shape은 [4, 6]이다.

실제 mu 숫자는 alpha, beta, delta, eta, gamma의
학습값에 따라 달라지므로 코드에 고정하지 않는다.
"""


from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from schema import (
    DTYPE,
    EPS,
    LATENT_STATES,
    NUM_CHANNELS,
    NUM_EVENTS,
    NUM_STATES,
    PilotABatch,
)


@dataclass
class RegressionOutput:
    """
    regression.py의 계산 결과를 한 객체로 묶는다.

    출력 필드:

    log_lambda
        [B, 4, 6]
        피해율 lambda에 log를 취한 값

    lambda_rate
        [B, 4, 6]
        원래 scale의 피해율 lambda

    mu
        [B, 4, 6]
        Negative Binomial likelihood에 넘길 기대피해 건수

    실제 시정촌 한 행은 각 필드에서 [4, 6] 부분을 가진다.

    4:
        00, 10, 01, 11

    6:
        사망, 중상, 경상, 전파, 반파, 일부파손
    """

    log_lambda: Tensor
    lambda_rate: Tensor
    mu: Tensor


class DamageRegression(nn.Module):
    """
    PGV, E, 이벤트 효과, 취약성 공변량,
    LS/LQ 잠재상태를 이용해 mu를 계산하는 회귀모형.

    학습 파라미터:

    alpha_channel [6]
        피해 채널별 기본 log 피해율

    alpha_event [9]
        지진 이벤트별 공통 효과
        기준 이벤트 1개는 0으로 고정

    alpha_event_free [8]
        기준 이벤트를 제외한 실제 자유 파라미터

    beta_pgv [6]
        채널별 log(PGV) 효과

    delta_wood [6]
        채널별 표준화 목조주택 비율(z_wood) 효과

    _gamma_ls_unconstrained [6]
        softplus 적용 전 LS 효과 내부 파라미터

    _gamma_lq_unconstrained [6]
        softplus 적용 전 LQ 효과 내부 파라미터


    제약:

    delta_wood
        부호 제약 없이 자유롭게 학습한다.

    gamma_ls / gamma_lq
        softplus를 거쳐 항상 0 이상이 된다.
    """

    def __init__(
        self,
        *,
        reference_event_idx: int = 0,
        gamma_init: float = 0.1,
        mtn_covariate: bool = False,
    ) -> None:
        """
        회귀모형 파라미터를 생성한다.

        reference_event_idx:
            alpha_e=0으로 고정할 기준 이벤트 index

        gamma_init:
            softplus 적용 후 실제 gamma 초기값
            기본값 0.1


        생성 직후:

        alpha_channel
            임시 0
            train 시작 전에 initialize_from_batch(batch)로
            실제 관측 피해율을 이용해 다시 초기화

        alpha_event
            모든 이벤트 효과 0

        beta_pgv
            모두 0

        delta_wood
            모두 0

        gamma_ls
            실제 값 0.1

        gamma_lq
            실제 값 0.1
        """

        super().__init__()

        if not 0 <= reference_event_idx < NUM_EVENTS:
            raise ValueError(
                f"reference_event_idx는 "
                f"0~{NUM_EVENTS - 1} 범위여야 합니다."
            )

        if gamma_init <= 0:
            raise ValueError(
                "gamma_init은 0보다 커야 합니다."
            )

        self.reference_event_idx = int(reference_event_idx)
        self.gamma_init = float(gamma_init)

        # ----------------------------------------------------
        # alpha_c
        # ----------------------------------------------------

        # alpha_c [6]
        #
        # 채널별 기본 log 피해율.
        # 실제 batch로 다시 초기화할 것이므로 우선 0으로 생성한다.
        self.alpha_channel = nn.Parameter(
            torch.zeros(
                NUM_CHANNELS,
                dtype=DTYPE,
            )
        )

        # ----------------------------------------------------
        # alpha_e
        # ----------------------------------------------------

        # 이벤트는 총 9개지만 기준 이벤트 하나는 alpha_e=0으로 고정한다.
        #
        # 따라서 optimizer가 실제로 움직이는 자유 파라미터는
        # NUM_EVENTS - 1 = 8개다.
        self.alpha_event_free = nn.Parameter(
            torch.zeros(
                NUM_EVENTS - 1,
                dtype=DTYPE,
            )
        )

        # ----------------------------------------------------
        # beta_c
        # ----------------------------------------------------

        # beta_c [6]
        #
        # log(PGV)가 피해 6채널 각각에 미치는 효과.
        self.beta_pgv = nn.Parameter(
            torch.zeros(
                NUM_CHANNELS,
                dtype=DTYPE,
            )
        )

        # ----------------------------------------------------
        # delta_c
        # ----------------------------------------------------

        # delta_c [6]
        #
        # 표준화된 목조주택 비율(z_wood)이
        # 피해 6채널 각각에 미치는 효과.
        #
        # 후속실험 1의 목적은
        # 지역 자체 취약성이 설명할 편차를
        # 회귀식이 충분히 흡수하도록 하는 것이다.
        #
        # 따라서 gamma처럼 양수 제약을 걸지 않고
        # 0에서 시작해 자유롭게 학습한다.
        self.delta_wood = nn.Parameter(
            torch.zeros(
                NUM_CHANNELS,
                dtype=DTYPE,
            )
        )

        # 후속실험 5 비교 조건: 산지 비율을 피해 회귀식에도 통제 변수로 넣는다.
        # "산지가 많은 곳 = 시골 = 인구 대비 피해가 적다"는 효과를 eta_mtn이 흡수해,
        # LS prior의 kappa가 그 효과를 산사태 쪽으로 잘못 가져가지 않게 한다.
        # 기본값(False)이면 파라미터 자체가 없어 기존과 같다.
        self.mtn_covariate = bool(mtn_covariate)
        if self.mtn_covariate:
            self.eta_mtn = nn.Parameter(
                torch.zeros(
                    NUM_CHANNELS,
                    dtype=DTYPE,
                )
            )

        # ----------------------------------------------------
        # gamma
        # ----------------------------------------------------

        # 실제 gamma = gamma_init이
        # softplus 적용 후 정확히 나오도록
        # 내부 unconstrained 초기값을 계산한다.
        gamma_unconstrained_init = _inverse_softplus(
            self.gamma_init
        )

        self._gamma_ls_unconstrained = nn.Parameter(
            torch.full(
                (NUM_CHANNELS,),
                gamma_unconstrained_init,
                dtype=DTYPE,
            )
        )

        self._gamma_lq_unconstrained = nn.Parameter(
            torch.full(
                (NUM_CHANNELS,),
                gamma_unconstrained_init,
                dtype=DTYPE,
            )
        )

    @property
    def alpha_event(self) -> Tensor:
        """
        기준 이벤트가 0으로 고정된
        전체 alpha_e [9]를 생성한다.

        예:

        reference_event_idx=0이면

            [
                0,
                free_0,
                free_1,
                ...
                free_7
            ]

        reference_event_idx=3이면

            [
                free_0,
                free_1,
                free_2,
                0,
                free_3,
                ...
                free_7
            ]

        기준 이벤트의 0은
        "효과가 없다"는 의미가 아니라
        다른 이벤트 효과를 비교하기 위한 기준점이다.
        """

        zero = self.alpha_event_free.new_zeros(1)

        return torch.cat(
            [
                self.alpha_event_free[
                    : self.reference_event_idx
                ],
                zero,
                self.alpha_event_free[
                    self.reference_event_idx :
                ],
            ],
            dim=0,
        )

    @property
    def gamma_ls(self) -> Tensor:
        """
        gamma_c^LS [6]을 반환한다.

        내부 unconstrained 값에 softplus를 적용하므로
        항상 0 이상이다.

        초기 실제 gamma 값은 모두 gamma_init,
        기본값은 0.1이다.
        """

        return F.softplus(
            self._gamma_ls_unconstrained
        )

    @property
    def gamma_lq(self) -> Tensor:
        """
        gamma_c^LQ [6]을 반환한다.

        내부 unconstrained 값에 softplus를 적용하므로
        항상 0 이상이다.

        초기 실제 gamma 값은 모두 gamma_init,
        기본값은 0.1이다.
        """

        return F.softplus(
            self._gamma_lq_unconstrained
        )

    @torch.no_grad()
    def initialize_from_batch(
        self,
        batch: PilotABatch,
    ) -> Tensor:
        """
        실제 batch를 이용해 회귀 파라미터를 초기화한다.

        입력:

        batch.y
            [B, 6]

        batch.E
            [B, 6]

        batch.obs_mask
            [B, 6]


        초기화 규칙:

        1.
            alpha_c
            = log(sum(y_c) / sum(E_c))

        2.
            alpha_e = 모두 0
            기준 이벤트는 구조적으로 계속 0

        3.
            beta_c = 0

        4.
            delta_c = 0

        5.
            gamma_c^LS = gamma_init
            기본값 0.1

        6.
            gamma_c^LQ = gamma_init
            기본값 0.1


        결측 피해 채널은 obs_mask=False이므로
        alpha_c 초기화의 피해합 / Exposure합에서
        모두 제외된다.
        """

        batch.validate()

        # True=1, False=0으로 바꿔
        # 관측된 채널만 합계에 포함한다.
        observed = batch.obs_mask.to(
            dtype=DTYPE
        )

        # 채널별 실제 관측 피해합
        total_y = (
            batch.y * observed
        ).sum(dim=0)

        # 같은 관측 위치의 Exposure 합
        total_E = (
            batch.E * observed
        ).sum(dim=0)

        if (total_E <= 0).any():
            raise ValueError(
                "alpha_c 초기화용 Exposure 합은 "
                "모든 채널에서 0보다 커야 합니다."
            )

        # 피해합이 0인 채널이 있으면
        # log(0)을 피하기 위해 EPS를 최소값으로 둔다.
        empirical_rate = torch.clamp(
            total_y / total_E,
            min=EPS,
        )

        alpha_init = torch.log(
            empirical_rate
        )

        # ----------------------------------------------------
        # 실제 parameter 객체는 유지하고
        # 초기값만 덮어쓴다.
        # ----------------------------------------------------

        self.alpha_channel.copy_(
            alpha_init
        )

        self.alpha_event_free.zero_()

        self.beta_pgv.zero_()

        self.delta_wood.zero_()

        if self.mtn_covariate:
            self.eta_mtn.zero_()

        # gamma 역시 initialize_from_batch() 호출 시
        # 항상 gamma_init 값으로 다시 초기화되도록 한다.
        gamma_unconstrained_init = _inverse_softplus(
            self.gamma_init
        )

        self._gamma_ls_unconstrained.fill_(
            gamma_unconstrained_init
        )

        self._gamma_lq_unconstrained.fill_(
            gamma_unconstrained_init
        )

        return alpha_init.clone()

    def forward(
        self,
        batch: PilotABatch,
    ) -> RegressionOutput:
        """
        batch의 모든 행을 4개 잠재상태로 확장해
        피해 6채널의 mu를 계산한다.

        실제 아쓰마초 입력 예:

            E
                [4838, 4838, 4838, 2121, 2121, 2121]

            pgv
                44.64

            z_wood
                표준화된 목조주택 비율

            event_idx
                5


        처리:

        1.
            pgv 하나에
            채널별 beta_c [6]이 각각 곱해진다.

        2.
            z_wood에는
            채널별 delta_c [6]이 각각 곱해진다.

        3.
            z_wood 효과는
            LS/LQ 상태와 무관한 지역 자체 공변량이므로
            00/10/01/11 네 상태 모두에 동일하게 들어간다.

        4.
            event_idx=5이면
            alpha_event[5]를 선택한다.

        6.
            00/10/01/11 네 상태 각각에서
            6채널 log_lambda를 계산한다.

        7.
            lambda = exp(log_lambda)

        8.
            mu = E * lambda


        출력:

            시정촌 한 행
                mu [4, 6]

            전체 B행
                mu [B, 4, 6]
        """

        batch.validate()

        B = batch.batch_size

        # ----------------------------------------------------
        # PGV
        # ----------------------------------------------------

        # PGV는 행마다 하나라 [B].
        #
        # [B,1,1]로 바꾸어
        # 상태 4개와 채널 6개에 broadcast한다.
        log_pgv = torch.log(
            batch.pgv
        ).view(
            B,
            1,
            1,
        )

        # ----------------------------------------------------
        # vulnerability covariates
        # ----------------------------------------------------

        # z_wood도 행마다 하나라 [B].
        #
        # 상태 축 크기를 1로 두기 때문에
        # 00/10/01/11 네 상태에
        # 같은 값이 broadcast된다.
        #
        # batch.z_mtn은 그대로 실려 오지만 이 브랜치에서는 쓰지 않는다.
        z_wood = batch.z_wood.view(
            B,
            1,
            1,
        )

        # ----------------------------------------------------
        # event effect
        # ----------------------------------------------------

        # 기준 이벤트 0을 포함한 전체 alpha_e [9]에서
        # 각 행 event_idx에 해당하는 효과를 선택한다.
        event_effect = self.alpha_event[
            batch.event_idx
        ].view(
            B,
            1,
            1,
        )

        # ----------------------------------------------------
        # channel parameters
        # ----------------------------------------------------

        # 각 채널별 파라미터 [6]을 [1,1,6]으로 바꾸어
        # 모든 행과 상태에 적용한다.
        channel_intercept = self.alpha_channel.view(
            1,
            1,
            NUM_CHANNELS,
        )

        pgv_effect = self.beta_pgv.view(
            1,
            1,
            NUM_CHANNELS,
        )

        wood_effect = self.delta_wood.view(
            1,
            1,
            NUM_CHANNELS,
        )

        gamma_ls = self.gamma_ls.view(
            1,
            1,
            NUM_CHANNELS,
        )

        gamma_lq = self.gamma_lq.view(
            1,
            1,
            NUM_CHANNELS,
        )

        # ----------------------------------------------------
        # latent states
        # ----------------------------------------------------

        # schema.py의 고정 잠재상태 순서
        #
        # 00
        # 10
        # 01
        # 11
        #
        # 를 Tensor [4,2]로 만든다.
        states = torch.tensor(
            LATENT_STATES,
            dtype=DTYPE,
            device=batch.E.device,
        )

        # [4] -> [1,4,1]
        ls_state = states[
            :,
            0,
        ].view(
            1,
            NUM_STATES,
            1,
        )

        lq_state = states[
            :,
            1,
        ].view(
            1,
            NUM_STATES,
            1,
        )

        # ----------------------------------------------------
        # log lambda
        # ----------------------------------------------------

        # 로그 선형 회귀식을 벡터화한 결과.
        #
        # shape:
        # [B,4,6]
        log_lambda = (
            channel_intercept
            + event_effect
            + pgv_effect * log_pgv
            + wood_effect * z_wood
            + gamma_ls * ls_state
            + gamma_lq * lq_state
        )

        # 후속실험 5 비교 조건: eta_c * z_mtn_i (네 상태 모두에 같은 값)
        if self.mtn_covariate:
            log_lambda = log_lambda + self.eta_mtn.view(1, 1, NUM_CHANNELS) * batch.z_mtn.view(B, 1, 1)

        # ----------------------------------------------------
        # lambda
        # ----------------------------------------------------

        # log 피해율을 원래 양수 scale로 되돌린다.
        lambda_rate = torch.exp(
            log_lambda
        )

        # ----------------------------------------------------
        # mu
        # ----------------------------------------------------

        # E [B,6]에 상태 축을 추가해 [B,1,6]으로 만든 뒤
        # lambda [B,4,6]과 곱한다.
        #
        # 결과:
        # mu [B,4,6]
        mu = (
            batch.E.unsqueeze(1)
            * lambda_rate
        )

        # ----------------------------------------------------
        # numerical validation
        # ----------------------------------------------------

        if not torch.isfinite(
            log_lambda
        ).all():
            raise ValueError(
                "log_lambda에 NaN 또는 Inf가 발생했습니다."
            )

        if not torch.isfinite(
            lambda_rate
        ).all():
            raise ValueError(
                "lambda_rate에 NaN 또는 Inf가 발생했습니다."
            )

        if not torch.isfinite(
            mu
        ).all():
            raise ValueError(
                "mu에 NaN 또는 Inf가 발생했습니다."
            )

        if (mu <= 0).any():
            raise ValueError(
                "mu는 모든 행/상태/채널에서 "
                "0보다 커야 합니다."
            )

        return RegressionOutput(
            log_lambda=log_lambda,
            lambda_rate=lambda_rate,
            mu=mu,
        )


def _inverse_softplus(
    value: float,
) -> float:
    """
    원하는 실제 gamma 초기값을
    softplus 이전 내부값으로 변환한다.

    예:

        gamma = 0.1

    이면 내부값은 약 -2.252이고,
    softplus를 거치면 다시 약 0.1이 된다.

    이 내부값은 별도의 의미를 가진 모델 파라미터라기보다
    gamma >= 0 제약을 구현하기 위한 내부 표현이다.
    """

    if value <= 0:
        raise ValueError(
            "inverse softplus 입력은 "
            "0보다 커야 합니다."
        )

    return float(
        torch.log(
            torch.expm1(
                torch.tensor(
                    value,
                    dtype=DTYPE,
                )
            )
        ).item()
    )