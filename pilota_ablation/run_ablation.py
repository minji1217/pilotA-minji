"""RQ4 ablation: TRACE(pilotA A2)에서 한 번에 하나만 바꿔 LOEO(지진 하나씩 빼기) 성능을 낸다.

pilotA 모델 코드는 고치지 않는다. 입력(batch)이나 부품(likelihood)만 바꿔 끼운다.
pilotA 코드: minji1217/pilotA 브랜치 followup4-lq-J-nolabels (A2 기본값). 그 폴더를 첫 인자로 준다.

    python run_ablation.py <pilotA 폴더> [--dpm5] [설정 ...]

--dpm5: 기준 학습을 "DPM 있는 5개 지진만"으로 (평가 지진을 빼고 나머지 4개로 학습, 평가는 그 5개 지진만).
        없으면 A2 기본 (평가 지진을 빼고 나머지 8개로 학습, 9개 지진 모두 평가).

설정 (기본: 전부)
  full          TRACE 그대로 (A2: 평가 지진 외 8개로 학습, 산사태 라벨 사용)
  uniform_prior   입력: 사전확률을 모든 시정촌에 같은 값으로 (평균). 재해마다, 학습 시정촌의 원래 사전확률
                  (κ = 0, b = 0일 때)을 평균한 값 하나로 고정. 산사태 · 액상화 똑같이
  uniform_learned 입력: 위 평균값에서 출발해 재해마다 같은 값 하나를 학습
  uniform_half    입력: 두 재해 모두 0.5로 고정
  no_pgv        입력: 피해 회귀식의 흔들림 항 β·log PGV 제거 (PGV = 1로 두어 log PGV = 0)
  poisson       구조: 음이항분포 대신 포아송분포 (과분산 없음)
  no_exposure   구조: 노출 offset 제거 (μ = E·λ 에서 E = 1)
  per_event     학습: 크로스이벤트 대신 예측할 지진 하나로만 학습. 그 지진 정답은 쓸 수 없으므로 라벨 없이

결과: <pilotA 폴더>/outputs/holdout_detail{_lq}_abl-<설정>_holdout-<지진>.csv
채점: score_ablation.py
"""
import contextlib
import dataclasses
import io
import os
import sys
import time

W = os.path.abspath(sys.argv[1])
os.chdir(W)
sys.path.insert(0, W)

import torch  # noqa: E402

import train as T  # noqa: E402
from likelihood import DamageLikelihood, LikelihoodOutput  # noqa: E402
from loader import attach_ls_labels, load_eval_ground_truth, load_pilot_a_batch, subset_batch  # noqa: E402
from schema import EVENT_TO_INDEX, EVENTS  # noqa: E402

torch.set_num_threads(4)
ALL = ["full", "uniform_prior", "uniform_learned", "uniform_half", "no_pgv", "poisson", "no_exposure", "per_event"]
ARGS = sys.argv[2:]
DPM5_ONLY = "--dpm5" in ARGS
SETTINGS = [a for a in ARGS if a != "--dpm5"] or ALL
DPM5 = ["2016 구마모토", "2018 오사카", "2018 훗카이도", "2021 후쿠시마", "2024 노토반도"]
TRAIN_KW = dict(seed=0, epochs=3000, lr=0.02, lam_gamma=10, area_mode="fixed-cov-mtn", lam_kappa=10,
                mtn_standardize="train", use_labels=True, lq_labels=False, lq_b_only=True)


class PriorUniform(T.AreaPrior):
    """균등 prior: 모든 시정촌에 같은 로짓. 산사태 · 액상화를 똑같이 처리한다.
    MODE = "mean"   : 학습 행의 원래 사전확률(κ = 0, b = 0일 때)을 평균한 값으로 고정
           "learned": 위 값에서 출발해, 재해마다 상수 하나를 학습
           "half"   : 0.5 (로짓 0) 고정
    C = (c_LS, c_LQ): "mean" · "learned"의 출발 로짓. 실행 때마다 학습 행으로 정한다."""
    MODE = "mean"
    C = (0.0, 0.0)

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.u_shift = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64), requires_grad=self.MODE == "learned")

    def z(self, pi_ls, *args, **kw):
        zero = torch.zeros_like(pi_ls)
        if self.MODE == "half":
            return zero, zero
        return zero + self.C[0] + self.u_shift[0], zero + self.C[1] + self.u_shift[1]


def uniform_const(batch, t):
    """학습 행의 원래 사전확률(κ = 0, b = 0일 때)의 평균을 로짓으로. 모델과 같은 식(log p̄를 EPS로 자름)."""
    from schema import EPS
    z_ls = torch.log(batch.pi_ls.clamp_min(EPS)) + batch.log_k_ls + batch.log_cov_ls
    z_lq = torch.log(batch.pi_lq.clamp_min(EPS)) + (batch.log_k_lq - batch.log_k_lq[t].mean())
    p_ls, p_lq = torch.sigmoid(z_ls[t]).mean(), torch.sigmoid(z_lq[t]).mean()
    return float(torch.logit(p_ls)), float(torch.logit(p_lq)), float(p_ls), float(p_lq)


class PoissonLikelihood(DamageLikelihood):
    """음이항 대신 포아송 log-PMF. 출력 형식(LikelihoodOutput)은 그대로."""

    def forward(self, batch, mu):
        y = batch.y.unsqueeze(1)
        log_p = y * torch.log(mu) - mu - torch.lgamma(y + 1.0)
        masked = torch.where(batch.obs_mask.unsqueeze(1), log_p, torch.zeros_like(log_p))
        return LikelihoodOutput(log_p_nb=log_p, masked_log_p_nb=masked, log_L=masked.sum(dim=-1))


def const_like(x, ref):
    """x와 같은 모양의 상수 텐서 (값 = ref 행들의 평균)."""
    return torch.full_like(x, float(ref.mean()))


def make_batch(setting, batch, train_mask):
    if setting == "no_pgv":
        return dataclasses.replace(batch, pgv=torch.ones_like(batch.pgv))
    if setting == "no_exposure":
        return dataclasses.replace(batch, E=torch.ones_like(batch.E))
    return batch


def main():
    base = load_pilot_a_batch(T.STATS_PATH, T.USGS_PATH)
    base = attach_ls_labels(T.GT_PATH, base, load_eval_ground_truth(T.GT_PATH, base))
    events = list(EVENTS)
    present = set(base.event_idx.tolist())
    events = [e for e in events if EVENT_TO_INDEX[e] in present]
    if DPM5_ONLY:
        events = [e for e in events if e in DPM5]
    dpm_idx = torch.tensor([EVENT_TO_INDEX[e] for e in DPM5])
    prefix = "abl-dpm5" if DPM5_ONLY else "abl"
    orig_like, orig_prior = T.DamageLikelihood, T.AreaPrior
    for s in SETTINGS:
        T.DamageLikelihood = PoissonLikelihood if s == "poisson" else orig_like
        uni = {"uniform_prior": "mean", "uniform_learned": "learned", "uniform_half": "half"}.get(s)
        T.AreaPrior = PriorUniform if uni else orig_prior
        PriorUniform.MODE = uni or "mean"
        for ev in events:
            ho = EVENT_TO_INDEX[ev]
            if s == "per_event":
                train_mask = base.event_idx == ho
            elif DPM5_ONLY:
                train_mask = torch.isin(base.event_idx, dpm_idx) & (base.event_idx != ho)
            else:
                train_mask = base.event_idx != ho
            batch = make_batch(s, base, train_mask)
            kw = dict(TRAIN_KW)
            note = ""
            if uni:
                c_ls, c_lq, p_ls0, p_lq0 = uniform_const(batch, train_mask)
                PriorUniform.C = (c_ls, c_lq)
                note = f"  균등 출발값 산사태 {p_ls0:.3f} 액상화 {p_lq0:.3f}" if uni != "half" else "  균등 0.5"
            if s == "per_event":
                kw["use_labels"] = False        # 예측할 지진의 정답은 학습에 쓸 수 없음
            t0 = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                reg, like, pri, _ = T.train(batch=subset_batch(batch, train_mask), **kw)
                with torch.no_grad():
                    out_r = reg(batch)
                    out_l = like(batch, out_r.mu)
                    lj, lp = T.marginalize(T.prior_log_w(pri, batch), out_l.log_L)
                    p_ls, p_lq = T.infer(lj, lp)
                T.report_holdout(batch, reg, like, pri, p_ls, p_lq, ev, f"_{prefix}-{s}_holdout-{ev.replace(' ', '')}")
            print(f"{s:14s} {ev}  학습행 {int(train_mask.sum())}  {time.time() - t0:.1f}s{note}", flush=True)
    T.DamageLikelihood, T.AreaPrior = orig_like, orig_prior


if __name__ == "__main__":
    main()
