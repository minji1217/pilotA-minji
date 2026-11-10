"""RQ4 ablation: TRACE(pilotA A2)에서 한 번에 하나만 바꿔 LOEO(지진 하나씩 빼기) 성능을 낸다.

pilotA 모델 코드는 고치지 않는다. 입력(batch)이나 부품(likelihood)만 바꿔 끼운다.
pilotA 코드: minji1217/pilotA 브랜치 followup4-lq-J-nolabels (A2 기본값). 그 폴더를 첫 인자로 준다.

    python run_ablation.py <pilotA 폴더> [설정 ...]

설정 (기본: 전부)
  full          TRACE 그대로 (A2: 평가 지진 외 8개로 학습, 산사태 라벨 사용)
  uniform_prior 입력: 사전확률을 모든 시정촌에 같은 값으로. 산사태는 USGS 확률·시정촌 크기·판독 범위를
                학습 행 평균 상수로, 산지 비율은 0으로 (산사태 사전확률 = 상수 하나).
                액상화는 USGS 확률·시정촌 크기를 상수로 (액상화 사전확률 = 학습되는 b 하나)
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
ALL = ["full", "uniform_prior", "no_pgv", "poisson", "no_exposure", "per_event"]
SETTINGS = sys.argv[2:] or ALL
TRAIN_KW = dict(seed=0, epochs=3000, lr=0.02, lam_gamma=10, area_mode="fixed-cov-mtn", lam_kappa=10,
                mtn_standardize="train", use_labels=True, lq_labels=False, lq_b_only=True)


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
    if setting == "uniform_prior":
        t = train_mask
        lp_ls, lp_lq = torch.log(batch.pi_ls), torch.log(batch.pi_lq)
        return dataclasses.replace(
            batch,
            pi_ls=torch.exp(const_like(lp_ls, lp_ls[t])), pi_lq=torch.exp(const_like(lp_lq, lp_lq[t])),
            log_k_ls=const_like(batch.log_k_ls, batch.log_k_ls[t]),
            log_k_lq=const_like(batch.log_k_lq, batch.log_k_lq[t]),
            log_cov_ls=const_like(batch.log_cov_ls, batch.log_cov_ls[t]),
            z_mtn=torch.zeros_like(batch.z_mtn),
        )
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
    orig_like = T.DamageLikelihood
    for s in SETTINGS:
        T.DamageLikelihood = PoissonLikelihood if s == "poisson" else orig_like
        for ev in events:
            ho = EVENT_TO_INDEX[ev]
            train_mask = (base.event_idx == ho) if s == "per_event" else (base.event_idx != ho)
            batch = make_batch(s, base, train_mask)
            kw = dict(TRAIN_KW)
            if s == "uniform_prior":
                kw["mtn_standardize"] = "all"   # 산지 비율을 0으로 둬서 학습 행 표준화(표준편차 0)를 건너뜀
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
                T.report_holdout(batch, reg, like, pri, p_ls, p_lq, ev, f"_abl-{s}_holdout-{ev.replace(' ', '')}")
            print(f"{s:14s} {ev}  학습행 {int(train_mask.sum())}  {time.time() - t0:.1f}s", flush=True)
    T.DamageLikelihood = orig_like


if __name__ == "__main__":
    main()
