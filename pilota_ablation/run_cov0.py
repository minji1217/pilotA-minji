"""산사태 판독 범위(cov) 항 점검: 학습은 그대로, 테스트 지진만 cov 항을 0으로 두고 정확도를 본다.

교수님 지시 (2026-11): "학습은 변함없음 (현재 학습되어있는 모델 그대로). 테스트 시, 테스트 대상 지진에 대한
cov 텀을 0으로 둔 상태에서 정확도 확인."

- 학습: AdminDER(pilotA A2) 그대로. pilotA는 무작위 요소가 없어 다시 학습해도 기존 모델과 똑같다.
- 테스트: 같은 모델로 두 번 추론한다.
    base : 원래대로 (테스트 지진 산사태 사전확률에 log cov 포함)
    cov0 : 테스트 지진 행의 log cov만 0으로 (cov = 1, 판독 범위 보정 없음). 학습 지진 값은 건드리지 않는다
  α_e 맞춤(피해 건수로 LBFGS)과 사후확률 계산은 각 경우의 사전확률로 다시 한다.
- 기준 학습 두 가지: 8개(뺀 지진 외 8개) / --dpm5 4개(뺀 지진 외 DPM 있는 4개). 평가는 DPM 있는 5개 지진.

    python run_cov0.py <pilotA 폴더 (followup4-lq-J-nolabels)> [--dpm5]
결과: <pilotA 폴더>/outputs/holdout_{summary,detail,detail_lq}_cov{base|0}{-dpm5}_holdout-<지진>.csv
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
from loader import attach_ls_labels, load_eval_ground_truth, load_pilot_a_batch, subset_batch  # noqa: E402
from schema import EVENT_TO_INDEX  # noqa: E402

torch.set_num_threads(4)
DPM5 = ["2016 구마모토", "2018 오사카", "2018 훗카이도", "2021 후쿠시마", "2024 노토반도"]
DPM5_ONLY = "--dpm5" in sys.argv
TRAIN_KW = dict(seed=0, epochs=3000, lr=0.02, lam_gamma=10, area_mode="fixed-cov-mtn", lam_kappa=10,
                mtn_standardize="train", use_labels=True, lq_labels=False, lq_b_only=True)


def evaluate(batch, reg, like, pri, ev, tag):
    with torch.no_grad():
        out_r = reg(batch)
        out_l = like(batch, out_r.mu)
        lj, lp = T.marginalize(T.prior_log_w(pri, batch), out_l.log_L)
        p_ls, p_lq = T.infer(lj, lp)
    T.report_holdout(batch, reg, like, pri, p_ls, p_lq, ev, tag)


def main():
    base = load_pilot_a_batch(T.STATS_PATH, T.USGS_PATH)
    base = attach_ls_labels(T.GT_PATH, base, load_eval_ground_truth(T.GT_PATH, base))
    dpm_idx = torch.tensor([EVENT_TO_INDEX[e] for e in DPM5])
    sfx = "-dpm5" if DPM5_ONLY else ""
    for ev in DPM5:
        ho = EVENT_TO_INDEX[ev]
        train_mask = (torch.isin(base.event_idx, dpm_idx) if DPM5_ONLY else torch.ones_like(base.event_idx, dtype=torch.bool)) \
            & (base.event_idx != ho)
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            reg, like, pri, _ = T.train(batch=subset_batch(base, train_mask), **TRAIN_KW)
            tag = ev.replace(" ", "")
            evaluate(base, reg, like, pri, ev, f"_covbase{sfx}_holdout-{tag}")
            # 테스트 지진 행만 log cov = 0 (판독 범위 보정 없음). 학습은 위에서 끝났으므로 모델은 같다.
            test = base.event_idx == ho
            b0 = dataclasses.replace(base, log_cov_ls=torch.where(test, torch.zeros_like(base.log_cov_ls), base.log_cov_ls))
            evaluate(b0, reg, like, pri, ev, f"_cov0{sfx}_holdout-{tag}")
        n_cov = int(((base.log_cov_ls != 0) & test).sum())
        print(f"{ev}  학습행 {int(train_mask.sum())}  테스트 지진에서 cov < 1인 행 {n_cov}  {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
