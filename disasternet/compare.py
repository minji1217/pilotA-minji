"""
DisasterNet vs pilotA 비교 러너.

    python -m disasternet.compare --dpm raw/DPM_municipality_v5.xlsx

하는 일
  1) pilotA 입력(통계+USGS)과 GT를 평소처럼 읽는다.
  2) DPM xlsx를 읽어 PilotABatch 행과 맞춘다 (5개 이벤트).
  3) pilotA를 학습한다. 기본은 평소처럼 9개 이벤트 전체로 학습
     (--pilota-train dpm-events 이면 DPM이 있는 5개 이벤트 행만으로 학습).
  4) DisasterNet을 학습한다. GT는 쓰지 않는다 (pilotA와 같은 비지도 조건).
  5) "DPM이 있는 행 ∩ GT 행"만 남겨서 USGS prior / pilotA / DisasterNet 셋을
     같은 eval.evaluate()로 채점한다.

산출물 (outputs/disasternet/ 아래, --tag가 있으면 파일명 뒤에 붙는다)
  compare_summary.csv     모델별 MSE / AUC (pooled, 이벤트 가중평균)
  compare_per_event.csv   모델 × 이벤트별 MSE / AUC
  predictions.csv         비교 행의 p_ls / p_lq (세 모델 나란히) + DPM + DisasterNet p_bd
  disasternet_params.csv  DisasterNet 학습 파라미터 (이벤트별)
  disasternet_loss.csv    DisasterNet 에폭별 loss
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from eval import evaluate
from infer import infer
from loader import load_eval_ground_truth, load_pilot_a_batch
from marginal import marginalize
from schema import INDEX_TO_EVENT
from train import GT_PATH, STATS_PATH, USGS_PATH, to_eval_gt, train

from .data import DEFAULT_DPM_COLUMN, align_dpm_to_batch, read_dpm_table
from .model import fit_disasternet

OUT_DIR = Path("outputs/disasternet")
KEYS = ["event_idx", "muni_code"]


def run_pilota(batch, *, train_rows=None, **train_kwargs):
    """pilotA를 학습하고 전체 batch 행의 p_ls, p_lq를 돌려준다.

    train_rows가 있으면 그 행만으로 학습한 뒤 전체 행에 대해 추론한다.
    (pilotA 파라미터는 행 단위가 아니라 채널/이벤트 단위라 학습에 안 쓴 행에도 추론할 수 있다.)
    """
    train_batch = batch if train_rows is None else _select_batch(batch, train_rows)
    reg, like, pri, _ = train(train_batch, **train_kwargs)
    with torch.no_grad():
        out_r = reg(batch)
        out_l = like(batch, out_r.mu)
        log_w = pri(batch.pi_ls, batch.pi_lq)
        log_joint, log_Py = marginalize(log_w, out_l.log_L)
        return infer(log_joint, log_Py)


def _select_batch(batch, idx):
    """PilotABatch에서 일부 행만 뽑는다. dataclass 필드를 그대로 따라간다."""
    import dataclasses
    idx = torch.as_tensor(sorted(set(idx)), dtype=torch.long)
    kw = {}
    for f in dataclasses.fields(batch):
        v = getattr(batch, f.name)
        if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == batch.batch_size:
            kw[f.name] = v[idx]
        elif isinstance(v, tuple) and len(v) == batch.batch_size:
            kw[f.name] = tuple(v[i] for i in idx.tolist())
        else:
            kw[f.name] = v
    return type(batch)(**kw)


def summarize(name, result):
    return {
        "model": name,
        "n_ls": result.n_ls, "n_lq": result.n_lq,
        "mse_ls": result.mse_ls, "mse_lq": result.mse_lq,
        "auc_ls": result.auc_ls, "auc_lq": result.auc_lq,
        "auc_ls_wavg": result.auc_ls_wavg, "auc_lq_wavg": result.auc_lq_wavg,
    }


def compare(gt_df, preds: dict):
    """preds: {모델명: DataFrame(event_idx, muni_code, p_ls, p_lq)} → (summary, per_event)."""
    rows, per_event = [], []
    for name, pred in preds.items():
        res = evaluate(gt_df, pred[KEYS + ["p_ls", "p_lq"]])
        rows.append(summarize(name, res))
        pe = res.per_event.drop(columns=[c for c in res.per_event.columns if "prior" in c])
        pe.insert(0, "model", name)
        per_event.append(pe)
    return pd.DataFrame(rows), pd.concat(per_event, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="DisasterNet vs pilotA 비교 (DPM이 있는 5개 이벤트)")
    ap.add_argument("--dpm", required=True, help="DPM_municipality_v5.xlsx 경로")
    ap.add_argument("--dpm-col", default=DEFAULT_DPM_COLUMN,
                    help="관측으로 쓸 DPM 컬럼 (기본 DPM_정규화z. 예: DPM_평균, DPM_p90, DPM_변화25초과비율)")
    ap.add_argument("--min-base-coh", type=float, default=None,
                    help="base상관도가 이 값 미만인 행 제외 (자료설명: 0.30 미만은 측정 실패에 가까움)")
    ap.add_argument("--stats", default=STATS_PATH)
    ap.add_argument("--usgs", default=USGS_PATH)
    ap.add_argument("--gt", default=GT_PATH)
    # DisasterNet
    ap.add_argument("--inference", default="vi", choices=["vi", "exact"],
                    help="vi=논문 방식(SVI+normalizing flow) / exact=정확 주변화(점검용)")
    ap.add_argument("--pooled", action="store_true", help="5개 이벤트를 한 모델로 학습 (기본: 이벤트별)")
    ap.add_argument("--no-bd", action="store_true", help="건물피해(BD) 노드를 빼는 ablation")
    ap.add_argument("--dn-prior-mode", default="free", choices=["free", "fixed", "bounded"])
    ap.add_argument("--dn-epochs", type=int, default=3000)
    ap.add_argument("--dn-lr", type=float, default=0.02)
    ap.add_argument("--mc-samples", type=int, default=8)
    ap.add_argument("--flow-layers", type=int, default=4)
    # pilotA (train.py와 같은 옵션)
    ap.add_argument("--pilota-train", default="all", choices=["all", "dpm-events"],
                    help="all=9개 이벤트 전체로 학습(평소대로) / dpm-events=DPM 5개 이벤트 행만")
    ap.add_argument("--prior-mode", default="free", choices=["free", "fixed", "bounded"])
    ap.add_argument("--lam-gamma", type=float, default=0.0)
    ap.add_argument("--b-bound", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    sfx = f"_{a.tag}" if a.tag else ""

    batch = load_pilot_a_batch(a.stats, a.usgs)
    eval_gt = load_eval_ground_truth(a.gt, batch)
    gt_df_all = to_eval_gt(eval_gt)

    dpm_rows = align_dpm_to_batch(
        read_dpm_table(a.dpm, column=a.dpm_col, min_base_coh=a.min_base_coh), batch)

    # ---- DisasterNet ----
    dn = fit_disasternet(
        dpm_rows, inference=a.inference, pooled=a.pooled, epochs=a.dn_epochs, lr=a.dn_lr,
        seed=a.seed, n_samples=a.mc_samples, use_bd=not a.no_bd,
        prior_mode=a.dn_prior_mode, flow_layers=a.flow_layers,
    )

    # ---- pilotA ----
    train_rows = None if a.pilota_train == "all" else dpm_rows.model_row_idx.tolist()
    if train_rows is not None:
        print(f"[pilotA] DPM 이벤트 행 {len(train_rows)}개로만 학습")
    p_ls, p_lq = run_pilota(batch, train_rows=train_rows, seed=a.seed, epochs=a.epochs,
                            lr=a.lr, lam_gamma=a.lam_gamma, prior_mode=a.prior_mode,
                            b_bound=a.b_bound)

    # ---- 비교 행: DPM 행 ∩ GT 행 ----
    idx = dn.predictions["model_row_idx"].to_numpy()
    base = dn.predictions[KEYS].copy()
    preds = {
        "USGS prior": base.assign(p_ls=batch.pi_ls[idx].numpy(), p_lq=batch.pi_lq[idx].numpy()),
        "pilotA": base.assign(p_ls=p_ls[idx].numpy(), p_lq=p_lq[idx].numpy()),
        "DisasterNet": dn.predictions[KEYS + ["p_ls", "p_lq"]],
    }
    keyset = set(map(tuple, base[KEYS].to_numpy().tolist()))
    in_common = [tuple(k) in keyset for k in gt_df_all[KEYS].to_numpy().tolist()]
    gt_df = gt_df_all.loc[in_common].reset_index(drop=True)
    print(f"\n비교 행: GT {len(gt_df_all)}행 중 DPM이 있는 {len(gt_df)}행 "
          f"(LS 평가 {int(gt_df['ls_eval_mask'].sum())} / LQ 평가 {int(gt_df['lq_eval_mask'].sum())})")

    summary, per_event = compare(gt_df, preds)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / f"compare_summary{sfx}.csv", index=False, encoding="utf-8-sig")
    per_event.to_csv(OUT_DIR / f"compare_per_event{sfx}.csv", index=False, encoding="utf-8-sig")
    side = base.copy()
    side.insert(1, "event", [INDEX_TO_EVENT[i] for i in side["event_idx"]])
    side["dpm"] = dn.predictions["dpm"].to_numpy()
    for name, p in preds.items():
        side[f"p_ls_{name}"] = p["p_ls"].to_numpy()
        side[f"p_lq_{name}"] = p["p_lq"].to_numpy()
    side["p_bd_DisasterNet"] = dn.predictions["p_bd"].to_numpy()
    side.to_csv(OUT_DIR / f"predictions{sfx}.csv", index=False, encoding="utf-8-sig")
    dn.params.to_csv(OUT_DIR / f"disasternet_params{sfx}.csv", index=False, encoding="utf-8-sig")
    dn.history.to_csv(OUT_DIR / f"disasternet_loss{sfx}.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print(summary.round(4).to_string(index=False))
    print("\n이벤트별:")
    print(per_event.round(4).to_string(index=False))
    print(f"\n저장: {OUT_DIR}/")
    return summary, per_event


if __name__ == "__main__":
    main()
