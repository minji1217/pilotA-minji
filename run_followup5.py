"""후속실험 5: 판독 범위(cov) + 산지 비율(kappa) + 학습/평가 분리.

조건 A(면적 항 fixed, z = log p̄ + log k) 위에 LS prior만 바꾼다.

    z_LS = log p̄ + log k + log cov + kappa * z_mtn,   P(LS=1) = sigmoid(z_LS)

- log cov  : 학습하지 않는 고정 오프셋. 라벨 없는 행과 보고서 기반 행은 cov=1이라 0.
- kappa    : 새로 학습하는 유일한 prior 파라미터(a=1, b=0, c=1 고정). loss에 lam_kappa * kappa^2.
- z_mtn    : 산지 비율을 '학습 행' 기준으로 표준화한 값. 평가 행에도 같은 평균·표준편차를 쓴다.
- 우도     : LS 라벨이 있는 학습 행은 LS를 라벨값으로 고정하고 LQ만 합산한다.
- 분리     : --test-event의 행 전체를 학습에서 뺀다. 평가 행의 사후는 라벨 없이(4상태 합산) 계산한다.

평가 이벤트의 alpha_e(이벤트 절편)는 학습 데이터가 없어 gradient를 받지 못한다.
--heldout-alpha로 채울 값을 정한다.
    mean : 학습된 이벤트 절편의 평균
    zero : 0 (학습되지 않은 초기값 그대로)
    fit  : 다른 파라미터를 모두 고정하고, 평가 이벤트 행의 '라벨 없음' 우도 sum log P(y_i)를
           최대로 만드는 alpha_e 하나를 찾는다. LS 라벨은 쓰지 않는다.
           평가 행의 사후 계산이 이미 y를 쓰므로(가이드 Step 4), y로 이벤트 수준만 맞추는 것이다.

--mtn-damage를 주면 피해 회귀식 log lambda에도 eta_c * z_mtn을 더한다(가이드와 다른 비교 조건).
기준 이벤트(alpha_e=0 고정)는 EVENTS[0]=2004 니가타라 돗토리·훗카이도 어느 회차에서도 학습에 남는다.

실행 예:
    python run_followup5.py --test-event "2018 훗카이도"
    python run_followup5.py --test-event "2000 돗토리"
    python run_followup5.py                      # 분리 없이 전체 학습(비교용)
"""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from infer import infer
from loader import load_eval_ground_truth, load_ls_coverage, load_pilot_a_batch
from marginal import marginalize
from prior import prior_log_w, prior_z
from schema import EVENT_TO_INDEX, EVENTS, INDEX_TO_EVENT, PilotABatch
from train import GT_PATH, STATS_PATH, USGS_PATH, train
from reporting import dump_params, save_loss_history


def subset_batch(batch: PilotABatch, idx: torch.Tensor) -> PilotABatch:
    """행 index로 batch를 잘라 낸다. Tensor 필드는 인덱싱하고 시정촌코드 tuple도 맞춰 자른다."""
    fields = {}
    for f in dataclasses.fields(batch):
        v = getattr(batch, f.name)
        if isinstance(v, torch.Tensor):
            fields[f.name] = v[idx]
        elif f.name == "municipality_code":
            fields[f.name] = tuple(v[i] for i in idx.tolist())
        else:
            fields[f.name] = v
    out = PilotABatch(**fields)
    out.validate()
    return out


def auc(score, label) -> float:
    """Mann-Whitney AUC. 동점은 0.5로 센다. 양성이나 음성이 없으면 nan."""
    score, label = np.asarray(score, dtype=float), np.asarray(label)
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0) + 0.5 * (diff == 0)).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-event", default=None, choices=list(EVENTS),
                    help="학습에서 행 전체를 뺄 평가 이벤트. 생략하면 분리 없이 전체 학습")
    ap.add_argument("--no-cov", action="store_true", help="log cov를 넣지 않는다(조건 A와 같은 prior)")
    ap.add_argument("--no-mtn", action="store_true", help="kappa * z_mtn을 넣지 않는다(kappa = 0)")
    ap.add_argument("--no-labels", action="store_true",
                    help="LS 라벨을 우도에 쓰지 않는다(모든 행을 라벨 없음 식으로)")
    ap.add_argument("--lam-gamma", type=float, default=10.0)
    ap.add_argument("--lam-kappa", type=float, default=None, help="기본값은 --lam-gamma와 같다")
    ap.add_argument("--save-alpha-curve", action="store_true",
                    help="--heldout-alpha fit일 때 alpha_e 격자별 우도·사후 AUC를 alpha_fit_curve.csv로 저장(설명용)")
    ap.add_argument("--mtn-damage", action="store_true",
                    help="비교 조건: 피해 회귀식에도 eta_c * z_mtn을 넣는다")
    ap.add_argument("--heldout-alpha", default="mean", choices=["mean", "zero", "fit"],
                    help="평가 이벤트의 alpha_e에 넣을 값")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="결과 폴더. 기본 results/followup5/<조건>")
    args = ap.parse_args()
    lam_kappa = args.lam_gamma if args.lam_kappa is None else args.lam_kappa
    mtn_prior = not args.no_mtn

    tag = "full" if args.test_event is None else f"test-{args.test_event.replace(' ', '_')}"
    tag += ("" if not args.no_cov else "_nocov") + ("" if mtn_prior else "_nomtn")
    tag += ("" if not args.no_labels else "_nolabels") + ("_mtndmg" if args.mtn_damage else "")
    tag += f"_alpha-{args.heldout_alpha}"
    out_dir = Path(args.out or f"results/followup5/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 데이터 ----
    batch = load_pilot_a_batch(STATS_PATH, USGS_PATH)
    eval_gt = load_eval_ground_truth(GT_PATH, batch)
    B = batch.batch_size

    ls_label = torch.full((B,), -1, dtype=torch.long)
    ls_rows = eval_gt.model_row_idx[eval_gt.ls_eval_mask]
    ls_label[ls_rows] = eval_gt.gt_ls[eval_gt.ls_eval_mask].to(torch.long)

    cov = load_ls_coverage(GT_PATH, batch)
    log_cov = torch.zeros(B, dtype=batch.pi_ls.dtype) if args.no_cov else torch.log(cov)

    test_mask = torch.zeros(B, dtype=torch.bool)
    if args.test_event is not None:
        test_mask = batch.event_idx == EVENT_TO_INDEX[args.test_event]
    train_idx = torch.nonzero(~test_mask).squeeze(-1)

    # z_mtn은 loader가 418행 전체로 표준화해 둔 값이다. 표준화는 선형 변환이라
    # 학습 행의 평균·표준편차로 한 번 더 표준화하면 원비율을 학습 행 기준으로 표준화한 것과 같다.
    z_all = batch.z_mtn
    m, s = z_all[train_idx].mean(), z_all[train_idx].std()
    z_mtn = (z_all - m) / s

    full = dataclasses.replace(batch, z_mtn=z_mtn, log_cov=log_cov, ls_label=ls_label)
    train_batch = subset_batch(full, train_idx)
    n_lab = int((train_batch.ls_label >= 0).sum())
    n_neg = int((train_batch.ls_label == 0).sum())
    print(f"[{tag}] 학습 {train_batch.batch_size}행 (LS 라벨 {n_lab}, 음성 {n_neg}) / 평가 {int(test_mask.sum())}행")

    # ---- 학습 ----
    reg, like, pri, hist = train(
        train_batch, seed=args.seed, epochs=args.epochs, lr=args.lr, lam_gamma=args.lam_gamma,
        area_mode="fixed", mtn_prior=mtn_prior, lam_kappa=lam_kappa,
        use_labels=not args.no_labels, mtn_damage=args.mtn_damage,
    )

    # 평가 이벤트의 alpha_e는 gradient를 받지 못해 초기값 0에 남아 있다.
    if args.test_event is not None:
        e = EVENT_TO_INDEX[args.test_event]
        ref = reg.reference_event_idx
        if e == ref:
            raise ValueError("평가 이벤트가 기준 이벤트입니다. reference_event_idx를 바꿔야 합니다.")
        slot = e if e < ref else e - 1
        with torch.no_grad():
            trained = [i for i in range(len(EVENTS)) if i != e]
            mean_alpha = float(reg.alpha_event[trained].mean())
            if args.heldout_alpha == "mean":
                reg.alpha_event_free[slot] = mean_alpha
            elif args.heldout_alpha == "zero":
                reg.alpha_event_free[slot] = 0.0
            else:
                # 평가 행만 떼어 '라벨 없음' 우도를 alpha_e 격자 위에서 계산한다.
                # alpha_e 하나짜리 1차원 문제라 격자 탐색 후 주변을 한 번 더 촘촘히 본다.
                test_batch = subset_batch(
                    dataclasses.replace(batch, z_mtn=z_mtn, log_cov=log_cov, ls_label=None),
                    torch.nonzero(test_mask).squeeze(-1),
                )
                log_w_test = prior_log_w(pri, test_batch)

                def total_loglik(a: float) -> float:
                    reg.alpha_event_free[slot] = a
                    out_l = like(test_batch, reg(test_batch).mu)
                    return float(marginalize(log_w_test, out_l.log_L)[1].sum())

                grid = np.arange(-8.0, 4.0 + 1e-9, 0.05)
                best = max(grid, key=total_loglik)
                fine = np.arange(best - 0.05, best + 0.05 + 1e-9, 0.001)
                best = max(fine, key=total_loglik)

                # 곡선 저장: alpha_e를 바꿔 가며 평가 행의 우도와 사후 LS AUC가 어떻게 변하는지 기록한다.
                # 결과에는 영향이 없고 설명용 자료다. 라벨은 AUC 기록에만 쓰고 alpha_e 선택에는 쓰지 않는다.
                if args.save_alpha_curve:
                    lab = ls_label[test_mask].numpy()
                    rows = []
                    for a in np.arange(-4.0, 1.5 + 1e-9, 0.05):
                        ll = total_loglik(float(a))
                        out_l = like(test_batch, reg(test_batch).mu)
                        lj, lpy = marginalize(log_w_test, out_l.log_L)
                        p_ls_t, _ = infer(lj, lpy)
                        has_lab = lab >= 0
                        rows.append({"alpha_e": round(float(a), 3), "loglik": ll,
                                     "post_auc": auc(p_ls_t.numpy()[has_lab], lab[has_lab]),
                                     **{f"post_{c}": float(p_ls_t[i]) for i, c in enumerate(test_batch.municipality_code)}})
                    pd.DataFrame(rows).to_csv(out_dir / "alpha_fit_curve.csv", index=False, encoding="utf-8-sig")
                reg.alpha_event_free[slot] = float(best)
            heldout_alpha_value = float(reg.alpha_event[e])
            print(f"평가 이벤트 alpha_e = {heldout_alpha_value:+.3f} "
                  f"({args.heldout_alpha}; 학습 이벤트 평균 {mean_alpha:+.3f})")
    else:
        heldout_alpha_value = None

    # ---- 전체 행 추론: 라벨 없이(4상태 합산) ----
    with torch.no_grad():
        out_r = reg(full)
        out_l = like(full, out_r.mu)
        log_w = prior_log_w(pri, full)
        log_joint, log_Py = marginalize(log_w, out_l.log_L)
        p_ls, _ = infer(log_joint, log_Py)
        z_ls, _ = prior_z(pri, full)
        z_A = torch.log(full.pi_ls.clamp_min(1e-6)) + full.log_k_ls  # 조건 A 자기 prior

    kappa = float(pri.kappa) if mtn_prior else 0.0
    pred = pd.DataFrame({
        "event": [INDEX_TO_EVENT[i] for i in full.event_idx.tolist()],
        "muni_code": list(full.municipality_code),
        "is_test": test_mask.numpy(),
        "ls_label": ls_label.numpy(),
        "cov": cov.numpy(),
        "mountain_z": z_mtn.numpy(),
        "pi_ls": full.pi_ls.numpy(),
        "z_condA": z_A.numpy(),
        "z_cov": (z_A + full.log_cov).numpy(),
        "z_prior": z_ls.numpy(),
        "prior_ls": torch.sigmoid(z_ls).numpy(),
        "post_ls": p_ls.numpy(),
    })
    pred.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    save_loss_history(hist, dir_=str(out_dir), tag=tag)
    dump_params(reg, like, pri, path=str(out_dir / "params.csv"))

    # ---- 이벤트별 LS AUC (라벨 있는 행) ----
    rows = []
    events = [args.test_event] if args.test_event else [e for e in EVENTS]
    for ev in events:
        d = pred[(pred.event == ev) & (pred.ls_label >= 0)]
        rows.append({
            "event": ev, "n": len(d), "neg": int((d.ls_label == 0).sum()),
            "auc_usgs": auc(d.pi_ls, d.ls_label),
            "auc_condA": auc(d.z_condA, d.ls_label),
            "auc_cov_k0": auc(d.z_cov, d.ls_label),
            "auc_prior": auc(d.z_prior, d.ls_label),
            "auc_post": auc(d.post_ls, d.ls_label),
        })
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "auc.csv", index=False, encoding="utf-8-sig")

    meta = {
        "test_event": args.test_event, "kappa": kappa, "lam_kappa": lam_kappa,
        "lam_gamma": args.lam_gamma, "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
        "use_cov": not args.no_cov, "mtn_prior": mtn_prior, "use_labels": not args.no_labels,
        "heldout_alpha": args.heldout_alpha,
        "heldout_alpha_value": heldout_alpha_value,
        "mtn_damage": bool(args.mtn_damage),
        "n_train": train_batch.batch_size, "n_train_labeled": n_lab, "n_train_negative": n_neg,
        "z_mtn_train_mean_of_loader_z": float(m), "z_mtn_train_std_of_loader_z": float(s),
        "final_loss": float(hist["loss_total"].iloc[-1]),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nkappa = {kappa:+.4f}")
    print(res.round(3).to_string(index=False))
    print(f"저장: {out_dir}")


if __name__ == "__main__":
    main()
