"""자연+혼재 정답의 AUC 상승을 두 원인으로 가른다.

  base     : ls_flag 1/0, NA 제외              (125행 · 양성 105)
  nat_ovl  : 같은 125행에 자연+혼재 라벨만 적용  (양성이 좁아진 효과만)
  nat      : 418행 전부                        (거기에 NA 293행이 음성으로 추가)

nat_ovl 이 base 보다 높으면 "양성을 좁힌 것"이 진짜 효과고,
nat 이 nat_ovl 보다 높으면 그 차이는 "쉬운 음성을 넣은 것"이 만든 것이다.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score

from infer import infer
from loader import load_pilot_a_batch
from marginal import marginalize
from prior import prior_log_w
from schema import INDEX_TO_EVENT
from train import DEFAULT_AREA_MODE, STATS_PATH, USGS_PATH, train


def wavg(df, score, gt):
    a, n = [], []
    for _, g in df.groupby("event"):
        if g[gt].nunique() < 2:
            continue
        a.append(roc_auc_score(g[gt], g[score])); n.append(len(g))
    return (float(np.average(a, weights=n)), len(a), int(sum(n))) if a else (float("nan"), 0, 0)


def main():
    gt = json.load(open("/tmp/gt_ruleB.json"))
    batch = load_pilot_a_batch(STATS_PATH, USGS_PATH)
    reg, like, pri, _ = train(batch, seed=0, epochs=3000, lam_gamma=10.0,
                              area_mode=DEFAULT_AREA_MODE)
    with torch.no_grad():
        out_l = like(batch, reg(batch).mu)
        log_joint, log_Py = marginalize(prior_log_w(pri, batch), out_l.log_L)
        p_ls, _ = infer(log_joint, log_Py)
        z_ls, _ = (pri.z(batch.pi_ls, batch.pi_lq, batch.log_k_ls, batch.log_k_lq)
                   if DEFAULT_AREA_MODE else pri.z(batch.pi_ls, batch.pi_lq))

    d = pd.DataFrame({
        "event": [INDEX_TO_EVENT[i] for i in batch.event_idx.tolist()],
        "muni": [int(m) for m in batch.municipality_code],
        "raw": batch.pi_ls.numpy(), "zp": torch.sigmoid(z_ls).numpy(), "p": p_ls.numpy()})
    d["key"] = d["event"] + "|" + d["muni"].astype(str)
    d["gb"] = d["key"].map(gt["base"])
    d["gn"] = d["key"].map(gt["nat"])

    sets = {"base":    d[d.gb.notna()].assign(g=lambda x: x.gb),
            "nat_ovl": d[d.gb.notna() & d.gn.notna()].assign(g=lambda x: x.gn),
            "nat":     d[d.gn.notna()].assign(g=lambda x: x.gn)}
    out = {"mode": DEFAULT_AREA_MODE}
    for k, s in sets.items():
        y = s["g"].to_numpy(float)
        raw, _, _ = wavg(s, "raw", "g"); zp, ne, _ = wavg(s, "zp", "g"); po, _, _ = wavg(s, "p", "g")
        out[k] = dict(raw=raw, zp=zp, post=po, gain=po - zp,
                      mse=float(np.mean((s["p"] - y) ** 2)),
                      bias=float(s["p"].mean() - y.mean()),
                      n=len(s), pos=int(y.sum()), ev=ne, rate=float(y.mean()))
    print("RESULT " + json.dumps(out, ensure_ascii=False))


main()
