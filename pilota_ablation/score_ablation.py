"""RQ4 ablation 채점: 지진 · 재해별 AUC를 채점 시정촌 수로 가중평균 (정답이 한쪽뿐인 조합은 제외).
DPM 있는 5개 지진 평가와 9개 지진 평가를 함께 낸다.

    python score_ablation.py <pilotA 폴더> [결과 CSV 경로]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

W = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else None
DPM5 = {"2016 구마모토", "2018 오사카", "2018 훗카이도", "2021 후쿠시마", "2024 노토반도"}
ORDER = ["full", "uniform_prior", "no_pgv", "poisson", "no_exposure", "per_event"]
NAME = {"full": "TRACE (전체)", "uniform_prior": "prior 균등", "no_pgv": "흔들림 항 제거",
        "poisson": "포아송", "no_exposure": "노출 offset 제거", "per_event": "이벤트별 단독 학습"}


def load(setting, hz):
    pre = "holdout_detail_lq" if hz == "LQ" else "holdout_detail"
    fs = glob.glob(os.path.join(W, "outputs", f"{pre}_abl-{setting}_holdout-*.csv"))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f, dtype={"muni_code": str}) for f in fs])
    col = "lq" if hz == "LQ" else "ls"
    return d.rename(columns={f"{col}_true": "y", f"p_{col}_alpha_fit": "p"})[["event", "y", "p"]].assign(h=hz)


rows, per_event = [], []
for s in ORDER:
    parts = [x for x in (load(s, "LS"), load(s, "LQ")) if x is not None]
    if not parts:
        continue
    d = pd.concat(parts)
    res = {}
    for (ev, h), g in d.groupby(["event", "h"]):
        if g.y.nunique() < 2:
            continue
        a = roc_auc_score(g.y, g.p)
        res[(ev, h)] = (a, len(g))
        per_event.append({"설정": NAME[s], "지진": ev, "재해": h, "채점 수": len(g), "AUC": round(a, 4)})
    for scope, keep in [("DPM 있는 5개", lambda e: e in DPM5), ("9개 전체", lambda e: True)]:
        r = {"설정": NAME[s], "평가": scope}
        for H in ["전체", "LS", "LQ"]:
            it = [(a, n) for (e, h), (a, n) in res.items() if keep(e) and (H == "전체" or h == H)]
            r[H] = round(sum(a * n for a, n in it) / sum(n for _, n in it), 4) if it else np.nan
        rows.append(r)

t = pd.DataFrame(rows)
for scope in ["DPM 있는 5개", "9개 전체"]:
    x = t[t["평가"] == scope].drop(columns="평가").reset_index(drop=True)
    if len(x):
        x["전체 차이"] = (x["전체"] - x.loc[x["설정"] == NAME["full"], "전체"].iloc[0]).round(4)
    print(f"\n== 평가: {scope}\n{x.to_string(index=False)}")
if OUT:
    t.to_csv(OUT, index=False, encoding="utf-8-sig")
    pd.DataFrame(per_event).to_csv(OUT.replace(".csv", "_지진별.csv"), index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT}")
