"""cov 항 점검 결과를 보고용 표 형식으로: 지진별 평가 시정촌 · 양성 · 음성, USGS 평균집계, 변환 prior(log p̄ + log k),
AdminDER(cov 있음), AdminDER(평가 지진 cov = 1). 산사태만. 정답이 한쪽뿐인 지진은 뺀다.
① 9개 LOEO (8개 학습): run_cov0.py 와 run_cov0.py --extra 결과.  ② DPM 5개 LOEO (4개 학습): run_cov0.py --dpm5 결과.

    python table_cov0.py <pilotA 폴더> [결과.csv] [--dpm5eval]
--dpm5eval: ①도 DPM 있는 5개 지진만 평가 (DisasterNet 비교 기준).
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ARGS = [a for a in sys.argv[1:] if a != "--dpm5eval"]
DPM5_EVAL = "--dpm5eval" in sys.argv
W = ARGS[0]
OUT = ARGS[1] if len(ARGS) > 1 else None
DPM5 = {"2016 구마모토", "2018 오사카", "2018 훗카이도", "2021 후쿠시마", "2024 노토반도"}
ORDER = ["2000 돗토리", "2004 니가타현주에쓰", "2007 니가타현주에쓰오키", "2008 이와테미야기",
         "2016 구마모토", "2018 오사카", "2018 훗카이도", "2021 후쿠시마", "2024 노토반도"]


def load(mode, sfx):
    import glob
    fs = glob.glob(os.path.join(W, "outputs", f"holdout_detail_cov{mode}{sfx}_holdout-*.csv"))
    return pd.concat([pd.read_csv(f, dtype={"muni_code": str}) for f in fs])


rows = []
for setname, sfx in [("① DPM 5개 평가 (8개 학습)" if DPM5_EVAL else "① 9개 LOEO (8개 학습)", ""), ("② DPM 5개 LOEO (4개 학습)", "-dpm5")]:
    b, z = load("base", sfx), load("0", sfx)
    z = z.set_index(["event", "muni_code"])
    evs = []
    for ev in ORDER:
        g = b[b.event == ev]
        if len(g) == 0 or g.ls_true.nunique() < 2 or (DPM5_EVAL and ev not in DPM5):
            continue
        g0 = z.loc[ev].loc[g.muni_code]
        y = g.ls_true.values
        r = {"세트": setname, "지진": ev, "평가 시정촌": len(g), "양성": int(y.sum()), "음성": int(len(g) - y.sum()),
             "USGS 평균집계": roc_auc_score(y, g.pi_ls),
             "변환 prior log p̄ + log k": roc_auc_score(y, np.log(g.pi_ls.clip(lower=1e-12)) + g.log_k_ls),
             "AdminDER (cov 있음)": roc_auc_score(y, g.p_ls_alpha_fit),
             "AdminDER (평가 지진 cov = 1)": roc_auc_score(y, g0.p_ls_alpha_fit.values)}
        rows.append(r)
        evs.append(r)
    scopes = [("가중평균", lambda e: True)]
    if sfx == "" and not DPM5_EVAL:
        scopes.append(("가중평균 (DPM 5개 지진만)", lambda e: e in DPM5))
    for name, keep in scopes:
        sel = [r for r in evs if keep(r["지진"])]
        n = sum(r["평가 시정촌"] for r in sel)
        w = {"세트": setname, "지진": name, "평가 시정촌": n, "양성": sum(r["양성"] for r in sel),
             "음성": sum(r["음성"] for r in sel)}
        for c in list(evs[0])[5:]:
            w[c] = sum(r[c] * r["평가 시정촌"] for r in sel) / n
        rows.append(w)
t = pd.DataFrame(rows)
num = t.columns[5:]
t[num] = t[num].astype(float).round(3)
pd.set_option("display.width", 250)
print(t.to_string(index=False))
if OUT:
    t.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT}")
