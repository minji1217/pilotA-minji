"""교수님 요청 (11/15): ②(DPM 지진으로 학습)에서도 2000 돗토리 · 2004 니가타현주에쓰 결과 확인. cov = 1 경우는 안 구함.
산사태만. 열: 평가 시정촌 · 양성 · 음성, USGS 평균집계, 변환 prior(log p̄ + log k), prior + cov + κ, AdminDER(cov 있음).

① 9개 LOEO (8개 학습): run_cov0.py, run_cov0.py --extra
② DPM 지진으로 학습: DPM 있는 지진은 run_cov0.py --dpm5 (평가 지진 외 4개로 학습),
   돗토리 · 니가타현주에쓰는 run_cov0.py --dpm5 --extra (DPM 있는 5개 전부로 학습. 평가 지진이 그 5개에 없어 뺄 것이 없음)

    python table_extra.py <pilotA 폴더> [결과.csv]
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
ORDER = ["2000 돗토리", "2004 니가타현주에쓰", "2007 니가타현주에쓰오키", "2008 이와테미야기",
         "2016 구마모토", "2018 오사카", "2018 훗카이도", "2021 후쿠시마", "2024 노토반도"]
COLS = ["USGS 평균집계", "변환 prior log p̄ + log k", "prior + cov + κ (cov 있음)", "AdminDER (cov 있음)"]


def load(sfx):
    fs = glob.glob(os.path.join(W, "outputs", f"holdout_detail_covbase{sfx}_holdout-*.csv"))
    return pd.concat([pd.read_csv(f, dtype={"muni_code": str}) for f in fs])


rows = []
for setname, sfx in [("① 9개 LOEO", ""), ("② DPM 지진으로 학습", "-dpm5")]:
    b = load(sfx)
    evs = []
    for ev in ORDER:
        g = b[b.event == ev]
        if len(g) == 0 or g.ls_true.nunique() < 2:
            continue
        y = g.ls_true.values
        ntr = 8 if sfx == "" else (4 if ev in DPM5 else 5)
        r = {"세트": f"{setname} ({ntr}개 학습)", "지진": ev, "평가 시정촌": len(g), "양성": int(y.sum()),
             "음성": int(len(g) - y.sum()),
             COLS[0]: roc_auc_score(y, g.pi_ls),
             COLS[1]: roc_auc_score(y, np.log(g.pi_ls.clip(lower=1e-12)) + g.log_k_ls),
             COLS[2]: roc_auc_score(y, g.z_cov_kappa),
             COLS[3]: roc_auc_score(y, g.p_ls_alpha_fit)}
        rows.append(r)
        evs.append(r)
    for name, keep in [("가중평균", lambda e: True), ("가중평균 (DPM 5개 지진만)", lambda e: e in DPM5)]:
        sel = [r for r in evs if keep(r["지진"])]
        n = sum(r["평가 시정촌"] for r in sel)
        w = {"세트": setname, "지진": name, "평가 시정촌": n, "양성": sum(r["양성"] for r in sel),
             "음성": sum(r["음성"] for r in sel)}
        for c in COLS:
            w[c] = sum(r[c] * r["평가 시정촌"] for r in sel) / n
        rows.append(w)
t = pd.DataFrame(rows)
t[COLS] = t[COLS].astype(float).round(3)
pd.set_option("display.width", 250)
print(t.to_string(index=False))
if OUT:
    t.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT}")
