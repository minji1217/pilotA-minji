"""run_cov0.py 결과 채점: 산사태(와 참고로 액상화) AUC, base 대 cov0, 지진별과 채점 시정촌 수 가중평균.

    python score_cov0.py <pilotA 폴더> [결과.csv]
"""
import glob
import os
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

W = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else None
DPM5 = {"2016 구마모토", "2018 오사카", "2018 훗카이도", "2021 후쿠시마", "2024 노토반도"}
rows = []
for setting, sfx in [("8개로 학습", ""), ("4개로 학습", "-dpm5")]:
    for mode in ["base", "0"]:
        for hz, pre, y, cols in [("산사태", "holdout_detail", "ls_true", {"사후확률": "p_ls_alpha_fit", "사전확률": "z_cov_kappa"}),
                                 ("액상화", "holdout_detail_lq", "lq_true", {"사후확률": "p_lq_alpha_fit"})]:
            fs = glob.glob(os.path.join(W, "outputs", f"{pre}_cov{mode}{sfx}_holdout-*.csv"))
            if not fs:
                continue
            d = pd.concat([pd.read_csv(f, dtype={"muni_code": str}) for f in fs])
            d = d[d.event.isin(DPM5)]        # 이 채점은 DPM 있는 5개 지진만 (9개 LOEO 표는 table_cov0.py)
            for what, col in cols.items():
                tot = n = 0
                for ev, g in d.groupby("event"):
                    if g[y].nunique() < 2:
                        continue
                    a = roc_auc_score(g[y], g[col])
                    rows.append({"기준": setting, "테스트 cov": "원래" if mode == "base" else "0", "재해": hz, "점수": what,
                                 "지진": ev, "채점 수": len(g), "AUC": round(a, 4)})
                    tot += a * len(g)
                    n += len(g)
                if n:
                    rows.append({"기준": setting, "테스트 cov": "원래" if mode == "base" else "0", "재해": hz, "점수": what,
                                 "지진": "가중평균", "채점 수": n, "AUC": round(tot / n, 4)})
t = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(t.pivot_table(index=["재해", "점수", "지진", "채점 수"], columns=["기준", "테스트 cov"], values="AUC").to_string())
if OUT:
    t.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT}")
