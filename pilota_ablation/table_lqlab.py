"""교수님 요청 (11/16): RQ4 표 2(② DPM 5개 지진 LOEO, 4개 학습)의 액상화 결과를 "AdminDER + 액상화 라벨 학습" 판으로.
score_ablation.py --dpm5 --lq-labels 가 만든 지진별 CSV를 표 2 모양(지진별 열 + 가중평균 + 기준 대비 Δ)으로 편다.

    python table_lqlab.py <지진별 CSV> [결과.csv]
"""
import sys

import pandas as pd

d = pd.read_csv(sys.argv[1])
OUT = sys.argv[2] if len(sys.argv) > 2 else None
COL = [("LS", "2016 구마모토"), ("LS", "2018 오사카"), ("LS", "2018 훗카이도"),
       ("LQ", "2016 구마모토"), ("LQ", "2018 훗카이도"), ("LQ", "2021 후쿠시마"), ("LQ", "2024 노토반도")]
NAME = {"LS": "산사태", "LQ": "액상화"}
rows = []
for s, g in d.groupby("설정", sort=False):
    r = {"조건": s}
    for h in ["LS", "LQ"]:
        x = g[g["재해"] == h]
        for hh, ev in COL:
            if hh == h:
                v = x.loc[x["지진"] == ev, "AUC"]
                r[f"{NAME[h]} {ev}"] = float(v.iloc[0]) if len(v) else float("nan")
        r[f"{NAME[h]} 가중평균"] = (x["AUC"] * x["채점 수"]).sum() / x["채점 수"].sum()
    rows.append(r)
t = pd.DataFrame(rows)
base = t.iloc[0]
for h in ["산사태", "액상화"]:
    t[f"{h} 기준 대비 Δ"] = t[f"{h} 가중평균"] - base[f"{h} 가중평균"]
num = t.columns[1:]
t[num] = t[num].astype(float).round(3)
pd.set_option("display.width", 300)
print(t.to_string(index=False))
if OUT:
    t.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT}")
