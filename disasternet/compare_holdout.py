"""
최신 pilotA(산사태 E3 + 액상화 J, 지진 분리) vs DisasterNet vs USGS prior 비교.

pilotA 최신 설정은 원본 저장소 pilotA의 followup4-lq-J-labels 브랜치에 있다.
그 브랜치에서 DPM이 있는 5개 지진을 하나씩 빼고 돌린 holdout_detail CSV를 읽어,
같은 라벨 행에서 DisasterNet 예측(python -m disasternet.compare 결과)과 나란히 채점한다.

1) followup4-lq-J-labels 체크아웃에서 (raw/시정촌_면적.csv는 tools/build_dataset.py --area-from-usgs)
     for ev in "2016 구마모토" "2018 오사카" "2018 훗카이도" "2021 후쿠시마" "2024 노토반도"; do
       python train.py --holdout-event "$ev" --lam-gamma 10 --tag "lqlab_holdout-${ev// /}"
       python train.py --holdout-event "$ev" --lam-gamma 10 --no-lq-labels --tag "lqnolab_holdout-${ev// /}"
     done
   (--lam-gamma 10 = 1001 발표 E3의 λ_γ = λ_κ = 10. 훗카이도 LS 사후 0.861, κ -0.0075 재현 확인)
2) 이 저장소에서
     python disasternet/compare_holdout.py <그 체크아웃 경로> outputs/disasternet/predictions_base.csv \
         outputs/disasternet/compare_E3_lqlab.csv lqlab

pilotA 사후는 E3 방식(평가 지진 α_e를 피해 통계로 맞춤, p_*_alpha_fit)을 쓴다.
"""
# 최신 pilotA(E3, 이벤트 분리) holdout 결과 vs DisasterNet(base) vs USGS prior, 같은 라벨 행에서 비교
import sys, glob, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
L = sys.argv[1]; DN = sys.argv[2]; OUT = sys.argv[3]; TAG = sys.argv[4]
dn = pd.read_csv(DN, dtype={"muni_code": str})
def auc(y, p):
    y = np.asarray(y); return roc_auc_score(y, p) if len(set(y)) == 2 else np.nan
rows = []; details = []
for kind, pat, ycol, cols in (
    ("LS", "holdout_detail_{TAG}_holdout-*.csv", "ls_true",
     {"USGS prior": "pi_ls", "pilotA prior(cov+κ)": "z_cov_kappa", "pilotA 사후(E3, α_e 맞춤)": "p_ls_alpha_fit"}),
    ("LQ", "holdout_detail_lq_{TAG}_holdout-*.csv", "lq_true",
     {"USGS prior": "pi_lq", "pilotA prior(J)": "z_kappa_lq", "pilotA 사후(E3, α_e 맞춤)": "p_lq_alpha_fit"})):
    for f in sorted(glob.glob(f"{L}/outputs/{pat.format(TAG=TAG)}")):
        d = pd.read_csv(f, dtype={"muni_code": str})
        ev = d["event"].iloc[0]
        m = d.merge(dn[dn.event == ev][["muni_code", f"p_{kind.lower()}_DisasterNet"]], on="muni_code", how="inner")
        m = m.rename(columns={f"p_{kind.lower()}_DisasterNet": "DisasterNet"})
        m["kind"] = kind; details.append(m)
        r = {"재해": kind, "이벤트": ev, "라벨 행": len(d), "DPM 있는 행": len(m),
             "양성": int(m[ycol].sum()), "음성": int((m[ycol] == 0).sum())}
        for name, c in {**cols, "DisasterNet": "DisasterNet"}.items():
            r[f"AUC {name}"] = auc(m[ycol], m[c])
        for name, c in (("pilotA 사후", cols[list(cols)[2]]), ("DisasterNet", "DisasterNet"), ("USGS prior", cols["USGS prior"])):
            r[f"MSE {name}"] = float(np.mean((m[ycol] - m[c]) ** 2)) if len(m) else np.nan
        rows.append(r)
t = pd.DataFrame(rows)
# 이벤트별 AUC를 (DPM 있는 행 수)로 가중평균 — pilotA eval의 wavg와 같은 방식
summ = []
for kind, g in t.groupby("재해"):
    s = {"재해": kind, "이벤트": "가중평균", "DPM 있는 행": int(g["DPM 있는 행"].sum())}
    for c in [c for c in t.columns if c.startswith("AUC") or c.startswith("MSE")]:
        ok = g[c].notna()
        s[c] = np.average(g.loc[ok, c], weights=g.loc[ok, "DPM 있는 행"]) if ok.any() else np.nan
    summ.append(s)
t = pd.concat([t, pd.DataFrame(summ)], ignore_index=True)
t.to_csv(OUT, index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(t.round(3).to_string(index=False))
