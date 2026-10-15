"""
원 논문(KDD 2023) 4.4절의 "합성 자료" 평가를 우리 설정(run_cellmix 가상 화소)으로 재현한다.

  ① 실제 DPM으로 모델을 돌려 사후 확률 q와 이벤트별 파라미터(w_0x, w_kx, w_ex)를 얻는다.
  ② q로 합성 정답을 만든다: 이벤트·재해마다 실제 정답이 있는 시정촌 중 q 상위 n개를 1로
     (n = 실제 양성 수). 논문의 "실제 정답 위치를 사후와 AUC ≈ 1이 되게 맞춤"에 해당한다.
     정답이 없는 시정촌도 같은 문턱(그 이벤트 n번째 q)으로 z를 정하고, 건물 피해는 z ~ Bernoulli(q_BD).
  ③ 모델 관측식을 정방향으로 돌려 합성 DPM을 만든다: log x = w_0x + Σ w_kx z_k + w_ex·ε, DPM = x − offset.
  ④ 합성 DPM으로 같은 설정을 seed마다 다시 돌려, 합성 정답에 대한 AUC를 잰다 (USGS prior와 비교).

    python -m disasternet.synth_eval --local raw/local_250m_byevent_usgs.csv --out-dir outputs/disasternet/synth_D2 --jobs 4
"""
import argparse
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from schema import INDEX_TO_EVENT

from .model import fit_disasternet
from .run_cellmix import expand, to_rows
from .run_input_v5 import load_table

LABELS = {"LS": ("LS_정답", "p_ls"), "LQ": ("LQ_정답", "p_lq")}


def muni_probs(pred, x):
    p = pred.copy()
    p[["muni_code", "행"]] = p["muni_code"].str.split("#", expand=True)
    p = p.merge(x[["event_idx", "muni_code", "행", "w"]], on=["event_idx", "muni_code", "행"])
    for k in ("p_ls", "p_lq", "p_bd"):
        p[k] = p[k] * p["w"]
    return p.groupby(["event_idx", "muni_code"])[["p_ls", "p_lq", "p_bd"]].sum().reset_index()


def fit(x, seed, epochs, offset, kw):
    return fit_disasternet(to_rows(x, "DPM_배경차"), seed=seed, epochs=epochs, obs_offset=offset, verbose=False, **kw)


def _job(args):
    import torch
    torch.set_num_threads(1)
    x, seed, epochs, offset, kw = args
    return seed, muni_probs(fit(x, seed, epochs, offset, kw).predictions, x)


def auc_table(df, score_cols, label_cols):
    rows = []
    for dom, lab in label_cols.items():
        for ev, g in df.groupby("이벤트"):
            g = g[g[lab].notna()]
            y = g[lab].astype(int)
            if y.nunique() < 2:
                continue
            r = {"dom": dom, "이벤트": ev, "w": len(g)}
            for name, col in score_cols[dom].items():
                r[name] = roc_auc_score(y, g[col])
            rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="원 논문식 합성 자료 평가")
    ap.add_argument("--input", default="raw/DisasterNet_입력_v5.xlsx")
    ap.add_argument("--local", default="raw/local_250m_byevent_usgs.csv")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--gen-seed", type=int, default=0, help="합성 자료를 만들 때 쓰는 실제 자료 학습의 seed")
    ap.add_argument("--obs-offset", type=float, default=2.0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out-dir", default="outputs/disasternet/synth_D2")
    a = ap.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    kw = {"bd_var_learn": True}

    table = load_table(a.input)
    x = expand(table, a.local, bd_off=True, usgs_unc=True)

    # ① 실제 자료로 학습
    res = fit(x, a.gen_seed, a.epochs, a.obs_offset, kw)
    q = muni_probs(res.predictions, x)
    prm = res.params.pivot(index="group", columns="param", values="value")
    t = table.assign(muni_code=table["muni_code"].astype(str).str.zfill(5)).merge(q, on=["event_idx", "muni_code"])

    # ② 합성 정답, ③ 합성 DPM
    rng = np.random.default_rng(a.gen_seed)
    t["z_BD"] = (rng.random(len(t)) < t["p_bd"]).astype(int)
    for dom, (lab, pc) in LABELS.items():
        t[f"z_{dom}"] = 0
        t[f"합성_{lab}"] = np.nan
        for e, g in t.groupby("event_idx"):
            gl = g[g[lab].notna()]
            n = int((gl[lab] == 1).sum())
            if n == 0 or n == len(gl):
                continue
            thr = gl[pc].sort_values(ascending=False).iloc[n - 1]
            t.loc[g.index, f"z_{dom}"] = (g[pc] >= thr).astype(int)
            top = gl[pc].sort_values(ascending=False).index[:n]
            t.loc[gl.index, f"합성_{lab}"] = 0
            t.loc[top, f"합성_{lab}"] = 1
    logx = np.zeros(len(t))
    for i, r in t.iterrows():
        p = prm.loc[INDEX_TO_EVENT[r["event_idx"]]]
        mu = p["w_0x"] + p["w_LSx"] * r["z_LS"] + p["w_LQx"] * r["z_LQ"] + p["w_BDx"] * r["z_BD"]
        logx[t.index.get_loc(i)] = mu + abs(p["w_ex"]) * rng.standard_normal()
    t["합성_DPM"] = np.exp(logx) - a.obs_offset
    t.to_csv(out / "synthetic_truth.csv", index=False, encoding="utf-8-sig")
    prm.to_csv(out / "gen_params.csv", encoding="utf-8-sig")

    # ④ 합성 DPM으로 다시 학습
    tab2 = table.copy()
    m = t.set_index(["event_idx", "muni_code"])["합성_DPM"]
    tab2["DPM_배경차"] = [m[(e, str(c).zfill(5))] for e, c in zip(tab2["event_idx"], tab2["muni_code"])]
    x2 = expand(tab2, a.local, bd_off=True, usgs_unc=True)
    jobs = [(x2, s, a.epochs, a.obs_offset, kw) for s in a.seeds]
    if a.jobs > 1:
        with get_context("fork").Pool(a.jobs) as pool:
            fits = pool.map(_job, jobs)
    else:
        fits = [_job(j) for j in jobs]

    rows = []
    for s, qq in fits:
        qq.to_csv(out / f"predictions_seed{s}.csv", index=False, encoding="utf-8-sig")
        d = t.drop(columns=["p_ls", "p_lq", "p_bd"]).merge(qq, on=["event_idx", "muni_code"])
        sc = {"LS": {"USGS": "LS_prior_평균", "합성 DPM": "합성_DPM", "실제 자료 사후(생성용)": "p_ls_gen", "DisasterNet": "p_ls"},
              "LQ": {"USGS": "LQ_prior_최대", "합성 DPM": "합성_DPM", "실제 자료 사후(생성용)": "p_lq_gen", "DisasterNet": "p_lq"}}
        d = d.merge(q.rename(columns={"p_ls": "p_ls_gen", "p_lq": "p_lq_gen", "p_bd": "p_bd_gen"}), on=["event_idx", "muni_code"])
        at = auc_table(d, sc, {"LS": "합성_LS_정답", "LQ": "합성_LQ_정답"})
        at["seed"] = s
        rows.append(at)
    allt = pd.concat(rows)
    allt.to_csv(out / "auc_synthetic.csv", index=False, encoding="utf-8-sig")
    cols = ["USGS", "합성 DPM", "실제 자료 사후(생성용)", "DisasterNet"]
    ev = allt.groupby(["dom", "이벤트"])[cols + ["w"]].mean()
    print(ev.round(3).to_string())
    for lab, mask in [("LS", ev.index.get_level_values(0) == "LS"), ("LQ", ev.index.get_level_values(0) == "LQ"), ("전체", slice(None))]:
        g = ev[mask]
        print(lab, {c: round(float(np.average(g[c], weights=g["w"])), 3) for c in cols})


if __name__ == "__main__":
    main()
