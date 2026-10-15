"""
시정촌 자료 + 칸 단위 끄기 지도 — 화소 래스터 없이 원본 가지치기를 시정촌 안에서 흉내 낸다.

시정촌 하나를 끄기 지도(local_map)의 경우별 "가상 화소" 행으로 나눈다.
  비탈(1)  산사태 켬 · 액상화 끔       평지(2)  산사태 끔 · 액상화 켬       중간(3)  둘 다 켬
  물(0)    칸은 빼고 나머지 비율을 다시 1로 맞춘다.
가상 화소 행은 그 시정촌의 DPM·USGS·PGV를 그대로 쓰고(화소 값이 없으므로), 끈 노드는 0으로 고정한다.
손실은 칸 비율로 가중해 시정촌마다 합이 1이 되게 하고, 시정촌 확률은 칸 비율 가중 평균으로 되돌린다.
--bd-off를 주면 건물용지가 없는 칸(local_map의 건물_켬=0)은 건물 피해도 0으로 고정한다 (없으면 모든 칸에서 켬).

    python -m disasternet.run_cellmix --local raw/local_250m.csv --out-dir outputs/disasternet/input_v5_cellmix --jobs 4
"""
import argparse
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from schema import DTYPE

from .data import DPMRows
from .model import fit_disasternet
from .run_input_v5 import load_table

CASE_OFF = {1: (False, True), 2: (True, False), 3: (False, False)}   # 경우 → (산사태 끔, 액상화 끔)


def expand(df, local_path, bd_off=False, usgs_unc=False, alpha="muni"):
    """시정촌 → 가상 화소 행.

    usgs_unc=False: 행 = (경우, 건물). 끄기는 경우(비탈·평지·중간)와 건물용지로만.
    usgs_unc=True : 원본처럼 USGS 불확실도도 쓴다 (usgs_cells로 칸마다 us_*/sd_*를 붙인 지도 필요).
                    칸마다 산사태 켬 = 비탈·중간이고 USGS 끔이 아님, 액상화 켬 = 평지·중간이고 USGS 끔이 아님.
                    행 = (산사태 켬, 액상화 켬, 건물). 행의 r^0 분산 = 켜진 칸들의 불확실도 평균 (원본 std = √qvar).
                    alpha="cells"면 행의 출발 확률도 켜진 칸들의 USGS 확률 평균, "muni"면 시정촌 값 그대로.
    """
    loc = pd.read_csv(local_path, dtype={"시정촌코드": str})
    loc = loc[loc["경우"] > 0].copy()
    loc["건물"] = loc["건물_켬"] if bd_off else 1
    if usgs_unc:
        from .usgs_cells import usgs_off
        loc["ls_on"] = (loc["경우"].isin([1, 3]) & ~usgs_off(loc["us_ls"], loc["sd_ls"])).astype(int)
        loc["lq_on"] = (loc["경우"].isin([2, 3]) & ~usgs_off(loc["us_lq"], loc["sd_lq"])).astype(int)
        keys = ["ls_on", "lq_on", "건물"]
        for k in ("ls", "lq"):
            on = loc[f"{k}_on"] == 1
            loc[f"v_{k}"] = loc[f"sd_{k}"].where(on)
            loc[f"a_{k}"] = loc[f"us_{k}"].where(on)
        agg = {"n": ("경우", "size"), "var_ls": ("v_ls", "mean"), "var_lq": ("v_lq", "mean"),
               "a_ls": ("a_ls", "mean"), "a_lq": ("a_lq", "mean")}
    else:
        keys = ["경우", "건물"]
        agg = {"n": ("경우", "size")}
    share = loc.groupby(["이벤트", "시정촌코드"] + keys).agg(**agg).reset_index()
    share["w"] = share["n"] / share.groupby(["이벤트", "시정촌코드"])["n"].transform("sum")
    share["muni_code"] = share["시정촌코드"].str.zfill(5)
    df = df.assign(muni_code=df["muni_code"].astype(str).str.zfill(5))
    cols = [c for c in share.columns if c not in ("시정촌코드", "n")]
    x = df.merge(share[cols], on=["이벤트", "muni_code"], how="left")
    if x["w"].isna().any():
        raise ValueError(f"끄기 지도에 칸이 없는 시정촌: {x.loc[x['w'].isna(), '시정촌명'].tolist()}")
    if usgs_unc:
        x["off_ls"], x["off_lq"] = x["ls_on"] == 0, x["lq_on"] == 0
        x["행"] = "s" + x["ls_on"].astype(str) + "q" + x["lq_on"].astype(str) + "b" + x["건물"].astype(str)
        if alpha == "cells":
            x["LS_prior_평균"] = x["a_ls"].fillna(0.0)
            x["LQ_prior_최대"] = x["LQ_prior_평균"] = x["a_lq"].fillna(0.0)
    else:
        x["off_ls"] = [CASE_OFF[c][0] for c in x["경우"]]
        x["off_lq"] = [CASE_OFF[c][1] for c in x["경우"]]
        x["행"] = x["경우"].astype(str) + "b" + x["건물"].astype(str)
    return x.sort_values(["event_idx", "muni_code", "행"]).reset_index(drop=True)


def to_rows(x, dpm_col, lq_col="LQ_prior_최대"):
    t = lambda c: torch.as_tensor(x[c].to_numpy(dtype="float64").copy(), dtype=DTYPE)
    off_ls = torch.as_tensor(x["off_ls"].to_numpy(dtype=bool))
    off_lq = torch.as_tensor(x["off_lq"].to_numpy(dtype=bool))
    r0_var = None
    if "var_ls" in x.columns:   # 원본: std = √qvar → 분산 = qvar. 꺼진 노드는 쓰이지 않으므로 1을 채운다.
        r0_var = {k: torch.as_tensor(x[f"var_{k.lower()}"].fillna(1.0).to_numpy(dtype="float64").copy(), dtype=DTYPE)
                  for k in ("LS", "LQ")}
    return DPMRows(
        y=t(dpm_col), pi_ls=t("LS_prior_평균"), pi_lq=t(lq_col), pgv=t("PGV"),
        event_idx=torch.as_tensor(x["event_idx"].to_numpy(), dtype=torch.long),
        municipality_code=tuple(f"{m}#{r}" for m, r in zip(x["muni_code"], x["행"])),
        model_row_idx=torch.arange(len(x)),
        off={"LS": off_ls, "LQ": off_lq, "BD": torch.as_tensor((x["건물"] == 0).to_numpy())},
        weight=t("w"),
        r0_var=r0_var,
    )


def _fit_one(job):
    x, dpm_col, offset, seed, epochs, out_dir, lq_col = job
    torch.set_num_threads(1)
    fit = fit_disasternet(to_rows(x, dpm_col, lq_col), seed=seed, epochs=epochs, obs_offset=offset, verbose=False)
    p = fit.predictions
    p[["muni_code", "행"]] = p["muni_code"].str.split("#", expand=True)
    p = p.merge(x[["event_idx", "muni_code", "행", "w"]], on=["event_idx", "muni_code", "행"])
    for k in ("p_ls", "p_lq", "p_bd"):
        p[k] = p[k] * p["w"]
    out = p.groupby(["event_idx", "muni_code"])[["p_ls", "p_lq", "p_bd"]].sum().reset_index()
    out.insert(0, "seed", seed)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(out_dir) / f"predictions_seed{seed}.csv", index=False, encoding="utf-8-sig")
    p.to_csv(Path(out_dir) / f"cells_seed{seed}.csv", index=False, encoding="utf-8-sig")
    print(f"seed {seed} 완료", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="시정촌 자료 + 칸 단위 끄기 지도")
    ap.add_argument("--input", default="raw/DisasterNet_입력_v5.xlsx")
    ap.add_argument("--local", default="raw/local_250m.csv")
    ap.add_argument("--dpm-col", default="DPM_배경차")
    ap.add_argument("--obs-offset", type=float, default=2.0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out-dir", default="outputs/disasternet/input_v5_cellmix")
    ap.add_argument("--bd-off", action="store_true", help="건물용지가 없는 칸은 건물 피해도 0으로 고정")
    ap.add_argument("--lq-prior-col", default="LQ_prior_최대",
                    help="액상화 출발 확률 열. 기본은 최대 (USGS 칸 대부분이 바닥값이라 평균은 위험한 칸을 희석함). 평균: LQ_prior_평균")
    a = ap.parse_args(argv)

    x = expand(load_table(a.input), a.local, a.bd_off)
    print(f"시정촌 {x.groupby(['event_idx', 'muni_code']).ngroups}곳 → 가상 화소 행 {len(x)}개 "
          f"(행 종류별 {x['행'].value_counts().sort_index().to_dict()})", flush=True)
    jobs = [(x, a.dpm_col, a.obs_offset, s, a.epochs, a.out_dir, a.lq_prior_col) for s in a.seeds]
    if a.jobs > 1:
        with get_context("fork").Pool(a.jobs) as pool:
            pool.map(_fit_one, jobs)
    else:
        for j in jobs:
            _fit_one(j)


if __name__ == "__main__":
    main()
