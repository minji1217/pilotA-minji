"""
시정촌 한 줄(가상 화소 없음)에 D′의 설정만 얹는다: 끄기 없음·soft에서 "가상 화소 효과"와 "D′ 설정 효과"를 떼어 본다.

D′에서 가져오는 것 (시정촌 단위로 줄여서)
  ① USGS 끄기   : 시정촌 안에 USGS가 켠 칸(확률 0·불확실도 0·999가 아닌 칸)이 하나도 없으면 그 노드를 0으로 고정
  ② r⁰ 분산     : 산사태·액상화 = 시정촌 안 USGS 켠 칸들의 불확실도 평균 (D′와 같은 값 규칙, 원본 std = √qvar)
  ③ 건물 분산   : 원본처럼 학습 [1, 5] (--bd-var-learn과 같음)
가져오지 않는 것: 경사·건물용지로 칸을 나눠 끄기 (가상 화소). 출발 확률·관측·학습 설정은 run_input_v5와 같다.
--terrain을 주면 soft처럼 prior 로짓에 log(지형 비율)을 더한다.

    python -m disasternet.run_muni_unc --out-dir outputs/disasternet/input_v5_muni_unc --jobs 5
    python -m disasternet.run_muni_unc --terrain raw/terrain_fractions.csv --out-dir outputs/disasternet/input_v5_muni_unc_soft --jobs 5
"""
import argparse
from multiprocessing import get_context
from pathlib import Path

import pandas as pd
import torch

from schema import DTYPE

from .model import fit_disasternet
from .run_input_v5 import attach_terrain, load_table, to_rows
from .usgs_cells import usgs_off


def muni_usgs(local_path):
    """칸별 USGS 확률·불확실도 → 시정촌별 (켠 칸 수, 불확실도 평균). 물 칸(경우 0)은 뺀다."""
    loc = pd.read_csv(local_path, dtype={"시정촌코드": str})
    loc = loc[loc["경우"] > 0].copy()
    for k in ("ls", "lq"):
        on = ~usgs_off(loc[f"us_{k}"], loc[f"sd_{k}"])
        loc[f"on_{k}"] = on.astype(int)
        loc[f"v_{k}"] = loc[f"sd_{k}"].where(on)
    g = loc.groupby(["이벤트", "시정촌코드"]).agg(n_on_ls=("on_ls", "sum"), n_on_lq=("on_lq", "sum"),
                                              var_ls=("v_ls", "mean"), var_lq=("v_lq", "mean")).reset_index()
    g["muni_code"] = g["시정촌코드"].str.zfill(5)
    return g.drop(columns="시정촌코드")


def build_rows(df, dpm_col, lq_col):
    rows = to_rows(df, dpm_col, mode="soft", lq_col=lq_col)       # frac_* 열이 있으면 soft의 log_frac, 없으면 없음
    t = lambda c: torch.as_tensor(df[c].to_numpy(dtype="float64").copy(), dtype=DTYPE)
    rows.off = {"LS": torch.as_tensor((df["n_on_ls"] == 0).to_numpy()),
                "LQ": torch.as_tensor((df["n_on_lq"] == 0).to_numpy()),
                "BD": torch.zeros(len(df), dtype=torch.bool)}
    # 꺼진 노드는 분산이 쓰이지 않으므로 1을 채운다 (run_cellmix.to_rows와 같은 규칙)
    rows.r0_var = {"LS": t("var_ls").nan_to_num(1.0), "LQ": t("var_lq").nan_to_num(1.0)}
    return rows


def _fit_one(job):
    df, dpm_col, offset, seed, epochs, out_dir, lq_col = job
    torch.set_num_threads(1)
    fit = fit_disasternet(build_rows(df, dpm_col, lq_col), seed=seed, epochs=epochs, obs_offset=offset,
                          verbose=False, bd_var_learn=True)
    p = fit.predictions[["event_idx", "muni_code", "p_ls", "p_lq", "p_bd"]].copy()
    p.insert(0, "seed", seed)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p.to_csv(Path(out_dir) / f"predictions_seed{seed}.csv", index=False, encoding="utf-8-sig")
    print(f"seed {seed} 완료", flush=True)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description="시정촌 한 줄 + D′ 설정 (USGS 끄기·분산, 건물 분산 학습)")
    ap.add_argument("--input", default="raw/DisasterNet_입력_v5.xlsx")
    ap.add_argument("--local", default="raw/local_250m_byevent_usgs.csv", help="usgs_cells 출력 (칸별 us_*/sd_*)")
    ap.add_argument("--terrain", default=None, help="주면 soft처럼 prior 로짓에 log(지형 비율)")
    ap.add_argument("--dpm-col", default="DPM_배경차")
    ap.add_argument("--obs-offset", type=float, default=2.0)
    ap.add_argument("--lq-prior-col", default="LQ_prior_최대")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out-dir", default="outputs/disasternet/input_v5_muni_unc")
    a = ap.parse_args(argv)

    df = load_table(a.input)
    if a.terrain:
        df = attach_terrain(df, a.terrain)
    df = df.merge(muni_usgs(a.local), on=["이벤트", "muni_code"], how="left", validate="one_to_one")
    if df[["n_on_ls", "n_on_lq"]].isna().any().any():
        raise ValueError(f"칸 지도가 없는 시정촌: {df.loc[df['n_on_ls'].isna(), '시정촌명'].tolist()}")
    print(f"시정촌 {len(df)}곳 · USGS로 산사태 끈 곳 {(df.n_on_ls == 0).sum()} · 액상화 끈 곳 {(df.n_on_lq == 0).sum()}"
          f" · 지형 비율 {'있음 (soft)' if a.terrain else '없음'}", flush=True)
    jobs = [(df, a.dpm_col, a.obs_offset, s, a.epochs, a.out_dir, a.lq_prior_col) for s in a.seeds]
    if a.jobs > 1:
        with get_context("fork").Pool(a.jobs) as pool:
            pool.map(_fit_one, jobs)
    else:
        for j in jobs:
            _fit_one(j)
    print(f"저장: {a.out_dir}/")


if __name__ == "__main__":
    main()
