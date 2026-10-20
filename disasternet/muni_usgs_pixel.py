"""
시정촌별 USGS 불확실도를 사전확률 표와 같은 방식으로 집계한다.

사전확률 표(재난프로젝트_시정촌별_USGS.xlsx)를 만든 방식
  - USGS 래스터 칸의 '중심점'이 시정촌 폴리곤(N03, 지진 발생시점 판, 정령지정도시는 구를 합침) 안이면 채택
  - 부분면적 가중 없음, 물도 포함
  - 산사태·액상화: 채택 칸의 산술평균 · 최댓값

여기서는 같은 규칙으로 불확실도 래스터(jessee/zhu의 std 또는 beta_sigma)를 집계한다. 확률 래스터와 불확실도
래스터는 같은 격자다.
  n_on_* : 채택 칸 중 불확실도가 0·999·nodata가 아닌 칸 수 (논문 기준 끄기: 0이면 그 재해를 끔)
  var_*  : 그 켜진 칸들의 불확실도 산술평균 (r⁰ 분산)
  확인용으로 확률의 산술평균(us_ls_mean)·최댓값(us_lq_max)도 같이 낸다.

    python -m disasternet.muni_usgs_pixel --ksj-dir <n03 폴더가 있는 곳> --usgs-dir <USGS 이벤트 폴더> \
        --out raw/muni_usgs_pixel.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from disasternet.terrain_fractions import assign, load_boundaries
from disasternet.usgs_cells import EVENT_ID, _pick
from disasternet.run_input_v5 import load_table

LAYERS = {"us_ls": ["jessee_2017_model.tif", "jessee_2018_model.tif"],
          "sd_ls": ["jessee_2017_std.tif", "jessee_2018_beta_sigma.tif"],
          "us_lq": ["zhu_2017_general_model.tif"],
          "sd_lq": ["zhu_2017_general_std.tif", "zhu_2017_general_beta_sigma.tif"]}


def pixel_table(path, bounds_ev):
    """경계 범위 안 래스터 칸들의 (중심 경위도, 값). nodata·NaN은 999."""
    x0, y0, x1, y1 = bounds_ev.total_bounds
    with rasterio.open(path) as r:
        win = rasterio.windows.from_bounds(x0, y0, x1, y1, r.transform).round_offsets().round_lengths()
        a = r.read(1, window=win, boundless=True, fill_value=np.nan).astype("float64")
        tr = r.window_transform(win)
        if r.nodata is not None and not np.isnan(r.nodata):
            a[a == r.nodata] = np.nan
    rows, cols = np.indices(a.shape)
    lon, lat = rasterio.transform.xy(tr, rows.ravel(), cols.ravel(), offset="center")
    v = a.ravel(); v[np.isnan(v)] = 999.0
    return np.asarray(lon), np.asarray(lat), v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="raw/DisasterNet_입력_v5.xlsx")
    ap.add_argument("--ksj-dir", required=True)
    ap.add_argument("--usgs-dir", required=True)
    ap.add_argument("--out", default="raw/muni_usgs_pixel.csv")
    a = ap.parse_args(argv)

    tab = load_table(a.input).rename(columns={"muni_code": "code"})
    bounds = load_boundaries(a.ksj_dir, tab[["이벤트", "code"]])
    out = []
    for ev, b in bounds.groupby("이벤트"):
        f = Path(a.usgs_dir) / EVENT_ID[ev]
        res = {}
        for col, names in LAYERS.items():
            lon, lat, v = pixel_table(_pick(f, names), b)
            idx, _, code = assign(lon, lat, b)
            res[col] = pd.DataFrame({"code": code, col: v[idx]})
        g = pd.DataFrame({"code": sorted(b["code"])})
        for k in ("ls", "lq"):
            sd = res[f"sd_{k}"]
            on = sd[(sd[f"sd_{k}"] != 0) & (sd[f"sd_{k}"] < 999)]
            g = g.merge(sd.groupby("code").size().rename(f"n_pix_{k}").reset_index(), on="code", how="left")
            g = g.merge(on.groupby("code")[f"sd_{k}"].agg(["size", "mean"]).rename(
                columns={"size": f"n_on_{k}", "mean": f"var_{k}"}).reset_index(), on="code", how="left")
            p = res[f"us_{k}"]; p = p[p[f"us_{k}"] < 999]
            g = g.merge(p.groupby("code")[f"us_{k}"].agg(["mean", "max"]).rename(
                columns={"mean": f"us_{k}_mean", "max": f"us_{k}_max"}).reset_index(), on="code", how="left")
        g[["n_on_ls", "n_on_lq"]] = g[["n_on_ls", "n_on_lq"]].fillna(0).astype(int)
        g.insert(0, "이벤트", ev)
        out.append(g)
        print(ev, len(g), "산사태 끔", int((g.n_on_ls == 0).sum()), "액상화 끔", int((g.n_on_lq == 0).sum()), flush=True)
    pd.concat(out).rename(columns={"code": "muni_code"}).to_csv(a.out, index=False, encoding="utf-8-sig")
    print("저장:", a.out)


if __name__ == "__main__":
    main()
