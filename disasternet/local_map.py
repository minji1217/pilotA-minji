"""
화소(250 m 칸) 단위 끄기 지도 — 원본 DisasterNet의 LOCAL 지도를 일본 자료로 만든다.

원본은 화소마다 "산사태만 / 액상화만 가능"을 코드로 정해 불가능한 노드를 0으로 고정한다.
여기서는 국토수치정보 경사(G04-d, 250 m)와 토지이용(L03-b, 100 m)으로 칸마다 경우를 나눈다.

경우 (기준은 결과를 보기 전에 정함)
  1 비탈   경사 ≥ 10°            산사태 켬 · 액상화 끔
  2 평지   경사 < 5°             산사태 끔 · 액상화 켬
  3 중간   5° ≤ 경사 < 10°       둘 다 켬 (확실한 곳만 끈다)
  0 물     칸 안 100 m 토지이용의 절반 이상이 하천·호소(1100) 또는 해수역(1500)   둘 다 끔
  경사 값이 없는 칸(unknown)은 0으로 둔다.
  건물 피해: 칸 안 100 m 토지이용에 건물용지(0700)가 하나라도 있으면 켬, 없으면 끔
             (원본에서 OSM 건물 윤곽이 화소에 걸치면 켜는 것과 같은 기준).

칸 → 시정촌: 칸 중심점이 들어가는 시정촌 경계 (terrain_fractions와 같은 규칙).

    python -m disasternet.local_map --ksj-dir <받은 폴더> --input raw/DisasterNet_입력_v5.xlsx \
        --out raw/local_250m.csv
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from disasternet.terrain_fractions import (
    SLOPE_LQ, SLOPE_LS, _read_zip_table, assign, load_boundaries, mesh100_centers, mesh250_centers,
)

WATER = {"1100", "1500"}
BUILDING = "0700"
CASE_NAMES = {0: "물·값 없음", 1: "비탈", 2: "평지", 3: "중간"}
ON = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}   # 경우 → (산사태 켬, 액상화 켬)


def cell250_key(lon, lat):
    """경위도 → 250 m 칸 번호. 5차 메시는 위도 7.5″, 경도 11.25″ 간격으로 원점에 맞춰져 있다."""
    return np.floor(lat * 3600 / 7.5).astype(np.int64) * 10**6 + np.floor(lon * 3600 / 11.25).astype(np.int64)


def main(argv=None):
    ap = argparse.ArgumentParser(description="250 m 칸 단위 끄기 지도")
    ap.add_argument("--ksj-dir", required=True, help="n03/, l03/, g04/ 폴더가 있는 곳")
    ap.add_argument("--input", default="raw/DisasterNet_입력_v5.xlsx")
    ap.add_argument("--out", default="raw/local_250m.csv")
    a = ap.parse_args(argv)

    table = pd.read_excel(a.input, sheet_name="입력")
    table["code"] = table["시정촌코드"].astype(int).astype(str).str.zfill(5)
    bounds = load_boundaries(a.ksj_dir, table)
    minx, miny, maxx, maxy = bounds.total_bounds

    cells = []
    for z in sorted(glob.glob(f"{a.ksj_dir}/g04/*.zip")):
        t = _read_zip_table(z, ["G04d_001", "G04d_010"])
        lon, lat = mesh250_centers(t["G04d_001"].astype(np.int64))
        box = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
        t, lon, lat = t[box].reset_index(drop=True), lon[box], lat[box]
        idx, ev, code = assign(lon, lat, bounds)
        cells.append(pd.DataFrame({
            "이벤트": ev, "시정촌코드": code, "mesh250": t["G04d_001"].to_numpy()[idx],
            "lon": lon[idx], "lat": lat[idx],
            "slope": pd.to_numeric(t["G04d_010"], errors="coerce").to_numpy()[idx],
        }))
    cells = pd.concat(cells, ignore_index=True)
    cells["key"] = cell250_key(cells["lon"].to_numpy(), cells["lat"].to_numpy())

    # 100 m 토지이용을 250 m 칸에 모아 물 비율을 낸다 (칸 번호는 100 m 칸 중심으로)
    water = []
    keys = set(cells["key"])
    for z in sorted(glob.glob(f"{a.ksj_dir}/l03/*.zip")):
        t = _read_zip_table(z, ["L03b_001", "L03b_002"])
        lon, lat = mesh100_centers(t["L03b_001"].astype(np.int64))
        k = cell250_key(lon, lat)
        m = np.isin(k, list(keys))
        lu = t["L03b_002"].to_numpy()[m]
        d = pd.DataFrame({"key": k[m], "water": np.isin(lu, list(WATER)), "bld": lu == BUILDING})
        water.append(d.groupby("key").agg(water=("water", "sum"), bld=("bld", "sum"), size=("water", "size")))
    w = pd.concat(water).groupby(level=0).sum()
    cells = cells.merge(pd.DataFrame({"water_frac": w["water"] / w["size"], "bld_frac": w["bld"] / w["size"]}),
                        left_on="key", right_index=True, how="left")

    s = cells["slope"]
    case = np.select([s >= SLOPE_LS, s < SLOPE_LQ, s.notna()], [1, 2, 3], default=0)
    case[(cells["water_frac"] >= 0.5).to_numpy()] = 0
    cells["경우"] = case
    cells["산사태_켬"] = [ON[c][0] for c in case]
    cells["액상화_켬"] = [ON[c][1] for c in case]
    cells["건물_켬"] = (cells["bld_frac"].fillna(0) > 0).astype(int)
    cells = cells.drop(columns="key")
    cells.to_csv(a.out, index=False, encoding="utf-8-sig")

    summary = cells.groupby(["이벤트", "경우"]).size().unstack(fill_value=0).rename(columns=CASE_NAMES)
    summary["전체"] = summary.sum(axis=1)
    print(summary.to_string())
    print("건물 피해 켬 비율:", cells.groupby("이벤트")["건물_켬"].mean().round(3).to_dict())
    print(f"저장: {a.out} ({len(cells):,}칸)")


if __name__ == "__main__":
    main()
