"""
시정촌별 지형 비율 계산 — DisasterNet의 가지치기(LOCAL 지도, 건물 footprint)를 시정촌 단위로 옮기기 위한 자료.

원본 DisasterNet은 픽셀마다 "비탈이면 액상화 불가, 평지면 산사태 불가, 건물 없으면 건물피해 불가"로
해당 노드를 0으로 고정한다. 시정촌에는 비탈·평지·건물이 섞여 있어 끌 수는 없으므로,
"그 재해가 가능한 땅의 비율"을 계산해 prior를 그만큼 기울이는 데 쓴다.

자료 (국토수치정보, nlftp.mlit.go.jp)
  N03      행정구역 (지진 시점 연도: 2016·2018·2021·2024)
  L03-b-21 토지이용세분메시 100 m (2021)
  G04-d-11 표고·경사도 5차메시 250 m (2011)

정의 (결과를 보기 전에 정함)
  frac_LS  비탈 비율   : 250 m 격자 평균 경사 ≥ 10°
  frac_LQ  평지 비율   : 250 m 격자 평균 경사 < 5°
  frac_BD  건물용지 비율 : 100 m 격자 토지이용 = 0700 건물용지
  분모는 육지 격자 (토지이용 1500 해수역·0000 범위 밖 제외 / 경사 격자는 시정촌 안에 중심이 있는 전부)

격자 → 시정촌: 격자 중심점이 들어가는 시정촌 경계. 정령지정도시는 구를 합쳐 시 하나로(통합표와 같게).

    python -m disasternet.terrain_fractions --ksj-dir <받은 폴더> --input raw/DisasterNet_입력_v5.xlsx \
        --out raw/terrain_fractions.csv
"""
import argparse
import glob
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

N03_SOURCES = {
    "2016 구마모토": ["N03-160101_43"],
    "2018 오사카": ["N03-180101_26", "N03-180101_27"],
    "2018 훗카이도": ["N03-180101_01"],
    "2021 후쿠시마": ["N03-20210101_04", "N03-20210101_07"],
    "2024 노토반도": ["N03-20240101_15", "N03-20240101_16", "N03-20240101_17"],
}
DESIGNATED_CITIES = {"43100", "26100", "27100", "01100", "04100", "15100"}
SLOPE_LS, SLOPE_LQ = 10.0, 5.0
LANDUSE_BUILDING, LANDUSE_SEA, LANDUSE_OUT = "0700", "1500", "0000"


def mesh100_centers(codes):
    """10자리 100 m 메시 코드 → 중심 (경도, 위도)."""
    c = np.asarray(codes, dtype=np.int64)
    p1, p2 = c // 10**6 // 100, c // 10**6 % 100            # 1차: 위도·경도 2자리
    s = c // 10**4 % 100; t = c // 100 % 100; u = c % 100    # 2차·3차·100 m
    lat = p1 / 1.5 + (s // 10) * (5 / 60) + (t // 10) * (30 / 3600) + (u // 10) * (3 / 3600) + 1.5 / 3600
    lon = p2 + 100 + (s % 10) * (7.5 / 60) + (t % 10) * (45 / 3600) + (u % 10) * (4.5 / 3600) + 2.25 / 3600
    return lon, lat


def mesh250_centers(codes):
    """10자리 5차(250 m) 메시 코드 → 중심. 4차·5차 자리는 1 남서 / 2 남동 / 3 북서 / 4 북동."""
    c = np.asarray(codes, dtype=np.int64)
    m3 = c // 100; q4 = c // 10 % 10; q5 = c % 10
    p1, p2 = m3 // 10**4 // 100, m3 // 10**4 % 100
    s = m3 // 100 % 100; t = m3 % 100
    lat = p1 / 1.5 + (s // 10) * (5 / 60) + (t // 10) * (30 / 3600)
    lon = p2 + 100 + (s % 10) * (7.5 / 60) + (t % 10) * (45 / 3600)
    lat = lat + ((q4 - 1) // 2) * 15 / 3600 + ((q5 - 1) // 2) * 7.5 / 3600 + 3.75 / 3600
    lon = lon + ((q4 - 1) % 2) * 22.5 / 3600 + ((q5 - 1) % 2) * 11.25 / 3600 + 5.625 / 3600
    return lon, lat


def load_boundaries(ksj_dir, table):
    need = table.groupby("이벤트")["code"].apply(set).to_dict()
    out = []
    for ev, srcs in N03_SOURCES.items():
        parts = []
        for s in srcs:
            f = glob.glob(f"{ksj_dir}/n03/{s}/**/*.shp", recursive=True)[0]
            g = gpd.read_file(f)
            col = [c for c in g.columns if c.startswith("N03_007")][0]
            parts.append(g.rename(columns={col: "code"})[["code", "geometry"]].dropna(subset=["code"]))
        g = pd.concat(parts)
        city = g["code"].str[:3] + "00"
        ward = city.isin(DESIGNATED_CITIES) & (g["code"] != city)
        g.loc[ward, "code"] = city[ward]
        g = g[g["code"].isin(need[ev])].dissolve(by="code").reset_index()
        missing = need[ev] - set(g["code"])
        if missing:
            raise ValueError(f"{ev}: 경계가 없는 시정촌 {sorted(missing)}")
        g["이벤트"] = ev
        out.append(g)
    return gpd.GeoDataFrame(pd.concat(out, ignore_index=True), crs=out[0].crs)


def _read_zip_table(path, keep):
    """메시 zip에서 속성표(dbf)만 읽는다."""
    z = zipfile.ZipFile(path)
    tmp = Path(path).with_suffix("")
    tmp.mkdir(exist_ok=True)
    names = [n for n in z.namelist() if n.endswith((".dbf", ".shp", ".shx", ".prj"))]
    z.extractall(tmp, members=names)
    shp = [n for n in names if n.endswith(".shp")][0]
    return gpd.read_file(tmp / shp, ignore_geometry=True, columns=keep)


def assign(lon, lat, bounds):
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs=bounds.crs)
    j = gpd.sjoin(pts, bounds[["이벤트", "code", "geometry"]], predicate="within", how="inner")
    return j.index.to_numpy(), j["이벤트"].to_numpy(), j["code"].to_numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description="시정촌별 비탈·평지·건물용지 비율")
    ap.add_argument("--ksj-dir", required=True, help="n03/, l03/, g04/ 폴더가 있는 곳")
    ap.add_argument("--input", default="raw/DisasterNet_입력_v5.xlsx")
    ap.add_argument("--out", default="raw/terrain_fractions.csv")
    a = ap.parse_args(argv)

    table = pd.read_excel(a.input, sheet_name="입력")
    table["code"] = table["시정촌코드"].astype(int).astype(str).str.zfill(5)
    bounds = load_boundaries(a.ksj_dir, table)
    minx, miny, maxx, maxy = bounds.total_bounds

    lu_counts, slope_rows = [], []
    for z in sorted(glob.glob(f"{a.ksj_dir}/l03/*.zip")):
        t = _read_zip_table(z, ["L03b_001", "L03b_002"])
        lon, lat = mesh100_centers(t["L03b_001"].astype(np.int64))
        box = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
        t, lon, lat = t[box].reset_index(drop=True), lon[box], lat[box]
        idx, ev, code = assign(lon, lat, bounds)
        d = pd.DataFrame({"이벤트": ev, "code": code, "lu": t["L03b_002"].to_numpy()[idx]})
        lu_counts.append(d.groupby(["이벤트", "code", "lu"]).size())
        print(f"토지이용 {Path(z).stem}: 시정촌 안 격자 {len(d):,}", flush=True)
    for z in sorted(glob.glob(f"{a.ksj_dir}/g04/*.zip")):
        t = _read_zip_table(z, ["G04d_001", "G04d_010"])
        lon, lat = mesh250_centers(t["G04d_001"].astype(np.int64))
        box = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
        t, lon, lat = t[box].reset_index(drop=True), lon[box], lat[box]
        idx, ev, code = assign(lon, lat, bounds)
        slope = pd.to_numeric(t["G04d_010"], errors="coerce").to_numpy()[idx]
        slope_rows.append(pd.DataFrame({"이벤트": ev, "code": code, "slope": slope}))
        print(f"경사도 {Path(z).stem}: 시정촌 안 격자 {len(idx):,}", flush=True)

    lu = pd.concat(lu_counts).groupby(level=[0, 1, 2]).sum().unstack(fill_value=0)
    land = lu.drop(columns=[c for c in (LANDUSE_SEA, LANDUSE_OUT) if c in lu.columns]).sum(axis=1)
    out = pd.DataFrame({
        "n_landuse_100m": land,
        "frac_BD": lu.get(LANDUSE_BUILDING, 0) / land,
        "frac_forest": lu.get("0500", 0) / land,
        "frac_paddy": lu.get("0100", 0) / land,
    })
    sl = pd.concat(slope_rows).dropna(subset=["slope"])
    g = sl.groupby(["이벤트", "code"])["slope"]
    out = out.join(pd.DataFrame({
        "n_slope_250m": g.size(),
        "frac_LS": g.apply(lambda s: float((s >= SLOPE_LS).mean())),
        "frac_LQ": g.apply(lambda s: float((s < SLOPE_LQ).mean())),
        "mean_slope": g.mean(),
    }), how="outer")
    out = out.reset_index().rename(columns={"code": "시정촌코드"})
    keyed = table[["이벤트", "code", "시정촌명"]].rename(columns={"code": "시정촌코드"})
    out = keyed.merge(out, on=["이벤트", "시정촌코드"], how="left")
    if out[["frac_LS", "frac_LQ", "frac_BD"]].isna().any().any():
        bad = out[out[["frac_LS", "frac_LQ", "frac_BD"]].isna().any(axis=1)]
        raise ValueError(f"비율이 비는 시정촌: {bad[['이벤트', '시정촌명']].to_dict('records')}")
    out.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(out.describe().round(3).to_string())
    print(f"저장: {a.out} ({len(out)}행)")


if __name__ == "__main__":
    main()
