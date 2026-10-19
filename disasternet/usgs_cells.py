"""
끄기 지도(local_map)의 250 m 칸마다 USGS Ground Failure 확률과 불확실도 값을 붙인다.

원본 DisasterNet은 USGS 불확실도 지도(qvar)를 두 군데 쓴다.
  (가) qvar == 0 또는 999 인 화소는 산사태·액상화를 끈다 (USGS가 계산하지 않았거나 확률 0).
  (나) 나머지 화소는 r^0 분포의 폭으로 쓴다: std = √qvar.
여기서는 칸 중심이 들어가는 USGS 래스터 칸의 값을 읽는다.

파일 (지진마다 USGS 이벤트 페이지의 ground-failure 제품)
  산사태  jessee_2017_model.tif / jessee_2017_std.tif       (2016·2018)
          jessee_2018_model.tif / jessee_2018_beta_sigma.tif (2021·2024)
  액상화  zhu_2017_general_model.tif / zhu_2017_general_std.tif 또는 _beta_sigma.tif
nodata는 999 (옛 판) 또는 NaN (2024 노토). 래스터 범위 밖 칸도 nodata로 둔다.

    python -m disasternet.usgs_cells --local raw/local_250m_byevent.csv --usgs-dir <받은 폴더> \
        --out raw/local_250m_byevent_usgs.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EVENT_ID = {"2016 구마모토": "us20005iis", "2018 오사카": "us1000eu1c", "2018 훗카이도": "us2000h8ty",
            "2021 후쿠시마": "us6000dher", "2024 노토반도": "us6000m0xl"}
NODATA = 999.0


def _pick(folder, stems):
    for s in stems:
        if (folder / s).exists():
            return folder / s
    raise FileNotFoundError(f"{folder}: {stems} 없음")


def sample(path, lon, lat):
    """칸 중심이 들어가는 래스터 칸 값. 범위 밖·nodata·NaN은 999로 통일한다."""
    import rasterio  # 래스터를 읽을 때만 필요 (끄기 판정 usgs_off만 쓰는 곳에서는 설치 안 해도 되게)
    with rasterio.open(path) as r:
        a = r.read(1).astype("float64")
        rows, cols = rasterio.transform.rowcol(r.transform, lon, lat)
        rows, cols = np.asarray(rows), np.asarray(cols)
        inside = (rows >= 0) & (rows < r.height) & (cols >= 0) & (cols < r.width)
        v = np.full(len(lon), NODATA)
        v[inside] = a[rows[inside], cols[inside]]
        if r.nodata is not None and not np.isnan(r.nodata):
            v[v == r.nodata] = NODATA
    v[np.isnan(v)] = NODATA
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description="250 m 칸에 USGS 확률·불확실도 붙이기")
    ap.add_argument("--local", default="raw/local_250m_byevent.csv")
    ap.add_argument("--usgs-dir", required=True, help="이벤트 ID 폴더(us20005iis/ …)가 있는 곳")
    ap.add_argument("--out", default="raw/local_250m_byevent_usgs.csv")
    a = ap.parse_args(argv)

    cells = pd.read_csv(a.local, dtype={"시정촌코드": str})
    parts = []
    for ev, g in cells.groupby("이벤트", sort=False):
        f = Path(a.usgs_dir) / EVENT_ID[ev]
        lon, lat = g["lon"].to_numpy(), g["lat"].to_numpy()
        g = g.copy()
        g["us_ls"] = sample(_pick(f, ["jessee_2017_model.tif", "jessee_2018_model.tif"]), lon, lat)
        g["sd_ls"] = sample(_pick(f, ["jessee_2017_std.tif", "jessee_2018_beta_sigma.tif"]), lon, lat)
        g["us_lq"] = sample(_pick(f, ["zhu_2017_general_model.tif"]), lon, lat)
        g["sd_lq"] = sample(_pick(f, ["zhu_2017_general_std.tif", "zhu_2017_general_beta_sigma.tif"]), lon, lat)
        parts.append(g)
        for k in ("ls", "lq"):
            off = usgs_off(g[f"us_{k}"], g[f"sd_{k}"])
            print(f"{ev} {k.upper()}: USGS 끔 {off.mean():.1%} (nodata {(g[f'sd_{k}'] == NODATA).mean():.1%})", flush=True)
    out = pd.concat(parts).sort_index()
    out.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"저장: {a.out} ({len(out):,}칸)")


def usgs_off(us, sd, rule="ours"):
    """원본 규칙: 불확실도가 0 또는 999면 끈다. 2024 노토 판은 확률 0인 칸에도 불확실도가 작게 남아 있어
    확률이 0이거나 nodata인 칸도 함께 끈다 (옛 판에서는 불확실도 0 ⇔ 확률 0이라 결과가 같다).
    rule="paper"면 공식 코드 그대로 불확실도만 본다."""
    us, sd = np.asarray(us), np.asarray(sd)
    off = (sd == 0) | (sd == NODATA)
    return off if rule == "paper" else off | (us == 0) | (us == NODATA)


if __name__ == "__main__":
    main()
