"""250 m 칸마다 USGS 지반 피해 확률 · 불확실도를 '칸과 겹친 넓이 비율'로 평균한다 (raw/local_250m_byevent_usgs_aw.csv.gz 를 만든 코드).

usgs_cells.py(칸 중심점이 든 래스터 칸 값 하나)와 다른 점: 칸 하나를 10 x 10 = 100개의 작은 점으로 나누고,
각 점이 든 래스터 칸 값을 읽어 평균한다. 점 하나가 칸 넓이의 1%를 대표하므로 '겹친 넓이 비율 가중평균'의 근사다.

세부 규칙
- 250 m 칸 크기: 경도 1/320도 x 위도 1/480도 (3차 메시를 4등분한 250 m 메시). 점 위치 = 칸 중심 ± ((i + 0.5)/10 - 0.5) x 칸 크기, i = 0..9
- 래스터: 이벤트 폴더(EVENT_ID) 안의 파일
    us_ls = jessee_2017_model.tif (없으면 jessee_2018_model.tif)      sd_ls = jessee_2017_std.tif (없으면 jessee_2018_beta_sigma.tif)
    us_lq = zhu_2017_general_model.tif                                sd_lq = zhu_2017_general_std.tif (없으면 zhu_2017_general_beta_sigma.tif)
- 평균에서 빼는 점: 래스터 범위 밖, nodata, 값 >= 999 (USGS가 계산 안 한 곳). 값 0은 정상 값으로 평균에 넣는다
- 칸 안 100개 점이 모두 빠지면 그 칸 값은 999 (계산 안 함)
- 확률과 불확실도를 같은 규칙으로 따로 평균한다 (불확실도도 넓이 평균)
- 나머지 열(이벤트, 시정촌코드, mesh250, lon, lat, slope, water_frac, ...)은 입력 파일 그대로, 칸(행)도 그대로 (물 칸 포함)

    python -m disasternet.usgs_cells_aw --local raw/local_250m_byevent_usgs.csv --usgs-dir raw/usgs_ground_failure \
        --out raw/local_250m_byevent_usgs_aw.csv.gz
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from .usgs_cells import EVENT_ID, _pick

DLON, DLAT, K = 1 / 320, 1 / 480, 10
FILES = [("us_ls", ["jessee_2017_model.tif", "jessee_2018_model.tif"]),
         ("sd_ls", ["jessee_2017_std.tif", "jessee_2018_beta_sigma.tif"]),
         ("us_lq", ["zhu_2017_general_model.tif"]),
         ("sd_lq", ["zhu_2017_general_std.tif", "zhu_2017_general_beta_sigma.tif"])]


def area_weighted(loc: pd.DataFrame, usgs_dir: Path) -> pd.DataFrame:
    off = (np.arange(K) + 0.5) / K - 0.5
    ox, oy = np.meshgrid(off * DLON, off * DLAT)
    ox, oy = ox.ravel(), oy.ravel()
    out = []
    for ev, g in loc.groupby("이벤트"):
        f = usgs_dir / EVENT_ID[ev]
        g = g.copy()
        lon = (g.lon.to_numpy()[:, None] + ox).ravel()
        lat = (g.lat.to_numpy()[:, None] + oy).ravel()
        for col, names in FILES:
            with rasterio.open(_pick(f, names)) as r:
                a = r.read(1).astype("float64")
                nod = r.nodata
                rr, cc = rasterio.transform.rowcol(r.transform, lon, lat)
                rr, cc = np.asarray(rr), np.asarray(cc)
                ins = (rr >= 0) & (rr < r.height) & (cc >= 0) & (cc < r.width)
                v = np.full(len(lon), np.nan)
                v[ins] = a[rr[ins], cc[ins]]
                if nod is not None and not np.isnan(nod):
                    v[v == nod] = np.nan
                v = v.reshape(len(g), K * K)
                valid = ~np.isnan(v) & (v < 999)
                s = np.where(valid, v, 0).sum(1)
                n = valid.sum(1)
                g[col] = np.where(n > 0, s / np.maximum(n, 1), 999.0)
        out.append(g)
        print(ev, len(g), flush=True)
    return pd.concat(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default="raw/local_250m_byevent_usgs.csv", help="칸 목록 (usgs_cells.py 출력, 중심점 판)")
    ap.add_argument("--usgs-dir", default="raw/usgs_ground_failure", help="이벤트별 USGS 래스터 폴더")
    ap.add_argument("--out", default="raw/local_250m_byevent_usgs_aw.csv.gz")
    a = ap.parse_args(argv)
    loc = pd.read_csv(a.local, dtype={"mesh250": str, "시정촌코드": str})
    area_weighted(loc, Path(a.usgs_dir)).to_csv(a.out, index=False, compression="gzip", encoding="utf-8-sig")
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
