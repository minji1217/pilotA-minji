"""
칸 결과(cells_*_seed*.csv.gz, 수십 MB)를 작게 줄여 GitHub로 보낼 수 있게 한다.

  1) 시정촌별 요약: 칸 수, 칸 최댓값, 칸 평균, 상위 1 · 5 · 10 % 칸 평균 (산사태 · 액상화)
  2) 몇몇 시정촌의 칸 전체 (산사태 피해 있음인데 비탈이 적은 곳과 비교 대상)

실험 폴더에서 실행 (결과를 git 폴더의 pc_results로 쓰면 push_results.bat이 올린다):
    py -3.10 -m disasternet.cell_extract --out C:\\work\\pilotA-minji-git\\pc_results\\cell_extract
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

EXPS = ["cells_max_local_unc_paper_aw", "cells_max_softunc_paper_aw_mid", "cells_max_softunc_off_paper_aw_mid",
        "cells_max_softunc_var_paper_aw_mid", "cells_max_aw"]
# 칸 전체를 보낼 시정촌 (지진 번호, 시정촌 코드): 구마모토 · 오사카 · 훗카이도 산사태 비교용
DETAIL = {(3, "43204"), (3, "43216"), (3, "43364"), (3, "43404"), (3, "43468"),
          (4, "27203"), (4, "27205"), (4, "27210"),
          (5, "01209"), (5, "01213"), (5, "01234"), (5, "01428"), (5, "01601")}
DETAIL_EXPS = EXPS[:2]
TOPS = (0.01, 0.05, 0.10)


def top_mean(v, q):
    v = np.sort(v)[::-1]
    return float(v[:max(1, int(np.ceil(len(v) * q)))].mean())


def summarize(c):
    rows = []
    for (e, m), g in c.groupby(["event_idx", "muni_code"]):
        r = {"event_idx": e, "muni_code": m, "n_cells": len(g)}
        for k in ("p_ls", "p_lq"):
            v = g[k].to_numpy()
            r[f"{k}_max"] = float(v.max())
            r[f"{k}_mean"] = float(v.mean())
            for q in TOPS:
                r[f"{k}_top{int(q * 100)}"] = top_mean(v, q)
        rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dn-dir", default="outputs/disasternet")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for exp in EXPS:
        fs = sorted(Path(a.dn_dir, exp).glob("cells_*_seed*.csv.gz"))
        if not fs:
            print(f"없음: {exp}")
            continue
        summ, det = [], []
        for f in fs:
            seed = int(re.search(r"_seed(\d+)\.csv\.gz$", f.name).group(1))
            c = pd.read_csv(f, dtype={"muni_code": str, "mesh250": str})
            c["muni_code"] = c["muni_code"].str.zfill(5)
            summ.append(summarize(c).assign(seed=seed))
            if exp in DETAIL_EXPS:
                keep = [(e, m) in DETAIL for e, m in zip(c["event_idx"], c["muni_code"])]
                det.append(c[keep].assign(seed=seed))
            print(f"{exp}  {f.name}  {len(c)}칸", flush=True)
        pd.concat(summ).to_csv(out / f"{exp}_muni_summary.csv", index=False, encoding="utf-8-sig")
        if det:
            pd.concat(det).to_csv(out / f"{exp}_detail.csv.gz", index=False, compression="gzip")
    print(f"끝: {out}")


if __name__ == "__main__":
    main()
