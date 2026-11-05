"""다시 돌린 실험이 예전 결과와 똑같은지 실험 · seed · 지진마다 비교한다.

시정촌 단위: predictions_seed{s}.csv 끼리 (event_idx, muni_code)로 맞춰 p_ls, p_lq 비교
칸 단위    : 새 폴더의 cells_{지진}_seed{s}.csv.gz 를 예전 cells_ 파일과 비교 (없으면 muni_ 파일, 그것도 없으면 predictions_seed{s}.csv의 그 지진 줄)

    python -m disasternet.rerun_compare --level muni  --pair DN outputs/disasternet/nobd_DN outputs/disasternet/rerun_muni/DN ...
    python -m disasternet.rerun_compare --level cells --pair LOCALU outputs/disasternet/cells_max_local_unc_paper_aw outputs/disasternet/rerun_cells/LOCALU ...
결과 표는 --out (기본: 새 폴더들의 상위 폴더/rerun_compare.csv)에 저장한다.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EVS = ["2016 구마모토", "2018 오사카", "2018 훗카이도", "2021 후쿠시마", "2024 노토반도"]
EV_IDX = {"2016 구마모토": 3, "2018 오사카": 4, "2018 훗카이도": 5, "2021 후쿠시마": 6, "2024 노토반도": 7}


def _cmp(o, n, key):
    m = o.merge(n, on=key, suffixes=("_old", "_new"))
    row = {"n_old": len(o), "n_new": len(n), "n_matched": len(m)}
    same = len(m) == len(o) == len(n)
    for col in ("p_ls", "p_lq"):
        x, y = m[f"{col}_old"].to_numpy(float), m[f"{col}_new"].to_numpy(float)
        eq = int((x == y).sum())
        row[f"{col}_same"] = eq
        row[f"{col}_max_diff"] = float(np.abs(x - y).max()) if len(m) else np.nan
        same &= eq == len(m)
    row["result"] = "같음" if same else "다름"
    return row


def _muni(name, old, new, seed):
    fo, fn = old / f"predictions_seed{seed}.csv", new / f"predictions_seed{seed}.csv"
    if not fn.exists():
        return [{"experiment": name, "seed": seed, "event": "전체", "level": "시정촌", "result": "아직 없음"}]
    if not fo.exists():
        return [{"experiment": name, "seed": seed, "event": "전체", "level": "시정촌", "result": "예전 파일 없음"}]
    o = pd.read_csv(fo, dtype={"muni_code": str})
    n = pd.read_csv(fn, dtype={"muni_code": str})
    out = []
    for ev, i in EV_IDX.items():
        r = _cmp(o[o.event_idx == i], n[n.event_idx == i], ["muni_code"])
        out.append({"experiment": name, "seed": seed, "event": ev, "level": "시정촌", **r})
    return out


def _cells(name, old, new, seed):
    out = []
    for ev in EVS:
        tag = ev.replace(" ", "")
        base = {"experiment": name, "seed": seed, "event": ev}
        fn = new / f"cells_{tag}_seed{seed}.csv.gz"
        if not fn.exists():
            out.append({**base, "level": "칸", "result": "아직 없음"})
            continue
        fo = old / f"cells_{tag}_seed{seed}.csv.gz"
        if fo.exists():
            o = pd.read_csv(fo, dtype={"muni_code": str, "mesh250": str})
            n = pd.read_csv(fn, dtype={"muni_code": str, "mesh250": str})
            out.append({**base, "level": "칸", **_cmp(o, n, ["muni_code", "mesh250"])})
            continue
        n = pd.read_csv(new / f"muni_{tag}_seed{seed}.csv", dtype={"muni_code": str})
        fo = old / f"muni_{tag}_seed{seed}.csv"
        if fo.exists():
            o = pd.read_csv(fo, dtype={"muni_code": str})
        elif (old / f"predictions_seed{seed}.csv").exists():
            o = pd.read_csv(old / f"predictions_seed{seed}.csv", dtype={"muni_code": str})
            o = o[o.event_idx == EV_IDX[ev]]
        else:
            out.append({**base, "level": "칸", "result": "예전 파일 없음"})
            continue
        out.append({**base, "level": "시정촌(칸 최댓값)", **_cmp(o, n, ["muni_code"])})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["muni", "cells"], required=True)
    ap.add_argument("--pair", nargs=3, action="append", required=True, metavar=("이름", "예전폴더", "새폴더"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows = []
    for name, old, new in a.pair:
        for s in a.seeds:
            rows += (_muni if a.level == "muni" else _cells)(name, Path(old), Path(new), s)
    df = pd.DataFrame(rows)
    out = Path(a.out) if a.out else Path(a.pair[0][2]).parent / "rerun_compare.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    done = df[df.result.isin(["같음", "다름"])]
    print(f"비교 {len(done)}건: 같음 {int((done.result == '같음').sum())} / 다름 {int((done.result == '다름').sum())}"
          f" / 아직 없음 {int((df.result == '아직 없음').sum())} / 예전 파일 없음 {int((df.result == '예전 파일 없음').sum())}")
    if (done.result == "다름").any():
        print(done[done.result == "다름"][["experiment", "seed", "event", "p_ls_max_diff", "p_lq_max_diff"]].to_string(index=False))
    print(f"저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
