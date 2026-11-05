"""칸 단위 DisasterNet을 같은 PC에서 다시 돌렸을 때 예전 결과와 똑같이 나오는지 확인한다.

예전 실행은 학습이 끝난 파라미터를 남기지 않았다. 같은 PC · 같은 seed · 같은 스레드 수로 다시 돌려서
예측값이 예전 파일과 완전히 같으면 파라미터도 같다고 볼 수 있다 (새 실행은 model_*.pt를 남긴다).

비교 순서: 칸별 예측(cells_*.csv.gz) → 시정촌 예측(muni_*.csv) → predictions_seed*.csv 의 그 지진 줄.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EV_IDX = {"2016 구마모토": 3, "2018 오사카": 4, "2018 훗카이도": 5, "2021 후쿠시마": 6, "2024 노토반도": 7}


def _old_frame(old, tag, seed, ev):
    f = old / f"cells_{tag}_seed{seed}.csv.gz"
    if f.exists():
        return "칸", pd.read_csv(f, dtype={"muni_code": str, "mesh250": str})
    f = old / f"muni_{tag}_seed{seed}.csv"
    if f.exists():
        return "시정촌", pd.read_csv(f, dtype={"muni_code": str})
    f = old / f"predictions_seed{seed}.csv"
    if f.exists():
        d = pd.read_csv(f, dtype={"muni_code": str})
        return "시정촌", d[d["event_idx"] == EV_IDX[ev]].reset_index(drop=True)
    return None, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-dir", required=True, help="예전 결과 폴더")
    ap.add_argument("--new-dir", required=True, help="다시 돌린 결과 폴더")
    ap.add_argument("--event", default="2018 오사카")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import torch
    print(f"버전: python {sys.version.split()[0]} / torch {torch.__version__} / numpy {np.__version__} / pandas {pd.__version__}")

    tag = a.event.replace(" ", "")
    old, new = Path(a.old_dir), Path(a.new_dir)
    level, o = _old_frame(old, tag, a.seed, a.event)
    if o is None:
        print(f"예전 결과 파일을 찾지 못함: {old}  (cells_/muni_{tag}_seed{a.seed}, predictions_seed{a.seed}.csv)")
        return 2
    if level == "칸":
        n = pd.read_csv(new / f"cells_{tag}_seed{a.seed}.csv.gz", dtype={"muni_code": str, "mesh250": str})
        key = ["muni_code", "mesh250"]
    else:
        n = pd.read_csv(new / f"muni_{tag}_seed{a.seed}.csv", dtype={"muni_code": str})
        key = ["muni_code"]
    m = o.merge(n, on=key, suffixes=("_old", "_new"))
    if len(m) != len(o) or len(m) != len(n):
        print(f"줄 수가 다름: 예전 {len(o)} / 새 {len(n)} / 맞춘 {len(m)}  → 다름")
        return 1

    print(f"{a.event} seed {a.seed}, {level} {len(m)}개 비교")
    same = True
    for col in ["p_ls", "p_lq"]:
        x, y = m[f"{col}_old"].to_numpy(float), m[f"{col}_new"].to_numpy(float)
        d = np.abs(x - y)
        n_eq = int((x == y).sum())
        print(f"  {col}: 완전히 같은 {level} {n_eq}/{len(m)}  최대 차이 {d.max():.3e}  평균 차이 {d.mean():.3e}")
        same &= n_eq == len(m)

    model_file = new / f"model_{tag}_seed{a.seed}.pt"
    if same:
        print("결과: 같음  → 이 PC에서 같은 설정으로 다시 돌리면 예전 파라미터를 되살릴 수 있음")
        print(f"  되살린 파라미터: {model_file}")
        return 0
    print("결과: 다름  → 다시 돌려도 예전 파라미터와 같아지지 않음 (차이가 아주 작으면 계산 순서 차이)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
