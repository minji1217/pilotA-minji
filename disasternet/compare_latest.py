"""
최신 실험 4개(DN · SOFT · DNU · SOFTU)를 USGS 기준선 · pilotA와 같은 행 · 같은 규칙으로 채점한다.
덱과 HANDOFF 문서의 AUC 숫자는 전부 이 스크립트로 낸 것이다.

채점 규칙
  - 행: pilotA 분리(holdout) 결과에도 있는 시정촌만 (공통 263곳)
  - 지진(이벤트)마다 따로 AUC. 정답이 한쪽(전부 1 또는 전부 0)뿐인 지진은 뺀다
  - DisasterNet은 seed 5개 각각 AUC → 평균
  - 전체 = Σ(이벤트 AUC × 시정촌 수) / Σ 시정촌 수   (시정촌 수 = 양성 + 음성)
  - ALL = 산사태 이벤트들 + 액상화 이벤트들을 같은 방식으로 한꺼번에 가중평균

pilotA 결과는 pilotA 저장소 followup4-lq-J-labels 브랜치를 지진마다 하나씩 빼고(--holdout-event) 돌린
outputs/holdout_detail_{lqlab|lqnolab}_holdout-*.csv (산사태), holdout_detail_lq_{...}_holdout-*.csv (액상화)를 읽는다.

    python -m disasternet.compare_latest --pilota-outputs <pilotA 저장소>/outputs
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .run_input_v5 import EVENT_TO_INDEX

RUNS = {
    "DN": "input_v5_lqmax",                     # ① 끄기 없음
    "SOFT": "input_v5_terrain_lqmax",           # ② 지형 비율 soft
    "DNU": "input_v5_muni_unc_lqmax",           # ③ USGS 불확실도
    "SOFTU": "input_v5_muni_unc_soft_lqmax",    # ④ soft + USGS 불확실도
}
EVS = ["2016 구마모토", "2018 오사카", "2018 훗카이도", "2021 후쿠시마", "2024 노토반도"]
# 도메인 → (정답 열, USGS 기준선 열, pilotA 사후 열, pilotA prior 열, pilotA 파일 접두)
DOMS = {"LS": ("LS_정답", "LS_prior_평균", "p_ls_alpha_fit", "z_cov_kappa", "holdout_detail_"),
        "LQ": ("LQ_정답", "LQ_prior_최대", "p_lq_alpha_fit", "z_kappa_lq", "holdout_detail_lq_")}


def pilota(outputs, prefix, tag):
    fs = glob.glob(f"{outputs}/{prefix}{tag}_holdout-*.csv")
    if not fs:
        raise FileNotFoundError(f"pilotA holdout 결과가 없습니다: {outputs}/{prefix}{tag}_holdout-*.csv")
    p = pd.concat([pd.read_csv(f, dtype={"muni_code": str}) for f in fs])
    p["muni_code"] = p.muni_code.str.zfill(5)
    return p.rename(columns={"event": "이벤트"})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="raw/DisasterNet_입력_v5.xlsx")
    ap.add_argument("--dn-dir", default="outputs/disasternet")
    ap.add_argument("--pilota-outputs", required=True)
    ap.add_argument("--out", default="outputs/disasternet/compare_latest.json")
    ap.add_argument("--runs", nargs="+", default=None, metavar="이름=폴더",
                    help="채점할 DisasterNet 결과 (기본: 최신 실험 4개). 예: DN=nobd_DN SOFT=nobd_SOFT")
    a = ap.parse_args(argv)
    runs = dict(r.split("=", 1) for r in a.runs) if a.runs else RUNS

    d = pd.read_excel(a.input, sheet_name="입력")
    d["muni_code"] = d["시정촌코드"].astype(int).astype(str).str.zfill(5)
    P = {k: [pd.read_csv(f, dtype={"muni_code": str}).assign(muni_code=lambda x: x.muni_code.str.zfill(5))
             for f in sorted(glob.glob(f"{a.dn_dir}/{v}/predictions_seed*.csv"))] for k, v in runs.items()}

    events = {}
    for dom, (ycol, usg, pa, pz, prefix) in DOMS.items():
        pc = "p_" + dom.lower()
        A1, A2 = pilota(a.pilota_outputs, prefix, "lqlab"), pilota(a.pilota_outputs, prefix, "lqnolab")
        for ev in EVS:
            g = d[(d.이벤트 == ev) & d[ycol].notna()]
            g = (g.merge(A1[["이벤트", "muni_code", pa, pz]].rename(columns={pa: "A1", pz: "PR"}), on=["이벤트", "muni_code"])
                  .merge(A2[["이벤트", "muni_code", pa]].rename(columns={pa: "A2"}), on=["이벤트", "muni_code"]))
            y = g[ycol].astype(int)
            if y.nunique() < 2:
                continue
            r = {"pos": int(y.sum()), "neg": int((y == 0).sum()), "w": len(g),
                 "USGS": roc_auc_score(y, g[usg]), "PR": roc_auc_score(y, g["PR"]),
                 "A1": roc_auc_score(y, g["A1"]), "A2": roc_auc_score(y, g["A2"])}
            for k, L in P.items():
                v = []
                for p in L:
                    m = g.merge(p[p.event_idx == EVENT_TO_INDEX[ev]][["muni_code", pc]], on="muni_code")
                    assert len(m) == len(g)
                    v.append(roc_auc_score(m[ycol].astype(int), m[pc]))
                r[k] = float(np.mean(v))
            events[f"{dom}|{ev}"] = r

    models = ["USGS", "PR", "A1", "A2"] + list(P)
    overall = {}
    for lab, doms in [("LS", ["LS"]), ("LQ", ["LQ"]), ("ALL", ["LS", "LQ"])]:
        E = [r for k, r in events.items() if k.split("|")[0] in doms]
        W = sum(r["w"] for r in E)
        overall[lab] = {m: sum(r[m] * r["w"] for r in E) / W for m in models}
        overall[lab]["W"] = W
    json.dump({"events": events, "overall": overall}, open(a.out, "w"), ensure_ascii=False, indent=1)

    names = {"USGS": "USGS 기준선", "PR": "pilotA prior", "A1": "pilotA A1", "A2": "pilotA A2",
             "DN": "① DN", "SOFT": "② SOFT", "DNU": "③ DNU", "SOFTU": "④ SOFTU"}
    names.update({k: k for k in runs if k not in names})
    print(f"{'':14s}  ALL    LS     LQ")
    for m in models:
        print(f"{names[m]:14s} {overall['ALL'][m]:.3f}  {overall['LS'][m]:.3f}  {overall['LQ'][m]:.3f}")
    print(f"시정촌 수: LS {overall['LS']['W']} · LQ {overall['LQ']['W']} · ALL {overall['ALL']['W']}")


if __name__ == "__main__":
    main()
