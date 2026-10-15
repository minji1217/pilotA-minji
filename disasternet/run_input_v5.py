"""
DisasterNet 입력 통합표(DisasterNet_입력_v5.xlsx, 시트 "입력" 264행)로 돌리는 실험.

    python -m disasternet.run_input_v5 --input raw/DisasterNet_입력_v5.xlsx --jobs 4

무엇을 하나
  기준선  USGS prior 단독(LS_prior_평균, LQ_prior_평균)의 AUC
  제안    prior + DPM 관측을 DisasterNet(disasternet/model.py)으로 결합한 사후확률의 AUC
  참고    DPM 값 자체를 점수로 쓴 AUC (모델이 DPM 위에 무엇을 더하는지 보려고)

입력 처리
  - prior·PGV·정답은 모두 통합표의 값을 쓴다 (pilotA loader를 거치지 않는다).
  - 관측은 DPM_배경차(주 지표). DisasterNet은 DPM을 로그정규로 두어 양수가 필요하므로
    모든 시정촌에 같은 상수 2를 더한다(--obs-offset). DPM은 상관도의 차이라 [-1, 1] 안이고,
    배경차(시정촌 평균 − 프레임 중앙값)는 이론상 [-2, 2] 안이다.
  - 학습은 이벤트마다 따로, 정답 없이. seed 여러 개로 반복한다.

평가 규칙 (요청 사항 그대로)
  - AUC는 이벤트별로 따로 계산한다. 이벤트를 섞지 않는다.
  - 전체 수치는 비교쌍 가중평균: Σ AUC_e·(양성_e×음성_e) / Σ(양성_e×음성_e).
  - 양성만 또는 음성만 있는 이벤트·도메인 조합은 AUC를 내지 않고 표에 "계산 불가"로 표시한다.
  - base상관도 < 0.30 행은 빼지 않는다. 민감도 분석으로만, 그 행을 채점에서 뺀 표를 따로 낸다
    (학습은 같은 모델 그대로).

산출물 (기본 outputs/disasternet/input_v5/, --out-dir로 변경)
  predictions_seed{N}.csv   시정촌별 사후확률
  auc_by_event.csv          이벤트 × 도메인 표 (양성·음성·쌍 수, prior / DPM / 사후 AUC)
  auc_overall.csv           비교쌍 가중 전체 AUC (LS · LQ · 합계)
  auc_by_event_coh30.csv    민감도: base상관도 ≥ 0.30 행만 채점
  auc_overall_coh30.csv
  movers.csv                prior 대비 사후 순위가 크게 오르거나 내린 정답 있는 시정촌
  summary.md                위 표들을 한 파일로
"""
import argparse
import os
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from loader import _normalize_municipality_code
from schema import DTYPE, EVENT_TO_INDEX

from .data import DPMRows
from .model import fit_disasternet

SHEET = "입력"
DOMAINS = {  # 도메인: (정답 컬럼, prior 컬럼)
    "LS": ("LS_정답", "LS_prior_평균"),
    "LQ": ("LQ_정답", "LQ_prior_평균"),
}
LOW_COH = 0.30
OUT_DIR = Path("outputs/disasternet/input_v5")


# ---------------------------------------------------------------- 입력
def load_table(path):
    df = pd.read_excel(path, sheet_name=SHEET)
    df["muni_code"] = df["시정촌코드"].map(_normalize_municipality_code)
    unknown = sorted(set(df["이벤트"]) - set(EVENT_TO_INDEX))
    if unknown:
        raise ValueError(f"schema.EVENTS에 없는 이벤트: {unknown}")
    df["event_idx"] = df["이벤트"].map(EVENT_TO_INDEX).astype(int)
    if df.duplicated(["event_idx", "muni_code"]).any():
        raise ValueError("(이벤트, 시정촌코드) 중복이 있습니다.")
    return df.sort_values(["event_idx", "muni_code"]).reset_index(drop=True)


FRAC_FLOOR = 0.01  # 비율 0인 시정촌의 log가 -inf가 되지 않게 (1% 미만은 1%로)


def attach_terrain(df, path):
    """terrain_fractions.csv(시정촌별 비탈·평지·건물용지 비율)를 붙인다."""
    t = pd.read_csv(path, dtype={"시정촌코드": str})
    t = t.rename(columns={"시정촌코드": "muni_code"})[["이벤트", "muni_code", "frac_LS", "frac_LQ", "frac_BD"]]
    out = df.merge(t, on=["이벤트", "muni_code"], how="left", validate="one_to_one")
    if out[["frac_LS", "frac_LQ", "frac_BD"]].isna().any().any():
        raise ValueError("지형 비율이 없는 시정촌이 있습니다.")
    return out


def to_rows(df, dpm_col):
    t = lambda c: torch.as_tensor(df[c].to_numpy(dtype="float64"), dtype=DTYPE)
    log_frac = None
    if "frac_LS" in df.columns:
        log_frac = {k: torch.log(torch.clamp(t(f"frac_{k}"), min=FRAC_FLOOR)) for k in ("LS", "LQ", "BD")}
    return DPMRows(
        y=t(dpm_col), pi_ls=t("LS_prior_평균"), pi_lq=t("LQ_prior_평균"), pgv=t("PGV"),
        event_idx=torch.as_tensor(df["event_idx"].to_numpy(), dtype=torch.long),
        municipality_code=tuple(df["muni_code"]),
        model_row_idx=torch.arange(len(df)),
        log_frac=log_frac,
    )


# ---------------------------------------------------------------- 학습
def _fit_one(job):
    path, dpm_col, offset, seed, epochs, use_bd, out_dir, terrain = job
    torch.set_num_threads(1)
    df = load_table(path)
    if terrain:
        df = attach_terrain(df, terrain)
    fit = fit_disasternet(to_rows(df, dpm_col), seed=seed, epochs=epochs, use_bd=use_bd,
                          obs_offset=offset, verbose=False)
    p = fit.predictions[["event_idx", "muni_code", "p_ls", "p_lq", "p_bd"]].copy()
    p.insert(0, "seed", seed)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p.to_csv(Path(out_dir) / f"predictions_seed{seed}.csv", index=False, encoding="utf-8-sig")
    print(f"seed {seed} 완료", flush=True)
    return p


# ---------------------------------------------------------------- 평가
def _auc(y, s):
    return float(roc_auc_score(y, s))


def by_event_table(df, post, dpm_col, keep=None):
    """이벤트 × 도메인 AUC 표. post: seed별 사후 [{seed: DataFrame(event_idx, muni_code, p_ls, p_lq)}]."""
    rows = []
    for dom, (ycol, pcol) in DOMAINS.items():
        pk = "p_" + dom.lower()
        for (eidx, ev), g in df.groupby(["event_idx", "이벤트"]):
            g = g[g[ycol].notna()]
            if keep is not None:
                g = g[keep.loc[g.index]]
            y = g[ycol].astype(int)
            n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
            r = {"도메인": dom, "이벤트": ev, "양성": n_pos, "음성": n_neg, "비교쌍": n_pos * n_neg}
            if n_pos == 0 or n_neg == 0:
                r["상태"] = "계산 불가 (" + ("양성만" if n_neg == 0 and n_pos else
                                          "음성만" if n_pos == 0 and n_neg else "정답 없음") + ")"
                rows.append(r)
                continue
            r["상태"] = "계산"
            r["AUC_prior"] = _auc(y, g[pcol])
            r["AUC_DPM"] = _auc(y, g[dpm_col])
            per_seed = []
            for s, p in post.items():
                m = g[["muni_code"]].merge(p[p.event_idx == eidx][["muni_code", pk]], on="muni_code")
                per_seed.append(_auc(y.to_numpy(), m[pk].to_numpy()))
            r["AUC_사후_평균"] = float(np.mean(per_seed))
            r["AUC_사후_표준편차"] = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0
            r["AUC_사후_seed별"] = " / ".join(f"{v:.3f}" for v in per_seed)
            rows.append(r)
    return pd.DataFrame(rows)


def overall_table(ev_tab, post_seeds):
    """비교쌍 가중 전체 AUC. 사후는 seed마다 가중평균을 낸 뒤 seed 평균 ± 표준편차."""
    out = []
    ok = ev_tab[ev_tab["상태"] == "계산"].copy()
    seeds_auc = ok["AUC_사후_seed별"].str.split(" / ").apply(lambda v: [float(x) for x in v])
    for label, mask in (("LS", ok["도메인"] == "LS"), ("LQ", ok["도메인"] == "LQ"), ("LS+LQ", ok["도메인"].notna())):
        g, w = ok[mask], ok.loc[mask, "비교쌍"].to_numpy()
        if w.sum() == 0:
            continue
        sa = np.array(seeds_auc[mask].tolist())             # [조합, seed]
        per_seed = (sa * w[:, None]).sum(0) / w.sum()
        out.append({
            "범위": label, "이벤트·도메인 조합": int(mask.sum()), "비교쌍": int(w.sum()),
            "AUC_prior": float(np.average(g["AUC_prior"], weights=w)),
            "AUC_DPM": float(np.average(g["AUC_DPM"], weights=w)),
            "AUC_사후_평균": float(per_seed.mean()),
            "AUC_사후_표준편차": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0,
        })
    return pd.DataFrame(out)


def movers(df, post, dpm_col, top=5):
    """정답 있는 시정촌 중 prior 대비 사후의 '이벤트 안 순위'가 가장 크게 바뀐 곳.

    AUC는 이벤트 안의 순위만 보므로 확률 차이보다 순위 백분위 변화로 고른다.
    사후는 seed 평균 확률을 쓴다.
    """
    mean_post = pd.concat(post.values()).groupby(["event_idx", "muni_code"])[["p_ls", "p_lq"]].mean().reset_index()
    d = df.merge(mean_post, on=["event_idx", "muni_code"])
    rows = []
    for dom, (ycol, pcol) in DOMAINS.items():
        pk = "p_" + dom.lower()
        g = d.copy()
        grp = g.groupby("event_idx")
        g["prior순위"] = grp[pcol].rank(pct=True)
        g["사후순위"] = grp[pk].rank(pct=True)
        g["DPM순위"] = grp[dpm_col].rank(pct=True)
        g["순위변화"] = g["사후순위"] - g["prior순위"]
        g = g[g[ycol].notna()]
        for direction, sub in (("상승", g.nlargest(top, "순위변화")), ("하락", g.nsmallest(top, "순위변화"))):
            for _, r in sub.iterrows():
                y = int(r[ycol])
                good = (y == 1) == (direction == "상승")
                rows.append({
                    "도메인": dom, "방향": direction, "이벤트": r["이벤트"], "시정촌명": r["시정촌명"],
                    "시정촌코드": r["muni_code"], "정답": y, "AUC에": "도움" if good else "해침",
                    "prior": r[pcol], "사후": r[pk], "prior순위%": 100 * r["prior순위"],
                    "사후순위%": 100 * r["사후순위"], "순위변화%p": 100 * r["순위변화"],
                    dpm_col: r[dpm_col], "DPM순위%": 100 * r["DPM순위"], "base상관도": r["base상관도"],
                    "주의": r["주의"] if isinstance(r["주의"], str) else "",
                })
    return pd.DataFrame(rows)


def _md(df, floatfmt=3):
    def f(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.{floatfmt}f}"
        return str(v)
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False))


# ---------------------------------------------------------------- 실행
def main(argv=None):
    ap = argparse.ArgumentParser(description="DisasterNet 입력 통합표 v5 실험")
    ap.add_argument("--input", default="raw/DisasterNet_입력_v5.xlsx")
    ap.add_argument("--dpm-col", default="DPM_배경차", help="관측 컬럼 (기본: 주 지표 DPM_배경차)")
    ap.add_argument("--obs-offset", type=float, default=2.0,
                    help="관측에 더할 고정 상수. 로그정규 관측이라 양수가 필요하다. "
                         "배경차는 이론상 -2까지 갈 수 있어 기본 2 (1로 돌려도 전체 AUC 차이 0.002)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--no-bd", action="store_true", help="건물피해 노드를 뺀 ablation")
    ap.add_argument("--jobs", type=int, default=1, help="seed를 동시에 돌릴 프로세스 수")
    ap.add_argument("--skip-fit", action="store_true", help="이미 저장된 predictions_seed*.csv로 채점만")
    ap.add_argument("--top", type=int, default=5, help="순위 변화 상위 몇 곳")
    ap.add_argument("--out-dir", default=str(OUT_DIR), help="결과 폴더")
    ap.add_argument("--terrain", default=None,
                    help="terrain_fractions.csv. 주면 원본 가지치기의 대용으로 prior 로짓에 log(가능 면적 비율)을 더한다")
    a = ap.parse_args(argv)
    out_dir = Path(a.out_dir)

    df = load_table(a.input)
    print(f"입력 {len(df)}행 / 이벤트별 {df.groupby('이벤트').size().to_dict()}")

    if a.skip_fit:
        post = {s: pd.read_csv(out_dir / f"predictions_seed{s}.csv", dtype={"muni_code": str}) for s in a.seeds}
    else:
        jobs = [(a.input, a.dpm_col, a.obs_offset, s, a.epochs, not a.no_bd, out_dir, a.terrain) for s in a.seeds]
        if a.jobs > 1:
            with get_context("fork").Pool(a.jobs) as pool:
                res = pool.map(_fit_one, jobs)
        else:
            res = [_fit_one(j) for j in jobs]
        post = {s: p for s, p in zip(a.seeds, res)}

    keep = df["base상관도"] >= LOW_COH
    ev = by_event_table(df, post, a.dpm_col)
    ov = overall_table(ev, post)
    ev_c = by_event_table(df, post, a.dpm_col, keep=keep)
    ov_c = overall_table(ev_c, post)
    mv = movers(df, post, a.dpm_col, top=a.top)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in (("auc_by_event", ev), ("auc_overall", ov), ("auc_by_event_coh30", ev_c),
                    ("auc_overall_coh30", ov_c), ("movers", mv)):
        t.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    show_ev = ["도메인", "이벤트", "양성", "음성", "비교쌍", "상태", "AUC_prior", "AUC_DPM", "AUC_사후_평균", "AUC_사후_표준편차"]
    md = [f"# DisasterNet · 입력 통합표 v5\n",
          f"- 관측: {a.dpm_col} + {a.obs_offset} (로그정규 관측) · seed {a.seeds} · BD 노드 {'제외' if a.no_bd else '포함'}"
          f" · 가지치기 대용 {'있음 (' + a.terrain + ')' if a.terrain else '없음'}",
          f"- 전체 AUC = 비교쌍(양성×음성) 가중평균. 계산 불가 조합은 제외.\n",
          "## 이벤트별 · 도메인별\n", _md(ev[[c for c in show_ev if c in ev.columns]]),
          "\n## 비교쌍 가중 전체\n", _md(ov),
          f"\n## 민감도: base상관도 ≥ {LOW_COH} 행만 채점 (제외 {int((~keep).sum())}행)\n",
          _md(ev_c[[c for c in show_ev if c in ev_c.columns]]), "\n", _md(ov_c),
          "\n## prior 대비 사후 순위가 크게 바뀐 시정촌 (정답 있는 곳)\n", _md(mv, 3)]
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")

    pd.set_option("display.width", 250)
    print(ev[[c for c in show_ev if c in ev.columns]].round(3).to_string(index=False))
    print(ov.round(4).to_string(index=False))
    print(f"\n[민감도 base상관도 ≥ {LOW_COH}]")
    print(ov_c.round(4).to_string(index=False))
    print(f"\n저장: {out_dir}/")


if __name__ == "__main__":
    main()
