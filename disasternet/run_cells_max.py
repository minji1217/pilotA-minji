"""
칸(250 m) 단위 DisasterNet → 시정촌 사후확률 = 칸 사후확률의 최댓값 (교수님 지시, 2026-10).

  - 칸마다 자기 DPM(dpm_250m_v8의 DPM_배경차)과 자기 USGS 확률(us_ls, us_lq)로 학습한다
  - 산사태·액상화만 (BD 없음), 끄기 없음, soft 없음
  - 학습 설정은 시정촌 실험(run_input_v5)과 같다: 지진마다 따로, 정답 없이, 관측 + 2, 3000 epoch
  - 학습이 끝나면 시정촌마다 칸 사후확률의 최댓값(그리고 참고용 평균)을 낸다

칸 자료
  raw/local_250m_byevent_usgs.csv : 264개 시정촌의 250 m 칸 (칸 중심이 속한 시정촌), 경사·USGS 확률
  raw/dpm_250m_v8_slim.csv.gz     : 같은 칸 체계(1/4 메시)의 칸별 DPM
  둘을 (이벤트, mesh250)으로 붙이고, 물 칸(경우 0)은 뺀다. 물은 상관도가 원래 낮아 DPM이 커서
  최댓값을 오염시키기 때문이다. USGS 확률 999(계산 안 함)는 확률 0으로 둔다.

    python -m disasternet.run_cells_max --out-dir outputs/disasternet/cells_max --jobs 4
"""
import argparse
import time
from multiprocessing import get_context
from pathlib import Path

import pandas as pd
import torch

from schema import DTYPE

from .data import DPMRows
from .model import DisasterNet, prepare_inputs, triplet_penalty
from .run_input_v5 import EVENT_TO_INDEX

EVS = ["2016 구마모토", "2018 오사카", "2018 훗카이도", "2021 후쿠시마", "2024 노토반도"]


def load_cells(local_path, dpm_path, keep_water=False, off_rule="ours"):
    loc = pd.read_csv(local_path, dtype={"mesh250": str, "시정촌코드": str},
                      usecols=["이벤트", "시정촌코드", "mesh250", "slope", "경우", "us_ls", "us_lq", "sd_ls", "sd_lq"])
    dpm = pd.read_csv(dpm_path, dtype={"mesh250": str}, usecols=["이벤트", "mesh250", "화소수", "DPM_배경차", "DPM_최댓값"])
    c = loc.merge(dpm, on=["이벤트", "mesh250"], how="inner")
    if not keep_water:
        c = c[c["경우"] > 0]
    c["muni_code"] = c["시정촌코드"].str.zfill(5)
    # U 끄기 판정은 원래 값(0·999 포함)으로 먼저 해 둔다 (usgs_cells.usgs_off, 기본 규칙)
    from .usgs_cells import usgs_off
    for k in ("ls", "lq"):
        c[f"off_{k}"] = usgs_off(c[f"us_{k}"].fillna(999), c[f"sd_{k}"].fillna(999), off_rule)
    for k in ("us_ls", "us_lq"):
        c[k] = c[k].where(c[k] < 999, 0.0).fillna(0.0)
    c["event_idx"] = c["이벤트"].map(EVENT_TO_INDEX)
    return c.sort_values(["event_idx", "muni_code", "mesh250"]).reset_index(drop=True)


def to_rows(c, dpm_col, soft=False, unc=False, unc_mode="both", const_var=None):
    """soft: 시정촌 soft를 칸에 그대로 — 칸의 비탈 비율 = (경사 ≥ 10°), 평지 비율 = (경사 < 5°), 1% 미만은 1%로.
             칸은 경사가 하나라 비율이 0 아니면 1 → 비탈 칸은 액상화 로짓 −4.6, 평지 칸은 산사태 로짓 −4.6.
       unc : 원 논문처럼 칸마다 USGS 불확실도 — 0·999 칸은 끄고, 나머지는 r⁰ 분산 = 그 칸의 불확실도.
       unc_mode: both = 끄기 + 분산 / off = 끄기만 (분산 1) / var = 분산만 (끄지 않음).
       const_var: 주면 모든 칸의 r⁰ 분산을 이 값으로 (USGS 불확실도 정보 없이 폭만 작게)."""
    t = lambda col: torch.as_tensor(c[col].to_numpy(dtype="float64").copy(), dtype=DTYPE)
    n = len(c)
    log_frac = off = r0_var = None
    if soft:
        fr = {"LS": (c["slope"] >= 10.0).astype(float), "LQ": (c["slope"] < 5.0).astype(float)}
        log_frac = {k: torch.log(torch.as_tensor(v.clip(lower=0.01).to_numpy(), dtype=DTYPE)) for k, v in fr.items()}
        log_frac["BD"] = torch.zeros(n, dtype=DTYPE)
    if unc and unc_mode in ("both", "off"):
        off = {"LS": torch.as_tensor(c["off_ls"].to_numpy(dtype=bool)), "LQ": torch.as_tensor(c["off_lq"].to_numpy(dtype=bool)),
               "BD": torch.zeros(n, dtype=torch.bool)}
    if unc and unc_mode in ("both", "var"):
        # 분산만 쓸 때(끄지 않음) 0·999 칸은 불확실도 정보가 없으므로 기본값 1
        r0_var = {k: torch.as_tensor(c[f"sd_{k.lower()}"].where(~c[f"off_{k.lower()}"], 1.0).fillna(1.0).to_numpy(dtype="float64"), dtype=DTYPE)
                  for k in ("LS", "LQ")}
    if const_var is not None:
        r0_var = {k: torch.full((n,), float(const_var), dtype=DTYPE) for k in ("LS", "LQ")}
    return DPMRows(y=t(dpm_col), pi_ls=t("us_ls"), pi_lq=t("us_lq"), pgv=torch.ones(n, dtype=DTYPE),
                   event_idx=torch.as_tensor(c["event_idx"].to_numpy(), dtype=torch.long),
                   municipality_code=tuple(c["muni_code"]), model_row_idx=torch.arange(n),
                   log_frac=log_frac, off=off, r0_var=r0_var)


def fit_cells(rows, *, epochs, seed, lr=0.01, n_samples=8, num_flows=4, lam_triplet=1.0, margin=1.0,
              obs_offset=2.0, final_samples=256, chunk=50_000, log_every=500, tag="", ckpt=None, ckpt_secs=300):
    """model._fit_group과 같은 학습(ELBO + β warm-up + triplet), BD 없음. 마지막 추론만 칸이 많아 나눠서 한다.

    ckpt: 중간 저장 파일 (ckpt_secs초마다 저장). 있으면 그 epoch부터 이어서 학습한다 (작업 환경이 재시작돼도 처음부터 다시 하지 않게).
    """
    torch.manual_seed(seed)
    a, log_x = prepare_inputs(rows, obs_offset)
    model = DisasterNet(rows.n, num_flows=num_flows, r0_var=1.0, use_bd=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    start = 0
    if ckpt is not None and Path(ckpt).exists():
        st = torch.load(ckpt, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        torch.set_rng_state(st["rng"])
        start = st["epoch"] + 1
        print(f"  [{tag}] 중간 저장에서 이어서: epoch {start}", flush=True)
    t0 = last_save = time.time()
    for epoch in range(start, epochs):
        beta = min(0.01 + epoch / max(epochs, 1), 1.0)
        post = model.posteriors(a["LS"], a["LQ"], a["BD"], n_samples, off=rows.off, r0_var=rows.r0_var)
        q, keep = post["q"], post["keep"]
        rec = model.log_p_x(log_x, q)
        log_q = sum(post["log_q"][k].mean(1) * keep[k] for k in model.nodes)
        kl = log_q - model.log_p_z(q, a, rows.log_frac, keep if rows.off is not None else None)
        loss = (-rec + beta * kl).sum() + lam_triplet * triplet_penalty(q, a, margin)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and epoch % log_every == 0:
            print(f"  [{tag}] {epoch:5d}  -ELBO/칸 {float((-rec + kl).detach().mean()):.4f}  {time.time() - t0:.0f}s", flush=True)
        if ckpt is not None and time.time() - last_save > ckpt_secs and epoch + 1 < epochs:
            last_save = time.time()
            tmp = Path(str(ckpt) + ".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "rng": torch.get_rng_state(),
                        "epoch": epoch}, tmp)
            tmp.replace(ckpt)

    # 최종 사후확률: 샘플 final_samples개 평균. 칸이 많아 chunk씩 나눠 계산한다 (flow·μ는 행마다 독립).
    out = {k: torch.empty(rows.n, dtype=DTYPE) for k in model.nodes}
    with torch.no_grad():
        for s in range(0, rows.n, chunk):
            e = min(s + chunk, rows.n)
            sub = _slice_model_posteriors(model, a, s, e, final_samples, rows)
            for k in model.nodes:
                out[k][s:e] = sub[k]
    w = {k: round(float(v.detach()), 4) for k, v in model.w_x.items() if k in model.nodes}
    return out, w


def _slice_model_posteriors(model, a, s, e, n_samples, rows=None):
    """행 [s, e) 만의 사후확률. FlowNode의 행별 파라미터(mu_local, flow_local)를 잘라서 계산한다.
    rows의 r0_var(칸별 분산)와 off(끄기)도 같은 범위로 잘라 적용한다."""
    q = {}
    for k in model.nodes:
        node = model.flow[k]
        h = a[k][s:e].unsqueeze(1)
        full_mu, full_fl = node.mu_local, node.flow_local
        node.mu_local = torch.nn.Parameter(full_mu[s:e], requires_grad=False)
        node.flow_local = torch.nn.Parameter(full_fl[s:e], requires_grad=False)
        var = None if rows is None or rows.r0_var is None else rows.r0_var[k][s:e]
        try:
            r, _ = node(h, n_samples, var)
        finally:
            node.mu_local, node.flow_local = full_mu, full_fl
        q[k] = torch.sigmoid(r).mean(1)
        if rows is not None and rows.off is not None:
            q[k] = q[k] * (~rows.off[k][s:e]).to(q[k].dtype)
    return q


def _job(args):
    ev, seed, a = args
    torch.set_num_threads(a.threads)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = ev.replace(" ", "")
    if (out / f"muni_{tag}_seed{seed}.csv").exists():
        print(f"이미 있음 {ev} seed {seed}", flush=True)
        return
    c = load_cells(a.local, a.dpm, a.keep_water, a.off_rule)
    c = c[c["이벤트"] == ev].reset_index(drop=True)
    rows = to_rows(c, a.dpm_col, soft=a.soft, unc=a.unc, unc_mode=a.unc_mode, const_var=a.const_var)
    ckpt = out / f"ckpt_{tag}_seed{seed}.pt"
    q, w = fit_cells(rows, epochs=a.epochs, seed=seed, tag=f"{ev} s{seed} {rows.n}칸", ckpt=ckpt)
    c = c.assign(p_ls=q["LS"].numpy(), p_lq=q["LQ"].numpy())
    g = c.groupby(["event_idx", "muni_code"])
    m = g.agg(n_cells=("mesh250", "size"), p_ls=("p_ls", "max"), p_lq=("p_lq", "max"),
              p_ls_mean=("p_ls", "mean"), p_lq_mean=("p_lq", "mean")).reset_index()
    m.insert(0, "seed", seed)
    m.to_csv(out / f"muni_{tag}_seed{seed}.csv", index=False, encoding="utf-8-sig")
    c[["event_idx", "muni_code", "mesh250", "us_ls", "us_lq", "DPM_배경차", "p_ls", "p_lq"]].to_csv(
        out / f"cells_{tag}_seed{seed}.csv.gz", index=False, compression="gzip")
    ckpt.unlink(missing_ok=True)
    print(f"완료 {ev} seed {seed}  w_x {w}", flush=True)


def merge_outputs(out_dir, seeds):
    out = Path(out_dir)
    for s in seeds:
        fs = sorted(out.glob(f"muni_*_seed{s}.csv"))
        if len(fs) == len(EVS):
            pd.concat([pd.read_csv(f, dtype={"muni_code": str}) for f in fs]).to_csv(
                out / f"predictions_seed{s}.csv", index=False, encoding="utf-8-sig")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default="raw/local_250m_byevent_usgs.csv")
    ap.add_argument("--dpm", default="raw/dpm_250m_v8_slim.csv.gz")
    ap.add_argument("--dpm-col", default="DPM_배경차")
    ap.add_argument("--keep-water", action="store_true", help="물 칸(경우 0)도 쓴다")
    ap.add_argument("--soft", action="store_true", help="칸 경사로 soft (비탈 칸 액상화·평지 칸 산사태 로짓 −4.6)")
    ap.add_argument("--off-rule", choices=["ours", "paper"], default="ours",
                    help="--unc 끄기 판정: ours = 불확실도 0·999 또는 확률 0·999 / paper = 공식 코드처럼 불확실도 0·999만")
    ap.add_argument("--unc-mode", choices=["both", "off", "var"], default="both",
                    help="--unc에서 쓸 것: both = 끄기 + 분산 / off = 끄기만 / var = 분산만")
    ap.add_argument("--const-var", type=float, default=None, help="모든 칸의 r⁰ 분산을 같은 값으로 (비교용)")
    ap.add_argument("--unc", action="store_true", help="칸마다 USGS 불확실도: 0·999 칸 끄기 + r⁰ 분산 = 그 칸 불확실도")
    ap.add_argument("--events", nargs="+", default=EVS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out-dir", default="outputs/disasternet/cells_max")
    a = ap.parse_args(argv)
    jobs = [(ev, s, a) for s in a.seeds for ev in a.events]
    if a.jobs > 1:
        with get_context("spawn").Pool(a.jobs) as p:
            p.map(_job, jobs, chunksize=1)
    else:
        for j in jobs:
            _job(j)
    merge_outputs(a.out_dir, a.seeds)


if __name__ == "__main__":
    main()
