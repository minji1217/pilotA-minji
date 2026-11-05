"""
DisasterNet(칸 단위 LOCALU)과 pilotA(A2)의 학습 · 추론 시간을 같은 컴퓨터에서 잰다.

  DisasterNet: 지진마다 그 지진의 칸으로 학습 (실제 실험과 같은 설정: LOCALU, 논문 기준 U 끄기, 스레드 2).
               3000 epoch을 다 돌리면 오래 걸리므로 --dn-epochs번만 돌려 epoch당 시간을 재고 3000배로 환산한다.
               추론(칸마다 샘플 256개)은 끝까지 잰다.
  pilotA     : 예측 지진을 뺀 8개 지진으로 3000 epoch 학습 (끝까지 실측), 추론은 사후확률 + α_e 맞춤.

실험 폴더(raw/가 있는 곳)에서 실행:
    py -3.10 -m disasternet.time_compare --pilota-dir ..\\pilotA-time
결과는 화면과 outputs/time_compare.txt에 나온다.
"""
import argparse, contextlib, io, os, sys, time
from pathlib import Path

import pandas as pd
import torch

EVS = ["2016 구마모토", "2018 오사카", "2018 훗카이도", "2021 후쿠시마", "2024 노토반도"]


def time_dn(epochs, threads, log):
    from disasternet.run_cells_max import load_cells, to_rows, _slice_model_posteriors
    from disasternet.model import DisasterNet, prepare_inputs, triplet_penalty
    torch.set_num_threads(threads)
    C = load_cells("raw/local_250m_byevent_usgs_aw.csv.gz", "raw/dpm_250m_v8_slim.csv.gz", False, "paper")
    res = []
    for ev in EVS:
        c = C[C["이벤트"] == ev].reset_index(drop=True)
        rows = to_rows(c, "DPM_배경차", unc=True, unc_mode="both", local_off=True)
        torch.manual_seed(0)
        a, log_x = prepare_inputs(rows, 2.0)
        model = DisasterNet(rows.n, num_flows=4, r0_var=1.0, use_bd=False)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        t = time.time()
        for epoch in range(epochs):
            beta = min(0.01 + epoch / 3000, 1.0)
            post = model.posteriors(a["LS"], a["LQ"], a["BD"], 8, off=rows.off, r0_var=rows.r0_var)
            q, keep = post["q"], post["keep"]
            rec = model.log_p_x(log_x, q)
            log_q = sum(post["log_q"][k].mean(1) * keep[k] for k in model.nodes)
            kl = log_q - model.log_p_z(q, a, rows.log_frac, keep)
            loss = (-rec + beta * kl).sum() + triplet_penalty(q, a, 1.0)
            opt.zero_grad(); loss.backward(); opt.step()
        train = (time.time() - t) / epochs * 3000
        t = time.time()
        with torch.no_grad():
            for s in range(0, rows.n, 50000):
                _slice_model_posteriors(model, a, s, min(s + 50000, rows.n), 256, rows)
        infer = time.time() - t
        res.append((ev, rows.n, train, infer))
        log(f"  DisasterNet {ev}: 칸 {rows.n:,}  학습 {train/60:.1f}분 (3000 epoch 환산)  추론 {infer:.1f}초")
    return res


def make_area(pdir):
    """pilotA가 읽는 raw/시정촌_면적.csv를 USGS 표의 area_km2로 만든다 (없을 때만)."""
    out = Path(pdir, "raw", "시정촌_면적.csv")
    if out.exists():
        return
    sys.path.insert(0, str(pdir))
    from schema import MUNICIPALITY_CODE_COLUMN as M, EVENTS
    rows = []
    for ev in EVENTS:
        x = pd.read_excel(Path(pdir, "raw", "재난프로젝트_시정촌별_USGS.xlsx"), sheet_name=ev, header=3)
        x = x[pd.to_numeric(x["시정촌코드"], errors="coerce").notna()]
        rows += [{"event": ev, M: str(int(c)).zfill(5), "area_km2": a} for c, a in zip(x["시정촌코드"], x["area_km2"])]
    pd.DataFrame(rows).drop_duplicates(["event", M]).to_csv(out, index=False)


def time_pilota(pdir, log):
    pdir = Path(pdir).resolve()
    make_area(pdir)
    here = os.getcwd(); os.chdir(pdir); sys.path.insert(0, str(pdir))
    try:
        import train as T
        from loader import load_pilot_a_batch, load_eval_ground_truth, attach_ls_labels, subset_batch
        from schema import EVENT_TO_INDEX
        batch = load_pilot_a_batch(T.STATS_PATH, T.USGS_PATH)
        batch = attach_ls_labels(T.GT_PATH, batch, load_eval_ground_truth(T.GT_PATH, batch))
        res = []
        for ev in EVS:
            tb = subset_batch(batch, batch.event_idx != EVENT_TO_INDEX[ev])
            t = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                reg, like, pri, _ = T.train(batch=tb, seed=0, epochs=3000, lr=0.02, lam_gamma=10,
                                            area_mode="fixed-cov-mtn", lam_kappa=10, mtn_standardize="train",
                                            use_labels=True, lq_labels=False, lq_b_only=True)
            train = time.time() - t
            t = time.time()
            with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
                out_r = reg(batch); out_l = like(batch, out_r.mu)
                lj, lp = T.marginalize(T.prior_log_w(pri, batch), out_l.log_L)
                p_ls, p_lq = T.infer(lj, lp)
            with contextlib.redirect_stdout(io.StringIO()):
                T.report_holdout(batch, reg, like, pri, p_ls, p_lq, ev, "_timecompare")
            infer = time.time() - t
            res.append((ev, train, infer))
            log(f"  pilotA {ev}: 학습 {train:.1f}초  추론 {infer:.2f}초")
        return res
    finally:
        os.chdir(here)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilota-dir", required=True, help="pilotA 저장소 followup4-lq-J-nolabels 체크아웃 (raw/ · validation/ 포함)")
    ap.add_argument("--dn-epochs", type=int, default=30)
    ap.add_argument("--dn-threads", type=int, default=2, help="실제 칸 실험과 같은 스레드 수")
    ap.add_argument("--out", default="outputs/time_compare.txt")
    a = ap.parse_args(argv)
    lines = []
    def log(s):
        print(s, flush=True); lines.append(s)
    log("== pilotA (A2, 지진 분리 5번) ==")
    pa = time_pilota(a.pilota_dir, log)
    log("== DisasterNet (칸 단위 LOCALU, 지진별) ==")
    dn = time_dn(a.dn_epochs, a.dn_threads, log)
    log("== 합계 (지진 5개, seed 하나) ==")
    log(f"  DisasterNet 학습: 차례로 {sum(r[2] for r in dn)/60:.1f}분 / 5개 동시(실제 실험 방식) 약 {max(r[2] for r in dn)/60:.1f}분")
    log(f"  DisasterNet 추론: 합계 {sum(r[3] for r in dn):.1f}초")
    log(f"  pilotA 학습: 합계 {sum(r[1] for r in pa)/60:.1f}분   추론: 합계 {sum(r[2] for r in pa):.2f}초")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
