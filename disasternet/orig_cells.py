"""
DisasterNet 공식 코드(orig_port.py와 같은 식)를 칸(250 m) 입력에 그대로 — 학습 방식을 원본대로 되돌린 비교판.

orig_port.py(시정촌판)와 다른 점
  · BD 노드 없음 (교수님 지시). BD 관련 항(사전·관측·log q·log det)을 모두 뺀다.
  · 칸마다 자기 DPM(DPM_배경차 + 2), 자기 USGS 확률·불확실도(면적 평균 파일), 물 칸 제외 — run_cells_max와 같은 칸
  · LOCAL: 끄기 지도(산사태_켬·액상화_켬)를 원본 번호로 — 둘 다 켬 6, 산사태만 끔 2, 액상화만 끔 1, 둘 다 끔 0
  · USGS 불확실도 0·999 → 원본 규칙대로 끔 (논문 기준). 확률 999는 0.
  · 배치: 원본 실행은 8,109칸이지만 u·w = Linear(N, K·N)이라 N = 8,109면 가중치 10억 개(메모리 17 GB 이상)다.
    기본값은 공식 코드 DisasterNet.py의 argparse 기본값 1,000 (--batch로 바꿀 수 있음). 지진마다 칸을 한 번 섞어 배치로 나누고, 마지막 배치의 모자란 자리는
    다른 칸으로 채워 돌린다 (채운 자리의 결과는 버림).
  · 결과 = 마지막 epoch에서 각 칸이 받은 q (한 번 뽑은 값). 시정촌 값 = 칸 최댓값.
  · 지진마다 muni_*.csv를 저장하고, 다시 실행하면 끝난 지진은 건너뛴다. 지진 안에서는 epoch마다 중간 저장(ckpt_*.pt).

    python -m disasternet.orig_cells --out-dir outputs/disasternet/orig_cells_aw_b1000
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .orig_port import Planar, log_normal_dist
from .run_cells_max import EVS, load_cells

LS_OFF = torch.tensor([0., 2., 4.])        # 원본: LOCAL 0·2·4면 산사태 끔
LF_OFF = torch.tensor([0., 1., 3., 5.])    # 원본: LOCAL 0·1·3·5면 액상화 끔


class OrigCellsNet(nn.Module):
    """orig_port.OrigDisasterNet에서 BD만 뺀 것."""

    def __init__(self, N, K=8):
        super().__init__()
        self.N, self.K, self.m = N, K, nn.Sigmoid()
        self.flow = Planar()
        self.amor_u = nn.Sequential(nn.Linear(N, K * N), nn.Hardtanh(-1, 1))
        self.amor_w = nn.Sequential(nn.Linear(N, K * N), nn.Hardtanh(-1, 1))
        self.amor_b = nn.Sequential(nn.Linear(N, K), nn.Hardtanh(-1, 1))
        p = lambda: nn.Parameter(torch.rand(1))
        for name in ("w_0LS", "w_0LF", "w_eLS", "w_eLF", "w_ex", "w_0x", "w_LSx", "w_LFx", "w_aLS", "w_aLF",
                     "w_mu_LS", "w_mu_LF"):
            setattr(self, name, p())

    def node(self, a1, w_mu, qvar, local, off_codes):
        K, N = self.K, self.N
        mu = w_mu * a1
        std = qvar.sqrt()
        std[qvar == 999] = 0
        z = torch.randn_like(std) * std + mu
        z[(qvar == 999) | (qvar == 0)] = 0
        z[z != 0] = self.m(z[z != 0])
        u = self.amor_u(a1).view(-1, K, N, 1)
        w = self.amor_w(a1).view(-1, K, 1, N)
        b = self.amor_b(a1).view(-1, K, 1, 1)
        zs, ldj = [z], 0.
        for k in range(K):
            zk, l = self.flow(zs[k], u[:, k], w[:, k], b[:, k])
            zk[(qvar == 999) | (qvar == 0)] = 0
            zk[torch.isin(local, off_codes)] = 0
            zs.append(zk)
            ldj = ldj + l
        zK = zs[-1]
        zK[zK != 0] = self.m(zK[zK != 0])
        return mu, zs[0], zK, ldj

    def forward(self, aLS, aLF, x, vLS, vLF, local):
        a1LS, a1LF = self.w_aLS * aLS, self.w_aLF * aLF
        muLS, z0LS, zLS, ldjLS = self.node(a1LS, self.w_mu_LS, vLS, local, LS_OFF)
        muLF, z0LF, zLF, ldjLF = self.node(a1LF, self.w_mu_LF, vLF, local, LF_OFF)
        pz = lambda z, w0, we, wa, a: (z * (-torch.log(1 + torch.exp(-(we ** 2) / 2 - w0 - wa * a)))
                                       + (1 - z) * (-torch.log(1 + torch.exp((we ** 2) / 2 + w0 + wa * a))))
        logp_zk = torch.mean(pz(zLS, self.w_0LS, self.w_eLS, self.w_aLS, aLS) + pz(zLF, self.w_0LF, self.w_eLF, self.w_aLF, aLF))
        lx, we2 = torch.log(x), self.w_ex ** 2
        log_p_xz = (-lx - torch.log(torch.abs(torch.sqrt(self.w_ex))) - lx ** 2 / (2 * we2)
                    + lx * (self.w_LSx * zLS + self.w_LFx * zLF + self.w_0x) / we2
                    - ((self.w_LSx ** 2) * zLS + (self.w_LFx ** 2) * zLF + we2 + self.w_0x ** 2) / (2 * we2)
                    - (self.w_LSx * self.w_LFx * zLS * zLF) / we2
                    - (self.w_LSx * self.w_0x * zLS + self.w_LFx * self.w_0x * zLF) / we2)
        return muLS, z0LS, zLS, ldjLS, muLF, z0LF, zLF, ldjLF, log_p_xz, logp_zk


def train_event(c, N, epochs, seed, lr=1e-5, K=8, tag="", ckpt=None):
    """exp.py의 run()을 칸 배치로. 반환: 마지막 epoch에서 각 칸이 받은 q."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(c)
    nb = int(np.ceil(n / N))
    pad = nb * N - n
    idx = np.concatenate([rng.permutation(n), rng.choice(n, pad, replace=False)]).reshape(nb, N)
    f = lambda col: torch.tensor(c[col].to_numpy(dtype="float32"))
    aLS, aLF, x = f("us_ls"), f("us_lq"), f("DPM_배경차") + 2.0
    vLS = torch.tensor(np.where(c["off_ls"], 999.0, c["sd_ls"].fillna(999)).astype("float32"))
    vLF = torch.tensor(np.where(c["off_lq"], 999.0, c["sd_lq"].fillna(999)).astype("float32"))
    ls_on, lq_on = c["산사태_켬"].to_numpy() != 0, c["액상화_켬"].to_numpy() != 0
    local = torch.tensor(np.select([ls_on & lq_on, ~ls_on & lq_on, ls_on & ~lq_on], [6, 2, 1], 0).astype("float32"))
    model = OrigCellsNet(N, K)
    opt = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9)
    tl = nn.TripletMarginLoss(margin=1.0, p=2)
    qLS, qLF = np.zeros(n, dtype="float32"), np.zeros(n, dtype="float32")
    start = 1
    if ckpt is not None and Path(ckpt).exists():
        st = torch.load(ckpt, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        torch.set_rng_state(st["rng"])
        start = st["epoch"] + 1
        print(f"  [{tag}] 중간 저장에서 이어서: epoch {start}", flush=True)
    t0 = time.time()
    for epoch in range(start, epochs + 1):
        for bi in range(nb):
            ii = torch.as_tensor(idx[bi])
            v = lambda t: t[ii].view(1, -1).clone()
            beta = min(0.01 + ((epoch - 1) * (nb / N) + bi + 1) / epochs, 1)   # exp.py 식 그대로
            vls, vlf = v(vLS), v(vLF)
            muLS, z0LS, zLS, ldjLS, muLF, z0LF, zLF, ldjLF, log_p_xz, logp_zk = model(
                v(aLS), v(aLF), v(x), vls.clone(), vlf.clone(), v(local))
            if epoch == epochs:   # 마지막 배치 뒤쪽 pad칸은 채운 자리라 버린다
                keep = N if bi < nb - 1 else N - pad
                qLS[idx[bi][:keep]] = zLS.detach().view(-1).numpy()[:keep]
                qLF[idx[bi][:keep]] = zLF.detach().view(-1).numpy()[:keep]
            log_q_z0 = (log_normal_dist(z0LS, muLS, vls.log(), 1) + log_normal_dist(z0LF, muLF, vlf.log(), 1)) / N
            penalty = tl(zLS, v(aLS), v(aLF)) + tl(zLF, v(aLF), v(aLS))
            kl = log_q_z0 - beta * logp_zk - (ldjLS + ldjLF)
            loss = beta * torch.abs(torch.mean(log_p_xz)) + kl + penalty
            opt.zero_grad()
            loss.sum().backward()
            opt.step()
        print(f"  [{tag}] epoch {epoch}/{epochs}  loss {float(loss.sum().detach()):.3f}  {time.time() - t0:.0f}s", flush=True)
        if ckpt is not None and epoch < epochs:
            tmp = Path(str(ckpt) + ".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "rng": torch.get_rng_state(), "epoch": epoch}, tmp)
            tmp.replace(ckpt)
    if ckpt is not None:
        Path(ckpt).unlink(missing_ok=True)
    return qLS, qLF


def main(argv=None):
    ap = argparse.ArgumentParser(description="DisasterNet 공식 코드 학습 방식 그대로 (칸 입력, BD 없음)")
    ap.add_argument("--local", default="raw/local_250m_byevent_usgs_aw.csv.gz")
    ap.add_argument("--dpm", default="raw/dpm_250m_v8_slim.csv.gz")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--events", nargs="+", default=EVS)
    ap.add_argument("--out-dir", default="outputs/disasternet/orig_cells_aw_b1000")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    allc = load_cells(a.local, a.dpm, off_rule="paper")
    parts = []
    for ev in a.events:
        f = out / f"muni_{ev.replace(' ', '')}_seed{a.seed}.csv"
        if f.exists():
            parts.append(pd.read_csv(f, dtype={"muni_code": str}))
            print(f"이미 있음 {ev}", flush=True)
            continue
        c = allc[allc["이벤트"] == ev].reset_index(drop=True)
        qls, qlf = train_event(c, a.batch, a.epochs, a.seed, tag=f"{ev} {len(c)}칸",
                               ckpt=out / f"ckpt_{ev.replace(' ', '')}_seed{a.seed}.pt")
        c = c.assign(p_ls=qls, p_lq=qlf)
        m = c.groupby(["event_idx", "muni_code"]).agg(n_cells=("mesh250", "size"), p_ls=("p_ls", "max"), p_lq=("p_lq", "max"),
                                                       p_ls_mean=("p_ls", "mean"), p_lq_mean=("p_lq", "mean")).reset_index()
        m.insert(0, "seed", a.seed)
        m.to_csv(f, index=False, encoding="utf-8-sig")
        parts.append(m)
        print(f"완료 {ev}", flush=True)
    if len(parts) == len(a.events):
        pd.concat(parts).to_csv(out / f"predictions_seed{a.seed}.csv", index=False, encoding="utf-8-sig")
    print(f"저장: {out}/")


if __name__ == "__main__":
    main()
