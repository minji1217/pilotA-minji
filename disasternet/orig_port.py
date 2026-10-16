"""
DisasterNet 공식 코드(method/model/model.py, method/optimization/exp.py)를 그대로 옮긴 버전.

우리 구현(model.py)에서 바꿨던 것을 모두 원본대로 되돌린다.
  · 중심 μ = w_mu × (w_a × α)            (시정촌별 오프셋 없음, ±6 묶기 없음)
  · 출발점은 위치마다 1개                  (8개 평균 아님)
  · r⁰ 직후 sigmoid, flow 8단계, 매 단계 불확실도·LOCAL로 0 덮기, 끝에 다시 sigmoid
  · planar flow는 배치 전체를 N차원 벡터 하나로 변환 (u, w ∈ R^N, b 스칼라, log det도 배치 전체에 하나)
  · u, w, b = Linear(N, K·N) / Linear(N, K) + Hardtanh(−1, 1)  (배치의 모든 위치 값을 한 벡터로 받음)
  · 건물 분산 = Linear(N, N) → Softplus → Hardtanh(1, 5)
  · 관측식 log p(x|z), prior log p(z), 손실(--anneal std), β 식, triplet, RMSprop(lr 1e-5, momentum 0.9), 30 epoch
  · 최종 확률 = 마지막 epoch에서 계산한 q (한 번 뽑은 값)

원본과 다를 수밖에 없는 것 (입력이 시정촌 단위라서)
  · 배치 = 지진 하나의 시정촌 전체 (원본은 화소 8,109개씩). 그래서 epoch마다 갱신은 1번.
  · LOCAL 표가 없음 → 어떤 노드도 끄지 않는 값(6)을 넣는다 (원본 번호 0~5 중에는 셋을 다 켜는 번호가 없다).
  · 건물 윤곽(ft)이 없음 → 전부 1 (끄지 않음).
  · qvar = 시정촌 안 USGS 켠 칸의 불확실도 평균, 켠 칸이 없으면 999 (원본 규칙으로 꺼짐).
  · α_BD = 지진 안 log PGV를 0~1로 (원본은 취약도 곡선 지도), 관측 x = DPM_배경차 + 2.
  · 원본 코드에 정의가 빠진 log_normal_dist는 같은 계열 코드(Sylvester flows)의 정의를 쓴다:
    Σ −½ (log σ² + (x − μ)² / σ²).

    python -m disasternet.orig_port --epochs 30 --out-dir outputs/disasternet/orig_port_e30 --jobs 5
"""
import argparse
import math
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .run_input_v5 import load_table
from .run_muni_unc import muni_usgs

NO_LOCAL = 6  # 원본 LOCAL 번호(0~5) 어디에도 걸리지 않는 값 → 아무것도 끄지 않음


def log_normal_dist(x, mean, logvar, dim):
    return torch.sum(-0.5 * (logvar + (x - mean) ** 2 * torch.exp(-logvar)), dim)


class Planar(nn.Module):
    """method/flow/flow.py 그대로."""

    def __init__(self):
        super().__init__()
        self.h = nn.Tanh()

    def forward(self, z, u, w, b):
        z = z.unsqueeze(2)
        prod = torch.bmm(w, z) + b
        f_z = z + u * self.h(prod)
        f_z = f_z.squeeze(2)
        psi = w * (1 - self.h(prod) ** 2)
        log_det_jacobian = torch.log(torch.abs(1 + torch.bmm(psi, u)))
        return f_z, log_det_jacobian.squeeze(2).squeeze(1)


class OrigDisasterNet(nn.Module):
    """method/model/model.py의 DisasterNet 그대로 (cuda 분기·쓰지 않는 인자만 뺐다)."""

    def __init__(self, batch_size, num_flows=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_flows = num_flows
        self.m = nn.Sigmoid()
        for k in range(num_flows):
            self.add_module("flow_LS_" + str(k), Planar())
        for k in range(num_flows):
            self.add_module("flow_LF_" + str(k), Planar())
        for k in range(num_flows):
            self.add_module("flow_BD_" + str(k), Planar())
        N = batch_size
        self.mu = nn.Sequential(nn.Linear(N, N), nn.Hardtanh(min_val=-2, max_val=2))  # 원본도 만들기만 하고 안 씀
        self.var = nn.Sequential(nn.Linear(N, N), nn.Softplus(), nn.Hardtanh(min_val=1, max_val=5))
        self.amor_u = nn.Sequential(nn.Linear(N, num_flows * N), nn.Hardtanh(min_val=-1, max_val=1))
        self.amor_w = nn.Sequential(nn.Linear(N, num_flows * N), nn.Hardtanh(min_val=-1, max_val=1))
        self.amor_b = nn.Sequential(nn.Linear(N, num_flows), nn.Hardtanh(min_val=-1, max_val=1))
        p = lambda: nn.Parameter(torch.rand(1, dtype=torch.float))
        for name in ("w_0LS", "w_0LF", "w_0BD", "w_eLS", "w_eLF", "w_eBD", "w_ex", "w_0x", "var_x",
                     "w_LSx", "w_LFx", "w_BDx", "w_LSBD", "w_LFBD", "w_aBD", "w_aLS", "w_aLF",
                     "w_mu_LS", "b_mu_LS", "w_mu_LF", "b_mu_LF", "w_mu_BD", "b_mu_BD"):
            setattr(self, name, p())

    def forward(self, a_LS, a_LF, a_BD, x, qvar_LS, qvar_LF, ft, local):
        K, N = self.num_flows, self.batch_size
        self.log_det_j_LS = torch.zeros([x.shape[0]])
        self.log_det_j_LF = torch.zeros([x.shape[0]])
        self.log_det_j_BD = torch.zeros([x.shape[0]])

        a_LS_1 = self.w_aLS * a_LS
        a_LF_1 = self.w_aLF * a_LF

        z_LS_mu = self.w_mu_LS * a_LS_1
        std_LS = qvar_LS.sqrt()
        std_LS[qvar_LS == 999] = 0
        eps_LS = torch.randn_like(std_LS)
        z_LS = eps_LS * std_LS + z_LS_mu
        z_LS[qvar_LS == 999] = 0
        z_LS[qvar_LS == 0] = 0
        z_LS[z_LS != 0] = self.m(z_LS[z_LS != 0])

        z_LF_mu = self.w_mu_LF * a_LF_1
        std_LF = qvar_LF.sqrt()
        std_LF[qvar_LF == 999] = 0
        eps_LF = torch.randn_like(std_LF)
        z_LF = eps_LF * std_LF + z_LF_mu
        z_LF[qvar_LF == 999] = 0
        z_LF[qvar_LF == 0] = 0
        z_LF[z_LF != 0] = self.m(z_LF[z_LF != 0])

        z_LS = [z_LS]
        z_LF = [z_LF]
        u_LS = self.amor_u(a_LS_1).view(-1, K, N, 1)
        w_LS = self.amor_w(a_LS_1).view(-1, K, 1, N)
        b_LS = self.amor_b(a_LS_1).view(-1, K, 1, 1)
        u_LF = self.amor_u(a_LF_1).view(-1, K, N, 1)
        w_LF = self.amor_w(a_LF_1).view(-1, K, 1, N)
        b_LF = self.amor_b(a_LF_1).view(-1, K, 1, 1)

        for k in range(K):
            z_LFk, ldj = getattr(self, "flow_LF_" + str(k))(z_LF[k], u_LF[:, k], w_LF[:, k], b_LF[:, k])
            z_LFk[qvar_LF == 999] = 0
            z_LFk[qvar_LF == 0] = 0
            z_LFk[(local == 0) | (local == 1) | (local == 3) | (local == 5)] = 0
            z_LF.append(z_LFk)
            self.log_det_j_LF = self.log_det_j_LF + ldj
        for k in range(K):
            z_LSk, ldj = getattr(self, "flow_LS_" + str(k))(z_LS[k], u_LS[:, k], w_LS[:, k], b_LS[:, k])
            z_LSk[qvar_LS == 999] = 0
            z_LSk[qvar_LS == 0] = 0
            z_LSk[(local == 0) | (local == 2) | (local == 4)] = 0
            z_LS.append(z_LSk)
            self.log_det_j_LS = self.log_det_j_LS + ldj

        z_LS_0, z_LF_0 = z_LS[0], z_LF[0]
        z_LS_K, z_LF_K = z_LS[-1], z_LF[-1]
        z_LS_K[z_LS_K != 0] = self.m(z_LS_K[z_LS_K != 0])
        z_LF_K[z_LF_K != 0] = self.m(z_LF_K[z_LF_K != 0])

        a_BD_1 = self.w_LSBD * z_LS_K.view(-1, N) + self.w_LFBD * z_LF_K.view(-1, N) + self.w_aBD * a_BD
        z_BD_mu = self.w_mu_BD * a_BD_1
        z_BD_var = self.var(a_BD_1)
        u_BD = self.amor_u(a_BD_1).view(-1, K, N, 1)
        w_BD = self.amor_w(a_BD_1).view(-1, K, 1, N)
        b_BD = self.amor_b(a_BD_1).view(-1, K, 1, 1)
        std_BD = z_BD_var.sqrt()
        eps_BD = torch.randn_like(std_BD)
        z_BD = eps_BD * std_BD + z_BD_mu
        z_BD[ft == 0] = 0
        z_BD[z_BD != 0] = self.m(z_BD[z_BD != 0])
        z_BD = [z_BD]
        for k in range(K):
            z_BDk, ldj = getattr(self, "flow_BD_" + str(k))(z_BD[k], u_BD[:, k], w_BD[:, k], b_BD[:, k])
            z_BDk[(local == 0) | (local == 1) | (local == 2) | (local == 5)] = 0
            z_BD.append(z_BDk)
            self.log_det_j_BD = self.log_det_j_BD + ldj
        z_BD_0, z_BD_K = z_BD[0], z_BD[-1]
        z_BD_K[z_BD_K != 0] = self.m(z_BD_K[z_BD_K != 0])

        BD_inner_1 = (torch.exp(-self.w_LSBD - self.w_LFBD) * z_LS_K * z_LF_K + (1 - z_LS_K) * (1 - z_LF_K)
                      + torch.exp(-self.w_LSBD) * z_LS_K * (1 - z_LF_K) + torch.exp(-self.w_LFBD) * z_LF_K * (1 - z_LS_K))
        BD_inner_2 = (torch.exp(self.w_LSBD + self.w_LFBD) * z_LS_K * z_LF_K + (1 - z_LS_K) * (1 - z_LF_K)
                      + torch.exp(self.w_LSBD) * z_LS_K * (1 - z_LF_K) + torch.exp(self.w_LFBD) * z_LF_K * (1 - z_LS_K))
        p_z_BD = (z_BD_K * (-torch.log(1 + torch.exp(-(self.w_eBD ** 2) / 2 - self.w_0BD - self.w_aBD * a_BD) * BD_inner_1))
                  + (1 - z_BD_K) * (-torch.log(1 + torch.exp((self.w_eBD ** 2) / 2 + self.w_0BD + self.w_aBD * a_BD) * BD_inner_2)))
        p_z_LS = (z_LS_K * (-torch.log(1 + torch.exp(-(self.w_eLS ** 2) / 2 - self.w_0LS - self.w_aLS * a_LS)))
                  + (1 - z_LS_K) * (-torch.log(1 + torch.exp((self.w_eLS ** 2) / 2 + self.w_0LS + self.w_aLS * a_LS))))
        p_z_LF = (z_LF_K * (-torch.log(1 + torch.exp(-(self.w_eLF ** 2) / 2 - self.w_0LF - self.w_aLF * a_LF)))
                  + (1 - z_LF_K) * (-torch.log(1 + torch.exp((self.w_eLF ** 2) / 2 + self.w_0LF + self.w_aLF * a_LF))))
        logp_zk = torch.mean(p_z_BD + p_z_LS + p_z_LF)

        log_p_xz = (- torch.log(x) - torch.log(torch.abs(torch.sqrt(self.w_ex)))
                    - (torch.log(x) ** 2) / (2 * (self.w_ex ** 2))
                    + (torch.log(x) * (self.w_BDx * z_BD_K + self.w_LSx * z_LS_K + self.w_LFx * z_LF_K + self.w_0x)) / (self.w_ex ** 2)
                    - ((self.w_BDx ** 2) * z_BD_K + (self.w_LSx ** 2) * z_LS_K + (self.w_LFx ** 2) * z_LF_K + self.w_ex ** 2 + self.w_0x ** 2) / (2 * (self.w_ex ** 2))
                    - (self.w_BDx * self.w_LSx * z_BD_K * z_LS_K + self.w_BDx * self.w_LFx * z_BD_K * z_LF_K + self.w_LSx * self.w_LFx * z_LS_K * z_LF_K) / (self.w_ex ** 2)
                    - (self.w_BDx * self.w_0x * z_BD_K + self.w_LSx * self.w_0x * z_LS_K + self.w_LFx * self.w_0x * z_LF_K) / (self.w_ex) ** 2)

        return (z_LS_mu, qvar_LS, z_LS_0, z_LS_K, self.log_det_j_LS, z_LF_mu, qvar_LF, z_LF_0, z_LF_K, self.log_det_j_LF,
                z_BD_mu, z_BD_var, z_BD_0, z_BD_K, self.log_det_j_BD, log_p_xz, logp_zk)


def train_event(t, seed, epochs, lr=1e-5, num_flows=8):
    """optimization/exp.py의 run()을 배치 하나(= 지진 하나의 시정촌 전체)로. 반환: 마지막 epoch의 q."""
    torch.manual_seed(seed)
    N = t["aLS"].shape[1]
    model = OrigDisasterNet(N, num_flows)
    opt = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9)
    n_batches, index, batch_size = 1, 0, N
    for epoch in range(1, epochs + 1):
        beta = min(0.01 + ((epoch - 1) * (n_batches / batch_size) + index + 1) / epochs, 1)
        (q_LS_mu, q_LS_var, q_LS_0, q_LS_K, ldj_LS, q_LF_mu, q_LF_var, q_LF_0, q_LF_K, ldj_LF,
         q_BD_mu, q_BD_var, q_BD_0, q_BD_K, ldj_BD, log_p_xz, logp_zk) = model(
            t["aLS"], t["aLF"], t["aBD"], t["obs"], t["varLS"].clone(), t["varLF"].clone(), t["ft"], t["loc"])
        out = (q_LS_K.detach().clone(), q_LF_K.detach().clone(), q_BD_K.detach().clone())
        log_q_z0 = (log_normal_dist(q_LS_0, q_LS_mu, q_LS_var.log(), 1) + log_normal_dist(q_LF_0, q_LF_mu, q_LF_var.log(), 1)
                    + log_normal_dist(q_BD_0, q_BD_mu, q_BD_var.log(), 1)) / batch_size
        log_det_jacobians = ldj_LS + ldj_LF + ldj_BD
        tl = nn.TripletMarginLoss(margin=1.0, p=2)
        penalty = tl(q_LS_K, t["aLS"], t["aLF"]) + tl(q_LF_K, t["aLF"], t["aLS"])
        kl = log_q_z0 - beta * logp_zk - log_det_jacobians
        loss = beta * torch.abs(torch.mean(log_p_xz)) + kl + penalty
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    return out, float(loss.sum().detach())


def event_tensors(g):
    f = lambda c: torch.tensor(g[c].to_numpy(dtype="float32")).view(1, -1)
    lp = np.log(np.clip(g["PGV"].to_numpy(dtype="float64"), 1e-6, None))
    a_bd = (lp - lp.min()) / ((lp.max() - lp.min()) or 1.0)
    var = lambda k: torch.tensor(np.where(g[f"n_on_{k}"] > 0, g[f"var_{k}"].fillna(999), 999.0).astype("float32")).view(1, -1)
    N = len(g)
    return {"aLS": f("LS_prior_평균"), "aLF": f("LQ_prior_최대"), "aBD": torch.tensor(a_bd, dtype=torch.float32).view(1, -1),
            "obs": torch.tensor((g["DPM_배경차"] + 2.0).to_numpy(dtype="float32")).view(1, -1),
            "varLS": var("ls"), "varLF": var("lq"), "ft": torch.ones(1, N), "loc": torch.full((1, N), float(NO_LOCAL))}


def _job(job):
    df, seed, epochs, out_dir = job
    torch.set_num_threads(1)
    parts = []
    for e, g in df.groupby("event_idx", sort=True):
        g = g.reset_index(drop=True)
        (ls, lf, bd), loss = train_event(event_tensors(g), seed, epochs)
        parts.append(pd.DataFrame({"seed": seed, "event_idx": e, "muni_code": g["muni_code"],
                                   "p_ls": ls.view(-1).numpy(), "p_lq": lf.view(-1).numpy(), "p_bd": bd.view(-1).numpy()}))
        print(f"seed {seed} event {e} N={len(g)} loss {loss:.3f}", flush=True)
    p = pd.concat(parts, ignore_index=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p.to_csv(Path(out_dir) / f"predictions_seed{seed}.csv", index=False, encoding="utf-8-sig")
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description="DisasterNet 공식 코드 그대로 (시정촌 입력)")
    ap.add_argument("--input", default="raw/DisasterNet_입력_v5.xlsx")
    ap.add_argument("--local", default="raw/local_250m_byevent_usgs.csv")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out-dir", default="outputs/disasternet/orig_port_e30")
    a = ap.parse_args(argv)
    df = load_table(a.input).merge(muni_usgs(a.local), on=["이벤트", "muni_code"], how="left", validate="one_to_one")
    jobs = [(df, s, a.epochs, a.out_dir) for s in a.seeds]
    if a.jobs > 1:
        with get_context("fork").Pool(a.jobs) as pool:
            pool.map(_job, jobs)
    else:
        for j in jobs:
            _job(j)
    print(f"저장: {a.out_dir}/")


if __name__ == "__main__":
    main()
