"""
DisasterNet - model.py

Li et al., "DisasterNet: Causal Bayesian Networks with Normalizing Flows for
Cascading Hazards Estimation from Satellite Imagery", KDD 2023 의 재구현.
원 논문은 40 m 픽셀 단위지만 우리 DPM은 시정촌 집계값이라 "한 행 = 한 위치"로 둔다.

인과 그래프 (논문 구조)
-----------------------
    USGS π_LS ─▶ LS ─┐
                     ├─▶ BD ─┐
    USGS π_LQ ─▶ LQ ─┘       ├─▶ DPM (y)
                 LS, LQ ─────┤
                       ε ────┘

    LS, LQ ~ Bern(σ(a·logit π + b))                    ← pilotA의 Prior와 같은 보정식
    BD     ~ Bern(σ(c0 + c_LS·LS + c_LQ·LQ + c_PGV·z(log PGV)))
    ε      ~ N(0, 1)                                    (환경 잡음 노드)
    y      ~ N(w0 + w_LS·LS + w_LQ·LQ + w_BD·BD + w_ε·ε, σ²)

    w_LS, w_LQ, w_BD, c_LS, c_LQ, w_ε, σ 는 softplus로 양수.
    "재해가 있으면 DPM(간섭도 손실)이 커진다"는 물리적 방향만 강제한다.

추론
----
inference="vi"    (기본, 논문 방식)
    위치별(local) 변분분포로 확률적 변분추론(SVI)을 한다.
      q(LS) q(LQ) q(BD)  : 행마다 Bernoulli (mean-field)
      q(ε)               : 행마다 N(m, s²)에 planar normalizing flow K층
    이산 노드 기댓값은 8상태를 열거해 정확히 계산하고, ε만 재파라미터화 샘플링한다.
    생성 파라미터와 변분 파라미터를 하나의 ELBO로 함께 학습한다.

inference="exact" (점검용)
    ε를 해석적으로 적분(y | 상태 ~ N(m_s, σ² + w_ε²))하고 8상태를 정확히 합산해
    주변우도를 최대화한다. pilotA와 같은 방식이다. VI가 제대로 수렴했는지 대조할 때 쓴다.

논문과 다른 점 / 원문 대조가 필요한 점
--------------------------------------
1. 단위: 픽셀 → 시정촌 집계 DPM.
2. BD prior: 논문은 건물 위치/피해 prior를 쓰지만 여기엔 그 자료가 없어 PGV로 대신한다.
3. triplet loss: 논문은 지구물리 prior로 식별성을 높이는 triplet loss를 쓴다.
   정확한 식을 확인하지 못해 넣지 않았다. 원문을 보고 elbo_loss()에 더하면 된다.
4. 이벤트마다 따로 학습한다(논문도 지진마다 따로 학습). --pooled로 합쳐 학습할 수도 있다.
"""
import math
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from prior import Prior
from schema import DTYPE, EPS, INDEX_TO_EVENT

from .data import DPMRows

# 8상태 (LS, LQ, BD). 앞 4개 (LS, LQ) 순서는 schema.LATENT_STATES와 같다.
STATES_LSLQ = ((0, 0), (1, 0), (0, 1), (1, 1))
STATES = tuple((ls, lq, bd) for bd in (0, 1) for (ls, lq) in STATES_LSLQ)


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(x))


def _standardize(x: Tensor) -> Tensor:
    sd = x.std()
    return (x - x.median()) / (sd if sd > 0 else 1.0)


class DisasterNet(nn.Module):
    """한 그룹(기본: 이벤트 1개)의 생성모형 + 위치별 변분분포."""

    def __init__(self, n_rows: int, *, use_bd=True, prior_mode="free", b_bound=2.0,
                 flow_layers=4):
        super().__init__()
        self.use_bd = use_bd
        self.prior = Prior(mode=prior_mode, b_bound=b_bound)

        p = lambda v: nn.Parameter(torch.tensor(float(v), dtype=DTYPE))
        # BD 노드
        self.c0 = p(-1.0)
        self.c_ls_raw = p(_inv_softplus(1.0))
        self.c_lq_raw = p(_inv_softplus(1.0))
        self.c_pgv = p(0.5)
        # 관측 노드 (y는 그룹 안에서 표준화해서 넣으므로 단위는 표준편차)
        self.w0 = p(-0.5)
        self.w_ls_raw = p(_inv_softplus(0.5))
        self.w_lq_raw = p(_inv_softplus(0.5))
        self.w_bd_raw = p(_inv_softplus(0.5))
        self.w_eps_raw = p(_inv_softplus(0.3))
        self.sigma_raw = p(_inv_softplus(0.5))

        # 위치별 변분 파라미터
        self.q_logit = nn.Parameter(torch.zeros(n_rows, 3, dtype=DTYPE))
        self.q_eps_m = nn.Parameter(torch.zeros(n_rows, dtype=DTYPE))
        self.q_eps_logs = nn.Parameter(torch.zeros(n_rows, dtype=DTYPE))
        K = flow_layers
        self.flow_u = nn.Parameter(0.01 * torch.randn(n_rows, K, dtype=DTYPE))
        self.flow_w = nn.Parameter(1.0 + 0.01 * torch.randn(n_rows, K, dtype=DTYPE))
        self.flow_b = nn.Parameter(torch.zeros(n_rows, K, dtype=DTYPE))

    # ---- 양수 제약 파라미터 ----
    c_ls = property(lambda self: F.softplus(self.c_ls_raw))
    c_lq = property(lambda self: F.softplus(self.c_lq_raw))
    w_ls = property(lambda self: F.softplus(self.w_ls_raw))
    w_lq = property(lambda self: F.softplus(self.w_lq_raw))
    w_bd = property(lambda self: F.softplus(self.w_bd_raw) if self.use_bd
                    else torch.zeros((), dtype=DTYPE))
    w_eps = property(lambda self: F.softplus(self.w_eps_raw))
    sigma = property(lambda self: F.softplus(self.sigma_raw) + 1e-3)

    def init_q_from_prior(self, rows: DPMRows, z_pgv: Tensor):
        with torch.no_grad():
            self.q_logit[:, 0] = torch.logit(rows.pi_ls, eps=EPS)
            self.q_logit[:, 1] = torch.logit(rows.pi_lq, eps=EPS)
            self.q_logit[:, 2] = self.c0 + self.c_pgv * z_pgv

    # ---- 생성모형 ----
    def log_prior_states(self, rows: DPMRows, z_pgv: Tensor) -> Tensor:
        """log p(LS, LQ, BD) [N, 8]."""
        log_w = self.prior(rows.pi_ls, rows.pi_lq)                     # [N,4]
        ls = torch.tensor([s[0] for s in STATES_LSLQ], dtype=DTYPE)
        lq = torch.tensor([s[1] for s in STATES_LSLQ], dtype=DTYPE)
        z_bd = (self.c0 + self.c_ls * ls + self.c_lq * lq).unsqueeze(0) \
            + (self.c_pgv * z_pgv).unsqueeze(1)                        # [N,4]
        if self.use_bd:
            log_bd0, log_bd1 = F.logsigmoid(-z_bd), F.logsigmoid(z_bd)
        else:  # BD 노드를 끄면 BD=0으로 고정
            log_bd0 = torch.zeros_like(z_bd)
            log_bd1 = torch.full_like(z_bd, -1e4)
        return torch.cat([log_w + log_bd0, log_w + log_bd1], dim=-1)   # [8] 순서 = STATES

    def obs_mean(self) -> Tensor:
        """상태별 E[y | 상태, ε=0] [8]."""
        s = torch.tensor(STATES, dtype=DTYPE)
        return self.w0 + self.w_ls * s[:, 0] + self.w_lq * s[:, 1] + self.w_bd * s[:, 2]

    # ---- 정확 추론 (점검용) ----
    def exact_log_joint(self, rows, y, z_pgv) -> Tensor:
        var = self.sigma ** 2 + self.w_eps ** 2
        log_lik = -0.5 * ((y.unsqueeze(1) - self.obs_mean()) ** 2 / var + torch.log(2 * math.pi * var))
        return self.log_prior_states(rows, z_pgv) + log_lik             # [N,8]

    # ---- 변분 추론 ----
    def sample_eps(self, n_samples: int):
        """planar flow를 거친 ε 샘플과 log q(ε). 모양 [N, S]."""
        N = self.q_eps_m.shape[0]
        n = torch.randn(N, n_samples, dtype=DTYPE)
        s = torch.exp(self.q_eps_logs).unsqueeze(1)
        z = self.q_eps_m.unsqueeze(1) + s * n
        log_q = -0.5 * (n ** 2 + math.log(2 * math.pi)) - torch.log(s)
        for k in range(self.flow_u.shape[1]):
            w = self.flow_w[:, k:k + 1]
            u = self.flow_u[:, k:k + 1]
            b = self.flow_b[:, k:k + 1]
            # 1차원 planar flow의 가역 조건 u·w > -1 을 보장하는 재파라미터화
            wu = w * u
            w_safe = torch.where(w.abs() < 1e-6, torch.full_like(w, 1e-6), w)
            u_hat = (-1.0 + F.softplus(wu)) / w_safe
            h = torch.tanh(w * z + b)
            log_det = torch.log((1.0 + u_hat * w * (1.0 - h ** 2)).abs() + 1e-12)
            z = z + u_hat * h
            log_q = log_q - log_det
        return z, log_q

    def elbo(self, rows, y, z_pgv, n_samples=8) -> Tensor:
        """행별 ELBO [N]."""
        logits = self.q_logit if self.use_bd else torch.cat(
            [self.q_logit[:, :2], torch.full_like(self.q_logit[:, 2:], -30.0)], dim=1)
        s = torch.tensor(STATES, dtype=DTYPE)                           # [8,3]
        log_q1, log_q0 = F.logsigmoid(logits), F.logsigmoid(-logits)    # [N,3]
        log_qs = log_q1 @ s.T + log_q0 @ (1 - s).T                     # [N,8]
        qs = log_qs.exp()

        eps, log_q_eps = self.sample_eps(n_samples)                     # [N,S]
        mean = self.obs_mean().view(1, 8, 1) + self.w_eps * eps.unsqueeze(1)
        var = self.sigma ** 2
        log_lik = -0.5 * ((y.view(-1, 1, 1) - mean) ** 2 / var + torch.log(2 * math.pi * var))
        e_log_lik = (qs * log_lik.mean(-1)).sum(-1)                     # [N]

        e_log_prior = (qs * self.log_prior_states(rows, z_pgv)).sum(-1)
        entropy_s = -(qs * log_qs).sum(-1)
        log_p_eps = -0.5 * (eps ** 2 + math.log(2 * math.pi))
        kl_eps = (log_q_eps - log_p_eps).mean(-1)
        return e_log_lik + e_log_prior + entropy_s - kl_eps

    def posterior(self, rows, y, z_pgv, inference):
        """P(LS=1|y), P(LQ=1|y), P(BD=1|y) [N]."""
        if inference == "exact":
            lj = self.exact_log_joint(rows, y, z_pgv)
            post = torch.softmax(lj, dim=-1)
            s = torch.tensor(STATES, dtype=DTYPE)
            p = post @ s
            return p[:, 0], p[:, 1], p[:, 2]
        p = torch.sigmoid(self.q_logit)
        p_bd = p[:, 2] if self.use_bd else torch.zeros_like(p[:, 2])
        return p[:, 0], p[:, 1], p_bd

    def param_rows(self, group: str):
        a, b = self.prior.a_value.detach(), self.prior.b_value.detach()
        vals = {
            "a_LS": a[0], "b_LS": b[0], "a_LQ": a[1], "b_LQ": b[1],
            "c0_BD": self.c0, "c_LS→BD": self.c_ls, "c_LQ→BD": self.c_lq, "c_PGV→BD": self.c_pgv,
            "w0": self.w0, "w_LS": self.w_ls, "w_LQ": self.w_lq, "w_BD": self.w_bd,
            "w_eps": self.w_eps, "sigma": self.sigma,
        }
        return [{"group": group, "param": k, "value": float(v.detach())} for k, v in vals.items()]


@dataclass
class FitResult:
    predictions: pd.DataFrame   # event_idx / muni_code / model_row_idx / dpm / p_ls / p_lq / p_bd
    params: pd.DataFrame
    history: pd.DataFrame


def _fit_group(rows: DPMRows, name: str, *, inference, epochs, lr, seed, n_samples,
               use_bd, prior_mode, b_bound, flow_layers, verbose):
    torch.manual_seed(seed)
    y = _standardize(rows.y)
    z_pgv = _standardize(torch.log(rows.pgv.clamp_min(EPS)))

    model = DisasterNet(rows.n, use_bd=use_bd, prior_mode=prior_mode, b_bound=b_bound,
                        flow_layers=flow_layers)
    model.init_q_from_prior(rows, z_pgv)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    hist = []
    for epoch in range(epochs):
        if inference == "exact":
            loss = -torch.logsumexp(model.exact_log_joint(rows, y, z_pgv), dim=-1).sum()
        else:
            loss = -model.elbo(rows, y, z_pgv, n_samples=n_samples).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append({"group": name, "epoch": epoch, "loss": float(loss.detach())})
        if verbose and epoch % 500 == 0:
            print(f"  [{name}] {epoch:5d}  loss/row {float(loss) / rows.n:.4f}")

    with torch.no_grad():
        p_ls, p_lq, p_bd = model.posterior(rows, y, z_pgv, inference)
    pred = pd.DataFrame({
        "event_idx": rows.event_idx.tolist(),
        "muni_code": list(rows.municipality_code),
        "model_row_idx": rows.model_row_idx.tolist(),
        "dpm": rows.y.numpy(),
        "p_ls": p_ls.numpy(), "p_lq": p_lq.numpy(), "p_bd": p_bd.numpy(),
    })
    return pred, model.param_rows(name), hist


def fit_disasternet(rows: DPMRows, *, inference="vi", pooled=False, epochs=3000, lr=0.02,
                    seed=0, n_samples=8, use_bd=True, prior_mode="free", b_bound=2.0,
                    flow_layers=4, verbose=True) -> FitResult:
    """이벤트별(기본) 또는 통합(pooled)으로 DisasterNet을 학습하고 사후확률을 돌려준다.

    통합 학습에서도 DPM은 이벤트 안에서 표준화한다. 프레임/이벤트마다 자가 다르기 때문이다.
    """
    if inference not in ("vi", "exact"):
        raise ValueError(f"inference는 'vi' 또는 'exact': {inference}")

    if pooled:
        y_std = torch.empty_like(rows.y)
        for e in rows.event_idx.unique().tolist():
            m = rows.event_idx == e
            y_std[m] = _standardize(rows.y[m])
        pooled_rows = DPMRows(y=y_std, pi_ls=rows.pi_ls, pi_lq=rows.pi_lq, pgv=rows.pgv,
                              event_idx=rows.event_idx, municipality_code=rows.municipality_code,
                              model_row_idx=rows.model_row_idx)
        groups = [("pooled", pooled_rows)]
    else:
        groups = [(INDEX_TO_EVENT[e], rows.subset(rows.event_idx == e))
                  for e in sorted(rows.event_idx.unique().tolist())]

    preds, params, hist = [], [], []
    for name, g in groups:
        if verbose:
            print(f"[DisasterNet/{inference}] {name}: {g.n}행")
        p, prm, h = _fit_group(g, name, inference=inference, epochs=epochs, lr=lr, seed=seed,
                               n_samples=n_samples, use_bd=use_bd, prior_mode=prior_mode,
                               b_bound=b_bound, flow_layers=flow_layers, verbose=verbose)
        if pooled:  # 표준화 전 원래 DPM 값을 남긴다
            p["dpm"] = rows.y.numpy()
        preds.append(p); params.extend(prm); hist.extend(h)

    pred = pd.concat(preds, ignore_index=True).sort_values(["event_idx", "muni_code"])
    return FitResult(pred.reset_index(drop=True), pd.DataFrame(params), pd.DataFrame(hist))
