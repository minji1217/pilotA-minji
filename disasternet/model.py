"""
DisasterNet - model.py

Li et al., "DisasterNet: Causal Bayesian Networks with Normalizing Flows for
Cascading Hazards Estimation from Satellite Imagery", KDD 2023 (pp. 4391-4403) 재구현.
공식 코드: https://github.com/SusuXu/DisasterNet (Haiti 전용 경로가 박혀 있어 그대로는 실행되지 않는다.
            세부 설정 확인용으로만 참고했다.)

원 논문은 DPM 픽셀 단위지만 우리 DPM은 시정촌 집계값이라 "한 행 = 한 위치 l"로 둔다.

인과 그래프 (논문 Fig. 2)
------------------------
    α_LS, ε_LS ─▶ z_LS ─┐
                        ├─▶ z_BD ─┐
    α_LF, ε_LF ─▶ z_LF ─┘  ▲       ├─▶ x (DPM)
                 α_BD, ε_BD┘       │
              z_LS, z_LF ──────────┤
                         ε_x ──────┘
    (논문 표기 LF = 우리 LQ)

노드 분포 (식 1, 9 / 부록 6.3)
    logit p(z_i=1 | 부모, ε_i) = Σ_j w_ji z_j + w_αi α_i + w_εi ε_i + w_0i ,  ε_i ~ N(0,1)
    BD의 부모 LS, LF는 noisy-OR 형태로 들어간다.

normalizing flow로 만드는 변분분포 (식 2-6)
    r_i^0 ~ N(μ_i, σ_i²),  μ_i = φ(부모; W_μ),  σ_i² = softplus(δ(부모; W_σ))
    r_i^K = f_K ∘ … ∘ f_1(r_i^0),  f(r) = r + u·h(w·r + b)  (planar, h = tanh)
    q_i   = (1/M) Σ_m sigmoid(r_i^K,m)                             ← P(z_i=1 | 관측)
    μ와 flow 파라미터 (u, w, b)는 부모 입력에서 amortize한 부분 + 위치별 자유 오프셋.

관측 (식 7)
    x | 부모 ~ LogNormal(Σ_i w_ix z_i + w_0x,  w_εx²)

손실 (식 8, 11, 12)
    L = -Σ_l [ E log p(x|z) + Σ_i E log p(z_i|·) - E log q(r^K) ]  +  λ Σ_l L_p
    E log p(z_i|·)  : Jensen 하한 E[-log(1+e^x)] ≥ -log(1+E[e^x]) (식 10, 부록 6.3)
    L_p (triplet)   : max{0, ||q_LS - α_LS|| - ||q_LS - α_LF|| + β}  와 LF 쪽 대칭항.
                      공식 코드 nn.TripletMarginLoss(margin=1, p=2)(q_LS, α_LS, α_LF) 를 따른다
                      (거리는 배치 안 위치들을 벡터로 본 유클리드 거리).

논문 / 공식 코드와 다른 점
--------------------------
1. 단위: 픽셀 → 시정촌 집계 DPM. 지역 가지치기(local pruning)는 쓸 일이 없어 뺐다.
2. α_BD: 논문은 건물 fragility curve 기반 prior. 그 자료가 없어 이벤트 안에서 min-max 정규화한
   log PGV로 대신한다 (논문 그래프에서도 지진동이 BD의 부모다).
3. σ_i²: 공식 코드는 LS/LF에 USGS 불확실도 지도(qvar)를 고정값으로 쓰고 BD는 [1,5]로 묶는다.
   불확실도 지도가 없어 상수(기본 1)로 고정한다. 논문 식대로 학습하면 E log q(r) 항 때문에
   분산이 발산해 사후가 상수로 뭉개진다 (r0_var=None으로 재현 가능).
3-1. 부호 제약: w_α, w_ix, w_LS→BD, w_LQ→BD 는 softplus로 양수. 논문·코드에는 없는 제약이다.
4. 부록 6.3의 w_ε²/2 부호: 공식 코드는 논문과 반대 부호다. ε ~ N(0,1)에서
   E[exp(-w_ε ε)] = exp(+w_ε²/2) 이므로 논문 식이 맞아 논문을 따른다.
5. E log p(x|z): 논문 [3]항 전개 대신 독립 Bernoulli(q)에 대한 정확한 기댓값
   (log x - m)² → (log x - w_0x - Σ w q)² + Σ w² q(1-q) 를 쓴다. 같은 양이다.
6. 부록 6.1 XOR 노드는 "plug-and-play" 선택 요소이고 공식 코드에도 없어 넣지 않았다.
7. 최적화: 공식 코드는 픽셀 수천 개 배치에 RMSprop lr=1e-5, 30 epoch.
   행이 이벤트당 30-80개라 전체 배치 Adam (기본 lr=0.01, 3000 epoch)으로 바꿨다.
   KL 항 β warm-up(0.01→1)은 공식 코드를 따랐다.
"""
import math
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from schema import DTYPE, EPS, INDEX_TO_EVENT

from .data import DPMRows

NODES = ("LS", "LQ", "BD")


class FlowNode(nn.Module):
    """한 노드의 변분분포: N(μ, σ²) 초기분포 + planar flow K층.

    입력 h [N, d]는 부모 노드 신호이고, 출력은 MC 샘플 r^K와 log q(r^K).

    μ와 flow 파라미터 = (부모로부터 amortize한 부분) + (위치별 자유 오프셋).
    위치별 오프셋이 없으면 q가 부모(α)만의 함수가 되어 관측 DPM이 사후에 들어갈 통로가 없다.
    공식 코드는 nn.Linear(batch_size, batch_size)로 배치의 모든 위치를 한 벡터로 받는데,
    배치 순서가 고정이라 사실상 위치별 자유 파라미터로 작동한다. 그것을 명시적으로 쓴 것이다
    (논문 Algorithm 1의 "위치별 사후 {q_i^l} 갱신").
    """

    def __init__(self, in_dim: int, num_flows: int, n_rows: int, r0_var=1.0):
        super().__init__()
        self.K = num_flows
        self.mu = nn.Linear(in_dim, 1, dtype=DTYPE)      # φ(·; W_μ)
        self.mu_local = nn.Parameter(torch.zeros(n_rows, 1, dtype=DTYPE))
        # r^0 분산. None이면 논문 식대로 δ(·; W_σ) → softplus 로 학습한다.
        # 다만 그렇게 두면 손실의 E log q(r) 항이 r 공간 밀도라 분산이 끝없이 커지고
        # 사후가 상수로 뭉개진다 (합성 데이터에서 확인). 공식 코드도 LS/LF는 USGS 불확실도
        # 지도로 고정, BD는 [1, 5]로 묶어 이를 막는다. 불확실도 지도가 없어 상수로 고정한다.
        self.r0_var = r0_var
        if r0_var is None or r0_var == "bounded":
            self.var = nn.Linear(in_dim, 1, dtype=DTYPE)
        # flow 파라미터 amortization. 공식 코드처럼 [-1, 1]로 묶는다.
        self.amor = nn.Linear(in_dim, 3 * num_flows, dtype=DTYPE)
        self.flow_local = nn.Parameter(0.01 * torch.randn(n_rows, 3 * num_flows, dtype=DTYPE))

    def forward(self, h: Tensor, n_samples: int, var_rows: Tensor | None = None):
        """var_rows: [N] 행마다 고정한 r^0 분산 (공식 코드의 USGS 불확실도 지도). 주면 r0_var 대신 쓴다."""
        N = h.shape[0]
        mu = 6.0 * torch.tanh((self.mu(h) + self.mu_local) / 6.0)      # [N,1], 부드럽게 ±6 안쪽
        if var_rows is not None:
            var = var_rows.unsqueeze(1)
        elif self.r0_var is None:
            var = F.softplus(self.var(h)) + 1e-3
        elif self.r0_var == "bounded":
            # 공식 코드의 BD: Linear → Softplus → Hardtanh(1, 5). 부모 입력으로 분산을 정하되 [1, 5]로 묶는다.
            var = torch.clamp(F.softplus(self.var(h)), 1.0, 5.0)
        else:
            var = torch.full_like(mu, float(self.r0_var))
        std = var.sqrt()
        n = torch.randn(N, n_samples, dtype=DTYPE)
        r = mu + std * n                                                 # r^0 [N,M]
        log_q = -0.5 * (n ** 2 + math.log(2 * math.pi)) - torch.log(std)

        if self.K:
            # 공식 코드는 u, w, b를 Hardtanh로 [-1, 1]에 묶는다. 경계에서 u·w = -1이 되면
            # 가역성이 깨지므로 0.99·tanh로 살짝 안쪽에 둔다 (|u·w| < 1 → 1+u·w·h' > 0).
            uwb = (0.99 * torch.tanh(self.amor(h) + self.flow_local)).view(N, 3, self.K)
            for k in range(self.K):
                u, w, b = uwb[:, 0, k:k + 1], uwb[:, 1, k:k + 1], uwb[:, 2, k:k + 1]
                t = torch.tanh(w * r + b)
                log_det = torch.log(1.0 + u * w * (1.0 - t ** 2))         # 식 4
                r = r + u * t                                            # 식 3
                log_q = log_q - log_det                                  # 식 15
        return r, log_q


def _neg_log1p_exp(x: Tensor) -> Tensor:
    """-log(1 + e^x), 수치안정."""
    return -F.softplus(x)


class DisasterNet(nn.Module):
    """한 그룹(기본: 이벤트 1개)의 DisasterNet."""

    def __init__(self, n_rows: int, num_flows=4, r0_var=1.0, use_bd=True, bd_var_learn=False):
        super().__init__()
        # use_bd=False: 건물피해 노드를 뺀 ablation. 파라미터는 똑같이 만들어 두고(난수 순서 유지)
        # 손실·관측식에서만 BD를 빼므로, use_bd=True 결과는 이 옵션이 생기기 전과 같다.
        self.use_bd = use_bd
        self.nodes = NODES if use_bd else ("LS", "LQ")
        p = lambda: nn.Parameter(torch.rand((), dtype=DTYPE))  # 공식 코드처럼 U(0,1) 초기화
        # 노드 CPD 가중치 (weight vertices). 모든 위치에서 공유된다.
        # 이름이 _raw인 것은 softplus로 양수 제약을 건다 (아래 property).
        # 논문·공식 코드는 부호 제약이 없지만, 행이 이벤트당 수십 개뿐이라 제약 없이 두면
        # "prior가 높을수록 덜 발생" / "재해가 있으면 DPM이 작아짐" 으로 부호가 뒤집힌 해에
        # 빠지는 것을 합성 데이터에서 확인했다. 물리적 방향만 고정한다.
        self.w_0 = nn.ParameterDict({k: p() for k in NODES})
        self.w_a_raw = nn.ParameterDict({k: p() for k in NODES})
        self.w_e = nn.ParameterDict({k: p() for k in NODES})
        self.w_LS_BD_raw, self.w_LQ_BD_raw = p(), p()
        # 관측 노드
        self.w_x_raw = nn.ParameterDict({k: p() for k in NODES})
        self.w_0x = p()
        self.w_ex = nn.Parameter(torch.tensor(1.0, dtype=DTYPE))
        # 노드별 변분분포 (부모 입력 차원: LS/LQ는 α 하나, BD는 q_LS·q_LQ·α_BD 셋)
        self.flow = nn.ModuleDict({
            "LS": FlowNode(1, num_flows, n_rows, r0_var),
            "LQ": FlowNode(1, num_flows, n_rows, r0_var),
            "BD": FlowNode(3, num_flows, n_rows, "bounded" if bd_var_learn else r0_var),
        })

    @property
    def w_a(self):
        return {k: F.softplus(v) for k, v in self.w_a_raw.items()}

    @property
    def w_x(self):
        return {k: F.softplus(v) for k, v in self.w_x_raw.items()}

    w_LS_BD = property(lambda self: F.softplus(self.w_LS_BD_raw))
    w_LQ_BD = property(lambda self: F.softplus(self.w_LQ_BD_raw))

    def posteriors(self, a_ls, a_lq, a_bd, n_samples=8, off=None, r0_var=None):
        """q_i = P(z_i=1|x)와 각 노드의 log q(r^K) [N,M].

        off: {"LS"|"LQ"|"BD": [N] bool}. True인 노드는 q를 0으로 고정한다 (원본 가지치기).
        r0_var: {"LS"|"LQ": [N]} 행마다 고정한 r^0 분산 (원본의 USGS 불확실도). 없는 노드는 r0_var 상수.
             행끼리 섞이지 않는 flow라 flow가 끝난 뒤 한 번 곱하면 원본의 단계별 재마스킹과 같다.
        """
        keep = {k: torch.ones_like(a_ls) if off is None else (~off[k]).to(a_ls.dtype) for k in NODES}
        out = {"keep": keep}
        rv = r0_var or {}
        r_ls, lq_ls = self.flow["LS"](a_ls.unsqueeze(1), n_samples, rv.get("LS"))
        r_lq, lq_lq = self.flow["LQ"](a_lq.unsqueeze(1), n_samples, rv.get("LQ"))
        q_ls = torch.sigmoid(r_ls).mean(1) * keep["LS"]                  # 식 6
        q_lq = torch.sigmoid(r_lq).mean(1) * keep["LQ"]
        if self.use_bd:
            h_bd = torch.stack([q_ls, q_lq, a_bd], dim=1)                # BD의 부모
            r_bd, lq_bd = self.flow["BD"](h_bd, n_samples, rv.get("BD"))
            q_bd = torch.sigmoid(r_bd).mean(1) * keep["BD"]
        else:
            q_bd, lq_bd = torch.zeros_like(q_ls), torch.zeros_like(lq_ls)
        out["q"] = {"LS": q_ls, "LQ": q_lq, "BD": q_bd}
        out["log_q"] = {"LS": lq_ls, "LQ": lq_lq, "BD": lq_bd}
        return out

    def log_p_z(self, q, a, log_frac=None, keep=None):
        """Σ_i E log p(z_i | ε_i, 부모) 의 Jensen 하한 [N] (식 9-10, 부록 6.3).

        log_frac: 원본 가지치기(LOCAL 지도·건물 footprint)의 시정촌 단위 대용.
                  {"LS"|"LQ"|"BD": [N]} = log(그 재해가 가능한 땅의 비율). 고정값으로 로짓에 더한다.
                  비율 1 → 영향 없음, 비율 → 0 → 그 노드를 끄는 것과 같다.
        """
        off = (lambda k: log_frac[k]) if log_frac is not None else (lambda k: 0.0)
        # keep: 0으로 고정한 노드는 확률변수가 아니므로 prior 항에서 뺀다
        kp = (lambda k: keep[k]) if keep is not None else (lambda k: 1.0)
        total = 0.0
        w_a = self.w_a
        for k in ("LS", "LQ"):
            s = self.w_0[k] + w_a[k] * a[k] + off(k)
            half = self.w_e[k] ** 2 / 2
            total = total + kp(k) * (q[k] * _neg_log1p_exp(half - s)
                                     + (1 - q[k]) * _neg_log1p_exp(half + s))
        if not self.use_bd:
            return total
        # BD: noisy-OR 부모. 독립 Bernoulli(q_LS), (q_LQ)에 대한 E[exp(∓Σ w z)]
        s = self.w_0["BD"] + w_a["BD"] * a["BD"] + off("BD")
        half = self.w_e["BD"] ** 2 / 2
        e_neg = (1 - q["LS"] + q["LS"] * torch.exp(-self.w_LS_BD)) \
            * (1 - q["LQ"] + q["LQ"] * torch.exp(-self.w_LQ_BD))
        e_pos = (1 - q["LS"] + q["LS"] * torch.exp(self.w_LS_BD)) \
            * (1 - q["LQ"] + q["LQ"] * torch.exp(self.w_LQ_BD))
        total = total + kp("BD") * (q["BD"] * _neg_log1p_exp(half - s + torch.log(e_neg))
                                    + (1 - q["BD"]) * _neg_log1p_exp(half + s + torch.log(e_pos)))
        return total

    def log_p_x(self, log_x, q):
        """E log p(x | z) [N], x ~ LogNormal(Σ w_ix z_i + w_0x, w_εx²) (식 7)."""
        w_x = self.w_x
        m = self.w_0x + sum(w_x[k] * q[k] for k in self.nodes)
        v = sum(w_x[k] ** 2 * q[k] * (1 - q[k]) for k in self.nodes)
        s2 = self.w_ex ** 2 + 1e-6
        return -log_x - 0.5 * torch.log(2 * math.pi * s2) - ((log_x - m) ** 2 + v) / (2 * s2)

    def param_rows(self, group: str):
        vals = {}
        w_a, w_x = self.w_a, self.w_x
        for k in NODES:
            vals.update({f"w_0{k}": self.w_0[k], f"w_a{k}": w_a[k],
                         f"w_e{k}": self.w_e[k], f"w_{k}x": w_x[k]})
        vals.update({"w_LS→BD": self.w_LS_BD, "w_LQ→BD": self.w_LQ_BD,
                     "w_0x": self.w_0x, "w_ex": self.w_ex})
        return [{"group": group, "param": k, "value": float(v.detach())} for k, v in vals.items()]


def triplet_penalty(q, a, margin):
    """공식 코드: TripletMarginLoss(margin, p=2)(q_LS, α_LS, α_LF) + (q_LF, α_LF, α_LS)."""
    tl = nn.TripletMarginLoss(margin=margin, p=2)
    return tl(q["LS"].unsqueeze(0), a["LS"].unsqueeze(0), a["LQ"].unsqueeze(0)) \
        + tl(q["LQ"].unsqueeze(0), a["LQ"].unsqueeze(0), a["LS"].unsqueeze(0))


def prepare_inputs(rows: DPMRows, obs_offset=None):
    """α와 log x를 만든다. LogNormal이라 DPM은 양수여야 한다.

    obs_offset: 관측에 더할 고정 상수. DPM_배경차처럼 음수가 있는 지표에 쓴다.
                DPM은 상관도의 차이라 [-1, 1] 안이므로 1을 더하면 모든 시정촌에 같은 처리로 양수가 된다.
                None이면 0 이하 값이 있을 때만 그 그룹 최솟값만큼 올린다(이전 동작).
    """
    x = rows.y
    if obs_offset is not None:
        x = x + obs_offset
        if (x <= 0).any():
            raise ValueError(f"obs_offset={obs_offset}을 더해도 0 이하인 관측이 있습니다.")
    elif (x <= 0).any():
        shift = float(-x.min()) + 1e-3
        print(f"  [경고] DPM에 0 이하 값이 있어 +{shift:.4f} 만큼 이동 (LogNormal 관측은 x>0 필요)")
        x = x + shift
    lp = torch.log(rows.pgv.clamp_min(EPS))
    span = lp.max() - lp.min()
    a_bd = (lp - lp.min()) / (span if span > 0 else 1.0)
    a = {"LS": rows.pi_ls, "LQ": rows.pi_lq, "BD": a_bd}
    return a, torch.log(x)


@dataclass
class FitResult:
    predictions: pd.DataFrame   # event_idx / muni_code / model_row_idx / dpm / p_ls / p_lq / p_bd
    params: pd.DataFrame
    history: pd.DataFrame


def _fit_group(rows: DPMRows, name, *, epochs, lr, seed, n_samples, num_flows,
               lam_triplet, margin, anneal, r0_var, use_bd, obs_offset, verbose, bd_var_learn=False):
    torch.manual_seed(seed)
    a, log_x = prepare_inputs(rows, obs_offset)
    model = DisasterNet(rows.n, num_flows=num_flows, r0_var=r0_var, use_bd=use_bd, bd_var_learn=bd_var_learn)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    hist = []
    for epoch in range(epochs):
        # KL 항 warm-up (공식 코드 anneal: β = min(0.01 + t/T, 1))
        beta = min(0.01 + epoch / max(epochs, 1), 1.0) if anneal else 1.0
        post = model.posteriors(a["LS"], a["LQ"], a["BD"], n_samples, off=rows.off, r0_var=rows.r0_var)
        q, keep = post["q"], post["keep"]
        rec = model.log_p_x(log_x, q)                                    # [N]
        log_q = sum(post["log_q"][k].mean(1) * keep[k] for k in model.nodes)    # E log q(r^K)
        kl = log_q - model.log_p_z(q, a, rows.log_frac, keep if rows.off is not None else None)
        trip = triplet_penalty(q, a, margin) if lam_triplet > 0 else torch.zeros((), dtype=DTYPE)
        per_row = -rec + beta * kl
        loss = (per_row if rows.weight is None else per_row * rows.weight).sum() + lam_triplet * trip
        opt.zero_grad()
        loss.backward()
        opt.step()

        nelbo = float((-rec + kl).sum().detach())
        hist.append({"group": name, "epoch": epoch, "loss": float(loss.detach()),
                     "neg_elbo": nelbo, "triplet": float(trip.detach()), "beta": beta})
        if verbose and epoch % 500 == 0:
            print(f"  [{name}] {epoch:5d}  -ELBO/row {nelbo / rows.n:.4f}  triplet {float(trip.detach()):.4f}")

    # 최종 사후확률은 MC 샘플을 넉넉히 뽑아 평균한다.
    with torch.no_grad():
        q = model.posteriors(a["LS"], a["LQ"], a["BD"], n_samples=256, off=rows.off, r0_var=rows.r0_var)["q"]
    pred = pd.DataFrame({
        "event_idx": rows.event_idx.tolist(),
        "muni_code": list(rows.municipality_code),
        "model_row_idx": rows.model_row_idx.tolist(),
        "dpm": rows.y.numpy(),
        "p_ls": q["LS"].numpy(), "p_lq": q["LQ"].numpy(), "p_bd": q["BD"].numpy(),
    })
    return pred, model.param_rows(name), hist


def fit_disasternet(rows: DPMRows, *, pooled=False, epochs=3000, lr=0.01, seed=0,
                    n_samples=8, num_flows=4, lam_triplet=1.0, margin=1.0, anneal=True,
                    r0_var=1.0, use_bd=True, obs_offset=None, verbose=True, bd_var_learn=False) -> FitResult:
    """이벤트별(기본, 논문과 같음) 또는 통합(pooled)으로 학습하고 사후확률을 돌려준다."""
    if pooled:
        groups = [("pooled", rows)]
    else:
        groups = [(INDEX_TO_EVENT[e], rows.subset(rows.event_idx == e))
                  for e in sorted(rows.event_idx.unique().tolist())]

    preds, params, hist = [], [], []
    for name, g in groups:
        if verbose:
            print(f"[DisasterNet] {name}: {g.n}행 / flow K={num_flows} / triplet λ={lam_triplet}")
        p, prm, h = _fit_group(g, name, epochs=epochs, lr=lr, seed=seed, n_samples=n_samples,
                               num_flows=num_flows, lam_triplet=lam_triplet, margin=margin,
                               anneal=anneal, r0_var=r0_var, use_bd=use_bd, obs_offset=obs_offset,
                               bd_var_learn=bd_var_learn,
                               verbose=verbose)
        preds.append(p); params.extend(prm); hist.extend(h)

    pred = pd.concat(preds, ignore_index=True).sort_values(["event_idx", "muni_code"])
    return FitResult(pred.reset_index(drop=True), pd.DataFrame(params), pd.DataFrame(hist))
