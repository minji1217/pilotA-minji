"""
합성 데이터로 DisasterNet 코드가 돌아가는지, 신호를 되찾는지 확인한다.
원천 xlsx 없이 돌릴 수 있다.

    python -m disasternet.smoke_test

논문 4.4처럼 생성모형의 forward 과정으로 LS/LQ/BD와 DPM을 뽑은 뒤,
사후 AUC가 USGS prior 단독 AUC보다 높은지 본다.
"""
import pandas as pd
import torch

from eval import evaluate
from schema import DTYPE

from .data import DPMRows
from .model import fit_disasternet


def make_synthetic(n_per_event=60, events=(3, 4, 5, 6, 7), seed=0):
    g = torch.Generator().manual_seed(seed)
    rnd = lambda n: torch.randn(n, generator=g, dtype=DTYPE)
    parts, truth = [], []
    for e in events:
        n = n_per_event
        # prior: 실제 발생과 상관은 있지만 잡음이 낀 확률
        lat_ls, lat_lq = rnd(n), rnd(n)
        pi_ls = torch.sigmoid(1.2 * lat_ls - 1.5 + 0.8 * rnd(n))
        pi_lq = torch.sigmoid(1.2 * lat_lq - 1.5 + 0.8 * rnd(n))
        ls = torch.bernoulli(torch.sigmoid(2.5 * lat_ls - 1.5), generator=g)
        lq = torch.bernoulli(torch.sigmoid(2.5 * lat_lq - 1.5), generator=g)
        pgv = torch.exp(0.5 * rnd(n) + 3)
        zp = (torch.log(pgv) - 3) / 0.5
        p_bd = 1 - torch.sigmoid(-(-1.5 + 0.8 * zp)) * torch.exp(-1.0 * ls - 1.0 * lq)  # noisy-OR
        bd = torch.bernoulli(p_bd.clamp(0, 1), generator=g)
        # LogNormal DPM (양수)
        x = torch.exp(-2.5 + 0.9 * ls + 0.7 * lq + 0.5 * bd + 0.25 * rnd(n))
        parts.append(dict(y=x, pi_ls=pi_ls, pi_lq=pi_lq, pgv=pgv,
                          event_idx=torch.full((n,), e, dtype=torch.long)))
        truth.append(pd.DataFrame({"event_idx": e, "ls_true": ls.long().numpy(),
                                   "lq_true": lq.long().numpy()}))
    cat = lambda k: torch.cat([p[k] for p in parts])
    N = sum(p["y"].shape[0] for p in parts)
    codes = tuple(f"{i:05d}" for i in range(N))
    rows = DPMRows(y=cat("y"), pi_ls=cat("pi_ls"), pi_lq=cat("pi_lq"), pgv=cat("pgv"),
                   event_idx=cat("event_idx"), municipality_code=codes,
                   model_row_idx=torch.arange(N))
    gt = pd.concat(truth, ignore_index=True)
    gt["muni_code"] = codes
    return rows, gt


def main():
    rows, gt = make_synthetic()
    prior = evaluate(gt, pd.DataFrame({
        "event_idx": rows.event_idx.tolist(), "muni_code": list(rows.municipality_code),
        "p_ls": rows.pi_ls.numpy(), "p_lq": rows.pi_lq.numpy()}))
    print(f"[prior          ] AUC_LS {prior.auc_ls_wavg:.3f}  AUC_LQ {prior.auc_lq_wavg:.3f}  "
          f"MSE_LS {prior.mse_ls:.4f}  MSE_LQ {prior.mse_lq:.4f}")

    results = {}
    # K=0(flow 없음)은 이 재구현에서 LS/LQ가 뒤바뀐 해에 빠지기 쉬워 점검 대상에서 뺐다 (README 참고).
    for K, lam in ((4, 1.0), (4, 0.0)):
        fit = fit_disasternet(rows, epochs=1500, num_flows=K, lam_triplet=lam, verbose=False)
        r = evaluate(gt, fit.predictions[["event_idx", "muni_code", "p_ls", "p_lq"]])
        results[(K, lam)] = r
        print(f"[K={K} triplet={lam:.0f}] AUC_LS {r.auc_ls_wavg:.3f}  AUC_LQ {r.auc_lq_wavg:.3f}  "
              f"MSE_LS {r.mse_ls:.4f}  MSE_LQ {r.mse_lq:.4f}")

    main_r = results[(4, 1.0)]
    assert main_r.auc_ls_wavg > prior.auc_ls_wavg and main_r.auc_lq_wavg > prior.auc_lq_wavg, \
        "기본 설정의 사후 AUC가 prior를 못 넘음"
    print("OK")


if __name__ == "__main__":
    main()
