"""
합성 데이터로 DisasterNet 코드가 돌아가는지, 신호를 되찾는지 확인한다.
원천 xlsx 없이 돌릴 수 있다.

    python -m disasternet.smoke_test

생성모형에서 LS/LQ/BD와 DPM을 직접 뽑은 뒤
  - DisasterNet 사후 AUC가 USGS prior 단독 AUC보다 높아야 하고
  - vi와 exact의 사후확률이 대체로 비슷해야 한다.
"""
import numpy as np
import pandas as pd
import torch

from eval import evaluate
from schema import DTYPE

from .data import DPMRows
from .model import fit_disasternet


def make_synthetic(n_per_event=60, events=(3, 4, 5, 6, 7), seed=0):
    g = torch.Generator().manual_seed(seed)
    parts, truth = [], []
    for e in events:
        n = n_per_event
        pi_ls = torch.rand(n, generator=g, dtype=DTYPE) ** 3 * 0.6 + 1e-3
        pi_lq = torch.rand(n, generator=g, dtype=DTYPE) ** 3 * 0.6 + 1e-3
        pgv = torch.exp(torch.randn(n, generator=g, dtype=DTYPE) * 0.6 + 3)
        ls = torch.bernoulli(pi_ls, generator=g)
        lq = torch.bernoulli(pi_lq, generator=g)
        zp = (torch.log(pgv) - 3) / 0.6
        bd = torch.bernoulli(torch.sigmoid(-1.5 + 1.0 * ls + 1.0 * lq + 0.8 * zp), generator=g)
        y = 0.05 + 0.25 * ls + 0.2 * lq + 0.15 * bd + 0.06 * torch.randn(n, generator=g, dtype=DTYPE)
        parts.append(dict(y=y, pi_ls=pi_ls, pi_lq=pi_lq, pgv=pgv,
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
    prior_pred = pd.DataFrame({"event_idx": rows.event_idx.tolist(),
                               "muni_code": list(rows.municipality_code),
                               "p_ls": rows.pi_ls.numpy(), "p_lq": rows.pi_lq.numpy()})
    res_prior = evaluate(gt, prior_pred)
    out = {}
    for inf in ("vi", "exact"):
        fit = fit_disasternet(rows, inference=inf, epochs=1500, verbose=False)
        out[inf] = fit.predictions
        r = evaluate(gt, fit.predictions[["event_idx", "muni_code", "p_ls", "p_lq"]])
        print(f"[{inf:5s}] AUC_LS {r.auc_ls:.3f} (prior {res_prior.auc_ls:.3f})  "
              f"AUC_LQ {r.auc_lq:.3f} (prior {res_prior.auc_lq:.3f})  "
              f"MSE_LS {r.mse_ls:.4f} (prior {res_prior.mse_ls:.4f})  "
              f"MSE_LQ {r.mse_lq:.4f} (prior {res_prior.mse_lq:.4f})")
        assert r.auc_ls > res_prior.auc_ls and r.auc_lq > res_prior.auc_lq, "사후가 prior를 못 넘음"

    diff = np.abs(out["vi"][["p_ls", "p_lq"]].to_numpy() - out["exact"][["p_ls", "p_lq"]].to_numpy())
    print(f"vi vs exact 사후확률 차이: 평균 {diff.mean():.3f} / 90% {np.quantile(diff, .9):.3f}")
    print("OK")


if __name__ == "__main__":
    main()
