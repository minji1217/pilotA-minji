from pathlib import Path

import pandas as pd
import torch

from likelihood import DamageLikelihood
from regression import DamageRegression
from prior import AreaPrior, Prior, prior_log_w
from marginal import marginalize
from infer import infer
from loader import load_eval_ground_truth, load_pilot_a_batch
from eval import evaluate
from schema import INDEX_TO_EVENT, EvalGroundTruthBatch, PilotABatch

from reporting import dump_params, save_eval, save_loss_history

STATS_PATH = "raw/재난프로젝트_시정촌별_통계데이터.xlsx"
USGS_PATH = "raw/재난프로젝트_시정촌별_USGS.xlsx"
GT_PATH = "validation/LS_LF 데이터자료.xlsx"

# 후속실험 3: 이 브랜치의 prior 조건. followup3-area-avg-<모드> 브랜치마다 이 값만 다르다.
# z = a·log p̄ + b + c·log k (prior.AreaPrior). None이면 기존 Prior(a·logit(pi)+b)를 쓴다.
DEFAULT_AREA_MODE = "lq-c062"


def save_predictions(batch, p_ls, p_lq, path="outputs/predictions.csv", extra=None):
    """전체 행의 사후확률을 저장한다. extra={컬럼명: 값}이면 옆에 붙인다."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "event_idx": batch.event_idx.tolist(),
        "event": [INDEX_TO_EVENT[i] for i in batch.event_idx.tolist()],
        "muni_code": list(batch.municipality_code),
        "p_ls": p_ls.detach().numpy(),
        "p_lq": p_lq.detach().numpy(),
    })
    for name, values in (extra or {}).items():
        df[name] = values
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"저장: {path}  ({len(df)}행)")
    return df


def to_eval_pred(batch: PilotABatch, p_ls, p_lq, eval_gt: EvalGroundTruthBatch, pri=None):
    """eval.py에 넘길 훗카이도 행만 뽑는다.

    loader가 만든 model_row_idx가 이미 GT와 같은 순서로 정렬돼 있으므로
    event_idx로 다시 마스킹하지 않고 그대로 쓴다.

    prior_ls / prior_lq는 보정 전 USGS prior 그대로다.
    사후가 사전보다 나아졌는지 보려면 같은 행에서 잰 prior AUC가 있어야 한다.
    """
    idx = eval_gt.model_row_idx

    out = pd.DataFrame({
        # 같은 시정촌코드가 여러 이벤트에 나오므로 event_idx까지 있어야 join이 성립한다.
        "event_idx": eval_gt.event_idx.tolist(),
        "muni_code": list(eval_gt.municipality_code),
        "p_ls": p_ls[idx].detach().numpy(),
        "p_lq": p_lq[idx].detach().numpy(),
        "prior_ls": batch.pi_ls[idx].detach().numpy(),
        "prior_lq": batch.pi_lq[idx].detach().numpy(),
    })

    # 모델 자신의 prior 점수. 면적 항 조건이면 z = a·log p̄ + b + c·log k 라서
    # 원값 prior와 순위가 다르다. 면적 항이 없는 조건은 인자 두 개짜리 z를 쓴다.
    if pri is not None and hasattr(pri, "z"):
        with torch.no_grad():
            if hasattr(batch, "log_k_ls"):
                z_ls, z_lq = pri.z(batch.pi_ls, batch.pi_lq, batch.log_k_ls, batch.log_k_lq)
            else:
                z_ls, z_lq = pri.z(batch.pi_ls, batch.pi_lq)
        out["zprior_ls"] = z_ls[idx].detach().numpy()
        out["zprior_lq"] = z_lq[idx].detach().numpy()

    return out

    # 모델 자신의 prior 점수. 면적 항 조건이면 z = a·log p̄ + b + c·log k 이고,
    # 원값 prior와 순위가 다르다. 둘을 같이 봐야 "면적 항이 올린 것"과
    # "피해 데이터가 올린 것"을 가를 수 있다.
    if pri is not None and hasattr(pri, "z"):
        with torch.no_grad():
            z_ls, z_lq = pri.z(batch.pi_ls, batch.pi_lq, batch.log_k_ls, batch.log_k_lq)
        out["zprior_ls"] = z_ls[idx].detach().numpy()
        out["zprior_lq"] = z_lq[idx].detach().numpy()

    return out


def to_eval_gt(eval_gt: EvalGroundTruthBatch):
    """EvalGroundTruthBatch를 eval.py가 쓰는 컬럼명으로 바꾼다."""
    return pd.DataFrame({
        "event_idx": eval_gt.event_idx.tolist(),
        "event": [INDEX_TO_EVENT[i] for i in eval_gt.event_idx.tolist()],
        "muni_code": list(eval_gt.municipality_code),
        "ls_true": eval_gt.gt_ls.tolist(),
        "lq_true": eval_gt.gt_lq.tolist(),
        # LS와 LQ는 원본 NA 처리 규칙이 달라 평가 가능한 행이 서로 다르다.
        "ls_eval_mask": eval_gt.ls_eval_mask.tolist(),
        "lq_eval_mask": eval_gt.lq_eval_mask.tolist(),
    })


def train(batch, *,seed=0,epochs=3000,lr=0.02,lam_gamma=0.0,prior_mode="free",b_bound=2.0,
          b_min=None,b_max=None,area_mode=None,c_min=0.0,c_max=2.0):
    """
    lam_gamma  : gamma에 거는 L2 정규화 계수. loss에 lam_gamma * sum(gamma^2)를 더한다.
                 gamma에 N(0, 1/(2*lam_gamma)) prior를 준 MAP 추정과 같다.
                 0이면 정규화 없음(= 기존 MLE).
    prior_mode : Prior의 a, b 제약 방식. "free" / "fixed" / "bounded"
    b_bound    : prior_mode="bounded"일 때 b의 범위. b in [-b_bound, +b_bound]
    b_min/b_max: 비대칭 b 범위가 필요할 때 b_bound 대신 쓴다.
                 후속실험 2의 b in [-2, 4]가 이 경우다. 둘 다 줘야 한다.
    area_mode  : 후속실험 3. AreaPrior 모드(prior.AreaPrior.MODES). 주면 prior_mode·b_bound는 쓰지 않는다.
                 b 범위는 b_min/b_max(기본 [-2, 4]), c 범위는 c_min/c_max(기본 [0, 2])다.
    """
    torch.manual_seed(seed)

    like=DamageLikelihood()
    reg=DamageRegression()
    if area_mode is None:
        pri=Prior(mode=prior_mode,b_bound=b_bound,b_min=b_min,b_max=b_max)
    else:
        # b를 학습하는 모드는 b 범위를 넓게 준다.
        # 중심화를 하면 b가 log k 평균(+8.0 LS / +6.7 LQ)만큼 이동한 자리에서 최적이 되고,
        # 중심화를 안 해도 기존 [-2, 4]로는 좁았다(bounded의 b_LQ가 -1.947로 하한에 붙었다).
        # 넓게 두고 어디서 멈추는지 보는 편이 낫다.
        _base = area_mode[:-4] if area_mode.endswith("-ctr") else area_mode
        wide = _base in ("tied", "bounded", "free", "b-only")
        default_b = (-20.0, 20.0) if wide else (-2.0, 4.0)
        pri=AreaPrior(mode=area_mode,
                      b_min=default_b[0] if b_min is None else b_min,
                      b_max=default_b[1] if b_max is None else b_max,
                      c_min=c_min,c_max=c_max)
        # 중심화 모드면 batch 전체의 log k 평균을 한 번 재 둔다.
        pri.fit_center(batch.log_k_ls, batch.log_k_lq)

    history = []
    reg.initialize_from_batch(batch)
    opt=torch.optim.Adam(list(like.parameters())+list(reg.parameters())+list(pri.parameters()),lr=lr)
    #total param??
    print(f"총 학습될 파라미터 : {sum(p.numel() for p in opt.param_groups[0]['params'])}개")
    

    for epoch in range(epochs):
        out_r=reg(batch)
        # -> 이제 람다들을 만들었으니 이걸... 어떻게 하더라 곱해서 L 하나 내뱉는걸로
        out_l=like(batch,out_r.mu)
        w_batch=prior_log_w(pri,batch)
        
        _,log_Py=marginalize(w_batch,out_l.log_L)
        nll=-log_Py.sum()

        # gamma는 softplus를 거친 실제 값에 건다. 변환 전 raw에 걸면 식의 gamma가 아니다.
        if lam_gamma > 0:
            penalty = lam_gamma * (
                (reg.gamma_ls ** 2).sum() + (reg.gamma_lq ** 2).sum()
            )
        else:
            penalty = torch.zeros((), dtype=nll.dtype)
        loss = nll + penalty

        opt.zero_grad() #기울기 누적 초기화
        loss.backward()
        opt.step()

        nll_val = float(nll.item())
        history.append({
            "epoch": epoch,
            "nll_total": nll_val,
            "nll_per_row": nll_val / batch.batch_size,
            "penalty": float(penalty.item()),
            "loss_total": float(loss.item()),
        })

        if epoch%100==0:
            print(f"{epoch}번째 학습=> loss :{loss}")

    return reg,like,pri,pd.DataFrame(history)



if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Pilot A 학습 1회 실행. 에폭별 loss와 학습 파라미터를 CSV로 저장한다."
    )
    ap.add_argument("--prior-mode", default="free", choices=["free", "fixed", "bounded"],
                    help="prior의 a, b 제약. free=제약없음(기본) / fixed=a1,b0 고정 / bounded=범위제한")
    ap.add_argument("--lam-gamma", type=float, default=0.0,
                    help="gamma L2 정규화 계수. 0이면 정규화 없음(기본)")
    ap.add_argument("--b-bound", type=float, default=2.0,
                    help="prior-mode=bounded일 때 b의 범위. b in [-b_bound, +b_bound]")
    ap.add_argument("--b-min", type=float, default=None,
                    help="비대칭 b 범위의 하한. --b-max와 함께 주면 --b-bound를 대신한다")
    ap.add_argument("--b-max", type=float, default=None,
                    help="비대칭 b 범위의 상한. 후속실험 2는 --b-min -2 --b-max 4")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--area-mode", default=DEFAULT_AREA_MODE or "none",
                    choices=[*AreaPrior.MODES, "none"],
                    help="후속실험 3 면적 prior 모드. none이면 기존 prior(--prior-mode)를 쓴다")
    ap.add_argument("--c-min", type=float, default=0.0, help="area-mode=free일 때 c의 하한")
    ap.add_argument("--c-max", type=float, default=2.0, help="area-mode=free일 때 c의 상한")
    ap.add_argument("--tag", default="",
                    help="출력 파일명 뒤에 붙일 꼬리표. 여러 설정을 비교할 때 서로 덮이지 않는다")
    args = ap.parse_args()

    area_mode = None if args.area_mode == "none" else args.area_mode
    if area_mode is not None and not args.tag:
        # 조건마다 결과 파일이 서로 덮이지 않게 한다.
        args.tag = f"area-{area_mode}"
    sfx = f"_{args.tag}" if args.tag else ""

    # 학습용 batch(382행)를 만들고, 선배가 만든 API로 평가용 GT를 정렬해 받는다.
    # GT는 eval 전용이라 train()에는 넘기지 않는다.
    batch = load_pilot_a_batch(STATS_PATH, USGS_PATH)
    eval_gt = load_eval_ground_truth(GT_PATH, batch)
    n_ev = int(eval_gt.event_idx.unique().numel())
    print(f"batch {batch.batch_size}행 / 평가 GT {eval_gt.batch_size}행 / 이벤트 {n_ev}개")
    if args.b_min is not None or args.b_max is not None:
        b_desc = f"b∈[{args.b_min}, {args.b_max}]"
    else:
        b_desc = f"b∈[-{args.b_bound}, {args.b_bound}]"
    if area_mode is None:
        print(f"설정: prior_mode={args.prior_mode} / lam_gamma={args.lam_gamma}"
              + (f" / {b_desc}" if args.prior_mode == "bounded" else ""))
    else:
        print(f"설정: area_mode={area_mode} (z = a·log p̄ + b + c·log k) / lam_gamma={args.lam_gamma}")

    reg, like, pri, hist = train(
        batch=batch, seed=args.seed, epochs=args.epochs, lr=args.lr,
        lam_gamma=args.lam_gamma, prior_mode=args.prior_mode, b_bound=args.b_bound,
        b_min=args.b_min, b_max=args.b_max,
        area_mode=area_mode, c_min=args.c_min, c_max=args.c_max,
    )
    save_loss_history(hist, tag=args.tag)
    dump_params(reg, like, pri, path=f"outputs/params{sfx}.csv")

    with torch.no_grad():                        # ← grad 안 만듦
        out_r = reg(batch)
        out_l = like(batch, out_r.mu)
        log_w = prior_log_w(pri, batch)
        log_joint, log_Py = marginalize(log_w, out_l.log_L)
        p_ls, p_lq = infer(log_joint, log_Py)

    gt = to_eval_gt(eval_gt)
    pred = to_eval_pred(batch, p_ls, p_lq, eval_gt)
    result = evaluate(gt, pred)

    # 후속실험 3: 면적 항을 넣은 '자기 prior' 단독 성능도 같은 행에서 잰다.
    # 사후가 이 값을 넘어야 모델(피해 우도)이 prior 위에 뭔가를 더한 것이다.
    own, extra = None, None
    if isinstance(pri, AreaPrior):
        with torch.no_grad():
            z_ls, z_lq = pri.z(batch.pi_ls, batch.pi_lq, batch.log_k_ls, batch.log_k_lq)
            own_ls, own_lq = torch.sigmoid(z_ls), torch.sigmoid(z_lq)
        own = evaluate(gt, to_eval_pred(batch, own_ls, own_lq, eval_gt))
        extra = {"log_k_ls": batch.log_k_ls.numpy(), "log_k_lq": batch.log_k_lq.numpy(),
                 "own_prior_ls": own_ls.numpy(), "own_prior_lq": own_lq.numpy()}

    save_predictions(batch, p_ls, p_lq, path=f"outputs/predictions{sfx}.csv", extra=extra)
    save_eval(result, gt, tag=args.tag, own_prior=own)

    print(
        f"MSE_LS {result.mse_ls:.4f} (n={result.n_ls}) / "
        f"MSE_LQ {result.mse_lq:.4f} (n={result.n_lq}) / 전체 {result.n}행"
    )
    print(
        f"AUC_LS {result.auc_ls:.4f} (prior {result.auc_prior_ls:.4f}) / "
        f"AUC_LQ {result.auc_lq:.4f} (prior {result.auc_prior_lq:.4f})"
    )
    print(
        f"이벤트내 가중평균 AUC_LS {result.auc_ls_wavg:.4f} "
        f"(prior {result.auc_prior_ls_wavg:.4f}) / "
        f"AUC_LQ {result.auc_lq_wavg:.4f} "
        f"(prior {result.auc_prior_lq_wavg:.4f})"
    )
    if own is not None:
        print(
            f"자기 prior(면적 항 포함) 단독 AUC_LS 가중 {own.auc_ls_wavg:.4f} / 전체 {own.auc_ls:.4f} · "
            f"AUC_LQ 가중 {own.auc_lq_wavg:.4f} / 전체 {own.auc_lq:.4f}"
        )
        a, b, c = pri.a_value.detach(), pri.b_value.detach(), pri.c_value.detach()
        print(f"prior 파라미터 LS a={float(a[0]):.4f} b={float(b[0]):+.4f} c={float(c[0]):.4f} / "
              f"LQ a={float(a[1]):.4f} b={float(b[1]):+.4f} c={float(c[1]):.4f}")
    print()
    print("이벤트별:")
    print(result.per_event.round(4).to_string(index=False))
