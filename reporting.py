"""
Pilot A - reporting.py

교수님 공유용 산출물 두 가지를 만든다.

1) dump_params()       -> outputs/params.csv
   학습된 파라미터 + 기준 이벤트 1개(고정 0)를 저장한다.

   컬럼:
     module      reg / lik / pri
     group       설계식에 등장하는 이름 (alpha_c, alpha_e, beta_c, delta_c, ...)
     param       코드상의 nn.Parameter 이름
     idx         채널 파라미터는 채널번호 0~5,
                 alpha_e는 EVENTS 번호 0~7, a/b는 0=LS 1=LQ.
                 ※ alpha_event_free는 코드상 길이가 NUM_EVENTS-1이지만 여기서는
                   벡터 위치가 아니라 이벤트 번호를 적는다.
                   기준 이벤트가 빠져 번호가 건너뛸 수 있다.
     label       사람이 읽는 이름
     transform   none / softplus / fixed(reference)
     raw_value   옵티마이저가 실제로 들고 있는 값 (변환 전)
     value       설계식에 등장하는 실제 값 (변환 후)

   transform=none이면 raw_value와 value가 같다.
   transform=fixed(reference)인 한 행만 학습 대상이 아니며, 나머지 54행이 학습 파라미터다.

2) save_loss_history() -> outputs/loss_history.csv, outputs/loss_curve.png
   에폭별 loss를 전부 저장하고 선형/로그 두 패널로 그린다.

3) save_eval()         -> outputs/eval_summary.csv, eval_detail.csv, eval_per_event.csv
   eval.evaluate()가 낸 결과를 CSV 3종으로 떨어뜨린다.
   원래 train.py의 __main__ 근처에 있었는데, 산출물 저장은 전부 이 파일에
   모으는 편이 브랜치를 오갈 때 덜 헷갈려서 옮겼다.

4) merge_runs()        -> outputs/comparison_auc.csv, comparison_params.csv
   브랜치별로 따로 돌린 실행 결과를 한 표로 합친다.
   조건이 여러 개일 때 파일을 번갈아 열지 않고 나란히 비교하려는 용도다.

모델 계산에는 관여하지 않는다. 학습이 끝난 뒤에만 호출한다.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # 창을 띄우지 않고 파일로만 저장한다
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F

from schema import (
    CHANNELS,
    DAMAGE_COLUMN_MAP,
    INDEX_TO_EVENT,
    NUM_EVENTS,
)

# 채널 라벨은 schema의 순서를 그대로 따른다.
# 여기서 리스트를 직접 적으면 schema가 바뀔 때 라벨만 조용히 어긋난다.
CHANNEL_LABELS = [DAMAGE_COLUMN_MAP[c] for c in CHANNELS]
CHANNEL_IDX = list(range(len(CHANNEL_LABELS)))

# 이벤트가 9개가 되면서 alpha_event_free가 7 -> 8이 되어 41 -> 42가 되었고,
# 후속실험 1에서 delta_c 6 + eta_c 6이 추가되어 42 -> 54가 되었다.
# 이 브랜치는 eta_c 6개를 다시 뺐으므로 48개다.
# prior_mode="fixed"이면 a, b 4개가 빠져 44개다.
# 실제 검증은 아래에서 optimizer가 들고 있는 수와 직접 대조한다.
EXPECTED_NUM_PARAMS = 48

# 설계식에 나오는 순서대로 정렬하기 위한 기준
GROUP_ORDER = ["alpha_c", "alpha_e", "beta_c", "delta_c",
               "gamma_LS", "gamma_LQ", "phi_c", "a", "b", "c"]


def dump_params(reg, like, pri, path="outputs/params.csv"):
    """학습된 파라미터를 CSV로 저장한다."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    ref = reg.reference_event_idx
    # alpha_event_free는 기준 이벤트를 제외한 NUM_EVENTS-1개다.
    # idx에는 벡터 위치가 아니라 EVENTS 번호를 적어야 기준 행과 같은 자로 읽힌다.
    event_free_idx = [i for i in range(NUM_EVENTS) if i != ref]
    event_free_labels = [INDEX_TO_EVENT[i] for i in event_free_idx]

    spec = [
        ("reg", "alpha_channel",           "alpha_c",  CHANNEL_IDX,    CHANNEL_LABELS,     "none"),
        ("reg", "alpha_event_free",        "alpha_e",  event_free_idx, event_free_labels,  "none"),
        ("reg", "beta_pgv",                "beta_c",   CHANNEL_IDX,    CHANNEL_LABELS,     "none"),
        # 후속실험 1에서 추가된 취약성 공변량. gamma와 달리 부호 제약이 없어 transform=none이다.
        # eta_mtn(산지)은 이 브랜치에서 제거했다.
        ("reg", "delta_wood",              "delta_c",  CHANNEL_IDX,    CHANNEL_LABELS,     "none"),
        ("reg", "_gamma_ls_unconstrained", "gamma_LS", CHANNEL_IDX,    CHANNEL_LABELS,     "softplus"),
        ("reg", "_gamma_lq_unconstrained", "gamma_LQ", CHANNEL_IDX,    CHANNEL_LABELS,     "softplus"),
        ("lik", "_phi_unconstrained",      "phi_c",    CHANNEL_IDX,    CHANNEL_LABELS,     "softplus"),
    ]

    # Prior는 mode에 따라 파라미터 이름과 변환이 달라진다.
    mode = getattr(pri, "mode", "free")
    # AreaPrior의 "-ctr" 접미사는 중심화 여부일 뿐 c 구조는 앞부분이 정한다.
    base_mode = getattr(pri, "base_mode", mode)
    # 후속실험 3의 AreaPrior는 c가 있고 모드 이름이 기존 Prior와 겹치므로 먼저 가른다.
    is_area = hasattr(pri, "c_value")
    if is_area:
        if pri.learn_ab:
            spec += [
                ("pri", "_a_raw", "a", [0, 1], ["LS", "LQ"], f"bounded[{pri.A_MIN},{pri.A_MAX}]"),
                ("pri", "_b_raw", "b", [0, 1], ["LS", "LQ"], f"bounded[{pri.B_MIN},{pri.B_MAX}]"),
            ]
        elif getattr(pri, "learn_b_only", False):
            # a와 c는 고정이고 b만 학습한다.
            spec += [
                ("pri", "_b_raw", "b", [0, 1], ["LS", "LQ"], f"bounded[{pri.B_MIN},{pri.B_MAX}]"),
            ]
        if base_mode == "free":
            spec += [
                ("pri", "_c_raw", "c", [0, 1], ["LS", "LQ"], f"bounded[{pri.C_MIN},{pri.C_MAX}]"),
            ]
        elif getattr(pri, "learn_c_lq", False):
            # c_LS는 고정이라 아래 고정행으로 가고, 학습하는 것은 c_LQ 하나뿐이다.
            spec += [
                ("pri", "_c_lq_raw", "c", [1], ["LQ"], f"bounded[{pri.C_MIN},{pri.C_MAX}]"),
            ]
    elif mode == "free":
        spec += [
            ("pri", "a", "a", [0, 1], ["LS", "LQ"], "none"),
            ("pri", "b", "b", [0, 1], ["LS", "LQ"], "none"),
        ]
    elif mode == "bounded":
        spec += [
            ("pri", "_a_raw", "a", [0, 1], ["LS", "LQ"], f"bounded[{pri.A_MIN},{pri.A_MAX}]"),
            ("pri", "_b_raw", "b", [0, 1], ["LS", "LQ"], f"bounded[{pri.B_MIN},{pri.B_MAX}]"),
        ]
    # 후속실험 5: z_LS의 kappa * z_mtn. 부호 제약이 없다.
    if getattr(pri, "mtn_prior", False):
        spec += [("pri", "kappa", "kappa", [0], ["LS"], "none")]
    # mode == "fixed"이면 a=1, b=0은 학습 파라미터가 아니므로 아래에서 고정행으로 넣는다.
    mods = {"reg": reg, "lik": like, "pri": pri}
    named = {k: dict(m.named_parameters()) for k, m in mods.items()}

    rows, total = [], 0
    for mod_key, pname, group, idxs, labels, transform in spec:
        if pname not in named[mod_key]:
            raise KeyError(
                f"{mod_key}에 '{pname}' 파라미터가 없습니다. "
                f"실제 이름: {sorted(named[mod_key])}"
            )

        raw = named[mod_key][pname].detach()
        if transform == "softplus":
            val = F.softplus(raw)
        elif transform.startswith("bounded"):
            # prior의 a, b는 sigmoid 재파라미터화라 raw와 실제 값이 다르다.
            c_val = getattr(pri, "c_value", None)
            val = {"_a_raw": pri.a_value, "_b_raw": pri.b_value,
                   "_c_raw": c_val,
                   # c_LQ만 학습하는 모드는 값도 LQ 한 칸만 꺼낸다.
                   "_c_lq_raw": None if c_val is None else c_val[1:]}[pname].detach()
        else:
            val = raw

        raw_np, val_np = raw.cpu().numpy(), val.cpu().numpy()
        if not (raw_np.size == len(labels) == len(idxs)):
            raise ValueError(
                f"{pname}: 값 {raw_np.size}개 / 라벨 {len(labels)}개 / "
                f"idx {len(idxs)}개가 서로 다릅니다."
            )
        total += raw_np.size

        for i, lab, r, v in zip(idxs, labels, raw_np, val_np):
            rows.append({
                "module": mod_key, "group": group, "param": pname,
                "idx": int(i), "label": lab, "transform": transform,
                "raw_value": float(r), "value": float(v),
            })

    # 기준 이벤트는 학습 파라미터가 아니라 식별을 위해 0으로 고정한 값이다.
    # alpha_e 그룹 안에 같은 idx 체계로 넣어야 8개 이벤트가 한자리에서 읽힌다.
    rows.append({
        "module": "reg", "group": "alpha_e", "param": "alpha_event_free",
        "idx": ref, "label": INDEX_TO_EVENT[ref], "transform": "fixed(reference)",
        "raw_value": 0.0, "value": 0.0,
    })

    if is_area:
        # 학습하지 않는 a, b, c도 어떤 값이 쓰였는지 남겨야 재현이 된다.
        fixed = []
        if getattr(pri, "learn_b_only", False):
            # b는 위에서 학습행으로 들어갔다. a만 고정행이다.
            fixed += [("a", pri.a_value, f"fixed(area={mode})", ["LS", "LQ"])]
        elif not pri.learn_ab:
            fixed += [("a", pri.a_value, f"fixed(area={mode})", ["LS", "LQ"]),
                      ("b", pri.b_value, f"fixed(area={mode})", ["LS", "LQ"])]
        if base_mode == "tied":
            fixed.append(("c", pri.c_value, "tied(c=a)", ["LS", "LQ"]))
        elif getattr(pri, "learn_c_lq", False):
            # c_LQ는 학습 파라미터라 위에서 기록됐다. 여기서는 고정된 c_LS만 남긴다.
            fixed.append(("c", pri.c_value, f"fixed(area={mode})", ["LS"]))
        elif base_mode != "free":
            fixed.append(("c", pri.c_value, f"fixed(area={mode})", ["LS", "LQ"]))
        # 중심화 모드는 실제로 뺀 값을 남겨야 재현이 된다.
        center = getattr(pri, "log_k_center", None)
        if center is not None and float(center.abs().sum()) > 0:
            fixed.append(("log_k_center", center, f"fixed(area={mode})", ["LS", "LQ"]))

        for group, vals, transform, labels in fixed:
            for i, lab in enumerate(labels):
                v = float(vals.detach()[i])
                rows.append({
                    "module": "pri", "group": group, "param": group,
                    "idx": i, "label": lab, "transform": transform,
                    "raw_value": v, "value": v,
                })
    elif mode == "fixed":
        # 학습 대상이 아니지만 어떤 값이 쓰였는지는 남겨야 재현이 된다.
        for i, lab in enumerate(["LS", "LQ"]):
            for group, value in (("a", 1.0), ("b", 0.0)):
                rows.append({
                    "module": "pri", "group": group, "param": group,
                    "idx": i, "label": lab, "transform": "fixed(mode=fixed)",
                    "raw_value": value, "value": value,
                })

    # 실제 optimizer가 들고 가는 파라미터 수와 대조한다.
    # 상수로 박아두면 prior mode를 바꿀 때마다 어긋난다.
    actual = sum(p.numel() for m in mods.values() for p in m.parameters())
    if total != actual:
        raise ValueError(
            f"열거한 파라미터 수({total})가 실제 학습 파라미터 수({actual})와 다릅니다."
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["group", "idx"],
        key=lambda s: s.map({g: i for i, g in enumerate(GROUP_ORDER)})
        if s.name == "group" else s,
    ).reset_index(drop=True)

    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"저장: {path}  (학습 {total}개 / 고정 {len(df) - total}개 / 합 {len(df)}행)")
    return df


def save_loss_history(hist_df, dir_="outputs", tag=""):
    """에폭별 loss를 CSV와 PNG로 저장한다.

    hist_df 컬럼: epoch, nll_total, nll_per_row, penalty, loss_total
        nll_total  : 데이터 항만. 정규화 계수가 달라도 이 값끼리는 비교할 수 있다.
        penalty    : lam_gamma * sum(gamma^2)
        loss_total : nll_total + penalty. 실제로 미분되는 값.

    tag가 있으면 파일명 뒤에 붙어 여러 설정의 결과가 서로 덮이지 않는다.
    """
    Path(dir_).mkdir(parents=True, exist_ok=True)

    sfx = f"_{tag}" if tag else ""
    csv_path = f"{dir_}/loss_history{sfx}.csv"
    png_path = f"{dir_}/loss_curve{sfx}.png"
    hist_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].plot(hist_df["epoch"], hist_df["nll_total"])
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("NLL (total)")
    ax[0].set_title("Training loss")
    ax[0].grid(alpha=0.3)

    # 초반 급락 때문에 선형 축에서는 후반이 바닥에 붙어 수렴 여부가 안 보인다.
    ax[1].plot(hist_df["epoch"], hist_df["nll_total"])
    ax[1].set_yscale("log")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("NLL (log scale)")
    ax[1].set_title("Training loss (log scale)")
    ax[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    first, last = hist_df["nll_total"].iloc[0], hist_df["nll_total"].iloc[-1]
    print(f"저장: {csv_path}, {png_path}")
    print(f"  loss {first:,.1f} -> {last:,.1f}  ({len(hist_df)} epochs)")
    return hist_df


def save_eval(result, gt_df, dir_="outputs", tag="", own_prior=None):
    """eval.evaluate() 결과를 CSV 3종으로 저장한다.

    원래 train.py에 있던 함수를 그대로 옮긴 것이다.
    own_prior(후속실험 3)를 주면 면적 항을 넣은 '자기 prior' 단독 성능을 같은 파일에 덧붙인다.
    """
    Path(dir_).mkdir(parents=True, exist_ok=True)
    sfx = f"_{tag}" if tag else ""

    ls_ok = gt_df["ls_eval_mask"].astype(bool)
    lq_ok = gt_df["lq_eval_mask"].astype(bool)

    # (1) 요약 — 나중에 여러 실험을 세로로 쌓기 좋게
    summary = pd.DataFrame([{
        "n_events": int(gt_df["event_idx"].nunique()),
        "n": result.n,
        "n_ls": result.n_ls,
        "n_lq": result.n_lq,
        "mse_ls": result.mse_ls,
        "mse_lq": result.mse_lq,
        # 후속실험 1의 완료기준. posterior가 prior 단독을 넘어야 한다.
        "auc_ls": result.auc_ls,
        "auc_lq": result.auc_lq,
        "auc_prior_ls": result.auc_prior_ls,
        "auc_prior_lq": result.auc_prior_lq,
        # 이벤트별 AUC를 행 수로 가중평균한 값
        "auc_ls_wavg": result.auc_ls_wavg,
        "auc_lq_wavg": result.auc_lq_wavg,
        "auc_prior_ls_wavg": result.auc_prior_ls_wavg,
        "auc_prior_lq_wavg": result.auc_prior_lq_wavg,
        # 양성 개수도 평가 가능한 행 안에서만 세야 placeholder가 섞이지 않는다.
        "n_pos_ls": int(gt_df.loc[ls_ok, "ls_true"].sum()),
        "n_pos_lq": int(gt_df.loc[lq_ok, "lq_true"].sum()),
        "note": "LS/LQ 각각의 eval_mask로 독립 평가",
    }])
    if own_prior is not None:
        summary["auc_ownprior_ls"] = own_prior.auc_ls
        summary["auc_ownprior_lq"] = own_prior.auc_lq
        summary["auc_ownprior_ls_wavg"] = own_prior.auc_ls_wavg
        summary["auc_ownprior_lq_wavg"] = own_prior.auc_lq_wavg
        summary["mse_ownprior_ls"] = own_prior.mse_ls
        summary["mse_ownprior_lq"] = own_prior.mse_lq
    summary.to_csv(f"{dir_}/eval_summary{sfx}.csv", index=False, encoding="utf-8-sig")

    # (2) 상세 — 시정촌별로 정답/예측/오차를 펼쳐서 확인용
    detail = result.merged.copy()
    detail["err_ls"] = (detail["p_ls"] - detail["ls_true"]) ** 2
    detail["err_lq"] = (detail["p_lq"] - detail["lq_true"]) ** 2

    # eval_mask=False인 행의 정답은 원본이 NA라서 넣어둔 placeholder 0이다.
    # 진짜 정답이 아니므로 오차를 계산해봐야 의미가 없고 MSE에도 안 들어간다.
    # 파일만 보고 오해하지 않도록 LS/LQ 각각 비워둔다.
    detail.loc[~detail["ls_eval_mask"].astype(bool), ["ls_true", "err_ls"]] = pd.NA
    detail.loc[~detail["lq_eval_mask"].astype(bool), ["lq_true", "err_lq"]] = pd.NA

    # 이벤트 순, 그 안에서 시정촌코드 순. 코드는 5자리 고정폭이라 문자열 정렬이 곧 번호 순이다.
    detail.sort_values(["event_idx", "muni_code"]).to_csv(
        f"{dir_}/eval_detail{sfx}.csv", index=False, encoding="utf-8-sig"
    )

    # (3) 이벤트별 — 완료기준이 "2004 니가타 LS AUC"라 이벤트 분해가 있어야 판정된다.
    per_event = result.per_event
    if own_prior is not None:
        own = own_prior.per_event[["event_idx", "auc_ls", "auc_lq"]].rename(
            columns={"auc_ls": "auc_ownprior_ls", "auc_lq": "auc_ownprior_lq"}
        )
        per_event = per_event.merge(own, on="event_idx", how="left")
    per_event.to_csv(
        f"{dir_}/eval_per_event{sfx}.csv", index=False, encoding="utf-8-sig"
    )

    print(f"저장: {dir_}/eval_summary{sfx}.csv, {dir_}/eval_detail{sfx}.csv, "
          f"{dir_}/eval_per_event{sfx}.csv")


# 합본에서 뽑아 쓸 AUC/MSE 열. eval_summary의 컬럼명 그대로다.
COMPARISON_METRICS = [
    ("auc_ls_wavg",       "LS AUC 가중평균"),
    ("auc_prior_ls_wavg", "LS prior 가중평균"),
    ("auc_ls",            "LS AUC 전체"),
    ("auc_prior_ls",      "LS prior 전체"),
    ("auc_ownprior_ls_wavg", "LS 자기prior 가중평균"),
    ("auc_ownprior_ls",      "LS 자기prior 전체"),
    ("auc_lq_wavg",       "LQ AUC 가중평균"),
    ("auc_prior_lq_wavg", "LQ prior 가중평균"),
    ("auc_lq",            "LQ AUC 전체"),
    ("auc_prior_lq",      "LQ prior 전체"),
    ("auc_ownprior_lq_wavg", "LQ 자기prior 가중평균"),
    ("auc_ownprior_lq",      "LQ 자기prior 전체"),
    ("mse_ls",            "MSE_LS"),
    ("mse_lq",            "MSE_LQ"),
    ("n_ls",              "LS 평가행"),
    ("n_lq",              "LQ 평가행"),
]


def merge_runs(runs, dir_="outputs", out_prefix="comparison"):
    """브랜치별로 따로 돌린 실행 결과를 한 표로 합친다.

    runs
        [(tag, 표시이름), ...]
        tag는 train.py --tag 에 준 값이다. dir_ 안의
        eval_summary_{tag}.csv / eval_per_event_{tag}.csv / params_{tag}.csv 를 읽는다.

    만드는 파일
        {out_prefix}_auc.csv       조건이 행, 지표가 열
        {out_prefix}_per_event.csv 이벤트가 행, 조건이 열 (LS/LQ 각각)
        {out_prefix}_params.csv    파라미터가 행, 조건이 열
                                   그 조건의 모델에 없는 항은 빈칸으로 남는다

    조건마다 파라미터 구성이 다르므로(공변량 유무) 파라미터 합본은 outer join한다.
    빈칸은 "값이 0"이 아니라 "그 모델에 그 항이 없음"을 뜻한다.
    """
    dir_ = Path(dir_)
    missing = [t for t, _ in runs if not (dir_ / f"eval_summary_{t}.csv").exists()]
    if missing:
        raise FileNotFoundError(
            f"{dir_}에 다음 tag의 eval_summary가 없습니다: {missing}"
        )

    # (1) AUC / MSE 요약
    # 조건마다 b 범위와 파라미터 수가 다를 수 있어 params CSV에서 같이 읽어 붙인다.
    rows = []
    for tag, name in runs:
        s = pd.read_csv(dir_ / f"eval_summary_{tag}.csv").iloc[0]
        row = {"조건": name, "tag": tag}
        ppath = dir_ / f"params_{tag}.csv"
        if ppath.exists():
            p = pd.read_csv(ppath)
            b_rows = p.loc[p["group"] == "b", "transform"]
            row["b 범위"] = b_rows.iloc[0] if len(b_rows) else ""
            c_rows = p.loc[p["group"] == "c", "transform"]
            row["c"] = c_rows.iloc[0] if len(c_rows) else ""
            # 고정값(fixed…)과 묶인 값(tied…)은 학습 파라미터가 아니다.
            row["학습 파라미터"] = int(
                (~p["transform"].astype(str).str.startswith(("fixed", "tied"))).sum()
            )
        # 자기prior 열은 후속실험 3 실행분에만 있다. 없으면 빈칸으로 둔다.
        row.update({label: s.get(col, float("nan")) for col, label in COMPARISON_METRICS})
        rows.append(row)
    auc = pd.DataFrame(rows)
    auc["LS − prior"] = auc["LS AUC 가중평균"] - auc["LS prior 가중평균"]
    auc["LQ − prior"] = auc["LQ AUC 가중평균"] - auc["LQ prior 가중평균"]
    auc.to_csv(dir_ / f"{out_prefix}_auc.csv", index=False, encoding="utf-8-sig")

    # (2) 이벤트별
    per = []
    for tag, name in runs:
        p = pd.read_csv(dir_ / f"eval_per_event_{tag}.csv")
        p.insert(0, "조건", name)
        per.append(p)
    per_all = pd.concat(per, ignore_index=True)
    per_all.to_csv(dir_ / f"{out_prefix}_per_event.csv", index=False, encoding="utf-8-sig")

    # (3) 파라미터 — 조건마다 항이 달라 outer join한다.
    #
    # transform은 키에 넣지 않는다. b의 transform 문자열에는 범위가 들어 있어
    # (bounded[-2.0,2.0] vs bounded[-2.0,4.0]) 조건마다 달라지고, 키에 넣으면
    # 같은 b_LS가 두 행으로 갈라진다. b 범위는 위 요약표에 따로 싣는다.
    KEY = ["module", "group", "idx", "label"]
    merged = None
    for tag, name in runs:
        path = dir_ / f"params_{tag}.csv"
        if not path.exists():
            print(f"  건너뜀: {path.name} 없음")
            continue
        p = pd.read_csv(path)[KEY + ["value"]].rename(columns={"value": name})
        merged = p if merged is None else merged.merge(p, on=KEY, how="outer")
    if merged is not None:
        order = {g: i for i, g in enumerate(GROUP_ORDER)}
        merged = merged.sort_values(
            by=["group", "idx"],
            key=lambda s: s.map(order) if s.name == "group" else s,
        ).reset_index(drop=True)
        merged.to_csv(dir_ / f"{out_prefix}_params.csv", index=False, encoding="utf-8-sig")

    print(f"저장: {dir_}/{out_prefix}_auc.csv, {dir_}/{out_prefix}_per_event.csv"
          + (f", {dir_}/{out_prefix}_params.csv" if merged is not None else ""))
    return auc, per_all, merged
