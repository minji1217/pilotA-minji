"""
DisasterNet - data.py

시정촌 단위 DPM 집계표(DPM_municipality_v5.xlsx, 시트 "집계")를 읽어
pilotA의 PilotABatch 행과 (event_idx, 시정촌코드)로 맞춘다.

DisasterNet이 쓰는 입력은 세 가지뿐이다.
  - DPM 관측값 y           ← DPM xlsx
  - USGS LS / LQ prior     ← PilotABatch.pi_ls / pi_lq  (pilotA와 같은 값)
  - 건물피해 prior용 PGV    ← PilotABatch.pgv

피해 통계(y 6채널)는 쓰지 않는다. 그게 pilotA와의 차이이자 비교 포인트다.

DPM이 있는 이벤트는 5개(구마모토 / 오사카 / 훗카이도 / 후쿠시마 / 노토)뿐이라
비교는 이 5개 이벤트 안에서, 두 모델 모두 예측이 있는 행에서만 한다.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from loader import _normalize_municipality_code
from schema import DTYPE, EVENT_TO_INDEX, PilotABatch

DPM_SHEET = "집계"

# 논문 식 7은 DPM을 LogNormal로 두므로 관측값이 양수여야 한다.
# DPM_평균 / DPM_정규화z는 음수가 있어 임의 이동이 필요하고,
# DPM_변화25초과비율(시정촌 화소 중 DPM>0.25 비율)은 항상 (0,1) 안이라 그대로 쓸 수 있다.
# 픽셀 DPM을 시정촌 하나로 요약할 때 "변화한 면적 비율"이라 해석도 자연스럽다.
DEFAULT_DPM_COLUMN = "DPM_변화25초과비율"

# 자료설명 시트: base상관도 0.30 미만은 "측정 실패에 가까움", 처리 방침 미정.
# 기본은 제외하지 않고(None) 옵션으로만 둔다.
BASE_COH_COLUMN = "base상관도"


@dataclass
class DPMRows:
    """DisasterNet 학습·추론 입력. 길이 N은 DPM과 PilotABatch가 둘 다 있는 행 수."""
    y: Tensor               # [N]  DPM 관측값
    pi_ls: Tensor           # [N]  USGS LS prior
    pi_lq: Tensor           # [N]  USGS LQ prior
    pgv: Tensor             # [N]
    event_idx: Tensor       # [N]  long
    municipality_code: tuple
    model_row_idx: Tensor   # [N]  long. PilotABatch에서의 행 번호 (pilotA 예측을 같은 행으로 뽑을 때 쓴다)

    @property
    def n(self) -> int:
        return int(self.y.shape[0])

    def subset(self, mask: Tensor) -> "DPMRows":
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        return DPMRows(
            y=self.y[idx], pi_ls=self.pi_ls[idx], pi_lq=self.pi_lq[idx],
            pgv=self.pgv[idx], event_idx=self.event_idx[idx],
            municipality_code=tuple(self.municipality_code[i] for i in idx.tolist()),
            model_row_idx=self.model_row_idx[idx],
        )


def read_dpm_table(path, *, column=DEFAULT_DPM_COLUMN, min_base_coh=None) -> pd.DataFrame:
    """DPM xlsx → (event_idx, muni_code, dpm, base_coh) DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DPM XLSX를 찾을 수 없습니다: {path}")

    raw = pd.read_excel(path, sheet_name=DPM_SHEET)
    for col in ("이벤트", "시정촌코드", column):
        if col not in raw.columns:
            raise ValueError(f"DPM 시트에 '{col}' 컬럼이 없습니다. 있는 컬럼: {list(raw.columns)}")

    unknown = sorted(set(raw["이벤트"]) - set(EVENT_TO_INDEX))
    if unknown:
        raise ValueError(f"schema.EVENTS에 없는 이벤트명: {unknown}")

    df = pd.DataFrame({
        "event_idx": raw["이벤트"].map(EVENT_TO_INDEX).astype(int),
        "muni_code": raw["시정촌코드"].map(_normalize_municipality_code),
        "dpm": pd.to_numeric(raw[column], errors="coerce"),
        "base_coh": pd.to_numeric(raw.get(BASE_COH_COLUMN), errors="coerce"),
    })

    bad = df["muni_code"].isna() | df["dpm"].isna()
    if bad.any():
        print(f"[DPM] 코드/값이 비어 제외한 행 {int(bad.sum())}개")
        df = df.loc[~bad]

    if min_base_coh is not None:
        low = df["base_coh"] < min_base_coh
        print(f"[DPM] base상관도 < {min_base_coh} 제외 {int(low.sum())}개")
        df = df.loc[~low]

    if df.duplicated(["event_idx", "muni_code"]).any():
        raise ValueError("DPM 시트에 중복 (이벤트, 시정촌코드)가 있습니다.")
    return df.reset_index(drop=True)


def align_dpm_to_batch(dpm_df: pd.DataFrame, batch: PilotABatch) -> DPMRows:
    """DPM 행을 PilotABatch 행과 맞춘다. 양쪽에 다 있는 행만 남는다."""
    model_df = pd.DataFrame({
        "model_row_idx": np.arange(batch.batch_size),
        "event_idx": batch.event_idx.tolist(),
        "muni_code": list(batch.municipality_code),
    })
    merged = dpm_df.merge(model_df, on=["event_idx", "muni_code"], how="left")

    no_model = merged["model_row_idx"].isna()
    if no_model.any():
        lost = merged.loc[no_model, ["event_idx", "muni_code"]].to_dict("records")
        print(f"[DPM] PilotABatch에 없는 DPM 행 {len(lost)}개 (USGS prior가 없어 제외): {lost[:10]}")
        merged = merged.loc[~no_model]

    merged = merged.sort_values(["event_idx", "muni_code"]).reset_index(drop=True)
    idx = torch.as_tensor(merged["model_row_idx"].astype(int).to_numpy(), dtype=torch.long)

    rows = DPMRows(
        y=torch.as_tensor(merged["dpm"].to_numpy(), dtype=DTYPE),
        pi_ls=batch.pi_ls[idx].to(DTYPE),
        pi_lq=batch.pi_lq[idx].to(DTYPE),
        pgv=batch.pgv[idx].to(DTYPE),
        event_idx=batch.event_idx[idx].long(),
        municipality_code=tuple(merged["muni_code"]),
        model_row_idx=idx,
    )
    counts = pd.Series(rows.event_idx.tolist()).value_counts().sort_index()
    print(f"[DPM] 정렬 완료 {rows.n}행 / 이벤트별 {counts.to_dict()}")
    return rows
