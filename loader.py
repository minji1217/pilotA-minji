"""
Pilot A - data/loader.py

이 모듈은 실제 피해 통계 XLSX와 USGS XLSX를 읽어 PilotABatch를 만든다.

처리 순서:
1. 이벤트 시트의 실제 헤더를 찾는다.
2. 합계/소계가 아닌 실제 시정촌 행만 남긴다.
3. 시정촌코드를 5자리 문자열로 정규화한다. 예: 1581 -> "01581"
4. 피해 6채널의 결측을 y=0 placeholder + obs_mask=False로 분리한다.
5. 같은 이벤트 안에서 5자리 시정촌코드로 통계와 USGS를 join한다.
6. LS_prior(평균), LQ_prior(평균), PGV와 Exposure가 모두 있는 행만 남긴다.
7. 통계 XLSX의 wooden_ratio / mountain_ratio를 최종 모델 행 기준으로 표준화해 z_wood / z_mtn을 만든다.
8. population / households_general로 E를 만든다.
8-1. (후속실험 3) raw/시정촌_면적.csv의 면적과 격자 칸 넓이로 log k(LS 7.5″, LQ 15″)를 만든다.
9. PyTorch PilotABatch로 변환하고 batch.validate()를 실행한다.
10. eval용 GT가 필요하면 LS_LF 데이터자료.xlsx를 읽어 EvalGroundTruthBatch를 별도로 만든다.

중요: 시정촌코드는 계산용 숫자가 아니라 ID이므로 int로 바꾸어 보관하지 않는다.
GT는 모델 입력이 아니므로 PilotABatch에 넣지 않고 EvalGroundTruthBatch로 분리한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from schema import (
    AREA_COLUMN,
    AREA_EVENT_COLUMN,
    AREA_PATH,
    CHANNELS,
    KM_PER_DEG_LAT,
    KM_PER_DEG_LON_EQUATOR,
    LQ_GRID_ARCSEC,
    LS_GRID_ARCSEC,
    PREFECTURE_CAPITAL_LAT,
    DAMAGE_COLUMN_MAP,
    DTYPE,
    EVENTS,
    EVENT_TO_INDEX,
    EXPECTED_NUM_MODEL_ROWS,
    EXPECTED_NUM_STATS_ROWS,
    EXPECTED_NUM_USGS_MATCHED_ROWS,
    EXPOSURE_COLUMN_BY_CHANNEL,
    HOUSEHOLDS_COLUMN,
    INDEX_DTYPE,
    MUNICIPALITY_CODE_COLUMN,
    MUNICIPALITY_CODE_WIDTH,
    MOUNTAIN_RATIO_COLUMN,
    POPULATION_COLUMN,
    USGS_LQ_PRIOR_COLUMN,
    USGS_LS_PRIOR_COLUMN,
    USGS_PGV_COLUMN,
    WOODEN_RATIO_COLUMN,
    EvalGroundTruthBatch,
    PilotABatch,
)


# 통계 XLSX에서 모델 입력을 만들기 위해 반드시 필요한 컬럼이다.
STATS_REQUIRED_COLUMNS: tuple[str, ...] = (
    MUNICIPALITY_CODE_COLUMN,
    *tuple(DAMAGE_COLUMN_MAP[channel] for channel in CHANNELS),
    POPULATION_COLUMN,
    HOUSEHOLDS_COLUMN,
    WOODEN_RATIO_COLUMN,
    MOUNTAIN_RATIO_COLUMN,
)

# USGS XLSX에서는 평균 prior 두 개와 PGV만 사용한다.
USGS_REQUIRED_COLUMNS: tuple[str, ...] = (
    MUNICIPALITY_CODE_COLUMN,
    USGS_LS_PRIOR_COLUMN,
    USGS_LQ_PRIOR_COLUMN,
    USGS_PGV_COLUMN,
)


# GT 파일 컬럼명이다.
GT_CODE_COLUMN: str = "muni_code"
GT_LS_FLAG_COLUMN: str = "ls_flag"
GT_LS_AREA_COLUMN: str = "ls_area_ha"
# 후속실험 5: 시정촌 면적 중 산사태 인벤토리 판독 범위 안에 든 비율(0~1). 폴리곤 이벤트 시트에만 있다.
GT_COVERAGE_COLUMN: str = "coverage_ratio"
GT_LQ_FLAG_COLUMN: str = "lq_flag"
GT_LQ_JSHIS_FLAG_COLUMN: str = "jshis_flag"

# train.py / experiment.py를 다음 단계에서 수정하기 전 import 오류를 막기 위한
# 임시 호환 상수다. 이제 GT는 한 이벤트가 아니라 EVENTS 전체를 평가 대상으로 한다.
EVAL_EVENT_NAME: str = "전체 이벤트"

# 2018 훗카이도 LQ GT는 삿포로시가 01101~01110의 10개 구로 나뉘어 있지만
# 모델 데이터는 삿포로시 01100 한 행이다.
# jshis_flag의 NA→0 규칙을 먼저 적용한 뒤 max(gt_lq)로 01100에 집계한다.
SAPPORO_CITY_CODE: str = "01100"
SAPPORO_WARD_CODES: frozenset[str] = frozenset(
    f"011{ward:02d}" for ward in range(1, 11)
)


def _normalize_municipality_code(value: object) -> str | None:
    """
    Excel에서 읽은 시정촌코드를 앞자리 0을 포함한 5자리 문자열로 변환한다.

    입력 예: 1581, 1581.0, "1581", "01581"
    출력 예: 모두 "01581"
    빈 값이나 숫자로 해석할 수 없는 값은 None을 반환한다.

    이유: 시정촌코드는 수치 계산 대상이 아니라 ID이므로 01581과 같은 앞자리 0을 보존해야 한다.
    """
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    # Excel 숫자 셀은 pandas에서 1581.0처럼 읽힐 수 있으므로 먼저 숫자로 해석한다.
    numeric = pd.to_numeric(text, errors="coerce")
    if pd.isna(numeric):
        return None

    # 시정촌코드는 정수형 ID이므로 1581.5 같은 값은 정상 코드로 인정하지 않는다.
    numeric_float = float(numeric)
    if not numeric_float.is_integer():
        return None

    # int 변환은 정규화 과정에서만 사용하고 최종 저장은 다시 5자리 문자열로 한다.
    code = str(int(numeric_float)).zfill(MUNICIPALITY_CODE_WIDTH)
    if len(code) != MUNICIPALITY_CODE_WIDTH:
        raise ValueError(
            f"시정촌코드는 {MUNICIPALITY_CODE_WIDTH}자리여야 합니다: raw={value!r}, normalized={code!r}"
        )

    return code


def _find_header_row(path: str | Path, sheet_name: str) -> int:
    """
    Excel 시트에서 실제 표의 헤더 행 번호를 찾는다.

    입력: XLSX 경로, 이벤트 시트명
    출력: 실제 헤더의 0-based 행 번호
    이유: 현재 파일은 제목·설명·빈 줄 뒤에 실제 표가 시작하므로 header=0을 고정하면 안 된다.
    """
    # header=None으로 모든 행을 데이터로 읽어 실제 헤더 후보를 직접 찾는다.
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)

    for row_idx, row in raw.iterrows():
        # NaN은 제외하고 셀 값을 문자열 집합으로 바꿔 컬럼명 존재 여부를 확인한다.
        values = {str(value).strip() for value in row.tolist() if pd.notna(value)}

        # 통계와 USGS의 실제 헤더에는 공통으로 '부현'과 '시정촌코드'가 존재한다.
        if "부현" in values and MUNICIPALITY_CODE_COLUMN in values:
            return int(row_idx)

    raise ValueError(
        f"[{sheet_name}] 실제 헤더 행을 찾지 못했습니다. "
        f"'{MUNICIPALITY_CODE_COLUMN}' 컬럼이 있는지 확인하세요."
    )


def _read_table(path: str | Path, sheet_name: str) -> pd.DataFrame:
    """
    제목·설명 행을 건너뛰고 실제 표만 DataFrame으로 읽는다.

    입력: XLSX 경로, 이벤트 시트명
    출력: 실제 컬럼명을 가진 pandas.DataFrame
    """
    header_row = _find_header_row(path, sheet_name)

    # 찾은 실제 헤더 행을 pandas의 header로 지정한다.
    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)

    # 완전히 빈 컬럼은 전처리에 필요하지 않으므로 제거한다.
    df = df.dropna(axis=1, how="all")

    return df.copy()


def _require_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    sheet_name: str,
    source_name: str,
) -> None:
    """
    필요한 컬럼이 실제 시트에 모두 있는지 검사한다.

    입력: DataFrame, 필수 컬럼 목록, 시트명, 자료 종류
    출력: 정상 None, 누락 컬럼이 있으면 ValueError
    """
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"[{sheet_name}] {source_name} 필수 컬럼이 없습니다: {missing}")


def _keep_municipality_rows(
    df: pd.DataFrame,
    *,
    sheet_name: str,
    source_name: str,
) -> pd.DataFrame:
    """
    합계·소계·설명 행을 제거하고 실제 시정촌 행만 남긴다.

    입력: 원본 표 DataFrame
    출력: 시정촌코드가 "01581" 같은 5자리 문자열로 정규화된 DataFrame

    합계 행처럼 시정촌코드가 비어 있거나 코드로 해석할 수 없는 행은 제거한다.
    """
    result = df.copy()

    # 각 셀을 5자리 문자열 코드로 정규화한다. 예: 1581.0 -> "01581"
    normalized_codes = result[MUNICIPALITY_CODE_COLUMN].map(_normalize_municipality_code)

    # 정상 코드가 만들어진 행만 실제 시정촌 행으로 남긴다.
    valid_code_mask = normalized_codes.notna()
    result = result.loc[valid_code_mask].copy()

    # int가 아니라 문자열 자체를 저장해 leading zero를 보존한다.
    result[MUNICIPALITY_CODE_COLUMN] = normalized_codes.loc[valid_code_mask].astype(str)

    # 한 이벤트 안에서 동일 시정촌코드가 중복되면 one-to-one join을 보장할 수 없으므로 즉시 중단한다.
    if result[MUNICIPALITY_CODE_COLUMN].duplicated().any():
        duplicates = result.loc[
            result[MUNICIPALITY_CODE_COLUMN].duplicated(keep=False),
            MUNICIPALITY_CODE_COLUMN,
        ].tolist()
        raise ValueError(f"[{sheet_name}] {source_name}에 중복 시정촌코드가 있습니다: {duplicates}")

    return result.reset_index(drop=True)


def _parse_damage_column(
    series: pd.Series,
    *,
    sheet_name: str,
    column_name: str,
) -> tuple[pd.Series, pd.Series]:
    """
    피해 한 채널을 숫자 y와 관측 여부 mask로 분리한다.

    입력 예: [1097, "-", 0]
    출력 예: values=[1097.0,0.0,0.0], observed=[True,False,True]

    '-'는 실제 0건이 아니라 결측이다. y에는 계산용 placeholder 0을 넣고 obs_mask=False로 보존한다.
    """
    # 문자열 앞뒤 공백을 없애 " - " 같은 값도 정상 결측으로 인식한다.
    stripped = series.astype("string").str.strip()

    # 빈 셀과 여러 종류의 dash 문자를 결측으로 본다.
    missing = series.isna() | stripped.isin(["", "-", "–", "—"])

    # 결측이 아닌 값만 숫자로 변환한다. 이상 문자열은 NaN이 되어 아래 invalid 검사에서 잡힌다.
    numeric = pd.to_numeric(series.where(~missing), errors="coerce")

    invalid = (~missing) & numeric.isna()
    if invalid.any():
        bad_values = series.loc[invalid].astype(str).unique().tolist()
        raise ValueError(
            f"[{sheet_name}] '{column_name}'에 숫자도 결측표시도 아닌 값이 있습니다: {bad_values}"
        )

    # PyTorch count Tensor는 숫자여야 하므로 결측 위치에는 0 placeholder를 넣는다.
    values = numeric.fillna(0.0).astype("float64")

    # 실제 관측값이면 True, 결측이면 False다.
    observed = (~missing).astype(bool)

    return values, observed


def _prepare_stats_sheet(path: str | Path, sheet_name: str) -> pd.DataFrame:
    """
    한 이벤트의 피해 통계 시트를 join 직전 형태로 정리한다.

    출력: 시정촌코드, 피해 6채널, obs_* 6개, population, households_general,
          wooden_ratio, mountain_ratio

    후속실험 1의 wooden_ratio / mountain_ratio는
    데이터 수집·합병·결측 대체가 끝난 0~1 원비율을 기대한다.
    여기서는 z-score를 만들지 않고, 9개 이벤트의 최종 모델 행이 모두 확정된 뒤 표준화한다.
    """
    df = _read_table(path, sheet_name)
    _require_columns(df, STATS_REQUIRED_COLUMNS, sheet_name=sheet_name, source_name="통계")
    df = _keep_municipality_rows(df, sheet_name=sheet_name, source_name="통계")

    result = pd.DataFrame()

    # 이미 _keep_municipality_rows()에서 5자리 문자열로 정규화된 코드를 그대로 복사한다.
    result[MUNICIPALITY_CODE_COLUMN] = df[MUNICIPALITY_CODE_COLUMN].astype(str)

    # schema.py의 CHANNELS 순서대로 피해값과 mask를 생성한다.
    for channel in CHANNELS:
        source_column = DAMAGE_COLUMN_MAP[channel]
        values, observed = _parse_damage_column(
            df[source_column],
            sheet_name=sheet_name,
            column_name=source_column,
        )
        result[channel] = values
        result[f"obs_{channel}"] = observed

    # Exposure 원본값은 여기서 숫자로만 변환하고 실제 사용 가능 여부는 join 이후에 판단한다.
    result[POPULATION_COLUMN] = pd.to_numeric(
        df[POPULATION_COLUMN],
        errors="coerce",
    ).astype("float64")
    result[HOUSEHOLDS_COLUMN] = pd.to_numeric(
        df[HOUSEHOLDS_COLUMN],
        errors="coerce",
    ).astype("float64")

    # 후속실험 1의 시정촌 취약성 공변량 원비율을 숫자로 읽는다.
    # 엑셀에는 z값이 아니라 0~1 원비율을 저장하고, 표준화는 최종 모델 행 확정 후 수행한다.
    result[WOODEN_RATIO_COLUMN] = pd.to_numeric(
        df[WOODEN_RATIO_COLUMN],
        errors="coerce",
    ).astype("float64")
    result[MOUNTAIN_RATIO_COLUMN] = pd.to_numeric(
        df[MOUNTAIN_RATIO_COLUMN],
        errors="coerce",
    ).astype("float64")

    # 값 자체의 결측/범위 검증은 여기서 하지 않는다.
    # 통계+USGS 조건으로 실제 모델 행을 먼저 확정한 뒤,
    # 최종 418행에 대해서만 검증하고 표준화한다.

    return result


def _prepare_usgs_sheet(path: str | Path, sheet_name: str) -> pd.DataFrame:
    """
    한 이벤트의 USGS 시트를 이번 Pilot에 필요한 값만 남겨 정리한다.

    출력: 시정촌코드, pi_ls=LS_prior(평균), pi_lq=LQ_prior(평균), pgv=PGV
    """
    df = _read_table(path, sheet_name)
    _require_columns(df, USGS_REQUIRED_COLUMNS, sheet_name=sheet_name, source_name="USGS")
    df = _keep_municipality_rows(df, sheet_name=sheet_name, source_name="USGS")

    result = pd.DataFrame()
    result[MUNICIPALITY_CODE_COLUMN] = df[MUNICIPALITY_CODE_COLUMN].astype(str)

    # 이번 Pilot에서 확정한 평균 prior를 모델 변수명으로 바꿔 저장한다.
    result["pi_ls"] = pd.to_numeric(df[USGS_LS_PRIOR_COLUMN], errors="coerce")
    result["pi_lq"] = pd.to_numeric(df[USGS_LQ_PRIOR_COLUMN], errors="coerce")
    result["pgv"] = pd.to_numeric(df[USGS_PGV_COLUMN], errors="coerce")

    # 실제 값이 존재하는 prior가 [0,1] 범위를 벗어나면 데이터 오류로 처리한다.
    for prior_name in ("pi_ls", "pi_lq"):
        valid = result[prior_name].notna()
        out_of_range = valid & ((result[prior_name] < 0) | (result[prior_name] > 1))
        if out_of_range.any():
            bad_codes = result.loc[out_of_range, MUNICIPALITY_CODE_COLUMN].tolist()
            raise ValueError(f"[{sheet_name}] {prior_name}가 [0,1] 범위를 벗어났습니다: {bad_codes}")

    return result


def _merge_event(
    stats_df: pd.DataFrame,
    usgs_df: pd.DataFrame,
    sheet_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    같은 이벤트의 통계와 USGS를 5자리 시정촌코드로 join한다.

    출력:
    - model_df: 현재 모델에 필요한 값이 모두 있는 행
    - excluded_df: USGS/PGV/prior/Exposure 문제로 제외된 행

    이벤트 시트별로 따로 처리하므로 실질적인 key는 이벤트 + 시정촌코드다.
    """
    # 통계를 기준으로 left join해 USGS에 없는 시정촌도 제외 사유를 확인할 수 있게 한다.
    merged = stats_df.merge(
        usgs_df,
        on=MUNICIPALITY_CODE_COLUMN,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    has_usgs_row = merged["_merge"].eq("both")
    has_usgs_values = merged[["pgv", "pi_ls", "pi_lq"]].notna().all(axis=1)
    has_exposure = merged[[POPULATION_COLUMN, HOUSEHOLDS_COLUMN]].notna().all(axis=1)
    positive_exposure = (
        (merged[POPULATION_COLUMN] > 0)
        & (merged[HOUSEHOLDS_COLUMN] > 0)
    )
    positive_pgv = merged["pgv"] > 0

    # 현재 회귀식과 prior 계산에 필요한 조건을 모두 만족해야 모델 행으로 사용한다.
    usable = (
        has_usgs_row
        & has_usgs_values
        & has_exposure
        & positive_exposure
        & positive_pgv
    )

    model_df = merged.loc[usable].drop(columns="_merge").copy()

    # 제외 행은 버리지 않고 원인 분석용으로 별도 보관한다.
    excluded_df = merged.loc[~usable].copy()
    excluded_df["exclude_reason"] = ""
    excluded_df.loc[~has_usgs_row, "exclude_reason"] += "USGS 행 없음; "
    excluded_df.loc[has_usgs_row & ~has_usgs_values, "exclude_reason"] += "PGV/prior 결측; "
    excluded_df.loc[~has_exposure, "exclude_reason"] += "Exposure 결측; "
    excluded_df.loc[has_exposure & ~positive_exposure, "exclude_reason"] += "Exposure 0 이하; "
    excluded_df.loc[has_usgs_values & ~positive_pgv, "exclude_reason"] += "PGV 0 이하; "

    # regression.py에서 이벤트별 alpha_e를 선택할 수 있도록 0~8 index를 저장한다.
    model_df["event_idx"] = EVENT_TO_INDEX[sheet_name]

    # 제외 데이터에는 사람이 읽을 수 있도록 이벤트명도 보존한다.
    excluded_df["event"] = sheet_name

    return model_df.reset_index(drop=True), excluded_df.reset_index(drop=True)


def _standardize_vulnerability_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """
    최종 모델 행 전체를 기준으로 후속실험 1의 취약성 공변량을 z-score 표준화한다.

    입력:
        wooden_ratio   [0,1] 원비율
        mountain_ratio [0,1] 원비율

    출력 추가 컬럼:
        z_wood = (wooden_ratio - mean) / std
        z_mtn  = (mountain_ratio - mean) / std

    중요한 점:
    - 439개 통계 원본 전체가 아니라 통계+USGS 조건을 통과한 최종 모델 행을 기준으로 계산한다.
    - 원비율은 그대로 보존하고 z_wood / z_mtn을 새 컬럼으로 추가한다.
    - std=0이면 공변량에 변이가 없어 회귀계수를 학습할 수 없으므로 오류 처리한다.
    """
    result = df.copy()

    for raw_column, z_column in (
        (WOODEN_RATIO_COLUMN, "z_wood"),
        (MOUNTAIN_RATIO_COLUMN, "z_mtn"),
    ):
        # 후속실험 1에서는 데이터 수집/합병/결측 대체까지 끝난
        # 0~1 원비율을 입력으로 기대한다.
        # 단, 검사는 실제 학습에 들어가는 최종 모델 행에 대해서만 수행한다.
        if result[raw_column].isna().any():
            bad_rows = result.loc[
                result[raw_column].isna(),
                ["event_idx", MUNICIPALITY_CODE_COLUMN],
            ].to_dict("records")
            raise ValueError(
                f"최종 모델 행의 {raw_column}에 결측/비숫자 값이 남아 있습니다: "
                f"{bad_rows[:10]}"
            )

        out_of_range = (result[raw_column] < 0) | (result[raw_column] > 1)
        if out_of_range.any():
            bad_rows = result.loc[
                out_of_range,
                ["event_idx", MUNICIPALITY_CODE_COLUMN, raw_column],
            ].to_dict("records")
            raise ValueError(
                f"최종 모델 행의 {raw_column}은 0~1 범위여야 합니다: "
                f"{bad_rows[:10]}"
            )

        mean = float(result[raw_column].mean())
        std = float(result[raw_column].std())

        if not (pd.notna(mean) and pd.notna(std)):
            raise ValueError(
                f"{raw_column} 표준화를 위한 mean/std 계산에 실패했습니다."
            )
        if std <= 0:
            raise ValueError(
                f"{raw_column}의 표준편차가 0 이하라 표준화할 수 없습니다: std={std}"
            )

        result[z_column] = (
            (result[raw_column] - mean) / std
        ).astype("float64")

    return result



def _grid_cell_km2(lat_deg, arcsec: float):
    """위도 lat_deg에서 arcsec × arcsec 격자 한 칸의 넓이(km²)."""
    deg = arcsec / 3600.0
    return (deg * KM_PER_DEG_LON_EQUATOR * np.cos(np.radians(lat_deg))) * (deg * KM_PER_DEG_LAT)


def _add_area_columns(df: pd.DataFrame, area_path: str | Path) -> pd.DataFrame:
    """
    후속실험 3: 시정촌 면적으로 면적 항 log k를 만든다.

        log_k_ls = log(면적 / 7.5″ 칸 넓이)
        log_k_lq = log(면적 / 15″ 칸 넓이)

    - 면적은 (이벤트, 시정촌코드)로 붙인다. 같은 시정촌도 이벤트 시점마다 합병 전후 면적이 다를 수 있다.
    - 최종 모델 행 중 면적이 없거나 0 이하이면 멈춘다. 조용히 빼면 학습 행 수가 달라진다.
    - 칸 넓이는 현청 소재지 위도로 계산한다(schema.PREFECTURE_CAPITAL_LAT).
    """
    area_path = Path(area_path)
    if not area_path.exists():
        raise FileNotFoundError(f"시정촌 면적 CSV를 찾을 수 없습니다: {area_path}")

    area = pd.read_csv(area_path, dtype={MUNICIPALITY_CODE_COLUMN: str})
    need = {AREA_EVENT_COLUMN, MUNICIPALITY_CODE_COLUMN, AREA_COLUMN}
    if need - set(area.columns):
        raise ValueError(f"{area_path}에 필요한 컬럼이 없습니다: {sorted(need - set(area.columns))}")
    area[MUNICIPALITY_CODE_COLUMN] = area[MUNICIPALITY_CODE_COLUMN].map(_normalize_municipality_code)
    area = area[[AREA_EVENT_COLUMN, MUNICIPALITY_CODE_COLUMN, AREA_COLUMN]].rename(
        columns={AREA_EVENT_COLUMN: "_area_event"}
    )
    if area.duplicated(["_area_event", MUNICIPALITY_CODE_COLUMN]).any():
        raise ValueError(f"{area_path}에 (이벤트, 시정촌코드) 중복이 있습니다.")

    result = df.copy()
    result["_area_event"] = result["event_idx"].map(lambda i: EVENTS[int(i)])
    n_before = len(result)
    result = result.merge(
        area, on=["_area_event", MUNICIPALITY_CODE_COLUMN], how="left", validate="many_to_one"
    )
    if len(result) != n_before:
        raise ValueError("면적 join 후 행 수가 바뀌었습니다.")

    bad = result[AREA_COLUMN].isna() | ~(result[AREA_COLUMN] > 0)
    if bad.any():
        raise ValueError(
            f"최종 모델 행에 면적이 없거나 0 이하입니다: "
            f"{result.loc[bad, ['_area_event', MUNICIPALITY_CODE_COLUMN]].to_dict('records')[:10]}"
        )

    lat = result[MUNICIPALITY_CODE_COLUMN].str[:2].map(PREFECTURE_CAPITAL_LAT)
    if lat.isna().any():
        raise ValueError(
            f"현청 위도가 없는 都道府県 코드가 있습니다: "
            f"{sorted(result.loc[lat.isna(), MUNICIPALITY_CODE_COLUMN].str[:2].unique())}"
        )

    for name, arcsec in (("ls", LS_GRID_ARCSEC), ("lq", LQ_GRID_ARCSEC)):
        cell = _grid_cell_km2(lat.to_numpy(dtype="float64"), arcsec)
        result[f"cell_km2_{name}"] = cell
        result[f"log_k_{name}"] = np.log(result[AREA_COLUMN].to_numpy(dtype="float64") / cell)

    return result.drop(columns="_area_event")


def _add_exposure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    population / households_general을 6채널 Exposure E로 확장한다.

    예: population=4838, households_general=2121
    -> E=[4838,4838,4838,2121,2121,2121]
    """
    result = df.copy()

    for channel in CHANNELS:
        source_column = EXPOSURE_COLUMN_BY_CHANNEL[channel]
        result[f"E_{channel}"] = result[source_column].astype("float64")

    return result


def _to_batch(df: pd.DataFrame) -> PilotABatch:
    """
    정리된 DataFrame을 A/B 공통 PilotABatch로 변환한다.

    시정촌코드는 Tensor로 바꾸지 않고 5자리 문자열 tuple로 그대로 보존한다.
    """
    y_columns = list(CHANNELS)
    e_columns = [f"E_{channel}" for channel in CHANNELS]
    mask_columns = [f"obs_{channel}" for channel in CHANNELS]

    batch = PilotABatch(
        y=torch.tensor(
            df[y_columns].to_numpy(dtype="float64"),
            dtype=DTYPE,
        ),
        E=torch.tensor(
            df[e_columns].to_numpy(dtype="float64"),
            dtype=DTYPE,
        ),
        pgv=torch.tensor(
            df["pgv"].to_numpy(dtype="float64"),
            dtype=DTYPE,
        ),
        z_wood=torch.tensor(
            df["z_wood"].to_numpy(dtype="float64"),
            dtype=DTYPE,
        ),
        z_mtn=torch.tensor(
            df["z_mtn"].to_numpy(dtype="float64"),
            dtype=DTYPE,
        ),
        pi_ls=torch.tensor(
            df["pi_ls"].to_numpy(dtype="float64"),
            dtype=DTYPE,
        ),
        pi_lq=torch.tensor(
            df["pi_lq"].to_numpy(dtype="float64"),
            dtype=DTYPE,
        ),
        log_k_ls=torch.tensor(
            df["log_k_ls"].to_numpy(dtype="float64"),
            dtype=DTYPE,
        ),
        log_k_lq=torch.tensor(
            df["log_k_lq"].to_numpy(dtype="float64"),
            dtype=DTYPE,
        ),
        event_idx=torch.tensor(
            df["event_idx"].to_numpy(dtype="int64"),
            dtype=INDEX_DTYPE,
        ),
        obs_mask=torch.tensor(
            df[mask_columns].to_numpy(dtype=bool),
            dtype=torch.bool,
        ),

        # ID는 계산 대상이 아니므로 "01581" 같은 문자열 자체를 tuple로 저장한다.
        municipality_code=tuple(
            df[MUNICIPALITY_CODE_COLUMN].astype(str).tolist()
        ),
    )

    batch.validate()
    return batch


def _load_all_events(
    stats_path: str | Path,
    usgs_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """
    9개 이벤트 전체에 같은 전처리 규칙을 적용한다.

    출력: 전체 model_df, 전체 excluded_df, 통계 원본 행 수, USGS 매칭 행 수
    """
    model_frames: list[pd.DataFrame] = []
    excluded_frames: list[pd.DataFrame] = []
    total_stats_rows = 0
    total_usgs_matched_rows = 0

    for event in EVENTS:
        stats_df = _prepare_stats_sheet(stats_path, event)
        usgs_df = _prepare_usgs_sheet(usgs_path, event)

        total_stats_rows += len(stats_df)

        model_df, excluded_df = _merge_event(stats_df, usgs_df, event)

        # model_df는 모두 USGS 매칭 행이고, excluded_df 중 _merge=="both"인 행도 USGS 자체는 존재한다.
        event_usgs_matched_rows = len(model_df) + int(
            excluded_df["_merge"].eq("both").sum()
        )
        total_usgs_matched_rows += event_usgs_matched_rows

        model_frames.append(model_df)
        excluded_frames.append(excluded_df)

    all_model_df = pd.concat(model_frames, ignore_index=True)
    all_excluded_df = pd.concat(excluded_frames, ignore_index=True)

    return (
        all_model_df,
        all_excluded_df,
        total_stats_rows,
        total_usgs_matched_rows,
    )


def load_pilot_a_batch(
    stats_path: str | Path,
    usgs_path: str | Path,
    *,
    area_path: str | Path = AREA_PATH,
    strict_expected_rows: bool = True,
) -> PilotABatch:
    """
    실제 통계 XLSX와 USGS XLSX를 읽어 최종 PilotABatch를 만든다.

    입력:
    - stats_path: 재난프로젝트_시정촌별_통계데이터.xlsx
    - usgs_path: 재난프로젝트_시정촌별_USGS.xlsx
    - area_path: 시정촌 면적 CSV (후속실험 3의 log k용, 기본 raw/시정촌_면적.csv)
    - strict_expected_rows: True이면 현재 확인한 439/419/418 행 수가 맞는지 검사

    출력: PilotABatch
    """
    stats_path = Path(stats_path)
    usgs_path = Path(usgs_path)

    if not stats_path.exists():
        raise FileNotFoundError(f"통계 XLSX를 찾을 수 없습니다: {stats_path}")
    if not usgs_path.exists():
        raise FileNotFoundError(f"USGS XLSX를 찾을 수 없습니다: {usgs_path}")

    model_df, _, total_stats_rows, total_usgs_matched_rows = _load_all_events(
        stats_path,
        usgs_path,
    )

    # 최종 모델 행(현재 기대 418행)을 기준으로 취약성 공변량을 표준화한다.
    model_df = _standardize_vulnerability_covariates(model_df)

    # 후속실험 3: 면적 항 log k
    model_df = _add_area_columns(model_df, area_path)

    model_df = _add_exposure_columns(model_df)

    # 현재 실제 파일이 우리가 확인한 상태와 동일한지 점검한다.
    if strict_expected_rows:
        if total_stats_rows != EXPECTED_NUM_STATS_ROWS:
            raise ValueError(
                f"통계 원본 행 수가 예상과 다릅니다: "
                f"expected={EXPECTED_NUM_STATS_ROWS}, actual={total_stats_rows}"
            )
        if total_usgs_matched_rows != EXPECTED_NUM_USGS_MATCHED_ROWS:
            raise ValueError(
                f"USGS 매칭 행 수가 예상과 다릅니다: "
                f"expected={EXPECTED_NUM_USGS_MATCHED_ROWS}, actual={total_usgs_matched_rows}"
            )
        if len(model_df) != EXPECTED_NUM_MODEL_ROWS:
            raise ValueError(
                f"최종 모델 행 수가 예상과 다릅니다: "
                f"expected={EXPECTED_NUM_MODEL_ROWS}, actual={len(model_df)}"
            )

    return _to_batch(model_df)


def load_excluded_rows(
    stats_path: str | Path,
    usgs_path: str | Path,
) -> pd.DataFrame:
    """
    현재 모델에서 제외되는 시정촌과 제외 사유를 반환한다.

    학습 함수가 아니라 데이터 점검용 함수다.
    """
    _, excluded_df, _, _ = _load_all_events(stats_path, usgs_path)
    return excluded_df


def _read_gt_sheet(
    gt_path: str | Path,
    event_name: str,
    kind: str,
) -> tuple[pd.DataFrame, str]:
    """
    이벤트별 GT 시트를 읽는다.

    기본 시트명은 f"{event_name}(LS)" / f"{event_name}(LQ)"다.
    현재 파일의 2018 훗카이도 LQ에만 남아 있는 공백 이름도 임시로 허용한다.
    """
    if kind not in {"LS", "LQ"}:
        raise ValueError(f"GT kind는 LS 또는 LQ여야 합니다: {kind}")

    gt_path = Path(gt_path)
    sheet_name = f"{event_name}({kind})"

    with pd.ExcelFile(gt_path) as xls:
        available = set(xls.sheet_names)

        if sheet_name not in available:
            # 최신 파일 정리가 끝나기 전까지의 legacy 이름만 임시 지원한다.
            legacy_name = f"{event_name} ({kind})"
            if legacy_name in available:
                sheet_name = legacy_name
            else:
                raise ValueError(
                    f"GT 시트를 찾을 수 없습니다: expected='{sheet_name}'"
                )

        df = pd.read_excel(xls, sheet_name=sheet_name)

    return df, sheet_name


def _prepare_gt_code_rows(
    df: pd.DataFrame,
    *,
    sheet_name: str,
) -> pd.DataFrame:
    """GT 시정촌코드를 5자리 문자열로 정규화하고 실제 코드 행만 남긴다."""
    if GT_CODE_COLUMN not in df.columns:
        raise ValueError(
            f"[{sheet_name}] GT 필수 컬럼이 없습니다: ['{GT_CODE_COLUMN}']"
        )

    result = pd.DataFrame(index=df.index)
    result[MUNICIPALITY_CODE_COLUMN] = df[GT_CODE_COLUMN].map(
        _normalize_municipality_code
    )
    result = result.loc[result[MUNICIPALITY_CODE_COLUMN].notna()].copy()

    if result[MUNICIPALITY_CODE_COLUMN].duplicated().any():
        duplicates = result.loc[
            result[MUNICIPALITY_CODE_COLUMN].duplicated(keep=False),
            MUNICIPALITY_CODE_COLUMN,
        ].tolist()
        raise ValueError(f"[{sheet_name}] 중복 시정촌코드가 있습니다: {duplicates}")

    return result


def _parse_binary_gt_flag(
    series: pd.Series,
    *,
    sheet_name: str,
    column_name: str,
) -> pd.Series:
    """0/1/NA만 허용하는 GT flag를 숫자형으로 변환한다."""
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")

    # 문자열 NA/빈칸은 허용하지만, 그 외 이상 문자열이 NaN으로 바뀌는 것은 오류다.
    text = series.astype("string").str.strip()
    allowed_na_text = series.isna() | text.isin(["", "NA", "NaN", "nan", "N/A"])
    invalid_text = numeric.isna() & ~allowed_na_text
    if invalid_text.any():
        bad_values = series.loc[invalid_text].astype(str).unique().tolist()
        raise ValueError(
            f"[{sheet_name}] {column_name}는 0/1/NA만 허용합니다: {bad_values}"
        )

    invalid_number = numeric.notna() & ~numeric.isin([0.0, 1.0])
    if invalid_number.any():
        bad_values = series.loc[invalid_number].astype(str).unique().tolist()
        raise ValueError(
            f"[{sheet_name}] {column_name}는 0/1/NA만 허용합니다: {bad_values}"
        )

    return numeric


def _prepare_ls_ground_truth(
    gt_path: str | Path,
    event_name: str,
) -> pd.DataFrame:
    """
    한 이벤트의 LS GT를 정리한다.

    확정 규칙:
    1. ls_flag 컬럼이 있으면 ls_flag를 우선 사용한다.
       1 -> 1, 0 -> 0, NA -> 평가 제외
    2. ls_flag가 없으면 ls_area_ha를 사용한다.
       >0 -> 1, 0 -> 0, NA -> 평가 제외, 음수 -> 오류

    NA는 "없었다"가 아니라 "모른다"다. 정답지가 둘을 의도적으로 갈라 놓았다.
    ls_flag=0 행의 근거는 "조사표에 행이 있고 산사태 칸만 비었다 = 조사 대상이었고 0"인 반면,
    NA 행의 근거는 "조사의 유무 자체를 확인할 수 없어 NA"(2000 돗토리),
    "항공사진 판독 범위 밖"(2007 니가타오키),
    "신고 기반이라 산지 내부를 못 잡을 수 있어 NA"(2018 오사카, 2021 후쿠시마)다.
    근거가 적힌 NA 149행 중 "확인 결과 없었다"고 말하는 행은 하나도 없다.

    LQ가 jshis_flag에서만 NA->0을 쓰는 것도 같은 이치다. jshis는 전국을 덮는 위험도
    지도라 빈칸이 곧 0이지만, lq_flag는 현장 기록이라 NA를 평가에서 뺀다.
    ls_flag는 현장 기록 쪽이므로 지도용 규칙을 쓰면 안 된다.

    그 결과 LS 평가는 125행 / 양성률 84%로 좁다. 음성이 9개 이벤트를 통틀어 20행뿐이라
    AUC가 불안정하다는 것은 결과를 읽을 때 반드시 같이 말해야 한다.
    """
    df, sheet_name = _read_gt_sheet(gt_path, event_name, "LS")
    result = _prepare_gt_code_rows(df, sheet_name=sheet_name)

    if GT_LS_FLAG_COLUMN in df.columns:
        raw = _parse_binary_gt_flag(
            df.loc[result.index, GT_LS_FLAG_COLUMN],
            sheet_name=sheet_name,
            column_name=GT_LS_FLAG_COLUMN,
        )
        # NA는 "모른다"이므로 평가에서 뺀다.
        result["ls_eval_mask"] = raw.notna()
        result["gt_ls"] = raw.fillna(0.0).astype("int64")

    elif GT_LS_AREA_COLUMN in df.columns:
        raw_series = df.loc[result.index, GT_LS_AREA_COLUMN]
        ls_area = pd.to_numeric(raw_series, errors="coerce").astype("float64")

        # ls_area_ha에는 숫자/NA만 허용한다.
        text = raw_series.astype("string").str.strip()
        allowed_na_text = raw_series.isna() | text.isin(["", "NA", "NaN", "nan", "N/A"])
        invalid_text = ls_area.isna() & ~allowed_na_text
        if invalid_text.any():
            bad_values = raw_series.loc[invalid_text].astype(str).unique().tolist()
            raise ValueError(
                f"[{sheet_name}] ls_area_ha에 숫자/NA가 아닌 값이 있습니다: {bad_values}"
            )

        negative = ls_area.notna() & (ls_area < 0)
        if negative.any():
            bad_codes = result.loc[negative, MUNICIPALITY_CODE_COLUMN].tolist()
            raise ValueError(
                f"[{sheet_name}] ls_area_ha는 음수가 될 수 없습니다: {bad_codes}"
            )

        # 면적이 비어 있는 것도 "모른다"이므로 평가에서 뺀다.
        result["ls_eval_mask"] = ls_area.notna()
        result["gt_ls"] = (ls_area.fillna(0.0) > 0).astype("int64")

    else:
        raise ValueError(
            f"[{sheet_name}] LS GT에는 '{GT_LS_FLAG_COLUMN}' 또는 "
            f"'{GT_LS_AREA_COLUMN}' 컬럼이 필요합니다."
        )

    # 후속실험 5: 판독 범위 비율 cov.
    # 폴리곤 기반 라벨(coverage_ratio > 0)은 그대로 쓰고, 나머지는 1이다.
    #   - 보고서 기반 시트(돗토리, 오사카 등)는 coverage_ratio 열이 없다.
    #   - 훗카이도의 ls_flag만 있는 2행(에니와시, 기타히로시마시)은 coverage_ratio=0이지만
    #     보고서 기반 라벨이라 1이다.
    #   - coverage_ratio=0인 나머지 행은 판독 범위 밖이라 애초에 라벨이 없다(NA).
    result["cov"] = 1.0
    if GT_COVERAGE_COLUMN in df.columns:
        cov = pd.to_numeric(df.loc[result.index, GT_COVERAGE_COLUMN], errors="coerce")
        bad = cov.notna() & ((cov < 0) | (cov > 1))
        if bad.any():
            raise ValueError(
                f"[{sheet_name}] coverage_ratio는 0~1이어야 합니다: "
                f"{result.loc[bad, MUNICIPALITY_CODE_COLUMN].tolist()}"
            )
        use = cov.notna() & (cov > 0)
        result.loc[use, "cov"] = cov[use].astype("float64")

    return result[
        [MUNICIPALITY_CODE_COLUMN, "gt_ls", "ls_eval_mask", "cov"]
    ].reset_index(drop=True)


def _prepare_lq_ground_truth(
    gt_path: str | Path,
    event_name: str,
) -> pd.DataFrame:
    """
    한 이벤트의 LQ GT를 정리한다.

    확정 규칙:
    1. jshis_flag가 있으면 우선 사용한다.
       1 -> 1, 0 -> 0, NA -> 0 (평가 포함)
    2. jshis_flag가 없으면 lq_flag를 사용한다.
       1 -> 1, 0 -> 0, NA -> NA (평가 제외)
    """
    df, sheet_name = _read_gt_sheet(gt_path, event_name, "LQ")
    result = _prepare_gt_code_rows(df, sheet_name=sheet_name)

    if GT_LQ_JSHIS_FLAG_COLUMN in df.columns:
        raw = _parse_binary_gt_flag(
            df.loc[result.index, GT_LQ_JSHIS_FLAG_COLUMN],
            sheet_name=sheet_name,
            column_name=GT_LQ_JSHIS_FLAG_COLUMN,
        )

        # JSHIS만 NA를 0으로 확정하므로 모든 실제 시정촌행을 LQ 평가에 사용한다.
        result["gt_lq"] = raw.fillna(0.0).astype("int64")
        result["lq_eval_mask"] = True

    elif GT_LQ_FLAG_COLUMN in df.columns:
        raw = _parse_binary_gt_flag(
            df.loc[result.index, GT_LQ_FLAG_COLUMN],
            sheet_name=sheet_name,
            column_name=GT_LQ_FLAG_COLUMN,
        )

        result["lq_eval_mask"] = raw.notna()
        result["gt_lq"] = raw.fillna(0.0).astype("int64")

    else:
        raise ValueError(
            f"[{sheet_name}] LQ GT에는 '{GT_LQ_JSHIS_FLAG_COLUMN}' 또는 "
            f"'{GT_LQ_FLAG_COLUMN}' 컬럼이 필요합니다."
        )

    # 현재 GT에서 삿포로 10개 구→삿포로시 집계가 필요한 것은 2018 훗카이도 LQ뿐이다.
    if event_name == "2018 훗카이도":
        ward_mask = result[MUNICIPALITY_CODE_COLUMN].isin(SAPPORO_WARD_CODES)
        result.loc[ward_mask, MUNICIPALITY_CODE_COLUMN] = SAPPORO_CITY_CODE

        result = (
            result.groupby(MUNICIPALITY_CODE_COLUMN, as_index=False, sort=False)
            .agg(
                gt_lq=("gt_lq", "max"),
                lq_eval_mask=("lq_eval_mask", "max"),
            )
            .reset_index(drop=True)
        )

    elif result[MUNICIPALITY_CODE_COLUMN].duplicated().any():
        duplicates = result.loc[
            result[MUNICIPALITY_CODE_COLUMN].duplicated(keep=False),
            MUNICIPALITY_CODE_COLUMN,
        ].tolist()
        raise ValueError(f"[{sheet_name}] 중복 시정촌코드가 있습니다: {duplicates}")

    return result[
        [MUNICIPALITY_CODE_COLUMN, "gt_lq", "lq_eval_mask"]
    ].reset_index(drop=True)


def _prepare_eval_ground_truth_df(gt_path: str | Path) -> pd.DataFrame:
    """9개 이벤트의 LS/LQ GT를 하나의 평가용 DataFrame으로 만든다."""
    gt_path = Path(gt_path)
    if not gt_path.exists():
        raise FileNotFoundError(f"GT XLSX를 찾을 수 없습니다: {gt_path}")

    frames: list[pd.DataFrame] = []

    for event_name in EVENTS:
        ls_df = _prepare_ls_ground_truth(gt_path, event_name)
        lq_df = _prepare_lq_ground_truth(gt_path, event_name)

        # 한쪽 GT 시트에 행 자체가 없으면 그 hazard는 0으로 확정하지 않고 평가 제외한다.
        merged = ls_df.merge(
            lq_df,
            on=MUNICIPALITY_CODE_COLUMN,
            how="outer",
            validate="one_to_one",
        )

        merged["gt_ls"] = merged["gt_ls"].fillna(0).astype("int64")
        merged["gt_lq"] = merged["gt_lq"].fillna(0).astype("int64")
        merged["ls_eval_mask"] = merged["ls_eval_mask"].fillna(False).astype(bool)
        merged["lq_eval_mask"] = merged["lq_eval_mask"].fillna(False).astype(bool)
        merged["event_idx"] = EVENT_TO_INDEX[event_name]

        frames.append(merged)

    gt_df = pd.concat(frames, ignore_index=True)

    duplicate_key = gt_df.duplicated(
        subset=["event_idx", MUNICIPALITY_CODE_COLUMN],
        keep=False,
    )
    if duplicate_key.any():
        bad = gt_df.loc[
            duplicate_key,
            ["event_idx", MUNICIPALITY_CODE_COLUMN],
        ].to_dict("records")
        raise ValueError(f"GT에 중복 (event_idx, 시정촌코드)가 있습니다: {bad[:10]}")

    return gt_df


def load_eval_ground_truth(
    gt_path: str | Path,
    model_batch: PilotABatch,
) -> EvalGroundTruthBatch:
    """
    9개 이벤트 GT를 전체 PilotABatch의 실제 모델 행과 정렬해 eval 입력을 만든다.

    매칭 key는 반드시 (event_idx, municipality_code)를 사용한다.
    같은 시정촌코드가 다른 지진 이벤트에 다시 등장할 수 있기 때문이다.

    LS/LQ 둘 중 하나라도 평가 가능한 모델 행만 EvalGroundTruthBatch에 포함한다.
    """
    model_batch.validate()
    gt_df = _prepare_eval_ground_truth_df(gt_path)

    model_df = pd.DataFrame(
        {
            "model_row_idx": list(range(model_batch.batch_size)),
            "event_idx": model_batch.event_idx.detach().cpu().tolist(),
            MUNICIPALITY_CODE_COLUMN: list(model_batch.municipality_code),
        }
    )

    if model_df.duplicated(
        subset=["event_idx", MUNICIPALITY_CODE_COLUMN]
    ).any():
        raise ValueError("PilotABatch에 중복 (event_idx, 시정촌코드)가 있습니다.")

    aligned = model_df.merge(
        gt_df,
        on=["event_idx", MUNICIPALITY_CODE_COLUMN],
        how="left",
        validate="one_to_one",
    )

    # GT 행 자체가 없으면 0으로 확정하지 않고 해당 hazard 평가에서 제외한다.
    aligned["gt_ls"] = aligned["gt_ls"].fillna(0).astype("int64")
    aligned["gt_lq"] = aligned["gt_lq"].fillna(0).astype("int64")
    aligned["ls_eval_mask"] = aligned["ls_eval_mask"].fillna(False).astype(bool)
    aligned["lq_eval_mask"] = aligned["lq_eval_mask"].fillna(False).astype(bool)

    any_eval = aligned["ls_eval_mask"] | aligned["lq_eval_mask"]
    aligned = aligned.loc[any_eval].reset_index(drop=True)

    if aligned.empty:
        raise ValueError("모델 행과 매칭되는 평가 가능한 GT가 없습니다.")

    eval_batch = EvalGroundTruthBatch(
        model_row_idx=torch.tensor(
            aligned["model_row_idx"].to_numpy(dtype="int64"),
            dtype=INDEX_DTYPE,
        ),
        gt_ls=torch.tensor(
            aligned["gt_ls"].to_numpy(dtype="int64"),
            dtype=INDEX_DTYPE,
        ),
        gt_lq=torch.tensor(
            aligned["gt_lq"].to_numpy(dtype="int64"),
            dtype=INDEX_DTYPE,
        ),
        ls_eval_mask=torch.tensor(
            aligned["ls_eval_mask"].to_numpy(dtype="bool"),
            dtype=torch.bool,
        ),
        lq_eval_mask=torch.tensor(
            aligned["lq_eval_mask"].to_numpy(dtype="bool"),
            dtype=torch.bool,
        ),
        event_idx=torch.tensor(
            aligned["event_idx"].to_numpy(dtype="int64"),
            dtype=INDEX_DTYPE,
        ),
        municipality_code=tuple(
            aligned[MUNICIPALITY_CODE_COLUMN].astype(str).tolist()
        ),
    )

    eval_batch.validate()
    return eval_batch


def load_ls_coverage(
    gt_path: str | Path,
    model_batch: PilotABatch,
) -> torch.Tensor:
    """
    후속실험 5: 모델 행 순서대로 LS 판독 범위 비율 cov [B]를 만든다.

    LS 라벨이 있는 행만 실제 cov를 갖고, 라벨이 없는 행(판독 범위 밖 포함)은 1이다.
    log(cov)는 LS prior에 학습하지 않는 고정 오프셋으로 더해진다.
    """
    gt_df = _prepare_eval_ground_truth_df(gt_path)
    model_df = pd.DataFrame(
        {
            "event_idx": model_batch.event_idx.detach().cpu().tolist(),
            MUNICIPALITY_CODE_COLUMN: list(model_batch.municipality_code),
        }
    )
    aligned = model_df.merge(
        gt_df[["event_idx", MUNICIPALITY_CODE_COLUMN, "cov", "ls_eval_mask"]],
        on=["event_idx", MUNICIPALITY_CODE_COLUMN],
        how="left",
        validate="one_to_one",
    )
    labeled = aligned["ls_eval_mask"].fillna(False).astype(bool)
    cov = aligned["cov"].where(labeled, 1.0).fillna(1.0).to_numpy(dtype="float64")
    return torch.tensor(cov, dtype=DTYPE)


def load_eval_inputs(
    stats_path: str | Path,
    usgs_path: str | Path,
    gt_path: str | Path,
    *,
    strict_expected_rows: bool = True,
) -> tuple[PilotABatch, EvalGroundTruthBatch]:
    """
    eval.py가 바로 사용할 모델 입력 + GT 입력을 함께 만든다.

    반환:
    - model_batch: 기존 학습/추론용 PilotABatch
    - eval_gt: 실제 GT가 존재하면서 모델 예측도 가능한 행만 정렬한 EvalGroundTruthBatch
    """
    model_batch = load_pilot_a_batch(
        stats_path,
        usgs_path,
        strict_expected_rows=strict_expected_rows,
    )
    eval_gt = load_eval_ground_truth(gt_path, model_batch)
    return model_batch, eval_gt


if __name__ == "__main__":
    # 아래 경로는 권장 프로젝트 구조 예시다. 실제 위치가 다르면 두 경로만 수정하면 된다.
    stats_file = Path("data/raw/재난프로젝트_시정촌별_통계데이터.xlsx")
    usgs_file = Path("data/raw/재난프로젝트_시정촌별_USGS.xlsx")

    batch = load_pilot_a_batch(stats_file, usgs_file)

    print("batch_size:", batch.batch_size)
    print("y:", tuple(batch.y.shape))
    print("E:", tuple(batch.E.shape))
    print("pgv:", tuple(batch.pgv.shape))
    print("z_wood:", tuple(batch.z_wood.shape))
    print("z_mtn:", tuple(batch.z_mtn.shape))
    print("pi_ls:", tuple(batch.pi_ls.shape))
    print("pi_lq:", tuple(batch.pi_lq.shape))
    print("event_idx:", tuple(batch.event_idx.shape))
    print("obs_mask:", tuple(batch.obs_mask.shape))
    print("municipality_code sample:", batch.municipality_code[:5])

    excluded = load_excluded_rows(stats_file, usgs_file)
    print("\n제외 행 수:", len(excluded))
    print(
        excluded[
            [MUNICIPALITY_CODE_COLUMN, "event", "exclude_reason"]
        ].to_string(index=False)
    )
