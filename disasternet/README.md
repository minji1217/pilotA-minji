# DisasterNet 재구현 · pilotA 비교

Li et al., *DisasterNet: Causal Bayesian Networks with Normalizing Flows for Cascading
Hazards Estimation from Satellite Imagery*, KDD 2023 을 시정촌 단위 DPM으로 재구현하고,
pilotA와 **같은 행 · 같은 GT · 같은 `eval.evaluate()`** 로 비교한다.

## 실행

저장소 루트에서 (원천 xlsx 3개는 평소 위치 `raw/`, `validation/`에 둔다):

```bash
# DPM 파일도 raw/에 둔다 (raw/는 .gitignore라 커밋되지 않음)
python -m disasternet.compare --dpm raw/DPM_municipality_v5.xlsx

# 합성 데이터 점검 (원천 xlsx 없이 동작)
python -m disasternet.smoke_test
```

결과는 `outputs/disasternet/`:

| 파일 | 내용 |
|---|---|
| `compare_summary.csv` | USGS prior / pilotA / DisasterNet 의 MSE · AUC (pooled, 이벤트 가중평균) |
| `compare_per_event.csv` | 모델 × 이벤트별 MSE · AUC |
| `predictions.csv` | 비교 행의 세 모델 p_ls · p_lq 나란히 + DPM + DisasterNet p_bd |
| `disasternet_params.csv` | 이벤트별 학습 파라미터 |
| `disasternet_loss.csv` | 에폭별 loss |

## 비교 조건

- 이벤트: DPM이 있는 5개 (2016 구마모토 · 2018 오사카 · 2018 훗카이도 · 2021 후쿠시마 · 2024 노토반도)
- 행: **DPM 행 ∩ PilotABatch 행 ∩ GT 행**. 세 모델 모두 이 행에서만 채점한다.
- GT는 두 모델 모두 학습에 쓰지 않는다 (비지도).
- pilotA는 기본적으로 평소처럼 9개 이벤트 전체로 학습한 뒤 위 행만 뽑아 채점한다.
  `--pilota-train dpm-events` 를 주면 DPM 5개 이벤트 행만으로 학습한다.

## 모델

```
USGS π_LS ─▶ LS ─┐
                 ├─▶ BD ─┐
USGS π_LQ ─▶ LQ ─┘       ├─▶ DPM
             LS, LQ ─────┤
                   ε ────┘

LS, LQ ~ Bern(σ(a·logit π + b))                       (pilotA Prior와 같은 보정식)
BD     ~ Bern(σ(c0 + c_LS·LS + c_LQ·LQ + c_PGV·z(log PGV)))
ε      ~ N(0,1)
DPM    ~ N(w0 + w_LS·LS + w_LQ·LQ + w_BD·BD + w_ε·ε, σ²)     (w, c ≥ 0)
```

- `--inference vi` (기본): 위치별 변분분포 q(LS)q(LQ)q(BD) + planar normalizing flow q(ε) 로
  ELBO를 최대화하는 확률적 변분추론. 논문 방식.
- `--inference exact`: ε를 적분하고 8상태를 정확히 합산 (점검용, pilotA와 같은 방식).
- 이벤트마다 따로 학습(논문과 동일). `--pooled` 로 통합 학습 가능.
- DPM은 그룹(이벤트) 안에서 표준화해 넣는다.

## 논문과 다른 점 · 원문 대조 필요

1. **단위**: 논문은 픽셀, 여기는 시정촌 집계 DPM(`DPM_정규화z` 기본, `--dpm-col` 로 변경).
2. **BD prior**: 건물 피해 prior 자료가 없어 PGV로 대신했다. `--no-bd` 로 BD 노드를 빼는 ablation 가능.
3. **triplet loss 미포함**: 논문은 지구물리 prior로 식별성을 높이는 triplet loss를 쓰는데,
   정확한 식을 확인하지 못해 넣지 않았다. 원문 확인 후 `model.py`의 ELBO에 더하면 된다.
4. **base상관도 < 0.30 (69곳)**: 처리 방침 미정이라 기본은 포함. `--min-base-coh 0.3` 으로 제외 실험.

## 권장 실험 세트

```bash
python -m disasternet.compare --dpm raw/DPM_municipality_v5.xlsx --tag base
python -m disasternet.compare --dpm raw/DPM_municipality_v5.xlsx --inference exact --tag exact
python -m disasternet.compare --dpm raw/DPM_municipality_v5.xlsx --min-base-coh 0.3 --tag coh03
python -m disasternet.compare --dpm raw/DPM_municipality_v5.xlsx --no-bd --tag nobd
python -m disasternet.compare --dpm raw/DPM_municipality_v5.xlsx --pilota-train dpm-events --tag p5
```
