# DisasterNet 재구현 · pilotA 비교

Li et al., *DisasterNet: Causal Bayesian Networks with Normalizing Flows for Cascading
Hazards Estimation from Satellite Imagery*, KDD 2023 (pp. 4391–4403) 를 시정촌 단위 DPM으로
재구현하고, pilotA와 **같은 행 · 같은 GT · 같은 `eval.evaluate()`** 로 비교한다.

공식 코드(<https://github.com/SusuXu/DisasterNet>)는 Haiti 데이터 경로가 박혀 있고 정의되지 않은
변수가 있어 그대로는 돌지 않는다. **논문 식을 기준으로 구현**하고, 공식 코드는 세부 설정
(triplet margin, flow 파라미터 범위, β warm-up 등) 확인용으로 참고했다.

## 실행

저장소 루트에서 (원천 xlsx 3개는 평소 위치 `raw/`, `validation/`에 둔다):

```bash
python -m disasternet.compare --dpm raw/DPM_municipality_v5.xlsx
python -m disasternet.smoke_test        # 합성 데이터 점검 (원천 xlsx 없이 동작, 약 4분)
```

결과는 `outputs/disasternet/`:

| 파일 | 내용 |
|---|---|
| `compare_summary.csv` | USGS prior / pilotA / DisasterNet 의 MSE · AUC (pooled, 이벤트 가중평균) |
| `compare_per_event.csv` | 모델 × 이벤트별 MSE · AUC |
| `predictions.csv` | 비교 행의 세 모델 p_ls · p_lq 나란히 + DPM + DisasterNet p_bd |
| `disasternet_params.csv` | 이벤트별 학습 파라미터 |
| `disasternet_loss.csv` | 에폭별 loss / -ELBO / triplet |

## 비교 조건

- 이벤트: DPM이 있는 5개 (2016 구마모토 · 2018 오사카 · 2018 훗카이도 · 2021 후쿠시마 · 2024 노토반도)
- 행: **DPM 행 ∩ PilotABatch 행 ∩ GT 행**. 세 모델 모두 이 행에서만 채점한다.
- GT는 두 모델 모두 학습에 쓰지 않는다 (비지도).
- pilotA는 기본적으로 평소처럼 9개 이벤트 전체로 학습한 뒤 위 행만 뽑아 채점한다.
  `--pilota-train dpm-events` 를 주면 DPM 5개 이벤트 행만으로 학습한다.

## 모델 (논문 3장)

```
α_LS, ε_LS ─▶ z_LS ─┐
                    ├─▶ z_BD ─┐
α_LF, ε_LF ─▶ z_LF ─┘  ▲       ├─▶ x (DPM)
             α_BD, ε_BD┘       │
          z_LS, z_LF ──────────┤          (논문 LF = 우리 LQ)
                     ε_x ──────┘
```

| 구성 | 식 | 논문 |
|---|---|---|
| 노드 | logit p(z_i=1) = Σ w_ji z_j + w_αi α_i + w_εi ε_i + w_0i, BD는 noisy-OR 부모 | 식 1, 9 |
| 변분분포 | r⁰ ~ N(μ, σ²) → planar flow K층 → q_i = mean sigmoid(r^K) | 식 2–6 |
| 관측 | x ~ LogNormal(Σ w_ix z_i + w_0x, w_εx²) | 식 7 |
| 하한 | E log p(z) 는 Jensen 하한 E[-log(1+eˣ)] ≥ -log(1+E[eˣ]) | 식 8–10, 부록 6.2–6.3 |
| triplet | max{0, ‖q_LS−α_LS‖ − ‖q_LS−α_LF‖ + β} + LF 대칭항, β=1 | 식 11 · 공식 코드 |
| 손실 | −Σ_l ELBO_l + λ·triplet, KL β warm-up | 식 12 · 공식 코드 |

- 이벤트마다 따로 학습 (논문과 동일). `--pooled` 로 통합 학습 가능.
- 주요 옵션: `--flows K`(기본 4, 논문은 이벤트별 3/4/8), `--lam-triplet`(기본 1), `--margin`(기본 1).

## 논문 · 공식 코드와 다른 점 (교수님 보고 시 명시)

1. **단위**: 픽셀 → 시정촌 집계 DPM. 지역 가지치기(local pruning)는 불필요해 뺐다.
2. **관측 컬럼**: LogNormal이라 x>0 이어야 한다. 기본은 `DPM_변화25초과비율`
   (시정촌 화소 중 DPM>0.25 비율, 항상 (0,1)). `--dpm-col` 로 바꾸면 0 이하 값은 이동시킨다.
3. **α_BD**: 논문은 건물 fragility curve prior. 자료가 없어 이벤트 안 min-max 정규화 log PGV로 대신.
4. **r⁰ 분산 고정(기본 1)**: 공식 코드도 LS/LF는 USGS 불확실도 지도로 고정, BD는 [1,5]로 묶는다.
   논문 식대로 학습시키면 E log q(r) 항 때문에 분산이 발산해 사후가 상수로 뭉개진다 (합성 데이터 확인).
5. **부호 제약**: w_α, w_ix, w_LS→BD, w_LQ→BD ≥ 0. 논문 · 코드에는 없다. 이벤트당 수십 행이라
   제약 없이 두면 부호가 뒤집힌 해에 빠졌다.
6. **위치별 변분 파라미터**: μ와 flow 파라미터 = 부모에서 amortize한 부분 + 위치별 오프셋.
   공식 코드의 `nn.Linear(batch_size, batch_size)` 가 사실상 하는 일을 명시적으로 쓴 것.
7. **부록 6.3 w_ε²/2 부호**: 공식 코드는 논문과 반대. 수학적으로 논문이 맞아 논문을 따랐다.
8. **XOR 노드(부록 6.1)**: 선택 요소이고 공식 코드에도 없어 생략.
9. **최적화**: 공식 코드 RMSprop lr=1e-5 · 30 epoch (픽셀 수천 개 배치) → 전체 배치 Adam lr=0.01 · 3000 epoch.

## 합성 데이터 점검 결과 (`smoke_test`)

논문 4.4처럼 생성모형 forward로 LS/LQ/BD/DPM을 뽑아 확인 (5개 이벤트 × 60행, 이벤트 가중 AUC).

| 설정 | AUC_LS | AUC_LQ |
|---|---|---|
| USGS prior | 0.826 | 0.819 |
| DisasterNet (K=4, triplet λ=1) — 기본 | **0.941** | **0.880** |
| DisasterNet (K=4, triplet 끔) | 0.815 | 0.821 |

- triplet loss가 없으면 prior를 넘지 못한다 → 논문이 triplet을 식별성 장치로 넣은 이유와 일치.
- K=0(flow 없음)은 이 재구현에서 LS/LQ가 뒤바뀐 해에 빠지기 쉽다 (논문 목적식에 z 엔트로피 항이
  없어 flow가 없으면 q가 끝으로 몰린다). 실제 실험에서 K=0 비교는 결과를 주의해서 볼 것.

## 권장 실험 세트

```bash
python -m disasternet.compare --dpm raw/DPM_municipality_v5.xlsx --tag base
python -m disasternet.compare --dpm raw/DPM_municipality_v5.xlsx --flows 8 --tag k8
python -m disasternet.compare --dpm raw/DPM_municipality_v5.xlsx --lam-triplet 0 --tag notrip
python -m disasternet.compare --dpm raw/DPM_municipality_v5.xlsx --min-base-coh 0.3 --tag coh03
python -m disasternet.compare --dpm raw/DPM_municipality_v5.xlsx --pilota-train dpm-events --tag p5
python -m disasternet.compare --dpm raw/DPM_municipality_v5.xlsx --dpm-col DPM_변화15초과비율 --tag ch15
```
