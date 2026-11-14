# 새 채팅 인수인계 (2026-11-10 기준, 이 파일 하나로 이어서 진행)

이 파일은 이전 Claude 세션(2026-10-12 ~ 11-10)의 모든 맥락을 모은 것이다. 새 채팅의 Claude는 **이 파일을 끝까지 먼저 읽고** 시작한다.
저장소: `minji1217/pilotA-minji`, 브랜치 `claude/dazzling-babbage-8c6yd4` (이 파일 위치: 저장소 루트 `HANDOFF_새채팅.md`).

---

## 0. 새 채팅 첫 메시지로 붙여 넣을 문장 (사용자용)

> `minji1217/pilotA-minji` 저장소의 `claude/dazzling-babbage-8c6yd4` 브랜치에 있는 `HANDOFF_새채팅.md`를 먼저 끝까지 읽고, 거기 적힌 작업 규칙대로 이어서 진행해줘.

---

## 1. 사람과 역할

| 누구 | 설명 |
|---|---|
| 사용자 (민지, "언니") | 이 프로젝트 담당 대학원생. 저장소 주인 `minji1217`. 지금 해외 출장 중(11/05 출국)이라 연구실 PC를 직접 못 만짐 |
| 교수님 | 지도교수. 결과 보고 대상. 미팅에서 요청한 것들은 아래 7장 |

- 사용자의 이메일 같은 개인정보는 파일 · 커밋 · 외부 서비스에 넣지 않는다.
- 사용자의 학교 Google 계정은 Google Drive용으로만 쓰였다 (Drive 공유는 사용자가 직접 함).

---

## 2. 작업 규칙 (꼭 지킬 것)

### 2-1. 말하는 방식
- **쉬운 한국어**로. 전문용어는 풀어서. 존댓말(해요체).
- 설명에 **화살표(→, ⇒ 등) 쓰지 말 것** (사용자 요청). 문장으로 풀어 쓴다.
- 교수님께 보고할 내용은 **구조화**(표, 짧은 항목). 말로 길게 풀어 쓰지 말 것.
- 확인된 사실 / 추측 / 모르는 것을 구분해서 답하라고 요청할 때가 있음.
- 시간 보고에서 기계 세부(코어 수 등)를 빼라고 한 적이 있었으나, 그 뒤 교수님이 **컴퓨팅 환경(CPU, GPU, 메모리, OS)을 같이 보고하라**고 하셔서 지금은 환경을 넣는다.
- 시간은 **예상치(환산) 금지, 실측치만**, 초 또는 분 단위로 정확히 (교수님 지시, 11/05).
- 보고서에 "예상치 없음" 같은 말은 넣지 말 것 (사용자 요청).
- 설명은 질문하지 않고 HTML(아티팩트)에 바로 반영하는 편을 좋아함. 단, "답변만 해줘, 코드 건드리지 마"라고 하면 코드 수정 · 새 실험 금지.

### 2-2. 실험 이름 (통일된 이름만 쓸 것)
- 시정촌 단위 10개: `DN`, `SOFT`, `DNU`, `SOFTU`, `DNU_끄기만`, `DNU_분산만`, `SOFTU_끄기만`, `SOFTU_분산만`, `DN_같은분산0.01`, `DN_같은분산0.001`
- 칸(픽셀) 단위 5개: `DN`, `SOFTU`, `SOFTU_끄기만`, `SOFTU_분산만`, `LOCALU` (LOCALU는 칸 단위에만 있음)
- 옛 칸 실험 번호와의 대응: ① baseline = DN, ② soft + U = SOFTU, ③ 끄기만 = SOFTU_끄기만, ④ 분산만 = SOFTU_분산만, ⑤ 끄기 지도 + U = LOCALU
- 폴더 이름은 영어: `DNU_off`=끄기만, `DNU_var`=분산만, `SOFTU_off`, `SOFTU_var`, `DN_cvar01`=같은분산0.01, `DN_cvar001`=같은분산0.001
- pilotA: "pilotA A2" = 지금까지 DisasterNet과 비교해 온 설정 (아래 4장). 공헌 2의 이름은 **"크로스 이벤트"** (교수님 지정).

### 2-3. HTML · 아티팩트 · 커밋 규칙
- HTML 페이지는 만들고 아티팩트로 게시 → 같은 내용을 저장소 `docs/html/`에 복사 → 커밋 · 푸시.
- 아티팩트를 고칠 때는 **반드시 기존 URL로 다시 게시**(새로 만들면 중복 링크가 생김; 예전에 중복을 만들었다가 사용자 승인 받고 지움). 새 채팅에서는 먼저 `Artifact action:"read"`로 읽고, 받은 파일을 고쳐서 `url`을 넣어 게시.
- 원본(편집용) HTML은 이전엔 scratchpad에 있었지만 **새 채팅에는 없음** → `docs/html/*.html`을 원본으로 삼아 고친 뒤 같은 URL로 게시.
- 아티팩트는 비공개. 교수님께 보내려면 사용자가 공유 메뉴에서 공유를 켜야 함 (매번 알려 줄 것).
- 커밋 후 항상 `git push -u origin claude/dazzling-babbage-8c6yd4`. 커밋 메시지 끝에 세션이 주는 attribution 줄을 붙인다. 커밋 메시지 · 코드에 모델 이름 넣지 말 것.
- 사용자가 PR을 요청한 적 없음 → PR 만들지 말 것.

### 2-4. 아티팩트 URL 목록

| 내용 | URL | 저장소 사본 |
|---|---|---|
| 연구일지 (가장 중요, 날짜별 기록) | https://claude.ai/artifact/A8WpZAke9d6uzdkfahqosh | `docs/html/6_연구일지.html` |
| pilotA 공헌점 (교수님 보고용) | https://claude.ai/artifact/SwaAyVd9acpFFPcWH3RDVa | `docs/html/12_pilotA_공헌점.html` |
| 인수인계 1편 (시정촌 단위) | https://claude.ai/artifact/UpGVai4nNwqVRKoLB6wPnu | `docs/html/10_인수인계_1_시정촌단위.html` |
| 인수인계 2편 (칸 단위) | https://claude.ai/artifact/ABVNTp3TEo7D7jB8xfU721 | `docs/html/11_인수인계_2_칸단위.html` |
| 학습·추론 시간 측정 (교수님 보고용) | https://claude.ai/artifact/5zVWTM8QE3voHYbhHUs94b | `docs/html/13_학습추론시간.html` |

- 그 밖의 `docs/html/` 페이지: `1_시정촌_실험기록`, `2_칸단위_실험기록`, `3_원본대우리코드`, `4_설정정리`, `5_USGS값_집계방법`, `7_시정촌대칸단위`, `8_칸단위_부록_시정촌으로모으기`, `9_칸단위_지난기록` (예전 페이지들, 아티팩트 URL은 이 파일에 없음).
- **주의: 공헌점 페이지(12)의 계산 시간은 옛 값**(칸 약 137분 · 48초, pilotA 약 2분)이다. 지금 정답은 6장의 11/07 재측정값. 공헌점 페이지를 다시 쓸 일이 있으면 시간을 고칠 것.

---

## 3. 저장소 · 브랜치 · 파일 위치

### 3-1. 저장소
| 저장소 / 브랜치 | 내용 |
|---|---|
| `minji1217/pilotA-minji` · `claude/dazzling-babbage-8c6yd4` | **작업 브랜치.** DisasterNet 재구현(`disasternet/`), 실행 파일(`*.bat`, `*.sh`), 문서(`docs/`), 결과 일부(`outputs/` 중 강제 추가한 것) |
| `minji1217/pilotA-minji` · `pc-results` | 연구실 PC가 자동으로 올리는 결과. `pc_results/`(옛 칸 실험 예측), `pc_rerun/`(11/05~07 다시 돌리기 결과, 아래 5장) |
| `minji1217/pilotA` · `followup4-lq-J-nolabels` | **pilotA A2 코드** (비교값을 만든 브랜치. 마지막 커밋 `ad44fa2 [Set] 이 브랜치 기본값: 액상화 J + 액상화 라벨 미사용(분리만)`) |

- **두 저장소 모두 공개(public) 저장소다** (GitHub API로 확인, 11/10). 연구 원자료를 더 올릴 때는 반드시 사용자에게 먼저 물어볼 것. 이미 올라간 자료: `raw/local_250m_byevent_usgs_aw.csv.gz`, `raw/usgs_ground_failure/`(USGS 공개 래스터), `pc_inputs/`(시정촌 입력 xlsx 등), `outputs/disasternet/rerun_muni/` 등.
- `raw/`와 `outputs/`는 `.git/info/exclude`로 무시됨 (로컬 설정). 올릴 때는 `git add -f`. 새로 클론하면 이 제외 설정이 없으니 조심.

### 3-2. 새 채팅에 없는 것 (다시 받아야 함)
이전 세션의 클라우드 작업 공간(scratchpad, `raw/` 원자료, pilotA 작업 폴더)은 새 채팅에 **없다**. 필요한 원자료는 사용자에게 다시 올려 달라고 해야 한다. 이전에 사용자가 올린 파일 이름:

| 파일 | 쓰임 |
|---|---|
| `DisasterNet_입력_v5.xlsx` | 시정촌 DisasterNet 입력 통합표 (정답 LS_정답/LQ_정답, USGS LS_prior_평균/최대, LQ_prior_평균/최대, DPM_배경차, base상관도 포함). `pc_inputs/disasternet_input_v5.xlsx`로 저장소에도 있음 |
| `DPM_municipality_v8.xlsx` (v5도 있음) | 시정촌 DPM. 시트: 집계, 궤도배정, 프레임목록, 집계_구규칙, 자료설명 |
| `dpm_250m_v8_slim.csv.gz`, `dpm_250m_v8_base.csv.gz`, `dpm_250m_v8_README.txt` | 칸 DPM (slim: 이벤트·시정촌코드·mesh250·화소수·DPM_배경차·DPM_최댓값; base: base상관도 소수3자리) |
| `local_250m_byevent_usgs.csv` (137MB) | 칸 목록 + 칸 중심점 USGS 값. 이걸로 `usgs_cells_aw.py`가 넓이평균 판을 만듦 |
| `재난프로젝트_시정촌별_USGS.xlsx`, `재난프로젝트_시정촌별_통계데이터.xlsx` | pilotA 입력 (`raw/`) |
| `LS_LF 데이터자료.xlsx` | pilotA 정답 (`validation/`) |
| `muni_usgs_pixel.csv`, `terrain_fractions.csv` | 시정촌 U · soft 입력 (`pc_inputs/`에 있음) |
| PPT `재난팀_1022_1~8.pptx`, `랩미팅_1023` | 발표 자료. 최신은 `1022_8` (실험 목록의 기준) |
| 논문 PDF `3580305.3599807.pdf` | DisasterNet 원 논문 (Li et al., KDD 2023) |
| 회의 녹음 `.m4a` (11/05) | 교수님 미팅. 받아쓰기 요지는 7장 |

- pilotA를 다시 돌리려면: `minji1217/pilotA`의 `followup4-lq-J-nolabels`를 받고, `raw/`에 USGS · 통계 xlsx, `validation/`에 `LS_LF 데이터자료.xlsx`를 넣는다. 시간 측정 스크립트는 `raw/시정촌_면적.csv`도 필요한데 `disasternet/time_compare.py`의 `make_area()`가 USGS xlsx의 `area_km2`로 만든다.

### 3-3. 연구실 PC (사용자 PC, 원격으로 못 만짐)
- 실험 폴더: `C:\work\pilotA-minji-claude-dazzling-babbage-8c6yd4` (브랜치를 받은 폴더, git 저장소 아님)
- 결과 올리는 git 폴더: `C:\work\pilotA-minji-git` (여기서 `pc-results` 브랜치로 강제 push)
- Python: `py -3.10` (Python 3.10.11, PyTorch 2.14.1+cpu, NumPy 2.2.6, pandas 2.3.3). **PyTorch 업데이트 금지** (재현성).
- Google Drive for desktop 설치됨: `G:\내 드라이브`. 학교 계정 Drive 용량 5GB 중 약 1.77GB 남음(11/05) → 큰 파일은 Drive에 안 올림.
- CPU · 메모리 사양은 모름 (사용자에게 물어봐야 함).
- PC 파일 받는 법(cmd, PowerShell 금지 · 인코딩 깨짐): `cd /d C:\work\pilotA-minji-git` → `git fetch origin claude/dazzling-babbage-8c6yd4` → `git show origin/claude/dazzling-babbage-8c6yd4:<경로> > C:\work\pilotA-minji-claude-dazzling-babbage-8c6yd4\<경로>`. 폴더째는 `git -C C:\work\pilotA-minji-git archive <브랜치> <폴더> | tar -xf - -C <실험폴더>`.
- **.bat 파일은 ASCII만** (한글이 있으면 cmd가 줄을 잘못 읽어 "은 was unexpected at this time." 오류로 멈춤. 11/05에 실제로 겪음). 한글 경로가 필요한 일은 파이썬(`disasternet/rerun_sync.py`)에서.
- 실행 중인 .bat 파일을 덮어쓰면 꼬임 → 창을 닫고 바꾼 뒤 다시 실행.
- Windows에는 multiprocessing `fork`가 없음 → `spawn` 사용 (`run_input_v5.py`, `run_muni_unc.py`에 반영됨).

### 3-4. 클라우드 컴퓨터 주의
- 세션마다 **클라우드 가상 컴퓨터가 바뀔 수 있다** (11/05엔 Xeon 2.80GHz, 11/06~07엔 Xeon 2.10GHz, 커널 버전도 바뀜). **시간 측정은 반드시 한 컴퓨터에서 한 번에 전부** 재야 한다. 다른 컴퓨터 숫자를 섞지 말 것. 재기 전에 `lscpu`, `free -h`, `uname -r`로 사양 기록.
- 백그라운드 명령은 최대 2시간 → 긴 작업은 지진 그룹으로 나눠서. 지진 하나 끝날 때마다 결과를 바로 저장하게 짠다.
- 측정 중에는 다른 계산을 돌리지 말 것 (코어 4개뿐).
- 인터넷: huggingface.co, *.hf.co는 사용자가 허용해 둠 (음성 받아쓰기 faster-whisper용).

---

## 4. 모델 설정 요약

### 4-1. DisasterNet (재구현: `disasternet/model.py`)
- 원 논문: Li et al., "DisasterNet: Causal Bayesian Networks with Normalizing Flows for Cascading Hazards Estimation from Satellite Imagery", KDD 2023. 공식 코드 github.com/SusuXu/DisasterNet (아이티 전용 경로 · `batch_size = 8109`가 박혀 있어 그대로 안 돌아감).
- **논문 확인 결과(11/05)**: 아이티 2021, 푸에르토리코 2020, 홋카이도 2018을 **지진마다 따로 학습·평가**. flow 개수 K도 지진마다 3/4/8. 정답 없이 "sensing observation의 marginal likelihood 최대화". 즉 지진 간 공유 파라미터 없음, 예측할 그 지진으로 학습(분리 실험 아님).
- 노드: 산사태(LS) · 액상화(LQ). **BD(건물) 노드는 교수님 지시로 제거**.
- 노드마다 사후분포: r⁰ ~ N(μ, 분산) → planar flow 4개 → sigmoid → 샘플 평균. 학습 중 샘플 8개, 마지막 추론 256개. μ = 6·tanh((Linear(USGS 확률) + 행별 보정값)/6). flow 파라미터도 Linear(α) + 행별 보정값, 0.99·tanh로 묶음.
- 사전확률(KL 항): sigmoid(w_0 + w_a·α + log_frac). w_a는 softplus로 양수. soft는 log_frac로만 들어감(고정값, 학습 안 함). **soft는 μ를 낮추는 게 아니라 사전확률(KL) 항에만 들어간다** (예전에 틀리게 설명했다가 정정, 연구일지 11.04).
- 관측식: log(DPM_배경차 + 2) ~ 정규분포(w_0x + w_LS·q_LS + w_LQ·q_LQ, w_εx²). w_x는 softplus로 양수, w_0x는 제약 없이 지진마다 학습. 노드들이 합쳐지는 곳은 관측식뿐.
- triplet 제약: TripletMarginLoss(q_LS, α_LS, α_LQ) + (q_LQ, α_LQ, α_LS), 가중 1, margin 1.
- 학습: Adam lr 0.01, 3000 epoch, β warm-up = min(0.01 + epoch/3000, 1). 손실 = Σ(−rec + β·KL) + triplet.
- 끄기(keep): q를 0으로 고정하고 그 노드를 prior · 관측 항에서 뺌.
- U(USGS 불확실도): 끄기(불확실도 0 또는 999인 곳) + r⁰ 분산 = 그 곳의 USGS 불확실도. 끄기 기준은 **논문 기준(`--off-rule paper`) = 불확실도 0·999만** (확률 0은 안 봄).
- soft(시정촌): frac_LS = 경사 10° 이상 비율, frac_LQ = 경사 5° 미만 비율, 1% 미만은 1%(로짓 −4.6 = ln 0.01). 5~10° 칸은 두 비율 어디에도 안 들어감(중앙값 13.5%, 최대 64%).
- soft(칸, `--soft --soft-mid`): 비탈 칸 액상화 · 평지 칸 산사태 로짓 −4.6, **5~10° 중간 칸은 둘 다 그대로**(고친 soft). LOCALU의 끄기 지도: 10° 이상 = 비탈(액상화 끔), 5° 미만 = 평지(산사태 끔), 5~10°는 둘 다 켬.
- 시정촌 U 값: 시정촌 안에 중심점이 든 USGS 불확실도 칸들의 단순 평균(0 · 999 제외). 켜진 칸이 0개면 끔 (`muni_usgs_pixel.csv`, 산사태 5곳 · 액상화 11곳 꺼짐).
- 칸 단위: 250 m 칸(1/4 지역메시, 경도 1/320° × 위도 1/480°, 약 232 × 281 m). 40 m DPM 화소를 화소 중심으로 칸에 배정해 평균(칸당 중앙 41개). USGS는 칸과 겹친 넓이 비율로 평균(아래 4-3). 물 칸 제외: 국토수치정보 토지이용 100 m에서 하천·호소·해수역, 칸 안 중심점의 50% 넘게 물이면 물 칸(약 2.5%). 시정촌 값 = 칸 사후확률의 **최댓값**.
- 칸 단위 학습도 칸 단위(시정촌으로 학습하고 칸으로 추론하는 게 아님, 교수님 확인 10/20).
- 평가: 지진 × 재해별 AUC, **채점 시정촌 수로 가중평균**. 한쪽 정답만 있는 조합은 제외. DisasterNet은 seed 0~4 평균.

### 4-2. DPM (v8, 사용자가 만든 자료)
- 화소 DPM = base 상관도 − event 상관도 (Sentinel-1 IW, HyP3 GAMMA INT40, 40 m). 유효 화소 = base, event 둘 다 > 0 (육지 마스크 기록 없음).
- 프레임 16장, 전부 하강. 시정촌마다 가장 많이 덮는 궤도 1개만(궤도 안 섞음). 같은 궤도 프레임이 겹치면 프레임별로 배경차를 먼저 구하고 화소 수로 가중 합성.
- DPM_배경차 = 칸(시정촌) 평균 − 그 프레임 전체 유효화소 중앙값. 프레임 중앙값: 구마모토 163-481 +0.1167, 163-486 +0.0869 (나머지 지진은 −0.059 ~ +0.016). 구마모토만 촬영 간격 24일(나머지 12일).
- 자료설명 기록: 보정 이유 "한 이벤트의 시정촌들이 서로 다른 프레임에서 나오기 때문", 효과 노토 +0.065 · 훗카이도 +0.041(**DPM만 AUC**), 프레임 vs path 단위는 미결(0.5791 대 0.5765, 교수님 확인 대기), "base상관도 0.30 미만은 측정 실패에 가까움 — 제외 아니라 플래그만"(v8), v5는 "(69곳) 처리 방침 미정". **0.4 기준은 v5 · v8 어디에도 없음.**
- 칸 DPM에서 base는 쓰지 않음. 물 칸도 유효화소가 있음(중앙 40개, base 중앙 0.306) → base>0으로 물이 안 걸러지므로 물 칸 제외가 필요.

### 4-3. 칸별 USGS 넓이평균 판 (`raw/local_250m_byevent_usgs_aw.csv.gz`)
- 만든 코드: `disasternet/usgs_cells_aw.py` (11/07 정리, 기존 파일과 차이 최대 3e-21). MD5 `3ea01292967f51cdcef5cf92bbacfe6d`.
- 규칙: 칸을 10 × 10 = 100점으로 나눠 각 점이 든 USGS 래스터 칸 값을 평균. 범위 밖 · nodata · 값 ≥ 999는 빼고, 전부 빠지면 999. 값 0은 평균에 넣음. 확률 · 불확실도 각각 같은 규칙. 행과 다른 열은 중심점 판과 같음(물 칸 포함).
- 래스터: `raw/usgs_ground_failure/<이벤트ID>/` (구마모토 us20005iis, 오사카 us1000eu1c, 훗카이도 us2000h8ty, 후쿠시마 us6000dher, 노토 us6000m0xl). 산사태 jessee_2017_model(없으면 2018), 불확실도 jessee_2017_std(없으면 jessee_2018_beta_sigma); 액상화 zhu_2017_general_model, 불확실도 zhu_2017_general_std(없으면 _beta_sigma).
- 중심점 판(`usgs_cells.py`, `local_250m_byevent_usgs.csv`)과 다름. 칸 실험에 쓰는 건 넓이평균 판.

### 4-4. pilotA A2 (`minji1217/pilotA` 브랜치 `followup4-lq-J-nolabels`)
- 9개 지진, 418행(지진 × 시정촌). 분리(holdout): 평가 지진 하나를 빼고 나머지로 학습, 뺀 지진 예측.
- 실행: `python3 train.py --holdout-event "<지진>" --lam-gamma 10 --tag "<태그>"` (브랜치 기본값이 A2). 파이썬 안에서는 `T.train(batch=..., seed=0, epochs=3000, lr=0.02, lam_gamma=10, area_mode="fixed-cov-mtn", lam_kappa=10, mtn_standardize="train", use_labels=True, lq_labels=False, lq_b_only=True)`.
- 산사태 사전확률(E3): log p̄ + log k + log cov + κ·산지. 학습하는 건 κ 하나(≈0). 액상화 사전확률(J): log p̄ + b + (log k − 평균). 학습하는 건 b 하나. 액상화는 USGS **평균** 사용(DisasterNet 비교의 USGS 기준선은 액상화 **최대**).
- 라벨: 학습 지진의 **산사태 라벨만** 사용. 액상화 라벨 안 씀. 뺀 지진 라벨은 절대 안 씀.
- α_e(지진 절편): 뺀 지진의 피해 건수로 LBFGS 맞춤(E3, 정답 미사용). E1은 0, E2는 학습 지진 평균.
- 피해 모델: 음이항(NB) 6채널(노출 E, log PGV, 목조 비율), 숨은 상태 4개를 샘플링 없이 정확히 합산. **seed와 무관하게 결과가 완전히 같음**(무작위 요소 없음, 초기값 고정; seed 0~4 확인).
- DPM은 쓰지 않음 (행정 자료 + USGS만). 그래서 DPM 없는 지진도 학습 · 예측 가능.
- 결과 파일: `outputs/holdout_detail{_lq}_<태그>_holdout-<지진>.csv` (`p_ls_alpha_fit`, `p_lq_alpha_fit`가 비교값).

---

## 5. 결과 숫자 (보고에 쓰는 값)

### 5-1. DPM 있는 5개 지진 평가, 전체 · 산사태 · 액상화 AUC
채점 시정촌: 산사태 49곳(양성 40, 음성 9: 구마모토 26/3, 오사카 5/2, 훗카이도 9/4), 액상화 151곳(양성 74, 음성 77). 합 200.

| 모델 | 전체 | 산사태 | 액상화 |
|---|---|---|---|
| USGS 기준선 (산사태 평균, 액상화 최대, 학습 없음) | 0.854 | 0.729 | 0.895 |
| pilotA A2 (8개 지진으로 학습) | 0.855 | 0.872 | 0.850 |
| pilotA A2 (DPM 있는 5개 지진만, 나머지 4개로 학습) | 0.862 | 0.887 | 0.854 |
| DisasterNet 칸 LOCALU (최고) | 0.808 | 0.790 | 0.813 |
| 칸 SOFTU_끄기만 | 0.796 | 0.786 | 0.799 |
| 칸 SOFTU | 0.785 | 0.773 | 0.789 |
| 칸 SOFTU_분산만 | 0.758 | 0.794 | 0.747 |
| 칸 DN | 0.704 | 0.728 | 0.696 |
| 시정촌 SOFTU_분산만 (시정촌 최고) | 0.751 | 0.792 | 0.738 |
| 시정촌 SOFTU | 0.738 | 0.791 | 0.721 |
| 시정촌 SOFTU_끄기만 | 0.722 | 0.770 | 0.706 |
| 시정촌 SOFT | 0.710 | 0.777 | 0.688 |
| 시정촌 DNU_분산만 | 0.681 | 0.610 | 0.705 |
| 시정촌 DNU | 0.675 | 0.621 | 0.693 |
| 시정촌 DN_같은분산0.01 | 0.657 | 0.579 | 0.682 |
| 시정촌 DNU_끄기만 | 0.638 | 0.538 | 0.670 |
| 시정촌 DN_같은분산0.001 | 0.616 | 0.540 | 0.641 |
| 시정촌 DN | 0.614 | 0.512 | 0.647 |
| DPM만 (점수로 바로) | 0.596 | 0.689 | 0.566 |

- 칸 추가 채점(새 학습 아님): 시정촌 값을 평균으로 모으면 전체 −0.029 ~ −0.085. 산사태 평균 + 액상화 최대 섞기는 LOCALU 0.811(산사태 0.805, 액상화 0.813).
- seed 안정성: 시정촌은 SOFTU_분산만이 seed 5개 모두 1등. 칸은 seed마다 1등이 바뀌지만 LOCALU가 평균(0.808)과 최악 seed(0.795) 모두 가장 높음. 칸 SOFTU_분산만 seed 0의 노토 액상화 0.35(확률 포화 + 끄기 없음)가 흔들림 원인.
- 칸 vs 시정촌: 끄기를 쓰는 SOFTU · SOFTU_끄기만은 칸이 0.05~0.07 높음.
- 지진별 pilotA(8개 / 4개 학습): 산사태 구마모토 0.846/0.859, 오사카 1.000/1.000, 훗카이도 0.861/0.889; 액상화 구마모토 0.949/0.957, 훗카이도 0.913/0.918, 후쿠시마 0.667/0.667, 노토 0.768/0.770.
- 지진별 시정촌 SOFTU_분산만 / 칸 LOCALU / USGS / DPM만: 산사태 구마모토 0.838/0.803/0.641/0.731, 오사카 0.980/0.980/1.000/0.400, 훗카이도 0.589/0.661/0.778/0.750; 액상화 구마모토 0.823/0.918/0.947/0.601, 훗카이도 0.716/0.608/0.926/0.468, 후쿠시마 0.767/0.533/0.500/0.667, 노토 0.695/0.907/0.887/0.594.

### 5-2. pilotA 세 실험 정리 (헷갈림 주의, 사용자가 여러 번 물음)
| | 매 회차 학습 | 추론 · 채점 | 회차 | 전체 | 산사태 | 액상화 |
|---|---|---|---|---|---|---|
| ① 지금까지 비교값 | 뺀 지진 외 8개 (DPM 없는 4개 포함) | DPM 있는 5개 | 5번 | 0.855 | 0.872 | 0.850 |
| ② DPM 지진만 학습 | 뺀 지진 외 4개 (DPM 있는 것만) | DPM 있는 5개 | 5번 | 0.862 | 0.887 | 0.854 |
| ③ 9개 전체 평가 | 뺀 지진 외 8개 | 9개 모두 | 9번 | 0.836 | 0.905 | 0.804 |

- "9개로 분리"는 9개를 한꺼번에 학습하는 게 아님. 매 회차 8개로 학습. 9개를 각각 돌려도 5개만 평가하면 ①과 똑같은 값(분리 실험은 회차끼리 독립).
- ③ USGS 기준선: 전체 0.825, 산사태 0.797, 액상화 0.837. 채점 265 = 산사태 83 + 액상화 182 (418행 중 정답 있는 행: 산사태 125, 액상화 229, 하나라도 278, 둘 다 없음 140; 한쪽 정답만인 조합 제외 산사태 −42, 액상화 −47). 추가 지진: 산사태 돗토리 0.933(USGS 0.904), 니가타2004 1.000(0.875); 액상화 돗토리 0.857(0.571), 니가타2004 0.483(0.550). 니가타2007 · 이와테미야기는 양성만 있어 제외.
- 녹음상 교수님 요청은 ②(DPM 지진만 학습)였다. 사용자가 미팅 때 "기존이 DPM 있는 지진만으로 학습"이라고 잘못 말씀드렸고, 실제 기존 값은 ①(8개 학습). 정정 보고함.
- 해석: DPM 없는 지진을 학습에 더해도 성능은 오르지 않음(0.862 대 0.855) → "적용 범위의 장점"으로 말할 것.

### 5-3. 학습 · 추론 시간 (최종, 11/07 재측정, 보고 페이지에 반영됨)
같은 컴퓨터, 지진 5개 합계, seed 0, 3000 epoch 실측. 다른 작업 없이 하나씩 차례로. 자료 읽기 시간 제외. 추론 = 학습 뒤 사후확률 계산(DisasterNet은 칸/시정촌마다 샘플 256개 평균, pilotA는 지진별 α_e 맞춤 포함).

| 모델 | 학습 | 추론 | AUC |
|---|---|---|---|
| DisasterNet 칸 LOCALU | 5,405.8초 (90.1분) | 40.13초 | 0.808 |
| DisasterNet 시정촌 SOFTU_분산만 | 61.0초 (1.0분) | 0.013초 | 0.751 |
| pilotA A2, 8개 지진으로 학습 | 67.9초 (1.1분) | 0.29초 | 0.855 |
| pilotA A2, DPM 있는 5개 지진만 학습 | 59.8초 (1.0분) | 0.31초 | 0.862 |

지진별(초) — 칸 LOCALU 학습/추론: 구마모토(96,404칸) 521.0/5.06, 오사카(44,811) 263.1/2.12, 훗카이도(294,110) 1,560.4/9.06, 후쿠시마(269,604) 1,331.4/10.97, 노토(320,110) 1,729.9/12.92. 시정촌 학습 11.7~12.9, 추론 0.002~0.004. pilotA 8개 학습 12.7~14.8, 5개만 11.3~12.7.

측정 환경: 클라우드 가상 컴퓨터(Claude 실행 환경), CPU Intel Xeon Processor @ 2.10GHz 4코어(코어당 1스레드), GPU 없음(CPU만), 메모리 15GB, Ubuntu 24.04.5 LTS (Linux 6.18), Python 3.13.16, PyTorch 2.14.1, NumPy 2.5.3, pandas 3.0.5, 계산 스레드 4.

- 칸 대비 pilotA(8개): 학습 약 80배, 추론 약 138배 빠름. 시정촌 DisasterNet과 pilotA는 시간이 비슷 → 시정촌 대비 장점은 정확도와 크로스 이벤트.
- 이전 값들(쓰지 말 것): 11/05 첫 컴퓨터(2.80GHz)에서 칸 8,025.7초 · 47.14초, 시정촌 113.4초 · 0.018초, pilotA 117.0초 · 1.32초 (`outputs/disasternet/time_localu_full_machine1/`, `time_measured_20261105.csv`). 그 전 "약 137분 · 48초"는 30 epoch × 100 예상치. 교수님이 말씀하신 "1시간 25분"은 연구실 PC에서 지진 5개 동시 실행한 학습 + 추론 벽시계 시간.
- 사용자가 교수님께 보낼 때는 보고 페이지(13)를 공유하거나 결과 표 + 컴퓨팅 환경 두 부분만 보내기로 함.
- 측정 파일: `disasternet/time_measure/` (`time_cells_localu.py`, `time_muni.py`, `time_pilota.py <pilotA폴더> [all8|dpm4]`), 결과 `outputs/disasternet/time_measured_20261107.csv`, `time_measured_env_20261107.txt`, `outputs/disasternet/time_localu_full/`(loss 포함).
- "pilotA 지진 분리 5번" 뜻: 평가 지진 5개를 하나씩 빼고 학습 · 예측한 5번의 합. 보고 페이지에서는 이 말 대신 "8개 지진으로 학습 / DPM 있는 5개 지진만 학습"으로 쓰고 주석으로 설명.

---

## 5-4. RQ4 Ablation (11/10 완료, 논문 초안 TRACE v1.1의 RQ4)
- 논문 초안: "TRACE: Inferring Cascading Earthquake Hazards from Web-Published Administrative Damage Statistics" (TRACE = pilotA). RQ1~RQ5 중 RQ4만 하기로 함 (사용자 지시 11/10).
- 스크립트: `pilota_ablation/run_ablation.py <pilotA 폴더> [설정...]`, `score_ablation.py <pilotA 폴더> [결과.csv]`. pilotA 모델 코드는 안 고치고 입력 · 부품만 바꿈. 결과 `pilota_ablation/ablation_result.csv`, `ablation_result_지진별.csv`.
- 기준 TRACE = A2(평가 지진 외 8개로 학습). LOEO 9개 지진 모두 돌리고, DPM 있는 5개 / 9개 전체로 채점.

| 설정 | 5개 전체 | 산사태 | 액상화 | 9개 전체 |
|---|---|---|---|---|
| TRACE | 0.855 | 0.872 | 0.850 | 0.836 |
| prior 균등 (0.5) | 0.602 | 0.725 | 0.562 | 0.580 |
| 흔들림 항 제거 | 0.862 | 0.836 | 0.871 | 0.855 |
| 포아송 | 0.524 | 0.608 | 0.498 | 0.538 |
| 노출 offset 제거 | 0.834 | 0.903 | 0.811 | 0.820 |
| 이벤트별 단독 학습 | 0.841 | 0.880 | 0.829 | 0.822 |

- 설정 정의: prior 균등 = 재해마다 모든 시정촌에 같은 사전확률 (산사태 · 액상화 똑같이). 세 방식: 평균(학습 시정촌 원래 사전확률 κ=0·b=0의 평균으로 고정), 학습(그 값에서 상수 학습), 0.5 고정. 본문 추천 0.5. 처음 "평균"은 USGS 확률 0인 행 때문에 산사태가 거의 0으로 잘못 계산됐었음(고침, 커밋 8992331). 흔들림 항 제거 = PGV를 1로(β·log PGV = 0). 포아송 = 음이항 대신 포아송 log-PMF. 노출 offset 제거 = E를 1로(인명은 인구, 주택은 일반세대 수였음). 이벤트별 단독 학습 = 예측할 지진만으로 라벨 없이 학습.
- 추가(11/10): prior 균등 세 방식(평균 · 학습 · 0.5) 비교와 기준 학습 "DPM 있는 5개 지진만"(`--dpm5`, 결과 `ablation_result_dpm5.csv`). 5개만 학습 기준 전체 AUC: TRACE 0.862, prior 균등 평균 0.666 · 학습 0.638 · 0.5 0.666, 흔들림 항 제거 0.863, 포아송 0.604, 노출 offset 제거 0.859, 이벤트별 단독 0.841. 8개 학습 기준 prior 균등 평균 0.629 · 학습 0.591 · 0.5 0.602 (9개 평가 0.602 · 0.576 · 0.580).
- 해석: prior · 음이항이 크게 기여, 노출 · 크로스이벤트는 작게 기여, 흔들림 항은 정확도 기여 없음(빼면 오름). 연구일지 11.10에 기록.
- 초안과 실제 구현의 차이(사용자에게 알림): 노출은 "세대 수"가 아니라 인명은 인구 · 주택은 일반세대 수; 과거 라벨은 산사태만 사용; 새 지진에는 α_e 하나를 피해 건수로 맞춤; RQ5의 (a, b)는 지금 모델에 그대로 없음(산사태 κ, 액상화 b); RQ2는 TRACE 추론만 대 DisasterNet 학습 + 추론으로 다시 짜야 함; 기준선 USGS 액상화는 최대, TRACE 입력은 평균.

## 5-5. 산사태 cov 항 점검 (11/14 완료, 교수님 지시)
- 교수님 지시: "학습은 변함없음 (현재 학습되어있는 모델 그대로). 테스트 시, 테스트 대상 지진에 대한 cov 텀을 0으로 둔 상태에서 정확도 확인". 베이스라인은 ②(USGS 평균 + 면적항 log k, 산사태 DPM 5개 0.783)로 하기로 함.
- 이유: 지금 LOEO 결과는 평가 지진의 실제 cov(사후 판독 범위 비율)를 써서 얻은 값. cov는 지진 뒤 판독이 끝나야 알 수 있음.
- 스크립트: `pilota_ablation/run_cov0.py <pilotA 폴더> [--dpm5]`, `score_cov0.py <pilotA 폴더> [결과.csv]`. 결과 `pilota_ablation/cov0_result.csv`. 학습 지진의 cov는 그대로 두고, 평가 지진 행만 log cov = 0으로 바꿔 α_e 맞춤과 사후확률을 다시 계산.
- 평가 지진에서 cov < 1인 시정촌 수: 구마모토 29, 훗카이도 10, 노토 5 (노토는 산사태 정답이 한쪽뿐이라 채점 제외), 오사카 · 후쿠시마 0.
- 원래 값(cov 사용)이 예전과 똑같이 나옴: 8개 학습 산사태 0.872, 4개 학습 0.887.

| 산사태 AUC (DPM 5개, 채점 49) | 8개 학습, cov 원래 | 8개 학습, cov 0 | 4개 학습, cov 원래 | 4개 학습, cov 0 |
|---|---|---|---|---|
| 사후확률 (AdminDER) | 0.872 | 0.784 | 0.887 | 0.784 |
| 사전확률만 | 0.879 | 0.783 | 0.864 | 0.783 |
| 구마모토 (29) 사후 | 0.846 | 0.846 | 0.859 | 0.846 |
| 오사카 (7) 사후 | 1.000 | 1.000 | 1.000 | 1.000 |
| 훗카이도 (13) 사후 | 0.861 | 0.528 | 0.889 | 0.528 |
| 액상화 사후 (참고, 151) | 0.850 | 0.849 | 0.854 | 0.855 |
| 전체 (200) | 0.855 | 0.833 | 0.862 | 0.837 |

- 해석: cov를 빼면 산사태 0.784로 베이스라인 ② 0.783과 거의 같음. 산사태 정확도 오름(0.783에서 0.872 · 0.887)은 거의 다 평가 지진의 cov에서 나옴. 떨어짐은 훗카이도에 몰림(0.86 · 0.89에서 0.53). κ가 거의 0이라 cov 0일 때 사전확률 순위가 ②와 같음. 액상화는 거의 그대로.
- 실행 환경 메모: 새 클라우드 컴퓨터에 torch가 없어 PyPI에서 torch 2.14.1(+cu130, CPU로 실행)을 깔았음 (download.pytorch.org는 막힘). sklearn · openpyxl · matplotlib · scipy도 새로 깔았음. `raw/시정촌_면적.csv`는 `disasternet/time_compare.py`의 `make_area()`로 만듦.

## 6. 파라미터 · loss 다시 돌리기 (11/05~11/07, 완료)

교수님이 파라미터와 loss를 요구하실 때를 대비해, PPT 1022_8의 모든 실험을 seed 0~4로 다시 돌려 **학습된 파라미터와 epoch별 loss를 CSV로** 남겼다. 예전에는 칸 실험의 파라미터를 학습 후 지웠음(중간 저장 파일 삭제 설계였음; 사용자에게 사과함).

### 6-1. 결과 상태
| | 어디서 | 예전(PPT) 결과와 비교 | 파일 위치 |
|---|---|---|---|
| 시정촌 10개 × seed 5 | 클라우드 (원래 클라우드에서 돌린 실험) | **250건 모두 완전히 같음** | 브랜치 `outputs/disasternet/rerun_muni/<실험>/` (params_seed*.csv, rowparams_seed*.csv, loss_seed*.csv, predictions_seed*.csv, rerun_compare.csv) |
| 칸 5개 × seed 5 | 연구실 PC (원래 PC에서 돌린 실험) | **125건 모두 칸 단위로 완전히 같음** (11/07 08:12 완료) | PC `outputs\disasternet\rerun_cells\<실험>\` 전부; GitHub `pc-results` 브랜치 `pc_rerun/cells/<실험>/`(loss, params, muni 예측, 비교표); Drive `내 드라이브\pilotA_rerun\cells\` |
| 시정촌 10개 (PC에서도 돌림) | 연구실 PC | **다름** (라이브러리 버전 차이) → **보고에는 클라우드 결과를 쓸 것** | PC `rerun_muni`, Drive `pilotA_rerun\muni`, GitHub `pc_rerun/muni` |

- 칸별 파라미터 `rowparams_<지진>_seed*.csv.gz`(약 6GB)와 칸 예측 `cells_*.csv.gz`는 **PC에만** 있다 (GitHub 용량 · Drive 5GB 한도 때문). 귀국 후 가져가야 함.
- 파일 형식: params = group(지진)/param/index/value (시정촌은 kind raw/derived도); rowparams = 행마다 flow.LS.mu_local, flow.LS.flow_local[0..11], LQ도 같음; loss = group/epoch/loss/neg_elbo/triplet/beta.
- 재현 근거: PC 오사카 LOCALU seed 0 시험(`run_repro_test.bat`, 11/05) 44,811칸 완전 일치 → 전체 재실행.
- PC 로그의 Traceback 4개(PermissionError, multiprocessing)는 11/05 시정촌 부분에서 창을 닫았다 다시 시작할 때 생긴 것. 문제 아님.
- 실행 파일: `run_rerun_all.bat`(PC, ASCII, 1부 시정촌 10개 → 2부 칸 5개 seed별, seed마다 `rerun_compare` + `rerun_sync`로 Drive · GitHub 업로드), `run_rerun_muni.sh`(클라우드 시정촌), `disasternet/rerun_compare.py`, `disasternet/rerun_sync.py`, `pc_inputs/`(시정촌 입력 + 비교용 예전 예측).
- 실험별 설정(PC 칸): 공통 `--local raw\local_250m_byevent_usgs_aw.csv.gz --off-rule paper --jobs 5 --threads 2`; LOCALU `--local-off --unc`; DN 추가 없음; SOFTU `--soft --soft-mid --unc`; SOFTU_끄기만 `+ --unc-mode off`; SOFTU_분산만 `+ --unc-mode var`. 예전 결과 폴더: `cells_max_local_unc_paper_aw`, `cells_max_aw`, `cells_max_softunc_paper_aw_mid`, `cells_max_softunc_off_paper_aw_mid`, `cells_max_softunc_var_paper_aw_mid`.
- 실험별 설정(시정촌): DN = `run_input_v5 --no-bd`; SOFT = `+ --terrain raw/terrain_fractions.csv`; DNU · SOFTU 계열 = `run_muni_unc --usgs-muni raw/muni_usgs_pixel.csv --no-bd [--terrain ...] [--unc off|var]`; 같은분산 = `run_muni_unc --no-bd --unc none --const-var 0.01|0.001` (예전 폴더 `nobd_DN`, `nobd_SOFT`, `px_DNU`, `px_DNU_off`, `px_DNU_var`, `px_SOFTU`, `px_SOFTU_off`, `px_SOFTU_var`, `nobd_DN_cvar01`, `nobd_DN_cvar001`).
- 코드 변경(11/05): `model.py`에 `model_param_tables`, `save_fit_tables`, FitResult에 raw_params/row_params; `run_cells_max.py`가 params/rowparams/loss 저장 + 중간저장에 loss 기록 포함; 계산 자체는 그대로(바꾸기 전 · 후 결과 동일 확인).
- 교수님께 보낼 것: 시정촌은 클라우드 결과(GitHub 브랜치 `outputs/disasternet/rerun_muni`), 칸은 Drive `pilotA_rerun\cells` 또는 GitHub `pc_rerun/cells`. 보기 좋은 정리본(실험별 파라미터 한 표 + loss 곡선)은 **아직 안 만듦** (제안만 함).

---

## 7. 교수님 요청과 상태

| 날짜 | 요청 | 상태 |
|---|---|---|
| 11/05 미팅(녹음) | 평가 방식: pilotA는 그 지진 빼고 학습해 예측, DisasterNet은 그 지진 하나 — "평가 세팅은 이렇게 해야 동일" | 지금 방식 유지 |
| 11/05 미팅 | DisasterNet도 여러 지진을 섞어 배울 수 있지 않나 → 논문 확인, 논문도 사건별이면 "논문 방식을 따랐다"고 쓰기 | 확인 완료(사건별) |
| 11/05 미팅 | pilotA를 DPM 있는 지진만으로 학습한 것도 보기 (녹음 04:55~05:58) | 완료(②, 0.862) |
| 11/05 미팅 | 공헌 2 이름 "크로스 이벤트" | 반영 |
| 11/05 | 학습/추론 시간 비교 | 완료 (5-3) |
| 11/05 메일 | 시간은 예측 말고 실측, 초·분 단위, Claude로 돌렸으면 정확한 시간 보고, 컴퓨팅 자원(CPU, GPU, memory, OS) 공유 | 완료 (11/07 재측정, 보고 페이지 13) |
| 11/05 메일 | 시정촌 버전 DisasterNet과의 학습/추론 시간 비교도 | 완료 |
| 11/05 메일 | "끈다" · DNU · SOFTU 개념 질문 | 사용자가 답장함 (초안 3번 수정) |
| (대비) | 파라미터 · loss | 완료 (6장), 보낼 정리본은 미작성 |
| 앞으로 갈 방향(사용자 언급 11/07) | pilotA를 **DPM 있는 5개 지진으로 학습하는 방향**으로 갈 것 같음 | 시간 · AUC 둘 다 보고 페이지에 넣음 |

- 사용자가 교수님께 보낸 정정 보고 요지(11/05): 기존 pilotA 결과는 8개(DPM 없는 4개 포함)로 학습한 것이었고, DPM 있는 지진만(4개) 학습 결과를 새로 돌려 0.862, 둘 다 DisasterNet 0.808보다 높음. 논문도 지진별 독립 최적화.
- 11/05 녹음 받아쓰기(faster-whisper, 받아쓰기 오류 많음: "디지스터넷/디프리즘" = DisasterNet, "가정평균" = 가중평균). 원문은 이전 scratchpad에만 있었음(없어짐).

---

## 8. pilotA 공헌점 (교수님께 보낸 구조, 페이지 12)
1. **위성 영상 없이, 더 가볍고 더 정확함**: AUC 표(pilotA 0.855, DisasterNet 칸 최고 0.808, USGS 0.854), 비용 표(입력: DisasterNet은 SAR 변화지도 40 m 화소를 250 m 칸으로 평균 + 칸별 경사 · USGS · 토지이용 / pilotA는 시정촌 표 418줄; 학습 단위 칸 4.5만~32만 개 대 시정촌 418개), 라벨 사용 차이 근거(산사태 성능 대부분은 학습 없는 사전확률에서: USGS 평균 0.729, 시정촌 크기 항 추가 0.783, 판독 범위 보정 추가 0.864; κ ≈ 0; 액상화는 라벨 미사용).
2. **크로스 이벤트**: 지난 지진에서 배운 공유 파라미터를 새 지진으로 옮김. 새 지진에는 α_e 하나만 피해 건수로 맞춤(정답 미사용). DisasterNet은 공유 파라미터 없이 지진마다 처음부터.
- 계획(미완): bootstrap 신뢰구간, `--no-labels` 실행, 전이 효과 직접 비교.
- 주의: "정답 없이"는 DisasterNet도 비지도라 pilotA만의 장점이 아님(사용자 지적). pilotA는 학습 지진의 산사태 라벨을 씀.
- 비교 조건: 7개 조합 중 6개에서 pilotA 우세(예외 노토 액상화).

---

## 9. 연구일지 주요 날짜 (`docs/html/6_연구일지.html`)
10.23(seed 의미, 시정촌 대 칸, LOCALU 최종, 실험 이름 통일, 노토 0.35 원인, 인수인계), 10.23(b), 10.28(자료 크기 맞추기), 10.30(pilotA E3 · J, 지형 처리 점검), 11.02(pilotA 공헌점), 11.04(soft 설명 정정: −4.6 = ln 0.01, 사후확률이 나오는 지점), 11.05(교수님 회의, 시간, DPM 지진만 pilotA, seed 무관, 9개 지진 분리, 세 실험 한눈에, 논문 확인, 시정촌 시간, 11.07 재측정 시간 표로 갱신). 다음 기록은 11.07 이후 일(파라미터 다시 돌리기 완료, USGS 넓이평균 파일 정리, 시간 재측정)을 새 날짜로 추가하면 됨 (**아직 연구일지에 안 넣은 것**: 6장 결과의 완료 상태, 4-3 넓이평균 파일 정리).

---

## 10. 남은 일 · 할 수 있는 일 (사용자가 요청하면)
1. 교수님께 보낼 **파라미터 · loss 정리본** (시정촌 = 클라우드 rerun_muni, 칸 = pc_rerun/cells; 실험별 파라미터 표 + loss 곡선 페이지).
2. 공헌점 페이지(12)의 계산 시간을 11/07 값으로 갱신(지금 옛 값).
3. 연구일지에 11.07 이후 내용 추가 (9장 끝 참고). RQ4는 11.10으로 이미 넣음.
4. 논문 초안 RQ3(γ 신뢰구간) · RQ5(민감도)는 아직 안 함 (사용자가 RQ4만 하라고 함, 11/10)
5. 훗카이도가 USGS보다 낮은 원인 분석 (w_LS · w_LQ 값 확인부터).
6. pilotA bootstrap 신뢰구간, `--no-labels` 실행, 크로스 이벤트 효과 직접 비교 (공헌점 계획).
7. PC 사양(CPU · 메모리) 확인 후 PC 실측 시간 참고값 정리 (PC 기록: 칸 LOCALU seed 0 지진 5개 동시 1시간 23분 52초, 학습 + 추론 + 저장 합).
8. 귀국 후 PC의 칸별 파라미터(`rowparams_*.csv.gz`, 약 6GB) 옮기기.
9. 배경 보정 단위(프레임 vs path) 교수님 확인 대기 상태 (미결).

---

## 11. 자주 하는 실수 · 함정 (이전 세션에서 실제로 있었던 것)
- 아티팩트를 `url` 없이 다시 게시해서 중복 생성.
- soft를 "μ를 낮춘다"고 잘못 설명(사실은 사전확률 항만).
- "40 m SAR 처리"라고 쓴 것 → "40 m 화소를 250 m 칸으로 평균"이 맞음.
- "7개 중 6개" 같은 표현은 "지진 5개 × 재해 = 채점 조합 7개"로 풀어 쓸 것.
- "정답 없이"를 pilotA만의 장점으로 쓴 것(DisasterNet도 비지도).
- 시간 표에 예상치(30 epoch × 100)를 쓴 것 → 교수님이 지적. 실측만.
- 서로 다른 클라우드 컴퓨터 시간을 섞을 뻔함.
- .bat에 한글 → cmd 오류. PowerShell로 `git show >` → 인코딩 깨짐.
- 칸 파라미터를 학습 후 지우게 짠 것(지금은 CSV로 남김).
- "9개로 분리" 같은 모호한 표현 → "뺀 지진 외 8개로 학습"처럼 쓸 것.
- 시정촌 비교표의 "시정촌 수"와 "채점 수" 혼동(265는 채점 수).
- 사용자가 PC를 못 만지는 동안에는 PC에서 할 일을 만들지 말 것.
