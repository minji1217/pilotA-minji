const pptxgen = require("pptxgenjs");

/* ── 팔레트: 주제에서 나온 색 ─────────────────────────────
   LS(산사태)=테라코타(흙), LQ(액상화)=틸(물). 두 hazard를 끝까지 이 색으로만 부른다. */
const INK   = "1C2A33";
const INK2  = "55636E";
const INK3  = "8A9BA8";
const BG    = "FFFFFF";
const BG2   = "F1F5F7";
const LS    = "B85042";
const LQ    = "1C7293";
const PRIOR = "A8B4BD";
const WARN  = "C9772F";
const GOOD  = "2C7A5A";

const KR = "맑은 고딕";
const NUM = "Calibri";
const W = 13.333, H = 7.5, M = 0.62;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Pilot A";
pres.title = "후속실험 3";

/* ── 헬퍼 ───────────────────────────────────────────── */
const slideBase = dark => { const s = pres.addSlide(); s.background = { color: dark ? INK : BG }; return s; };

function head(s, kicker, title, dark) {
  if (kicker) s.addText(kicker, { x: M, y: 0.44, w: 8.5, h: 0.26, isTextBox: true, margin: 0,
    fontFace: NUM, fontSize: 11, bold: true, charSpacing: 2, color: INK3 });
  s.addText(title, { x: M, y: kicker ? 0.76 : 0.6, w: W - 2 * M, h: 0.95, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 29, bold: true, color: dark ? BG : INK, valign: "top", lineSpacing: 35 });
}

const card = (s, o) => s.addShape(pres.ShapeType.roundRect, {
  x: o.x, y: o.y, w: o.w, h: o.h, rectRadius: 0.06,
  fill: { color: o.fill || BG2 }, line: { color: o.fill || BG2, width: 0 } });

function note(s, x, y, w, h, title, body, color) {
  card(s, { x, y, w, h, fill: color === LS ? "FBF1EE" : color === LQ ? "EDF4F7"
            : color === GOOD ? "EAF4EE" : BG2 });
  s.addText(title, { x: x + 0.3, y: y + 0.2, w: w - 0.6, h: 0.34, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 14.5, bold: true, color: color || INK });
  s.addText(body, { x: x + 0.3, y: y + 0.58, w: w - 0.6, h: h - 0.78, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 12, color: INK2, lineSpacing: 17 });
}

const chartFrame = () => ({
  showLegend: false,
  catAxisLabelColor: INK2, valAxisLabelColor: INK3,
  catAxisLabelFontFace: KR, valAxisLabelFontFace: NUM,
  catAxisLabelFontSize: 9, valAxisLabelFontSize: 9,
  valGridLine: { color: "E3E9EC", size: 1 }, catGridLine: { style: "none" },
  catAxisLineShow: false, valAxisLineShow: false, showTitle: false,
  dataLabelFontFace: NUM, dataLabelFontSize: 9, dataLabelColor: INK2,
});

/* ── 데이터 (규칙 B, lam_gamma=10) ──────────────────────────── */
const NL = "\n";
const NAMES = ["1 3번 b4","2 3번 b10","3 6번 b10","4 3번 자연혼재","5 6번 자연혼재",
  "A fixed","B tied","C bounded","D free","E c=0.5","F c=2",
  "G LQ c=0.5","H LQ c=0.75","I LQ c 학습","J b만 학습"];
const ZP_LS   = [0.7969,0.7969,0.7969,0.8883,0.8883,0.8400,0.8400,0.8163,0.7969,0.8163,0.8679,0.8400,0.8400,0.8400,0.8400];
const POST_LS = [0.7348,0.7635,0.7506,0.8474,0.8481,0.8550,0.8204,0.8031,0.7710,0.7915,0.8658,0.8550,0.8507,0.8550,0.8235];
const ZP_LQ   = [0.7154,0.7154,0.8372,0.7154,0.8372,0.7698,0.7698,0.7709,0.7784,0.7570,0.7687,0.7570,0.7725,0.7547,0.7698];
const POST_LQ = [0.7504,0.7663,0.7753,0.7663,0.7753,0.7812,0.7871,0.7813,0.7814,0.7914,0.7681,0.7905,0.7955,0.7732,0.8100];
const BIAS_LS = [-0.308,-0.210,-0.211,0.151,0.148,0.006,-0.270,-0.269,-0.192,-0.494,0.146,0.008,0.007,0.008,-0.273];
const BIAS_LQ = [-0.180,-0.204,-0.204,-0.204,-0.204,0.256,-0.229,-0.214,-0.208,-0.103,0.501,-0.106,0.104,-0.166,-0.192];
const C_AXIS  = ["0.50","0.75","1.00","1.25","1.50","1.75","2.00"];
const SW_LS   = [-0.4976,-0.2177,0.0065,0.1062,0.1356,0.1430,0.1464];
const SW_LQ   = [-0.1055,0.1039,0.2561,0.3612,0.4271,0.4731,0.5008];
const SWM_LS  = [0.4229,0.1946,0.0997,0.1103,0.1327,0.1414,0.1443];
const SWM_LQ  = [0.1933,0.1865,0.2493,0.3330,0.3999,0.4512,0.4896];

/* ══════════ 1. 표지 ══════════ */
{
  const s = slideBase(true);
  s.addText("PILOT A · 후속실험 3", { x: M, y: 1.65, w: 10, h: 0.35, isTextBox: true, margin: 0,
    fontFace: NUM, fontSize: 13, bold: true, charSpacing: 3, color: INK3 });
  s.addText("유도식 c = 1 이" + NL + "산사태에서는 맞았다", { x: M, y: 2.15, w: 11.4, h: 2.0, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 42, bold: true, color: BG, lineSpacing: 54 });
  s.addText("면적 항을 유도한 대로 두었을 때(a=1, b=0, c=1) 산사태 예측 편향이 +0.006이다." + NL +
            "액상화만 c를 0.62로 낮춰야 한다 — 두 hazard가 서로 다른 c를 원한다.", {
    x: M, y: 4.35, w: 11.4, h: 0.8, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 14, color: "AFC0CB", lineSpacing: 22 });
  const kpi = [["15","실험 조건"],["125","LS 평가 행"],["418","③ 평가 행"],["49","c 스윕 격자"]];
  kpi.forEach(([v,l],i)=>{
    const x = M + i*2.5;
    s.addText(v, { x, y: 5.52, w: 2.3, h: 0.72, isTextBox: true, margin: 0,
      fontFace: NUM, fontSize: 34, bold: true, color: BG });
    s.addText(l, { x, y: 6.22, w: 2.3, h: 0.3, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 11.5, color: INK3 });
  });
}

/* ══════════ 2. 한 장 요약 ══════════ */
{
  const s = slideBase(false);
  head(s, "요약", "네 가지 결론", false);
  const items = [
    ["1", "유도식 c = 1 이 산사태에서 정확히 맞는다",
     "편향이 0을 지나는 곳이 c_LS ≈ 0.99, MSE 최소도 c_LS = 1.0이다. A fixed의 편향 +0.006 / MSE 0.0998로 기저확률 0.1344를 크게 밑돈다. “격자 칸끼리 독립” 가정이 산사태에서는 성립한다.", LS],
    ["2", "액상화는 c 를 0.62로 낮춰야 한다",
     "c_LQ ≈ 0.62에서 편향이 0을 지난다. 액상화 감수성이 충적층을 따라 공간적으로 상관돼 유효 칸 수가 명목보다 적다는 뜻이다. 두 hazard가 서로 다른 c를 원한다.", LQ],
    ["3", "b 를 학습하면 오히려 나빠진다",
     "b·a를 학습하는 B·C·J가 편향 −0.27, MSE 0.21~0.26으로 최악이다. prior의 자유도를 늘리는 것이 답이 아니다.", WARN],
    ["4", "피해 데이터의 기여는 판정할 수 없다",
     "A·G·H·I는 보탠양이 +0.011~+0.015로 양수지만, 부트스트랩 95% 구간이 [−0.041, +0.073]으로 0을 포함한다. LS 평가에 음성이 20행뿐이기 때문이다.", INK3],
  ];
  items.forEach(([n,t,d,c],i)=>{
    const y = 1.85 + i*1.28;
    s.addShape(pres.ShapeType.ellipse, { x: M, y: y+0.05, w: 0.38, h: 0.38, fill: { color: c } });
    s.addText(n, { x: M, y: y+0.1, w: 0.38, h: 0.3, isTextBox: true, margin: 0,
      align: "center", fontFace: NUM, fontSize: 14, bold: true, color: "FFFFFF" });
    s.addText(t, { x: M+0.58, y, w: 11.6, h: 0.36, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 16, bold: true, color: INK });
    s.addText(d, { x: M+0.58, y: y+0.38, w: 11.6, h: 0.78, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 12, color: INK2, lineSpacing: 17 });
  });
}

/* ══════════ 3. 규칙 ① 학습 ══════════ */
{
  const s = slideBase(false);
  head(s, "규칙 1/3", "모든 조건이 똑같이 따르는 학습 설정", false);
  s.addText("조건 사이에서 실제로 달라지는 것은 prior 식의 a · b · c 몇 개뿐이다. 나머지는 전부 같다.", {
    x: M, y: 1.78, w: 12.1, h: 0.32, isTextBox: true, margin: 0, fontFace: KR, fontSize: 12.5, color: INK2 });
  s.addTable([
    ["항목","값"].map(t=>({text:t,options:{bold:true,color:INK,fill:{color:BG2}}})),
    ["seed","0"],["epoch","3000"],["optimizer","Adam · lr 0.02"],["λ_γ (gamma L2)","10"],
    ["회귀·likelihood 파라미터","44개 — 모든 조건에서 학습"],
    ["prior 파라미터","조건마다 0~6개"],
  ].map(r=>r.map((c,i)=>typeof c==="string"
      ? {text:c,options:{align:"left",fontFace:i?NUM:KR,color:INK2}} : c)), {
    x: M, y: 2.24, w: 6.0, colW:[2.7,3.3], rowH:0.4, fontFace:KR, fontSize:12, valign:"middle",
    border:{type:"solid",color:"E3E9EC",pt:1}, fill:{color:BG},
  });
  note(s, 6.9, 2.24, 5.8, 2.8, "정답(GT)은 학습에 전혀 쓰지 않는다",
    "train()이 받는 것은 피해 통계와 USGS prior뿐이다. 정답은 학습이 끝난 뒤 evaluate()에서만 쓰인다." + NL + NL +
    "그래서 정답 규칙을 바꿔도 모델은 한 줄도 바뀌지 않고 채점만 바뀐다. 같은 시험지를 다시 채점하는 것이지 시험을 다시 보는 것이 아니다.", GOOD);
  s.addText("피해 통계 XLSX  +  USGS prior XLSX   →   train()   →   posterior" + NL +
            "LS / LF 정답 XLSX   →   evaluate()   ← 여기서만 쓰인다", {
    x: M, y: 5.3, w: 12.1, h: 0.9, isTextBox: true, margin: 0,
    fontFace: NUM, fontSize: 14, color: INK, lineSpacing: 26 });
  s.addText("조건마다 브랜치가 다르다. 면적 항 조건은 prior.py / loader.py 자체가 다르기 때문에 한 checkout에서 전부 돌 수 없고, run_followup3.py가 조건별로 git worktree를 띄운다.", {
    x: M, y: 6.4, w: 12.1, h: 0.5, isTextBox: true, margin: 0, fontFace: KR, fontSize: 11.5, color: INK3, lineSpacing: 16 });
}

/* ══════════ 4. 규칙 ② 정답 ══════════ */
{
  const s = slideBase(false);
  head(s, "규칙 2/3", "정답(GT) 규칙 — 여기서 결론이 갈린다", false);
  s.addText("NA는 “없었다”가 아니라 “모른다”다. 정답지가 0과 NA를 의도적으로 갈라 놓았다.", {
    x: M, y: 1.78, w: 12.1, h: 0.32, isTextBox: true, margin: 0, fontFace: KR, fontSize: 12.5, color: INK2 });
  s.addTable([
    ["규칙","양성","음성","평가 제외","적용"].map(t=>({text:t,options:{bold:true,color:INK,fill:{color:BG2},fontSize:10.5}})),
    ["LS 기본","ls_flag = 1\n(또는 면적 > 0)","ls_flag = 0\n(또는 면적 = 0)","NA","① ② ④ ⑤"],
    ["LS 자연+혼재","ls_type ∈ {자연, 혼재}","그 밖 전부\n(인공·불명·빈칸·NA)","없음","③ (4·5번)"],
    ["LQ — jshis_flag","1","0 과 NA","없음","구마모토·훗카이도·노토반도"],
    ["LQ — lq_flag","1","0","NA","나머지 6개 이벤트"],
  ].map(r=>r.map((c,i)=>typeof c==="string"
      ? {text:c,options:{align:"left",fontFace:KR,color:INK2}} : c)), {
    x: M, y: 2.24, w: 12.1, colW:[2.1,2.7,2.9,1.5,2.9], rowH:0.62, fontFace:KR, fontSize:10.5, valign:"middle",
    border:{type:"solid",color:"E3E9EC",pt:1}, fill:{color:BG},
  });
  note(s, M, 5.02, 5.9, 2.0, "왜 LS의 NA를 빼나 — 정답지가 그렇게 적었다",
    "ls_flag = 0 :「조사표에 행이 있고 산사태 칸만 비었다 = 調査対象이고 0」" + NL +
    "NA :「調査の有無自体を確認できずNA」「항공사진 판독범위 밖」「신고 기반이라 산지 내부를 못 잡을 수 있어 NA」" + NL +
    "근거가 적힌 NA 149행 중 “확인 결과 없었다”는 행은 하나도 없다.", LS);
  note(s, 6.82, 5.02, 5.9, 2.0, "왜 ③만 제외가 없나 — 질문이 다르다",
    "③이 묻는 것은 “이 시정촌에서 자연사면 붕괴가 확인되었는가”다." + NL +
    "그 질문에는 인공만 무너졌든, 종별을 모르든, 기록이 아예 없든 답이 모두 “아니오”다." + NL +
    "반면 기본 규칙의 “산사태가 있었는가”에는 답을 모르는 행이 실제로 있다.", GOOD);
}

/* ══════════ 5. 규칙 ③ 평가 ══════════ */
{
  const s = slideBase(false);
  head(s, "규칙 3/3", "평가 집합과 지표", false);
  s.addTable([
    ["평가 집합","행","양성","음성","이벤트","MSE 기저확률"].map(t=>({text:t,options:{bold:true,color:INK,fill:{color:BG2}}})),
    ["LS 기본","125","105","20","5","0.1344"],
    ["LS 자연+혼재","418","77","341","9","0.1503"],
    ["LQ","229","109","120","6","0.2494"],
  ].map(r=>r.map((c,i)=>typeof c==="string"
      ? {text:c,options:{align:i?"right":"left",fontFace:i?NUM:KR,color:INK2}} : c)), {
    x: M, y: 1.95, w: 7.6, colW:[2.3,0.9,0.9,0.9,1.0,1.6], rowH:0.42, fontFace:KR, fontSize:12, valign:"middle",
    border:{type:"solid",color:"E3E9EC",pt:1}, fill:{color:BG},
  });
  note(s, 8.3, 1.95, 4.4, 1.68, "⚠ LS 음성이 20행뿐이다",
    "9개 이벤트 중 4개는 음성이 0개라 AUC를 잴 수조차 없어 빠진다. AUC 차이는 통계적으로 구분되지 않는다.", WARN);
  const defs = [
    ["가중평균 AUC","이벤트마다 AUC를 재고 그 이벤트의 평가 행 수로 가중평균한다. 이벤트를 섞는 pooled AUC는 다른 지진 사이의 쌍까지 채점해 의미가 없다."],
    ["원값 prior","USGS p̄ 단독. 면적 항도 학습도 거치지 않은 값."],
    ["자기 prior","그 조건의 prior sigmoid(z) 단독. 면적 항과 학습된 a·b·c가 들어 있고 피해 데이터만 안 거쳤다."],
    ["면적항 기여","자기 prior − 원값 prior. 면적 항이 prior 순위에 보탠 양."],
    ["보탠양","posterior − 자기 prior. 피해 데이터가 보탠 양. 음수면 오히려 깎았다는 뜻."],
    ["예측 편향","평균 예측확률 − 실제 양성률. 0이 정답. 양수면 과대예측."],
  ];
  defs.forEach(([t,d],i)=>{
    const y = 3.95 + i*0.52;
    s.addText(t, { x: M, y, w: 2.0, h: 0.32, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 12, bold: true, color: INK });
    s.addText(d, { x: M+2.1, y, w: 10.0, h: 0.42, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 11, color: INK2, lineSpacing: 15 });
  });
}

/* ══════════ 6. 실험 지도 ══════════ */
{
  const s = slideBase(false);
  head(s, "실험", "네 갈래 15개 조건", false);
  const groups = [
    ["②","교수님 피드백 — b 범위","3개","b 상한을 4 / 10으로 넓혀 안쪽에서 멈추는지 본다","교수님 지시",LS],
    ["③","자연+혼재 GT 재평가","2개","②의 모델을 그대로 두고 LS 정답만 자연사면으로 좁힌다","교수님 지시",GOOD],
    ["④","면적 항","10개","z = a·log p̄ + b + c·log k 의 a·b·c를 어떻게 다룰지","A~F 지시 / G~J 추가",LQ],
    ["⑤","면적 계수 c 훑기","49점+","c를 0.5~2.0 격자로 돌려 최적값을 찾는다","④ 결과를 보고 추가",WARN],
  ];
  groups.forEach(([n,t,cnt,d,src,c],i)=>{
    const y = 2.0 + i*1.22;
    card(s, { x: M, y, w: 12.1, h: 1.05, fill: BG2 });
    s.addShape(pres.ShapeType.ellipse, { x: M+0.3, y: y+0.3, w: 0.46, h: 0.46, fill: { color: c } });
    s.addText(n, { x: M+0.3, y: y+0.36, w: 0.46, h: 0.34, isTextBox: true, margin: 0,
      align: "center", fontFace: KR, fontSize: 14, bold: true, color: "FFFFFF" });
    s.addText(t, { x: M+0.95, y: y+0.18, w: 5.0, h: 0.34, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 15, bold: true, color: INK });
    s.addText(cnt + " · " + src, { x: M+0.95, y: y+0.56, w: 5.0, h: 0.3, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 11, color: INK3 });
    s.addText(d, { x: M+6.2, y: y+0.34, w: 5.6, h: 0.5, isTextBox: true, margin: 0,
      fontFace: KR, fontSize: 12, color: INK2, lineSpacing: 16 });
  });
  s.addText("모두 같은 데이터 · 같은 seed · lam_gamma = 10.  정답은 학습에 쓰지 않으므로 ③은 학습을 다시 하지 않는다.", {
    x: M, y: 6.95, w: 12.1, h: 0.3, isTextBox: true, margin: 0, fontFace: KR, fontSize: 11.5, color: INK3 });
}

/* ══════════ 7. 갈래 ② 상세 ══════════ */
{
  const s = slideBase(false);
  head(s, "갈래 ②", "교수님 피드백 — b 상한을 넓히면 안쪽에서 멈추는가", false);
  s.addText("prior 식은  z = a · logit(p̄) + b  하나뿐이다(면적 항 없음). a는 USGS 값을 얼마나 믿을지(기울기), b는 확률 수준을 위아래로 옮기는 값(높이)이고, a ∈ [0.5, 2]다.", {
    x: M, y: 1.78, w: 12.1, h: 0.5, isTextBox: true, margin: 0, fontFace: KR, fontSize: 12.5, color: INK2, lineSpacing: 17 });
  const rows = [
    ["기준","3번 기준","followup1-1-wood","평균","[−2, 2]","1.989","작년 실험. 비교 출발점"],
    ["기준","6번 기준","followup2-1-wood","최대","[−2, 4]","3.956","작년 실험. LQ 집계만 다르다"],
    ["1","3번 b4","followup3-avg-b4","평균","[−2, 4]","3.954 ⚠","b 상한을 4로 넓혀 본다"],
    ["2","3번 b10","followup3-avg-b10","평균","[−2, 10]","9.523 ⚠","b 상한을 10까지 넓혀 본다"],
    ["3","6번 b10","followup3-lqmax-b10","최대","[−2, 10]","9.589 ⚠","2번과 같되 LQ를 최대집계로"],
  ];
  const hi = new Set(["1","2","3"]);
  s.addTable([
    ["#","이름","브랜치","LQ 집계","b 범위","학습된 b_LS","무엇을 보려는 조건인가"].map((h,i)=>
      ({text:h,options:{bold:true,color:INK,fill:{color:BG2},align:i>=3&&i<=5?"right":"left",fontSize:10.5}})),
    ...rows.map(r=>r.map((c,i)=>({text:c,
      options:{align:i>=3&&i<=5?"right":"left",fontFace:i>=4&&i<=5?NUM:KR,
               bold:hi.has(r[0]),color:hi.has(r[0])?INK:INK3,fill:{color:hi.has(r[0])?BG:"FAFCFD"}}}))),
  ], { x: M, y: 2.42, w: 12.1, colW:[0.6,1.5,2.9,1.0,1.15,1.45,3.5], rowH:0.36,
       fontFace:KR, fontSize:10.5, valign:"middle", border:{type:"solid",color:"E3E9EC",pt:1} });
  note(s, M, 4.62, 5.9, 2.2, "⚠ 답: 안 멈춘다",
    "b_LS가 상한 4에서 3.954, 상한 10에서 9.523으로 둘 다 끝에 붙는다." + NL +
    "최적값을 찾은 것이 아니라 잘린 것이다." + NL + NL +
    "규칙 B에서 다시 보면 이 조건들은 편향이 −0.21 ~ −0.31로 확률을 크게 과소예측한다. b를 키우는 방향 자체가 틀렸다.", WARN);
  note(s, 6.82, 4.62, 5.9, 2.2, "3번 vs 6번의 차이는 하나뿐",
    "LQ prior를 시정촌 안 격자들의 평균으로 집계하느냐, 최대로 집계하느냐." + NL + NL +
    "최대집계가 LQ prior AUC를 0.7154 → 0.8372로 크게 올린다. 그런데 posterior는 0.775에 그쳐 −0.062로 오히려 깎는다 — 좋아진 prior를 모델이 망가뜨린다.", LQ);
}

/* ══════════ 8. 갈래 ③ 상세 ══════════ */
{
  const s = slideBase(false);
  head(s, "갈래 ③", "자연+혼재 재평가 — 학습을 다시 하지 않는다", false);
  s.addText("USGS 산사태 사전모형(Nowicki Jessee 2018)이 설명하는 것은 자연사면의 붕괴다. 그런데 정답의 양성에는 성토·옹벽·법면 같은 인공사면 붕괴가 섞여 있다. 모형이 원리적으로 맞힐 수 없는 사건까지 양성으로 세고 있는 셈이라, 자연·혼재만 양성으로 두고 다시 채점한다.", {
    x: M, y: 1.78, w: 12.1, h: 0.72, isTextBox: true, margin: 0, fontFace: KR, fontSize: 12.5, color: INK2, lineSpacing: 18 });
  s.addTable([
    ["#","이름","실행 브랜치","결과 이름","평가"].map(h=>({text:h,options:{bold:true,color:INK,fill:{color:BG2},fontSize:10.5}})),
    ["4","3번 자연혼재","followup3-avg-b10  (2번과 동일)","followup3-avg-b10-natmix","418행"],
    ["5","6번 자연혼재","followup3-lqmax-b10  (3번과 동일)","followup3-lqmax-b10-natmix","418행"],
  ].map(r=>r.map(c=>typeof c==="string"?{text:c,options:{align:"left",fontFace:KR,color:INK2}}:c)), {
    x: M, y: 2.62, w: 12.1, colW:[0.6,2.0,4.3,3.6,1.6], rowH:0.4, fontFace:KR, fontSize:11, valign:"middle",
    border:{type:"solid",color:"E3E9EC",pt:1}, fill:{color:BG} });
  note(s, M, 3.75, 5.9, 1.55, "브랜치를 따로 만들지 않는다",
    "학습이 정답을 전혀 쓰지 않으므로 같은 seed면 모델이 완전히 같다. 2·3번 브랜치에서 --gt-variant natmix 옵션만 켜고 결과 파일 이름으로 구분한다.", GOOD);
  note(s, 6.82, 3.75, 5.9, 1.55, "평가가 넓어진다",
    "125행(양성률 84%) → 418행(양성률 18.4%). 제외하는 행이 없기 때문이다. 음성이 20행 → 341행이 되어 AUC가 훨씬 안정된다.", GOOD);
  s.addChart(pres.ChartType.bar, [
    { name: "자기 prior", labels: ["4 3번 자연혼재","5 6번 자연혼재"], values: [0.8883,0.8883] },
    { name: "posterior",  labels: ["4 3번 자연혼재","5 6번 자연혼재"], values: [0.8474,0.8481] },
  ], { x: M, y: 5.5, w: 5.9, h: 1.5, barDir:"bar", barGrouping:"clustered", barGapWidthPct:30,
       chartColors:[PRIOR,GOOD], ...chartFrame(), showLegend:true, legendPos:"b",
       legendFontFace:KR, legendFontSize:10, legendColor:INK2,
       showValue:true, dataLabelPosition:"outEnd", dataLabelFormatCode:"0.000",
       valAxisMinVal:0.7, valAxisMaxVal:0.95, valAxisLabelFormatCode:"0.0" });
  note(s, 6.82, 5.5, 5.9, 1.5, "여기서도 posterior가 prior를 못 넘는다",
    "−0.041 / −0.040. LS MSE는 0.1404 / 0.1371로 기저확률 0.1503을 밑돌아 눈금은 괜찮다. 편향은 +0.15로 과대예측이 남는다.", INK3);
}

/* ══════════ 9. 갈래 ④ 상세 ══════════ */
{
  const s = slideBase(false);
  head(s, "갈래 ④", "면적 항 A~J — 무엇을 왜 넣었나", false);
  s.addText("z = a · log p̄ + b + c · (log k − m)      k = 시정촌 면적 ÷ 격자 한 칸 넓이", {
    x: M, y: 1.76, w: 8.0, h: 0.3, isTextBox: true, margin: 0, fontFace: NUM, fontSize: 13.5, bold: true, color: INK });
  s.addText("USGS 값 p̄는 “발생 확률”이 아니라 “격자 한 칸 중 덮이는 면적 비율”이다. 넓은 시정촌일수록 한 곳이라도 무너질 확률이 높다. 칸이 독립이면 λ = p̄ × k 이고 P(하나라도) = 1 − exp(−λ). 포화를 피해 순위가 같은 log λ = log p̄ + log k 를 쓴다 — 그래서 a=1, b=0, c=1 이 유도식 그대로이고, 면적 항이 있는 조건만 logit이 아니라 log를 쓴다.", {
    x: M, y: 2.1, w: 12.1, h: 0.66, isTextBox: true, margin: 0, fontFace: KR, fontSize: 11.5, color: INK2, lineSpacing: 16 });
  const rows = [
    ["지시","A","fixed","avg-fixed","1.00","0.00","1.00","1.00","0.00","1.00","0","아무것도 학습 안 함 — 기준점"],
    ["지시","B","tied","avg-tied","0.89","4.91","= a","0.50","0.80","= a","4","면적의 힘을 USGS와 같게 묶고 크기만 학습"],
    ["지시","C","bounded","avg-bounded","1.98","11.06","1.00","0.51","0.83","1.00","4","면적의 힘은 1로 두고 기울기·높이만 학습"],
    ["지시","D","free","avg-free","1.86","11.49","0.01","0.50","0.91","0.78","6","여섯 값 전부 데이터에 맡긴다"],
    ["지시","E","c=0.5","avg-a1-c05","1.00","0.00","0.50","1.00","0.00","0.50","0","면적의 영향을 절반으로 줄여 고정"],
    ["지시","F","c=2","avg-a1-c2","1.00","0.00","2.00","1.00","0.00","2.00","0","면적의 영향을 두 배로 키워 고정"],
    ["추가","G","lq-c05","avg-lq-c05","1.00","0.00","1.00","1.00","0.00","0.50","0","액상화 쪽만 절반 (LS는 A와 동일)"],
    ["추가","H","lq-c075","avg-lq-c075","1.00","0.00","1.00","1.00","0.00","0.75","0","액상화 쪽만 0.75 (LS는 A와 동일)"],
    ["추가","I","lq-c-free","avg-lq-c-free","1.00","0.00","1.00","1.00","0.00","0.43","1","액상화 쪽 c만 데이터가 고른다"],
    ["요청","J","b-only","avg-b-only","1.00","5.53","1.00","1.00","2.69","1.00","2","순위는 유도식대로, 확률 높이만 보정"],
  ];
  const fill = { "지시":"FAFCFD", "추가":"FFF6EC", "요청":"EAF4EE" };
  s.addTable([
    ["출처","","이름","브랜치 followup3-","a_LS","b_LS","c_LS","a_LQ","b_LQ","c_LQ","학습","무엇을 보려는 조건인가"].map((h,i)=>
      ({text:h,options:{bold:true,color:INK,fill:{color:BG2},align:i>=4&&i<=10?"right":"left",fontSize:9.5}})),
    ...rows.map(r=>r.map((c,i)=>({text:c,
      options:{align:i>=4&&i<=10?"right":"left",fontFace:i>=4&&i<=9?NUM:KR,
               bold:i===1,color:INK2,fill:{color:fill[r[0]]}}}))),
  ], { x: M, y: 2.92, w: 12.1, colW:[0.6,0.38,1.0,1.85,0.72,0.82,0.72,0.72,0.72,0.72,0.58,3.27],
       rowH:0.27, fontFace:KR, fontSize:9.5, valign:"middle", border:{type:"solid",color:"E3E9EC",pt:1} });
  s.addText("a = USGS를 얼마나 믿을지(기울기)     b = 확률 높이     c = 면적을 얼마나 반영할지     ‘학습’은 prior의 a·b·c 중 학습하는 개수 (회귀 44개는 늘 학습)", {
    x: M, y: 5.98, w: 12.1, h: 0.3, isTextBox: true, margin: 0, fontFace: KR, fontSize: 11, color: INK2 });
  s.addText("출처 —  회색: 교수님이 지시하신 A~F     주황: 실험 도중 우리가 만든 G·H·I     초록: 요청으로 추가한 J", {
    x: M, y: 6.32, w: 12.1, h: 0.3, isTextBox: true, margin: 0, fontFace: KR, fontSize: 11, bold: true, color: WARN });
  s.addText("b를 학습하는 B·C·D·J는 b 범위가 [−20, 20]이고 log k를 평균 중심화한다(m = LS 8.042 / LQ 6.656). b=0 고정 조건은 중심화하지 않는다 — 빼면 유도식이 깨진다.", {
    x: M, y: 6.66, w: 12.1, h: 0.3, isTextBox: true, margin: 0, fontFace: KR, fontSize: 10.5, color: INK3 });
}


/* ══════════ 9-2. G~J 는 왜 그 값인가 ══════════ */
{
  const s = slideBase(false);
  head(s, "갈래 ④", "G · H · I · J 는 왜 그 값으로 정했나", false);
  s.addText("A~F를 돌리고 나서 관찰된 것: 같은 c = 1 인데 두 hazard의 눈금이 정반대로 어긋난다.", {
    x: M, y: 1.78, w: 12.1, h: 0.32, isTextBox: true, margin: 0, fontFace: KR, fontSize: 12.5, color: INK2 });
  const cmp = [
    ["산사태 LS 편향","−0.494  무너짐","+0.006  거의 완벽","+0.146"],
    ["액상화 LQ 편향","−0.103","+0.256  크게 과대","+0.501"],
  ];
  s.addTable([
    ["","c = 0.5  (E)","c = 1.0  (A)","c = 2.0  (F)"].map(h=>({text:h,options:{bold:true,color:INK,fill:{color:BG2}}})),
    ...cmp.map(r=>r.map((c,i)=>({text:c,options:{align:i?"right":"left",fontFace:i?NUM:KR,
      color:c.includes("완벽")?GOOD:(c.includes("무너짐")||c.includes("과대"))?LS:INK2,
      bold:c.includes("완벽")}}))),
  ], {
    x: M, y: 2.24, w: 9.4, colW:[2.5,2.3,2.3,2.3], rowH:0.44, fontFace:KR, fontSize:12, valign:"middle",
    border:{type:"solid",color:"E3E9EC",pt:1}, fill:{color:BG} });
  s.addText("→ 두 hazard가 서로 다른 c를 원한다.  LS는 유도값 1을 건드리지 말고, LQ만 내려 보자.", {
    x: M, y: 3.42, w: 12.1, h: 0.34, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 13.5, bold: true, color: INK });

  const rows = [
    ["G","c_LQ = 0.5 고정","E에서 이미 확인된 반대쪽 값. LQ 편향이 −0.103으로 부호가 뒤집히는 지점을 그대로 가져왔다."],
    ["H","c_LQ = 0.75 고정","관찰값 두 개(0.5 → −0.103, 1.0 → +0.256)로 편향 0 교차점을 선형 추정하면 0.643이다. 0.25 간격 격자에서 그 위쪽 점을 잡았다."],
    ["I","c_LQ 만 학습","사람이 고르지 말고 우도가 고르게 둔다. c ∈ [0, 2]에서 학습해 0.434로 수렴했다."],
    ["J","b 만 학습","c를 건드리면 순위까지 같이 움직인다. b는 z에 더해지는 상수라 hazard 안에서 순위를 전혀 바꾸지 않고 확률 높이만 옮긴다 — 눈금을 고치는 올바른 손잡이는 c가 아니라 b라는 가설."],
  ];
  rows.forEach(([k,t,d],i)=>{
    const y = 3.95 + i*0.78;
    s.addShape(pres.ShapeType.ellipse, { x: M, y: y+0.04, w: 0.34, h: 0.34, fill:{color: k==="J"?GOOD:WARN} });
    s.addText(k, { x: M, y: y+0.08, w: 0.34, h: 0.28, isTextBox:true, margin:0,
      align:"center", fontFace:KR, fontSize:12.5, bold:true, color:"FFFFFF" });
    s.addText(t, { x: M+0.5, y, w: 2.5, h: 0.32, isTextBox:true, margin:0,
      fontFace:NUM, fontSize:12.5, bold:true, color:INK });
    s.addText(d, { x: M+3.1, y, w: 9.0, h: 0.66, isTextBox:true, margin:0,
      fontFace:KR, fontSize:11.5, color:INK2, lineSpacing:16 });
  });
  s.addText("⑤ 스윕이 나중에 실제 교차점을 c_LQ ≈ 0.618 로 확정했다.  G(0.5)와 H(0.75)가 정답을 사이에 두고 있었고, 선형 추정 0.643은 0.02 차이로 맞았다.  반면 I가 학습한 0.434는 아래로 지나쳤다.", {
    x: M, y: 6.95, w: 12.1, h: 0.52, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 11.5, bold: true, color: GOOD, lineSpacing: 16 });
}

/* ══════════ 10. 갈래 ⑤ 상세 ══════════ */
{
  const s = slideBase(false);
  head(s, "갈래 ⑤", "면적 계수 c 훑기 — 유도값 1이 정말 맞나", false);
  s.addText("④에서 c를 0.5 / 1 / 2 세 점으로만 봤다. 그 사이와 바깥을 직접 돌려 편향이 0을 지나는 c를 찾는다. AreaPrior(mode=\"grid\")로 c_LS와 c_LQ를 직접 지정하고 a=1, b=0은 고정하므로 prior에서 학습하는 값이 0개다 — 순수하게 c만 다른 실험이다.", {
    x: M, y: 1.78, w: 12.1, h: 0.56, isTextBox: true, margin: 0, fontFace: KR, fontSize: 12.5, color: INK2, lineSpacing: 17 });
  s.addTable([
    ["항목","값"].map(t=>({text:t,options:{bold:true,color:INK,fill:{color:BG2}}})),
    ["브랜치","followup3-area-sweep"],
    ["거친 격자","c_LS × c_LQ ∈ {0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0}  =  49점"],
    ["미세 격자","최적점 주변 0.05 간격"],
    ["학습 prior 파라미터","0개"],
    ["검증","격자점 (1.0, 1.0)이 조건 A와 완전히 일치"],
  ].map(r=>r.map((c,i)=>typeof c==="string"?{text:c,options:{align:"left",fontFace:i?NUM:KR,color:INK2}}:c)), {
    x: M, y: 2.5, w: 12.1, colW:[2.6,9.5], rowH:0.4, fontFace:KR, fontSize:11.5, valign:"middle",
    border:{type:"solid",color:"E3E9EC",pt:1}, fill:{color:BG} });
  note(s, M, 5.1, 5.9, 1.8, "두 hazard의 c는 서로 간섭하지 않는다",
    "한쪽 c를 0.5~2.0으로 바꿔도 다른 쪽 편향은 0.004 안에서만 움직인다." + NL +
    "c_LS와 c_LQ를 따로 골라도 된다는 뜻이고, 실제로 두 값이 크게 다르게 나왔다.", GOOD);
  note(s, 6.82, 5.1, 5.9, 1.8, "왜 재학습이 필요한가",
    "정답은 학습에 안 쓰이지만 c는 prior를 바꾸므로 posterior가 달라진다. 그래서 c마다 회귀 44개를 새로 적합한다." + NL +
    "한 점에 14초, 49점에 12분이다.", INK3);
}

/* ══════════ 11-12. 결과 — prior vs posterior ══════════ */
function pairSlide(kicker, title, dek, zp, post, col, capt, noteT, noteB) {
  const s = slideBase(false);
  head(s, kicker, title, false);
  s.addText(dek, { x: M, y: 1.78, w: 11.6, h: 0.34, isTextBox: true, margin: 0,
    fontFace: KR, fontSize: 13, color: INK2 });
  s.addChart(pres.ChartType.bar, [
    { name: "자기 prior", labels: NAMES, values: zp },
    { name: "posterior",  labels: NAMES, values: post },
  ], { x: M, y: 2.2, w: 8.0, h: 4.6, barDir:"bar", barGrouping:"clustered", barGapWidthPct:25,
       chartColors:[PRIOR,col], ...chartFrame(), showLegend:true, legendPos:"t",
       legendFontFace:KR, legendFontSize:11, legendColor:INK2,
       showValue:true, dataLabelPosition:"outEnd", dataLabelFormatCode:"0.000",
       valAxisMinVal:0.65, valAxisMaxVal:0.95, valAxisLabelFormatCode:"0.0" });
  s.addText(capt, { x: M, y: 6.85, w: 8.0, h: 0.3, isTextBox: true, margin: 0, align:"center",
    fontFace: KR, fontSize: 10.5, color: INK3 });
  note(s, 8.85, 2.2, 3.85, 4.6, noteT, noteB, col);
}
pairSlide("결과", "산사태 LS — 자기 prior vs posterior",
  "회색이 자기 prior 단독, 색이 posterior다. 색 막대가 회색보다 길어야 피해 데이터가 보탠 것이 있다.",
  ZP_LS, POST_LS, LS, "가중평균 AUC · ①②④는 125행 / ③은 418행",
  "A · G · H · I 가 넘는다",
  "네 조건 모두 c_LS = 1 이고 b = 0 고정이다. 보탠양 +0.011 ~ +0.015." + NL + NL +
  "⚠ 다만 부트스트랩 95% 구간이 [−0.041, +0.073]으로 0을 포함한다. LS 평가에 음성이 20행뿐이라 AUC로는 판정할 수 없다." + NL + NL +
  "F(c=2)는 prior가 0.868로 가장 높지만 편향 +0.146으로 눈금이 나쁘다 — 순위가 좋은 c와 눈금이 맞는 c가 다르다.");
pairSlide("결과", "액상화 LQ — 자기 prior vs posterior",
  "LQ는 정답 규칙을 바꾸지 않았다. 229행 그대로다.",
  ZP_LQ, POST_LQ, LQ, "가중평균 AUC · 229행 · 양성률 47.6%",
  "대부분 보탠다",
  "15개 중 13개가 자기 prior를 넘는다(+0.003 ~ +0.040). J(b만 학습)가 0.8100 / +0.040으로 1위다." + NL + NL +
  "예외는 3·5번(LQ 최대집계)과 F다. 3·5번은 prior가 0.837로 가장 좋은데 posterior가 0.775에 그쳐 −0.062로 가장 크게 깎는다." + NL + NL +
  "LS와 정반대 그림이다.");

/* ══════════ 13. 결과 — 예측 편향 ══════════ */
{
  const s = slideBase(false);
  head(s, "결과", "예측 편향 — 여기가 진짜 지표다", false);
  s.addText("평균 예측확률 − 실제 양성률. 0이 정답이고 양수면 과대예측이다. 음성이 20행뿐이라 AUC는 못 믿지만 편향과 MSE는 행 전체를 쓰므로 훨씬 안정적이다.", {
    x: M, y: 1.78, w: 12.1, h: 0.5, isTextBox: true, margin: 0, fontFace: KR, fontSize: 12.5, color: INK2, lineSpacing: 17 });
  s.addChart(pres.ChartType.bar, [{ name:"LS 편향", labels: NAMES, values: BIAS_LS }], {
    x: M, y: 2.4, w: 5.9, h: 4.0, barDir:"bar", chartColors:[LS], ...chartFrame(),
    showValue:true, dataLabelPosition:"outEnd", dataLabelFormatCode:"0.000",
    valAxisMinVal:-0.55, valAxisMaxVal:0.55, valAxisLabelFormatCode:"0.0" });
  s.addChart(pres.ChartType.bar, [{ name:"LQ 편향", labels: NAMES, values: BIAS_LQ }], {
    x: 6.82, y: 2.4, w: 5.9, h: 4.0, barDir:"bar", chartColors:[LQ], ...chartFrame(),
    showValue:true, dataLabelPosition:"outEnd", dataLabelFormatCode:"0.000",
    valAxisMinVal:-0.55, valAxisMaxVal:0.55, valAxisLabelFormatCode:"0.0" });
  s.addText("산사태 LS — 실제 양성률 0.840", { x: M, y: 6.45, w: 5.9, h: 0.28, isTextBox:true, margin:0,
    align:"center", fontFace:KR, fontSize:11, color:INK3 });
  s.addText("액상화 LQ — 실제 양성률 0.476", { x: 6.82, y: 6.45, w: 5.9, h: 0.28, isTextBox:true, margin:0,
    align:"center", fontFace:KR, fontSize:11, color:INK3 });
  s.addText("A · G · H · I 는 LS 편향이 0.008 안이다.  b를 학습하는 B · C · J 는 −0.27 로 최악이고, E(c=0.5)는 −0.494 로 무너진다.", {
    x: M, y: 6.82, w: 12.1, h: 0.3, isTextBox:true, margin:0, fontFace:KR, fontSize:12, bold:true, color:INK });
}

/* ══════════ 14. 결과 — c 스윕 ══════════ */
{
  const s = slideBase(false);
  head(s, "갈래 ⑤ 결과", "산사태는 c = 1, 액상화는 c = 0.62", false);
  s.addText("c를 0.5부터 2.0까지 직접 돌린 결과다. 편향이 0을 지나는 자리가 두 hazard에서 크게 다르다.", {
    x: M, y: 1.78, w: 12.1, h: 0.32, isTextBox: true, margin: 0, fontFace: KR, fontSize: 12.5, color: INK2 });
  s.addChart(pres.ChartType.line, [
    { name: "산사태 LS", labels: C_AXIS, values: SW_LS },
    { name: "액상화 LQ", labels: C_AXIS, values: SW_LQ },
  ], { x: M, y: 2.3, w: 5.9, h: 3.1, chartColors:[LS,LQ], ...chartFrame(),
       showLegend:true, legendPos:"t", legendFontFace:KR, legendFontSize:11, legendColor:INK2,
       lineSize:3, lineSmooth:false, valAxisMinVal:-0.55, valAxisMaxVal:0.55, valAxisLabelFormatCode:"0.0" });
  s.addText("예측 편향 — 0이 정답.  가로축 c", { x: M, y: 5.45, w: 5.9, h: 0.28, isTextBox:true, margin:0,
    align:"center", fontFace:KR, fontSize:10.5, color:INK3 });
  s.addChart(pres.ChartType.line, [
    { name: "산사태 LS", labels: C_AXIS, values: SWM_LS },
    { name: "액상화 LQ", labels: C_AXIS, values: SWM_LQ },
  ], { x: 6.82, y: 2.3, w: 5.9, h: 3.1, chartColors:[LS,LQ], ...chartFrame(),
       showLegend:true, legendPos:"t", legendFontFace:KR, legendFontSize:11, legendColor:INK2,
       lineSize:3, lineSmooth:false, valAxisMinVal:0.0, valAxisMaxVal:0.55, valAxisLabelFormatCode:"0.0" });
  s.addText("MSE — 낮을수록 좋음.  기저확률 LS 0.1344 / LQ 0.2494", { x: 6.82, y: 5.45, w: 5.9, h: 0.28,
    isTextBox:true, margin:0, align:"center", fontFace:KR, fontSize:10.5, color:INK3 });
  note(s, M, 5.82, 5.9, 1.32, "산사태 — 유도값이 맞는다",
    "편향 0 교차 c_LS ≈ 0.99, MSE 최소 c_LS = 1.0 (0.0997). “칸끼리 독립” 가정이 성립한다.", LS);
  note(s, 6.82, 5.82, 5.9, 1.32, "액상화 — 낮춰야 한다",
    "편향 0 교차 c_LQ ≈ 0.62, MSE 최소 0.625 (0.1765). 감수성이 충적층을 따라 상관돼 유효 칸 수가 적다.", LQ);
}

/* ══════════ 15. 전체 결과 표 ══════════ */
{
  const s = slideBase(false);
  head(s, "결과", "전체 15개 조건", false);
  const rows = [
    ["1","3번 b4","125","0.797","0.797","0.735","−0.062","0.2793","−0.308","0.750","+0.035","0.2280","−0.180"],
    ["2","3번 b10","125","0.797","0.797","0.763","−0.033","0.2510","−0.210","0.766","+0.051","0.2428","−0.204"],
    ["3","6번 b10","125","0.797","0.797","0.751","−0.046","0.2524","−0.211","0.775","−0.062","0.2409","−0.204"],
    ["4","3번 자연혼재","418","0.888","0.888","0.847","−0.041","0.1404","+0.151","0.766","+0.051","0.2428","−0.204"],
    ["5","6번 자연혼재","418","0.888","0.888","0.848","−0.040","0.1371","+0.148","0.775","−0.062","0.2409","−0.204"],
    ["A","fixed","125","0.797","0.840","0.855","+0.015","0.0998","+0.006","0.781","+0.011","0.2497","+0.256"],
    ["B","tied","125","0.797","0.840","0.820","−0.020","0.2148","−0.270","0.787","+0.017","0.2530","−0.229"],
    ["C","bounded","125","0.797","0.816","0.803","−0.013","0.2563","−0.269","0.781","+0.010","0.2507","−0.214"],
    ["D","free","125","0.797","0.797","0.771","−0.026","0.2527","−0.192","0.781","+0.003","0.2411","−0.208"],
    ["E","c=0.5","125","0.797","0.816","0.791","−0.025","0.4192","−0.494","0.791","+0.034","0.1932","−0.103"],
    ["F","c=2","125","0.797","0.868","0.866","−0.002","0.1443","+0.146","0.768","−0.001","0.4896","+0.501"],
    ["G","LQ c=0.5","125","0.797","0.840","0.855","+0.015","0.0994","+0.008","0.791","+0.033","0.1929","−0.106"],
    ["H","LQ c=0.75","125","0.797","0.840","0.851","+0.011","0.0998","+0.007","0.796","+0.023","0.1868","+0.104"],
    ["I","LQ c 학습","125","0.797","0.840","0.855","+0.015","0.0993","+0.008","0.773","+0.019","0.2155","−0.166"],
    ["J","b만 학습","125","0.797","0.840","0.824","−0.017","0.2218","−0.273","0.810","+0.040","0.2217","−0.192"],
  ];
  const hi = new Set(["A","G","H","I"]);
  const hdr = ["#","이름","행","원값","자기","post","보탠양","LS MSE","LS 편향","LQ post","LQ 보탠양","LQ MSE","LQ 편향"];
  s.addTable([
    hdr.map((h,i)=>({text:h,options:{bold:true,color:INK,fill:{color:BG2},align:i>=2?"right":"left",fontSize:9.5}})),
    ...rows.map(r=>r.map((c,i)=>({text:c,
      options:{align:i>=2?"right":"left",fontFace:i>=2?NUM:KR,
               bold:hi.has(r[0]),color:hi.has(r[0])?INK:INK2,
               fill:{color:hi.has(r[0])?"EAF4EE":BG}}}))),
  ], { x: M, y: 1.95, w: 12.1, colW:[0.42,1.35,0.62,0.85,0.85,0.85,1.0,1.0,1.0,1.0,1.1,1.0,1.06],
       rowH:0.29, fontFace:KR, fontSize:9.5, valign:"middle", border:{type:"solid",color:"E3E9EC",pt:1} });
  s.addText("원값 = USGS p̄ 단독 · 자기 = 그 조건의 prior 단독 · 보탠양 = post − 자기.  MSE 기저확률: LS 0.1344 (4·5번은 0.1503) / LQ 0.2494.  초록 = LS 보탠양이 양수인 네 조건.", {
    x: M, y: 6.72, w: 12.1, h: 0.3, isTextBox: true, margin: 0, fontFace: KR, fontSize: 10.5, color: INK3 });
}

/* ══════════ 16. 한계와 다음 ══════════ */
{
  const s = slideBase(true);
  head(s, "마무리", "한계와 다음", true);
  const items = [
    ["LS 평가가 125행 · 음성 20행 · 이벤트 5개뿐이다", "이번 판의 가장 큰 제약이다. NA 304행을 되살리려면 「조사 대상이었음이 확인되나 기록 없음」과 「조사 여부 불명」을 근거별로 갈라야 한다. location_note가 구조화되어 있어 149행은 분류 가능하고, 면적 시트 4개(155행)는 원자료를 다시 봐야 한다.", LS],
    ["③과 ④·⑤를 결합하지 않았다", "A~J는 전부 기본 규칙 125행으로만 채점했다. predictions_*.csv에 418행 전부의 예측이 있으므로 재학습 없이 재채점만 하면 된다.", WARN],
    ["c_LS = 1.0, c_LQ ≈ 0.62 를 정식 조건으로 안 돌렸다", "스윕 격자 위의 점으로만 확인했다. 브랜치를 만들어 정식 조건으로 올려야 한다.", GOOD],
    ["LQ 최대집계 + 면적 항을 안 해봤다", "가장 좋은 LQ prior(0.8372)와 면적 항을 한 번도 같이 쓰지 않았다. λ = p̄·k 유도가 평균집계를 전제하므로 이론 정리가 먼저다.", LQ],
    ["왜 b를 학습하면 나빠지는지 모른다", "B·C·J의 편향이 −0.27이다. 양성률 84%짜리 좁은 집합에서 우도가 b를 어느 방향으로 끄는지 확인이 필요하다.", INK3],
  ];
  items.forEach(([t,d,c],i)=>{
    const y = 1.9 + i*1.02;
    s.addShape(pres.ShapeType.ellipse, { x: M, y: y+0.08, w: 0.22, h: 0.22, fill:{color:c} });
    s.addText(t, { x: M+0.44, y, w: 11.6, h: 0.32, isTextBox:true, margin:0,
      fontFace:KR, fontSize:14.5, bold:true, color:BG });
    s.addText(d, { x: M+0.44, y: y+0.34, w: 11.6, h: 0.6, isTextBox:true, margin:0,
      fontFace:KR, fontSize:11.5, color:"AFC0CB", lineSpacing:15 });
  });
  s.addText("재현:  python run_followup3.py --data-dir .        정답 규칙은 loader.py에 있고 학습에는 쓰이지 않는다.", {
    x: M, y: 7.02, w: 11.6, h: 0.3, isTextBox:true, margin:0, fontFace:NUM, fontSize:11, color:INK3 });
}

/* ══════════ 17. 부록 — 이전 판과의 차이 ══════════ */
{
  const s = slideBase(false);
  head(s, "부록", "이전 판에서 무엇이 뒤집혔나", false);
  s.addText("이전 판은 LS 정답의 NA를 0으로 세었다(418행). 정답지 정의를 확인한 뒤 NA를 평가에서 빼는 원래 규칙으로 되돌렸고, 다음이 뒤집혔다. 모델은 하나도 바뀌지 않았다 — 채점만 바뀌었다.", {
    x: M, y: 1.78, w: 12.1, h: 0.5, isTextBox: true, margin: 0, fontFace: KR, fontSize: 12.5, color: INK2, lineSpacing: 17 });
  const rows = [
    ["최적 c_LS","0.62","0.99  ≈ 유도값 1"],
    ["A fixed 편향","+0.436  (크게 과대예측)","+0.006"],
    ["E c=0.5 편향","−0.097","−0.494  (무너짐)"],
    ["b 학습 (B·C·J) 편향","+0.08 ~ +0.09  (양호)","−0.27  (최악)"],
    ["LS 보탠양","15개 중 14개 음수","A·G·H·I 양수"],
    ["LS 평가 집합","418행 · 양성률 25.1%","125행 · 양성률 84.0%"],
  ];
  s.addTable([
    ["항목","이전 판 (NA → 0)","지금 (NA 평가 제외)"].map(h=>({text:h,options:{bold:true,color:INK,fill:{color:BG2}}})),
    ...rows.map(r=>r.map((c,i)=>({text:c,
      options:{align:i?"right":"left",fontFace:i?NUM:KR,color:i===2?INK:INK3,bold:i===2}}))),
  ], { x: M, y: 2.45, w: 12.1, colW:[3.5,4.3,4.3], rowH:0.48, fontFace:KR, fontSize:12, valign:"middle",
       border:{type:"solid",color:"E3E9EC",pt:1} });
  note(s, M, 5.42, 12.1, 1.72, "왜 이렇게 크게 갈리나",
    "NA 304행은 대부분 산림 내부·판독 범위 밖처럼 prior가 어차피 낮게 매기는 곳이다. 그것을 전부 “정답 음성”으로 만들면 prior에 유리한 쉬운 문제가 293개 추가된다." + NL +
    "그러면 prior AUC가 0.797 → 0.878로 오르고, 확률을 크게 부풀리는 조건(c=1, A fixed)이 상대적으로 잘 맞아 보인다. 정답 정의 하나가 결론 전체를 좌우했다.", WARN);
}

pres.writeFile({ fileName: "후속실험3.pptx" }).then(f => console.log("생성:", f));
