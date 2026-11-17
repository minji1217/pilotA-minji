# pilotA A2 분리 학습(지진 5번) 학습 · 추론 시간. 사용: python time_pilota.py <pilotA 폴더 (followup4-lq-J-nolabels 브랜치)> [all8|dpm4] [lqlab]
# lqlab: 학습 지진의 액상화 라벨도 학습에 쓰는 판 (AdminDER + 액상화 라벨 학습)
# grid : 평가 지진 α_e를 원고 각주와 같은 격자 탐색으로 맞춘다 ([-8, 4]를 0.05 간격 → 최고점 ±0.05 안을 0.001 간격).
#        추론 = 전체 행 사후 계산 + α_e 격자 맞춤 + 평가 지진 사후 + LS · LQ AUC + 평가 지진 사후 CSV 저장.
#        grid가 없으면 예전처럼 train.report_holdout(α_e를 LBFGS로 맞춤, 여러 비교 AUC와 CSV 저장 포함)을 잰다.
import time, sys, os, io, contextlib, torch
W=sys.argv[1]; os.chdir(W); sys.path.insert(0,W)
import train as T
from loader import load_pilot_a_batch, load_eval_ground_truth, attach_ls_labels, subset_batch
from schema import EVENT_TO_INDEX
torch.set_num_threads(4)
t=time.time()
batch=load_pilot_a_batch(T.STATS_PATH,T.USGS_PATH); eval_gt=load_eval_ground_truth(T.GT_PATH,batch)
batch=attach_ls_labels(T.GT_PATH,batch,eval_gt)
print(f'자료 읽기 {time.time()-t:.1f}s')
tt=ti=0
DPM=["2016 구마모토","2018 오사카","2018 훗카이도","2021 후쿠시마","2024 노토반도"]
MODE=sys.argv[2] if len(sys.argv)>2 else "all8"
LQLAB="lqlab" in sys.argv[3:]
GRID="grid" in sys.argv[3:]
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
idx=torch.tensor([EVENT_TO_INDEX[e] for e in DPM])
for ev in DPM:
    keep=(batch.event_idx!=EVENT_TO_INDEX[ev]) if MODE=="all8" else (torch.isin(batch.event_idx,idx)&(batch.event_idx!=EVENT_TO_INDEX[ev]))
    tb=subset_batch(batch,keep)
    t=time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        reg,like,pri,hist=T.train(batch=tb,seed=0,epochs=3000,lr=0.02,lam_gamma=10,area_mode="fixed-cov-mtn",lam_kappa=10,
                                  mtn_standardize="train",use_labels=True,lq_labels=LQLAB,lq_b_only=True)
    tr=time.time()-t
    t=time.time()
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        out_r=reg(batch); out_l=like(batch,out_r.mu); log_w=T.prior_log_w(pri,batch)
        lj,lp=T.marginalize(log_w,out_l.log_L); p_ls,p_lq=T.infer(lj,lp)
    if GRID:
        with torch.no_grad():
            ho=subset_batch(batch,batch.event_idx==EVENT_TO_INDEX[ev]); lw=T.prior_log_w(pri,ho); mu0=reg(ho).mu
            ll=lambda a: float(T.marginalize(lw,like(ho,mu0*np.exp(a)).log_L)[1].sum())   # a = 학습값에서 옮기는 양
            k=float(reg.alpha_event[EVENT_TO_INDEX[ev]])
            g=max(np.arange(-8.0,4.0+1e-9,0.05),key=lambda v: ll(v-k)); a=max(np.arange(g-0.05,g+0.05+1e-9,0.001),key=lambda v: ll(v-k))
            lj,lp=T.marginalize(lw,like(ho,mu0*np.exp(a-k)).log_L); pls,plq=T.infer(lj,lp)
        auc={}
        for hz,p,m,y in [("LS",pls,ho.ls_label_mask,ho.ls_label),("LQ",plq,ho.lq_label_mask,ho.lq_label)]:
            m=m.numpy(); yy=y.numpy()[m]
            auc[hz]=roc_auc_score(yy,p.numpy()[m]) if len(set(yy))==2 else float("nan")
        pd.DataFrame({"muni_code":ho.municipality_code,"p_ls":pls.numpy(),"p_lq":plq.numpy()}).to_csv(
            f"outputs/timing_grid_{MODE}{'_lqlab' if LQLAB else ''}_{ev.replace(' ','')}.csv",index=False)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            T.report_holdout(batch,reg,like,pri,p_ls,p_lq,ev,"_timing_"+MODE+("_lqlab" if LQLAB else ""))
    inf=time.time()-t
    tt+=tr; ti+=inf
    print(f'{ev} 학습행 {tb.batch_size}  학습 {tr:.1f}s  추론(α_e 맞춤 포함) {inf:.2f}s'+(f'  α_e(격자) {a:+.3f}  AUC LS {auc["LS"]:.4f} LQ {auc["LQ"]:.4f}' if GRID else ''))
print(f'합계: 학습 {tt:.1f}s ({tt/60:.1f}분)  추론 {ti:.2f}s')
