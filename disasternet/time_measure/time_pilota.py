# pilotA A2 분리 학습(지진 5번) 학습 · 추론 시간. 사용: python time_pilota.py <pilotA 폴더 (followup4-lq-J-nolabels 브랜치)>
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
for ev in ["2016 구마모토","2018 오사카","2018 훗카이도","2021 후쿠시마","2024 노토반도"]:
    tb=subset_batch(batch,batch.event_idx!=EVENT_TO_INDEX[ev])
    t=time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        reg,like,pri,hist=T.train(batch=tb,seed=0,epochs=3000,lr=0.02,lam_gamma=10,area_mode="fixed-cov-mtn",lam_kappa=10,
                                  mtn_standardize="train",use_labels=True,lq_labels=False,lq_b_only=True)
    tr=time.time()-t
    t=time.time()
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        out_r=reg(batch); out_l=like(batch,out_r.mu); log_w=T.prior_log_w(pri,batch)
        lj,lp=T.marginalize(log_w,out_l.log_L); p_ls,p_lq=T.infer(lj,lp)
    with contextlib.redirect_stdout(io.StringIO()):
        T.report_holdout(batch,reg,like,pri,p_ls,p_lq,ev,"_timing")
    inf=time.time()-t
    tt+=tr; ti+=inf
    print(f'{ev}  학습 {tr:.1f}s  추론(α_e 맞춤 포함) {inf:.2f}s')
print(f'합계: 학습 {tt:.1f}s ({tt/60:.1f}분)  추론 {ti:.2f}s')
