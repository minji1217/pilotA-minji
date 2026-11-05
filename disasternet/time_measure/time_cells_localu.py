# 칸 단위 LOCALU를 3000 epoch 끝까지 학습하며 학습·추론 시간을 따로 잰다 (seed 0, 지진 하나씩 차례로, 4스레드).
# 지진 하나가 끝날 때마다 시간 · 파라미터 · loss 기록을 바로 저장하고, 이미 끝난 지진은 건너뛴다.
import time, sys, csv, os, torch
sys.path.insert(0,'.')
from disasternet.run_cells_max import load_cells, to_rows, _slice_model_posteriors, EVS
from disasternet.model import DisasterNet, prepare_inputs, triplet_penalty
torch.set_num_threads(4)
E=3000
D='outputs/disasternet/time_localu_full'; os.makedirs(D,exist_ok=True)
OUT=f'{D}/time_localu_full.csv'
SEL=sys.argv[1].split(',') if len(sys.argv)>1 else EVS
done=set()
if os.path.exists(OUT):
    done={r['지진'] for r in csv.DictReader(open(OUT,encoding='utf-8'))}
t0=time.time()
C=load_cells('raw/local_250m_byevent_usgs_aw.csv.gz','raw/dpm_250m_v8_slim.csv.gz',False,'paper')
print(f'자료 읽기 {time.time()-t0:.1f}s',flush=True)
for ev in [e for e in EVS if e in SEL and e not in done]:
    tag=ev.replace(' ','')
    c=C[C['이벤트']==ev].reset_index(drop=True)
    rows=to_rows(c,'DPM_배경차',unc=True,unc_mode='both',local_off=True)
    torch.manual_seed(0)
    a,log_x=prepare_inputs(rows,2.0)
    model=DisasterNet(rows.n,num_flows=4,r0_var=1.0,use_bd=False)
    opt=torch.optim.Adam(model.parameters(),lr=0.01)
    hist=[]
    t_start=t=time.time()
    for epoch in range(E):
        beta=min(0.01+epoch/E,1.0)
        post=model.posteriors(a['LS'],a['LQ'],a['BD'],8,off=rows.off,r0_var=rows.r0_var)
        q,keep=post['q'],post['keep']
        rec=model.log_p_x(log_x,q)
        log_q=sum(post['log_q'][k].mean(1)*keep[k] for k in model.nodes)
        kl=log_q-model.log_p_z(q,a,rows.log_frac,keep)
        trip=triplet_penalty(q,a,1.0)
        loss=(-rec+beta*kl).sum()+trip
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append((epoch,float(loss.detach()),float((-rec).detach().sum()),float(kl.detach().sum()),float(trip.detach()),beta))
        if epoch%500==0: print(f'  {ev} epoch {epoch}  {time.time()-t:.0f}s',flush=True)
    tr=time.time()-t
    t=time.time()
    with torch.no_grad():
        for s in range(0,rows.n,50000):
            _slice_model_posteriors(model,a,s,min(s+50000,rows.n),256,rows)
    inf=time.time()-t
    torch.save({'model':model.state_dict(),'event':ev,'seed':0,'epochs':E},f'{D}/model_{tag}_seed0.pt')
    with open(f'{D}/loss_{tag}_seed0.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['epoch','loss','neg_rec','kl','triplet','beta']); w.writerows(hist)
    new=not os.path.exists(OUT)
    with open(OUT,'a',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        if new: w.writerow(['지진','칸','학습_초','추론_초','시작','끝'])
        w.writerow([ev,rows.n,f'{tr:.1f}',f'{inf:.2f}',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(t_start)),time.strftime('%Y-%m-%d %H:%M:%S')])
    print(f'{ev} 칸 {rows.n}  학습 {tr:.1f}초 ({tr/60:.1f}분, 3000 epoch 실측)  추론 {inf:.2f}초',flush=True)
