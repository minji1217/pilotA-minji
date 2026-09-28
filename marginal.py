import torch
from schema import LATENT_STATES

# marginal(주변화)는 이제 민지님께서 likeligood.py에서 전달 해주신 L=[B,4]를
# 내가 만든 prior.py에서 만들어진 w=[B,4]와 곱하고 다 더하는 작업
# P(y)=w00*L00 + w10*L10 + w01*L01 + w11*L11

# 후속실험 5: LS 라벨이 있는 행은 LS를 라벨값으로 고정하고 LQ만 합산한다.
#   라벨 있음: log sum_q P(LS=l)P(LQ=q)p(y|l,q)
#   라벨 없음: log sum_s sum_q P(LS=s)P(LQ=q)p(y|s,q)   (기존과 같음)
# LS가 라벨과 다른 상태를 -inf로 가리면 logsumexp가 위 식 그대로가 된다.
_LS_OF_STATE = torch.tensor([ls for ls, _ in LATENT_STATES])


def marginalize(log_w, log_L, ls_label=None):
    log_wl=log_w+log_L
    if ls_label is not None:
        labeled = (ls_label >= 0).unsqueeze(-1)                                   # [B,1]
        mismatch = _LS_OF_STATE.to(ls_label.device).unsqueeze(0) != ls_label.unsqueeze(-1)  # [B,4]
        log_wl = log_wl.masked_fill(labeled & mismatch, float("-inf"))
    log_Py=torch.logsumexp(log_wl,dim=-1)
    return log_wl,log_Py

#민망할 정도로 짧다...