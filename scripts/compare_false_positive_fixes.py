"""Compare transient masking and hard-negative over-sampling on held-out mixtures and clean backgrounds.

Paths are local; edit them. Prints held-out score, St Francis scores, mean prediction on
clean held-out backgrounds and the mean prediction on Test.csv windows (inputs only).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from vesselnoise.data import load_records, load_test
from vesselnoise.windows import build_windows
from vesselnoise.vtuad import load_vtuad_references
from vesselnoise.mixtures import make_mixtures
from vesselnoise.features import FeatureBuilder
from vesselnoise.splits import chronological_split, assign_reference_splits

V5 = dict(station_balance=0, atb_range_db=(-20, 5), gain_db=10, bg_tilt_db=6, ref_tilt_db=6, ref_jitter_db=1,
          busy_p=0.4, busy_atb_range_db=(-15, 5))
from vesselnoise.windows import mask_run_start_transients
from vesselnoise.pipeline import hard_negative_weights
from vesselnoise.metrics import leaderboard_score
from sklearn.ensemble import HistGradientBoostingRegressor
rec,_=load_records('data/South_African_Real_Acoustics')
mrec,_=mask_run_start_transients(rec)
Xu,mu=build_windows(rec); Xc,mc=build_windows(mrec)
assert len(Xu)==len(Xc) and (mu.window_id.values==mc.window_id.values).all()
Xt,mt,_=load_test('data/Test.csv')
refs=load_vtuad_references('data/references/vtuad',background='none')
sp=chronological_split(mc); rs=assign_reference_splits(refs)
tr,te=sp=='train',sp=='test'
rtr=[r for r in refs if rs[r.vessel_id]=='train']; rte=[r for r in refs if rs[r.vessel_id]=='test']
mtr=mc[tr].reset_index(drop=True); mte=mc[te].reset_index(drop=True)
# common evaluation sets on CLEANED held-out data
Xe,ye,_,ie=make_mixtures(Xc[te],mte,rte,5000,seed=22,**V5)
sf_te=mte.station.str.contains('Francis').to_numpy()
hnw,oof=hard_negative_weights(Xc[tr],mtr,rtr,V5,strength=10,n=12000)
print('out-of-fold clean-bg predictions on TRAIN backgrounds: mean %.3f, St Francis %.3f'%(oof.mean(),oof[mtr.station.str.contains('Francis').to_numpy()].mean()))
print('hard-negative weight: St Francis mean %.2f, others %.2f'%(hnw[mtr.station.str.contains('Francis').to_numpy()].mean(),hnw[~mtr.station.str.contains('Francis').to_numpy()].mean()))
rows=[]
for name,Xtrain,w in [('v5 (no fix)',Xu[tr],None),('A: mask transients',Xc[tr],None),('A+B: + hard negatives',Xc[tr],hnw)]:
    Xa,ya,_,ia=make_mixtures(Xtrain,mtr,rtr,15000,seed=1,bg_weights=w,**V5)
    fb=FeatureBuilder(station_aware=False).fit(Xtrain,mtr)
    mdl=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,random_state=0).fit(fb.transform(Xa,ia),ya)
    P=lambda X_,m_: np.clip(mdl.predict(fb.transform(X_,m_)),0,1)
    pe=P(Xe,ie); pz=P(Xc[te],mte); pt=P(Xt,mt)
    st_sf=ie.station.str.contains('Francis').to_numpy()
    rows.append({'config':name,'heldout_score':round(leaderboard_score(ye,pe),4),'heldout_StFrancis_mix_score':round(leaderboard_score(ye[st_sf],pe[st_sf]),4),
                 'clean_bg_pred_all':round(pz.mean(),3),'clean_bg_pred_StFrancis':round(pz[sf_te].mean(),3),'clean_bg_pred_others':round(pz[~sf_te].mean(),3),
                 'test_pred_mean':round(pt.mean(),3),'train_StFrancis_share':round(ia.station.str.contains('Francis').mean(),3)})
    print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv('outputs/fp_fix.csv',index=False)
