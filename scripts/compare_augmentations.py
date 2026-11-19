"""Compare augmentation settings on common held-out mixtures, clean held-out backgrounds and Test.csv inputs.

Reports local scores, mean prediction on clean backgrounds (should be ~0), the
distribution of predictions on Test.csv windows (no labels used) and adversarial
AUC (training mixtures vs test windows). Paths below are local; edit them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vesselnoise.data import load_records, load_test
from vesselnoise.windows import build_windows
from vesselnoise.vtuad import load_vtuad_references
from vesselnoise.mixtures import make_mixtures
from vesselnoise.features import FeatureBuilder
from vesselnoise.splits import chronological_split, assign_reference_splits
from vesselnoise.metrics import leaderboard_score
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
rec,_=load_records('data/South_African_Real_Acoustics'); X,m=build_windows(rec)
Xt,mt,_=load_test('data/Test.csv')
refs=load_vtuad_references('data/references/vtuad',background='none')
sp=chronological_split(m); rs=assign_reference_splits(refs)
tr=sp=='train'; te=sp=='test'
rtr=[r for r in refs if rs[r.vessel_id]=='train']; rte=[r for r in refs if rs[r.vessel_id]=='test']
mtr=m[tr].reset_index(drop=True); mte=m[te].reset_index(drop=True)
base=dict(station_balance=0,atb_range_db=(-20,10))
LOUD=dict(gain_db=10,bg_tilt_db=6,ref_tilt_db=6,ref_jitter_db=1)
BUSY=dict(busy_p=0.4,busy_atb_range_db=(-15,5))
fb=FeatureBuilder(station_aware=False).fit(X[tr],mtr)
# common evaluation sets
Xe,ye,_,ie=make_mixtures(X[te],mte,rte,4000,seed=22,**base)                     # plain held-out
Xe2,ye2,_,ie2=make_mixtures(X[te],mte,rte,4000,seed=23,**base,**LOUD,**BUSY)   # augmented held-out
Fe,Fe2=fb.transform(Xe,ie),fb.transform(Xe2,ie2)
idx=np.flatnonzero(te); pick=np.random.default_rng(0).choice(idx,6000,replace=False)
mz=m.iloc[pick].reset_index(drop=True); Fz=fb.transform(X[pick],mz)
Ft=fb.transform(Xt,mt); drop=[c for c in Ft.columns if 'missing' in c]
rows=[]
for name,kw in [('none',{}),('loudness',LOUD),('busy',BUSY),('both',{**LOUD,**BUSY})]:
    Xa,ya,_,ia=make_mixtures(X[tr],mtr,rtr,15000,seed=1,**base,**kw)
    Fa=fb.transform(Xa,ia)
    mdl=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,random_state=0).fit(Fa,ya)
    P=lambda F: np.clip(mdl.predict(F),0,1)
    pz=P(Fz); pt=P(Ft)
    A=pd.concat([Fa.sample(4000,random_state=0).drop(columns=drop),Ft.drop(columns=drop)]); lab=np.r_[np.zeros(4000),np.ones(len(Ft))]
    adv=roc_auc_score(lab,cross_val_predict(HistGradientBoostingClassifier(max_iter=150,random_state=0),A,lab,cv=5,method='predict_proba')[:,1])
    rows.append(dict(aug=name,train_label_mean=round(ya.mean(),3),score_plain_heldout=round(leaderboard_score(ye,P(Fe)),4),
        score_aug_heldout=round(leaderboard_score(ye2,P(Fe2)),4),clean_bg_pred=round(pz.mean(),3),
        stfrancis_bg_pred=round(pz[mz.station.str.contains('Francis').to_numpy()].mean(),3),
        test_pred_mean=round(pt.mean(),3),test_pred_median=round(np.median(pt),3),test_gt_0p2=round((pt>0.2).mean(),2),adv_auc=round(adv,3)))
    print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv('outputs/aug_compare.csv',index=False)
