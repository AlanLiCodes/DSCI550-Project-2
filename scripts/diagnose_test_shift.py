"""Adversarial validation: how distinguishable are training mixtures from Test.csv windows?

Uses test *inputs* only (no labels) to judge how realistic the mixture design is.
Lower AUC = mixtures look more like the test windows. Paths below are local; edit them.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
rec,_=load_records('data/South_African_Real_Acoustics'); X,m=build_windows(rec)
Xt,mt,_=load_test('data/Test.csv')
# Remove the trivial missingness cue: compare on fully observed minutes only by masking test-style? Simply drop missing features.
fb=FeatureBuilder(station_aware=False).fit(X,m); Ft=fb.transform(Xt,mt)
drop=[c for c in Ft.columns if 'missing' in c]
refs={b:load_vtuad_references('data/references/vtuad',background=b) for b in ['onc','none']}
def adv(Fm):
    A=pd.concat([Fm.drop(columns=drop),Ft.drop(columns=drop)]); lab=np.r_[np.zeros(len(Fm)),np.ones(len(Ft))]
    p=cross_val_predict(HistGradientBoostingClassifier(max_iter=150,random_state=0),A,lab,cv=5,method='predict_proba')[:,1]
    return roc_auc_score(lab,p)
rows=[]
for name,b,sb,st,atb,pz in [('onc_sub,bal.5',  'onc',0.5,1,(-20,20),.2),
                            ('onc_sub,bal0',   'onc',0.0,1,(-20,20),.2),
                            ('raw,bal0',       'none',0.0,1,(-20,20),.2),
                            ('raw,bal0,str4',  'none',0.0,4,(-20,20),.2),
                            ('raw,bal0,atb-20..10','none',0.0,1,(-20,10),.2),
                            ('raw,bal0,atb-10..20','none',0.0,1,(-10,20),.2),
                            ('raw,bal0,pz0',   'none',0.0,1,(-20,20),0.0)]:
    Xm,y,pres,info=make_mixtures(X,m,refs[b],4000,seed=3,station_balance=sb,stretch_range=(1,st),atb_range_db=atb,p_zero=pz)
    Fm=fb.transform(Xm,info)
    rows.append(dict(variant=name,adv_auc=round(adv(Fm),3),label_mean=round(y.mean(),3),
      high_pk=round(Fm.high_peak_minus_median.median(),1),lmh=round(Fm.low_minus_high.median(),1),pk=round(Fm.bb_peak_minus_median.median(),1)))
    print(rows[-1],flush=True)
print('test: high_pk %.1f lmh %.1f pk %.1f'%(Ft.high_peak_minus_median.median(),Ft.low_minus_high.median(),Ft.bb_peak_minus_median.median()))
