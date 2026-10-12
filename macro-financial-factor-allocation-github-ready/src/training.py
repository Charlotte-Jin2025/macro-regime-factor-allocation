"""Initial-sample diagnostics and a fixed 12-month-block predictor permutation."""
from pathlib import Path
import numpy as np
import pandas as pd
from . import us_core as u

def run_analysis(base):
    base=Path(base);out=base/'results/training';out.mkdir(parents=True,exist_ok=True)
    data,_=u.load_data(base)
    x=u.feature_data(data)
    train=x.loc[:u.TRAIN_END].dropna(subset=u.PREDICTORS+[f+'_next' for f in u.FACTORS])
    regressions,adf,corr=u.training_diagnostics(x)
    regressions.to_csv(out/'macro_regression.csv',index=False)
    adf.to_csv(out/'adf.csv',index=False)
    corr.to_csv(out/'predictor_correlation.csv')
    X=np.column_stack([np.ones(len(train)),train[u.PREDICTORS].to_numpy()])
    Y=train[[f+'_next' for f in u.FACTORS]].to_numpy(float)
    def r2(a):
        prediction=a@np.linalg.lstsq(a,Y,rcond=None)[0]
        return 1-((Y-prediction)**2).sum(axis=0)/((Y-Y.mean(axis=0))**2).sum(axis=0)
    observed=r2(X);rng=np.random.default_rng(u.SEED)
    chunks=[np.arange(i,min(i+12,len(train))) for i in range(0,len(train),12)]
    simulated=np.asarray([r2(X[np.concatenate([chunks[i] for i in rng.permutation(len(chunks))])]) for _ in range(500)])
    pd.DataFrame(dict(Factor=u.FACTORS,Training_R2=observed,
        Block_Permutation_Share_At_Least_Observed=(1+(simulated>=observed).sum(axis=0))/501,
        Draws=500,Block_Length=12,N=len(train))).to_csv(out/'training_block_placebo.csv',index=False)
    pd.DataFrame(simulated,columns=u.FACTORS).to_csv(out/'training_block_placebo_distribution.csv',index=False)
