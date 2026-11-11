"""Verify refactored frozen procedure reproduces original V4 fold zero."""
import json
import joblib
import numpy as np
from topic1.run import TASKS,write_json
from topic1.final_validation import register,data,take,OUT,V4,read,store_fit
from topic1.final_models import fit_procedure

register()
for name in TASKS:
    folder=OUT/name/'equivalence'
    if (folder/'result.json').exists():continue
    task,matrices,info,rep=data(name);cl=task.metadata.task_type=='classification'
    train,y=task.get_train_and_val_data(0);test=task.get_test_data(0)
    ai=task.df.index.get_indexer(train.index);bi=task.df.index.get_indexer(test.index)
    bundle,record,oofs=fit_procedure(take(matrices,ai),y,info.loc[train.index,'formula'].to_numpy(),cl,0,42)
    p=bundle.predict(take(matrices,bi));old=read(V4/name/'runs/V4/Selected/fold_0.csv')
    np.testing.assert_allclose(p,old.prediction,rtol=1e-10,atol=1e-8)
    store_fit(folder,bundle,record,oofs,train.index,y)
    write_json(folder/'result.json',dict(pass_all=True,fold=0,max_absolute_prediction_difference=float(np.max(np.abs(p-old.prediction.to_numpy()))),
                                       inner_fits=record['inner_fits'],refits=record['refits']))
    print('EQUIVALENCE OK',name,flush=True)
