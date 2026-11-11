"""Isolate robust scaling from tuning in the previously diagnosed failure."""
import json
import numpy as np
import pandas as pd
from matbench.task import MatbenchTask
from threadpoolctl import threadpool_limits
from topic1.run import OUT as V1,scores,write_json
from topic1.advanced import OUT,make_model


def main():
    task=MatbenchTask('matbench_expt_gap');name=task.dataset_name
    matrix=np.load(V1/name/'features_C.npy')
    dest=OUT/name/'linear_ablation';dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    for fold in task.folds:
        train,y=task.get_train_and_val_data(fold);test=task.get_test_data(fold)
        model=make_model('Linear',{'regularization':10.},False)
        with threadpool_limits(limits=1):
            model.fit(matrix[task.df.index.get_indexer(train.index)],y)
            pred=model.predict(matrix[task.df.index.get_indexer(test.index)])
        _,yt=task.get_test_data(fold,include_target=True)
        pd.DataFrame(dict(mbid=test.index,target=yt,prediction=pred)).to_csv(dest/f'fold_{fold}.csv',index=False)
        s=scores(yt,pred,False);rows.append(dict(setting='Robust_alpha10',fold=fold,**s))
        for setting,base in [('Standard_alpha10',V1),('Robust_tuned',OUT)]:
            r=json.loads((base/name/'runs/C/Linear'/f'fold_{fold}.json').read_text())
            rows.append(dict(setting=setting,fold=fold,**r['metrics']))
    pd.DataFrame(rows).to_csv(OUT/'linear_ablation_metrics.csv',index=False)
    write_json(dest/'protocol.json',dict(new_fits=5,task=name,representation='C',alpha=10,
               purpose='fixed-alpha scaling control; exploratory diagnosis; no test-based selection',
               note='Constant-feature filtering is also present in Robust; constant columns carry no Ridge signal.'))
    print('Fixed-alpha scaling ablation complete: five official folds, five new fits.')


if __name__=='__main__':main()
