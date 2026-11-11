"""Post-hoc failure analysis; does not change or select baseline settings."""
import json
import numpy as np
import pandas as pd
from matbench.task import MatbenchTask
from topic1.run import OUT,pipeline,write_json


def main():
    task=MatbenchTask('matbench_expt_gap')
    folder=OUT/task.dataset_name
    information=json.loads((folder/'features.json').read_text())
    evidence=[]
    for rep in ['B','C']:
        records=[json.loads(p.read_text()) for p in (folder/'runs'/rep/'Linear').glob('*.json')]
        fold=max(records,key=lambda r:r['metrics']['mae'])['fold']
        matrix=np.load(folder/f'features_{rep}.npy')
        train,y=task.get_train_and_val_data(fold)
        test=task.get_test_data(fold)
        ai=task.df.index.get_indexer(train.index);bi=task.df.index.get_indexer(test.index)
        pipe=pipeline('Linear',False).fit(matrix[ai],y)
        prediction=pipe.predict(matrix[bi])
        saved=pd.read_csv(folder/'runs'/rep/'Linear'/f'fold_{fold}.csv')
        np.testing.assert_allclose(prediction,saved.prediction,rtol=1e-8,atol=1e-8)
        worst=int(np.argmax(saved.abs_error))
        transformed=pipe[:-1].transform(matrix[bi])
        contributions=transformed[worst]*pipe[-1].coef_
        selected=np.argsort(np.abs(contributions))[-8:][::-1]
        row=dict(representation=rep,fold=int(fold),sample=str(test.index[worst]),formula=str(saved.formula.iloc[worst]),
                 target=float(saved.target.iloc[worst]),prediction=float(prediction[worst]),
                 intercept=float(pipe[-1].intercept_),
                 contributions=[dict(feature=information[rep]['names'][int(k)],
                                     train_mean=float(pipe[1].mean_[k]),train_scale=float(pipe[1].scale_[k]),
                                     train_min=float(np.nanmin(matrix[ai,k])),train_max=float(np.nanmax(matrix[ai,k])),
                                     test_value=float(matrix[bi[worst],k]),standardized_value=float(transformed[worst,k]),
                                     coefficient=float(pipe[-1].coef_[k]),contribution=float(contributions[k])) for k in selected])
        evidence.append(row)
    write_json(OUT/'ridge_failure_diagnostic.json',dict(purpose='post-hoc explanation only; no settings changed',diagnostic_refits=2,results=evidence))
    print(json.dumps(evidence,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
