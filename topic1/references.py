"""Re-score public strong-model predictions; never call these local training."""
import gzip
import json
import shutil
import numpy as np
import pandas as pd
from matbench.task import MatbenchTask
from topic1.run import ROOT,TASKS,write_json,sha,scores
from topic1.advanced import OUT


def main():
    sources={'MODNet':'matbench_v0.1_modnet_v0.1.12','CrabNet':'matbench_v0.1_CrabNet'}
    provenance=[];rows=[]
    for label,directory in sources.items():
        folder=ROOT/'third_party/matbench/benchmarks'/directory
        raw=folder/'results.json.gz';data=json.load(gzip.open(raw,'rt'))
        dest=OUT/'references'/label;dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(folder/'info.json',dest/'source_info.json')
        record=dict(model=label,source_url=f'https://github.com/materialsproject/matbench/tree/07e50bcac791ce7660684d71e59646a1b6b82faa/benchmarks/{directory}',
                    original_results_sha256=sha(raw),origin='published_predictions_rescored_not_locally_trained',tasks=[])
        for name in TASKS:
            if name not in data['tasks']:continue
            task=MatbenchTask(name);cl=task.metadata.task_type=='classification'
            rec=MatbenchTask(name,autoload=False);rec.df=task.df;task_rows=[]
            for fold in task.folds:
                test,y=task.get_test_data(fold,include_target=True)
                predictions=data['tasks'][name]['results'][f'fold_{fold}']['data']
                assert set(predictions)==set(test.index)
                p=np.array([predictions[i] for i in test.index],float)
                probability=not np.isin(p,[0,1]).all() if cl else None
                metric=scores(y,p,cl)
                # Do not pretend hard-label AUROC is probability ranking quality.
                if cl and not probability:metric.pop('roc_auc')
                rows.append(dict(task=name,model=label,fold=fold,probability_available=probability,**metric))
                task_rows.append(pd.DataFrame(dict(mbid=test.index,fold=fold,target=np.asarray(y,float),prediction=p)))
                rec.record(fold,p,params={'origin':record['origin'],'source':record['source_url']})
            rec.validate();rec.to_file(str(dest/f'{name}.json.gz'))
            pd.concat(task_rows).to_csv(dest/f'{name}_predictions.csv',index=False)
            record['tasks'].append(name)
        provenance.append(record)
    write_json(OUT/'references/provenance.json',provenance)
    pd.DataFrame(rows).to_csv(OUT/'reference_fold_metrics.csv',index=False)
    print('Published predictions re-scored:',len(rows),'folds. These are external references, not new training.')


if __name__=='__main__':main()
