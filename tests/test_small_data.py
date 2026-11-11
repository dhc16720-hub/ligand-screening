import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from experiments import small_data as study


class SmallDataTests(unittest.TestCase):
    def test_subsets_are_nested_and_full_data_is_not_permuted(self):
        for n in [249,508,3683]:
            for seed in study.SEEDS:
                a=study.subset_positions(n,.2,seed,2)
                b=study.subset_positions(n,.5,seed,2)
                np.testing.assert_array_equal(a,b[:len(a)])
                self.assertEqual(len(set(b)),len(b))
                np.testing.assert_array_equal(study.subset_positions(n,1,seed,2),np.arange(n))

    def test_full_points_not_triplicated_and_all_folds_present(self):
        configs=list(study.configurations())
        self.assertEqual(len(configs),1050)
        self.assertEqual(sum(c['fraction']==1 for c in configs),150)
        self.assertEqual(len({tuple(c.values()) for c in configs}),1050)
        for task in study.TASKS:
            self.assertEqual({c['fold'] for c in configs if c['task']==task},set(range(5)))

    def test_paired_gain_direction_and_fold_first_aggregation(self):
        rows=[]
        for task in [study.TASKS[0],study.TASKS[3]]:
            for fold in range(5):
                for fraction in study.FRACTIONS:
                    for seed in (study.SEEDS if fraction<1 else [42]):
                        for rep in study.REPRESENTATIONS:
                            base=10+fold if task==study.TASKS[0] else .7+.01*fold
                            gain=(2 if fraction<1 else 1) if task==study.TASKS[0] else .05
                            score=base if rep=='A' else base+(-gain if task==study.TASKS[0] else gain)
                            rows.append(dict(task=task,model='RF',representation=rep,fraction=fraction,fold=fold,
                                             subset_seed=seed,n_train=100,score=score))
        fold,summary,paired,contrast=study.paired_tables(pd.DataFrame(rows))
        self.assertTrue((paired.total_folds==5).all())
        selected=paired[(paired.task==study.TASKS[0])&(paired.comparison=='C-A')&(paired.fraction==.2)].iloc[0]
        self.assertEqual(selected.mean_improvement,2)
        self.assertEqual(selected.improved_folds,5)
        self.assertTrue((paired[paired.comparison=='C-A'].mean_improvement>0).all())
        self.assertEqual(contrast[(contrast.task==study.TASKS[0])&(contrast.comparison=='C-A')].iloc[0].low_data_gain_minus_full,1)
        self.assertEqual(set(fold.repeats),{1,3})

    def test_modified_checkpoint_cannot_be_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);config=next(study.configurations())
            for name in ['model.joblib','predictions.csv']:(folder/name).write_text('original')
            (folder/'result.json').write_text(json.dumps(dict(config=config,sha256={n:study.sha(folder/n) for n in ['model.joblib','predictions.csv']})))
            study.validate_checkpoint(folder,config)
            (folder/'predictions.csv').write_text('changed')
            with self.assertRaises(RuntimeError):study.validate_checkpoint(folder,config)


if __name__=='__main__': unittest.main()
