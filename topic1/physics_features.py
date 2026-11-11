"""Stateless, input-only V4 physical descriptors; never drop failed samples."""
import json
import os
from pathlib import Path
import warnings
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'outputs/topic1_v4/mpl-cache'))
import numpy as np
from pymatgen.core import Composition
from matbench.task import MatbenchTask
from matminer.featurizers.composition import Miedema, YangSolidSolution
from matminer.featurizers.structure import SiteStatsFingerprint
from topic1.run import ROOT, OUT as V1, write_json, sha

OUT = ROOT/'outputs/topic1_v4'
TARGETS = ['matbench_steels','matbench_jdft2d','matbench_glass']


def alloy_features(inputs):
    m = Miedema(impute_nan=False)
    y = YangSolidSolution(impute_nan=False)
    rows, issues = [], []
    for mbid, value in inputs.items():
        comp = Composition(value)
        row = []
        try:
            row.extend(m.featurize(comp))
        except Exception as exc:
            row.extend([np.nan]*3)
            issues.append(dict(mbid=mbid,block='Miedema',reason=str(exc)))
        # Yang's implementation clamps a zero enthalpy denominator. Explicitly
        # reject missing binary parameters before that clamp could mask NaNs.
        elements = list(comp.elements)
        pairs = [y.dhf_mix.get_mixing_enthalpy(a,b) for i,a in enumerate(elements) for b in elements[:i]]
        try:
            omega = y.compute_omega(comp) if np.isfinite(pairs).all() else np.nan
            delta = y.compute_delta(comp)
        except Exception as exc:
            omega, delta = np.nan, np.nan
            issues.append(dict(mbid=mbid,block='Yang',reason=str(exc)))
        row.extend([omega,delta])
        if not np.isfinite(row).all():
            issues.append(dict(mbid=mbid,block='finite_check',reason='Unavailable elemental or binary parameters; retained as NaN'))
        rows.append(row)
    return np.array(rows,float), m.feature_labels()+y.feature_labels(), issues


def structure_features(inputs,n_jobs=4):
    f=SiteStatsFingerprint.from_preset('CrystalNNFingerprint_ops',stats=['mean','std_dev'])
    f.set_n_jobs(n_jobs)
    values=f.featurize_many(list(inputs),ignore_errors=True,return_errors=True,pbar=False)
    errors=[dict(mbid=mbid,block='CrystalNN',reason=row[-1]) for mbid,row in zip(inputs.index,values) if isinstance(row[-1],str)]
    return np.array([row[:-1] for row in values],float), f.feature_labels(), errors


def prepare(name):
    folder=OUT/name
    path=folder/'features_D.npy'
    if path.exists():
        info=json.loads((folder/'features.json').read_text())
        assert info['source_sha256']==sha(__file__) and info['matrix_sha256']==sha(path)
        return
    task=MatbenchTask(name)
    base_rep='C_structure' if name=='matbench_jdft2d' else 'C'
    inputs=task.df[task.metadata.input_type]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        extra,labels,issues=structure_features(inputs) if name=='matbench_jdft2d' else alloy_features(inputs)
    extra[~np.isfinite(extra)]=np.nan
    old=np.load(V1/name/f'features_{base_rep}.npy')
    matrix=np.column_stack([old,extra])
    folder.mkdir(parents=True,exist_ok=True)
    np.save(path,matrix)
    base_labels=json.loads((V1/name/'features.json').read_text())[base_rep]['names']
    write_json(folder/'features.json',dict(source_sha256=sha(__file__),matrix_sha256=sha(path),
               base_representation=base_rep,base_matrix_sha256=sha(V1/name/f'features_{base_rep}.npy'),
               ids=list(inputs.index),names=base_labels+labels,extra_names=labels,n_samples=len(inputs),dimension=matrix.shape[1],
               extra_missing_fraction=np.isnan(extra).mean(axis=0).tolist(),rows_with_any_extra_missing=int(np.isnan(extra).any(axis=1).sum()),
               warnings=sorted(set(str(w.message) for w in caught)),issues=issues,
               preprocessing='No fitting/target access. Missing values remain NaN until fold-local median imputation.'))
    print('FEATURES',name,matrix.shape,'rows with missing extras',np.isnan(extra).any(axis=1).sum(),flush=True)


if __name__=='__main__':
    for name in TARGETS: prepare(name)
