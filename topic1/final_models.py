"""Frozen V4 selection procedure fitted on an arbitrary proper-training set."""
from dataclasses import dataclass
import numpy as np
from threadpoolctl import threadpool_limits
from topic1 import advanced as v2
from topic1 import refine_models as v3
from topic1 import physics_models as v4


@dataclass
class FinalBundle:
    models: dict
    recipe: dict
    classification: bool

    def predict(self, matrices):
        predictions={key:v3.predict(model,matrices[key.split(':')[0]],self.classification) for key,model in self.models.items()}
        return v3.combine(predictions,self.recipe)


def specifications(classification,has_physics):
    specs={f'C:{f}':configs for f,configs in v2.CONFIGS.items()}
    specs.update({f'C:{f}':configs for f,configs in v3.configs(classification).items()})
    if has_physics:specs.update({f'D:{f}':configs for f,configs in v4.configs(classification).items()})
    return specs


def estimator(key,parameters,classification,seed):
    rep,family=key.split(':')
    if rep=='D':return v4.make_model(family,parameters,classification,seed)
    if family in v2.FAMILIES:return v2.make_model(family,parameters,classification,seed)
    return v3.make_model(family,parameters,classification,seed)


def fit_procedure(matrices,y,groups,classification,fold=0,seed=42):
    """No evaluation targets or calibration inputs accepted by this function."""
    y=np.asarray(y);inner,grouped=v2.inner_folds(y,np.asarray(groups),classification,fold)
    oofs={};traces={};params={}
    specs=specifications(classification,'D' in matrices)
    for key,configs in specs.items():
        x=matrices[key.split(':')[0]];candidates=[];predictions=[]
        for setting in configs:
            oof=np.full(len(y),np.nan)
            for a,b in inner:
                model=estimator(key,setting,classification,seed)
                with threadpool_limits(limits=1):model.fit(x[a],y[a]);oof[b]=v3.predict(model,x[b],classification)
            losses=[v2.loss(y[b],oof[b],classification) for _,b in inner]
            candidates.append(dict(parameters=setting,losses=losses,mean=float(np.mean(losses))))
            predictions.append(oof)
        winner=int(np.argmin([c['mean'] for c in candidates]));traces[key]=candidates
        params[key]=configs[winner];oofs[key]=predictions[winner]
    old={k:p for k,p in oofs.items() if k.startswith('C:')}
    recipe,old_candidates=v3.select(y,old,inner,classification)
    record=dict(search=traces,parameters=params,old_recipe=recipe,old_candidates=old_candidates,grouped=grouped,
                seed=seed,inner=[dict(train=a.tolist(),validation=b.tolist()) for a,b in inner],inner_fits=sum(len(c)*3 for c in specs.values()))
    if 'D' in matrices:
        new={k:p for k,p in oofs.items() if k.startswith('D:')}
        new_recipe,new_candidates=v3.select(y,new,inner,classification)
        decision=v4.gate(y,v3.combine(old,recipe),v3.combine(new,new_recipe),inner,classification)
        record.update(new_recipe=new_recipe,new_candidates=new_candidates,gate=decision)
        if decision['use_new']:recipe=new_recipe
    fitted={}
    for key in recipe['members']:
        model=estimator(key,params[key],classification,seed)
        with threadpool_limits(limits=1):model.fit(matrices[key.split(':')[0]],y)
        fitted[key]=model
    record.update(recipe=recipe,refits=len(fitted))
    return FinalBundle(fitted,recipe,classification),record,oofs
