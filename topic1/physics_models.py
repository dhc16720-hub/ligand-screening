"""V4 representation controls and conservative inner-validation gate."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier,RandomForestRegressor,RandomForestClassifier
from topic1.refine_models import make_model as boost_model, configs as boost_configs

TREE_CONFIGS=[{'max_features':1.,'min_samples_leaf':1}, {'max_features':.7,'min_samples_leaf':1}, {'max_features':.4,'min_samples_leaf':2}]


def configs(classification):
    return {'ET':TREE_CONFIGS,'RF':[{'max_features':1.,'min_samples_leaf':1},{'max_features':.5,'min_samples_leaf':1},{'max_features':'sqrt','min_samples_leaf':2}],
            'CatBoost':boost_configs(classification)['CatBoost']}


def make_model(family,params,classification,seed=42):
    if family=='CatBoost': return boost_model(family,params,classification,seed)
    cls={'ET':(ExtraTreesRegressor,ExtraTreesClassifier),'RF':(RandomForestRegressor,RandomForestClassifier)}[family][int(classification)]
    return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),cls(n_estimators=128,random_state=seed,n_jobs=4,**params))


def gate(y,old_oof,new_oof,inner,classification):
    from topic1.advanced import loss
    old=np.array([loss(y[b],old_oof[b],classification) for _,b in inner])
    new=np.array([loss(y[b],new_oof[b],classification) for _,b in inner])
    gains=old-new
    threshold=float(np.std(gains,ddof=1)/np.sqrt(len(gains)))
    accepted=bool(np.mean(gains)>threshold and np.count_nonzero(gains>0)>=2)
    return dict(use_new=accepted,old_inner_losses=old.tolist(),new_inner_losses=new.tolist(),
                mean_gain=float(np.mean(gains)),threshold=threshold,improved_inner_folds=int(np.count_nonzero(gains>0)),
                rule='Accept D only when mean paired inner loss gain > paired SE and at least 2/3 inner folds improve; heuristic, not a statistical test')
