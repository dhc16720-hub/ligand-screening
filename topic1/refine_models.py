"""Importable V3 estimators and invertible target transforms."""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


def signed_log1p(y):
    return np.sign(y) * np.log1p(np.abs(y))


def signed_expm1(y):
    return np.sign(y) * np.expm1(np.abs(y))


def configs(classification):
    result = {'CatBoost': [dict(depth=d, loss_function='Logloss' if classification else loss)
                            for d, loss in [(4, 'MAE'), (6, 'MAE'), (6, 'RMSE')]]}
    if classification:
        result['CatBoost'][-1]['depth'] = 8
    else:
        for name in ['XGB_Squared', 'XGB_MAE', 'XGB_Log']:
            result[name] = [dict(max_depth=d) for d in [3, 5, 7]]
    return result


def make_model(family, params, classification, seed=42):
    if family == 'CatBoost':
        from catboost import CatBoostClassifier, CatBoostRegressor
        cls = CatBoostClassifier if classification else CatBoostRegressor
        model = cls(iterations=512, learning_rate=.05, random_seed=seed,
                    thread_count=4, verbose=False, allow_writing_files=False,
                    l2_leaf_reg=3, **params)
    else:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=512, learning_rate=.05, random_state=seed,
                             n_jobs=4, tree_method='hist', subsample=1., colsample_bytree=.9,
                             min_child_weight=3, verbosity=0,
                             objective='reg:absoluteerror' if family == 'XGB_MAE' else 'reg:squarederror', **params)
        if family == 'XGB_Log':
            model = TransformedTargetRegressor(regressor=model, func=signed_log1p, inverse_func=signed_expm1)
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), model)


def predict(model, x, classification):
    p = model.predict_proba(x)[:, list(model[-1].classes_).index(True)] if classification else model.predict(x)
    return np.asarray(p, dtype=float)


def combine(predictions, recipe):
    values = np.array([predictions[f] for f in recipe['members']], dtype=float)
    return np.median(values, axis=0) if recipe['operation'] == 'median' else np.mean(values, axis=0)


def select(y, oofs, inner, classification):
    from topic1.advanced import mean_inner_loss
    ranked = sorted(oofs, key=lambda f: mean_inner_loss(y, oofs[f], inner, classification))
    recipes = [dict(members=ranked[:n], operation='mean') for n in [1, 2, 3]]
    if not classification:
        recipes.append(dict(members=ranked[:3], operation='median'))
    for r in recipes:
        r['inner_loss'] = mean_inner_loss(y, combine(oofs, r), inner, classification)
    return min(recipes, key=lambda r: r['inner_loss']), recipes
