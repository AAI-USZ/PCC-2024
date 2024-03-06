import random

import numpy as np
import pandas as pd
import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils import compute_class_weight

import simtools


def train_model(model, features=None):
    if features is None:
        features = slice(None)

    df = pd.read_csv('../../data/ALL_features.csv', sep=';')

    df = simtools.prep.misc.process_scrappy_dataset(df)

    meta = simtools.misc.read_meta()

    X = df[features]
    y = df[meta.target]

    w = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y.to_numpy().ravel())
    w = {False: w[0], True: w[1]}

    cv = StratifiedKFold(n_splits=10, shuffle=True)

    ct = ColumnTransformer(
        [
            (
                'cat',
                OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
                meta.multicat,
            ),
            (
                'std',
                StandardScaler(),
                list(set(features).difference(meta.multicat))
            ),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    )
    ct.set_output(transform='pandas')

    X = ct.fit_transform(X)
    y = y.to_numpy().ravel()

    results = cross_validate(
        model,
        X,
        y,
        scoring=['f1', 'precision', 'recall', 'accuracy'],
        cv=cv,
        n_jobs=4,
    )

    result_df = pd.DataFrame.from_dict({'#': ['mean', 'min', 'max']} | {
        k.removeprefix('test_'): [v.mean(), v.min(), v.max()]
        for k, v in results.items()
        if k.startswith('test_')
    })

    return result_df


def train_ensemble(features=None):
    if features is None:
        features = slice(None)

    df = pd.read_csv('../../data/ALL_features.csv', sep=';')

    df = simtools.prep.misc.process_scrappy_dataset(df)

    meta = simtools.misc.read_meta()

    X = df[features]
    y = df[meta.target]

    w = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y.to_numpy().ravel())
    w = {False: w[0], True: w[1]}

    cv = StratifiedKFold(n_splits=10, shuffle=True)

    ct = ColumnTransformer(
        [
            #(
            #    'cat',
            #    OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
            #    meta.multicat,
            #),
            (
                'std',
                StandardScaler(),
                list(set(features).difference(meta.multicat))
            ),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    )
    ct.set_output(transform='pandas')

    X = ct.fit_transform(X)
    y = y.to_numpy().ravel()

    models = [
        LogisticRegression(max_iter=500),
        MLPClassifier(max_iter=500),
        RandomForestClassifier(),
        SGDClassifier(),
        SVC(),
    ]
    stacking = StackingClassifier(
        estimators=[(model.__class__.__name__, model) for model in models],
        final_estimator=MLPClassifier(hidden_layer_sizes=tuple(), max_iter=500),
        cv=cv,
        n_jobs=4,
    )

    results = cross_validate(
        estimator=stacking,
        X=X,
        y=y,
        scoring=['f1', 'precision', 'recall', 'accuracy'],
        cv=cv,
        n_jobs=4,
        return_estimator=True,
    )

    return results


def stacking_experiment():
    meta = simtools.misc.read_meta()

    results_list = []
    for run_id in tqdm.tqdm(range(10), desc='Stacking Experiment'):
        results = train_ensemble(meta.best_v1)
        results_list.append(results)


def mlp_experiment():
    meta = simtools.misc.read_meta()

    result_list = []
    for run_id in tqdm.tqdm(range(10), desc='Experiment'):
        results = train_model(MLPClassifier(max_iter=500), meta.best_v1)
        result_list.append(results)

    [el.to_csv(f'.cache/mlp_runs/best_v1/run:{i:02d}', index=False) for i, el in enumerate(result_list)]
    for el in result_list:
        print('-- ----------------------------------------------- --')
        print(el)
        print('** ----------------------------------------------- **')


def main():
    random.seed(4242)
    np.random.seed(4242)

    stacking_experiment()

    exit()


if __name__ == '__main__':
    main()
