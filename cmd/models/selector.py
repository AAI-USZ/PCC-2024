import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tqdm
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import compute_class_weight

import simtools


def model_selector(models: Iterable[ClassifierMixin], features=None):
    if features is None:
        features = slice(None)

    meta = simtools.misc.read_meta()
    df = pd.read_csv('data/features.v2.csv', sep=';')

    df = simtools.prep.misc.process_scrappy_dataset(df)

    X = df[features]
    y = df[meta.target]

    w = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y.to_numpy().ravel())
    w = {False: w[0], True: w[1]}

    def pipeline(model):
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

        return Pipeline([
            ('pre', ct),
            ('cls', model)
        ])

    output = []

    for model in tqdm.tqdm(models, desc='Model validation'):
        model.class_weight = w
        model = pipeline(model)
        results = cross_validate(
            model,
            X,
            y.to_numpy().ravel(),
            scoring=['f1', 'precision', 'recall', 'accuracy'],
            cv=StratifiedKFold(n_splits=10, shuffle=True),
            n_jobs=4,
        )

        result_df = pd.DataFrame.from_dict({'#': ['mean', 'min', 'max']} | {
            k.removeprefix('test_'): [v.mean(), v.min(), v.max()]
            for k, v in results.items()
            if k.startswith('test_')
        })

        output.append(result_df)

    return output


def generate_results():
    random.seed(4242)
    np.random.seed(4242)
    meta = simtools.misc.read_meta()
    for run_id in tqdm.tqdm(range(10), desc='Experiment'):
        models = [
            DecisionTreeClassifier(),
            GaussianNB(),
            KNeighborsClassifier(),
            LinearDiscriminantAnalysis(),
            LogisticRegression(max_iter=500),
            MLPClassifier(max_iter=500),
            RandomForestClassifier(),
            SGDClassifier(),
            SVC(),
        ]
        results = model_selector(models, features=meta.static + meta.tool + meta.dist + meta.embed)

        for res, model in zip(results, models):
            res.to_csv(f'.cache/model_selection/run:{run_id:02d}_model:{model.__class__.__name__}.csv', index=False)


def read_results():
    results = [
        pd.read_csv(f)
        for f in sorted(Path('.cache/model_selection').iterdir())
    ]
    return results


def mean_run_results():
    results = read_results()
    models = [
        'DecisionTreeClassifier', 'GaussianNB', 'KNeighborsClassifier',
        'LinearDiscriminantAnalysis', 'LogisticRegression', 'MLPClassifier',
        'RandomForestClassifier', 'SGDClassifier', 'SVC',
    ]
    mean_res = {m: np.asarray([df['f1'].values[0] for df in results[i::9]]).mean() for i, m in enumerate(models)}
    pd.DataFrame.from_dict({
        'model': mean_res.keys(), 'f1 mean': mean_res.values()
    }).to_csv('.cache/model_runs/mean_results.csv', index=False)
    #pd.DataFrame.from_dict(mean_res, orient='index').to_csv('.cache/model_runs/mean_results.csv', index=False)


def main():
    mean_run_results()
    print('End of program')


if __name__ == '__main__':
    main()
