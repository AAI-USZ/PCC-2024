import random

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import compute_class_weight

import simtools


def select_features():
    df = pd.read_csv('../../data/ALL_features.csv', sep=';')

    df = simtools.prep.misc.process_scrappy_dataset(df)

    meta = simtools.misc.read_meta()

    X = df[meta.all]
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
                list(set(meta.all).difference(meta.multicat))
            ),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    )
    ct.set_output(transform='pandas')

    X = ct.fit_transform(X)
    y = y.to_numpy().ravel()

    rfecv = RFECV(
        estimator=RandomForestClassifier(class_weight=w),
        step=10,
        cv=cv,
        scoring='f1',
        min_features_to_select=10,
        n_jobs=4,
    )
    rfecv.fit(X, y)

    selected = rfecv.get_feature_names_out()
    return selected


def feature_performance(feature_list):
    df = pd.read_csv('data/features.v2.csv', sep=';')

    df = simtools.prep.misc.process_scrappy_dataset(df)

    meta = simtools.misc.read_meta()
    result_list = []

    for feat in feature_list:
        X = df[meta.all]
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
                    list(set(meta.all).difference(meta.multicat))
                ),
            ],
            remainder='passthrough',
            verbose_feature_names_out=False,
        )
        ct.set_output(transform='pandas')

        X = ct.fit_transform(X)
        X = X[feat]
        y = y.to_numpy().ravel()

        results = cross_validate(
            estimator=RandomForestClassifier(class_weight=w),
            X=X,
            y=y,
            scoring=['f1', 'precision', 'recall', 'accuracy'],
            cv=cv,
            n_jobs=4,
        )

        result_list.append(results)

    _f1_means = [res['test_f1'].mean() for res in result_list]
    return result_list


def load_best():
    with open('.cache/feat_selection/best-20.csv', 'r') as fp:
        lines = fp.readlines()

    return [
        ln.strip().split(',')
        for ln in lines
    ]


def find_best_overall():
    bests = load_best()
    best = set(bests[0])
    for b in bests:
        best = best.intersection(b)

    print(','.join(best))


def select_best_features():
    for i in range(20):
        selected = select_features()
        print(f'{i:02d}: {",".join(selected)}')


def main():
    np.random.seed(4242)
    random.seed(4242)
    #select_best_features()
    features = load_best()
    results = feature_performance(features)
    #[pd.DataFrame.from_dict(
    #    {'f1': res['test_f1'].mean(), 'precision': res['test_precision'].mean(), 'recall': res['test_recall'].mean(),
    #     'accuracy': res['test_accuracy'].mean(), 'features': feat}, orient='index').to_csv(
    #    f'.cache/feat_runs/run:{i:02d}.csv') for i, (feat, res) in enumerate(zip(features, results))]
    return results


if __name__ == '__main__':
    main()
