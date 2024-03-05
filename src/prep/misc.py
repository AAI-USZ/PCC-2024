import numpy as np
import pandas as pd


def process_scrappy_dataset(df: pd.DataFrame):
    """
    Do not look at me... I'm just trying to make some csv parsed data frame workable.

    :param df: dataframe to be processed
    :return: processed dataframe, hopefully not so different from the original
    """

    df = df.rename(columns={col: col.strip() for col in df.columns})
    del df['jensenshannon_distance']

    for col in df.columns:
        df.loc[:, col] = [v.strip() if isinstance(v, str) else v for v in df[col].values]

    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            null_indices = df[col].isna()
            df.loc[null_indices, col] = 0

    # Aggregate scores
    df['structural_score'] = df['structural_score'].map(lambda el: np.fromstring(el.strip('[]'), sep=',').sum())
    df['conceptual_score'] = df['conceptual_score'].map(lambda el: np.fromstring(el.strip('[]'), sep=',').sum())

    # Process object fields
    for col in df.columns:
        if df[col].dtype.kind == 'O':
            numerics = np.array([
                el.isdigit() if isinstance(el, str) else
                True if isinstance(el, (int, float)) else
                False
                for el in df[col]
            ])
            if numerics.sum() / len(numerics) >= 0.9:
                df.loc[~numerics, col] = 0

    return df
