import numpy as np
import pandas as pd
import tqdm

def count_sampling_coeffs(df, margins_idx, margins):
    df.loc[:, 'sampling_coeff'] = [1.0] * len(df)
    pop_size = margins[0].sum()
    for i in range(len(df)):
        row = df.iloc[i].astype(int)
        coeff = 1
        for idx, margin in zip(margins_idx, margins):
            vals = tuple(row[idx].values)
            coeff *= margin[vals] / pop_size
        df.at[i, 'sampling_coeff'] = coeff


def sample_data(df, margins_idx, margins):
    n = margins[0].sum()
    for margin in margins[1:]:
        assert margin.sum() == n, "All margins must sum to the same total"
    sampled_indices = []
    for _ in tqdm.tqdm(range(n)):
        count_sampling_coeffs(df, margins_idx, margins)
        p = df['sampling_coeff'] / df['sampling_coeff'].sum()
        new_idx = np.random.choice(df.index, replace=False, p=p)
        sampled_indices.append(new_idx)
        for margin_idx, margin in zip(margins_idx, margins):
            vals = tuple(df.loc[new_idx, margin_idx].astype(int).values)
            margin[vals] -= 1
    return df.loc[sampled_indices].reset_index(drop=True)


def sample_data_optimized(df, margins_idx, margins):
    n = margins[0].sum()
    for margin in margins[1:]:
        assert margin.sum() == n, "All margins must sum to the same total"
    suitable = []
    for m_idx, margin in zip(margins_idx, margins):
        shape = margin.shape
        values = np.arange(shape[0])
        if len(shape) > 1:
            values = np.stack(np.indices(shape), axis=-1).reshape((-1, len(shape)))
        indices = np.zeros(shape, dtype=object)
        for val in values:
            mask = (df[m_idx] == val).values.all(axis=1)
            idx = val if len(m_idx) == 1 else tuple(val)
            indices[idx] = df.index[mask]
        suitable.append(indices)
    sampled_indices = []
    count_sampling_coeffs(df, margins_idx, margins)
    for _ in tqdm.tqdm(range(n)):
        p = df['sampling_coeff'] / df['sampling_coeff'].sum()
        new_idx = np.random.choice(df.index, replace=False, p=p)
        sampled_indices.append(new_idx)
        for margin_idx, margin, indices in zip(margins_idx, margins, suitable):
            vals = tuple(df.loc[new_idx, margin_idx].astype(int).values)
            mask = indices[vals]
            df.loc[mask, 'sampling_coeff'] *= (margin[vals] - 1) / margin[vals]
            margin[vals] -= 1
    return df.loc[sampled_indices].reset_index(drop=True)


def sample_hh_data():
    pass