import numpy as np
import pandas as pd
import numba
import tqdm

def count_sampling_coeffs(df, margins_idx, margins):
    sample_coeff = np.ones(len(df), dtype='float64')
    pop_size = margins[0].sum()
    for i in tqdm.tqdm(range(len(df))):
        row = df.iloc[i].astype(int)
        coeff = 1
        for idx, margin in zip(margins_idx, margins):
            vals = tuple(row[idx].values)
            coeff *= margin[vals] / pop_size
        sample_coeff[i] = coeff
    return sample_coeff


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


@numba.njit('void(float64[:], int64[:], float64)', cache=True, fastmath=True, nogil=True)
def multiply_numba(array, idx, y):
    for i in numba.prange(idx.shape[0]):
        array[idx[i]] *= y


@numba.njit('int64(float64[:], int64[:])', cache=True)
def weighted_random_choice_numba(p, indices):
    cumulative = 0.0
    r = np.random.random()
    
    for i in range(len(p)):
        cumulative += p[i]
        if r <= cumulative:
            return indices[i]
    return indices[-1]


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
            indices[idx] = np.array(df.index[mask], dtype='int64')
        suitable.append(indices)
    sampled_indices = []
    sample_coeff = count_sampling_coeffs(df, margins_idx, margins)
    df_indices = np.array(df.index, dtype='int64')
    for _ in tqdm.tqdm(range(n)):
        sum_coeff = sample_coeff.sum()
        if sum_coeff == 0:
            break
        p = sample_coeff / sum_coeff
        new_idx = weighted_random_choice_numba(p, df_indices)
        sampled_indices.append(new_idx)
        for margin_idx, margin, indices in zip(margins_idx, margins, suitable):
            vals = tuple(df.loc[new_idx, margin_idx].astype(int).values)
            mask = indices[vals]
            if margin[vals] <= 0:
                print("ERROR!!! Marginals for sampled individuals shouldn't be 0!")
                print(margin_idx, p[new_idx])
                return sample_coeff, sampled_indices
            multiply_numba(sample_coeff, mask, (margin[vals] - 1) / margin[vals])
            margin[vals] -= 1
    return df.loc[sampled_indices].reset_index(drop=True)


def sample_hh_data():
    pass