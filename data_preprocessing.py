import pandas as pd

def categorize_age(df, splits, age_col='age', grp_col='age_grp'):
    """
    Categorize the `cc_age` column into age groups using `splits` as cut points.

    Parameters:",
    df - pandas DataFrame containing a numeric `cc_age` column (or values coercible to numeric).
    splits - iterable of numeric cut points (e.g. [18, 30, 45])

    """
    import numpy as np
    # validate inputs
    if age_col not in df.columns:
        raise KeyError("DataFrame must contain a " + age_col + " column")
    if splits is None:
        raise ValueError("splits must be a non-empty iterable of numeric cut points")

    # Normalize splits to a sorted list of floats and remove duplicates
    try:
        splits_list = sorted(set(float(x) for x in splits))
    except Exception:
        raise ValueError("splits must be an iterable of numbers")
    if len(splits_list) == 0:
        raise ValueError("splits must contain at least one cut point")

    bins = [-float('inf')] + splits_list + [float('inf')]
    ages = pd.to_numeric(df[age_col], errors='coerce')

    # Use pandas.cut to create categorical intervals (right-inclusive by default)
    age_cat = pd.cut(ages, bins=bins, include_lowest=True, labels=False)
    df.loc[:, grp_col] = age_cat