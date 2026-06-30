import pandas as pd


def nearest_index(index, date):
    #snap an arbitrary calendar date onto the nearest actual trading day in the index. rebalance
    #dates (month-ends) routinely land on weekends/holidays, this keeps the lookup from KeyError-ing.
    if date in index:
        return date
    return index[index.get_indexer([date], method='nearest')[0]]


def broadcast_series(series, columns, index):
    #stretch a 1d series (e.g. the spy regime flag) across every column so it can be combined with
    #the per-stock boolean masks elementwise
    import numpy as np
    return pd.DataFrame(
        np.tile(series.to_numpy().reshape(-1, 1), (1, len(columns))),
        index=index,
        columns=columns,
    )
