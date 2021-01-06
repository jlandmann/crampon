import pandas as pd


def compare_geosatclim(old, new):
    """
    Compare geosatclim version via the mass balance that they produce.

    Parameters
    ----------
    old: xr.Dataset
        MB (produced by climatology_from_daily.make_mb_clim) with old
        geosatclim version
    new: xr.Dataset
        MB (produced by climatology_from_daily.make_mb_clim) with new
        geosatclim version

    Returns
    -------
    None
    """
    df = (old.MB.sel(model=['PellicciottiModel']).cumsum() -
          new.MB.sel(model=['PellicciottiModel']).cumsum()).to_dataframe()
    df_nonzero = df[df != 0.]
    df_nonzero_pelli = df_nonzero[['MB']]
    df_nonzero_pelli = df_nonzero_pelli.rename(columns={
        'MB': 'PellicciottiModel'})
    df = (old.MB.sel(model=['OerlemansModel']).cumsum() -
          new.MB.sel(model=['OerlemansModel']).cumsum()).to_dataframe()
    df_nonzero = df[df != 0.]
    df_nonzero_oerle = df_nonzero[['MB']]
    df_nonzero_oerle = df_nonzero_oerle.rename(columns={
        'MB': 'OerlemansModel'})

    pd.concat([df_nonzero_oerle, df_nonzero_pelli], axis=1).boxplot()
