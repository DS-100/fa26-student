import numpy as np
import polars as pl


def remove_outliers(data, variable, lower=-np.inf, upper=np.inf):
    """
    Input:
      data (data frame): the table to be filtered
      variable (string): the column with numerical outliers
      lower (numeric): observations with values lower than or equal to this will be removed
      upper (numeric): observations with values higher than or equal to this will be removed

    Output:
      a data frame with outliers removed

    Note: This function should not change the contents of data.
    """
    return data.filter((pl.col(variable) > lower) & (pl.col(variable) < upper))


def add_total_bathrooms(data):
    """
    Input:
      data (DataFrame): a DataFrame containing at least the Description column.

    Output:
      a DataFrame with a new column "Bathrooms" containing floats.

    """
    rooms_regex = r'([\d\.]+) of which are bathrooms'
    return data.with_columns(
        pl.col('Description').str.extract(rooms_regex, 1).cast(pl.Float64).alias('Bathrooms')
    )


def add_total_bedrooms(data):
    """
    Input:
      data (data frame): a data frame containing at least the Description column.
    """
    rooms_regex = r'(\d+) of which are bedrooms'
    return data.with_columns(
        pl.col('Description').str.extract(rooms_regex, 1).cast(pl.Int64).alias('Bedrooms')
    )


def find_expensive_neighborhoods(data, n=3, metric=np.median):
    """
    Input:
      data (data frame): should contain at least a string-valued Neighborhood
        and a numeric 'Sale Price' column
      n (int): the number of top values desired
      metric (function): function used for aggregating the data in each neighborhood.
        for example, np.median for median prices

    Output:
      a list of the top n richest neighborhoods as measured by the metric function
    """
    neighborhoods = (
        data
        .group_by('Neighborhood Code')
        .agg(
            pl.col('Log Sale Price')
            .map_elements(lambda values: metric(values.to_numpy()), return_dtype=pl.Float64)
        )
        .sort('Log Sale Price', descending=True)
        .head(n)
        ['Neighborhood Code']
        .to_list()
    )

    # This makes sure the final list contains the generic int type used in Python3, not specific ones used in numpy.
    return [int(code) for code in neighborhoods]


def add_in_expensive_neighborhood(data, neighborhoods):
    """
    Input:
      data (data frame): a data frame containing a 'Neighborhood Code' column with values
        found in the codebook
      neighborhoods (list of strings): strings should be the names of neighborhoods
        pre-identified as rich
    Output:
      data frame identical to the input with the addition of a binary
      in_rich_neighborhood column
    """
    return data.with_columns(
        pl.col('Neighborhood Code').is_in(neighborhoods).cast(pl.Int32).alias('in_expensive_neighborhood')
    )


def substitute_wall_material(data):
    """
    Input: data (DataFrame): a DataFrame containing a 'Wall Material' column.  Its values should be limited to those found in the codebook
    Output: new DataFrame identical to the input except with a refactored 'Wall Material' column
    """
    # BEGIN SOLUTION
    replacements = {
        1: 'Wood',
        2: 'Masonry',
        3: 'Wood&Masonry',
        4: 'Stucco'
    }

    new_data = data.with_columns(
        pl.col('Wall Material').replace_strict(replacements, default=None)
    )
    # END SOLUTION
    return new_data


from sklearn.preprocessing import OneHotEncoder


def ohe_wall_material(data):
    """
    One-hot-encodes wall material. New columns are of the form "Wall Material_MATERIAL".
    """
    ...
    oh_enc = OneHotEncoder()
    oh_enc.fit(data[['Wall Material']])
    oh_enc_columns = pl.DataFrame(
        oh_enc.transform(data[['Wall Material']]).toarray(),
        schema=list(oh_enc.get_feature_names_out())
    )

    return data.hstack(oh_enc_columns)


def select_columns(data, *columns):
    """Select only columns passed as arguments."""
    return data.select(list(columns))


def log_transform(data, col):
    """Add the log transformation of a column to the data frame"""
    return data.with_columns(pl.col(col).log().alias('Log ' + col))


def rmse(predicted, actual):
    """
    Calculates RMSE from actual and predicted values.
    Input:
      predicted (1D array): Vector of predicted/fitted values
      actual (1D array): Vector of actual values
    Output:
      A float, the RMSE value.
    """
    return np.sqrt(((actual - predicted) ** 2).mean())


def mape_interval(df, start, end):
    subset_df = df.filter(
        (pl.col('True Log Sale Price') >= start) & (pl.col('True Log Sale Price') < end)
    )
    actual = subset_df['True Log Sale Price']
    predicted = subset_df['Predicted Log Sale Price']
    percent_error = np.abs((actual - predicted) / actual)
    return percent_error.mean()
