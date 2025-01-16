import os
import json
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# You can safely assume that `build_dataset` is correctly implemented
# i could not .... needed to make changes
def build_dataset():
    dir_general = os.getcwd()
    dir_data = dir_general+r"\data\MLA_100k.jsonlines"
    data = [json.loads(x) for x in open(dir_data)]
    target = lambda x: x.get("condition")
    N = -10000
    X_train = data[:N]
    X_test = data[N:]
    y_train = [target(x) for x in X_train]
    y_test = [target(x) for x in X_test]

    df_X_train = pd.json_normalize(X_train, sep='_')
    df_X_test = pd.json_normalize(X_test, sep='_')

    df_X_test = df_X_test.loc[:,df_X_train.columns.values]
    df_X_test.drop(columns="condition", inplace=True)

    return df_X_train, y_train, df_X_test, y_test

def drop_cols(df,lst_cols):
    """
    Drops specified columns from a DataFrame.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame from which columns will be removed.
    lst_cols : list of str
        A list of column names to be dropped from the DataFrame.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame with the specified columns removed.
    """
    drop_df = df.drop(columns=lst_cols)
    return drop_df

def to_binary_exists(df,cols):
    """
    Transform the column to binary, depending on the presence of non-null values.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame from which columns will be transformed.
    lst_cols : list of str
        A list of column names to be transformed from the DataFrame.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame with the specified columns transformed.
    """
    df[cols] = (~df[cols].isna()).astype(int)
    return df

def rename_cols_to_binary_exists(df,dict_cols):
    """
    Renames columns in a DataFrame based on a provided dictionary and converts the renamed columns 
    into binary indicators, indicating the existence of values.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame containing the columns to be renamed and transformed.
    dict_cols : dict
        A dictionary where keys are the current column names in `df`, and values are the new column 
        names to which they should be renamed.

    Returns:
    -------
    pandas.DataFrame
        The modified DataFrame with columns renamed according to `dict_cols` and converted to binary 
        indicators (1 for non-null or non-zero values, 0 otherwise).
    """    
    df.rename(columns=dict_cols, inplace=True)
    df = to_binary_exists(df, list(dict_cols.values()))
    return df

def to_binary_boolean_to_numeric(df,cols):
    """
    Transform the column to int, columns passed should be of Boolean Type.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame from which columns will be transformed.
    lst_cols : list of str
        A list of column names which should be of Boolean type.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame with the specified columns transformed.
    """
    df[cols] = df[cols].astype(int).fillna(0)
    return df

def to_binary_exists_json(df,cols):
    """
    Transform the column to a binary value if the elements stored in the json format are different than null or empty, 
    columns passed should have a json stored inside.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame from which columns will be transformed.
    lst_cols : list of str
        A list of column names which should contain a json structure.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame with the specified columns transformed.
    """
    df[cols] = df[cols].apply(lambda x : (x.str.len().fillna(0) > 0).astype(int))
    return df

def log_transform(df,cols):
    df[cols] = df[cols].map(lambda x: np.log(x) if x > 0 else np.nan)
    df[cols] = df[cols].fillna(-1)
    return df

def to_binary_str_is_in(df,dict_transform):
    """
    Replaces the specified columns to binary indicators based on whether the columns in the key values from the input dictionary matches, 
    the value pair stored in the dictionary.

    The naming of the new column is flg_(column_name)_(value_searched)
    The columns passed as parameters are dropped from the returning dataframe

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame containing columns to be transformed.
    dict_transform : dict
        A dictionary where keys are column names in `df`, and values are the specific values to match.
        For each key-value pair, a new binary column will be created, indicating if the value in the column 
        matches the specified criterion (1 if match, 0 otherwise).

    Returns:
    -------
    pandas.DataFrame
        The DataFrame with new binary columns for each condition, and the original columns dropped.

    """
    for column, (values, custom_name) in dict_transform.items():
        if len(values) == 1 and custom_name is None:
            column_name = f"flg_{column}_{values[0]}".replace(" ", "_").lower()
        else:
            column_name = f"flg_{custom_name}".replace(" ", "_").lower()
        # Create binary flag column with the custom name
        df[column_name] = df[column].isin(values).astype(int).fillna(0)
    
    # Drop the original columns
    df.drop(columns=dict_transform.keys(), inplace=True)
    
    return df

def transform_get_variation(df, base_column, compare_column, new_column):
    """
    Computes the variation between two columns in a DataFrame, assigns a sign to the result, 
    and stores it in a new column. The comparison column is dropped from the DataFrame after processing.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame containing the columns to compare.
    base_column : str
        The name of the column that serves as the base for the comparison.
    compare_column : str
        The name of the column to be compared against the `base_column`.
    new_column : str
        The name of the new column where the sign of the variation is stored.
        The values in this column will represent:
        - 1: If `base_column` > `compare_column`.
        - -1: If `base_column` < `compare_column`.
        - 0: If `base_column` == `compare_column` or for missing values.

    Returns:
    -------
    pandas.DataFrame
        The modified DataFrame with the new column added and the `compare_column` removed.
    """
    df[new_column] = (np.sign(df[base_column] - df[compare_column])).fillna(0)
    df.drop(columns=[compare_column], inplace=True)
    return df

def transform_num_elements_json(df,dict_transformations,drop=True):
    """
    Transform the column to a number that represents the number elements stored in the json format, 
    columns passed should have a json stored inside.

    the column given as a parameter will be eliminated from the returning df

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame from which columns will be transformed.
    col : str
        column name which should contain a json structure.
    new_name : str
        name of the new column to be created

    Returns:
    -------
    pandas.DataFrame
        A DataFrame with the specified columns transformed.
    """
    for original_column, config in dict_transformations.items():
            if isinstance(config, tuple):
                new_column, drop = config
            else:
                new_column, drop = config, drop

            df[new_column] = df[original_column].str.len().fillna(0)
            
            if drop:
                df.drop(columns=[original_column], inplace=True)

    return df

def transform_payment_methods(df,drop=True):
    """
    Transforms the payment methods in a DataFrame into categorical columns by mapping specific 
    payment method descriptions to standardized categories. Optionally drops the original 
    `non_mercado_pago_payment_methods` column.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame containing at least the following columns:
        - `id`: A unique identifier for each entry.
        - `non_mercado_pago_payment_methods`: A list-like column with payment method details 
          in a JSON-like structure.

    drop : bool, optional, default=True
        Indicates whether to drop the `non_mercado_pago_payment_methods` column from the output DataFrame.

    Returns:
    -------
    pandas.DataFrame
        The transformed DataFrame with binary columns for each payment method category. Each column 
        contains 0 or 1, indicating the absence or presence of the respective payment method for each `id`.
    """
    dict_metodos_de_pago = {
        "Acordar con el comprador": "efectivo_o_acuerdo",
        "Efectivo": "efectivo_o_acuerdo",
        "Transferencia bancaria": "transferencia_bancaria",
        "Giro postal": "giro_postal",
        "Cheque certificado": "cheque",
        "American Express": "tarjeta_credito",
        "Diners": "tarjeta_credito",
        "MasterCard": "tarjeta_credito",
        "Mastercard Maestro": "tarjeta_credito",
        "Visa": "tarjeta_credito",
        "Visa Electron": "tarjeta_credito",
        "Tarjeta de crédito": "tarjeta_credito",
        "MercadoPago": "MercadoPago",
        "Contra reembolso": "contra_reembolso"
    }
    
    df_payment_methods = df.loc[:,["id","non_mercado_pago_payment_methods"]].\
                            explode("non_mercado_pago_payment_methods").\
                            reset_index()

    df_payment_methods_explode = pd.json_normalize(df_payment_methods["non_mercado_pago_payment_methods"])
    df_payment_methods_explode.rename(columns={"description":"description_shipping_free_methods",
                                                    "id":"id_shipping_free_methods"},
                                            inplace=True)

    df_payment_methods = pd.concat([df_payment_methods[["id"]], 
                                        df_payment_methods_explode], 
                                        axis=1)[["id","description_shipping_free_methods","id_shipping_free_methods"]]
    

    df_payment_methods["metodo_de_pago"] = df_payment_methods["description_shipping_free_methods"].map(dict_metodos_de_pago)
    df_payment_methods["ones"] = 1
    gr_df_payment_methods = df_payment_methods.groupby(["id","metodo_de_pago"])[["ones"]].sum().reset_index()
    gr_df_payment_methods = gr_df_payment_methods.pivot(index="id",columns="metodo_de_pago", values="ones").fillna(0)

    df = pd.merge(df, gr_df_payment_methods, on="id", how="left").fillna(0)
    df.loc[:,gr_df_payment_methods.columns.values] = df[gr_df_payment_methods.columns.values].fillna(0)

    if drop:
        df.drop(columns=["non_mercado_pago_payment_methods"])

    return df

def transform_listing_type(df):
    """
    Encodes the listing type of each product with a numeric score based on a predefined mapping.

    This function maps each unique `listing_type_id` in the DataFrame to a corresponding numeric score, 
    which represents the listing's quality or priority level. The mapping is defined by `dict_transform_listing_type`. 
    If a `listing_type_id` does not match any key in the mapping, it is assigned a default score of 0.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame containing a `listing_type_id` column, which holds the listing type for each product.

    Returns:
    -------
    pandas.DataFrame
        The DataFrame with the `listing_type_id` column transformed to numeric scores.
    """
    dict_transform_listing_type = {'free':0,
                               'bronze':1,
                               'silver':2,
                               'gold':3, 
                               'gold_premium':4,
                               'gold_special':5, 
                               'gold_pro':6}
    
    df["listing_type_id"] = df["listing_type_id"].map(dict_transform_listing_type).fillna(0)

    return df

class sellerIdTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_column=True):
        self.drop_column = drop_column
        self.dict_seller = None

    def fit(self,X, y=None):
        X_copy = X[["id","seller_id","condition"]].copy()
        X_copy["binary_used"] = (X_copy["condition"] == "used").astype(int)

        gr_seller_id = X_copy.groupby(["seller_id"]).agg({"binary_used":"sum","id":"count"})
        gr_seller_id["score_seller_used"] = gr_seller_id["binary_used"]/gr_seller_id["id"]

        self.dict_seller = gr_seller_id[["score_seller_used"]].to_dict()["score_seller_used"]
        
        return self

    def transform(self,X):
        if self.dict_seller is None:
            raise ValueError("El transformador no ha sido entrenado, ejecute la funcion 'fit' primero.")
        
        X_copy = X.copy()
        X_copy["score_seller"] = X_copy["seller_id"].map(self.dict_seller).fillna(0)
        
        if self.drop_column:
            X_copy.drop(columns=["seller_id"], inplace=True)

        return X_copy

class CategoryIdPopularityTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer for calculating a seller's "used item score" based on the proportion of used items 
    sold by each seller and mapping it back to the input DataFrame. Optionally drops the `seller_id` 
    column after transformation.

    Parameters:
    ----------
    drop_column : bool, optional, default=True
        Determines whether the `seller_id` column is dropped from the DataFrame after transformation.

    Attributes:
    ----------
    dict_seller : dict
        A dictionary where keys are `seller_id` values and values are the computed "used item scores" 
        (proportion of used items sold by each seller).

    Methods:
    -------
    fit(X, y=None):
        Computes the "used item scores" for each seller based on the input DataFrame.

    transform(X):
        Maps the precomputed "used item scores" to the input DataFrame and optionally drops the 
        `seller_id` column. Raises a `ValueError` if the transformer is not fitted.
    """    
    def __init__(self, drop_category_id=True):
        self.drop_category_id = drop_category_id
        self.dict_category_popularity = None

    def fit(self, X, y=None):
        X_copy = X[["id","category_id","condition"]].copy()

        X_copy["binary_used"] = (X_copy["condition"] == "used").astype(int)

        gr_seller_id = X_copy.groupby("category_id").agg({"binary_used": "sum", "id": "count"})
        gr_seller_id["score_category_used"] = gr_seller_id["binary_used"] / gr_seller_id["id"]

        self.dict_category_popularity = gr_seller_id["score_category_used"].to_dict()
        
        return self

    def transform(self, X):
        if self.dict_category_popularity is None:
            raise ValueError("El transformador no ha sido entrenado, ejecute la funcion 'fit' primero.")

        X_copy = X.copy()
        X_copy["score_popularity_category"] = X_copy["category_id"].map(self.dict_category_popularity).fillna(0)

        if self.drop_category_id:
            X_copy.drop(columns=["category_id"], inplace=True)

        return X_copy

class CategoryIdTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that calculates a normalized score for each `category_id` based on the 
    frequency of occurrences in the input DataFrame. Optionally drops the `category_id` column 
    after transformation.

    Parameters:
    ----------
    drop_category_id : bool, optional, default=True
        Determines whether the `category_id` column is dropped from the DataFrame after transformation.

    Attributes:
    ----------
    dict_category : dict
        A dictionary where keys are `category_id` values and values are the normalized scores 
        (frequency of the `category_id` relative to the most frequent `category_id`).

    Methods:
    -------
    fit(X, y=None):
        Computes the normalized scores for each `category_id` based on its frequency in the input DataFrame.

    transform(X):
        Maps the precomputed normalized scores to the `category_id` values in the input DataFrame and 
        optionally drops the `category_id` column. Raises a `ValueError` if the transformer is not fitted.
    """
    def __init__(self, drop_category_id=True):
        self.drop_category_id = drop_category_id
        self.dict_category = None

    def fit(self, X, y=None):
        gr_category_id = X.groupby("category_id")[["id"]].count().sort_values("id",ascending=False)
        gr_category_id = gr_category_id/gr_category_id.max()

        self.dict_category = gr_category_id.to_dict()["id"]

        return self

    def transform(self, X):
        if self.dict_category is None:
            raise ValueError("El transformador no ha sido entrenado, ejecute la funcion 'fit' primero.")

        X_copy = X.copy()
        X_copy["score_category_id"] = X_copy["category_id"].map(self.dict_category).fillna(0)

        if self.drop_category_id:
            X_copy.drop(columns=["category_id"], inplace=True)

        return X_copy

class FlgOutlierTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that identifies outliers in specified numeric columns based on the 
    Interquartile Range (IQR) method. It creates binary flag columns indicating the presence 
    of outliers for each specified column.

    Parameters:
    ----------
    lst_columns : list of str
        A list of column names in the DataFrame for which outliers will be identified.

    Attributes:
    ----------
    dict_upper_lower_bounds : dict
        A dictionary containing the calculated lower and upper bounds for each column 
        in `lst_columns`. The keys are column names, and the values are lists of the form 
        `[lower_bound, upper_bound]`.

    Methods:
    -------
    fit(X, y=None):
        Calculates the IQR and determines the lower and upper bounds for each column in `lst_columns`.

    transform(X):
        Flags outliers in the specified columns by comparing their values to the calculated bounds. 
        Adds binary columns prefixed with `flg_outliers_` for each column in `lst_columns` to the DataFrame.
        Raises a `ValueError` if the transformer is not properly initialized or fitted.
    """
    def __init__(self, lst_columns):
        self.dict_upper_lower_bounds = None
        self.lst_columns = lst_columns

    def fit(self, X, y=None):
        
        dict_param_outliers = {}
        q1 = X[self.lst_columns].quantile(0.25)
        q3 = X[self.lst_columns].quantile(0.75)
        
        # Calcula IQR
        iqr = q3 - q1
        
        # lower y upper bounds
        lower_bounds = q1 - 1.5 * iqr
        upper_bounds = q3 + 1.5 * iqr
        
        # Combine into a dictionary
        self.dict_upper_lower_bounds = {col: [lower_bounds[col], upper_bounds[col]] for col in self.lst_columns}

        return self
    
    def transform(self, X):
        if self.lst_columns is None:
            raise ValueError("El transformador fue inicializado incorrectamente, suministre una lista de columnas como parametro de entrada")
        if self.dict_upper_lower_bounds  is None:
            raise ValueError("El transformador no ha sido entrenado, ejecute la funcion 'fit' primero.")
        
        lower_bounds = pd.Series({col: bounds[0] for col, bounds in self.dict_upper_lower_bounds.items()})
        upper_bounds = pd.Series({col: bounds[1] for col, bounds in self.dict_upper_lower_bounds.items()})

        is_outlier = (X[self.lst_columns] < lower_bounds) | (X[self.lst_columns] > upper_bounds)
        outlier_flags = is_outlier.astype(int).add_prefix("flg_outliers_")

        X_transformed = pd.concat([X, outlier_flags], axis=1)

        return X_transformed
    
class DatasetTransformer(BaseEstimator, TransformerMixin):
    """
    A comprehensive transformer for applying multiple preprocessing transformations to a dataset, 
    including feature engineering, binary transformations, and sub-transformer applications. 
    This class is designed to automate extensive data preparation for machine learning models.

    Parameters:
    ----------
    lst_drop_cols : list of str
        A list of column names to be dropped from the dataset.
    lst_binary_exists : list of str
        A list of columns to transform into binary indicators based on the presence of values.
    lst_binary_boolean_to_numeric : list of str
        A list of boolean columns to convert into numeric values (1 for True, 0 for False).
    lst_json_binary : list of str
        A list of columns containing JSON data to transform into binary indicators.
    lst_log_transform : list of str
        A list of numeric columns to apply log transformation to.
    dict_transform_str_is_in : dict
        A dictionary mapping column names to values, used to create binary flags if a value 
        exists in the column.
    dict_inmobiliario : dict
        A dictionary mapping column names to new column names for real estate-related features 
        (transformed to binary indicators).
    dict_transform_num_elements : dict
        A dictionary where keys represent JSON-like columns, and values are numerical thresholds 
        to be transformed.
    lst_outliers : list of str
        A list of numeric columns for which outlier detection will be performed using IQR bounds.

    Attributes:
    ----------
    trans_seller_id : sellerIdTransformer
        A transformer for calculating and mapping seller scores based on item conditions.
    trans_categ_id_popularity : CategoryIdPopularityTransformer
        A transformer that calculates and maps category popularity based on item frequency.
    trans_categ_id : CategoryIdTransformer
        A transformer for normalizing category ID scores based on item frequencies.
    trans_flg_outliers : FlgOutlierTransformer
        A transformer for identifying and flagging outliers in specified numeric columns.

    Methods:
    -------
    fit(X, y=None):
        Fits all sub-transformers using the provided dataset.

    transform(X):
        Applies all specified transformations to the dataset, including column transformations, 
        sub-transformers, and feature engineering. Returns the transformed dataset.
    """
    def __init__(self, 
                 lst_drop_cols, 
                 lst_binary_exists, 
                 lst_binary_boolean_to_numeric, 
                 lst_json_binary, 
                 lst_log_transform,
                 dict_transform_str_is_in, 
                 dict_inmobiliario, 
                 dict_transform_num_elements, 
                 lst_outliers):

        self.lst_drop_cols = lst_drop_cols
        self.lst_binary_exists = lst_binary_exists
        self.lst_binary_boolean_to_numeric = lst_binary_boolean_to_numeric
        self.lst_json_binary = lst_json_binary
        self.lst_log_transform = lst_log_transform
        self.dict_transform_str_is_in = dict_transform_str_is_in
        self.dict_inmobiliario = dict_inmobiliario
        self.dict_transform_num_elements = dict_transform_num_elements
        self.lst_outliers = lst_outliers
        
        self.trans_seller_id = sellerIdTransformer()
        self.trans_categ_id_popularity = CategoryIdPopularityTransformer(drop_category_id=False)
        self.trans_categ_id = CategoryIdTransformer()
        self.trans_flg_outliers = FlgOutlierTransformer(lst_columns=lst_outliers)

    def fit(self, X, y=None):
        self.trans_seller_id.fit(X, y)
        self.trans_categ_id_popularity.fit(X, y)
        self.trans_categ_id.fit(X, y)
        self.trans_flg_outliers.fit(X, y)
        return self

    def transform(self, X):
        df = X.copy()

        # transformations
        df = drop_cols(df, self.lst_drop_cols)
        df = to_binary_exists(df, self.lst_binary_exists)
        df = to_binary_boolean_to_numeric(df, self.lst_binary_boolean_to_numeric)
        df = to_binary_exists_json(df,self.lst_json_binary)
        df = to_binary_str_is_in(df, self.dict_transform_str_is_in)
        df = rename_cols_to_binary_exists(df, self.dict_inmobiliario)
        
        #df = transform_get_variation(df, base_column="price", compare_column="base_price", new_column="variation_base_price")
        #df = transform_get_variation(df, base_column="price", compare_column="original_price", new_column="variation_original_price")
        df = transform_get_variation(df, base_column="initial_quantity", compare_column="available_quantity", new_column="variation_available_quantity")

        df = transform_payment_methods(df, drop=False)
        df = transform_num_elements_json(df, self.dict_transform_num_elements)

        df = transform_listing_type(df)

        # sub-transformers
        #df = self.trans_seller_id.transform(df)
        df = self.trans_categ_id_popularity.transform(df)
        df = self.trans_categ_id.transform(df)
        df = self.trans_flg_outliers.transform(df)

        df = log_transform(df,self.lst_log_transform)

        # Drop
        if 'condition' in df.columns:
            df.drop(columns=['condition'], inplace=True)

        if 'id' in df.columns:
            df.drop(columns=['id'], inplace=True)

        return df