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
    drop_df = df.drop(columns=lst_cols)
    return drop_df

def to_binary_exists(df,cols):
    df[cols] = (~df[cols].isna()).astype(int)
    return df

def rename_cols_to_binary_exists(df,dict_cols):
    df.rename(columns=dict_cols, inplace=True)
    df = to_binary_exists(df, list(dict_cols.values()))

    return df

def to_binary_boolean_to_numeric(df,cols):
    df[cols] = df[cols].astype(int).fillna(0)
    return df

def to_binary_exists_json(df,cols):
    df[cols] = df[cols].apply(lambda x : (x.str.len().fillna(0) > 0).astype(int))
    return df

def log_transform(df,cols):
    df[cols] = df[cols].map(lambda x: np.log(x) if x > 0 else np.nan)
    df[cols] = df[cols].fillna(-1)
    return df

def to_binary_str_is_in(df,dict_transform):
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
    df[new_column] = (np.sign(df[base_column] - df[compare_column])).fillna(0)
    df.drop(columns=[compare_column], inplace=True)
    return df

def transform_num_elements_json(df,dict_transformations,drop=True):
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