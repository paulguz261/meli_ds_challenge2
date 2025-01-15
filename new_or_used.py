"""
Exercise description
--------------------

Description:
In the context of Mercadolibre's Marketplace an algorithm is needed to predict if an item listed in the markeplace is new or used.

Your tasks involve the data analysis, designing, processing and modeling of a machine learning solution 
to predict if an item is new or used and then evaluate the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k.jsonlines` and a function to read that dataset in `build_dataset`.

For the evaluation, you will use the accuracy metric in order to get a result of 0.86 as minimum. 
Additionally, you will have to choose an appropiate secondary metric and also elaborate an argument on why that metric was chosen.

The deliverables are:
--The file, including all the code needed to define and evaluate a model.
--A document with an explanation on the criteria applied to choose the features, 
  the proposed secondary metric and the performance achieved on that metrics. 
  Optionally, you can deliver an EDA analysis with other formart like .ipynb



"""

import json
import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
sys.path.append('../funciones')
from funciones import funciones as func 

if __name__ == "__main__":
    print("Loading dataset...")
    # Train and test data following sklearn naming conventions
    # X_train (X_test too) is a list of dicts with information about each item.
    # y_train (y_test too) contains the labels to be predicted (new or used).
    # The label of X_train[i] is y_train[i].
    # The label of X_test[i] is y_test[i].
    df_X_train, y_train, df_X_test, y_test = func.build_dataset()

    print(df_X_train.columns)

    # Insert your code below this line:
    lst_drop_cols = ["thumbnail",
                 "secure_thumbnail",
                 "permalink",
                 "site_id",
                 "location_country_id",
                 "location_country_name",
                 "international_delivery_mode",
                 "seller_contact",
                 "location_zip_code",
                 "differential_pricing",
                 "location_open_hours",
                 "subtitle",
                 "listing_source",
                 "seller_contact_webpage",
                 "catalog_product_id",
                 "shipping_dimensions",
                 "seller_contact_phone2",
                 "seller_contact_area_code2",
                 "seller_contact_other_info",
                 "coverage_areas",
                 "shipping_methods",
                 "shipping_tags",
                 "deal_ids",
                 "seller_address_country_id",
                 "seller_address_search_location_neighborhood_id",
                 "seller_address_search_location_state_id",
                 "seller_address_search_location_city_id",
                 "seller_address_city_id",
                 "seller_address_state_id",
                 "location_neighborhood_id",
                 "location_city_id",
                 "location_state_id",
                 "geolocation_latitude",
                 "geolocation_longitude",
                 "seller_address_latitude",
                 "seller_address_longitude",
                 "seller_address_search_location_state_name",
                 "seller_address_id",
                 "seller_address_search_location_city_name",
                 "last_updated",
                 "start_time",
                 "stop_time",
                 "date_created",
                 "descriptions",
                 "seller_address_comment",
                 "seller_address_address_line",
                 "title",
                 "seller_address_city_name",
                 "seller_address_zip_code",
                 "seller_address_search_location_neighborhood_name",
                 "sub_status",
                 "status",
                 "currency_id",
                 "seller_address_country_name",
                 "seller_contact_contact",
                 "seller_contact_area_code",
                 "seller_contact_phone",
                 "location_neighborhood_name",
                 "location_longitude",
                 "location_address_line",
                 "location_latitude",
                 "location_city_name",
                 "location_state_name",
                 "shipping_free_methods",
                 "base_price",
                 "original_price"]

    #transformacioness
    lst_binary_boolean_to_numeric = ["accepts_mercadopago", 
                                        "automatic_relist", 
                                        "shipping_local_pick_up",
                                        "shipping_free_shipping"
                                    ]

    lst_binary_exists = ["warranty",
                            "parent_item_id", 
                            "official_store_id",
                            "video_id"
                            ]

    #lst_json_binary = ["shipping_free_methods"]
    lst_json_binary = []

    dict_transform_str_is_in = {"buying_mode":(["buy_it_now"],None),
                                "shipping_mode":(["not_specified"],None),
                                "seller_address_state_name":(["Capital Federal"],None)}

    dict_inmobiliario = {"seller_contact_email":"flg_inmobiliario"}

    dict_transform_num_elements = {
            "variations": "num_variations",
            "attributes": "num_attributes",
            "tags":"num_tags",
            "pictures": "num_pictures",
            "non_mercado_pago_payment_methods":"num_payment_methods"
        }

    list_variaciones = ["base_price","original_price"]

    lst_scores = ["seller_id","category_id"]

    lst_recategorizacion = ["listing_type_id",
                            "non_mercado_pago_payment_methods"]

    lst_correlacion = ["available_quantity"]

    lst_outliers = ["price", "initial_quantity", "sold_quantity"]

    lst_log_transform = []

    transf_dataset = func.DatasetTransformer(
                 lst_drop_cols, 
                 lst_binary_exists, 
                 lst_binary_boolean_to_numeric, 
                 lst_json_binary, 
                 lst_log_transform,
                 dict_transform_str_is_in, 
                 dict_inmobiliario, 
                 dict_transform_num_elements, 
                 lst_outliers)

    df_X_train_processed = transf_dataset.fit_transform(df_X_train)
    df_X_test_processed = transf_dataset.transform(df_X_test)
    y_train_transform = [int(x == "used") for x in y_train]
    y_test_transform = [int(x == "used") for x in y_test]

    rf = RandomForestClassifier(random_state=123, n_jobs=-1, criterion= "gini", max_depth=23, max_features=4, n_estimators=110)
    rf.fit(df_X_train_processed, y_train_transform)

    y_pred_train = rf.predict(df_X_train_processed)
    y_pred_test = rf.predict(df_X_test_processed)

    train_accuracy = accuracy_score(y_train_transform, y_pred_train)
    train_f1 = f1_score(y_train_transform, y_pred_train)
    
    test_accuracy = accuracy_score(y_test_transform, y_pred_test)
    test_f1 = f1_score(y_test_transform, y_pred_test)
    
    print(f'Train Accuracy: {train_accuracy:.4f}')
    print(f'Train F1-Score: {train_accuracy:.4f}')

    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test F1-Score: {test_accuracy:.4f}')
