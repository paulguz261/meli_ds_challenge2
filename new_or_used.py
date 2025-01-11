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

# You can safely assume that `build_dataset` is correctly implemented
def build_dataset():
    data = [json.loads(x) for x in open("MLA_100k.jsonlines")]
    target = lambda x: x.get("condition")
    N = -10000
    X_train = data[:N]
    X_test = data[N:]
    y_train = [target(x) for x in X_train]
    y_test = [target(x) for x in X_test]
    for x in X_test:
        del x["condition"]
    return X_train, y_train, X_test, y_test

def transform_y(y):
    """
    Converts the y values into a binary element 1 if the condition is "used" and 0 otherwise.

    Parameters:
    ----------
    y : list of str
        A list of product conditions, where each value is expected to be either "used" or another condition.

    Returns:
    -------
    list of int
        A list of integers where each entry is 1 if the corresponding entry in `y` is "used" and 0 otherwise.
    """
    try:
        return [int(v == "used") for v in y]
    
    except Exception as e:
        raise Exception(f"Error in function 'transform_y': {str(e)}") from e

def read_data():
    """
    Reads and preprocesses the training and test datasets for model input.

    This function loads raw training and test data by calling `build_dataset()`, 
    applies JSON normalization to flatten nested structures in `X_train` and `X_test`, 
    and converts `y_train` and `y_test` to binary labels using `transform_y()`.

    Returns:
    -------
    tuple
        A tuple containing four elements: X_train, y_train, X_test, y_test
    """
    try:
        X_train, y_train, X_test, y_test = build_dataset()

        X_train = pd.json_normalize(X_train, sep='_')
        X_test = pd.json_normalize(X_test, sep='_')
        y_train = transform_y(y_train)
        y_test = transform_y(y_test)

        return X_train, y_train, X_test, y_test

    except Exception as e:
        raise Exception(f"Error in function 'read_data': {str(e)}") from e

if __name__ == "__main__":
    print("Loading dataset...")
    # Train and test data following sklearn naming conventions
    # X_train (X_test too) is a list of dicts with information about each item.
    # y_train (y_test too) contains the labels to be predicted (new or used).
    # The label of X_train[i] is y_train[i].
    # The label of X_test[i] is y_test[i].
    X_train, y_train, X_test, y_test = build_dataset()

    # Insert your code below this line:
    # ...


