from hashlib import md5
from itertools import islice
from pathlib import Path
import requests
import time

from joblib import dump
import numpy as np
import polars as pl


# ### LAB UTILITIES ####

def fetch_and_cache(data_url, file, data_dir="data", force=False):
    """
    Download and cache a url and return the file object.

    data_url: the web address to download
    file: the file in which to save the results.
    data_dir: (default="data") the location to save the data
    force: if true the file is always re-downloaded

    return: The pathlib.Path object representing the file.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir / Path(file)
    # If the file already exists and we want to force a download then
    # delete the file first so that the creation date is correct.
    if force and file_path.exists():
        file_path.unlink()
    if force or not file_path.exists():
        print('Downloading...', end=' ')
        resp = requests.get(data_url)
        with file_path.open('wb') as f:
            f.write(resp.content)
        print('Done!')
        last_modified_time = time.ctime(file_path.stat().st_mtime)
    else:
        last_modified_time = time.ctime(file_path.stat().st_mtime)
        print("Using cached version that was downloaded (UTC):", last_modified_time)
    return file_path


def download_lab3_data():
    data_dir = 'data'
    data_url = 'http://www.ds100.org/sp24/resources/assets/datasets/lab03_data_sp24.zip'
    file_name = 'lab03_data_sp24.zip'
    return fetch_and_cache(data_url=data_url, file=file_name, data_dir=data_dir)


# ### PROJECT UTILITIES ####

def run_linear_regression_test(
    final_model,
    process_data_fm,
    threshold,
    train_data_path,
    test_data_path,
    is_test=False,
    is_ranking=False,
    return_predictions=False
):
    def rmse(predicted, actual):
        return np.sqrt(((actual - predicted) ** 2).mean())

    training_data = pl.read_csv(train_data_path).drop("")
    X_train, y_train = process_data_fm(training_data)
    if is_test:
        test_data = pl.read_csv(test_data_path).drop("")
        X_test = process_data_fm(test_data, is_test_set=True)
        assert len(test_data) == len(X_test), 'You may not remove data points from the test set!'

    final_model.fit(X_train, y_train)
    if is_test:
        return final_model.predict(X_test)
    else:
        y_predicted = final_model.predict(X_train)
        loss = rmse(y_predicted, y_train)
        if is_ranking:
            print('Your RMSE loss is: {}'.format(loss))
            return loss
        return loss < threshold


def run_linear_regression_test_optim(
    final_model,
    process_data_fm,
    train_data_path,
    test_data_path,
    is_test=False,
    is_ranking=False,
    return_predictions=False
):
    def rmse(predicted, actual):
        return np.sqrt(((actual - predicted) ** 2).mean())

    training_data = pl.read_csv(train_data_path).drop("")
    X_train, y_train = process_data_fm(training_data)
    if is_test:
        test_data = pl.read_csv(test_data_path).drop("")
        X_test = process_data_fm(test_data, is_test_set=True)
        assert len(test_data) == len(X_test), 'You may not remove data points from the test set!'

    final_model.fit(X_train, y_train)
    if is_test:
        return final_model.predict(X_test)
    else:
        y_predicted = final_model.predict(X_train)
        loss = rmse(y_predicted, y_train)
        if is_ranking:
            print('Your RMSE loss is: {}'.format(loss))
            return loss
        fn = (lambda threshold: loss < threshold)
        fn.loss = loss
        fn.signature = (process_data_fm, train_data_path, test_data_path)
        return fn
