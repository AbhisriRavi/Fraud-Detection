import numpy as np
import pandas as pd

from features import FEATURES, build_features


def make_row(time=0.0, amount=0.0):
    row = {"Time": time, "Amount": amount, **{f"V{i}": 0.0 for i in range(1, 29)}}
    return pd.DataFrame([row])


def test_columns_match_training_features():
    assert list(build_features(make_row()).columns) == FEATURES


def test_log_amount():
    assert build_features(make_row(amount=0.0))["log_amount"].iloc[0] == 0.0
    assert np.isclose(build_features(make_row(amount=99.0))["log_amount"].iloc[0], np.log(100))


def test_hour_wraps_after_one_day():
    # 90,000 seconds = 25 hours, which should be 1am on day two
    assert build_features(make_row(time=90000))["hour"].iloc[0] == 1