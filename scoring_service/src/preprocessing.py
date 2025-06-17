import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import category_encoders as ce

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)  

R_EARTH = 6_372_800  


def haversine_distance(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in *metres*."""
    logger.debug("Computing haversine distance …")
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R_EARTH * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def bearing_degree(lat1, lon1, lat2, lon2):
    """Initial bearing / forward azimuth [0, 360)."""
    logger.debug("Computing bearing degree …")
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def add_geospatial_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Adding geospatial features …")
    df = df.copy()
    df["bearing_degree_1"] = bearing_degree(df["lat"], df["lon"], df["merchant_lat"], df["merchant_lon"])
    df["bearing_degree_2"] = bearing_degree(df["lat"], df["lon"], 0, 0)
    df["bearing_degree_3"] = bearing_degree(0, 0, df["merchant_lat"], df["merchant_lon"])

    df["hav_dist_1"] = haversine_distance(df["lat"], df["lon"], df["merchant_lat"], df["merchant_lon"])
    df["hav_dist_2"] = haversine_distance(df["lat"], df["lon"], 0, 0)
    df["hav_dist_3"] = haversine_distance(0, 0, df["merchant_lat"], df["merchant_lon"])

    df["same_location"] = (
        (df["lat"].round(3) == df["merchant_lat"].round(3))
        & (df["lon"].round(3) == df["merchant_lon"].round(3))
    ).astype(int)

    df["lat_diff"] = (df["lat"] - df["merchant_lat"]).abs()
    df["lon_diff"] = (df["lon"] - df["merchant_lon"]).abs()
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Adding time features …")
    df = df.copy()
    if not np.issubdtype(df["transaction_time"].dtype, np.datetime64):
        df["transaction_time"] = pd.to_datetime(df["transaction_time"])

    df["hour"] = df["transaction_time"].dt.hour
    df["dayofweek"] = df["transaction_time"].dt.weekday
    df["day"] = df["transaction_time"].dt.day
    df["month"] = df["transaction_time"].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_night"] = df["hour"].between(0, 6).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["weekday_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    df["ts_transaction_time"] = df["transaction_time"].astype("int64") // 10**9
    return df


def fit_target_encoder(
    train_df: pd.DataFrame,
    cat_columns: List[str],
    target_col: str = "target",
) -> ce.CatBoostEncoder:
    logger.debug("Fitting CatBoost target encoder …")
    enc = ce.CatBoostEncoder(cols=cat_columns, return_df=True)
    enc.fit(train_df[cat_columns], train_df[target_col])
    return enc


def apply_target_encoder(df: pd.DataFrame, encoder: ce.CatBoostEncoder) -> pd.DataFrame:
    logger.debug("Applying target encoder …")
    encoded = encoder.transform(df[encoder.cols]).add_suffix("_cb")
    return df.join(encoded)


def fit_preprocessor(
    train_df: pd.DataFrame,
    cat_columns: List[str],
    target_col: str = "target",
) -> Dict:
    """Return anything that needs to be *saved* for inference."""
    logger.info("Fitting full preprocessor …")
    encoder = fit_target_encoder(train_df, cat_columns, target_col)
    return {"encoder": encoder, "cat_columns": cat_columns}


def transform_preprocessor(df: pd.DataFrame, helpers: Dict) -> pd.DataFrame:
    """Stateless transform that produces exactly ``MODEL_FEATURES``."""
    logger.info("Applying full preprocessor …")
    df_ = df.copy()
    df_ = add_geospatial_features(df_)
    df_ = add_time_features(df_)
    df_ = apply_target_encoder(df_, helpers["encoder"])

    missing = [f for f in MODEL_FEATURES if f not in df_.columns]
    if missing:
        logger.warning("Missing expected features after preprocess: %s", missing)
    return df_[MODEL_FEATURES]


MODEL_FEATURES: List[str] = [
    "amount",
    "bearing_degree_1",
    "bearing_degree_2",
    "bearing_degree_3",
    "cat_id_cb",
    "gender_cb",
    "hav_dist_1",
    "hav_dist_2",
    "hav_dist_3",
    "jobs_cb",
    "lat",
    "lon",
    "merch_cb",
    "merchant_lat",
    "merchant_lon",
    "name_1_cb",
    "name_2_cb",
    "one_city_cb",
    "population_city",
    "post_code_cb",
    "street_cb",
    "us_state_cb",
    "ts_transaction_time",
    "hour",
    "dayofweek",
    "day",
    "month",
    "is_weekend",
    "is_night",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "lat_diff",
    "lon_diff",
    "same_location",
]

__all__ = [
    "fit_preprocessor",
    "transform_preprocessor",
    "MODEL_FEATURES",
]


