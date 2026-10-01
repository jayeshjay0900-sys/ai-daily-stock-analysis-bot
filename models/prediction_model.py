from __future__ import annotations

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from utils.indicators import add_indicators


# ============================================================
# ML FEATURES
# ============================================================

FEATURES = [
    "SMA_20",
    "SMA_50",
    "SMA_200",
    "EMA_20",
    "EMA_50",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "ATR_14",
    "Volume_SMA_20",
    "Volume_Change_Pct",
    "Daily_Return",
]


# ============================================================
# MAIN MODEL
# ============================================================

def run_direction_model(
    df: pd.DataFrame,
):
    """
    Train a time-ordered Random Forest model.

    Target:
        1 = next closing price is higher
        0 = next closing price is lower/equal

    The function automatically creates technical
    indicators if they are missing.
    """

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    if df is None:

        return {
            "available": False,
            "message": "No market data was provided.",
        }


    if not isinstance(df, pd.DataFrame):

        return {
            "available": False,
            "message": "Market data is not a pandas DataFrame.",
        }


    if df.empty:

        return {
            "available": False,
            "message": "Market data is empty.",
        }


    # --------------------------------------------------------
    # COPY DATA
    # --------------------------------------------------------

    data = df.copy()


    # --------------------------------------------------------
    # NORMALIZE COLUMN NAMES
    # --------------------------------------------------------

    # Sometimes yfinance can return MultiIndex columns.
    # Convert them into normal column names.

    if isinstance(
        data.columns,
        pd.MultiIndex,
    ):

        new_columns = []

        for column in data.columns:

            if isinstance(
                column,
                tuple,
            ):

                # Usually first element is the actual
                # OHLCV column name.
                new_columns.append(
                    str(column[0])
                )

            else:

                new_columns.append(
                    str(column)
                )

        data.columns = new_columns


    # Make column names strings.
    data.columns = [
        str(column)
        for column in data.columns
    ]


    # --------------------------------------------------------
    # CHECK CLOSE COLUMN
    # --------------------------------------------------------

    if "Close" not in data.columns:

        return {
            "available": False,
            "message": (
                "The market data does not contain "
                "a Close price column."
            ),
        }


    # --------------------------------------------------------
    # CREATE TECHNICAL INDICATORS
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in data.columns
    ]


    if missing_features:

        try:

            data = add_indicators(
                data
            )

        except Exception as exc:

            return {
                "available": False,
                "message": (
                    "Unable to create technical "
                    f"indicators: {exc}"
                ),
            }


    # --------------------------------------------------------
    # CHECK FEATURES AFTER INDICATOR CREATION
    # --------------------------------------------------------

    still_missing = [
        feature
        for feature in FEATURES
        if feature not in data.columns
    ]


    if still_missing:

        return {
            "available": False,
            "message": (
                "The following technical indicators "
                "could not be created: "
                + ", ".join(still_missing)
            ),
        }


    # --------------------------------------------------------
    # CREATE TARGET
    # --------------------------------------------------------

    # Tomorrow's close > today's close
    # 1 = UP
    # 0 = DOWN / SAME

    data["Target"] = (
        data["Close"].shift(-1)
        > data["Close"]
    ).astype(int)


    # The final row has no actual next-day value.
    # Remove it before training.

    data = data.iloc[:-1].copy()


    # --------------------------------------------------------
    # CLEAN INF / NAN VALUES
    # --------------------------------------------------------

    data = data.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )


    # Convert features to numeric.

    for feature in FEATURES:

        data[feature] = pd.to_numeric(
            data[feature],
            errors="coerce",
        )


    data["Target"] = pd.to_numeric(
        data["Target"],
        errors="coerce",
    )


    # Remove rows with missing features.

    data = data.dropna(
        subset=FEATURES + ["Target"]
    )


    # --------------------------------------------------------
    # CHECK DATA SIZE
    # --------------------------------------------------------

    if len(data) < 80:

        return {
            "available": False,
            "message": (
                "Not enough usable historical data "
                "after calculating technical indicators. "
                f"Only {len(data)} usable rows are available. "
                "Try selecting 1y, 2y, or a longer period."
            ),
        }


    # --------------------------------------------------------
    # FEATURES AND TARGET
    # --------------------------------------------------------

    X = data[FEATURES].copy()

    y = data["Target"].astype(int).copy()


    # --------------------------------------------------------
    # CHECK TARGET CLASSES
    # --------------------------------------------------------

    if y.nunique() < 2:

        return {
            "available": False,
            "message": (
                "The training data contains only one "
                "target class. The model needs both "
                "UP and DOWN examples."
            ),
        }


    # --------------------------------------------------------
    # TIME-ORDERED TRAIN / TEST SPLIT
    # --------------------------------------------------------

    split_index = int(
        len(data) * 0.80
    )


    # Make sure both sections contain data.

    if split_index < 30:

        return {
            "available": False,
            "message": (
                "Not enough historical observations "
                "for a training/test split."
            ),
        }


    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]


    # --------------------------------------------------------
    # CHECK TRAINING CLASSES
    # --------------------------------------------------------

    if y_train.nunique() < 2:

        return {
            "available": False,
            "message": (
                "The training portion contains only "
                "one class. Try a longer historical period."
            ),
        }


    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )


    # --------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )


    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )


    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )


    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )


    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )


    # --------------------------------------------------------
    # LATEST FEATURE ROW
    # --------------------------------------------------------

    latest_features = (
        data[FEATURES]
        .iloc[-1:]
        .copy()
    )


    # --------------------------------------------------------
    # LATEST DIRECTION
    # --------------------------------------------------------

    latest_prediction = int(
        model.predict(
            latest_features
        )[0]
    )


    next_direction = (
        "UP"
        if latest_prediction == 1
        else "DOWN"
    )


    # --------------------------------------------------------
    # MODEL PROBABILITY
    # --------------------------------------------------------

    confidence = None

    try:

        probabilities = model.predict_proba(
            latest_features
        )[0]

        # Probability belonging to predicted class.
        confidence = float(
            probabilities[
                latest_prediction
            ]
        )

    except Exception:

        confidence = None


    # --------------------------------------------------------
    # WARNING
    # --------------------------------------------------------

    warning = (
        "This machine-learning output is a historical "
        "classification experiment, not a guaranteed "
        "prediction of future stock prices. Past market "
        "patterns can change and the model can be wrong."
    )


    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {

        "available": True,

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        ),

        "confusion_matrix": (
            cm.tolist()
        ),

        "next_direction": (
            next_direction
        ),

        "confidence": confidence,

        "warning": warning,

        "training_rows": int(
            len(X_train)
        ),

        "testing_rows": int(
            len(X_test)
        ),

        "total_rows": int(
            len(data)
        ),

        "features": FEATURES,
    }