
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, recall_score, r2_score
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


st.set_page_config(
    page_title="FAANG Stock Analysis & ML",
    page_icon="📈",
    layout="wide"
)

st.title("📈 FAANG Stock Analysis & Machine Learning")
st.caption("EDA, feature engineering, regression, classification, and XGBoost tuning")


# --------------------------------------------------
# Functions
# --------------------------------------------------
@st.cache_data
def load_data(uploaded_file):
    return pd.read_excel(uploaded_file)


def prepare_data(raw_df):
    df = raw_df.copy()

    required_columns = [
        "Source.Name", "Date", "Close", "High", "Low", "Adj Close"
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(
        by=["Source.Name", "Date"]
    ).reset_index(drop=True)

    # EDA features
    df["EDA_Daily_Return"] = (
        df.groupby("Source.Name")["Adj Close"].pct_change()
    )

    df["SMA_50_Adj"] = (
        df.groupby("Source.Name")["Adj Close"]
        .transform(lambda x: x.rolling(50).mean())
    )

    df["SMA_200_Adj"] = (
        df.groupby("Source.Name")["Adj Close"]
        .transform(lambda x: x.rolling(200).mean())
    )

    # ML features
    df["Close_Lag1"] = (
        df.groupby("Source.Name")["Close"].shift(1)
    )
    df["Close_Lag5"] = (
        df.groupby("Source.Name")["Close"].shift(5)
    )

    df["SMA_10"] = (
        df.groupby("Source.Name")["Close"]
        .transform(lambda x: x.rolling(10).mean())
    )

    df["SMA_50"] = (
        df.groupby("Source.Name")["Close"]
        .transform(lambda x: x.rolling(50).mean())
    )

    df["High_Low_Range"] = df["High"] - df["Low"]

    df["Daily_Return"] = (
        df.groupby("Source.Name")["Close"].pct_change() * 100
    )

    # Targets
    df["Target_Price"] = (
        df.groupby("Source.Name")["Close"].shift(-1)
    )

    df["Target_Direction"] = (
        df["Target_Price"] > df["Close"]
    ).astype(int)

    # Same behavior as the notebook:
    # remove rows created by lag/rolling/target operations
    df = df.dropna().reset_index(drop=True)

    return df


@st.cache_data
def run_models(df):
    features = [
        "Close_Lag1",
        "Close_Lag5",
        "SMA_10",
        "SMA_50",
        "High_Low_Range",
        "Daily_Return"
    ]

    X = df[features]
    y_reg = df["Target_Price"]
    y_cls = df["Target_Direction"]

    train_size = int(len(df) * 0.8)

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]

    y_train_reg = y_reg.iloc[:train_size]
    y_test_reg = y_reg.iloc[train_size:]

    y_train_cls = y_cls.iloc[:train_size]
    y_test_cls = y_cls.iloc[train_size:]

    # ---------------- REGRESSION ----------------
    reg_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),
        "XGBoost Regressor": XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
    }

    reg_results = []
    reg_pipelines = {}

    for name, model in reg_models.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train_reg)
        preds = pipeline.predict(X_test)

        reg_results.append({
            "Model": name,
            "MAE ($)": round(
                mean_absolute_error(y_test_reg, preds), 2
            ),
            "RMSE ($)": round(
                np.sqrt(mean_squared_error(y_test_reg, preds)), 2
            ),
            "R2 Score": round(
                r2_score(y_test_reg, preds), 4
            )
        })

        reg_pipelines[name] = pipeline

    reg_df = pd.DataFrame(reg_results).sort_values(
        by="MAE ($)"
    ).reset_index(drop=True)

    # ---------------- CLASSIFICATION ----------------
    num_neg = (y_train_cls == 0).sum()
    num_pos = (y_train_cls == 1).sum()
    scale_pos_weight = (
        num_neg / num_pos if num_pos > 0 else 1.0
    )

    cls_models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced"
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42
        ),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
    }

    cls_results = []
    cls_pipelines = {}

    for name, model in cls_models.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train_cls)
        preds = pipeline.predict(X_test)

        cls_results.append({
            "Model": name,
            "Accuracy": round(
                accuracy_score(y_test_cls, preds), 4
            ),
            "Precision": round(
                precision_score(y_test_cls, preds, zero_division=0), 4
            ),
            "Recall": round(
                recall_score(y_test_cls, preds, zero_division=0), 4
            ),
            "F1 Score": round(
                f1_score(y_test_cls, preds, zero_division=0), 4
            )
        })

        cls_pipelines[name] = pipeline

    cls_df = pd.DataFrame(cls_results).sort_values(
        by="F1 Score",
        ascending=False
    ).reset_index(drop=True)

    # ---------------- XGBOOST TUNING ----------------
    tscv = TimeSeriesSplit(n_splits=3)

    xgb_pipe = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            XGBClassifier(
                random_state=42,
                scale_pos_weight=scale_pos_weight
            )
        )
    ])

    param_grid = {
        "model__n_estimators": [50, 100],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.01, 0.05]
    }

    grid_search = GridSearchCV(
        estimator=xgb_pipe,
        param_grid=param_grid,
        cv=tscv,
        scoring="f1",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train_cls)

    tuned_xgb_cls = grid_search.best_estimator_

    # Best regression = same selection logic as notebook
    best_reg_name = reg_df.iloc[0]["Model"]
    best_reg_pipeline = reg_pipelines[best_reg_name]

    df_test = df.iloc[train_size:].copy()

    df_test["Predicted_Close_Price"] = (
        best_reg_pipeline.predict(X_test)
    )

    df_test["Predicted_Market_Direction"] = (
        tuned_xgb_cls.predict(X_test)
    )

    return {
        "features": features,
        "reg_df": reg_df,
        "cls_df": cls_df,
        "reg_pipelines": reg_pipelines,
        "cls_pipelines": cls_pipelines,
        "best_reg_name": best_reg_name,
        "best_reg_pipeline": best_reg_pipeline,
        "tuned_xgb_cls": tuned_xgb_cls,
        "best_params": grid_search.best_params_,
        "df_test": df_test,
        "train_size": train_size,
        "X_test": X_test,
        "y_test_reg": y_test_reg,
        "y_test_cls": y_test_cls
    }


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("📂 Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload Cleaned_FAANG_Data 1.xlsx",
    type=["xlsx"]
)

if uploaded_file is None:
    st.info(
        "Upload your cleaned FAANG Excel file from the sidebar to start."
    )
    st.markdown("""
### Expected columns
The uploaded file should contain at least:

- `Source.Name`
- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Adj Close`
- `Volume`
""")
    st.stop()

try:
    raw_df = load_data(uploaded_file)
    df = prepare_data(raw_df)
except Exception as e:
    st.error(f"Could not prepare the data: {e}")
    st.stop()


# --------------------------------------------------
# Basic information
# --------------------------------------------------
companies = sorted(df["Source.Name"].unique())

st.sidebar.header("🔎 Filters")

selected_company = st.sidebar.selectbox(
    "Company",
    ["All"] + companies
)

if selected_company == "All":
    view_df = df.copy()
else:
    view_df = df[df["Source.Name"] == selected_company].copy()

st.sidebar.write(f"Rows available: {len(view_df):,}")


# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA",
    "🤖 Regression",
    "🎯 Classification",
    "🔮 Predictions"
])


# ==================================================
# EDA
# ==================================================
with tab1:
    st.header("Exploratory Data Analysis")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", f"{len(raw_df):,}")
    c2.metric("Companies", raw_df["Source.Name"].nunique())
    c3.metric("Missing Values", int(raw_df.isnull().sum().sum()))
    c4.metric("Duplicates", int(raw_df.duplicated().sum()))

    st.subheader("Dataset Preview")
    st.dataframe(
        view_df.head(20),
        use_container_width=True
    )

    st.subheader("Price Trend")

    fig, ax = plt.subplots(figsize=(12, 5))

    for company in (
        companies if selected_company == "All"
        else [selected_company]
    ):
        temp = view_df[
            view_df["Source.Name"] == company
        ]

        ax.plot(
            temp["Date"],
            temp["Adj Close"],
            label=company
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Moving Averages")

    if selected_company == "All":
        ma_company = st.selectbox(
            "Choose company for moving averages",
            companies,
            key="ma_company"
        )
    else:
        ma_company = selected_company

    ma_df = df[
        df["Source.Name"] == ma_company
    ].copy()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        ma_df["Date"],
        ma_df["Adj Close"],
        label="Adj Close"
    )
    ax.plot(
        ma_df["Date"],
        ma_df["SMA_50_Adj"],
        label="SMA 50"
    )
    ax.plot(
        ma_df["Date"],
        ma_df["SMA_200_Adj"],
        label="SMA 200"
    )

    ax.set_title(f"{ma_company} - Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Company Statistics")

    company_stats = df.groupby(
        "Source.Name"
    )["Adj Close"].agg(
        ["mean", "median", "std", "min", "max"]
    )

    st.dataframe(
        company_stats.round(3),
        use_container_width=True
    )

    st.subheader("Daily Return Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))

    for company in (
        companies if selected_company == "All"
        else [selected_company]
    ):
        temp = view_df[
            view_df["Source.Name"] == company
        ]

        sns.histplot(
            temp["EDA_Daily_Return"].dropna(),
            kde=True,
            label=company,
            ax=ax,
            alpha=0.35
        )

    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Return Correlation")

    pivot_returns = df.pivot_table(
        index="Date",
        columns="Source.Name",
        values="Daily_Return"
    )

    corr = pivot_returns.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )
    ax.set_title("Correlation of Daily Returns")
    st.pyplot(fig)
    plt.close(fig)


# ==================================================
# Run ML
# ==================================================
with st.spinner("Training the models..."):
    results = run_models(df)


# ==================================================
# Regression
# ==================================================
with tab2:
    st.header("Regression Models")
    st.write(
        "Target: predict the next day's closing price."
    )

    st.subheader("Model Comparison")
    st.dataframe(
        results["reg_df"],
        use_container_width=True
    )

    selected_reg = st.selectbox(
        "Select regression model",
        results["reg_df"]["Model"].tolist()
    )

    st.write(
        f"**Selected model:** {selected_reg}"
    )

    selected_row = results["reg_df"][
        results["reg_df"]["Model"] == selected_reg
    ].iloc[0]

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f'{selected_row["MAE ($)"]:.2f}')
    m2.metric("RMSE", f'{selected_row["RMSE ($)"]:.2f}')
    m3.metric("R²", f'{selected_row["R2 Score"]:.4f}')

    st.subheader("Actual vs Predicted")

    model = results["reg_pipelines"][selected_reg]
    predictions = model.predict(results["X_test"])

    comparison = pd.DataFrame({
        "Actual": results["y_test_reg"].values,
        "Predicted": predictions
    })

    st.line_chart(
        comparison.set_index(
            pd.RangeIndex(len(comparison))
        )
    )

    st.download_button(
        "Download Regression Results",
        results["reg_df"].to_csv(index=False),
        "regression_results.csv",
        "text/csv"
    )


# ==================================================
# Classification
# ==================================================
with tab3:
    st.header("Classification Models")
    st.write(
        "Target: predict whether the next day's price goes up or down."
    )

    st.subheader("Model Comparison")
    st.dataframe(
        results["cls_df"],
        use_container_width=True
    )

    selected_cls = st.selectbox(
        "Select classification model",
        results["cls_df"]["Model"].tolist()
    )

    row = results["cls_df"][
        results["cls_df"]["Model"] == selected_cls
    ].iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f'{row["Accuracy"]:.4f}')
    c2.metric("Precision", f'{row["Precision"]:.4f}')
    c3.metric("Recall", f'{row["Recall"]:.4f}')
    c4.metric("F1 Score", f'{row["F1 Score"]:.4f}')

    st.subheader("XGBoost Hyperparameter Tuning")

    st.write("Best parameters found by TimeSeriesSplit:")
    st.json(results["best_params"])


# ==================================================
# Predictions
# ==================================================
with tab4:
    st.header("Prediction Output")

    st.write(
        f"Best regression model selected by the notebook logic: "
        f"**{results['best_reg_name']}**"
    )

    prediction_df = results["df_test"].copy()

    if selected_company != "All":
        prediction_df = prediction_df[
            prediction_df["Source.Name"] == selected_company
        ]

    display_cols = [
        "Source.Name",
        "Date",
        "Close",
        "Target_Price",
        "Predicted_Close_Price",
        "Target_Direction",
        "Predicted_Market_Direction"
    ]

    st.dataframe(
        prediction_df[display_cols].tail(100),
        use_container_width=True
    )

    st.subheader("Predicted vs Actual Next-Day Price")

    if len(prediction_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            prediction_df["Date"],
            prediction_df["Target_Price"],
            label="Actual Next-Day Price"
        )

        ax.plot(
            prediction_df["Date"],
            prediction_df["Predicted_Close_Price"],
            label="Predicted Price"
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(alpha=0.2)

        st.pyplot(fig)
        plt.close(fig)

    csv_data = results["df_test"].to_csv(index=False)

    st.download_button(
        "⬇️ Download FAANG Predictions CSV",
        csv_data,
        "FAANG_Predictions_For_BI.csv",
        "text/csv"
    )

st.divider()
st.caption(
    "Model workflow follows the uploaded FAANG notebook: "
    "EDA → Feature Engineering → Time-based Train/Test Split → "
    "Regression & Classification → XGBoost Tuning → Predictions."
)
