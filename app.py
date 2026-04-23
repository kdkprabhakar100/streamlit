import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="Air Quality Forecasting Dashboard",
    page_icon="🌫️",
    layout="wide"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .main {
        background-color: #0e1117;
    }

    h1, h2, h3 {
        color: white;
    }

    .hero-box {
        background: linear-gradient(135deg, #1f2937, #111827);
        padding: 28px;
        border-radius: 18px;
        margin-bottom: 20px;
        border: 1px solid #2d3748;
    }

    .hero-title {
        color: #ffffff;
        font-size: 40px;
        font-weight: 700;
        margin-bottom: 10px;
        text-align: center;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 18px;
        text-align: center;
        margin-bottom: 0px;
    }

    .small-note {
        color: #cbd5e1;
        font-size: 15px;
    }

    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #253046;
        padding: 14px;
        border-radius: 14px;
    }

    .result-box {
        padding: 18px;
        border-radius: 16px;
        margin-top: 10px;
        margin-bottom: 10px;
        border: 1px solid #253046;
        background-color: #111827;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-box">
    <div class="hero-title">🌫️ Intelligent Air Quality Forecasting System</div>
    <div class="hero-subtitle">
        Predict PM2.5 concentration and analyze pollution risk using machine learning
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("**Developed by: Manushi Paudel**")

st.markdown("""
### 📘 Project Overview

This system predicts PM2.5 air pollution levels and classifies pollution risk
based on environmental conditions such as temperature, pressure, dew point,
wind speed, rain, and snow.
""")

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    if "No" in df.columns:
        df = df.drop(columns=["No"])

    if all(col in df.columns for col in ["year", "month", "day", "hour"]):
        df["date"] = pd.to_datetime(df[["year", "month", "day", "hour"]], errors="coerce")

    df = df[df["pm2.5"].notna()].copy()

    if "month" in df.columns:
        df["season"] = df["month"] % 12 // 3 + 1
        df["season_name"] = df["season"].map({
            1: "Winter",
            2: "Spring",
            3: "Summer",
            4: "Autumn"
        })

    if "hour" in df.columns:
        df["time_of_day"] = pd.cut(
            df["hour"],
            bins=[-1, 5, 11, 17, 23],
            labels=["Night", "Morning", "Afternoon", "Evening"]
        )

    return df


def pollution_risk(pm):
    if pm <= 50:
        return "Low"
    elif pm <= 100:
        return "Moderate"
    elif pm <= 150:
        return "High"
    return "Very High"


@st.cache_resource
def train_models(df):
    df = df.copy()
    df["risk"] = df["pm2.5"].apply(pollution_risk)

    numeric_features = [
        col for col in ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir", "year", "month", "day", "hour"]
        if col in df.columns
    ]
    categorical_features = [
        col for col in ["cbwd", "season_name", "time_of_day"]
        if col in df.columns
    ]
    selected_features = numeric_features + categorical_features

    X = df[selected_features].copy()
    y_reg = df["pm2.5"].copy()
    y_clf = df["risk"].copy()

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    reg_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        ))
    ])

    clf_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        ))
    ])

    reg_model.fit(X_train_reg, y_train_reg)
    clf_model.fit(X_train_clf, y_train_clf)

    reg_pred = reg_model.predict(X_test_reg)
    clf_pred = clf_model.predict(X_test_clf)

    reg_metrics = {
        "MAE": float(mean_absolute_error(y_test_reg, reg_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test_reg, reg_pred))),
        "R²": float(r2_score(y_test_reg, reg_pred)),
    }

    clf_metrics = {
        "Accuracy": float(accuracy_score(y_test_clf, clf_pred)),
        "F1 Score": float(f1_score(y_test_clf, clf_pred, average="weighted")),
    }

    feature_importance_df = None
    try:
        model = reg_model.named_steps["model"]
        transformed_names = reg_model.named_steps["preprocessor"].get_feature_names_out()
        importances = model.feature_importances_

        feature_importance_df = pd.DataFrame({
            "Feature": transformed_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
    except Exception:
        feature_importance_df = None

    return {
        "reg_model": reg_model,
        "clf_model": clf_model,
        "reg_metrics": reg_metrics,
        "clf_metrics": clf_metrics,
        "features_num": numeric_features,
        "features_cat": categorical_features,
        "selected_features": selected_features,
        "feature_importance_df": feature_importance_df
    }


DATA_PATH = "PRSA_data_2010.1.1-2014.12.31.csv"
df = load_data(DATA_PATH)
models = train_models(df)

if "risk" not in df.columns:
    df["risk"] = df["pm2.5"].apply(pollution_risk)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "EDA", "Model Results", "Predict", "Impact"]
)

with tab1:
    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Average PM2.5", f"{df['pm2.5'].mean():.2f}")

    st.markdown("### Preview of Dataset")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Pollution Risk Distribution")
    risk_counts = df["risk"].value_counts().reindex(
        ["Low", "Moderate", "High", "Very High"]
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(risk_counts.index, risk_counts.values)
    ax.set_title("Pollution Risk Categories")
    ax.set_xlabel("Risk")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.markdown("### 📌 Key Insights")
    st.write(f"- Average PM2.5 is **{df['pm2.5'].mean():.2f}**.")
    st.write("- Pollution varies significantly with atmospheric conditions.")
    st.write("- Wind speed, dew point, and pressure may influence air quality patterns.")
    st.write("- Pollution risk classification helps translate raw PM2.5 values into meaningful categories.")

with tab2:
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df["pm2.5"].dropna(), bins=40)
        ax.set_title("PM2.5 Distribution")
        ax.set_xlabel("PM2.5")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot(df["pm2.5"].dropna(), vert=False)
        ax.set_title("PM2.5 Boxplot")
        ax.set_xlabel("PM2.5")
        st.pyplot(fig)

    numeric_df = df.select_dtypes(include=np.number)

    if not numeric_df.empty:
        st.subheader("Correlation Heatmap")
        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(corr, aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)
        ax.set_title("Correlation Heatmap")
        fig.colorbar(im)
        st.pyplot(fig)

    scatter_features = [col for col in ["TEMP", "PRES", "DEWP", "Iws"] if col in df.columns]
    if scatter_features:
        st.subheader("Feature vs PM2.5 Analysis")
        sample_df = df.sample(min(3000, len(df)), random_state=42)

        for feature in scatter_features:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(sample_df[feature], sample_df["pm2.5"], alpha=0.5)
            ax.set_title(f"{feature} vs PM2.5")
            ax.set_xlabel(feature)
            ax.set_ylabel("PM2.5")
            st.pyplot(fig)

    if "date" in df.columns:
        st.subheader("Daily Average PM2.5 Over Time")
        daily = df.set_index("date")["pm2.5"].resample("D").mean().dropna()

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(daily.index, daily.values)
        ax.set_title("Daily Average PM2.5")
        ax.set_xlabel("Date")
        ax.set_ylabel("PM2.5")
        st.pyplot(fig)

with tab3:
    st.subheader("Model Performance")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Regression")
        for k, v in models["reg_metrics"].items():
            st.metric(k, f"{v:.3f}")

    with c2:
        st.markdown("### Classification")
        for k, v in models["clf_metrics"].items():
            st.metric(k, f"{v:.3f}")

    st.markdown("### 📊 Model Insight")
    st.write("""
    Random Forest performs better for this kind of problem because it can capture
    more complex nonlinear relationships between environmental variables and pollution levels.
    """)

    st.markdown("### Features Used")
    st.write("**Numerical:**", models["features_num"])
    st.write("**Categorical:**", models["features_cat"])

    if models["feature_importance_df"] is not None:
        st.markdown("### Feature Importance")
        st.dataframe(models["feature_importance_df"].head(15), use_container_width=True)

        chart_df = models["feature_importance_df"].head(10).set_index("Feature")
        st.bar_chart(chart_df)

with tab4:
    st.subheader("Make a Prediction")
    st.sidebar.header("Input Environmental Values")

    def get_default(col, fallback):
        if col in df.columns:
            return float(df[col].median())
        return fallback

    temp = st.sidebar.number_input("Temperature (TEMP)", value=get_default("TEMP", 20.0))
    pres = st.sidebar.number_input("Pressure (PRES)", value=get_default("PRES", 1010.0))
    dewp = st.sidebar.number_input("Dew Point (DEWP)", value=get_default("DEWP", 10.0))
    iws = st.sidebar.number_input("Wind Speed (Iws)", value=get_default("Iws", 1.0))
    is_val = st.sidebar.number_input("Cumulated Hours of Snow (Is)", value=get_default("Is", 0.0))
    ir_val = st.sidebar.number_input("Cumulated Hours of Rain (Ir)", value=get_default("Ir", 0.0))

    year = int(st.sidebar.number_input(
        "Year",
        value=int(df["year"].mode()[0]) if "year" in df.columns else 2014,
        step=1
    ))
    month = int(st.sidebar.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=int(df["month"].mode()[0]) if "month" in df.columns else 1,
        step=1
    ))
    day = int(st.sidebar.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=int(df["day"].mode()[0]) if "day" in df.columns else 1,
        step=1
    ))
    hour = int(st.sidebar.number_input(
        "Hour",
        min_value=0,
        max_value=23,
        value=int(df["hour"].mode()[0]) if "hour" in df.columns else 12,
        step=1
    ))

    cbwd_options = sorted(df["cbwd"].dropna().astype(str).unique().tolist()) if "cbwd" in df.columns else ["cv"]
    cbwd = st.sidebar.selectbox("Wind Direction (cbwd)", options=cbwd_options)

    season_num = month % 12 // 3 + 1
    season_name = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}[season_num]

    if hour <= 5:
        time_of_day = "Night"
    elif hour <= 11:
        time_of_day = "Morning"
    elif hour <= 17:
        time_of_day = "Afternoon"
    else:
        time_of_day = "Evening"

    input_row = {
        "DEWP": dewp,
        "TEMP": temp,
        "PRES": pres,
        "Iws": iws,
        "Is": is_val,
        "Ir": ir_val,
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "cbwd": cbwd,
        "season_name": season_name,
        "time_of_day": time_of_day
    }

    input_df = pd.DataFrame([input_row])
    input_df = input_df[[col for col in models["selected_features"] if col in input_df.columns]]

    st.markdown("### Current Input Summary")
    st.dataframe(input_df, use_container_width=True)

    if st.button("Predict Air Quality"):
        pm_pred = float(models["reg_model"].predict(input_df)[0])
        risk_pred = models["clf_model"].predict(input_df)[0]

        st.markdown("### 🔍 Prediction Result")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Predicted PM2.5", f"{pm_pred:.2f}")
        with col_b:
            st.metric("Predicted Risk", str(risk_pred))

        if risk_pred == "Low":
            st.success(f"🟢 Risk Level: {risk_pred}")
        elif risk_pred == "Moderate":
            st.warning(f"🟡 Risk Level: {risk_pred}")
        elif risk_pred == "High":
            st.warning(f"🟠 Risk Level: {risk_pred}")
        else:
            st.error(f"🔴 Risk Level: {risk_pred}")

        st.markdown(
            f"""
            <div class="result-box">
                <h4 style="color:white;">Interpretation</h4>
                <p class="small-note">
                    Based on the environmental values you entered, the model predicts a PM2.5 concentration
                    of <b>{pm_pred:.2f}</b>, which falls into the <b>{risk_pred}</b> pollution category.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

with tab5:
    st.subheader("🌍 Real-World Impact")

    st.markdown("""
### 🌍 Why This Matters

Air pollution is a major global health risk. This system helps:
- detect dangerous pollution levels early
- support environmental monitoring
- improve public awareness
""")

    st.markdown("""
- Helps monitor air pollution trends  
- Supports public health risk awareness  
- Can be extended into real-time smart city systems  
- Useful for environmental monitoring and forecasting  
""")

    st.markdown("### Project Summary")
    st.write("""
    This dashboard combines machine learning, environmental analysis, and visualization
    to forecast PM2.5 and classify pollution risk. It demonstrates how data science
    can support environmental decision-making in a practical and interactive way.
    """)

    st.markdown("### Future Improvements")
    st.write("""
    - Add advanced models like XGBoost or LSTM  
    - Connect live API-based weather data  
    - Add more pollutants and health-based alerts  
    """)