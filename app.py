import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Cars Analytics App",
    page_icon="🚗",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():

    df = pd.read_csv("cars_dataset.csv")

    # Drop unnecessary columns
    drop_cols = ["Unnamed: 0", "vin", "lot"]

    for col in drop_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Handle missing values
    if "mileage" in df.columns:
        df["mileage"] = df["mileage"].fillna(df["mileage"].median())

    # Remove duplicates
    df = df.drop_duplicates()

    return df

df = load_data()

# =========================
# MODEL TRAINING
# =========================

X = df.drop("price", axis=1)
y = df["price"]

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols = X.select_dtypes(exclude="object").columns.tolist()

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model_pipeline.fit(X_train, y_train)

predictions = model_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("🚗 Navigation")

page = st.sidebar.radio(
    "Go To",
    ["Home", "Insights Dashboard", "Prediction"]
)

# =========================
# HOME PAGE
# =========================

if page == "Home":

    st.title("🚗 Cars Price Analytics & Prediction App")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Cars", len(df))

    with col2:
        st.metric("Average Price", f"${int(df['price'].mean())}")

    with col3:
        st.metric("Average Mileage", f"{int(df['mileage'].mean())}")

    st.markdown("---")

    st.subheader("📌 About Dataset")

    st.write(
        "This application analyzes used cars data and predicts car prices "
        "using Machine Learning."
    )

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📈 Model Performance")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("MAE", round(mae, 2))

    with c2:
        st.metric("R² Score", round(r2, 2))

# =========================
# INSIGHTS PAGE
# =========================

elif page == "Insights Dashboard":

    st.title("📊 Insights Dashboard")

    st.markdown("---")

    # FILTERS
    st.sidebar.header("Dashboard Filters")

    selected_brand = st.sidebar.multiselect(
        "Select Brand",
        options=df["brand"].unique(),
        default=df["brand"].unique()
    )

    selected_color = st.sidebar.multiselect(
        "Select Color",
        options=df["color"].unique(),
        default=df["color"].unique()
    )

    filtered_df = df[
        (df["brand"].isin(selected_brand)) &
        (df["color"].isin(selected_color))
    ]

    # KPIs
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric("Cars Count", len(filtered_df))

    with k2:
        st.metric(
            "Average Price",
            f"${int(filtered_df['price'].mean())}"
        )

    with k3:
        st.metric(
            "Maximum Price",
            f"${int(filtered_df['price'].max())}"
        )

    with k4:
        st.metric(
            "Average Mileage",
            f"{int(filtered_df['mileage'].mean())}"
        )

    st.markdown("---")

    # CHARTS

    col1, col2 = st.columns(2)

    with col1:

        brand_chart = px.bar(
            filtered_df.groupby("brand")["price"]
            .mean()
            .reset_index(),
            x="brand",
            y="price",
            title="Average Price by Brand"
        )

        st.plotly_chart(brand_chart, use_container_width=True)

    with col2:

        year_chart = px.line(
            filtered_df.groupby("year")["price"]
            .mean()
            .reset_index(),
            x="year",
            y="price",
            title="Average Price by Year"
        )

        st.plotly_chart(year_chart, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:

        scatter_chart = px.scatter(
            filtered_df,
            x="mileage",
            y="price",
            color="brand",
            title="Mileage vs Price"
        )

        st.plotly_chart(scatter_chart, use_container_width=True)

    with col4:

        color_chart = px.pie(
            filtered_df,
            names="color",
            title="Cars Distribution by Color"
        )

        st.plotly_chart(color_chart, use_container_width=True)

    st.subheader("📋 Filtered Dataset")
    st.dataframe(filtered_df)

# =========================
# PREDICTION PAGE
# =========================

elif page == "Prediction":

    st.title("🤖 Car Price Prediction")

    st.markdown("Enter car information to predict price")

    st.markdown("---")

    # INPUTS

    brand = st.selectbox(
        "Brand",
        sorted(df["brand"].unique())
    )

    model = st.selectbox(
        "Model",
        sorted(df["model"].unique())
    )

    year = st.number_input(
        "Year",
        min_value=int(df["year"].min()),
        max_value=int(df["year"].max()),
        value=2018
    )

    title_status = st.selectbox(
        "Title Status",
        sorted(df["title_status"].unique())
    )

    mileage = st.number_input(
        "Mileage",
        min_value=0,
        value=50000
    )

    color = st.selectbox(
        "Color",
        sorted(df["color"].unique())
    )

    state = st.selectbox(
        "State",
        sorted(df["state"].unique())
    )

    country = st.selectbox(
        "Country",
        sorted(df["country"].unique())
    )

    condition = st.selectbox(
        "Condition",
        sorted(df["condition"].unique())
    )

    if st.button("Predict Price"):

        input_df = pd.DataFrame({

            "brand": [brand],
            "model": [model],
            "year": [year],
            "title_status": [title_status],
            "mileage": [mileage],
            "color": [color],
            "state": [state],
            "country": [country],
            "condition": [condition]

        })

        predicted_price = model_pipeline.predict(input_df)[0]

        st.success(
            f"Predicted Car Price: ${predicted_price:,.2f}"
        )

        st.balloons()
