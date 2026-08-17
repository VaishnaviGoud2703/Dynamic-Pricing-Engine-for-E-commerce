import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
st.title("Dynamic Pricing Engine for E-commerce")
df = pd.read_csv("ecommerce_dynamic_pricing_dataset.csv")

st.subheader("📊 Dataset")
st.write(df.head())
features = [
    "product_category",
    "region",
    "base_price",
    "competitor_price",
    "inventory",
    "demand_score",
    "sales_last_7_days",
    "customer_rating",
    "discount_percent",
    "season",
    "promotion",
    "page_views",
    "price_elasticity"
]

target = "recommended_price"

X = df[features]
y = df[target]


categorical_features = [
    "product_category",
    "region",
    "season"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)



model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("MAE", f"{mae:.2f}")

with col2:
    st.metric("R² Score", f"{r2:.2f}")

st.subheader("Predict Recommended Price")

col1, col2 = st.columns(2)

with col1:

    category = st.selectbox(
        "Product Category",
        df["product_category"].unique()
    )

    region = st.selectbox(
        "Region",
        df["region"].unique()
    )

    base_price = st.number_input(
        "Base Price",
        min_value=100.0,
        value=1000.0
    )

    competitor_price = st.number_input(
        "Competitor Price",
        min_value=100.0,
        value=1000.0
    )

    inventory = st.number_input(
        "Inventory",
        min_value=1,
        value=100
    )

    demand_score = st.slider(
        "Demand Score",
        10.0,
        100.0,
        50.0
    )

    sales_last_7_days = st.number_input(
        "Sales in Last 7 Days",
        min_value=0,
        value=50
    )


with col2:

    customer_rating = st.slider(
        "Customer Rating",
        1.0,
        5.0,
        4.0
    )

    discount_percent = st.slider(
        "Discount (%)",
        0.0,
        50.0,
        10.0
    )

    season = st.selectbox(
        "Season",
        df["season"].unique()
    )

    promotion = st.selectbox(
        "Promotion",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    page_views = st.number_input(
        "Page Views",
        min_value=0,
        value=1000
    )

    price_elasticity = st.slider(
        "Price Elasticity",
        0.3,
        2.0,
        1.0
    )



if st.button("🚀 Calculate Recommended Price"):

    input_data = pd.DataFrame({
        "product_category": [category],
        "region": [region],
        "base_price": [base_price],
        "competitor_price": [competitor_price],
        "inventory": [inventory],
        "demand_score": [demand_score],
        "sales_last_7_days": [sales_last_7_days],
        "customer_rating": [customer_rating],
        "discount_percent": [discount_percent],
        "season": [season],
        "promotion": [promotion],
        "page_views": [page_views],
        "price_elasticity": [price_elasticity]
    })

    recommended_price = model.predict(input_data)[0]

    st.success(
        f"💰 Recommended Price: ₹{recommended_price:,.2f}"
    )