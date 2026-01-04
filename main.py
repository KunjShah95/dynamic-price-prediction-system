import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from fastapi import FastAPI
from typing import List, Optional
from scipy.stats import ttest_rel

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

DATA_PATH = "data/simulated_product_sales.csv"
RANDOM_STATE = 42

def load_and_prepare_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df.sort_values(["product_id", "date"], inplace=True)

    df["price_ratio"] = df["price"] / df["base_price"]
    df["competitor_diff"] = df["price"] - df["competitor_price"]

    df["lag_1"] = df.groupby("product_id")["demand"].shift(1)
    df["lag_7"] = df.groupby("product_id")["demand"].shift(7)

    df["rolling_7"] = (
        df.groupby("product_id")["demand"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    return df.dropna()

#train demand model
def train_demand_model(df):
    FEATURES=[
         "price", "price_ratio", "discount_pct",
        "competitor_diff", "seasonality",
        "lag_1", "lag_7", "rolling_7"
    ]

    train =df[df["date"] < "2023-01-01"]
    test = df[df["date"] >= "2023-10-01"]

    
    X_train, y_train = train[FEATURES], train["demand"]
    X_test, y_test   = test[FEATURES], test["demand"]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Demand MAE:", mean_absolute_error(y_test, preds))

    return model, FEATURES, test

#price simulation and optimization
def simulate_prices(
    model,
    base_row,
    FEATURES,
    min_price_ratio=0.85,
    max_price_ratio=1.15
):
    price_ratios = np.linspace(min_price_ratio, max_price_ratio, 20)
    rows = []

    for r in price_ratios:
        row = base_row.copy()
        row["price"] = row["base_price"] * r
        row["price_ratio"] = r
        row["discount_pct"] = max(0, 1 - r)

        X = row[FEATURES].values.reshape(1, -1)
        demand = max(0, model.predict(X)[0])
        revenue = row["price"] * demand

        rows.append((row["price"], demand, revenue))

    return pd.DataFrame(rows, columns=["price", "predicted_demand", "revenue"])

#elasticity estimation
def estimate_elasticity(sim_df):
    sim_df["log_price"] = np.log(sim_df["price"])
    sim_df["log_demand"] = np.log(sim_df["predicted_demand"] + 1)

    elasticity = np.polyfit(
        sim_df["log_price"],
        sim_df["log_demand"],
        1
    )[0]

    return elasticity

# visualization
def plot_revenue_curve(sim_df, optimal_price):
    plt.figure()
    plt.plot(sim_df["price"], sim_df["revenue"])
    plt.axvline(optimal_price)
    plt.xlabel("Price")
    plt.ylabel("Revenue")
    plt.title("Revenue Optimization Curve")
    plt.show()

#shap explainability
def explain_demand_predictions(model,X_sample,feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type="bar"
    )


# --- A/B test simulator (moved here) ---
def estimate_own_elasticities(df: pd.DataFrame) -> dict:
    """Estimate log-log elasticity per product using historical price & demand.

    Returns mapping product_id -> elasticity (slope in log-demand vs log-price).
    """
    elasts = {}
    for pid, g in df.groupby("product_id"):
        tmp = g.copy()
        tmp = tmp[tmp["price"] > 0]
        if len(tmp) < 3:
            # fallback to true_elasticity column if available
            elasts[pid] = float(tmp.get("true_elasticity", pd.Series([-1.0])).iloc[0])
            continue

        tmp["log_price"] = np.log(tmp["price"])
        tmp["log_demand"] = np.log(tmp["demand"] + 1)

        # simple linear fit on logs
        coef = np.polyfit(tmp["log_price"], tmp["log_demand"], 1)[0]
        elasts[pid] = float(coef)

    return elasts


def create_cross_price_matrix(
    products: List[str],
    own_elasticities: dict,
    cross_share: float = 0.25,
) -> pd.DataFrame:
    """Create a simple cross-price/cannibalization matrix.

    The column j represents the impact on other products when product j price changes.
    Entry M[i, j] is the fraction of a demand-change in product j that flows to product i.
    """
    mat = pd.DataFrame(0.0, index=products, columns=products)

    for j in products:
        own = abs(own_elasticities.get(j, 1.0))
        others = [p for p in products if p != j]
        if not others:
            continue

        # distribute a fraction of the own-effect to other products proportional to their baseline sizes
        per_other = (cross_share * own) / len(others)
        for i in others:
            mat.at[i, j] = per_other

    return mat


def simulate_ab_test(
    df: pd.DataFrame,
    treated_products: List[str],
    price_multiplier: float = 0.9,
    days: Optional[List[pd.Timestamp]] = None,
    cross_matrix: Optional[pd.DataFrame] = None,
    use_estimated_elasticities: bool = True,
) -> dict:
    """Simulate an A/B price experiment across multiple products accounting for cannibalization.

    Args:
        df: historical data containing columns ['date','product_id','price','demand']
        treated_products: list of product_ids to apply the price change to
        price_multiplier: multiplier applied to price for treated SKUs (e.g., 0.9 = 10% discount)
        days: optional list of dates to simulate; if None, uses all dates in df
        cross_matrix: optional pre-built cross-price matrix; if None it's created automatically

    Returns:
        dict with keys: 'baseline_daily', 'test_daily', 'summary', 'cross_matrix'
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()

    if days is not None:
        data = data[data["date"].isin(days)]

    products = sorted(data["product_id"].unique())

    # baseline time-series (daily demand and price per product)
    baseline = (
        data.groupby(["date", "product_id"])[["demand", "price"]]
        .first()
        .reset_index()
    )

    # estimate elasticities if needed
    if use_estimated_elasticities:
        elasts = estimate_own_elasticities(data)
    else:
        # fall back to using any provided true_elasticity column, or a default
        elasts = {}
        for pid in products:
            sample = data[data["product_id"] == pid]
            if "true_elasticity" in sample.columns:
                elasts[pid] = float(sample["true_elasticity"].iloc[0])
            else:
                elasts[pid] = -1.0

    # build cross-price / cannibalization matrix if not supplied
    if cross_matrix is None:
        cross_matrix = create_cross_price_matrix(products, elasts)

    # pivot to wide format for easier vectorized ops
    demand_pivot = baseline.pivot(index="date", columns="product_id", values="demand").fillna(0)
    price_pivot = baseline.pivot(index="date", columns="product_id", values="price").fillna(0)

    # prepare predicted demand frame initialized as baseline
    predicted = demand_pivot.copy()

    # apply treatment: change prices for treated products and compute own-price effect
    price_test = price_pivot.copy()
    for p in products:
        if p in treated_products:
            price_test[p] = price_pivot[p] * price_multiplier

    # apply own-elasticity effect
    for p in products:
        eps = elasts.get(p, -1.0)
        # avoid division by zero
        old_price = price_pivot[p].replace(0, np.nan)
        ratio = (price_test[p] / old_price).fillna(1.0)
        # predicted demand via log-log elasticity: demand * ratio^elasticity
        predicted[p] = demand_pivot[p] * (ratio ** eps)

    # apply cannibalization from treated SKU demand changes onto others
    # for each treated SKU j, its demand change redistributes to other products
    for j in treated_products:
        delta_j = predicted[j] - demand_pivot[j]
        for i in products:
            if i == j:
                continue
            # amount of delta_j that flows to product i
            frac = cross_matrix.at[i, j] if (i in cross_matrix.index and j in cross_matrix.columns) else 0.0
            predicted[i] = predicted[i] - frac * delta_j

    # ensure non-negative
    predicted = predicted.clip(lower=0.0)

    # compute revenue series
    baseline_revenue = (demand_pivot * price_pivot).sum(axis=1)
    test_revenue = (predicted * price_test).sum(axis=1)

    # paired t-test on daily total revenue
    try:
        tstat, pvalue = ttest_rel(test_revenue.values, baseline_revenue.values)
    except Exception:
        tstat, pvalue = float('nan'), float('nan')

    summary = {
        "baseline_total_revenue": float(baseline_revenue.sum()),
        "test_total_revenue": float(test_revenue.sum()),
        "absolute_uplift": float(test_revenue.sum() - baseline_revenue.sum()),
        "percent_uplift": float((test_revenue.sum() / baseline_revenue.sum() - 1) * 100) if baseline_revenue.sum() != 0 else None,
        "daily_p_value": float(pvalue) if not pd.isna(pvalue) else None,
        "daily_t_stat": float(tstat) if not pd.isna(tstat) else None,
    }

    return {
        "baseline_daily": baseline_revenue,
        "test_daily": test_revenue,
        "summary": summary,
        "cross_matrix": cross_matrix,
        "predicted_demands": predicted,
        "price_test": price_test,
    }

#running the pipeline 
if __name__ == "__main__":
    df = load_and_prepare_data(DATA_PATH)

    model, FEATURES, test_data = train_demand_model(df)

    sample = test_data.iloc[0]
    simulation = simulate_prices(model, sample, FEATURES)

    optimal = simulation.loc[simulation["revenue"].idxmax()]
    elasticity = estimate_elasticity(simulation)

    print(f"Optimal Price: ₹{optimal['price']:.2f}")
    print(f"Estimated Elasticity: {elasticity:.2f}")

    plot_revenue_curve(simulation, optimal["price"])

    X_sample = test_data[FEATURES].iloc[[0]]
    explain_demand_predictions(model, X_sample, FEATURES)

    # --- A/B test simulator demo (multi-product cannibalization) ---
    # pick two products to treat
    products = df["product_id"].unique().tolist()
    treated = products[:2]

    # use last 60 days for the simulated experiment
    days = sorted(df["date"].unique())[-60:]

    sim_result = simulate_ab_test(df, treated_products=treated, price_multiplier=0.9, days=days)
    print("\nA/B Test Simulator Summary:")
    for k, v in sim_result["summary"].items():
        print(f"{k}: {v}")

app = FastAPI(title="Dynamic Pricing API")

@app.post("/optimize_price")
def optimize_price(payload: dict):
    row = pd.DataFrame([payload])

    simulation = simulate_prices(model, row.iloc[0], FEATURES)
    optimal = simulation.loc[simulation["revenue"].idxmax()]

    return {
        "optimal_price": float(optimal["price"]),
        "expected_demand": float(optimal["predicted_demand"]),
        "expected_revenue": float(optimal["revenue"])
    }