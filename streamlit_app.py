import streamlit as st
import pandas as pd
import io

from main import load_and_prepare_data, simulate_ab_test

DATA_PATH = "data/simulated_product_sales.csv"


def load_dataset(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return load_and_prepare_data(DATA_PATH)

    try:
        uploaded.seek(0)
    except Exception:
        pass

    df = pd.read_csv(uploaded, parse_dates=["date"]) if isinstance(uploaded, io.IOBase) else pd.read_csv(uploaded, parse_dates=["date"])
    # basic normalization
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def main():
    st.title("A/B Test Simulator — Multi-product Cannibalization")

    st.sidebar.header("Data & Experiment")
    uploaded = st.sidebar.file_uploader("Upload dataset CSV (optional)", type=["csv"])
    cross_upload = st.sidebar.file_uploader("Upload cross-elasticity CSV (optional)", type=["csv"] , key="cross")

    df = load_dataset(uploaded) if uploaded is not None else load_and_prepare_data(DATA_PATH)

    products = sorted(df["product_id"].unique())

    # product search/filter
    q = st.sidebar.text_input("Filter products (type to search)")
    if q:
        products_filtered = [p for p in products if q.lower() in str(p).lower()]
    else:
        products_filtered = products

    treated = st.sidebar.multiselect("Treated products", products_filtered, default=products_filtered[:2])
    price_mult = st.sidebar.slider("Price multiplier for treated (e.g. 0.9 = 10% off)", 0.5, 1.2, 0.9, 0.01)
    last_n = st.sidebar.slider("Use last N days", 7, min(365, len(df["date"].unique())), 60)

    custom_cross = None
    if cross_upload is not None:
        try:
            custom_cross = pd.read_csv(cross_upload, index_col=0)
        except Exception as e:
            st.sidebar.error(f"Could not read cross matrix CSV: {e}")

    st.sidebar.markdown("---")
    run = st.sidebar.button("Run simulation")

    if run:
        days = sorted(df["date"].unique())[-last_n:]
        res = simulate_ab_test(df, treated_products=treated, price_multiplier=price_mult, days=days, cross_matrix=custom_cross)

        st.subheader("Summary")
        st.json(res["summary"]) 

        st.subheader("Daily revenue (baseline vs test)")
        df_rev = pd.DataFrame({
            "baseline": res["baseline_daily"],
            "test": res["test_daily"]
        })
        st.line_chart(df_rev)

        st.subheader("Cross-cannibalization matrix")
        st.dataframe(res["cross_matrix"])

        st.subheader("Per-product demand (baseline vs predicted)")
        selected = st.multiselect("Select products to view", products, default=treated[:3])
        if selected:
            # baseline demand pivot
            baseline_pivot = res["predicted_demands"].copy()
            # predicted_demands in our implementation is the simulated predicted; baseline available via baseline_daily breakdown
            # we can reconstruct baseline per-product demand from res["predicted_demands"] + delta, but simpler: re-run pivot from df
            df_sub = df[df["date"].isin(days)]
            baseline = df_sub.groupby(["date","product_id"])["demand"].first().reset_index()
            baseline_p = baseline.pivot(index="date", columns="product_id", values="demand").fillna(0)

            chart_df = pd.DataFrame()
            for p in selected:
                if p in baseline_p.columns:
                    chart_df[f"{p} - baseline_demand"] = baseline_p[p]
                if p in res["predicted_demands"].columns:
                    chart_df[f"{p} - predicted_demand"] = res["predicted_demands"][p]

            st.line_chart(chart_df)

        st.subheader("Per-product revenue change")
        # compute per-product revenue sums baseline vs test
        price_test = res["price_test"]
        baseline_price = df[df["date"].isin(days)].groupby(["date","product_id"])["price"].first().reset_index()
        baseline_price_p = baseline_price.pivot(index="date", columns="product_id", values="price").fillna(0)

        per_prod_baseline_rev = (baseline_p.multiply(baseline_price_p)).sum(axis=0)
        per_prod_test_rev = (res["predicted_demands"].multiply(price_test)).sum(axis=0)
        rev_change = pd.DataFrame({
            "baseline_rev": per_prod_baseline_rev,
            "test_rev": per_prod_test_rev,
        })
        rev_change["abs_uplift"] = rev_change["test_rev"] - rev_change["baseline_rev"]
        rev_change["pct_uplift"] = (rev_change["test_rev"] / rev_change["baseline_rev"] - 1) * 100
        st.dataframe(rev_change.sort_values("abs_uplift", ascending=False))


if __name__ == "__main__":
    main()
