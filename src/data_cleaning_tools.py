import pandas as pd
from pathlib import Path


def import_df(file_path, drop_nulls=False, **read_csv_kwargs):
    """Load a CSV file and optionally remove unnamed columns and null rows."""
    df = pd.read_csv(file_path, **read_csv_kwargs)
    unnamed_cols = [col for col in df.columns if col.startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    if drop_nulls:
        df = df.dropna().reset_index(drop=True)
    return df


def denormalize_olist_datasets(datasets_dir="../datasets", drop_nulls=True):
    """
    Build a denormalized Olist dataframe at the order-item level for EDA.

    The final granularity is one row per purchased item so review analysis
    can be done without duplicating observations because of repeated
    payment rows or geolocation rows.
    """
    datasets_dir = Path(datasets_dir)

    # Load each source table used to reconstruct the denormalized dataset.
    customers = import_df(datasets_dir / "olist_customers_dataset.csv")
    geolocation = import_df(datasets_dir / "olist_geolocation_dataset.csv")
    order_items = import_df(datasets_dir / "olist_order_items_dataset.csv")
    payments = import_df(datasets_dir / "olist_order_payments_dataset.csv")
    reviews = import_df(datasets_dir / "olist_order_reviews_dataset.csv")
    orders = import_df(datasets_dir / "olist_orders_dataset.csv")
    products = import_df(datasets_dir / "olist_products_dataset.csv")
    sellers = import_df(datasets_dir / "olist_sellers_dataset.csv")
    category_translation = import_df(
        datasets_dir / "product_category_name_translation.csv",
        encoding="utf-8-sig",
    )

    # Keep a single review record per order to avoid one-to-many duplication.
    review_dedup = reviews.sort_values(
        by=["order_id", "review_answer_timestamp", "review_creation_date"],
        na_position="last",
    ).drop_duplicates(subset="order_id", keep="last")

    # Aggregate payment behavior to the order level before joining.
    payments_agg = payments.groupby("order_id", as_index=False).agg(
        payment_sequential_max=("payment_sequential", "max"),
        payment_installments_max=("payment_installments", "max"),
        payment_value_total=("payment_value", "sum"),
        payment_type_count=("payment_type", "nunique"),
        payment_type=(
            "payment_type",
            lambda x: ", ".join(sorted(set(x.dropna().astype(str)))),
        ),
    )

    # Collapse repeated zip-code geolocations into a single representative row.
    geolocation_agg = geolocation.groupby(
        "geolocation_zip_code_prefix", as_index=False
    ).agg(
        geolocation_lat=("geolocation_lat", "mean"),
        geolocation_lng=("geolocation_lng", "mean"),
        geolocation_city=("geolocation_city", "first"),
        geolocation_state=("geolocation_state", "first"),
    )

    # Enrich products with their English category translation.
    products_enriched = products.merge(
        category_translation,
        on="product_category_name",
        how="left",
    )

    # Split customer and seller geolocation features to preserve clear semantics.
    customer_geo = geolocation_agg.rename(
        columns={
            "geolocation_zip_code_prefix": "customer_zip_code_prefix",
            "geolocation_lat": "customer_lat",
            "geolocation_lng": "customer_lng",
            "geolocation_city": "customer_geo_city",
            "geolocation_state": "customer_geo_state",
        }
    )

    seller_geo = geolocation_agg.rename(
        columns={
            "geolocation_zip_code_prefix": "seller_zip_code_prefix",
            "geolocation_lat": "seller_lat",
            "geolocation_lng": "seller_lng",
            "geolocation_city": "seller_geo_city",
            "geolocation_state": "seller_geo_state",
        }
    )

    # Build the order-level side of the model-ready table.
    orders_enriched = orders.merge(customers, on="customer_id", how="left")
    orders_enriched = orders_enriched.merge(review_dedup, on="order_id", how="left")
    orders_enriched = orders_enriched.merge(payments_agg, on="order_id", how="left")
    orders_enriched = orders_enriched.merge(
        customer_geo, on="customer_zip_code_prefix", how="left"
    )

    # Build the item-level side with product and seller attributes.
    order_items_enriched = order_items.merge(
        products_enriched, on="product_id", how="left"
    )
    order_items_enriched = order_items_enriched.merge(
        sellers, on="seller_id", how="left"
    )
    order_items_enriched = order_items_enriched.merge(
        seller_geo, on="seller_zip_code_prefix", how="left"
    )

    # Combine item-level and order-level information into one EDA dataframe.
    denormalized_df = order_items_enriched.merge(
        orders_enriched, on="order_id", how="left"
    )

    # Convert known date columns so time-based features can be engineered later.
    date_columns = [
        "shipping_limit_date",
        "review_creation_date",
        "review_answer_timestamp",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for column in date_columns:
        if column in denormalized_df.columns:
            denormalized_df[column] = pd.to_datetime(
                denormalized_df[column], errors="coerce"
            )

    # Optionally remove incomplete rows to simplify downstream analysis.
    if drop_nulls:
        denormalized_df = denormalized_df.dropna().reset_index(drop=True)

    return denormalized_df


def numeric_column_to_binary(df, column_name, positive_value, output_column=None):
    """
    Convert a numeric column into a binary column using a threshold.

    If output_column is not provided, the original column is overwritten.
    """
    transformed_df = df.copy()
    target_column = output_column or column_name
    transformed_df[target_column] = (
        transformed_df[column_name] >= positive_value
    ).astype(int)
    return transformed_df
