# Reviews E-Commerce: Bad Review Risk Analysis

## Project Overview

This project analyzes Brazilian e-commerce order data to understand which operational, product, payment, and delivery-related variables are associated with low customer review scores.

The main business question is:

> Can we identify orders with higher risk of receiving a bad review and use that information to prioritize preventive actions?

The project combines:

- Data denormalization from multiple relational CSV files.
- Exploratory Data Analysis (EDA).
- Feature engineering for logistics, pricing, freight, and product dimensions.
- Binary classification models to predict review quality.
- Threshold tuning to understand the trade-off between detecting more bad reviews and creating more false alerts.
- Model interpretation using CatBoost feature importance and SHAP.

The final goal is not to claim direct causality, but to build a practical risk-detection workflow that can support business decisions.

## Dataset

The project uses the public Olist Brazilian E-Commerce dataset structure. The raw data is split across several files:

- `olist_customers_dataset.csv`
- `olist_geolocation_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `olist_orders_dataset.csv`
- `olist_products_dataset.csv`
- `olist_sellers_dataset.csv`
- `product_category_name_translation.csv`

The datasets were denormalized into a single analysis table using custom functions in:

- `src/data_cleaning_tools.py`
- `src/data_cleaning_tools.ipynb`

## Project Structure

```text
Reviews_ECommerce/
├── datasets/
├── images/
├── notebooks/
│   ├── eda_model01.ipynb
│   └── prediction_model01.ipynb
├── src/
│   ├── data_cleaning_tools.py
│   └── data_cleaning_tools.ipynb
└── README.md
```

## Data Preparation

The denormalization function joins the main Olist datasets into one modeling-ready dataframe. The joins preserve order, customer, review, payment, product, seller, and geographic information.

Important engineered features include:

- `delivery_delay`: days between actual delivery date and estimated delivery date.
- `shipping_time`: days between purchase timestamp and customer delivery date.
- `approval_delay`: days between purchase timestamp and order approval.
- `freight_ratio`: freight value divided by product price.
- `product_volume_cm3`: product length x height x width.
- `is_late_delivery`: binary indicator for late delivery.

The final modeling dataset contains `11,112` rows after selecting relevant variables and dropping rows with missing values.

## Exploratory Data Analysis

The EDA focuses on how review scores behave against:

- Price and total payment value.
- Freight value and freight ratio.
- Product weight, dimensions, and volume.
- Shipping time, approval delay, and delivery delay.
- Product categories.

Key EDA observations:

- Longer shipping times tend to be associated with lower review scores.
- Delivery delays show a negative relationship with review scores.
- High total payment value and higher freight pressure appear related to lower satisfaction.
- Large and heavy products show slightly higher bad-review risk.
- Some categories have higher bad-review rates, especially categories related to furniture, home goods, computers/accessories, baby products, and watches/gifts.

Important visual outputs are stored in the `images/` folder, including:

- `review_score_regression_plots.png`
- `delivery_delay_review_score.png`
- `shipping_time_review_score.png`
- `approval_delay_review_score.png`
- `correlation heatmap_binary_review_score.png`
- `XGBoost_ConfusionMatrix.png`

## Prediction Modeling

The first modeling version used:

- `bad = 0` when `review_score <= 2`
- `good = 1` when `review_score >= 3`

This allowed the model to classify whether a review was generally good or bad. The target distribution was:

```text
good    8,478
bad     2,634
```

Models tested:

- Logistic Regression
- CatBoostClassifier
- XGBoostClassifier

### Initial Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.693 | 0.831 | 0.750 | 0.788 | 0.670 |
| CatBoostClassifier | 0.813 | 0.818 | 0.971 | 0.888 | 0.732 |
| XGBoostClassifier | 0.830 | 0.837 | 0.965 | 0.896 | 0.755 |

The best XGBoost configuration was:

```text
n_estimators = 200
learning_rate = 0.10
max_depth = 6
```

XGBoost had the best overall metrics, but the business problem is not only predicting good reviews. The more important question is whether the model can detect bad reviews early enough to support preventive action.

## Business-Focused Bad Review Model

To align the model with the business goal, the target was reframed:

```text
target_bad_review = 1 if review_score <= 2
target_bad_review = 0 if review_score >= 3
```

This makes the positive class the event we want to prevent: a bad review.

CatBoost was then retrained using this bad-review target and class weights to help the model pay more attention to the minority class.

### CatBoost Bad-Review Target Results

With the default threshold of `0.50`:

| Metric | Value |
|---|---:|
| Accuracy | 0.784 |
| Bad Review Precision | 0.543 |
| Bad Review Recall | 0.565 |
| Bad Review F1 | 0.554 |
| Bad Review PR-AUC | 0.594 |
| ROC-AUC | 0.762 |

Interpretation:

- The model detects about `56.5%` of actual bad reviews.
- When it predicts a bad review, it is correct about `54.3%` of the time.
- This is a more balanced operating point.

## Threshold Tuning as a Trade-Off Experiment

The model outputs probabilities, not just labels. A threshold converts those probabilities into a decision.

Example:

```text
If bad_review_probability >= threshold -> predict bad review
If bad_review_probability < threshold  -> predict not bad review
```

The default threshold is usually `0.50`. In this project, the `0.50` threshold is the most balanced operating point because it keeps precision and recall at a reasonable level.

I also tested a lower threshold as an experiment. The goal was not to present it as the best final model, but to show the trade-off between detecting more bad reviews and generating more false alerts.

The aggressive threshold selected by F2-score was:

```text
threshold = 0.24
```

With this aggressive threshold experiment:

| Metric | Value |
|---|---:|
| Accuracy | 0.369 |
| Bad Review Precision | 0.267 |
| Bad Review Recall | 0.951 |
| Bad Review F1 | 0.417 |
| Bad Review PR-AUC | 0.594 |
| ROC-AUC | 0.762 |

Interpretation:

- This aggressive threshold detects about `95.1%` of bad reviews.
- Precision drops to about `26.7%`, meaning many normal orders are also flagged as risky.
- Accuracy also drops because the model becomes overly sensitive and predicts too many orders as high risk.
- This is useful as a learning experiment, not necessarily as the final business configuration.
- If interventions are expensive or limited, the default threshold of `0.50` is more practical and easier to defend.

The main takeaway is:

> Lowering the threshold increases bad-review recall, but it also increases false alerts. The model does not magically become better; we simply choose a more aggressive risk-detection strategy.

## Validation

### Cross-Validation

Using 5-fold stratified cross-validation with the aggressive threshold experiment:

| Metric | Mean | Std |
|---|---:|---:|
| Accuracy | 0.312 | 0.008 |
| Bad Review Precision | 0.251 | 0.002 |
| Bad Review Recall | 0.958 | 0.011 |
| Bad Review F1 | 0.398 | 0.004 |
| Bad Review PR-AUC | 0.579 | 0.021 |
| ROC-AUC | 0.752 | 0.011 |

The high-recall strategy is stable across random splits, but it creates many false positives. This confirms that the aggressive threshold is useful for understanding the trade-off, not necessarily for final deployment.

### Temporal Validation

A temporal split was also tested:

- Train period: `2017-05-04` to `2018-07-31`
- Test period: `2018-07-31` to `2018-08-29`

Temporal validation results:

| Metric | Value |
|---|---:|
| Accuracy | 0.418 |
| Bad Review Precision | 0.247 |
| Bad Review Recall | 0.842 |
| Bad Review F1 | 0.382 |
| Bad Review PR-AUC | 0.445 |
| ROC-AUC | 0.680 |

Temporal validation is more realistic because it simulates training on past orders and predicting future orders. The drop in ROC-AUC suggests the model is useful, but not strong enough to be treated as a fully reliable automated decision system.

## Model Interpretation

CatBoost feature importance and SHAP were used to understand which variables influenced bad-review predictions the most.

Top model drivers:

| Feature | Importance |
|---|---:|
| `payment_value_total` | 15.125 |
| `delivery_delay` | 12.307 |
| `price` | 9.560 |
| `shipping_time` | 9.077 |
| `freight_value` | 7.733 |
| `product_category_name_english` | 6.993 |
| `approval_delay` | 5.671 |
| `product_height_cm` | 5.251 |
| `freight_ratio` | 5.050 |
| `product_width_cm` | 4.744 |

These results suggest that bad reviews are strongly associated with:

- Delivery performance.
- Total order value.
- Freight cost and freight pressure.
- Product category.
- Product size and logistics complexity.

## Business Recommendations

Based on the EDA and model interpretation, the business could prioritize:

- Monitoring orders with long shipping times.
- Flagging orders at risk of late delivery.
- Giving proactive updates for orders with high delivery-delay risk.
- Reviewing high-risk categories such as furniture, home goods, computers/accessories, baby products, and watches/gifts.
- Paying attention to large, heavy, or expensive products where customer expectations may be higher.
- Using the model as a risk-ranking system instead of an automatic decision system.

The most practical use case is:

> Score incoming orders, rank them by bad-review probability, and prioritize the highest-risk orders for proactive support or communication.

For a realistic business implementation, the threshold should be chosen based on operational capacity. If the team can only handle a limited number of alerts, a balanced threshold such as `0.50` is more appropriate. If the intervention is cheap and automated, a lower threshold can be tested.

## Limitations

This project identifies patterns and predictive signals, but it does not prove causality.

Important limitations:

- Some variables, such as actual delivery delay, are only known after delivery.
- Dropping missing values reduces the dataset and may introduce sample bias.
- Review score `3` is treated as not bad, but in some businesses it could be considered neutral or risky.
- A model can identify risk, but it cannot prove that a specific action will prevent bad reviews.
- To prove prevention, the next step would be a real experiment or A/B test.

## Final Conclusion

This project shows that customer review outcomes can be partially predicted using order, delivery, payment, product, and category information.

The strongest signals are related to delivery performance, order value, freight cost, and product characteristics. The model is not perfect, but it can support a practical business workflow by identifying orders with a higher probability of receiving a bad review.

The threshold tuning section should be interpreted as a sensitivity experiment. It shows how the model behaves when the priority shifts from balanced performance to catching as many bad reviews as possible.

The best portfolio framing is:

> This model is a decision-support tool for prioritizing customer experience interventions, not a final automated decision system.

## Tools and Libraries

- Python
- pandas
- NumPy
- matplotlib
- seaborn
- scikit-learn
- CatBoost
- XGBoost
- SHAP
- Jupyter Notebook
