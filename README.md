# Predicting Late Delivery Risk Using Machine Learning

**Dataset:** DataCo Supply Chain Dataset  
**Objective:** Develop a machine learning model capable of predicting whether an order is at risk of late delivery (`Late_delivery_risk`).

---

## 1. Executive Summary

### Business Problem
Late deliveries negatively impact customer satisfaction, operational efficiency, and supply chain performance. The objective of this project is to predict whether an order is likely to experience late delivery using historical supply chain and transactional data.

### Final Outcome
Three machine learning models were evaluated:
- **Logistic Regression**
- **Random Forest**
- **XGBoost**

**Random Forest** achieved the best performance and was selected as the final model.

### Final Test Performance
| Metric | Score |
|---|---|
| **Accuracy** | 74.20% |
| **Precision** | 83.32% |
| **Recall** | 66.19% |
| **F1 Score** | 73.78% |
| **ROC-AUC** | 82.98% |
| **PR-AUC** | 87.03% |

---

## 2. Dataset Description

### Dataset Overview
The project used the **DataCo Supply Chain Dataset**.
- **Main Dataset:** 180,519 records across 53 original variables.
- Each row represents an individual order item rather than a complete order.

### Target Variable
`Late_delivery_risk`:
| Value | Description |
|---|---|
| `0` | No Late Delivery Risk |
| `1` | Late Delivery Risk |

### Class Distribution
| Class | Percentage |
|---|---|
| `1` (Risk) | 54.83% |
| `0` (No Risk) | 45.17% |

*The dataset is reasonably balanced and did not require SMOTE.*

---

## 3. Data Preprocessing

### Missing Values
- **Removed:**
  - Product Description (100% missing)
  - Order Zipcode (86% missing)
- **Imputed:**
  - Customer Zipcode (median imputation)

### Feature Engineering
- **Created:**
  - Order Year
  - Order Month
  - Order Day
  - Shipping Delay

### Data Leakage Prevention
- **Removed leakage-prone variables:**
  - Delivery Status
  - Days for shipping (real)
  - Shipping date (`DateOrders`)
  - Shipping Delay
- **Removed identifiers:**
  - Order Id, Customer Id, Product Card Id, etc.

### Encoding
- **Applied:** One-Hot Encoding
- **High-cardinality variables removed:** Order City, Order State, Customer City

---

## 4. Methodology

### Train-Test Strategy
Dataset was split using stratified sampling:
| Dataset | Percentage |
|---|---|
| **Training** | 70% |
| **Validation** | 15% |
| **Test** | 15% |

### Feature Scaling
- **Applied StandardScaler for:** Logistic Regression
- **Not required for:** Random Forest, XGBoost

### Baseline Model
- **DummyClassifier Performance:** Accuracy = `54.83%`

---

## 5. Model Development

### Candidate Models
1. **Logistic Regression**
2. **Random Forest**
3. **XGBoost**
   - *Advantages:* Gradient boosting, strong tabular-data performance, regularization support.

---

## 6. Model Comparison

### Validation Results
| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** | 69.62% | 55.94% | 66.88% | 74.42% |
| **Random Forest** | **74.32%** | **66.40%** | **73.93%** | **83.33%** |
| **XGBoost** | 70.41% | 56.86% | 67.82% | 77.22% |

### Cross Validation Results
| Model | Mean CV ROC-AUC |
|---|---|
| **Logistic Regression** | 0.7425 |
| **Random Forest** | **0.8099** |
| **XGBoost** | 0.7637 |

*Random Forest demonstrated the strongest and most stable performance.*

---

## 7. Hyperparameter Tuning

`RandomizedSearchCV` was applied to Random Forest.

**Best Parameters:**
```json
{
  "n_estimators": 300,
  "min_samples_split": 15,
  "min_samples_leaf": 1,
  "max_features": "sqrt",
  "max_depth": null,
  "class_weight": "balanced"
}
```

- **Best Tuned CV ROC-AUC:** `0.8052`
- The tuned model did not outperform the default Random Forest (`0.8099`).
- Therefore, the default Random Forest was retained.

---

## 8. Final Model Evaluation

### Test Set Results
| Metric | Score |
|---|---|
| **Accuracy** | 74.20% |
| **Precision** | 83.32% |
| **Recall** | 66.19% |
| **F1 Score** | 73.78% |
| **ROC-AUC** | 82.98% |
| **PR-AUC** | 87.03% |

### Error Analysis
**Confusion Matrix:**
| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 10,265 | 1,967 |
| **Actual 1** | 5,019 | 9,827 |

*False negatives represent late-delivery risks missed by the model and remain the primary area for improvement.*

---

## 9. Model Explainability

### Top Predictive Features
| Feature | Importance |
|---|---|
| Days for shipment (scheduled) | 0.0919 |
| Shipping Mode_Standard Class | 0.0761 |
| Latitude | 0.0503 |
| Longitude | 0.0477 |
| Order Profit Per Order | 0.0445 |

### Key Insight
Delivery risk is driven primarily by:
- Shipping characteristics
- Geographic location
- Transaction value
- Customer location

---

## 10. Robustness & Limitations

### Robustness Checks
Performed:
- Stratified Cross Validation
- Learning Curve Analysis
- Validation vs Test Evaluation

*Learning curves indicated moderate overfitting but acceptable generalization.*

### Limitations
The dataset does not include external operational variables such as:
- Weather conditions
- Traffic disruptions
- Logistics bottlenecks
- Carrier-specific delays
- Customs delays

---

## 11. Business Recommendations

1. **High-Risk Monitoring:** Flag orders predicted as high-risk for proactive monitoring.
2. **Policy Review:** Review shipping-mode policies and scheduled shipment durations.
3. **Regional Investigation:** Investigate geographical regions associated with higher delivery risk.
4. **Integration:** Deploy the model as an early-warning system within supply chain operations.

---

## 12. Conclusion

This project successfully developed a machine learning solution for predicting late-delivery risk. Among the evaluated models, **Random Forest** achieved the strongest performance with:
- **ROC-AUC:** 82.98%
- **PR-AUC:** 87.03%
- **Accuracy:** 74.20%

The model generalized well to unseen data and provides actionable business insights into the factors driving delivery delays. As future work, incorporating external logistics and environmental data may further improve predictive performance.