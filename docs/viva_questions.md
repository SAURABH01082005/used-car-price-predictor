# Viva Preparation — Questions and Answers

## Fundamentals

**1. What is regression?**
A supervised learning task where the model predicts a continuous numeric
output (here, `selling_price`) from input features, as opposed to
classification, which predicts a discrete category.

**2. Why is this a regression problem, not classification?**
The target, `selling_price`, is a continuous numeric value, not a finite
set of labeled classes.

**3. What is supervised learning?**
Learning a mapping from inputs to outputs using a labeled training dataset
where the correct output (here, actual past selling prices) is known.

**4. What is the difference between supervised and unsupervised learning?**
Supervised learning uses labeled data (input-output pairs); unsupervised
learning (e.g. clustering) finds structure in unlabeled data. This
project is entirely supervised.

**5. Why use Linear Regression here at all, if it performs worse?**
As an interpretable baseline. It establishes a performance floor
(test R² ≈ 0.71) that more complex models must beat to justify their added
complexity, and its coefficients are easy to explain.

**6. Why use Ridge Regression in addition to plain Linear Regression?**
Ridge adds L2 regularization, which shrinks coefficients and reduces
sensitivity to multicollinearity (e.g. `car_age` and `year` correlate
with each other) — a useful comparison point even though, on this
dataset, it performed almost identically to plain Linear Regression.

**7. Why use Random Forest?**
An ensemble of decision trees trained on bootstrapped samples with random
feature subsets; it captures non-linear relationships and feature
interactions without manual engineering, and is robust to outliers.

**8. Why use Gradient Boosting?**
It builds trees sequentially, each correcting the errors of the previous
ones. It achieved the best test-set RMSE in this project's actual run
(see `reports/results.md`).

**9. What is overfitting?**
When a model learns the training data's noise/specifics too closely and
performs much worse on new, unseen data than on the training data itself.

**10. What is underfitting?**
When a model is too simple to capture the real pattern in the data, so it
performs poorly on both training and test data (e.g. Linear Regression's
test R² of 0.71 relative to the tree ensembles' ~0.93 suggests it is
underfitting the non-linear price relationships present here).

**11. How did you check for overfitting in this project?**
By comparing the final model's 5-fold cross-validation R² (0.8698) against
its held-out test R² (0.9281) — the test score was not markedly worse
(in fact slightly better, within CV variance), which is evidence against
overfitting. See `reports/results.md`.

## Data & Preprocessing

**12. What is data leakage, and how did you avoid it?**
Leakage is when information from outside the training data (often the test
set, or the target itself) improperly influences training. This project
avoids it by splitting into train/test BEFORE fitting any preprocessing
statistic (imputation medians, one-hot categories, scaling), so those
statistics never "see" the test set.

**13. Why split data into training and testing sets?**
To get an honest estimate of how the model performs on data it has never
seen, simulating real-world deployment.

**14. What is cross-validation, and why use it?**
Repeatedly splitting the training data into k folds, training on k-1 and
validating on the remaining fold, rotating through all folds. It gives a
more stable performance estimate (mean ± std across folds) than a single
train/validation split, and it never touches the test set.

**15. Why 5-fold specifically?**
A common, reasonable trade-off between computational cost and the
stability of the performance estimate — enough folds to average out
split-specific noise without excessive retraining cost.

**16. What is feature engineering, and what did you engineer here?**
Deriving new input features from raw data to make patterns easier for the
model to learn. Here: `car_age` (from `year`), `brand` (extracted from the
free-text `name` column), and rare-brand grouping into `"Other"`.

**17. Why compute `car_age` instead of using `year` directly?**
Buyers reason in terms of "how old is this car," and age has a more
directly interpretable, roughly monotonic relationship with depreciation
than an absolute calendar year.

**18. Why extract `brand` instead of using the full `name` column?**
`name` has thousands of unique values (specific trims/variants) — far too
high-cardinality to one-hot encode usefully. `brand` alone is a strong,
low-cardinality signal that is trivial to extract reliably.

**19. Why group rare brands into "Other"?**
Brands with very few listings produce one-hot columns that are almost
always zero, and any coefficient/split learned from so few examples is
high-variance and unreliable. Grouping stabilizes the feature space.

**20. Why is categorical encoding required?**
ML models require numeric input; `OneHotEncoder` converts categories
(e.g. `fuel = "Diesel"`) into binary indicator columns the model can use.

**21. Why OneHotEncoder specifically, not label encoding?**
Label encoding would impose a false numeric ordering on unordered
categories (e.g. implying `"Diesel" < "Petrol"`), which linear models in
particular would misinterpret as a meaningful magnitude relationship.

**22. Why `handle_unknown="ignore"` on the OneHotEncoder?**
So a category never seen during training (e.g. a brand not in the
training data) doesn't crash the app at prediction time — it's encoded as
all-zeros for that feature instead of raising an error.

**23. Why use `Pipeline` and `ColumnTransformer`?**
`ColumnTransformer` applies different preprocessing to numeric vs.
categorical columns in one object; `Pipeline` chains preprocessing and the
model together so the exact same fitted transformation is guaranteed at
both training and inference time, and cross-validation/tuning can treat
preprocessing + model as a single unit without leakage.

**24. Why median imputation for numeric features, not mean?**
The median is robust to outliers (e.g. a few extreme mileage/price values
won't skew it the way a mean would).

**25. Why drop the `torque` column instead of parsing it?**
Its free-text format mixes units (Nm vs. kgm) and RPM notations
inconsistently across rows; attempting to parse it risked introducing
silently incorrect numeric values, which is worse than not using the
feature at all.

## Model Evaluation

**26. What is MAE?**
Mean Absolute Error — the average absolute difference between predicted
and actual price, in rupees. Easy to explain directly.

**27. What is RMSE, and why can it be larger than MAE?**
Root Mean Squared Error squares each error before averaging, so large
individual errors are penalized more than in MAE. RMSE ≥ MAE always; a
larger gap between them indicates a few large-error predictions are
present (this project's actual RMSE/MAE gap is discussed in
`reports/results.md`).

**28. What is R²?**
The fraction of variance in the target explained by the model, relative
to always predicting the mean. This project's final model achieved a test
R² of 0.9281.

**29. What is MAPE, and why report it alongside RMSE?**
Mean Absolute Percentage Error expresses error as a percentage of the true
price, which is more interpretable across a wide price range than a raw
rupee figure — a ₹20,000 error matters more on a cheap car than an
expensive one.

**30. What is hyperparameter tuning?**
Searching over a model's configuration settings (e.g. tree depth, number
of estimators) — not learned from data directly — to find a combination
that improves validation performance.

**31. What is RandomizedSearchCV, and why not GridSearchCV?**
It samples a fixed number of random combinations from the hyperparameter
space rather than exhaustively trying every combination, which is far
cheaper computationally while still finding good configurations — an
appropriate trade-off given the bounded compute available for this
project.

**32. Why did tuning not always improve results in this project?**
Tuning improved cross-validation RMSE for XGBoost but gave only a marginal
change for Random Forest on the test set, and the untuned Gradient
Boosting model still beat every tuned model on the held-out test set. This
is a legitimate outcome, not a bug: with defaults already reasonable and a
training set of ~5,500 rows, tuning gains can be small and can even
reorder narrowly-separated top models on a single test split. See
`reports/results.md` for the actual numbers and discussion.

**33. How was the final model selected?**
Objectively, by lowest RMSE on the held-out test set, computed after every
candidate (baseline and tuned) was evaluated exactly once under identical
conditions — not assumed in advance.

**34. Why might tree-based models outperform linear regression here?**
Used-car pricing involves non-linear effects (e.g. steep early
depreciation that flattens with age, threshold effects around ownership
count) that a linear model cannot capture without manual interaction
terms; tree ensembles learn these non-linearities automatically.

## Interpretability & Limitations

**35. What is feature importance?**
For tree-based models, a measure of how much each feature contributed to
reducing prediction error across all trees/splits in the ensemble.

**36. Does feature importance imply causation?**
No. It reflects association strength *within the trained model*, not a
causal claim — confounded features (e.g. age and mileage correlate) can
distort which one appears "more important."

**37. What are the main limitations of this project?**
Single-region dataset (India), no vehicle-condition data, `torque`
dropped, rare brands grouped, and an empirical (not statistically
calibrated) price range — full list in `docs/limitations.md`.

**38. What happens if the used-car market changes significantly?**
The model's learned pricing patterns will drift out of date — it has no
mechanism to detect or adapt to market shifts on its own; periodic
retraining on fresh data would be required (see `docs/future_scope.md`).

## Implementation & Deployment

**39. Why is Streamlit used for the GUI?**
It turns a Python script into a web app with minimal boilerplate,
appropriate for a mini project's scope, and integrates directly with
pandas/matplotlib/Plotly without a separate frontend stack.

**40. How is the trained model loaded in the app?**
`joblib.load()` reads the serialized `sklearn.Pipeline` from
`models/used_car_price_model.pkl` once (cached via `functools.lru_cache`),
and the app calls `.predict()` on it per request — it is never retrained
at request time.

**41. What is joblib, and why use it instead of `pickle` directly?**
A serialization library optimized for objects containing large NumPy
arrays (common in fitted sklearn models); it is the standard,
sklearn-recommended tool for persisting fitted pipelines.

**42. How do you test the application?**
40 automated `pytest` tests across four files covering data validation,
preprocessing correctness, model loading/error handling, and prediction/
input-validation/reliability, run via `pytest tests/ -v`. See
`reports/test_cases.md`.

**43. What is model validation, and how does it differ from testing (software)?**
Model validation measures predictive performance (accuracy/error metrics)
on unseen data; software testing (the `pytest` suite) verifies the code's
*behavior* is correct (validation logic rejects bad input, the pipeline
loads, predictions are deterministic) — this project does both.

**44. What would you do differently with more time/resources?**
Use a larger, multi-region, more recent dataset; add SHAP-based
per-prediction explanations; implement a properly calibrated prediction
interval (e.g. conformal prediction) instead of the empirical residual
range currently used — see `docs/future_scope.md` for the full list.
