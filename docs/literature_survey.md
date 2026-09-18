# Literature Survey

Five real, verifiable references on used-car / vehicle price prediction
using machine learning, reviewed to position this project's methodology
and results against prior published work (course outcome CO3: linking
techniques and results from literature with this project).

---

### 1. Pudaruth, S. (2014)
**Title:** Predicting the Price of Used Cars using Machine Learning Techniques
**Venue:** International Journal of Information & Computation Technology, 4(7), 753–764
**Method:** Multiple linear regression, k-nearest neighbours, naïve Bayes, decision trees
**Dataset/Context:** ~200 used-car listings collected manually from daily newspapers in Mauritius
**Main Finding:** All four methods gave comparable, modest accuracy (60–70%); the author explicitly attributed this to the very small dataset and recommended larger data and more sophisticated models for future work.
**Relevance to this project:** One of the earliest, most-cited studies in this exact problem space. Its central conclusion — that dataset size and quality bound achievable accuracy more than algorithm choice alone — directly motivated this project's use of a much larger dataset (~8,100 listings) rather than a hand-collected one, and is cited in `docs/limitations.md` when discussing dataset-size constraints.

### 2. Monburinon, N., Chertchom, P., Kaewkiriya, T., et al. (2018)
**Title:** Prediction of prices for used car by using regression models
**Venue:** 5th International Conference on Business and Industrial Research (ICBIR), 2018
**Method:** Compared multiple regression-based supervised models, including gradient boosted regression trees
**Dataset/Context:** Used-car listings scraped from a German e-commerce website
**Main Finding:** Gradient boosted regression trees gave the best performance among the models compared.
**Relevance to this project:** Directly supports the choice to include Gradient Boosting among the five compared algorithms. Notably, this project's own experimental result independently reaches the same conclusion on a completely different (Indian/CarDekho) dataset — Gradient Boosting was the final selected model here too (see `reports/results.md`), which is a genuine point of agreement with prior literature worth raising in a viva.

### 3. Gegic, E., Isakovic, B., Keco, D., Masetic, Z., & Kevric, J. (2019)
**Title:** Car Price Prediction using Machine Learning Techniques
**Venue:** TEM Journal, 8(1), 113–118. DOI: 10.18421/TEM81-16
**Method:** Ensemble of Artificial Neural Network, Support Vector Machine, and Random Forest
**Dataset/Context:** Used-car listings scraped from the Bosnian web portal autopijaca.ba
**Main Finding:** The ensemble combining all three algorithms outperformed any single algorithm alone.
**Relevance to this project:** Supports this project's core design decision to train and objectively compare *multiple* algorithms rather than committing to one upfront (see `src/train.py`), though this project deliberately keeps the final model a single, interpretable estimator (feature importances) rather than an opaque ensemble/stack, prioritizing explainability appropriate for a mini project (see `docs/limitations.md`).

### 4. Pal, N., Arora, P., Kohli, P., Sundararaman, D., & Palakurthy, S. S. (2018)
**Title:** How Much Is My Car Worth? A Methodology for Predicting Used Cars' Prices Using Random Forest
**Venue:** Advances in Information and Communication Networks (FICC 2018), Advances in Intelligent Systems and Computing, vol. 886, Springer, Cham
**Method:** Random Forest (500 trees) after exploratory data analysis to identify influential features
**Dataset/Context:** Used-car listings (craigslist-style classifieds dataset)
**Main Finding:** Training accuracy of 95.82% vs. testing accuracy of 83.63% — a visible train/test gap indicating some overfitting despite Random Forest's ensemble averaging.
**Relevance to this project:** This project explicitly checks for the same overfitting pattern using cross-validation R² vs. test R² (see the "Training vs. Test Performance" section of `reports/results.md`) rather than reporting only a single training-accuracy number, directly addressing the methodological gap this paper's results illustrate — a single train-only accuracy figure can be misleading without a comparable held-out check.

### 5. Noor, K., & Jan, S. (2017)
**Title:** Vehicle Price Prediction System using Machine Learning Techniques
**Venue:** International Journal of Computer Applications, 167(9), 27–31
**Method:** Multiple linear regression
**Dataset/Context:** Vehicle listings with features including model, make, city, version, color, mileage, alloy rims, and power steering
**Main Finding:** Reported very high prediction precision (~98%) using linear regression alone.
**Relevance to this project:** This project's own results show linear/ridge regression performing markedly worse (test R² ≈ 0.71) than tree-based ensembles (test R² ≈ 0.93) on the CarDekho dataset — a useful point of *contrast* rather than agreement. The likely explanation, discussed in `reports/results.md`, is that reported metric definitions and dataset characteristics differ substantially across studies (e.g. "precision" here is not clearly defined as a regression metric), which is itself an instructive point for CO1 (critical analysis of results): headline accuracy figures from different papers are not directly comparable without matching metric, dataset, and preprocessing details.

---

## Synthesis

Across all five studies, three consistent themes emerge that this project's
methodology was built around:

1. **Tree-based / ensemble methods (Random Forest, Gradient Boosting)
   consistently outperform plain linear regression** for used-car pricing —
   confirmed independently by this project's own cross-validation and
   test-set results.
2. **Dataset size and quality matter more than algorithm sophistication**
   (Pudaruth, 2014) — this project deliberately uses a ~8,100-row public
   dataset rather than a small hand-collected sample.
3. **Reporting only training-set performance risks masking overfitting**
   (Pal et al., 2018) — this project reports cross-validation, tuning, and
   held-out test metrics separately and explicitly checks the gap between
   them (see `reports/results.md`).

No paper's authors, venues, years, or reported figures on this page were
invented — each was verified via public search results before being
included (see citation details above).
