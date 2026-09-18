# Future Scope

Realistic, incremental improvements beyond this mini project's scope:

1. **Larger, more diverse datasets** — combining multiple sources/regions
   to improve generalization and reduce the "Other" brand-grouping loss.
2. **Real-time market data** — periodic retraining or online learning from
   live listing feeds instead of a static historical snapshot, to track
   market drift.
3. **Regional pricing models** — separate models (or a regional feature)
   per city/state, since used-car prices vary geographically in ways this
   dataset does not capture.
4. **Explainable AI (XAI)** — SHAP or LIME per-prediction explanations
   instead of only global feature importance, so a user can see exactly
   why *their specific* prediction came out the way it did.
5. **Statistically calibrated prediction intervals** — conformal
   prediction or quantile regression to replace the current empirical
   residual-quantile range with a genuine, per-instance calibrated
   interval.
6. **Advanced ensembling** — stacking/blending the top-performing models
   (Gradient Boosting, Random Forest, XGBoost) rather than selecting one
   outright, if the added complexity is justified by a real accuracy gain.
7. **Vehicle condition from images** — a computer-vision model estimating
   cosmetic condition from listing photos as an additional input feature.
8. **REST API deployment** — wrapping `src/predict.py` in a FastAPI service
   so the model can be consumed by other applications, not just the
   Streamlit UI.
9. **Cloud deployment** — containerizing and deploying the Streamlit app
   (e.g. Streamlit Community Cloud, a small cloud VM) for public access.
10. **Mobile application** — a lightweight mobile front-end calling the
    same prediction API.
11. **Personalized recommendations** — "cars similar to this one at a
    better price" suggestions using nearest-neighbor search over the
    dataset.
12. **Online/incremental learning** — updating the model incrementally as
    new listings arrive, instead of full retraining from scratch each time.
