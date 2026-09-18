# Limitations

Honest, specific limitations of this project — important for CO4
(appreciating practical implications and constraints) and for answering
viva questions credibly rather than overclaiming.

1. **Regional and currency scope.** The dataset consists of Indian used-car
   listings (CarDekho). Prices, depreciation patterns, and feature
   importance will not transfer to other countries' used-car markets
   without retraining on region-appropriate data.

2. **Dataset age / market drift.** The dataset is a snapshot scraped at
   some point in the past. Used-car prices move with fuel prices,
   interest rates, new-car pricing, and general economic conditions — a
   model trained on older listings will drift out of calibration over
   time without periodic retraining on fresh data.

3. **`torque` dropped entirely.** The raw `torque` column mixes units
   (Nm, kgm), RPM ranges, and inconsistent notations (e.g.
   `"190Nm@ 2000rpm"` vs. `"22.4 kgm at 1750-2750rpm"`) that could not be
   reliably parsed without risking incorrect values; it was dropped rather
   than partially/incorrectly parsed. Some price-relevant signal is likely
   lost as a result.

4. **Rare brands grouped into "Other."** Brands with fewer than 20
   listings are merged into a single `"Other"` category to avoid unstable,
   overfit one-hot columns. This means the model cannot give a
   brand-specific estimate for low-volume/exotic manufacturers — it falls
   back to a generic estimate for them.

5. **No vehicle-condition data.** The dataset has no field for accident
   history, cosmetic condition, service history, or modifications — all of
   which meaningfully affect real resale price but are simply not
   observable from this data.

6. **Dealer markup vs. private-sale price not separately modeled beyond
   the `seller_type` category.** The model treats `seller_type` as one
   categorical feature; it does not model the systematic pricing
   differences between dealers and private sellers beyond what that single
   feature captures.

7. **The "price range" shown in the app is an empirical, not a
   statistically calibrated, interval.** It is built from the 10th/90th
   percentile of the *test set's* residual distribution for the selected
   model — a reasonable, explainable approximation, but not a per-instance
   calibrated prediction interval (e.g. not conformal prediction). This is
   explicitly labeled in the UI to avoid overclaiming statistical rigor.

8. **Feature importance ≠ causation.** The feature-importance plot shows
   which features the trained model relies on most to reduce prediction
   error — it does not establish that changing a feature (e.g. `mileage`)
   *causes* a price change of a corresponding amount; confounding between
   features (e.g. age and mileage are correlated) is not disentangled.

9. **Generalization to unseen vehicle types.** The model was trained on
   passenger cars in this dataset's range (2–10 seats, standard fuel
   types). It has no data on electric vehicles beyond whatever placeholder
   handling exists for an unseen category, and will not generalize
   reliably to vehicle classes outside its training distribution
   (commercial trucks, motorcycles, etc.).

10. **Single train/test split.** While 5-fold cross-validation is used for
    model selection, the final reported test metrics come from one 80/20
    split. A different random split would give a somewhat different (though
    likely similar, given the CV standard deviations reported in
    `reports/results.md`) final number.

This project does not claim its predictions are guaranteed accurate for
any individual vehicle — it is a data-driven estimate with a measured,
reported error rate (MAPE ≈ 21%, see `reports/results.md`), presented
transparently rather than as a definitive valuation.
