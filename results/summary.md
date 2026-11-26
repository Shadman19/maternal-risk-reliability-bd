# Results summary

Data file SHA256: `2b60ea74439e1e838c381ab81577feb2e01284dc4f3d78e014b7bc68fce32f16`

## Data audit

- Rows: 1014; unique rows: 452; redundant exact copies: 562.
- Feature vectors shared by two or more rows: rows involved 884; largest group 34.
- Feature vectors that carry conflicting labels: 35.
- Heart rate below 40: 2 rows; age below 15: 54; age above 50: 95.

## Leakage under a random 80/20 split

Share of test rows whose exact feature vector also appears in the training split (across seeds): 79.4% (74.5 to 84.6).

## Accuracy and macro F1 by split scheme

Mean across repeated 5-fold cross-validation, with the 2.5 to 97.5 percentile range across repeats.

| Model | Scheme | Accuracy | Macro F1 |
|---|---|---|---|
| logistic_regression | row_level | 61.8% (61.2 to 62.4) | 60.7% (60.2 to 61.3) |
| logistic_regression | grouped | 60.5% (59.9 to 61.5) | 59.4% (58.5 to 60.4) |
| logistic_regression | deduplicated_grouped | 66.8% (66.2 to 67.6) | 53.2% (52.4 to 54.2) |
| random_forest | row_level | 84.2% (83.2 to 85.1) | 84.6% (83.5 to 85.5) |
| random_forest | grouped | 66.0% (64.9 to 67.3) | 63.9% (62.6 to 65.6) |
| random_forest | deduplicated_grouped | 71.3% (70.6 to 72.3) | 63.7% (62.5 to 64.9) |
| gradient_boosting | row_level | 84.3% (83.3 to 85.4) | 84.7% (83.7 to 85.7) |
| gradient_boosting | grouped | 66.1% (64.7 to 69.1) | 64.3% (62.3 to 68.3) |
| gradient_boosting | deduplicated_grouped | 69.7% (68.6 to 71.2) | 63.4% (61.6 to 65.3) |

## Calibration under duplicate-aware cross-validation

| Model | Version | Top-label ECE | Brier | NLL |
|---|---|---|---|---|
| logistic_regression | uncalibrated | 0.064 (0.041 to 0.077) | 0.494 (0.487 to 0.501) | 0.810 (0.798 to 0.823) |
| logistic_regression | sigmoid_calibrated | 0.077 (0.062 to 0.100) | 0.497 (0.492 to 0.503) | 0.810 (0.802 to 0.818) |
| random_forest | uncalibrated | 0.124 (0.104 to 0.140) | 0.473 (0.443 to 0.491) | 0.857 (0.789 to 0.939) |
| random_forest | sigmoid_calibrated | 0.051 (0.030 to 0.077) | 0.447 (0.433 to 0.460) | 0.733 (0.715 to 0.752) |
| gradient_boosting | uncalibrated | 0.247 (0.214 to 0.260) | 0.557 (0.505 to 0.581) | 1.353 (1.189 to 1.476) |
| gradient_boosting | sigmoid_calibrated | 0.055 (0.042 to 0.068) | 0.438 (0.428 to 0.450) | 0.714 (0.700 to 0.733) |

## Screening operating point (random forest)

Threshold on P(high risk) chosen inside each training fold to reach 90% sensitivity on inner out-of-fold scores, then applied to the held-out fold.

- Held-out sensitivity: 89.3% (87.4 to 91.2)
- Held-out specificity: 82.3% (81.0 to 83.7)
- Share of all mothers flagged: 36.9% (35.4 to 38.3)

## Conformal prediction sets (random forest, target coverage 90%)

- Coverage: 90.7% (87.3 to 93.7)
- Mean set size: 1.768 (1.638 to 1.838)
- Single-label sets: 36.2% (31.9 to 44.2)

| True class | Coverage |
|---|---|
| low risk | 97.4% (96.2 to 98.5) |
| mid risk | 80.0% (71.8 to 87.2) |
| high risk | 93.8% (91.6 to 96.0) |

## Age bands (random forest, duplicate-aware CV, seed 0)

| Age band | Rows | High-risk rows | High-risk recall | 95% group bootstrap interval |
|---|---|---|---|---|
| under 20 | 279 | 39 | 61.5% | 39.3% to 81.0% |
| 20 to 34 | 417 | 79 | 82.3% | 67.7% to 93.6% |
| 35 and over | 318 | 154 | 92.2% | 85.4% to 96.9% |

Settings: repeats 10, trees 300, folds 5, runtime 626.7 s.
