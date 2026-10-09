# Model Card

## Model Details

- **Developer:** Luke Bazill
- **Date:** 10/9/2026
- **Model type:** Binary classifier, scikit-learn `RandomForestClassifier` (100 trees, `random_state=42`, `n_jobs=-1`, all other hyperparameters at their defaults). No hyperparameter tuning was done.
- **Task:** Predict whether a person's annual income is `>50K` or `<=50K`.
- **Inputs:** 8 categorical features (workclass, education, marital-status, occupation, relationship, race, sex, native-country), one-hot encoded with `OneHotEncoder(handle_unknown="ignore")`, plus the continuous features (age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week), which are passed through unscaled. The label is binarized with `LabelBinarizer`.
- **Software:** Python 3.12.9, scikit-learn 1.5.1, pandas 2.2.2.
- **Artifacts:** `model/model.pkl` and `model/encoder.pkl`.

## Intended Use

- **Intended use:** Demonstrating a deployable ML pipeline on a public dataset.
- **Intended users:** Anyone looking to see a basic, functional ML model and pipeline.
- **Out of scope:** Any current, real world analysis. This project uses data from 1994.

## Training Data

- **Source:** `data/census.csv`, the UCI Census Income ("Adult") dataset, which is extracted from the 1994 US Census database. It has 32,561 rows and 15 columns.
- **Target:** `salary`, either `>50K` or `<=50K`.
- **Split:** A random 80/20 train/test split with `random_state=42`, giving 26,048 training rows.
- **Preprocessing:** One-hot encoding of the categorical features, fitted on the training split only. Missing values appear in the data as the literal string `?` and are treated as their own category.

## Evaluation Data

The remaining 20% of the data was held out from training. The encoder and label binarizer fitted on the training split were reused to transform it. The same test set was used for the overall metrics and for the slice analysis.

## Metrics

Precision, recall and F1 (beta = 1) are computed for the positive class (`>50K`).

**Overall performance on the test set:**

| Precision | Recall | F1 |
|-----------|--------|--------|
| 0.7419 | 0.6384 | 0.6863 |

**Performance on selected slices** (the full results for every categorical value are in `slice_output.txt`):

| Slice | Count | Precision | Recall | F1 |
|-------|------:|----------:|-------:|-----:|
| sex: Female | 2,126 | 0.7229 | 0.5150 | 0.6015 |
| sex: Male | 4,387 | 0.7445 | 0.6599 | 0.6997 |
| race: White | 5,595 | 0.7404 | 0.6373 | 0.6850 |
| race: Black | 599 | 0.7273 | 0.6154 | 0.6667 |
| race: Asian-Pac-Islander | 193 | 0.7857 | 0.7097 | 0.7458 |
| marital-status: Married-civ-spouse | 2,950 | 0.7346 | 0.6900 | 0.7116 |
| marital-status: Never-married | 2,126 | 0.8302 | 0.4272 | 0.5641 |
| marital-status: Divorced | 920 | 0.7600 | 0.3689 | 0.4967 |
| education: HS-grad | 2,085 | 0.6594 | 0.4377 | 0.5261 |
| education: Bachelors | 1,053 | 0.7523 | 0.7289 | 0.7404 |
| education: Masters | 369 | 0.8271 | 0.8551 | 0.8409 |
| education: Doctorate | 77 | 0.8644 | 0.8947 | 0.8793 |
| relationship: Husband | 2,590 | 0.7370 | 0.6923 | 0.7140 |
| relationship: Own-child | 1,019 | 1.0000 | 0.1765 | 0.3000 |
| native-country: United-States | 5,870 | 0.7392 | 0.6321 | 0.6814 |
| native-country: Mexico | 114 | 1.0000 | 0.3333 | 0.5000 |

**Patterns in the slice results:**
- Performance is strongest for higher-education and typically high-paying professional careers.
- Recall is much lower for groups where high income is less common, including Own-child (0.18), Widowed (0.16), Other-service (0.19), Divorced (0.37) and Never-married (0.43). The model tends to predict `<=50K` for these groups.
- Recall for women (0.515) is about 14.5 percentage points lower than for men (0.660).

## Ethical Considerations

- The data includes protected and sensitive attributes such as race, sex, native country, and marital status, and the model uses them as inputs. Predictions may therefore reflect or reinforce historical patterns of discrimination in the data.
- The slice results show unequal performance across groups. The clearest gap is that the model finds fewer high earners among women than among men, so it would under-predict high income for women. Race slices differ less among the larger groups, but the smaller groups are measured on very few examples.

## Caveats and Recommendations

- **The data is from 1994.** Income thresholds, job markets, education levels and demographics have changed. A `>50K` threshold in 1994 dollars does not mean the same thing today, so the model should not be applied to current populations. It also includes entries that don't exist anymore, such as Yugoslavia.
- **Small slices are unreliable.** Many slices have under 30 test rows (for example Cambodia has 3 and Yugoslavia has 2), so their scores are not going to be as accurate as slices with far more test rows. Scores of 1.0000 or 0.0000 on such slices usually come from tiny counts.
- **Single split, no tuning.** Results come from one random split and default hyperparameters, so they carry some variance.
- **Overfitting and size.** Default random forests grow deep trees and tend to overfit. The pickled model is about 74 MB, which is large for deployment.
