# How this project works, explained simply

This is my walkthrough of every file and every important decision, so I can explain the project from memory. All numbers come from `results/metrics.json` and `results/tuning.json`.

---

## 1. The big picture

1. **Load** the NSL-KDD records. Each record summarises one network connection with 41 features, such as protocol, service, bytes sent and error rates.
2. **Prepare** them: turn text into numbers and put all numbers on the same scale.
3. **Train** three detectors on `KDDTrain+`: a neural network, an SVM and a K-Means anomaly detector.
4. **Test** them on `KDDTest+`, which includes attack types the models have never seen.
5. **Compare** them on false positives and false negatives, overall and per attack category.
6. **Tune** their settings using the training data only.
7. **Detect:** replay records one at a time through the best model and print JSON alerts.

---

## 2. File by file

### `src/ids/data.py`: loading the data

| Name | What it does |
|---|---|
| `FEATURE_COLUMNS` | The 41 feature names in file order. The raw files have no header row, so I supply the names. |
| `CATEGORICAL_COLUMNS` / `NUMERIC_COLUMNS` | Splits the 41 features into the 3 text columns and the 38 numeric ones, because they need different preparation. |
| `RAW_COLUMNS` | The 41 features plus `attack_name` and `difficulty`, so 43 columns in total. |
| `ATTACK_CATEGORIES` | A dictionary mapping each attack name to DoS, Probe, R2L or U2R. It includes names that only appear in the test set (e.g. `mscan`, `apache2`). |
| `attack_category(name)` | Looks up one name. `normal` stays `normal`. An unknown name returns `unknown` instead of crashing. |
| `load_nsl_kdd(path)` | Reads a file and adds `attack_category` and the binary target `is_attack` (0 normal, 1 attack). It gives a clear error if the file is missing. |
| `split_features_target(df)` | Returns X (the 41 features) and y (`is_attack`). It deliberately leaves out the label, category and difficulty columns, because giving those to a model would leak the answer. |

**The four attack categories:**
- **DoS:** flooding a service so legitimate users can't reach it.
- **Probe:** scanning the network for hosts and open ports.
- **R2L (remote to local):** getting into a machine from outside, for example by guessing passwords.
- **U2R (user to root):** a logged-in user escalating to administrator, for example through a buffer overflow.

### `src/ids/features.py`: preparing the data

| Name | What it does |
|---|---|
| `build_preprocessor()` | Builds a `ColumnTransformer`. It one-hot encodes the 3 text columns and standard-scales the 38 numeric ones. `handle_unknown="ignore"` means a service never seen in training produces zeros instead of an error. `remainder="drop"` means any other column (such as a label) can never reach the model by accident. |
| `output_feature_names(pre)` | Lists the 122 column names the model actually sees, e.g. `categorical__service_http`. |

### `src/ids/models.py`: the three detectors

| Name | What it does |
|---|---|
| `RANDOM_STATE = 42` | One seed used everywhere, so results are reproducible. |
| `SVM_SAMPLE_SIZE = 20_000` | The number of training rows used for the SVM. |
| `KMeansDetector` | My anomaly detector (explained below). It follows scikit-learn conventions, so it works inside a `Pipeline` and `GridSearchCV`. |
| `KMeansDetector.fit(X, y)` | Removes attack rows, clusters the normal rows, and sets `threshold_` to the chosen percentile of their distances to the nearest centre. |
| `KMeansDetector.anomaly_score(X)` | Distance from each record to its nearest cluster centre. Bigger means more unusual. |
| `KMeansDetector.predict(X)` | Returns 1 (anomaly) if the score is above the threshold, otherwise 0. |
| `build_mlp_pipeline()` | Preprocessor plus `MLPClassifier` (layers of 64 and 32 neurons, early stopping). |
| `build_svm_pipeline()` | Preprocessor plus `SVC` with an RBF kernel. |
| `build_kmeans_pipeline()` | Preprocessor plus `KMeansDetector`. |
| `build_all_models()` | All three untrained pipelines in a dictionary. |
| `stratified_sample(X, y, n)` | Picks n rows at random while keeping the same normal/attack ratio. |

### `src/ids/evaluate.py`: measuring and charting

| Name | What it does |
|---|---|
| `confusion_counts()` | Counts TN, FP, FN and TP. |
| `compute_metrics()` | Calculates accuracy, precision, recall, F1, FPR and FNR from those four counts. Each formula is written out by hand so it is easy to explain. Dividing by zero returns 0 instead of crashing. |
| `summarise_folds()` | Mean and standard deviation of each metric across cross-validation folds. |
| `pick_best_model()` | Returns the name of the model with the highest F1. |
| `detection_rate_by_category()` | Recall calculated separately for DoS, Probe, R2L and U2R. |
| `detection_rate_seen_vs_unseen()` | Recall for attack types that were in training compared with types that were not. |
| `plot_confusion_matrix()`, `plot_error_rates()`, `plot_detection_by_category()`, `plot_before_after()`, `plot_kmeans_sweep()` | Save the charts in `results/`. Each model keeps the same colour in every chart. |

### `train.py`: train and compare

| Function | What it does |
|---|---|
| `training_rows_for()` | Uses the 20k sample for the SVM and all rows for the other models. |
| `fit_and_evaluate()` | Trains one model, times it, predicts on the test set and computes the metrics. |
| `cross_validate()` | Stratified 5-fold cross-validation on the training set (with `--cv`). |
| `save_charts()` | Writes the five Stage 4 charts. |
| `print_results_table()` | Prints the metrics table. |
| `main()` | Runs everything, saves the best model by F1 to `models/best_model.joblib`, and writes `results/metrics.json`. |

### `tune.py`: tuning

| Function | What it does |
|---|---|
| `PARAM_GRIDS` | The settings tried for each model (8 for the MLP, 9 for the SVM, 16 for K-Means). |
| `grid_search()` | Runs `GridSearchCV` with 3-fold stratified cross-validation on training data, using every CPU core. |
| `best_params_by_f1()` | Picks the settings with the highest mean cross-validation F1. |
| `cv_table()`, `kmeans_f1_grid()` | Record every setting tried, for the JSON report and the heatmap. |
| `main()` | Refits the chosen settings on the same rows `train.py` uses, tests once, saves before and after, and saves the best tuned model. |

### `detect.py`: alerts

| Name | What it does |
|---|---|
| `read_records()` | Reads the input file **one line at a time** and yields each record. Lines with or without labels both work. |
| `build_alert()` | Builds the JSON alert: timestamp, record number, verdict, model, protocol, service, flag, `src_bytes`, `dst_bytes`. |
| `Summary` | Keeps running totals. If the file has labels, it also counts true alerts, false alarms and missed attacks. |
| `run()` | Loads the saved model, scores each record as it arrives, prints alerts, and can pause between records (`--delay`) to imitate a live feed. |

### `tests/`

There are 56 pytest tests:
- **Loading and the category mapping** (`test_data.py`).
- **Preprocessing** (`test_features.py`): shape and scaling.
- **KMeansDetector and the pipelines** (`test_models.py`).
- **Metrics** (`test_evaluate.py`): calculations worked out by hand and checked against scikit-learn.
- **detect.py** (`test_detect.py`).

`conftest.py` makes a small synthetic dataset for fast tests. It is never used for results.

---

## 3. Why I made each design decision

**Why does K-Means train on normal traffic only?**
The detector's idea is "learn what normal looks like, flag anything far from it". If attacks were in the training data, some cluster centres would sit on top of attack traffic, and attacks would look normal. Training on normal traffic only also means the detector never needs examples of an attack to flag it. That is why the baseline K-Means caught 37.2% of R2L and 62.7% of U2R attacks, more than either supervised model. Strictly, using labels to pick out the normal rows makes it *novelty detection* rather than fully unsupervised learning.

**What does the percentile threshold control?**
It sets how far from normal a record must be before it is flagged.
- **Mechanism:** at the 95th percentile, the 5% of normal training records furthest from a centre sit above the threshold. So about 5% false positives is expected on traffic like the training data. Cross-validation measured 4.97%.
- **Trade-off:** raising the percentile cuts false alarms but misses more attacks, and lowering it does the opposite. In the sweep at 32 clusters, going from the 90th to the 97.5th percentile changed the cross-validated FPR from 0.0998 to 0.0254, while recall fell from 0.9604 to 0.9175.
- **In a SOC:** it is the dial that sets alert volume.

**Why one-hot encoding?**
Models need numbers. Replacing `tcp`, `udp` and `icmp` with 1, 2 and 3 would wrongly suggest an order and a distance between them. One-hot encoding gives each value its own 0/1 column, so no false order is introduced.

**Why scaling?**
Features have very different ranges: `src_bytes` can be in the millions, while rates run from 0 to 1. The SVM's RBF kernel and K-Means both rely on distances, so unscaled byte counts would dominate. Standard scaling gives every numeric feature mean 0 and standard deviation 1. The scaler is fitted on training data only, so no information from the test set leaks into training.

**Why does the SVM use a sample?**
SVM training time grows roughly with the square of the number of rows or worse, so 126k rows would be slow on a laptop. A 20,000-row stratified sample keeps the normal/attack balance and trained in a few seconds. Even so, the SVM had the fewest false positives.

**Why use F1 to pick the best model?**
- **Why not accuracy:** accuracy can look good while a model is useless. A detector that never alerts still scores 43% accuracy on KDDTest+, because 43% of it is normal traffic.
- **What F1 balances:** precision ("when it alerts, is it right?") and recall ("does it catch the attacks?").
- **How it is calculated:** as a harmonic mean, so a model can't score well by being strong on only one of the two.

**Why a Pipeline?**
The preprocessor and model are fitted and saved together. `detect.py` therefore prepares new records in exactly the same way as training did, which avoids a classic deployment bug.

**Why fixed seeds?**
So anyone, including me, can re-run the code and get the same numbers. I checked this: re-running `train.py` reproduced the metrics exactly.

---

## 4. Ten likely interview questions, with answers

**1. Give me a two-minute summary of the project.**
I compared a neural network, an SVM and a K-Means anomaly detector for detecting malicious traffic on the NSL-KDD benchmark, focusing on false positives and false negatives. I evaluated on a test set containing attack types not seen in training. After tuning, the neural network had the best F1 (0.7808) and the SVM the fewest false positives (2.65%). The K-Means detector was best on the rare R2L and U2R attacks. Finally, `detect.py` replays records and prints JSON alerts. It's a rebuild of my MSc project using public data and AI coding assistance.

**2. Which model would you deploy in a SOC, and why?**
It depends on which error the team can afford. If analysts are overloaded, I'd choose the SVM: its false positive rate was 3.39%, the lowest of the baseline models, but it missed 37% of attacks. If missing attacks is the bigger risk, I'd choose the neural network, which missed 32%. In practice I'd consider combining a supervised model with the anomaly detector, because the K-Means detector caught R2L and U2R attacks the classifiers missed.

**3. Why are your cross-validation scores around 0.99 but test scores around 0.78?**
Two reasons. First, KDDTest+ has 17 attack types that are not in training (3,750 records); detection of those was only 29% to 47%. Second, the class mix shifts: R2L is 0.79% of training but 12.8% of the test set. Cross-validation only measures known attacks, while the test set measures how well a model generalises.

**4. What is the difference between a false positive and a false negative, and which is worse?**
A false positive is an alert on harmless traffic. It wastes analyst time and causes alert fatigue. A false negative is a missed attack, which means an undetected breach. A single false negative is usually worse, but false positives arrive in volume: the tuned neural network still raised 709 false alarms on about 9,700 normal records. A SOC has to balance the two, which is why I report both.

**5. How does your K-Means detector work?**
It clusters normal training traffic into groups. For each normal record it measures the distance to the nearest cluster centre, and it sets a threshold at a percentile of those distances. A new record further away than the threshold is flagged. Tuning chose 32 clusters and the 97.5th percentile, which raised its test F1 from 0.5329 to 0.7775.

**6. Why did tuning make the neural network and SVM worse?**
The settings were chosen by cross-validation on the training data, where both models were already near 0.99 F1. The winning settings fitted known attacks slightly better but generalised slightly worse to new ones. The SVM's F1 fell from 0.7602 to 0.7190, although its false positive rate improved. The lesson is that you can't tune your way around a shift between training and real-world data.

**7. Why didn't you tune on the test set?**
The test set would then no longer be an independent measure, and the reported results would be optimistic. All settings were chosen by cross-validation on KDDTrain+, and the test set was used only to report the results. One honest caveat: the final choice of model is made by test F1, as the brief specified. I list that as a limitation.

**8. What are the limitations of NSL-KDD?**
- **Age:** it comes from 1998 DARPA simulation data and doesn't reflect modern traffic, encryption or current attacks.
- **Known criticisms:** these were raised by McHugh (2000).
- **Scope:** it summarises connections rather than raw packets.

It is still useful for comparing methods on a common benchmark, but these results don't predict performance on a real network.

**9. How would you make this work on real traffic?**
- **Data:** run Zeek or Suricata in an isolated lab and read their connection logs.
- **Features:** map their fields to equivalent features and retrain on that lab's normal traffic.
- **Speed:** score records in batches. Scoring one record at a time ran at about 41 records per second, far too slow for a real network.
- **Alerts:** send them to a SIEM with a severity score.

**10. Did you use AI to build this? How do I know you understand it?**
Yes. I rebuilt my lost MSc project with AI coding assistance, and the README says so. I wrote the requirements, chose the design (for example, training K-Means on normal traffic only and selecting by F1), reviewed every file, and checked every number against real runs. I can explain any function, and I can explain why tuning hurt the supervised models, which only comes from understanding the results.
