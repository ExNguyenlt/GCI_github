# Claude's Work History in the GCI_github Repository

This document records Claude Code's work sessions (cloud session, branch `claude/cool-noether-usxp6b`), reconstructed from the conversation transcript, the git history and the experiment logs. All times are UTC (Vietnam time = UTC + 7).

# PART I — Overview by Work Session

## Overview

| Session | Time (UTC) | Main content | Result |
|---|---|---|---|
| 1 | 08/11 12:11–13:12 | Repository survey; mechanism for tracking `main` | `CLAUDE.md` + SessionStart hook, merged into `main` |
| 2 | 08/11 13:15–16:50 | Initial modelling, Optuna + stacking | AUC 0.76273 |
| 3 | 09/11 08:49–11:26 | Set up the 3-agent loop, round 1 | AUC 0.76790 |
| 4 | 10/11–12/11 | Round 2 (interrupted by a pause and by usage limits) | AUC 0.77081 |
| 5 | 12/11 18:13–13/11 08:33 | Automatic resume, round 3 (final round) | AUC 0.77149, loop stopped |
| 6 | 13/11 19:39 | Writing this document | `LICH_SU_LAM_VIEC.md` |
| 7 | 16/11–20/11 | Code archive + reproducible submission notebook | `code/`, `code/nop_bai/` (stacking AUC 0.77095) |

All AUC values in this document are out-of-fold (OOF) AUC from 5-fold cross-validation (StratifiedKFold, shuffle, random_state = 42) on the 171,202 training rows; they are not the organizer's leaderboard score.

---

## Session 1 — Repository survey and tracking of the `main` branch (08/11)

**What was done**
- Listed the files in the repository; explained that `github.txt` was on `main`, so Claude could only see it after merging.
- Installed a mechanism so that every time Claude works, it knows what has changed on `main`.
- Renamed commits to `000` as requested, then force-pushed (with the user's permission) so that the branch held a single `000` commit on top of `origin/main`.

**How**
- `CLAUDE.md`: mandatory rules — `git fetch origin main` → report new commit/file names → `git merge origin/main` before editing → on a conflict, stop and ask; clear notebook outputs; never push directly to `main`.
- `.claude/settings.json` + `.claude/hooks/check-main.sh`: a SessionStart hook that fetches and reports changes on `main` (report only, never merges, always exits with code 0).

**Result**
- The user merged this mechanism into `main`. An initial push error (403) was fixed after the user reconnected GitHub.

---

## Session 2 — Initial modelling (08/11)

**Request:** create `com/submission.csv` for the Home Credit task (predicting loan default, metric ROC-AUC), target ≥ 0.80, using legitimate methods only.

**What was done and how**
1. Loaded the data: `train_part1.csv` + `train_part2.csv` (171,202 rows, ~8% defaults), `test.csv` (61,500 rows), 33 application columns; the column names contain a BOM, so files are read with `encoding="utf-8-sig"`.
2. LightGBM baseline → added ~40 features (credit/income ratios, EXT_SOURCE combinations, …) → combined LightGBM/XGBoost/CatBoost by rank averaging.
3. Optuna (TPE) hyperparameter tuning and logistic-regression stacking (as requested: "run Optuna and stacking").
4. Read the 5 PDF documents in `com/advanced techniques/` (answer only, no changes); explained the meaning of AUC and why the data sets the upper bound of the score.

**Results**

| Step | AUC |
|---|---|
| LightGBM baseline | 0.7615 |
| + 40 features | 0.7615 |
| 3-model blend | 0.7625 |
| Optuna + stacking | 0.76273 |

- Leakage checks: AUC of the customer ID alone ≈ 0.503 (carries no information); train/test drift check ≈ 0.512 (the two sets are homogeneous).

---

## Session 3 — 3-agent loop, round 1 (09/11)

**Loop design (as specified by the user)**
- Agent 1: research NEW optimization methods. Agent 2: draw lessons from the previous round. These two agents run in parallel.
- Agent 3: implement based on the assessments of Agents 1 and 2, run, and compare results.
- Stop conditions (exactly one of three): (1) AUC ≥ 0.80; (2) Agent 1 has run out of new ideas AND every lesson from Agent 2 has been carried out, after running one final round; (3) the user orders a stop. No stopping because of saturation or a round limit.
- Lessons are tracked in a "LESSON CHECKLIST" table (codes L<round>.<number>, status NOT DONE / DONE / NOT FEASIBLE).

**Round 1 — techniques kept**
- Regression-based imputation of EXT_SOURCE.
- Implied interest rate reconstructed from the annuity equation (AMT_CREDIT, AMT_ANNUITY).
- Reconstructed loan term.

**Rejected:** date features, KNN neighbour target mean, MLP network, adding round-0 models to the ensemble.

**Result:** AUC **0.76790** (LightGBM 0.76696, XGBoost 0.76643, CatBoost 0.76545). `com/submission.csv` updated, committed as `000`, pushed.

---

## Session 4 — Round 2 (10/11 – 12/11)

**Timeline:** the user ordered "Pause everything" (09/11), then resumed (10/11). Agents stopped several times because of the 5-hour usage limit and were resumed from where they stopped, without restarting from scratch.

**What was done**
- Agent 1: "loan-product population profile" — for each product (key = reconstructed term × rate), compute the average characteristics of all applicants with the same product over train + test, without using labels.
- Agent 2: target encoding of the product, computed nested inside each fold; identified the customer groups where the model is still weak (over 60, pensioners, missing 2 EXT scores).
- Agent 3: implemented both approaches, carried out the 10 outstanding checklist items, ran all 5 folds for the 3 models, then ensembled.

**Results**
- Kept: product profile (+0.0028); product code as a categorical feature for CatBoost (+0.0040 on CatBoost); target encoding in LightGBM (+0.0003).
- Rejected (≈ 0): separate models for weak groups, pseudo-labelling, EBM, KNN at the meta level, …
- AUC **0.77081** (+0.0029, better in 5/5 folds). `com/submission.csv` updated, committed as `000`, pushed (12/11 18:32).

---

## Session 5 — Automatic resume and round 3 (12/11 18:13 – 13/11 08:33)

**Automatic resume after the usage limit resets**
- Created a scheduled task "Auto-continue the 3-agent loop". Initially it checked every hour; it was later changed to a 5-hour cycle at the user's request.
- The first time, it missed the reset at 21:30 because the start of the cycle was mis-estimated; it was re-aligned to 03:40 / 08:40 / 13:40 / 18:40 / 23:40 UTC.
- Disabled at the user's request (13/11 08:31).

**Round 3**
- Agent 1: tried 9 new directions (OpenFE, soft-label self-distillation, pricing residual, monotone constraints, …); none passed the threshold → **NO_NEW_METHODS**.
- Agent 2: confirmed there was no label leakage; OOF slightly optimistic by about +0.00025 because of early stopping; proposed 6 final items.
- Agent 3 (final round) was stopped by the model's weekly usage limit after all models had finished; the main session completed the ensembling step, the submission and the log itself.

| Item | Result |
|---|---|
| Product profile × 5 credit-amount levels | Stable gain → kept |
| CatBoost 5 seeds | Small gain → kept |
| TabM neural network (torch CPU) | Gain when ensembled → kept |
| LightGBM DART / extra trees / linear tree | Diversify the ensemble → kept |
| Dropping XGBoost | No change → XGBoost not used |
| Null Importance feature selection | Lower score → rejected |
| Penalizing high rates in term reconstruction | Unstable → rejected |

**Result**
- Final ensemble: nested logistic-regression stacking over LightGBM, CatBoost, binned logistic regression, DART, extra trees, linear tree and TabM.
- AUC **0.77149** (+0.00068 vs round 2, better in 5/5 folds); honest estimate without early stopping ≈ 0.7713.
- `com/submission.csv` updated, committed as `000`, pushed (13/11 08:33).
- **The loop stopped under condition 2.**

---

---

# PART II — DETAILS OF EACH AGENT IN EACH ROUND (full version, not summarized)

This part reproduces **in full** the report of each agent (files `loop/round<round>_agent<1|2>.md`) and the sections written by Agent 3 in the shared log `loop/history.md`, translated into English; only heading levels were lowered to fit the layout. The symbol `$S` in the reports is the session's scratch directory:
`/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad` (not in the repository). Codes `L<round>.<number>` are lesson codes in Agent 2's checklist; `R<number>` are rules for combining techniques; `A1 #<number>` is proposal number <number> of Agent 1 in round 1.

## Loop rules (`loop/rules.md`)
#### Procedure for each round
1. Agent 1 (research NEW methods, not yet in history.md) and Agent 2 (draw lessons from the previous round) run IN PARALLEL.
2. Agent 3 implements based on the assessments of 1 and 2, runs, and compares results.
3. If no stop condition is met → go back to step 1.

#### STOP conditions — exactly one of three
1. OOF AUC ≥ 0.80.
2. Agent 1 reports `STATUS: NO_NEW_METHODS` AND, at the same time, every lesson/direction proposed by Agent 2 has been carried out by Agent 3
   (→ run one final round with that knowledge, then stop).
3. The user orders a stop.

There is NO stop condition for saturation or for a maximum number of rounds.

#### Lesson tracker (required to check condition 2)
- Agent 2: at the end of each round's report there must be a `LESSON CHECKLIST` table — each row: code (L<round>.<number>), the actionable direction/lesson, status `NOT DONE` / `DONE (round X, result)` / `NOT FEASIBLE (reason)`. Carry over the rows not yet done from previous rounds.
- Agent 3: in its ROUND section of history.md it must state clearly which lesson codes were carried out and with what result.

## Common context and Round 0 (before the loop — carried out by the main session, no agents)
Shared working directory (scratchpad): `/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad`
(abbreviated `$S` below). This log lives at `$S/loop/history.md`.

### The problem
- Data: `/home/user/GCI_github/com/` — `train_part1.csv` + `train_part2.csv` (combined = 171,202 rows), `test.csv` (61,500 rows),
  `sample_submission.csv` (columns `SK_ID_CURR,TARGET`), `HomeCredit_columns_description.xlsx`.
- 33 columns (ONLY the application table, NO bureau/previous_application/installments...). Share of TARGET=1 ≈ 8.0%.
- Note: the CSV files have a BOM — strip `﻿` from column names. pandas 3.x: text columns have dtype `str`; use `select_dtypes(exclude='number')`.
- Metric: ROC-AUC. User's target: AUC ≥ 0.80.
- Reference: the course's sample leaderboard rank 1 = 0.748. The original Home Credit (Kaggle) top ≈ 0.80 but uses 7 tables; the application table alone ≈ 0.76–0.77.
- Machine: 4 CPUs, no GPU. Installed: lightgbm, xgboost, catboost, scikit-learn, optuna 5, scipy, pandas 3.

### Mandatory constraints
- LEGITIMATE methods only: no label leakage, no leaderboard probing, no use of test labels.
  (Checked: AUC by SK_ID_CURR = 0.503 → the ID does not leak.)
- Techniques that use the UNLABELLED test data (pseudo-labels, adversarial validation, statistics pooled over train+test) are legitimate as long as test labels are not used.
- Do not create files in the repository `/home/user/GCI_github` except `com/submission.csv`. All scripts/intermediate results go in `$S`.
- Evaluate with 5-fold StratifiedKFold(shuffle=True, random_state=42) so that OOF predictions are comparable with existing files.
  Stacking/blending must be evaluated with nested CV (the meta-model is scored on unseen data).

### Files available in `$S`
- `fe.py` — function `load()` returns `(df, y, n, cats, ids)`: the current feature set (~80 columns). `run.py <lgb|cat|xgb> $S`, `opt.py <model> $S <n_trials>`.
- OOF/test preds (same folds rs=42): `oof_{lgb,cat,xgb,lgb_opt,cat_opt,xgb_opt}.npy`, `pt_*.npy`; `y.npy`; `ids.npy` (test IDs in sample_submission order).
- Logs: `log_cx.txt` (CatBoost/XGB), `log_opt.txt` (Optuna).

### ROUND 0 (done before the loop) — results
| # | Change | OOF AUC | Notes |
|---|---|---|---|
| 0.1 | LightGBM baseline + ~20 FE (EXT mean/std/prod/min/max/weighted, credit/annuity, annuity/income, credit/income, ratios /DAYS_BIRTH, group mean by ORG/OCC), 3 seeds × 5 folds | 0.7615 | `hc.py` |
| 0.2 | ~40 more FE (EXT×age, pairwise EXT products, frequency encoding, diff from mean by ORG/OCC/INCOME_TYPE/EDU, income/median(ORG,OCC), TERM_ROUND, REQ_SUM) + LGB lr 0.01 leaves16 ff0.3 | 0.76152 | NO gain over 0.1 |
| 0.3 | CatBoost (depth6 lr0.03, native categorical) | 0.7608 | |
| 0.4 | XGBoost (depth5, colsample0.3, native categorical) | 0.7610 | |
| 0.5 | Rank-average of 3 models | 0.76246 | +0.001 |
| 0.6 | Grid search of blend weights | 0.76258 | only +0.0001 → not used |
| 0.7 | Optuna (LGB 30, XGB 20, Cat 12 trials; 3 folds, high lr then retrain with low lr) | LGB 0.76153 / XGB 0.76118 / Cat 0.76063 | ≈ 0; the best parameters are all SHALLOW trees (depth 3–4) + strong regularization |
| 0.8 | Rank-avg of 6 models (old + opt) | 0.76266 | |
| 0.9 | LogisticRegression stacking on the logits of 6 OOF models (nested CV) | **0.76273** | CURRENTLY SUBMITTED VERSION. XGB weight ≈ 0 |

Methods ALREADY used (Agent 1 must not propose them again as "new"):
GBDT (LGB/XGB/Cat), basic ratio/interaction features, frequency encoding, group-mean differences, rank averaging, weighted blending,
Optuna TPE, LogisticRegression stacking, multi-seed averaging, handling of DAYS_EMPLOYED=365243.
From the course materials (already known): log transform, label/count/one-hot encoding, OWN_CAR_AGE>60 outlier, missing-value filling, class_weight balanced, bagging/boosting/stacking theory.

Main observation of round 0: 3 different algorithms converge to ~0.761 → suspected information ceiling of the data ~0.76–0.77.

---

---

## ROUND 1

### Agent 1 — Research of new methods (`loop/round1_agent1.md`)
STATUS: NEW_METHODS_FOUND

#### ROUND 1 — Agent 1 (Research): proposed new methods

Context read: `loop/history.md` (best OOF 0.76273, 3 GBDT algorithms converging at ~0.761) and `fe.py` (~80 columns).
None of the methods below is in the "already used" list. All are legitimate: no test labels are used; anything that uses train labels
(KNN target, target encoding) MUST be computed nested inside the folds (nested OOF), as described in each item.

Network note: kaggle.com, medium.com and speakerdeck.com are blocked in this session, so some sources could only be read through search-engine
snippets; the source code of the 2nd-place solution (KazukiOnodera, GitHub) could be read directly.

##### Quick checks on the data (no model training, scripts in `$S/loop/a1/explore*.py`)
- C/A (AMT_CREDIT/AMT_ANNUITY) of Cash loans is concentrated in DISCRETE CLUSTERS (e.g. 34.1: 5,757 rows; 23.5: 4,841; 19.5: 4,467; 20.7; 12.6; 30.8...),
  i.e. "products" with a fixed (term, interest rate). Revolving loans: C/A = 20.000 for 99.8% of rows → C/A is meaningless for revolving loans.
- Rate reconstruction (choose the term n with the highest rate density over train+test) → within each C/A bin, the univariate AUC of the estimated rate
  reaches 0.646 (bin 4/20) and 0.56 (bin 0) → there is NEW information that C/A (TERM_ROUND) does not fully capture.
- Age at passport issue (DAYS_ID_PUBLISH − DAYS_BIRTH) clusters at 20.0–20.5 years (16.5k) and 45.0–45.5 years (39.7k), exactly as Russian passport rules require
  (renewal at 20 and 45). The "passport renewal delay" has AUC 0.53–0.546 WITHIN each age group.
- Repeat customers: a time-invariant key (DAYS_BIRTH−DAYS_ID_PUBLISH, DAYS_BIRTH−DAYS_REGISTRATION, DAYS_BIRTH−DAYS_EMPLOYED, gender)
  puts 2.2% of rows in groups of size >1 (e.g. 2 test rows offset by exactly 113 days in both DAYS_BIRTH and DAYS_EMPLOYED → the same person). Small coverage.

##### Ranking (score ≈ expected gain × probability of success / cost)

| Rank | Method | Expected OOF gain | CPU cost | Main risk |
|---|---|---|---|---|
| 1 | Loan term & interest-rate reconstruction + "product code" | +0.002 → +0.008 | a few seconds | wrong term assignment (only noise; trees filter it) |
| 2 | KNN target mean over 500 neighbours (a strong feature of the winning team) — nested OOF | +0.001 → +0.004 | 5–15 min | leakage if not nested in folds |
| 3 | Model-based imputation of EXT_SOURCE_1/2/3 (2nd-place solution) + residuals | +0.001 → +0.003 | 5–10 min | low (no TARGET used) |
| 4 | Domain date/age features (age at milestones, passport renewal delay at 20/45, age at loan payoff) | +0.0005 → +0.002 | ~0 | low |
| 5 | Modern tabular neural networks RealMLP / TabM (TabArena 2025) to diversify stacking | +0.001 → +0.003 (at ensemble level) | 30–90 min | installing torch ~3 GB; a single NN is weaker than GBDT |
| 6 | OOF target encoding for categorical combinations (ORG×OCC, product×contract type…) | +0.0003 → +0.0015 | minutes | leakage if not nested in folds |
| 7 | Feature selection with Null Importance (Olivier's Home Credit kernel) | 0 → +0.001 | 20–40 min | overfitting the selection to CV |
| 8 | Repeat-client matching through invariant date keys | +0.0002 → +0.001 | minutes | wrong matches; 2% coverage |
| 9 | Boosting variants for diversity: LightGBM DART / extra_trees / linear_tree | 0 → +0.001 | 10–30 min | DART is slow, no early stopping |
| 10 | (Not recommended now) TabICL v2 / TabPFN-2.5 | ? | very high on CPU | infeasible for 171k rows without GPU |

---

##### 1. Loan term & interest-rate reconstruction
- **Source**: Home Credit 2nd-place solution (team ikiri_DS, slides "Home Credit Default Risk 2nd place solutions",
  https://speakerdeck.com/hoxomaxwell/home-credit-default-risk-2nd-place-solutions — slides 9, 13): the interest rate inferred from (principal, annuity, term)
  has "extremely high" importance; the team suspected it is a META-FEATURE from Home Credit's internal scoring model (rate = risk price);
  the application table lacks the term, so they built an "Interest Rate Estimator" (R² 0.998). Source code: https://github.com/KazukiOnodera/Home-Credit-Default-Risk (py/000.py has
  `AMT_CREDIT-d-AMT_ANNUITY`, `AMT_GOODS_PRICE-d-AMT_ANNUITY`, `age_finish_payment`).
- **Mechanism**: Home Credit prices by risk → the approved rate contains the bank's own risk assessment. C/A mixes term and rate
  (C/A=34.1 could be 36 months @3.6%/year, 48 months @18% or 60 months @25%). We do not have CNT_PAYMENT, but because products use a fixed grid of
  (term, rate), the correct pair is the one whose implied rate falls on a "density peak" across all of train+test. Quick check:
  the AUC of the estimated rate within the same C/A bin reaches 0.646 → genuinely new information.
- **Implementation** (uses only AMT_*, pooled train+test, no TARGET):
  ```python
  TERMS=[6,10,12,18,24,30,36,48,60]   # actual HC terms (CNT_PAYMENT in previous_application); also try 42,54
  def rate(C,A,n):   # solve A = C*r/(1-(1+r)^-n) by bisection, r∈[0,0.2]/month; NaN if A*n<C
  for n in TERMS: r_n = rate(C,A,n)*12
      key = np.round(r_n*1000)   # bin of 0.1%/year (try 0.25%)
      dens_n = key.map(key.value_counts())   # density over train+test (cash loans)
  term_hat = argmax_n dens_n ; rate_hat = r_{term_hat}; dens_max; dens_ratio = top1/top2 (confidence)
  overpay = A*term_hat - C ; overpay_rel = overpay/C ; age_end = -DAYS_BIRTH/365 + term_hat/12
  PRODUCT = round(C/A, 2) (+ round(C/GOODS,3)) → frequency encoding + use as a categorical for CatBoost
  for revolving loans: set term_hat/rate_hat = NaN (C/A=20 fixed)
  ```
  Advanced version (like the 2nd-place "estimator"): choose n by maximizing the posterior = prior(n) × density(r_n), with prior(n) taken from the frequency
  of standard terms; or smooth the density with a 1-D KDE. Keep the raw r_12, r_24, r_36, r_48, r_60 as well (let the GBDT choose).
- **Expected**: +0.002…+0.008 (the highest in the list; in the original competition the rate was one of the most important features of the previous-application group).
- **Cost**: a few seconds (vectorized numpy). **Risk**: wrong term for rare clusters → only noise; no leakage.

##### 2. KNN target mean (500 neighbours) — a feature of the winning team
- **Source**: comment by the winning team on Kaggle (https://www.kaggle.com/c/home-credit-default-risk/discussion/64480, linking to the
  1st-place write-up discussion/64821): "one interesting feature we used that was very strong was the mean targets of the 500 nearest neighbors, where
  neighbors were defined by the three external credit ratings and the credit amount/annuity ratio". Repeated in the case study
  https://medium.com/thecyphy/home-credit-default-risk-part-2-84b58c1ab9d5 (`TARGET_NEIGHBORS_500_MEAN`, highest importance).
- **Mechanism**: a non-parametric, smooth estimate of default probability in the 4 strongest dimensions (EXT1..3, C/A). GBDT approximates this surface with
  axis-parallel steps; KNN gives a smooth surface with full interactions → complementary, especially for data with few columns as here.
- **Implementation** (sklearn available):
  ```python
  Z = [EXT1, EXT2, EXT3, C/A]  -> QuantileTransformer(normal) fitted on train+test; NaN: fill with the predicted EXT from item 3
       (or median + an extra flag column); also try the variant Z2=[EXT_MEAN, EXT2, C/A, DAYS_BIRTH]
  for outer fold k (5-fold rs=42):
      Tk = train rows outside fold k
      # feature for rows in Tk: inner 5-fold on Tk (KDTree fitted on inner-train, queried on inner-valid)
      # feature for fold k (valid) and for test: KDTree fitted on all of Tk
      feat = mean(y[neighbors]) with K∈{100, 500, 2000}; add an inverse-distance-weighted mean
  # submission: final training on all train using 5-fold OOF; test queried against all train
  ```
  Use `sklearn.neighbors.KDTree` (4 dimensions → very fast), `n_jobs=4`.
- **Expected**: +0.001…+0.004. **Cost**: 5–15 min (30 KDTree fits/queries on ~137k points).
- **Risk**: LEAKAGE if a single-level OOF is computed and then used in CV (the train-set feature contains labels of the valid fold) → nesting as above is mandatory.
  There may be a slight distribution shift between train (inner 80%) and valid (100% of Tk) — acceptable, as with standard target encoding.

##### 3. Model-based imputation of EXT_SOURCE (2nd-place solution) + residuals
- **Source**: 2nd-place code `py/011_EXT_1.py … 016_EXT_3.py` (https://github.com/KazukiOnodera/Home-Credit-Default-Risk/tree/master/py):
  LGB regression (lr 0.01, num_leaves 63, max_depth 6, min_data_in_leaf 150, colsample 0.9) learns EXT_SOURCE_k from the other columns on the train+test rows
  that HAVE a value, and predicts the missing rows; TARGET is not used.
- **Mechanism**: EXT_SOURCE_1 is ~56% missing and EXT_SOURCE_3 ~20%, and they are the 3 strongest features; the aggregates EXT_MEAN/PROD/W are currently computed on the
  non-missing part → noisy. Plausible imputed values make EXT_MEAN/PROD/KNN (item 2) more accurate. The residual (EXT_k − prediction, computed OOF on rows with a value)
  measures how "abnormal the credit score is relative to the profile" — a new signal.
- **Implementation**: for k=1,2,3: X = all original columns + FE (dropping EXT_k and columns derived from EXT_k), data = train+test rows with EXT_k; 5-fold KFold →
  `EXT_k_PRED` (OOF for rows with a value, fold average for missing rows), `EXT_k_FILL = EXT_k.fillna(PRED)`, `EXT_k_RES = EXT_k − PRED`.
  Recompute EXT_MEAN_F / EXT_PROD_F / EXT_W_F on the FILL version; KEEP the missing flags.
- **Expected**: +0.001…+0.003. **Cost**: 3 models × 5 folds of LGB regression ~5–10 min. **Risk**: low; no labels used.

##### 4. Domain date/age features
- **Source**: 2nd place (py/000.py: differences `DAYS_X-m-DAYS_BIRTH`, `age_finish_payment`); Russian passport rules (issued at 14, renewed at 20 and 45) —
  confirmed on the data: peaks at 20.0–20.5 and 45.0–45.5 years.
- **Mechanism**: the existing /DAYS_BIRTH ratios cannot express "age at an event" or "delay relative to a legal milestone". People who renew their passport late
  (lack of administrative discipline) default more: AUC within age group 0.53–0.546.
- **Implementation**:
  ```python
  age = -DAYS_BIRTH/365.25
  for c in [DAYS_ID_PUBLISH, DAYS_REGISTRATION, DAYS_EMPLOYED, DAYS_LAST_PHONE_CHANGE]: df[c+'_AGE_AT'] = (c - DAYS_BIRTH)/365.25
  ida = ID_PUBLISH_AGE_AT; df['ID_LATE'] = where(ida>=45, ida-45, where(ida>=20, ida-20, ida-14))
  df['ID_ON_TIME'] = ID_LATE < 0.1 ; df['ID_OVERDUE'] = (age>=45)&(ida<45)  | (age>=20)&(ida<20)
  df['YEARS_TO_45'] = 45-age ; df['AGE_END_LOAN'] = age + CREDIT_ANNUITY/12 (or term_hat/12 from item 1)
  df['PHONE_CHANGED_TODAY'] = DAYS_LAST_PHONE_CHANGE==0 ; df['GOODS_ANNUITY'] = AMT_GOODS_PRICE/AMT_ANNUITY
  df['EMP_START_AGE'] = (DAYS_EMPLOYED-DAYS_BIRTH)/365.25 ; df['ID_MINUS_EMP'] = DAYS_ID_PUBLISH-DAYS_EMPLOYED
  ```
- **Expected**: +0.0005…+0.002. **Cost**: ~0. **Risk**: low.

##### 5. Modern tabular neural networks (RealMLP, TabM) to diversify stacking
- **Source**: TabArena (Erickson et al., NeurIPS 2025, https://arxiv.org/abs/2506.16791): with tuning + ensembling, RealMLP (Elo ~1569) and TabM (~1552)
  match or beat LightGBM (~1532) and CatBoost (~1489). RealMLP: Holzmüller et al. 2024 "Better by default" (package `pytabkit`);
  TabM: Gorishniy et al., ICLR 2025 (package `tabm`). The Home Credit winning team also used NNs (M. Jahrer's DAE+NN) as level-1 models.
- **Mechanism**: NN errors are less correlated with GBDT (smooth surface, periodic/PLR numeric embeddings) → the current LR stacking (XGB weight ≈0 because it is too similar
  to LGB) gains a genuinely diverse component. A single NN is expected at ~0.752–0.760, but the blend improves.
- **Implementation**: `pip install torch` (PyPI torch 2.15.0, ~3 GB including CUDA libraries — download.pytorch.org/whl/cpu is blocked by the proxy; 26 GB of disk left),
  `pip install pytabkit==1.8.0` (download verified), or `tabm==0.0.3`.
  ```python
  from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier
  m = RealMLP_TD_Classifier(device='cpu', n_threads=4, random_state=0, val_metric_name='1-auc_ovr', n_epochs=256)
  # preprocessing: original columns + important FE; let pytabkit one-hot/embed the cats; NaN: add a flag + median fill
  # keep the same folds rs=42; in each fold hold out 10% as an internal validation set for early stopping
  # TabM_D_Classifier(device='cpu', tabm_k=16) if time allows
  ```
  Then feed the OOF logits into the nested LR stacking as in round 0.9.
- **Expected**: +0.001…+0.003 at the ensemble level. **Cost**: RealMLP ~3–8 min/fold, TabM ~10–25 min/fold on 4 cores.
  **Risk**: heavy installation; NNs are sensitive to preprocessing (need QuantileTransformer/robust scaling, built into RealMLP-TD).

##### 6. OOF target encoding for categorical combinations
- **Source**: 2nd place `py/003_TargetEncoding.py`, `004_TargetEncoding.py` (OOF mean target over 5 folds); Micci-Barreca (2001) smoothing.
- **Mechanism**: ORGANIZATION_TYPE (58 levels) × OCCUPATION_TYPE (19) × NAME_INCOME_TYPE create many-level combinations that GBDT splits poorly; the existing frequency encoding /
  group-mean differences contain no label information. Add "PRODUCT" (item 1) × NAME_CONTRACT_TYPE.
- **Implementation**: smoothing m=50–200; NESTED: in each outer fold, the TE of train rows is computed with an inner 5-fold, and the TE of valid/test on all of Tk.
  Or cheaper: CatBoost with string combination columns + `max_ctr_complexity=3` (CatBoost does ordered TS itself, no leakage).
- **Expected**: +0.0003…+0.0015. **Cost**: minutes. **Risk**: leakage if not nested; very rare combinations → noise (use min_count≥30).

##### 7. Feature selection with Null Importance
- **Source**: the kernel "Feature Selection with Null Importances" (Olivier Grellier) written for this very Home Credit competition
  (https://www.kaggle.com/code/ogrellier/feature-selection-with-null-importances).
- **Mechanism**: round 0.2 added 40 features without an OOF gain → many noisy/duplicate columns; with shallow trees (depth 3–4, per Optuna) noisy columns steal splits.
  Compare the real importance with the distribution of importance under shuffled labels (50–80 times) → drop columns whose gain ≤ the 75th percentile of the null.
- **Implementation**: fast LGB (lr 0.1, 200 trees, feature_fraction 0.8) × 60 TARGET shuffles; score = log(1e-10 + real_imp/(1+P75(null_imp)));
  try thresholds {0,10,20,30} and choose with 5-fold rs=42. Do this after adding items 1–4.
- **Expected**: 0…+0.001. **Cost**: 20–40 min. **Risk**: choosing the threshold by CV → slight overfitting; keep thresholds coarse.

##### 8. Repeat-client matching
- **Source**: inferred from the data structure (days are relative to the application date → the difference between two DAYS columns is invariant per person); the
  "find the hidden user" technique common in Kaggle competitions (IEEE-CIS Fraud 2019 — uid from D-columns). Check: 2.2% of rows have ≥1 record with the same key.
- **Implementation**: uid = (DAYS_BIRTH−DAYS_ID_PUBLISH, DAYS_BIRTH−DAYS_REGISTRATION, DAYS_BIRTH−DAYS_EMPLOYED, CODE_GENDER); features:
  UID_CNT (train+test), time gap to the previous/next application (difference of DAYS_BIRTH), change of income/credit vs other applications, and (optional) OOF
  mean TARGET of the OTHER applications of the same uid in train (nested as in item 2; for test use only train labels).
- **Expected**: +0.0002…+0.001 (small coverage). **Risk**: wrong matches (DAYS_EMPLOYED=365243 → excluded from the key), leakage if the TE is not nested.

##### 9. Boosting variants for diversity
- LightGBM `boosting='dart'` (drop_rate 0.1, skip_drop 0.5, ~1500–3000 rounds, lr 0.02, no early stopping — choose the number of rounds by CV),
  `extra_trees=True`, `linear_tree=True` (+ `linear_lambda`); `path_smooth`. Use as new level-1 models for stacking.
- Source: LightGBM documentation (https://lightgbm.readthedocs.io/en/latest/Parameters.html); DART (Rashmi & Gilad-Bachrach, AISTATS 2015);
  many Home Credit kernels use DART.
- **Expected**: 0…+0.001 (mainly through diversity). **Cost**: DART is 3–5× slower (~10–30 min).

##### 10. (Not recommended in this round) Tabular foundation models TabICL v2 / TabPFN-2.5
- TabPFN-2.5 (Prior Labs, arXiv 2511.08667): designed for ≤50k rows (tested up to 100k) and needs a GPU (160k rows ≈ an 80 GB GPU); the weights require accepting
  a licence on HuggingFace. `tabpfn` 9.2.0 and `tabicl` 2.2.0 are on PyPI, but the ~O(n²) attention cost on 137k rows × 5 folds × several ensemble
  members on a 4-core CPU means many hours. Only worth trying as a side model: a sampled context of 10–20k rows, as one extra level-1 model — low expectations.

##### Not recommended
- Pseudo-labelling on test: with AUC and 8% positives the gain is usually ≈0, plus the risk of self-confirming errors.
- Adversarial validation: diagnostic use only (AUC by ID = 0.503 → most likely a random split, no benefit).

##### Suggested order of trials for Agent 2
1+4 (cheap, same LGB run) → 3 → 2 (using the imputed EXT) → 6 → rerun LGB/Cat with the new set → 5 and 9 for more diversity → nested LR stacking → 7 if time remains.
Compare each step against 3-seed LGB (OOF 0.7615); treat gains < 0.0005 as noise (seed-to-seed variation).

##### Honest assessment of the AUC ceiling
- Original competition: 1st place reached ~0.805 on the private LB, but used all 7 tables (bureau, previous, installments… contributed most of the gain from ~0.76 to ~0.80).
  Popular notebooks that use only application_train (122 columns, 307k rows) usually reach about 0.755–0.77.
- Here there are only 33/122 columns (FLAG_DOCUMENT_*, housing information, social circle, REGION_RATING_W_CITY, application hour… are missing) and 171k rows (≈56% of the original rows),
  so the realistic ceiling is lower. That 3 different GBDT algorithms all stop at ~0.761 and stacking only reaches 0.7627 confirms this.
- If items 1–5 add up well, a reasonable expectation is OOF **~0.765–0.770**, optimistically ~0.772. **0.80 is almost certainly not achievable
  with legitimate methods** using only the 33-column application table. Only item 1 (the interest rate) has the potential for a jump >0.005, because it recovers part of
  Home Credit's internal risk assessment.

##### Sources
- Kaggle Home Credit discussion 64480 (winning team's comment, KNN-500): https://www.kaggle.com/c/home-credit-default-risk/discussion/64480 ; 1st-place write-up: /discussion/64821
- 2nd-place slides (ikiri_DS): https://speakerdeck.com/hoxomaxwell/home-credit-default-risk-2nd-place-solutions
- 2nd-place code: https://github.com/KazukiOnodera/Home-Credit-Default-Risk (py/000.py, 003_TargetEncoding.py, 011–016_EXT_*.py)
- KNN feature case study: https://medium.com/thecyphy/home-credit-default-risk-part-2-84b58c1ab9d5
- Top-17 solution (#1 solo): https://github.com/NoxMoon/home-credit-default-risk
- TabArena: https://arxiv.org/abs/2506.16791 ; TabPFN-2.5: https://arxiv.org/abs/2511.08667
- Null importance kernel: https://www.kaggle.com/code/ogrellier/feature-selection-with-null-importances
- pip packages checked: pytabkit 1.8.0, tabm 0.0.3, tabicl 2.2.0, tabpfn 9.2.0, gplearn 0.4.3, torch 2.15.0 (PyPI; the pytorch CPU index is blocked).

### Agent 2 — Lessons from Round 0 (`loop/round1_agent2.md`)
#### ROUND 1 — Agent 2 (Lessons learned)

Sources: `history.md`, `fe.py`, `run.py`, `opt.py`, `log_cx.txt`, `log_opt.txt`, the `oof_*/pt_*.npy` files.
My light-weight analysis: scripts in `$S/a2/` (`a1.py` … `a6.py`). No file in the repository was modified.

Conventions for the numbers below:
- "OOF" = AUC over the 5 folds rs=42 (comparable with the round 0 table).
- "Quick test F0/F1" = LightGBM lr 0.05, shallow trees (leaves 14, depth 4, ff 0.2, l1 10), early stopping 200,
  run on fold 0 and fold 1 × 2 seeds = 4 paired evaluations. Absolute scores are lower than the real OOF (high lr), but
  the PAIRED DIFFERENCES are usable.

---------------------------------------------------------------------------------------------------
##### 1. Failures & causes

| Experiment | Result | Evidence | Cause |
|---|---|---|---|
| 0.2: adding ~40 FE (EXT×age, pairwise EXT products, freq enc, group diffs, TERM_ROUND, REQ_SUM) | 0.7615 → 0.76152 (≈ 0) | Quick test on fold 0 (mean of 3 seeds): **33 original columns + 8 core FE = 0.75749** vs **all 93 columns = 0.75768** → the 85 extra columns contribute only ~+0.0002, less than the seed noise (0.0012). 23/93 columns have gain < 0.3%, 2 columns have gain = 0. | These FE are re-transformations of the SAME information (EXT_SOURCE, AMT_*, DAYS_*) that the GBDT learns by itself through splits. They bring no NEW information. |
| 0.7: Optuna (LGB 30, XGB 20, Cat 12 trials) | LGB 0.76152→0.76153, XGB 0.76100→0.76118, Cat 0.76080→0.76063 | Paired per-fold difference lgb_opt − lgb: [+0.0002, −0.0004, +0.0004, −0.0009, +0.0006], mean −0.00003, std 0.00054 → **zero**. Cat_opt loses to cat in 5/5 folds (mean −0.0009 vs lgb). The score range across trials (0.7583–0.7616) ≈ seed noise. | (a) Flat parameter landscape: every shallow-tree + regularized configuration gives ~0.761. (b) The optimized objective (3 folds, lr 0.05) ≠ the final configuration (5 folds, lr 0.01) → the "best" parameters do not transfer. (c) Winner's curse: the best trial is selected on those same 3 folds, so its score is optimistic. (d) The data is information-limited, not the model. |
| 0.6 / 0.9: weighted blend, LR stacking | 0.76246 → 0.76258 / 0.76273 (+0.0001–0.0003) | Bootstrap SE of the overall AUC = 0.0024; SE of the paired difference rank6 − lgb ≈ 0.0002 → +0.0001 is within noise. Stacking gives XGB a weight ≈ 0. | OOF Spearman correlation: LGB–XGB **0.995**, LGB–lgb_opt 0.992, XGB–xgb_opt 0.993 → the models are almost identical; nothing is left to optimize in the weights. |
| Adding "models" by re-tuning the same algorithm (lgb_opt, xgb_opt, cat_opt) | rank 3 → rank 6: 0.76246 → 0.76266 (+0.0002) | OOF correlation between the original and the opt version of the same algorithm 0.988–0.993 | Same features + same GBDT family → no diversity. |
| Feature selection by gain (top-40) | fold 0: 0.75449 vs 0.75713–0.75835 (full) | drop of ~0.003, larger than seed noise | Gain on one fold is unstable; dropping low-gain columns loses interaction information. → Do NOT prune features by gain. |

##### 2. Tips from the successes

1. **Core FE around EXT_SOURCE and the loan is the biggest lever seen so far**: fold 0, 33 original columns = 0.74737 → adding 8 core FE
   (CREDIT_ANNUITY, EXT_MEAN, EXT_PROD, EXT_W, EXT_NA, ANNUITY_INCOME, CREDIT_GOODS, DAYS_EMPLOYED_R_BIRTH) = 0.75749 (**+0.010**).
   Current top gain: ORGANIZATION_TYPE_EXT_MEAN_DIFF 8%, EXT_W 7.2%, EXT_MIN 6.6%, OCCUPATION_TYPE 3.4%.
   → Lesson: gains come from features carrying **new information / information hard for trees to learn** (EXT combinations compensating for missing values, credit ratios), not from the number of columns.
2. **Rank-averaging DIFFERENT algorithms is a real gain**: per-fold rank6 − lgb = [+0.0015, +0.0011, +0.0011, +0.0008, +0.0012],
   mean +0.00114, std 0.00023 → positive in 5/5 folds, t ≈ 10. The main contribution comes from CatBoost (correlation with LGB only 0.973, the lowest).
3. **Shallow trees + strong regularization** (depth 3–4, large l1, feature_fraction 0.1–0.3) is the common optimum of all 3 algorithms → use it as the default, no need to re-tune.
4. **Simple rank-avg ≈ stacking**: when model correlation > 0.97, equal weights suffice; stacking only makes sense with a genuinely different model (corr < 0.95).

##### 3. Surprising findings (light analysis, round 1)

1. **Model-based imputation of EXT_SOURCE (no TARGET used) gives a clear gain**: train an LGB regressor that predicts EXT_SOURCE_k from the other columns
   on train+test (3-fold OOF for rows with a value). Prediction–actual correlation: EXT1 **0.695**, EXT2 0.496, EXT3 0.369.
   Adding `EXT_k_PRED`, `EXT_k_FILL` (actual value, prediction when missing), `EXT_k_RES` (actual − prediction), `EXTF_MEAN/PROD/W`:
   quick test F0/F1 mean 0.75919 → **0.76144 (+0.0022)**, paired [+0.0024, +0.0031, +0.0028, +0.0007] → positive 4/4, ~2× the seed noise.
   Reason: EXT1 is 69.5% missing (higher than the original Kaggle set, ~56%), EXT3 32% missing; the group with missing EXT is where the model is weakest (see item 5).
2. **The loan's implied interest rate (the price the bank sets by risk) gives a clear gain**: with k = AMT_CREDIT/AMT_ANNUITY, solve
   (1 − (1+r)^−n)/r = k for terms n ∈ {12, 24, 36, 48, 60} → `RATE_n` (annual rate; NaN if k ≥ n).
   Quick test F0/F1: 0.75919 → **0.76136 (+0.0022)**, paired [+0.0029, +0.0018, +0.0022, +0.0019] → positive 4/4, very stable.
   Trees cannot learn this non-linear function from CREDIT_ANNUITY by themselves (the old TERM_ROUND only rounds k and carries no rate information).
   Note: Revolving loans have k = 20 fixed (annuity = 5% of the limit) → RATE is meaningless for those 9.5% of rows, so leave it NaN.
   **Combining 1 and 2**: quick test F0/F1 = **0.76299 (+0.0038)**, paired [+0.0045, +0.0038, +0.0038, +0.0032] → almost ADDITIVE (2 independent sources of information).
   Gain in the combined model: EXTF_W 15.4%, EXTF_MEAN 11.2%, EXTF_PROD 7.0%, EXT_SOURCE_3_PRED 3.5%, RATE_24/RATE_48 1.7% each
   → the IMPUTED EXT combinations take over the top positions from the old EXT combinations.
   (Reusable code: `$S/a2/a5.py` — the imputation part and the `rate()` function; `$S/a2/a6.py` — the combined run.)
3. **Train and test share the same distribution**: adversarial AUC = **0.512** (close to random); missing rates of columns in train/test almost identical (EXT1 0.695/0.698, EXT3 0.319/0.320).
   No duplicate rows within train, none between train and test. → CV is trustworthy, no adversarial weighting needed; LB score ≈ OOF ± sampling noise.
4. **Performance by group (rank6 OOF, overall AUC 0.76266)**:
   - Number of missing EXT: 0 → 0.7735 (21%), 1 → 0.7668 (56%), **2 → 0.7399 (23%, highest default rate 9.1%)**.
   - EXT3 missing: 0.7430 vs present: 0.7711.
   - DAYS_EMPLOYED = 365243 (pensioner/unemployed): **0.7337** (18% of rows). Age > 60: **0.7214**; < 30: 0.7354.
   - Contract: Cash 0.7618, Revolving 0.7573. No difference by gender (0.758/0.758).
   → The weakness lies with people missing EXT_SOURCE and with pensioners/older people — exactly where EXT imputation helps.
5. **Concrete measurement noise** (used to set thresholds):
   - Std of AUC across the 5 folds of one model: 0.0022–0.0034 (fold 2 is always the highest ~0.766, fold 3 the lowest ~0.757 for every model → differences due to the fold, not the model).
   - Seed noise on one fold (lr 0.05): 0.75713 / 0.75835 / 0.75755 → about 0.0012.
   - Bootstrap SE of the overall OOF AUC: 0.0024; but the SE of the PAIRED DIFFERENCE between 2 models on the same OOF is only ~0.0002–0.0005.
6. **Small optimistic bias in OOF**: `run.py`/`opt.py` use the validation fold itself for early stopping → OOF slightly optimistic
   (a few 1e-4), and Optuna selects the trial on those same 3 folds and then rescores on 5 folds that contain those 3 folds. It is not test-label leakage, but it makes
   differences of +0.0001 even more meaningless.
7. The data is a 33/122-column subset of the original Kaggle set (missing FLAG_DOCUMENT_*, housing indicators, SOCIAL_CIRCLE, HOUR_APPR_PROCESS, REGION_RATING_W_CITY…),
   with SK_ID_CURR renumbered 0..232701. The 0.80 target is almost certainly **not achievable** with this data alone; the realistic ceiling is estimated at ~0.765–0.770.

##### 4. Combining techniques & logic optimization — rules for Agent 3

**Evaluation**
- R1. Always compare **paired** (same fold, same seed) with the baseline, report per-fold differences, not only the overall OOF.
- R2. Significance threshold: a change is kept when the OOF difference is ≥ **+0.0005** AND positive in **≥ 4/5 folds**
  (or in a quick test: positive in ≥ 4/4 evaluations and mean ≥ +0.001). Below +0.0003 counts as 0; do not report it as a "gain".
- R3. Screen quickly first: fold 0+1 × 2 seeds, lr 0.05 (~10 s each) → only ideas that pass the screen get the full 5-fold run at lr 0.01.
  Saves ~10× compute compared with a full run for every idea.
- R4. Every feature that uses TARGET (target encoding, KNN target mean, stacking) must be computed INSIDE the fold (only from the fold's train part), including when built for test.
  Statistics that do not use TARGET (frequencies, group means, imputation models) may use train+test.

**Compute allocation**
- R5. **Stop Optuna/hyperparameter tuning** for GBDT: proven to be 0. Use the existing opt parameters as defaults. Only tune if the feature set changes a lot, and then with ≤ 15 trials, scored on 5 folds with a fixed lr.
- R6. Priority order: (1) **new information** (features trees cannot derive: EXT imputation, implied interest rate, KNN target) >
  (2) **model diversity** (algorithms from a different family, OOF corr < 0.97) > (3) multi-seed > (4) blend weights > (5) hyperparameters.
- R7. Do not add batches of features of the same kind (ratio/diff/pairwise products) — round 0 proved that 85 such columns ≈ 0. Add by **hypothesis-driven group**, each group checked separately with R2/R3.
- R8. Do not prune features by gain (top-40 lost ~0.003).

**Combining techniques**
- R9. A new model only helps the ensemble if its OOF Spearman correlation with the current rank-avg is < ~0.97 AND its single AUC ≥ ~0.755.
  Check the correlation before investing in tuning.
- R10. Blending: equal-weight rank-average as the default; use stacking/weights only when there is ≥ 1 model with corr < 0.95, and always evaluate with nested CV.
- R11. When the feature set changes, rerun all 3 algorithms (LGB/Cat/XGB) on the new set and then rank-avg — the FE gain (+0.002) and the ensemble gain (+0.001) seem additive because they come from different sources.
- R12. Record every result in history.md with: OOF, per-fold difference vs the baseline, number of seeds. Save oof/pt under distinct names for re-blending.

##### 5. Diagnosis & priority directions

**Bottleneck: DATA/FEATURES, not the model or the hyperparameters.**
- Models: 3 algorithms × tuning ≈ 0.7606–0.7615, corr 0.97–0.995 → saturated.
- Evaluation: fine (adversarial 0.512, train/test identically distributed); only the paired-comparison discipline (R1–R3) is needed.
- Features: the first two groups of new information tried (EXT imputation, implied interest rate) both give a stable +0.002
  → the remaining headroom is here. Weakest groups: ≥ 2 missing EXT (AUC 0.740), pensioners (0.734), > 60 years old (0.721).

**Priority directions for the next round (in order):**
1. **Add to fe.py: model-based EXT_SOURCE imputation + RATE_n**, run all 5 folds for LGB/Cat/XGB, rank-avg.
   Expected single-LGB OOF ~0.764–0.765 (quick test +0.0038), ensemble ~0.766. Extensions: impute with several algorithms, use extra terms n=6,10,18,30,42 for RATE;
   compute RATE with AMT_GOODS_PRICE as the principal (the credit−goods difference is insurance/fees); filter plausible RATE by NAME_CONTRACT_TYPE.
2. **In-fold KNN target mean** (like Kaggle's "neighbors_target_mean_500"): mean TARGET of the k=50/500 nearest neighbours
   in the normalized space {EXT_1/2/3 (imputed version), CREDIT_ANNUITY, DAYS_BIRTH}, computed only from the fold's train part. A "smooth" local signal
   that trees learn poorly; KNN can also be used as a separate model for diversity.
3. **Models from a different family for diversity**: an MLP with embeddings for categorical variables + quantile-normalized numeric features; or Logistic Regression
   on WOE/binning features (usually corr ~0.9 with GBDT). Keep only if R9 is satisfied.
4. **Multi-seed (3 seeds) for the final configuration** of each algorithm — cheap, reduces noise by ~0.0003–0.0005; do it AFTER the features are fixed.
5. (Optional, low risk, low benefit) High-confidence pseudo-labels on test; separate models for the Revolving/pensioner groups. Only try if 1–3 are done.

Realistic expectation: a ceiling of ~0.765–0.770 with the 33-column data; AUC 0.80 is not legitimately achievable — this should be reported to the user.

### Agent 3 — Implementation (ROUND 1 section in `loop/history.md`)
Scripts (in `$S`): `fe_r1.py` (`load_r1()` = `fe.load()` + groups IMP/RATE/TERM/DATE; imputation cache `r1_feats.pkl`), `r1_screen.py`, `r1_screen2.py` (screening),
`knn_r1.py` (nested KNN target → `r1_knn.pkl`), `run_r1.py <lgb|cat|xgb> IMP,RATE,TERM 0 <seeds> <tag>` (full 5 folds), `div_r1.py <lrbin|mlp>`,
`ens_final_r1.py` (ensemble comparison + writing the submission). Logs: `log_r1_{screen,screen2,knn,lgb,xgb,cat,lrbin,mlp}.txt`.
Final feature set: **fe.py (~80 columns) + IMP + RATE + TERM**.

Definitions of the new feature groups (none uses TARGET; computed on train+test):
- **IMP**: LGB regression (KFold 5, 800 rounds, lr 0.05) predicting EXT_SOURCE_k from the columns not containing EXT → `EXT_k_PRED` (OOF for rows with a value), `EXT_k_FILL`, `EXT_k_RES`, `EXTF_MEAN/PROD/MIN/MAX/STD/W` on the imputed version.
- **RATE**: implied annual rate `RATE_n` (n = 12/24/36/48/60) solving A = C·r/(1−(1+r)^−n) by bisection; Revolving → NaN.
- **TERM**: reconstructed term = argmax_n of the rate density (bin 0.25%/year, n ∈ {6,10,12,18,24,30,36,42,48,54,60}) → `TERM_HAT, RATE_HAT, TERM_DENS, TERM_DENS_RATIO, OVERPAY_REL, AGE_END_LOAN, PRODUCT_FREQ (freq of round(C/A,2)), CG_FREQ (freq of round(C/GOODS,3))`.
- DATE (dropped): age at the DAYS_* milestones, passport renewal delay at 14/20/45, AGE_FRAC, YEARS_TO_45, ID_MINUS_EMP/REG, GOODS_ANNUITY.
- KNN (dropped): mean TARGET of 50/200/500 neighbours in 2 quantile-normal spaces {EXT1/2/3_FILL, C/A} and {EXTF_MEAN, EXT2/3_FILL, C/A, DAYS_BIRTH}; nested (inner 5-fold for the train part, all of Tk for valid/test).

##### Quick screening (LGB lr 0.05, shallow trees, fold 0/1 × 2 seeds = 4 paired evaluations)
| Change | Mean of 4 | Paired difference | Positive | Decision |
|---|---|---|---|---|
| BASE (fe.py) | 0.75919 | — | — | |
| + IMP | 0.76157 | +0.00238 | 4/4 | KEEP |
| + RATE | 0.76165 | +0.00246 | 4/4 | KEEP |
| + IMP + RATE | 0.76321 | +0.00403 vs BASE (almost additive) | 4/4 | KEEP |
| IMP+RATE + TERM | **0.76531** | +0.00210 vs IMP+RATE | 4/4 | KEEP |
| IMP+RATE + DATE | 0.76278 | −0.00043 | 1/4 | DROP |
| IMP+RATE+TERM + DATE | 0.76444 | −0.00087 vs IR+TERM | — | DROP |
| IMP+RATE+TERM + KNN500 (2 columns) | 0.76477 | −0.00054 | 1/4 | DROP |
| IMP+RATE+TERM + all KNN (6 columns) | 0.76458 | −0.00074 | 1/4 | DROP |
(KNN-500 univariate AUC ≈ 0.73 — strong but redundant with the imputed EXT + C/A, and adds noise because of the inner/outer distribution shift.)

##### Full 5-fold runs (StratifiedKFold rs=42, early stopping on the valid fold as in round 0)
| Model (IMP+RATE+TERM set) | Seeds | OOF AUC | Per-fold AUC | Paired difference vs the round-0 version of the same algorithm |
|---|---|---|---|---|
| LGB (lr 0.01, leaves 16, ff 0.3) `lgb_r1` | 3 | **0.76696** | 0.76446 / 0.76733 / 0.77261 / 0.76322 / 0.76739 | +0.00545 +0.00477 +0.00618 +0.00540 +0.00537 → **+0.00543**, 5/5 |
| XGB (depth 5, colsample 0.3) `xgb_r1` | 1 | 0.76643 | 0.76373 / 0.76667 / 0.77194 / 0.76262 / 0.76739 | +0.00493 +0.00474 +0.00541 +0.00590 +0.00610 → **+0.00542**, 5/5 |
| CatBoost (depth 6, lr 0.03) `cat_r1` | 1 | 0.76545 | 0.76463 / 0.76483 / 0.77040 / 0.76142 / 0.76621 | +0.00465 vs cat (0.76080) |
| LR on 16-bin quantile one-hot binning (additive GAM, C=0.05; also uses DATE) `lrbin_r1` | — | 0.75688 | 0.75415 / 0.75628 / 0.76097 / 0.75394 / 0.75919 | new model; Spearman vs r1 GBDT rank-avg = 0.926 |
| sklearn MLP (128,64) on quantile-normal `mlp_r1` | — | 0.75482 | 0.75288 / 0.75536 / 0.75999 / 0.75105 / 0.75747 | new model; Spearman = 0.937 |
→ The FE gain at the full level (+0.0054) ≈ 90% of the screening level (IMP+RATE+TERM − BASE = +0.0061 at lr 0.05) — transfers well.

##### Ensemble (meta-model scored on unseen folds; all level-1 OOF use the same folds rs=42)
| Method | OOF | Per-fold AUC | Difference vs round-0 stack (0.76273) |
|---|---|---|---|
| rank-avg LGB/Cat/XGB r1 | 0.76751 | 0.76544 / 0.76772 / 0.77285 / 0.76362 / 0.76812 | +0.00477, 5/5 (vs single lgb_r1: +0.00055, 5/5) |
| LR stack LGB/Cat/XGB r1 | 0.76750 | | +0.00479 (≈ rank-avg, as R10 predicts) |
| rank-avg r1 GBDT + lrbin + mlp | 0.76747 | | +0.00478 (equal-weight rank-avg CANNOT absorb weaker models) |
| LR stack r1 GBDT + lrbin + mlp | 0.76786 | | +0.00517 |
| **LR stack r1 GBDT + lrbin** (CHOSEN) | **0.76790** | **0.76579 / 0.76811 / 0.77310 / 0.76412 / 0.76872** | **+0.00517, 5/5** |
| LR stack r1 GBDT + mlp | 0.76746 | | mlp adds ≈ 0 |
| LR stack of the 6 round-0 models + r1 GBDT + lrbin + mlp | 0.76800 | | +0.00015 vs (G1+D1), below the R2 threshold, negative coefficients for xgb/xgb_opt → DROP |
Contribution of lrbin in stacking (G1+lrbin − G1): +0.0003 +0.0004 +0.0002 +0.0005 +0.0006 → **+0.0004, 5/5** (borderline per R2 but stable → KEEP).
MLP: +0.00002 once lrbin is present → DROP. Stack coefficients: lgb_r1 0.292, cat_r1 0.319, xgb_r1 0.265, lrbin_r1 0.175.

**Final OOF AUC of round 1 = 0.76790** (round 0: 0.76273, **+0.00517**, positive in 5/5 folds).
Candidate submission: `$S/loop/sub_round1.csv` (61,500 rows, order = sample_submission, TARGET = normalized rank of the stack logit, no NaN). Not yet written to the repository.

##### Methods ALREADY used (added in round 1 — Agent 1 must not propose them again)
EXT_SOURCE imputation with LGB regression + residuals + imputed EXT aggregates; implied interest rate for fixed terms; term reconstruction via rate density
(TERM_HAT/RATE_HAT/confidence/overpay/age at payoff); frequency of the "product code" round(C/A), round(C/GOODS); domain date/age features & passport renewal delay (tried, dropped);
nested KNN target mean 50/200/500 (tried, dropped); LR on quantile binning (GAM/WOE-like); sklearn MLP on quantile-normal; LR stacking with model selection by paired ablation.

##### Surprising findings
1. The FE gain transfers intact to all 3 algorithms (+0.0054 LGB, +0.0054 XGB, +0.0047 Cat) — genuinely new information, not noise of one model.
2. KNN target 500 (the "very strong" feature of the Kaggle winners) actually LOWERS the score here: once EXT is imputed, KNN has no new information left; also the inner(80%)/outer(100%) distribution shift misleads the trees.
3. Date/passport features (univariate AUC within age groups 0.53–0.55) lower the score — the signal is too weak relative to the noise of adding 12 columns.
4. TERM (reconstructed term) adds +0.0021 on top of raw RATE → choosing the right term via rate density carries information the raw RATE_n does not.
5. Model diversity: only LR-binning helps (+0.0004); the MLP has corr 0.937 but lower AUC, so it adds nothing; the round-0 models no longer add value.

##### Lessons carried out (cross-checked against round1_agent2.md and round1_agent1.md)
| Code | Direction / lesson | Status |
|---|---|---|
| L1.1 (A2 §5.1) | Add EXT imputation + RATE_n to the feature set, run LGB/Cat/XGB fully, rank-avg | DONE (round 1): LGB 0.76696, XGB 0.76643, Cat 0.76545; rank-avg 0.76751 |
| L1.1b (A2 §5.1 extension) | Extra terms n=6,10,18,30,42; filter RATE by contract type | PARTLY DONE: used in TERM (11 terms), Revolving → NaN |
| L1.1c (A2 §5.1 extension) | RATE computed with AMT_GOODS_PRICE as the principal; imputation with several algorithms | NOT DONE |
| L1.2 (A2 §5.2, A1 #2) | Nested KNN target mean (k=50/200/500, 2 spaces) | DONE: −0.0005…−0.0007, 1/4 → DROPPED. KNN as a separate model: NOT DONE |
| L1.3 (A2 §5.3) | Models from another family: LR binning/WOE, MLP | DONE: lrbin 0.75688 (kept, +0.0004 in the stack), MLP 0.75482 (dropped, ≈0) |
| L1.4 (A2 §5.4) | Multi-seed for the final configuration | DONE for LGB (3 seeds); XGB/Cat 1 seed (time limit) |
| L1.5 (A2 §5.5) | Pseudo-labels on test; separate models for the Revolving/pensioner groups | NOT DONE |
| R1–R3 (A2 §4) | Paired comparison, threshold +0.0005 & ≥4/5, screening fold0/1×2 seeds | APPLIED to every keep/drop decision |
| R4 | Features using TARGET computed in-fold | APPLIED (nested KNN) |
| R5 | No more Optuna tuning | APPLIED (round-0 parameters used) |
| R7 | Add hypothesis-driven groups, test separately | APPLIED (IMP/RATE/TERM/DATE/KNN tested group by group) |
| R8 | Do not prune features by gain | APPLIED |
| R9/R10 | Check correlation first; stack only with a model of corr < 0.95, nested CV | APPLIED (lrbin corr 0.926 → stacking > rank-avg) |
| R11 | Feature change → rerun all 3 GBDTs | DONE |
| R12 | Save oof/pt under distinct names, record the number of seeds | DONE (`oof_/pt_{lgb,xgb,cat,lrbin,mlp}_r1.npy`) |
| A1 #1 | Term & rate reconstruction + product code | DONE (RATE + TERM): +0.0025 / +0.0021 in screening. The "posterior prior×density"/KDE version: NOT DONE |
| A1 #2 | KNN target mean 500 | DONE → DROPPED (see L1.2) |
| A1 #3 | Model-based EXT imputation + residuals | DONE (IMP): +0.0024 in screening |
| A1 #4 | Domain date/age features, passport at 20/45 | DONE → DROPPED (−0.0004, 1/4) |
| A1 #5 | RealMLP / TabM (pytabkit, needs torch ~3 GB) | NOT DONE (replaced by an sklearn MLP, which did not help) |
| A1 #6 | OOF target encoding for categorical combinations (ORG×OCC, PRODUCT×contract type) | NOT DONE |
| A1 #7 | Null Importance feature selection | NOT DONE |
| A1 #8 | Repeat-client matching (uid from DAYS differences) | NOT DONE |
| A1 #9 | LightGBM DART / extra_trees / linear_tree for diversity | NOT DONE |
| A1 #10 | TabICL / TabPFN | NOT FEASIBLE (O(n²), needs a GPU, 171k rows × 4-core CPU) |
| A1 (not recommended) | Adversarial validation | DONE by A2 (AUC 0.512) — diagnostic only |

---

## ROUND 2

### Agent 1 — Research of new methods (`loop/round2_agent1.md`)
STATUS: NEW_METHODS_FOUND

#### ROUND 2 — Agent 1 (Research): new methods

Read: `rules.md`, `history.md` (baseline OOF 0.76790), `round1_agent1.md`, `fe_r1.py`, `round2_agent2.md`.
NOT counted as mine: in-fold target encoding of PRODUCT=round(C/A,2) (Agent 2, L2.1/L2.2).
Scripts and intermediate data: `$S/loop/a1r2/` (`h2.py`, `t1.py`, `t2.py`, `mkprof.py`, `screen2.py`, `x*.py`, `em_term.py`, logs `log_screen2/3/4.txt`).
Network: web search this round gave no valuable new sources; the main finding comes from data analysis.

##### One-line summary
Found ONE new feature group with strong evidence: **the "loan-product population profile" (label-free, computed on train+test)** — LGB screening exactly per R3: **+0.0026 … +0.0031, positive 4/4** on top of REF IMP+RATE+TERM (on par with IMP or RATE alone in round 1). The other directions I tried are all ≈ 0 (listed below so they are not repeated).

##### Two measurement tools used
1. **"Remaining information" harness** (`h2.py`): a shallow LGB (depth 3, lr 0.03) boosting FURTHER from the logit of the round-1 stack (init_score = calibrated stack OOF), using only the candidate feature group, 5 folds rs=42; reports the AUC difference at 100 rounds. Controls: 5 noise columns = −0.00006; raw columns C/A, C/G, RATE_HAT, TERM_HAT = −0.0002; imputed EXT = −0.0006 → the harness does not generate positive scores by itself.
2. **R3 screening** (`screen2.py`, identical to the round-1 screening configuration: fold 0/1 × 2 seeds, LGB lr 0.05 depth 4). REF reproduced exactly at 0.76531 as in history.md.

##### Ranking

| Rank | Method | Evidence | Expected OOF gain | Cost | Risk |
|---|---|---|---|---|---|
| 1 | Loan-product population profile (target-free, train+test) as columns for the GBDT | R3 screening: +0.0026…+0.0031, 4/4 (6 variants) | +0.002 … +0.003 for the single model and the stack | a few seconds + rerunning 3 GBDTs | same signal as A2's TE PRODUCT (not additive) |
| 2 | "Prediction profile": mean predicted PD (round 1) by product, including test rows | not measured; inferred from #1 | 0 … +0.001 on top of #1 | minutes, must be nested | indirect leakage if not nested |
| 3 | EBM / GA2M (`interpret`) as a level-1 model from a different family | not measured; additive lrbin already gave +0.0004 | 0 … +0.0005 in the stack | 20–60 min | may be too correlated with GBDT |
| — | Tried, NO signal (do not redo) | see "Negative results" | ≈ 0 | — | — |

---

##### 1. Loan-product population profile

**Idea.** Home Credit prices by risk: each product (term, rate) is sold to a customer segment chosen by the bank's internal model. So "who buys this product" (mean EXT, mean age, mean employment length… of ALL customers with the same product in train+test) is an estimate of the risk grade the bank assigns to that product. Unlike A2's TE: it does NOT use TARGET → no nesting needed, no inner/outer distribution shift (the cause of KNN target's failure in round 1), and it can use the extra 61.5k test rows (+36% data for the group statistics).
Unlike the "group-mean diff by ORG/OCC" used in round 0: the group key here is the PRODUCT, and only this key carries signal (see the key table).

**Evidence 1 — remaining-information harness (AUC difference vs the stack 0.76792, 100 rounds; number in brackets = positive folds/5):**

| Group key | Levels | Target-free profile | Nested TE (comparison) |
|---|---|---|---|
| round(C/A, 2) (cash) | 2,204 | **+0.00129 (5)** | +0.00111 (5) |
| TERM_HAT × round(RATE_HAT, 0.5%) | 682 | **+0.00114 (5)** | +0.00110 (5) |
| round(A/C, 4) | 815 | +0.00103 (5) | +0.00111 (5) |
| round(C/A, 1) | 299 | +0.00095 (5) | +0.00091 (5) |
| round(C/A, 3) | 11,084 | +0.00084 (5) | +0.00052 (5) |
| C/A × insurance package k | 3,298 | +0.00090 (5) | +0.00066 (5) |
| C/A × REGION_RATING | 3,489 | +0.00082 (5) | +0.00060 (4) |
| exact AMT_ANNUITY | 12,981 | +0.00030 (5) | +0.00022 (5) |
| REGION_POPULATION_RELATIVE (81 regions) | 81 | −0.00002 | −0.00007 |
| insurance package k, AMT_GOODS_PRICE, AMT_CREDIT, AMT_INCOME, ORG×OCC, REGION×ORG, birth year×gender | — | −0.0002 … −0.0006 (0–2) | ≤ 0 |

- Profile + TE together (2-digit C/A key): +0.00129 = the same as the profile alone → **the same signal as A2's TE, not additive**.
- Column by column (2-digit C/A key): mean EXT_SOURCE_3 **+0.00116**, EXTF_MEAN +0.00093, EXT_1_PRED +0.00067, age +0.00061, DAYS_EMPLOYED +0.00054, EXT_2 +0.00052, share of men +0.00026; income, credit amount, share with insurance, group row count, (individual EXT − group EXT) ≤ 0.

**Evidence 2 — R3 screening (LGB, columns added to REF IMP+RATE+TERM = 0.76531):**

| Variant | Columns | Mean of 4 | Paired difference | Positive |
|---|---|---|---|---|
| TR16: key TERM×RATE(0.5%), 16 columns (`prof_tr.pkl`) | 16 | 0.76844 | **+0.00312** | 4/4 |
| RATEn: 5 keys round(RATE_n, 0.5%) n=12/24/36/48/60 × {e, e3, age, emp} (`pv_RATEn.pkl`) | 20 | 0.76838 | **+0.00306** | 4/4 |
| RATEn + TR16 | 36 | 0.76819 | +0.00287 | 4/4 |
| TRmulti: key TERM×RATE at 3 resolutions (2% / 0.5% / 0.1%) | 24 | 0.76812 | +0.00281 | 4/4 |
| TR + CA2 (8 columns per key) | 16 | 0.76802 | +0.00271 | 4/4 |
| RATEn + TR16 + CA2 | 44 | 0.76799 | +0.00268 | 4/4 |
| TRwide: key TERM×RATE, 43 columns (adding occupation, education, housing shares…) | 43 | 0.76795 | +0.00264 | 4/4 |
| TRcore: key TERM×RATE, 8 columns | 8 | 0.76793 | +0.00261 | 4/4 |
| CA2S: key round(C/A,2), 8 columns | 8 | 0.76725 | +0.00194 | 4/4 |

Conclusion: every variant is positive 4/4; **saturated at around +0.003** — adding keys or columns no longer helps (the 0.0005 spread across variants is within noise). Choose the compact version.
Note: the harness measures on the stack (+0.0013) while the screening measures on a single LGB at lr 0.05 (+0.003); the gain on the final stack has not been measured directly and is expected to lie between the two numbers.

**Implementation sketch (exactly the code that was run, `mkprof.py`):**
```python
# df = load_r1() (train+test pooled); y is NOT used
cash = df.NAME_CONTRACT_TYPE=='Cash loans'
core = DataFrame(e=EXTF_MEAN, e2=EXT_SOURCE_2, e3=EXT_SOURCE_3, e1p=EXT_SOURCE_1_PRED,
                 age=DAYS_BIRTH, emp=DAYS_EMPLOYED (365243→NaN), male=(CODE_GENDER=='M'))
def profile(key, cols, pre, minc=30):
    cnt = number of rows per key (train+test)
    out[pre+'_'+c] = groupby(key)[c].transform('mean')     # mean ignores NaN
    out[pre+'_estd'] = groupby(key).e.transform('std')
    out[cnt < minc] = NaN                                   # rare groups → NaN
# Option A (recommended, 8–16 columns): key = TERM_HAT*100000 + round(RATE_HAT*200); Revolving → its own key (-5)
# Option B (independent of TERM_HAT, 20 columns): for n ∈ {12,24,36,48,60}: key_n = round(RATE_n*200); columns {e, e3, age, emp}
```
Prebuilt files (232,702 rows, in `load_r1()` order): `$S/loop/a1r2/prof_tr.pkl` (A, 16 columns), `pv_TRcore.pkl` (A, 8 columns), `pv_RATEn.pkl` (B), `prof_ca2s.pkl`.

**What Agent 3 should do:**
1. Add the PROF group (A or B, one of the two) to `fe_r1.py`, rerun LGB/XGB/Cat on all 5 folds (R11), then stack with lrbin.
2. Compare with/without A2's TE PRODUCT at the meta level: since it is the same signal, once the GBDT has PROF the TE column at the meta level will most likely lose its effect (R16) — report both configurations.
3. Add PROF to lrbin as well (an additive model benefits directly from a "product risk grade" column).

**Risk.** No label leakage (TARGET is not used anywhere). Using features of test rows is legitimate under the constraints. Real risks: the gain on the full 5 folds at lr 0.01 is smaller than in screening (round 1 transferred ~90%), and it overlaps with A2's TE.

##### 2. Prediction profile (not measured)
Instead of the mean EXT, take the mean predicted logit of the round-1 model over all rows with the same product (train + test). This is a group-level form of "self-training": it uses the unlabelled test data and is smoother than TE because each row contributes a probability instead of a 0/1 label.
Nesting is mandatory: for outer fold k, predict all rows (train rows in the fold: inner OOF; valid + test: the model trained on Tk) before aggregating by product; with a single-level OOF the valid-fold labels leak indirectly into the feature. Expected 0…+0.001 on top of #1 (same signal source), much higher cost → only try after #1.

##### 3. EBM / GA2M (not measured)
Explainable Boosting Machine (`interpret`; the wheel `interpret_core-0.6.9` was downloaded to `$S/loop/a1r2_pip/`): an additive model + pairwise interactions, cyclic boosting feature by feature. lrbin (purely additive, 0.7569) already gave +0.0004 in the stack; EBM has pairwise interactions, so its single AUC is expected at ~0.760–0.764. Keep only if its Spearman with the GBDT rank-avg is < 0.95 (R9). Expected 0…+0.0005. Suggested configuration: `ExplainableBoostingClassifier(interactions=20, max_bins=256, outer_bags=4, n_jobs=4)`, same folds rs=42.

##### Negative results (measured, do not repeat)
| Direction | Result |
|---|---|
| Term reconstruction by EM (latent mixture, conditional on the insurance package; `em_term.py`, 8 columns TERM_EM/RATE_EM/POST_MAX/TERM_EXP/RATE_EXP/TERM_ENT/INS_K/INS_PER_M) | R3 screening: **−0.00001, 2/4** → no help. Related to A2's L2.3 and to the unfinished "posterior" part of A1 #1 in round 1 |
| Hidden insurance fee: (C/G − 1) is a multiple of 1.32% for 89.8% of cash loans with C>G (k = 3…50); default rate by k from 2.6% (k=3) to 18.3% (k=32) | the information is already in CREDIT_GOODS: harness one-hot/profile/TE by k all ≤ 0 |
| Term inferred from the insurance package (discreteness of the rate for each k; `x9.py`) | cannot identify the term (HHI almost equal across n) |
| Roundness of amounts (annuity/income/goods price divisible by 4.5k/9k/22.5k/45k/225k) | harness: −0.0001 … 0 |
| Repetition count of EXT_SOURCE_1/2/3 values | ≈ 0; two rows with the same EXT_SOURCE_2 are NOT the same person (DAYS_BIRTH match 0%) |
| Region (REGION_POPULATION_RELATIVE = 81 region codes) as a group key / TE | ≈ 0 |
| New 2025–2026 tabular methods (TabPFN-3, TabICLv2, LimiX…) | all need a GPU at 171k rows; TabArena reports that TabM/ModernNCA lose quality on CPU with a time limit → not feasible here |

##### Honest assessment of 0.80
- After round 2, a reasonable estimate: 0.7679 + (0.0015 … 0.003 from the product signal; PROF and TE are not additive) + (0 … 0.0005 from model diversity/multi-seed) ≈ **0.770 – 0.771**.
- Every "remaining information" probe outside the product key gives ≈ 0 → the 33-column data is nearly exhausted. The gap to 0.80 is still ~0.03, about 10× the gain of a good round.
- **0.80 is not achievable with legitimate methods on this data.** Once #1 is implemented, I expect round 3 to have great difficulty finding further new methods with evidence; #2 and #3 are the remaining unmeasured directions.

### Agent 2 — Lessons from Round 1 (`loop/round2_agent2.md`)
#### ROUND 2 — Agent 2 (Lessons learned from round 1)

Sources: `rules.md`, `history.md` (round 0 + round 1), `round1_agent2.md`, `fe_r1.py`, `run_r1.py`, `div_r1.py`, `ens_final_r1.py`, `log_r1_*.txt`, OOF `*_r1.npy`.
My light-weight analysis: `$S/a2r2/b1.py … b6.py`, logs `$S/a2r2/b1.log … b6.log`. The repository was not modified.
Baseline: LR stack (lgb_r1 + cat_r1 + xgb_r1 + lrbin_r1) = **0.76790** (my reconstruction: 0.76790; round-0 stack reconstructed: 0.76271).

Reliability notes:
- [MEASURED] = measured directly on the 5-fold rs=42 OOF. [INFERRED] = reasoning, not yet verified.
- The screening "TE as a feature for LGB" did NOT finish (the machine was CPU-contended by another agent, 2 time-outs) → no numbers yet; do not treat it as tested.

---------------------------------------------------------------------------------------------------
##### 1. Round-1 failures & causes

| Experiment | Result | Cause |
|---|---|---|
| DATE (12 columns: age at milestones, passport renewal delay…) | −0.0004 (1/4), −0.0009 with TERM | [INFERRED] All are linear combinations of DAYS_* that the trees can already split on; univariate AUC 0.53–0.55. Adding 12 weak columns at once with feature_fraction 0.2–0.3 dilutes the probability of picking strong columns (EXTF_MEAN holds 20% of the gain). Violates the spirit of R7 (group too large, signal too weak). |
| KNN target 50/200/500 as features | −0.0005…−0.0007 (1/4) | (a) The KNN space = {imputed EXT, C/A, age} → redundant with EXTF_* (already 35.5% of the gain [MEASURED]). (b) Inner(80%)/outer(100%) shift of the feature between train and valid. (c) KNN univariate 0.73 < model 0.767, so it is only a "blurred" copy of the model itself. |
| sklearn MLP (128,64) | 0.75482; ≈ 0 when added to the stack | [MEASURED] corr with lgb_r1 0.932 (different enough) but worse than lgb_r1 **uniformly by −0.010 in EVERY group**, and corr 0.865 with lrbin. The model is 0.012 weaker and not better in any group → nothing for the stack to take. The sklearn MLP has no embeddings, no NaN handling (median fill), no early stopping on AUC. |
| LR-binning (kept, but small benefit: +0.0004) | 0.75688 | [MEASURED] lrbin − lgb_r1 by group: −0.0022 (TERM 24) … −0.018 (age > 60), −0.015 (pensioners). lrbin wins in no group → the benefit comes only from averaging out noise, not from covering blind spots. An additive model has no EXT×age×product interactions. |
| Adding the 6 round-0 models to the stack | +0.00015, negative coefficients | [MEASURED] corr lgb(round 0)–lgb_r1 = 0.975 but 0.0054 weaker and an information SUBSET of r1 → the meta-model assigns negative coefficients (fitting noise). Lesson: old models on a subset of the features are NOT a source of diversity. |
| Equal-weight rank-avg when adding lrbin/mlp | 0.76747 < 0.76751 | Equal rank-avg only fits when the models are equally strong; a model 0.01 weaker needs a small weight (the stack gives lrbin 0.175). |
| XGB/Cat with only 1 seed | — | Time limit (XGB 838 s, Cat ~17 min/seed). Still owed (L1.4). |

Methodological failures (not results):
- Screening used only LGB. A label-using feature (KNN) was rejected in its "column for the trees" form but never tried at the meta level — while at the meta level, a different label-based signal (TE of the product code, item 3) gives the largest benefit found so far.
- TERM was accepted as an 8-column block; nobody checked whether TERM_HAT is correct (see 3.4).

##### 2. Tips from the round-1 successes

1. **IMP works because it turns "missing" into "estimated"** — and EXT is 80% of the model. [MEASURED, fold 0, lr 0.05, AUC 0.76417]:
   jointly permuting the 15 IMP columns → −0.156; all EXT columns (old + IMP) → **−0.208**; whereas RATE+TERM → −0.017, age → −0.013, ORG/OCC → −0.005, old AMT → −0.005.
   Gain: IMP 35.5%, RATE 5.6%, TERM 7.9%, the 80 old columns 51%. `EXTF_MEAN` alone 20.3% of the gain.
   The gain by group matches the hypothesis [MEASURED, stack r1 − stack r0]: 2 missing EXT **+0.0086** (0.7401→0.7487), missing EXT3 **+0.0087**, 1 missing EXT +0.0044, all 3 EXT present +0.0035.
2. **Within IMP, the useful parts are `_PRED` and the `EXTF_*` aggregates, not the residuals**: permuting the 3 `_PRED` columns −0.0087; the 3 `_RES` columns only −0.0008 (gain 1%).
   → `_PRED` is a "demographic credit score" learned WITHOUT labels (EXT1_PRED is mainly age/gender/employment). Consequence: it also helps people with ALL EXT present (+0.0035).
3. **RATE/TERM work because they are information about the PRODUCT, independent of EXT**: Spearman(RATE_HAT, EXTF_MEAN) = −0.125 [MEASURED] → nearly orthogonal, hence additive with IMP (+0.0024 + +0.0025 ≈ +0.0040).
   The largest gain is at TERM_HAT=60 (**+0.0119**), age 50–60 (+0.0084), civil servants (+0.0096).
4. **Three layers from different sources add up**: FE (+0.0054 per GBDT) + 3-algorithm ensemble (+0.00055) + a different-family model at the meta level (+0.0004). r1 correlations [MEASURED]: lgb–xgb **0.996** (almost identical), lgb–cat 0.978, GBDT–lrbin 0.91–0.93. XGB is nearly redundant next to LGB; CatBoost is the real source of diversity within the GBDTs.
5. **Screening with 4 paired evaluations correctly predicts the full result** (≈ 90% transfer) → keep trusting the R3 procedure.

##### 3. Surprising findings (all [MEASURED], scripts `a2r2/`)

1. **In-fold target encoding of the "product code" is a signal all 4 models currently miss — the largest found this round.**
   Key `PRODUCT = round(AMT_CREDIT/AMT_ANNUITY, 2)` (2,119 levels), smoothed TE `(sum + α·prior)/(count + α)`.
   Adding logit(TE) as the 5th column of the meta-LR, STRICTLY nested (the TE of the meta train part computed with an inner 5-fold within the train part; the TE of the scored fold computed from the whole train part; labels of the scored fold are never used) — `b4.py`, `b5.py`:

   | Column added to the meta-model | Per-fold difference vs 0.76790 | Mean | Positive |
   |---|---|---|---|
   | TE PRODUCT (2 digits), α=10 | +0.00138 +0.00159 +0.00121 +0.00190 +0.00165 | **+0.00155** | 5/5 |
   | TE PRODUCT, α=5 / 20 / 100 / 300 | | +0.00153 / +0.00149 / +0.00108 / +0.00072 | 5/5 |
   | TE PRODUCT 3 digits, α=20 / 100 | | +0.00110 / +0.00067 | 5/5 |
   | TE (TERM_HAT × RATE_HAT rounded to 1%), α=20 / 100 | | +0.00125 / +0.00107 | 5/5 |
   | TE (exact AMT_CREDIT × AMT_ANNUITY), α=20 | | +0.00084 | 5/5 |
   | TE PRODUCT + TE TERM×RATE together (α=100) | | +0.00111 | 5/5 (NOT additive → the same signal) |
   | TE ORG×OCC / ORG×INC×EDU / CREDIT×GOODS | | −0.00005 / −0.00007 / +0.00003 | useless |

   → estimated new stack ≈ **0.7694** with just one column, ~4× the contribution of lrbin. Small α (5–20) beats large α.
   TE PRODUCT univariate AUC 0.669, Spearman with the stack only 0.49. The signal lies in common products: PRODUCT groups with 200–1000 rows have TE AUC 0.690, 30–200 rows 0.665; < 30 rows ≈ 0.51–0.56 (noise).
   Why the GBDT misses it [INFERRED]: the default rate by product code is NOT monotone in C/A (each discrete C/A value is a separate product/programme); shallow depth-4 trees + feature_fraction 0.2 cannot cut ~2,000 narrow intervals; `PRODUCT_FREQ` only says how common a product is, not how risky.
   WARNING: so far only tested at the META level. The "TE column for the GBDT" form has no result yet (the run did not finish).
2. **A smarter meta-model does NOT help**: meta-LR with group indicators (2 missing EXT, pensioner, Revolving, > 60, male) and logit×group products: −0.00033, **0/5**. A 4-leaf meta LGB on the logits + 6 original variables: −0.00040, 1/5.
   → No beneficial "per-group weighting" exists; improvement must come from NEW signal at level 1 or a new column at the meta level, not from a more complex meta-model.
3. **The weak groups are still the old ones; the gap has barely narrowed** (stack r1 AUC): age > 60 **0.7274**, pensioners 0.7405, age < 30 0.7397, 2 missing EXT 0.7487, missing EXT3 0.7517; vs all EXT present 0.7770.
   By product: TERM_HAT 48 → **0.7315** (17.7% of the data), 10 → 0.7371, 18 → 0.7398 (default rate 13.8%, the highest), 42 → 0.7465; whereas 12 → 0.7946, 24 → 0.7833, 54 → 0.7790.
   Men gained only +0.0031 from round 1, women +0.0068. Revolving +0.0037 (no RATE/TERM).
4. **TERM_HAT is suspect; the model hardly uses it**: gain of `TERM_HAT` = 0.05% (the lowest among the 28 new columns); the trees use `RATE_36`, `RATE_HAT`, `TERM_DENS_RATIO`, `AGE_END_LOAN`.
   58% of Cash loans have RATE_HAT > 35%/year, median 40%, p90 = 55% (close to the 60% filter cap) → the density algorithm is biased towards LONG terms/HIGH rates. Only 18% of rows have TERM_DENS_RATIO > 2 (clear term), 64.5% have a ratio ≤ 1.2 (almost a tie).
   The sign of RATE_HAT within each term is inconsistent: AUC(RATE_HAT→default) = 0.47 (n=12), 0.455 (24), 0.47 (36) but 0.61 (54), 0.61 (60).
   → TERM currently acts as a "product hash" rather than the true term; this is also why TE by PRODUCT (a raw key that needs no term guess) is strong.
5. **The largest error is noise that the current features cannot reduce**: 683 defaulters (4.9% of positives) are ranked in the safest 25% — median EXTF_MEAN 0.636 (higher even than NON-defaulters: 0.537), age 48, 78% women, 24% pensioners. Conversely, 3,164 non-defaulters are in the riskiest 3%: EXTF_MEAN 0.275, age 35.6, 55% men, 89% low education, 98% Cash.
   The model is already "right according to EXT"; its mistakes are cases where EXT contradicts reality. The riskiest decile contains 34.5% of defaults (rate 27.9%), the top 30% contains 66.3%.
6. **Repeat customers barely exist**: the key (gender, DAYS_ID_PUBLISH−DAYS_BIRTH, DAYS_REGISTRATION−DAYS_BIRTH) gives 9,650/232,702 duplicate rows (4,701 groups), but adding the condition of equal EXT1 leaves only 124 rows; in train, the duplicate groups have a default rate of 6.2% (below the 8.1% base), 247/2,675 groups with mixed labels. → mostly coincidental matches; the "uid" direction (A1 #8) is expected to give ≈ 0.

##### 4. Combining techniques & logic optimization for round 2 — rules for Agent 3

Keep R1–R12 from round 1. Additions:

- **R13 (label-based features → try at the META level first).** Every signal that uses TARGET (TE, KNN, WOE) is tested at the meta level first (seconds, no GBDT retraining), following exactly the strict nesting pattern of `a2r2/b4.py` (`nested_te` + `metaF`). Only if the meta result is positive ≥ 4/5 should it be tried as a GBDT column.
- **R14 (nesting TE correctly).** For each outer fold (a = train, b = valid): TE for a with an inner 5-fold WITHIN a; TE for b from all of a; TE for test from the whole train (or the average of the 5 per-fold versions). Never use a "global 5-fold OOF" TE as a training feature and then score on those same folds. Choose α in {5, 10, 20} — do not tune more finely (the difference between α 5/10/20 is only 0.00006).
- **R15 (one signal = one column).** TE PRODUCT and TE TERM×RATE are not additive (+0.00111 together < +0.00149 alone) → choose ONE key (PRODUCT 2 digits, α=10). Do not add 5 TE variants at once.
- **R16 (when putting TE into the GBDT).** If tried: compare paired with REF IMP+RATE+TERM per R3; if the GBDT absorbs the TE, the TE column at the meta level will lose its effect → report BOTH configurations (TE only at meta / TE in the GBDT) and choose by the final stack OOF, not by the single-model OOF.
- **R17 (no complex meta-model).** Meta = LR on logits, standardized within the fold. Measured: group interactions −0.0003 (0/5), meta LGB −0.0004 (1/5). Do not retry.
- **R18 (ensemble).** Do not include models that are information subsets of a stronger model (round-0 models, sklearn MLP). Condition for admitting a new model into the stack: corr < 0.95 with the GBDT rank-avg AND (AUC ≥ 0.760 OR it beats the GBDT in at least one group covering ≥ 5% of the data). XGB and LGB overlap at 0.996 → if short of time, drop multi-seed XGB first and prioritize seeds for CatBoost/LGB.
- **R19 (small feature groups).** Each tested group ≤ 5 columns; larger groups must be split and screened part by part (lesson from DATE with 12 columns, TERM with 8 columns of which TERM_HAT has 0.05% gain).
- **R20 (report by group).** Each round, reprint the AUC-by-group table (`a2r2/b1.py`, runs < 1 minute when the machine is idle) for the new stack: missing EXT, pensioners, age, TERM_HAT — to know where the gain comes from.
- **R21 (resources).** The 4-core machine is shared by several agents: one 35 s LGB screening can become > 5 minutes. Do not run more than 1 process with `n_jobs=4` in parallel; log with `flush=True` at each step.

Suggested combination order for round 2: (1) meta + TE PRODUCT → fix the new baseline; (2) screen TE/TERM fixes in the GBDT; (3) if the feature set changes → rerun the 3 GBDTs (R11), then restack with lrbin + TE; (4) multi-seed last.

##### 5. Diagnosis of the bottleneck & priority directions

**The bottleneck is still INFORMATION, more specifically: (i) the product-code signal has not been exploited in the right form; (ii) the groups with missing EXT / pensioners / old / young people have no alternative source of information.**
- Models/meta: saturated (lgb–xgb 0.996; more complex meta-models are negative).
- Evaluation: stable; every conclusion above holds in 5/5 folds.
- The remaining errors are mainly "high EXT yet defaulted" cases → not fixable with 33 columns. Realistic ceiling estimated at ~0.770–0.772; **0.80 is not legitimately feasible** (no evidence to the contrary after 2 rounds).

**Priority directions (by expected benefit / cost):**
1. **In-fold TE of PRODUCT=round(C/A,2), α=10, as the 5th column of the meta-LR** (L2.1). Measured +0.0015, 5/5 → ~0.7694. Cost: seconds. A test TE (from the whole train) must also be built to produce the submission.
2. **Bring the product signal into level 1** (L2.2): a nested TE PRODUCT column for LGB/Cat/XGB, and/or PRODUCT as a categorical variable for CatBoost (built-in ordered target statistics, naturally leak-resistant). Expected +0.001–0.002 if the trees can combine TE with EXT/age; risk: an inner/outer shift as with KNN → must pass R3, R16.
3. **Fix the term reconstruction** (L2.3): add a prior on plausible rates (penalize rates > 35–40%/year) or use EM/posterior (overlaps the unfinished part of A1 #1); check by: the share of TERM_DENS_RATIO > 2 rises, the gain of TERM_HAT rises, the screening is positive. Benefit uncertain (TE PRODUCT may already capture it all); priority after 1–2.
4. **Weak groups** (L1.5b → made concrete as L2.4): features/models for pensioners & > 60 (18% of the data, AUC 0.7405/0.7274) and Revolving (9.5%, no RATE/TERM, 0.7609): e.g. TE by (NAME_INCOME_TYPE × education × 5-year age band) or a GBDT trained on the group alone and then fed into the stack. Note that the per-group meta-model was negative → it must be a level-1 model/feature.
5. **Multi-seed Cat/LGB** (the rest of L1.4) and finishing the outstanding items L1.1c, L1.2b, L1.5a — cheap but small expected gains (≤ +0.0005 each); do them last to close the checklist.

##### 6. LESSON CHECKLIST

| Code | Actionable direction / lesson | Status |
|---|---|---|
| L1.1 | Model-based EXT imputation + RATE_n in the feature set, run LGB/Cat/XGB fully, rank-avg | DONE (round 1: LGB 0.76696, XGB 0.76643, Cat 0.76545; rank-avg 0.76751) |
| L1.1b | Extra terms n=6,10,18,30,42; RATE by contract type | DONE (round 1, within TERM with 11 terms; Revolving → NaN; TERM +0.0021 in screening) |
| L1.1c | RATE computed with AMT_GOODS_PRICE as the principal; impute EXT with a second algorithm (e.g. CatBoost/ridge) and average | NOT DONE |
| L1.2 | Nested KNN target mean as a feature (k=50/200/500) | DONE (round 1: −0.0005…−0.0007, 1/4 → dropped) |
| L1.2b | KNN as a separate MODEL / a meta-level column (per R13: try logit(KNN-500) as a meta column, using the existing `r1_knn.pkl`) | NOT DONE |
| L1.3 | Models from a different family: LR-binning, MLP | DONE (round 1: lrbin 0.75688 kept, +0.0004; MLP 0.75482 dropped) |
| L1.4 | Multi-seed final configuration for all 3 algorithms | PARTLY DONE (round 1: LGB 3 seeds). The rest — Cat ≥ 2 seeds, XGB time permitting (R18): NOT DONE |
| L1.5a | High-confidence pseudo-labels on test | NOT DONE |
| L1.5b | Separate models/features for weak groups (pensioners, > 60, Revolving) at LEVEL 1 — see L2.4 | NOT DONE (merged into L2.4) |
| R1–R12 | Round-1 rules for evaluation/allocation/combination | DONE (round 1, applied to every decision) |
| L2.1 | Add logit(in-fold TE of PRODUCT=round(C/A,2), α=10) as a meta-LR column, nested per R14; build the test TE from the whole train; produce a submission | NOT DONE (A2 did a preliminary check: +0.00155, 5/5 → ≈0.7694; Agent 3 needs to implement it properly + test) |
| L2.2 | Bring the product signal into level 1: (a) nested TE PRODUCT column for the GBDT, R3 screening; (b) PRODUCT as a CatBoost categorical. Compare both configurations per R16 | NOT DONE (A2's screening did not finish because of CPU contention) |
| L2.3 | Fix the term reconstruction: prior/penalty on rates > 35–40%/year or posterior; check via TERM_DENS_RATIO, TERM_HAT gain, R3 screening | NOT DONE |
| L2.4 | Weak groups at level 1: in-fold TE (INCOME_TYPE × EDU × 5-year age band) at the meta level per R13, and/or separate GBDTs for pensioners+>60 and for Revolving fed into the stack | NOT DONE |
| L2.5 | Keep meta = simple LR; do NOT use a meta-model with group interactions / a meta GBDT (R17) | DONE (round 2 by A2: −0.00033 0/5; −0.00040 1/5 → rejected, Agent 3 need not redo it) |
| L2.6 | Do not add round-0 models / the sklearn MLP to the stack; model admission criterion R18 | DONE (round 1: +0.00015 and +0.00002 → rejected; R18 applies from now on) |
| L2.7 | Repeat-client matching (uid) | NOT FEASIBLE in a useful way (A2 measured: only 124 duplicate rows when EXT1 must also match; 247/2,675 duplicate groups with mixed labels → no signal) |
| L2.8 | Print the AUC-by-group table for the new stack every round (R20) and record it in history.md | NOT DONE |
| L2.9 | Feature groups tested with ≤ 5 columns; split TERM to test it piece by piece — specifically: screen dropping `TERM_HAT`, `EXT_k_RES` (gain 0.05% / 1%) to see whether the score drops (R19; different from R8 because this is a paired ablation, not pruning by gain) | NOT DONE |

Total: **10 items NOT DONE** (L1.1c, L1.2b, the rest of L1.4, L1.5a, L2.1, L2.2, L2.3, L2.4, L2.8, L2.9; L1.5b merged into L2.4).

### Agent 3 — Implementation (ROUND 2 section in `loop/history.md`)
Scripts (in `$S`): `fe_r2.py` (`load_r2()` = `load_r1()` + groups PROF/PROFX/PROFR/PROFC/RATEG/IMPR/IMPB; `nested_te`, `te_map`, `PRODUCT_KEY`; caches `r2_df.pkl`, `r2_imp_ridge.pkl`),
`r2_screen.py` (R3 screening, results in `r2_screen.json`), `run_r2.py <lgb|cat|xgb> <spec[~TE][~PCAT]> <seeds> <tag> [nfold]` (5 folds, checkpoints `ck_<tag>.npz`),
`div_r2.py <lrbin|ebm>`, `r2_grp.py` (per-group models), `ens_r2.py` + `ens_final_r2.py` (ensemble, writing the submission), `r2_meta1.py`/`r2_meta2.py` (meta-level checks), `r2_groups.py` (group table).
Logs: `log_r2_{screen1..4,meta1,catscr,grp,lgb,lgbte,cat,xgb,lrbin,ebm,ens1,ens_final,groups}.txt`. EBM: `interpret-core 0.7.8` installed into `$S/pylib` (the pre-downloaded 0.6.9 failed with numpy 2.5).
Final feature set: **IMP + RATE + TERM + PROF (8 columns)**; LGB adds the nested TE PRODUCT column; CatBoost adds PRODUCT as a categorical variable.

New definitions:
- **PROF** (no TARGET, train+test): key = TERM_HAT × round(RATE_HAT, 0.5%) (Revolving = its own key); per-key mean of {EXTF_MEAN, EXT_2, EXT_3, EXT_1_PRED, DAYS_BIRTH, DAYS_EMPLOYED, share of men} + std of EXTF_MEAN; keys with < 30 rows → NaN. Identical to A1's `pv_TRcore.pkl` (zero difference).
- **TE PRODUCT** (uses TARGET, strictly nested per R14): PRODUCT = round(C/A, 2), α = 10; for an outer fold (a, b): rows of a use an inner 5-fold within a; rows of b and test use all of a. Labels of the scored fold are never used.
- **PCAT**: PRODUCT (levels with < 30 rows merged into `rare`) as a native categorical variable for CatBoost (ordered target statistics).

##### Quick screening (LGB lr 0.05, fold 0/1 × 2 seeds; REF = IMP+RATE+TERM = 0.76531)
| Change | Mean of 4 | Paired difference | Positive | Decision |
|---|---|---|---|---|
| + PROF (8 columns) | 0.76793 | +0.00261 vs REF | 4/4 | **KEEP** (most compact version) |
| + PROF + PROFX (14 columns) | 0.76828 | +0.00296 vs REF; +0.00035 vs PROF (3/4) | 4/4 | DROP PROFX (below threshold) |
| + PROFR (20 columns, RATE_n keys) | 0.76838 | +0.00306 vs REF; +0.00045 vs PROF (3/4) | 4/4 | DROP (below threshold, 20 columns) |
| + PROFC (2-digit C/A key, 4 columns) | 0.76710 | +0.00178 vs REF | 4/4 | DROP (worse than PROF) |
| + nested TE PRODUCT (L2.2a) | 0.76744 | +0.00212 vs REF | 4/4 | worse than PROF |
| + PROF + TE PRODUCT | 0.76842 | +0.00310 vs REF; +0.00049 vs PROF (2/4) | 4/4 | screening inconclusive → measured on the full 5 folds (below): KEEP for LGB |
| + PROFR + TE | 0.76823 | +0.00291 vs REF | 4/4 | DROP |
| PROF − TERM_HAT (L2.9) | 0.76806 | +0.00013 vs PROF | 3/4 | no change → keep the column (dropping or keeping ≈ 0) |
| PROF − the 3 `_RES` columns (L2.9) | 0.76797 | +0.00005 | 2/4 | same as above |
| PROF − TERM_HAT − `_RES` | 0.76807 | +0.00014 | 3/4 | same as above |
| PROF + RATEG (rate with GOODS as the principal, 5 columns; L1.1c) | 0.76789 | −0.00004 | 2/4 | DROP |
| PROF + IMPR (EXT predicted by Ridge, 3 columns; L1.1c) | 0.76792 | −0.00001 | 3/4 | DROP |
| PROF, IMP replaced by IMPB (average of LGB + Ridge; L1.1c) | 0.76807 | +0.00015 | 2/4 | DROP |
| PROF + TE (INCOME_TYPE × EDU × 5-year age band) (L2.4) | 0.76791 | −0.00001 | 2/4 | DROP |
| PROF + PP (A1 round 2 #2: mean predicted logit by product, nested, including test) | 0.76785 | −0.00007 | 2/4 | DROP |
| PROF + soft pseudo-labels on all of test (weights p / 1−p; L1.5a) | 0.76790 | −0.00003 | 2/4 | DROP |
| PROF + hard pseudo-labels (lowest 50% → 0, highest 1% → 1; L1.5a) | 0.76820 | +0.00027 | 4/4 | DROP (below the +0.0005 threshold) |
| PROF + PRODUCT as a categorical for **LGB** | 0.76499 | **−0.00294** | 0/4 | DROP (LGB overfits a 2,000-level variable) |
(Pseudo-labels in the screening use test predictions of a model trained only on the fold's train part → no leakage of the scored fold's labels.)

CatBoost screening (lr 0.1, fold 0/1, 1 seed; L2.2b):
| Configuration | fold 0 / fold 1 | Mean | Difference vs PROF |
|---|---|---|---|
| PROF | 0.76484 / 0.76758 | 0.76621 | — |
| PROF + PCAT | 0.76694 / 0.76959 | 0.76826 | **+0.00205** (2/2) → KEEP |
| PROF + nested TE column | 0.76580 / 0.76863 | 0.76722 | +0.00101 (2/2), worse than PCAT |
| PROF + PCAT + TERM×RATE key as a cat | 0.76669 / 0.76851 | 0.76760 | −0.00066 vs PCAT → DROP |
| PROF + PCAT + round(C/GOODS,3) as a cat | 0.76760 / 0.76750 | 0.76755 | −0.00071 → DROP |
| PROF + PCAT + AMT_ANNUITY as a cat | 0.76587 / 0.76791 | 0.76689 | −0.00137 → DROP |

Meta-level checks on the round-1 OOF (stack r1 = 0.76790; strictly nested):
| Column added to the meta-model | OOF | Mean difference | Positive |
|---|---|---|---|
| TE PRODUCT α=5 / 10 / 20 (L2.1) | 0.76944 / 0.76946 / 0.76940 | +0.00153 / +0.00155 / +0.00149 | 5/5 (A2's numbers reproduced exactly) |
| logit KNN1_500 / KNN2_500 / both / KNN1_50 (L1.2b) | 0.76792 / 0.76789 / 0.76790 / 0.76790 | +0.00002 / 0 / +0.00001 / 0 | 4/5, 2/5, 3/5, 2/5 → DROP |
| TE (INCOME × EDU × 5-year age band) / × gender × occupation added (L2.4a) | 0.76788 / 0.76788 | −0.00002 | 1/5 → DROP |

##### Full 5-fold runs (StratifiedKFold rs=42)
| Model | Seeds | OOF AUC | Per-fold AUC | vs round 1, same algorithm |
|---|---|---|---|---|
| LGB + PROF `lgb_r2` | 3 | 0.76978 | 0.76726 / 0.77063 / 0.77518 / 0.76604 / 0.76993 | +0.00280, 5/5 |
| LGB + PROF + nested TE PRODUCT `lgbte_r2` | 3 | **0.77008** | 0.76748 / 0.77106 / 0.77563 / 0.76650 / 0.76996 | +0.00312; +0.00030 vs lgb_r2, 5/5 |
| CatBoost + PROF + PCAT `cat_r2` | 2 | 0.76947 | 0.76834 / 0.77021 / 0.77339 / 0.76586 / 0.76970 | **+0.00402**, 5/5 |
| XGB + PROF `xgb_r2` | 2 | 0.76929 | 0.76679 / 0.76987 / 0.77494 / 0.76547 / 0.76960 | +0.00286, 5/5 |
| LR-binning + PROF `lrbin_r2` | — | 0.75851 | 0.75619 / 0.75831 / 0.76240 / 0.75531 / 0.76043 | +0.00163 |
| EBM (interactions 10, outer_bags 4; A1 round 2 #3) `ebm_r2` | — | 0.76715 | 0.76519 / 0.76761 / 0.77196 / 0.76468 / 0.76662 | new; Spearman vs GBDT rank-avg = 0.980 (> 0.95) |
| Per-group LGB (L2.4b, lr 0.02, 2 seeds) — within-group AUC: separate model vs lgb_r2 | | | pensioners/>60: 0.7340 vs 0.7421; Revolving: 0.7480 vs 0.7593; <30: 0.7361 vs 0.7411; 2 missing EXT: 0.7455 vs 0.7520 | the separate models are WORSE than the general model by 0.005–0.011 within their own group |

##### Ensemble (meta scored on unseen folds; difference vs the round-1 stack = 0.76790)
| Method | OOF | Mean difference | Notes |
|---|---|---|---|
| rank-avg lgb/cat/xgb r2 | 0.77042 | +0.00248, 5/5 | |
| LR stack lgb/cat/xgb r2 | 0.77047 | +0.00255 | xgb coefficient ≈ −0.06 |
| + lrbin_r2 | 0.77062 | +0.00270 | lrbin_r2 +0.00014 |
| + lrbin_r2 + TE PRODUCT at meta | 0.77074 | +0.00282 | the meta TE adds only +0.00013 (5/5) once level 1 has PROF (previously +0.00155) |
| + lrbin_r2 + EBM | 0.77063 | +0.00271 | EBM +0.00001 (2/5) → DROP |
| + lrbin_r2 + per-group models (old / rev / young / ext2na) | 0.77052 / 0.77061 / 0.77064 / 0.77060 | −0.00010 … +0.00001 | DROP |
| + lrbin_r2 + lgbte_r2 (5 models) | 0.77082 | +0.00290 | negative xgb coefficient |
| **LR stack lgbte_r2 + cat_r2 + xgb_r2 + lrbin_r2 (CHOSEN)** | **0.77081** | **+0.00289, 5/5** | coefficients 0.506 / 0.396 / 0.036 / 0.123; +0.00019 (5/5) vs the version using lgb_r2 |
| same as above + TE at meta | 0.77083 | +0.00291 | meta TE redundant once LGB has TE (+0.00002) |
| 3 GBDT r2 + 3 GBDT r1 + lrbin_r2 + meta TE (8 columns) | 0.77090 | +0.00298 | +0.00009 vs the chosen version, violates R18 → DROP |

Three configurations of the product signal (R16): TE at meta only (on the round-1 models) 0.76946; PROF profile at level 1 only 0.77062; PROF + TE at meta 0.77074; PROF + TE in LGB + PCAT in CatBoost **0.77081**. The versions with PROF differ by ≤ 0.0002 (below the R2 threshold) — the 4-model version with all positive coefficients is chosen.

**Final OOF AUC of round 2 = 0.77081** (round 1: 0.76790, **+0.00289**, positive 5/5). Per fold: 0.76875 / 0.77172 / 0.77550 / 0.76727 / 0.77105.
Candidate submission: `$S/loop/sub_round2.csv` (61,500 rows, order = sample_submission, no NaN; Spearman with sub_round1 = 0.985). Not yet written to the repository.
Target 0.80: NOT reached (0.029 short).

##### AUC-by-group table (L2.8 / R20) — round-1 stack → round-2 stack
| Group | Level | n | share | default rate | AUC stack r1 | AUC stack r2 | Difference |
|---|---|---|---|---|---|---|---|
| ALL | all | 171202 | 1.000 | 0.081 | 0.7679 | 0.7708 | +0.0029 |
| EXT_NA | 0 | 36164 | 0.211 | 0.071 | 0.7770 | 0.7800 | +0.0030 |
| EXT_NA | 1 | 96310 | 0.563 | 0.080 | 0.7712 | 0.7734 | +0.0022 |
| EXT_NA | 2 | 38611 | 0.226 | 0.091 | 0.7487 | 0.7534 | +0.0047 |
| EXT3_NA | have | 116616 | 0.681 | 0.077 | 0.7749 | 0.7775 | +0.0026 |
| EXT3_NA | na | 54586 | 0.319 | 0.089 | 0.7517 | 0.7553 | +0.0036 |
| CONTRACT | Cash loans | 154988 | 0.905 | 0.083 | 0.7671 | 0.7703 | +0.0032 |
| CONTRACT | Revolving loans | 16214 | 0.095 | 0.056 | 0.7609 | 0.7606 | -0.0003 |
| PENSION | 0 | 140304 | 0.820 | 0.087 | 0.7679 | 0.7707 | +0.0028 |
| PENSION | 1 | 30898 | 0.180 | 0.054 | 0.7405 | 0.7445 | +0.0040 |
| AGE | (0, 30] | 25125 | 0.147 | 0.115 | 0.7397 | 0.7422 | +0.0025 |
| AGE | (30, 40] | 45673 | 0.267 | 0.096 | 0.7692 | 0.7716 | +0.0024 |
| AGE | (40, 50] | 42732 | 0.250 | 0.077 | 0.7617 | 0.7649 | +0.0031 |
| AGE | (50, 60] | 37981 | 0.222 | 0.060 | 0.7637 | 0.7675 | +0.0038 |
| AGE | (60, 100] | 19691 | 0.115 | 0.049 | 0.7274 | 0.7340 | +0.0066 |
| GENDER | F | 112549 | 0.657 | 0.070 | 0.7650 | 0.7681 | +0.0031 |
| GENDER | M | 58649 | 0.343 | 0.101 | 0.7606 | 0.7635 | +0.0029 |
| INC_TYPE | Commercial associate | 40007 | 0.234 | 0.075 | 0.7610 | 0.7631 | +0.0021 |
| INC_TYPE | Pensioner | 30895 | 0.180 | 0.054 | 0.7402 | 0.7442 | +0.0040 |
| INC_TYPE | State servant | 12007 | 0.070 | 0.059 | 0.7800 | 0.7844 | +0.0044 |
| INC_TYPE | Working | 88268 | 0.516 | 0.095 | 0.7655 | 0.7684 | +0.0030 |
| TERM_HAT | -1 | 16214 | 0.095 | 0.056 | 0.7609 | 0.7606 | -0.0003 |
| TERM_HAT | 10 | 5754 | 0.034 | 0.067 | 0.7371 | 0.7390 | +0.0019 |
| TERM_HAT | 12 | 14120 | 0.082 | 0.047 | 0.7946 | 0.7974 | +0.0029 |
| TERM_HAT | 18 | 14712 | 0.086 | 0.138 | 0.7398 | 0.7438 | +0.0040 |
| TERM_HAT | 24 | 9501 | 0.055 | 0.103 | 0.7833 | 0.7880 | +0.0046 |
| TERM_HAT | 30 | 9854 | 0.058 | 0.095 | 0.7637 | 0.7672 | +0.0036 |
| TERM_HAT | 36 | 10480 | 0.061 | 0.070 | 0.7571 | 0.7590 | +0.0020 |
| TERM_HAT | 42 | 19542 | 0.114 | 0.084 | 0.7465 | 0.7486 | +0.0021 |
| TERM_HAT | 48 | 30241 | 0.177 | 0.072 | 0.7315 | 0.7350 | +0.0035 |
| TERM_HAT | 54 | 27869 | 0.163 | 0.086 | 0.7790 | 0.7818 | +0.0028 |
| TERM_HAT | 60 | 12915 | 0.075 | 0.076 | 0.7595 | 0.7640 | +0.0045 |
Comment: the largest gains are for age > 60 (+0.0066), 2 missing EXT (+0.0047), TERM_HAT 24/60 (+0.0046/+0.0045), state servants (+0.0044), pensioners (+0.0040). Revolving does NOT benefit (−0.0003: no product key). The weakest groups are still > 60 (0.7340), TERM_HAT 48 (0.7350), 10 (0.7390), < 30 (0.7422).

##### Methods ALREADY used (added in round 2 — Agent 1 must not propose them again)
Loan-product population profile (label-free group means by the keys TERM×RATE, RATE_n, C/A; train+test; 8/14/20/4 columns); nested target encoding of PRODUCT=round(C/A,2) at the meta level AND as a column for LGB/CatBoost;
PRODUCT as a native categorical for CatBoost (ordered TS) and for LGB (tried, dropped); extra categorical keys TERM×RATE / C÷GOODS / AMT_ANNUITY for CatBoost (tried, dropped);
implied rate with AMT_GOODS_PRICE as the principal (tried, dropped); EXT imputation with a second algorithm (Ridge on one-hot bins) and the average of two imputers (tried, dropped);
ablation of TERM_HAT / `_RES`; TE by INCOME_TYPE×EDU×age band (meta and level 1; tried, dropped); separate GBDTs for pensioners/>60, Revolving, <30, 2 missing EXT (tried, dropped);
KNN target as a meta column (tried, dropped); soft and hard pseudo-labels on test (tried, dropped); nested model-prediction profile (tried, dropped); EBM/GA2M `interpret` (tried, dropped);
multi-seed LGB 3 / CatBoost 2 / XGB 2; AUC-by-group table.

##### Surprising findings
1. The product signal is ONE signal, saturating at around +0.003: PROF, TE, PCAT and meta TE substitute for each other. Once level 1 has PROF, the meta TE column falls from +0.00155 to +0.00013.
2. CatBoost benefits the most (+0.0040 vs round 1, more than LGB/XGB at +0.0028) thanks to PRODUCT as a categorical (screening +0.0021 on top of PROF); the same variable makes LGB DROP by −0.0029 (0/4). Adding a second categorical key lowers the score again (R15 holds).
3. The PROF+TE screening for LGB was inconclusive (2/4), but on the full 5 folds at lr 0.01 it is positive 5/5 (+0.0003) — too small for a 4-evaluation screening to resolve; screening noise between equivalent variants ≈ ±0.0005.
4. The separate models for weak groups are worse than the general model by 0.005–0.011 within those very groups: the groups are weak because of missing information, not because the general model neglects them.
5. EBM reaches 0.76715 in only ~1 minute/fold but has correlation 0.980 with the GBDTs → adds nothing; lrbin (0.7585, corr 0.93) is still more useful despite being 0.009 weaker.
6. Every direction outside the product key is ≈ 0 (RATEG, second imputer, pseudo-labels, KNN meta, demographic TE, PP, EBM) — consistent with A1/A2's view that the 33 columns are nearly exhausted; realistic ceiling ≈ 0.771.

##### Lessons carried out (cross-checked against round2_agent2.md §6 and round1/2_agent1.md)
| Code | Direction / lesson | Status |
|---|---|---|
| L1.1 / L1.1b | EXT imputation + RATE_n; extra terms | DONE (round 1) |
| L1.1c | RATE with AMT_GOODS_PRICE as the principal; EXT imputation with a second algorithm then averaging | DONE (round 2): RATEG −0.00004 (2/4); Ridge PRED +3 columns −0.00001; LGB+Ridge average +0.00015 (2/4) → all three DROPPED |
| L1.2 | KNN target mean as a feature | DONE (round 1, dropped) |
| L1.2b | KNN as a meta-level column | DONE (round 2): +0.00002 … 0 → DROPPED. (KNN as a stand-alone model = exactly this column) |
| L1.3 | LR-binning, MLP | DONE (round 1); round 2 lrbin + PROF = 0.75851, +0.00014 in the stack → KEPT |
| L1.4 | Multi-seed for all 3 algorithms | DONE (round 2): LGB 3 seeds, CatBoost 2 seeds, XGB 2 seeds |
| L1.5a | Pseudo-labels on test | DONE (round 2, screening): soft −0.00003 (2/4); hard +0.00027 (4/4, below threshold) → DROPPED |
| L1.5b | Separate models/features for weak groups | DONE (merged into L2.4) |
| R1–R12 | Round-1 rules | APPLIED |
| L2.1 | TE PRODUCT (α=10) as a meta-LR column, nested per R14, with a test TE | DONE: on the round-1 models +0.00155 (0.76946, 5/5); on the round-2 models only +0.00013 … +0.00002 → not used at the meta level in the chosen version (TE lives inside LGB) |
| L2.2 | Product signal at level 1: (a) nested TE column for the GBDT; (b) PRODUCT categorical for CatBoost; compare configurations per R16 | DONE: (a) LGB screening +0.00212 alone / +0.00049 on top of PROF; full lgbte_r2 0.77008 (+0.00030 vs lgb_r2, 5/5) → KEPT; (b) CatBoost +0.00205 in screening, cat_r2 0.76947 → KEPT. Table of the 3 configurations in the Ensemble section |
| L2.3 | Fix the term reconstruction (prior/rate penalty, posterior/EM) | DONE by A1 in round 2 (conditional EM: −0.00001, 2/4) → noted, Agent 3 did not repeat it. The separate "penalize rates > 35–40%" version: NOT DONE (expected ≈ 0 because PROF/TE use the raw key that already captures the signal) |
| L2.4 | Weak groups at level 1: TE (INCOME×EDU×age) and per-group GBDTs | DONE: TE at meta −0.00002 (1/5), as an LGB column −0.00001 (2/4); separate GBDTs for 4 groups (pensioners/>60, Revolving, <30, 2 missing EXT) worse than the general model within the group by 0.005–0.011, in the stack −0.00010 … +0.00001 → DROPPED |
| L2.5 / R17 | Meta = simple LR | APPLIED |
| L2.6 / R18 | Do not add information-subset models to the stack | APPLIED (the 8-model r1+r2 version only +0.00009 → rejected; EBM corr 0.98 → rejected) |
| L2.7 | uid of repeat customers | NOT FEASIBLE in a useful way (measured by A2) |
| L2.8 / R20 | AUC-by-group table | DONE (table above) |
| L2.9 / R19 | Ablation of TERM_HAT, `_RES` | DONE: dropping TERM_HAT +0.00013 (3/4), dropping `_RES` +0.00005 (2/4), dropping both +0.00014 (3/4) → no effect, kept as is |
| R13–R16, R21 | Try TE at meta first; nesting per R14; one signal one column; report both configurations; ≤ 1 four-thread process | APPLIED |
| A1 round 1 #1 | Term/rate reconstruction; posterior/KDE version | DONE (round 1) + A1's EM version in round 2 (≈ 0) |
| A1 round 1 #2, #3, #4 | KNN / EXT imputation / date-passport | DONE (round 1) |
| A1 round 1 #5 | RealMLP / TabM (torch) | NOT DONE (needs torch ~3 GB; A1 round 2 noted that TabM/ModernNCA lose quality on CPU with a time limit) |
| A1 round 1 #6 | OOF TE for categorical combinations | DONE (round 2): PRODUCT useful; ORG×OCC (A2: −0.00005), INCOME×EDU×age (−0.00002) useless |
| A1 round 1 #7 | Null Importance | NOT DONE (the L2.9 ablation shows that dropping low-gain columns ≈ 0 → expected ≈ 0) |
| A1 round 1 #8 | uid | NOT FEASIBLE in a useful way (L2.7) |
| A1 round 1 #9 | LGB DART / extra_trees / linear_tree | NOT DONE |
| A1 round 1 #10 | TabICL / TabPFN | NOT FEASIBLE (needs a GPU) |
| A1 round 2 #1 | Loan-product population profile | DONE: screening +0.00261 (4/4); full LGB +0.0028, XGB +0.0029, Cat +0.0040 (with PCAT); stack +0.0029 |
| A1 round 2 #2 | Model-prediction profile (nested) | DONE (screening): −0.00007 (2/4) → DROPPED |
| A1 round 2 #3 | EBM / GA2M | DONE: 0.76715, corr 0.980, +0.00001 in the stack → DROPPED |

Still NOT DONE after round 2: A1 round 1 #5 (RealMLP/TabM), #7 (Null Importance), #9 (DART/extra_trees/linear_tree); the "penalize high rates" part of L2.3.

---

## ROUND 3 (FINAL ROUND)

### Agent 1 — Research of new methods (`loop/round3_agent1.md`)
STATUS: NO_NEW_METHODS

#### ROUND 3 — Agent 1 (Research): no new method with positive evidence found

Read: `rules.md`, `history.md` (rounds 0–2), `round1_agent1.md`, `round2_agent1.md`, `round2_agent2.md`, `fe_r2.py`, `run_r2.py`, `r2_screen.py`.
Baseline: round-2 stack = 0.77081. My scripts/logs: `$S/loop/a1r3/` (`base.py` screening harness, `q0.py`, `e1.py`, `mk2.py`+`e2.py`, `e3.py`, `ofe.py`+`e6.py`, `e5.py`; logs `log_e1/e2/e3/e5/e6.txt`, `log_ofe0.txt`; results `scr.json`).

##### One-line conclusion
I screened 9 directions NOT yet in the history/earlier reports (14 variants). None passed the R2 threshold (+0.0005 and ≥ 4/5): the best was +0.00025 (7/10 evaluations) — within the ±0.0005 screening noise recorded in round 2. I therefore report `NO_NEW_METHODS` rather than proposing something for the sake of it.

##### Measurement
The harness is identical to the round-2 R3 screening: BASE = IMP + RATE + TERM + PROF (129 columns), LGB lr 0.05 depth 4, fold 0/1 × seed 0/1, paired comparison.
BASE reproduces EXACTLY the numbers in `r2_screen.json` ('+PROF'): [0.76695, 0.76692, 0.76896, 0.76888], mean 0.76793.
The two most positive candidates were additionally scored on fold 2/3/4 × 2 seeds (10 paired evaluations in total).
Limitation: the machine was CPU-contended by Agent 2 throughout the session (load 11–19 on 4 cores), so no full 5-fold lr 0.01 run was possible; all numbers below are screening-level.

##### (2) Proposed new methods
None.

##### (3) Screened — all ≈ 0 or negative (do not repeat)

| # | Direction (why it is not a duplicate) | Result vs BASE | Positive | Conclusion |
|---|---|---|---|---|
| 1 | **"Pricing" residual relative to the customer profile** (`mk2.py`): target-free OOF LGB regression on train+test predicting RATE_HAT, log(C/A), log income and employment length from the customer columns → `_PRED` + `_RES` columns (8 columns). Different from IMP (which only imputes EXT) | fold 0/1: +0.00057 (4/4); fold 2/3/4: +0.00004 (3/6) → **pooled over 10: +0.00025, 7/10** | 7/10 | Below threshold. Both subsets alone are negative: RATE/C÷A only −0.00023 (0/4), income/employment only −0.00009 (2/4) → the initial +0.00057 was noise |
| 2 | **OpenFE** (Zhang et al., ICML 2023 — automatic feature generation + feature boosting; `ofe.py`). Features selected ONLY with the labels of fold 0's train part (an 80k-row sample, init_score = in-fold OOF logit of BASE), 477 candidates passed stage 2 | top 10: **−0.00048** (1/4); top 40: **−0.00032** (0/4) (fold 0 × 4 seeds) | 0–1/4 | DROP. The top features are meaningless combinations (e.g. `DAYS_REGISTRATION + AMT_REQ_YEAR`) → no useful second-order interaction is left |
| 3 | **Self-distillation / soft labels** (`e3.py`): inner 5-fold OOF within the train part, retrain on labels (1−α)·y + α·p, cross_entropy objective. Different from pseudo-labelling (which uses test) | α=0.3: +0.00006 (2/4); α=0.6: +0.00026 (3/4); α=0.9: −0.00145 (0/4); cross_entropy without distillation: +0.00011 (2/4) | — | DROP (below threshold, not monotone in α) |
| 4 | **Monotone constraints** on 11 EXT columns (decreasing), method `advanced` | −0.00001 | 2/4 | DROP |
| 5 | **path_smooth = 50** (smoothing leaf values towards the parent node) | fold 0/1: +0.00024 (3/4); fold 2/3/4: +0.00014 (4/6) → pooled +0.00018, 7/10 | 7/10 | Below threshold; harmless, could be enabled but does not count as an improvement |
| 6 | **max_bin = 1023** (so that trees can isolate each C/A product cluster) | +0.00010 | 2/4 | DROP — the product signal is already fully captured by PROF/TE |
| 7 | **GOSS** instead of bagging | −0.00022 | 2/4 | DROP |
| 8 | scale_pos_weight = 3 | −0.02197 | 0/4 | DROP (breaks early stopping/calibration at lr 0.05; not worth pursuing) |
| 9 | **Hidden weekday**: DAYS_* mod 7 (6 columns; documents are issued on working days ⇒ infer the application weekday) | −0.00002 | 1/4 | DROP |
| 10 | **Row order / SK_ID_CURR** (`q0.py`): ID = file order (0…171201 train, followed by test). Label autocorrelation at lag 1/2/5 ≈ −0.001; moving average of neighbouring labels (windows 50/500/5000) AUC 0.497–0.500; correlation of features with ID ≤ 0.008 | no signal | — | The data is shuffled; nothing to exploit (and no drift over time) |

Web search this round (Home Credit solutions using only the application table, GBDT/credit-scoring tips 2025–2026): no CPU-feasible technique that is not already in the log. Recent papers are all tabular foundation models (need a GPU at 171k rows) or resampling (ADASYN/SMOTE — the literature reports AUC benefits that are not statistically significant).

##### Priority of the OLD proposals still not done (not counted as new)
| Item | Priority | Reason |
|---|---|---|
| A1 round 1 #9 — LGB DART / extra_trees / linear_tree | Medium-low | Cheap, only for stack diversity; EBM (corr 0.98) gave +0.00001, so expected 0 … +0.0002 |
| A1 round 1 #5 — RealMLP / TabM | Low | Needs torch ~3 GB, the 4 CPU cores are contended; the sklearn MLP gave ≈ 0; expected 0 … +0.0005 in the stack |
| A1 round 1 #7 — Null Importance | Very low | The L2.9 ablation and the OpenFE experiment (adding columns → lower) both show that adding/removing columns ≈ 0 |
| L2.3 "penalize high rates" | Very low | max_bin 1023 and the RATE residual are both ≈ 0 → the product signal is saturated |

##### (4) Estimate of the AUC ceiling
- Three independent probes all show no remaining information: OpenFE (automatic search for second-order interactions over 39 core columns) is negative; the pricing residual ≈ 0; distillation/monotone constraints/smoothing (variance reduction) ≈ 0.
- Realistic ceiling of the 33 application columns with legitimate methods: **≈ 0.771 – 0.772** (currently 0.77081; what remains is only +0.0002 … +0.0008 from more seeds, path_smooth and diverse models).
- **0.80 is not achievable** with legitimate methods on this data: the gap of 0.029 is ~10× the gain of the best round, and round 3 found no further source of signal. The 0.80 level in the original Kaggle competition came from 6 history tables (bureau, previous_application, installments…) that are not available here.

### Agent 2 — Lessons from Round 2 (`loop/round3_agent2.md`)
#### ROUND 3 — Agent 2 (Lessons learned from round 2)

Sources: `rules.md`, `history.md` (rounds 0–2), `round2_agent1.md`, `round2_agent2.md`, `fe_r2.py`, `run_r2.py`, `r2_screen.py`, `ens_r2.py`, `ens_final_r2.py`, `log_r2_*.txt`, OOF `*_r2.npy`.
My analysis: `$S/a2r3/` — `c1.py`/`c1b.py` (OOF diagnostics, no training), `c2.py` (early-stopping optimism; LGB + PCAT), `c3.py` (combination TE at the meta level), `c_cat.py`/`c_cat1.py` (CatBoost variants), `c4.py`/`c5.py` (remaining-information harness; product key × credit-amount level); logs with the same names `.log`. The repository was not modified.
Baseline: LR stack (lgbte_r2 + cat_r2 + xgb_r2 + lrbin_r2) = **0.77081** (my reconstruction is exactly 0.77081, each fold matches history.md).

Notes: [MEASURED] = measured directly by me. [INFERRED] = reasoning not yet verified.
Reliability note: the machine was heavily CPU-contended (load 13–17) and restarted once midway; all numbers below come from runs that completed. CatBoost screening used only fold 0/1 and 1–2 seeds (large noise, see 3.4).

---------------------------------------------------------------------------------------------------
##### 0. Validity check (a specific requirement of this round)

| Item | Conclusion | Evidence |
|---|---|---|
| PROF (means by the TERM×RATE key, train+test) | NO label leakage | Read `fe_r2.py::profile`: it uses only EXT/age/employment/gender, no TARGET; test rows receive exactly the same columns → OOF and test share the same distribution. Legitimate under the constraints (using unlabelled test data). |
| IMP (EXT imputation) | NO label leakage | `fe_r1.py::impute`: regresses EXT_k on other columns, KFold over train+test, no TARGET. |
| TE PRODUCT in LGB | Correctly nested per R14 | `run_r2.py`: train a = inner 5-fold within a; valid b and test = from all of a; labels of b are not used. |
| PCAT in CatBoost | Legitimate | ordered target statistics; the valid set is encoded with train labels. The "rare" level is based on train+test counts (no labels). |
| Stack | Correctly nested | `ens_r2.py::stack`: meta-LR fitted on 4 folds, scored on the remaining fold; standardization within the fold. |
| **Early stopping on the scored fold itself** | **Small optimism, present since round 0** | [MEASURED] LGB lr 0.05, 5 folds: AUC at the best ES round 0.76899 vs AUC at the number of rounds = mean best_iter of the 4 OTHER folds 0.76874 → **+0.00025** (positive 5/5; version with TE: +0.00026). Every level-1 model (LGB/XGB/Cat, each seed) chooses its number of rounds on the scored fold. |
| Pooling OOF across folds | No optimism | [MEASURED] pooled AUC = mean fold AUC = AUC after ranking within each fold (lgbte 0.77008 / 0.77012 / 0.77012). Per-fold calibration: slope 0.96–1.03, mean p 0.079–0.081 vs a rate of 0.0807. |
| ID | No leakage | [MEASURED] SK_ID_CURR is already sorted ascending in the file; AUC(y ~ mean y of the ±1/5/50/500 ID neighbours) = 0.500/0.498/0.495/0.501; correlation of residuals with neighbouring residuals ≤ 0.005 → no time/batch signal by ID. |

Conclusion: there is no label leakage. The OOF 0.77081 is optimistic by about **0.0002–0.0003** because of early stopping (honest estimate ≈ 0.7705–0.7706). The bias is the same in every round, so comparisons between rounds remain valid; but it is of the same size as the +0.0002–0.0003 "improvements" under consideration → see R22.

##### 1. Round-2 failures & causes

| Experiment | Result | Cause |
|---|---|---|
| PRODUCT as a categorical for LGB | −0.0029 (0/4) | [MEASURED, `c2.py B`] I reproduced −0.00300 (0/4). With tightened categorical parameters (cat_smooth 1000, cat_l2 300, min_data_per_group 1000, max_cat_threshold 4) the harm disappears: **+0.00016 (2/4)**; at a medium level (300/100/300/8): −0.00016. → The cause is how LGB handles categoricals: it sorts ~2,000 levels by gradient statistics of the training data ITSELF and then splits greedily, with no "ordered" mechanism → each split is a target encoding that leaks within train. Tightening enough to remove the harm also removes the benefit → no LGB-categorical configuration is worth using. |
| A second categorical key for CatBoost (TERM×RATE, C/GOODS, AMT_ANNUITY) | −0.0007 … −0.0014 | The same product signal (R15); adding many-level variables increases the number of CTR combinations CatBoost must estimate → noise. Note these numbers were measured with 1 seed, noise ±0.001 (3.4) — the direction is right but the magnitude is uncertain. |
| EBM | 0.76715, +0.00001 in the stack | corr 0.980 with the GBDTs; the same function class (additive + pairwise interactions ⊂ shallow trees). |
| Per-group GBDTs | worse by 0.005–0.011 within their own group | The weak groups lack information; the general model can use structure learned from the other 82–90% of the data. |
| Pseudo-labels, RATEG, second imputer, demographic TE, KNN meta, PP | ≈ 0 | No new information (all are functions of existing columns). [MEASURED, `c4.py H`] A shallow LGB boosting further from the r2 stack logit using ALL 129 model columns: −0.0010 (50 rounds), −0.0017 (100), −0.0036 (300), 0/5 → the stack has already extracted everything a GBDT can extract from the current columns; only NEW COLUMNS can help. |
| XGB in the stack | coefficient 0.036 | [MEASURED] dropping xgb_r2: 0.77085, **+0.00004 (5/5)**. xgb–lgb Spearman 0.9977 → XGB is pure cost (2 seeds × 5 folds). |
| TE PRODUCT as an LGB column (lgbte) | +0.0003 over 5 folds | [MEASURED] At lr 0.05, 5 folds, 1 seed: PROF+TE 0.76885 vs PROF 0.76899 (−0.00014). The +0.0003 gain of lgbte_r2 is within seed noise; the real value of lgbte in the stack: dropping it −0.00024 (0/5), but partly because it is the only remaining LGB model. |

Methodological failures:
- CatBoost screening with 1 seed × 2 folds: [MEASURED] same configuration PROF+PCAT, seed 0 = 0.76694/0.76959, seed 1 = 0.76577/0.76877 → a difference of **0.0010** between two seeds. Any CatBoost conclusion of size < 0.001 from 1 seed cannot be resolved.
- The conclusion "the product signal saturates at around +0.003" was drawn when the variants differed only in key or number of columns; making the key FINER by credit amount had not been tried (see 3.2).

##### 2. Tips from the round-2 successes

1. **PROF works because it is "the risk grade the bank assigns to the product", measured by the very population that buys it.** No labels → no inner/outer shift (what killed KNN in round 1), it uses the extra 61.5k test rows, and it is a monotone numeric column that a shallow tree can cut with 1–2 splits (instead of ~2,000 narrow intervals on C/A).
   [MEASURED] Decomposition: AUC of "mean stack prediction per PRODUCT" = 0.691; AUC of the within-product deviation = 0.700; stack 0.7708 → about half of the discriminative power is BETWEEN products. PROF gain by product row count (cat_r2 − cat_r1): 30–200 rows **+0.0067**, 200–1000 +0.0040, > 1000 +0.0012, < 30 +0.0024 → the largest benefit is for medium-sized products (enough rows for a stable mean, narrow enough to characterize one segment).
2. **PCAT works for CatBoost through TWO mechanisms, and the second is larger** [MEASURED, `c_cat*.py`, lr 0.1, fold 0/1]:
   - the ordered TS of PRODUCT alone (`max_ctr_complexity=1`): 0.76689 (seed 0) / 0.76646 (seed 1) vs no PCAT 0.76621 → only about +0.0005;
   - allowing COMBINATIONS (default `max_ctr_complexity=4`): 0.76826 / 0.76727 → disabling combinations loses **−0.00138 (seed 0), −0.00081 (seed 1)**; 3/4 evaluations negative, 1 tie.
   CatBoost builds target statistics for PRODUCT × (other categoricals and the numeric splits already used in the tree) — i.e. "the risk of THIS product for THIS type of customer", which a single TE/PROF column cannot represent.
3. **The same variable hurts LGB** because LGB has no ordered statistics (section 1, row 1). General tip: a many-level categorical carrying label signal → native CatBoost; LGB/XGB only get the label-free form (PROF) or a nested TE.
4. **CatBoost is the only worthwhile source of diversity**: [MEASURED] dropping cat_r2 from the stack −0.00057 (the largest), dropping lgbte −0.00024, dropping lrbin −0.00015, dropping xgb +0.00004. Spearman cat–lgbte 0.982, lgb–xgb 0.998, lrbin–GBDT 0.92–0.93.
5. **Seed averaging helps CatBoost twice as much as LGB**: [MEASURED from the logs] fold AUC 1 seed → seed average: CatBoost 0.76910 → 0.76950 (**+0.00040** with 2 seeds); LGB 0.76961 → 0.76981 (+0.00020 with 3 seeds); XGB +0.00015. CatBoost still has a lot of seed variance left to remove.
6. **The default CatBoost parameters are good** [MEASURED, seed 0, vs PCAT 0.76826]: depth 4 −0.00056; l2_leaf_reg 20 −0.00050; rsm 0.3 −0.00117. (All within the ±0.001 noise, but none is positive → no reason to change.)

##### 3. Surprising findings (all [MEASURED])

1. **The product profile is NOT fully saturated — refining the key by credit amount still adds.** Group **PQ** = label-free profile (means of EXTF_MEAN, EXT_3, age, employment length; train+test; < 30 rows → NaN) by the key `TERM×RATE(0.5%) × AMT_CREDIT quintile`, 4 columns, added to PROF. LGB lr 0.05, **5 folds**, paired with the same seed (`c4.py S`, `c5.py`):

   | Seed | PROF | PROF + PQ | Per-fold difference | Mean | Positive |
   |---|---|---|---|---|---|
   | 0 | 0.76899 | 0.76947 | +0.00046 −0.00014 −0.00002 +0.00110 +0.00104 | +0.00049 | 3/5 |
   | 1 | 0.76870 | 0.76930 | +0.00007 +0.00111 +0.00120 +0.00031 +0.00032 | +0.00060 | 5/5 |
   | 2 | 0.76876 | 0.76939 | +0.00035 +0.00043 +0.00035 +0.00156 +0.00046 | +0.00063 | 5/5 |

   → **+0.00057, positive 13/15** for the single LGB model. WARNING: the harness on the r2 stack with only the 4 PQ columns = −0.0001 (on par with noise columns, −0.00015) → at the stack level the benefit may be much smaller (expected 0 … +0.0003). Not yet measured with 5 folds at lr 0.01, nor for CatBoost.
2. **At the meta level, the TE of the combination PRODUCT × AMT_CREDIT quintile is the best column** (`c3.py`, α = 10): on a stack without cat and without TE (0.76989): +0.00042 (5/5), vs plain TE PRODUCT +0.00023; on the full stack without xgb (0.77085): **+0.00013 (5/5) → 0.77098**. Other combinations on the full stack: × REGION_RATING +0.00009 (4/5), × EXT3_NA +0.00007 (5/5), × gender +0.00006 (5/5), × housing +0.00005 (5/5); × education / income type / family / occupation / age / EXT / insurance / employment ≈ 0. All below the R2 threshold → not used at the meta level, but they point in the same direction as PQ: **the credit amount further splits the product**.
   As an LGB column it does NOT help: PROF + TE(P×Cq5) −0.00006 (2/5); PROF + TE(P) + TE(P×Cq5) +0.00020 (2/5) → the label-free form (PQ) beats the TE form, repeating exactly the round-2 lesson PROF > TE.
3. **The stack has extracted all information from the current columns** (harness over all columns: −0.0010 … −0.0036, 0/5). Every "different model on the same columns" direction is expected at ≈ 0.
4. **CatBoost seed noise in screening is ±0.001** (section 1). Twice that of LGB screening (±0.0005).
5. **Dropping XGB IMPROVES the stack** (+0.00004, 5/5) — small but consistent; saves ~1/3 of level-1 time.
6. **CatBoost is clearly worse than LGB in some groups**: products with < 30 rows (merged into `rare`: −0.0033), TERM_HAT 60 (−0.0036), age > 60 (−0.0023), incomplete higher education (−0.0043); better than LGB at TERM_HAT 10/12/24 (+0.0005 … +0.0014), age 30–40 (+0.0008), products with > 1000 rows (+0.0006). So PCAT is strong for common products and weak for rare ones — consistent with ordered TS.
7. **Revolving still gets nothing**: cat_r2 − cat_r1 only +0.0006 (Cash +0.0043); a TE of the Revolving credit limit (AMT_CREDIT) at the meta level −0.00001. This group (9.5%) has no usable product key within the 33 columns.

##### 4. Combining techniques & logic optimization for round 3 — rules for Agent 3

Keep R1–R21. Additions:

- **R22 (report AUC without early stopping).** For the final configuration, in addition to the usual OOF, report the "fixed number of rounds" OOF (each fold uses the number of rounds = mean best_iter of the other 4 folds), at least for LGB. An improvement < 0.0003 between two configurations only counts as real if it is also positive in the fixed-rounds version.
- **R23 (CatBoost screening).** At least 2 seeds × fold 0/1 (4 paired evaluations with the same seed), acceptance threshold +0.001; 1 seed is not enough.
- **R24 (form of a group signal).** Priority order for a new group key: (1) a label-free train+test profile for every GBDT; (2) a native categorical for CatBoost (only ONE key); (3) a nested TE — only if (1) is not feasible. Never feed a many-level key as a categorical to LGB/XGB.
- **R25 (ensemble configuration).** LR stack = {LGB (with or without TE — the difference is within noise), CatBoost + PCAT with several seeds, lrbin}; NO XGB. Shift XGB's time budget to CatBoost seeds.

What adds up / does not add up (updated):
- ADDITIVE: IMP + RATE/TERM + PROF (three different sources); PROF + PCAT in CatBoost (+0.002 in screening, thanks to combinations); CatBoost + LGB in the stack (+0.0006); lrbin (+0.00015); CatBoost seeds (+0.0004 for 2 seeds).
- NOT ADDITIVE: PROF + TE + TE at meta (one signal); PROF + PROFX/PROFR/PROFC; XGB + LGB; EBM + GBDT; a second categorical key in CatBoost; previous-round models + this round's models.
- UNCLEAR (needs measuring at the stack level): PROF + PQ (single LGB +0.0006).

Suggested order for round 3: (1) screen PQ per R3 for LGB and per R23 for CatBoost → if kept, add it to the feature set; (2) rerun LGB (3 seeds) + CatBoost (5 seeds) + lrbin on the final set, without XGB; (3) LR stack, compare with 0.77081 fold by fold; (4) report the fixed-rounds OOF (R22); (5) the R20 group table.

##### 5. Diagnosis of the bottleneck & priority directions

The bottleneck is still INFORMATION. New evidence: the all-columns harness is negative; every model variant ≈ 0; the weak groups are unchanged (age > 60: 0.734; TERM_HAT 48: 0.735; < 30: 0.742; 2 missing EXT: 0.753; products with < 30 rows: 0.718).
Honest forecast: the directions below add up to an expected **+0.0003 … +0.0008** → ≈ 0.7711–0.7716. **0.80 is not achievable with legitimate methods on these 33 columns**; after three rounds there is no evidence to the contrary, and the remaining gap (0.029) is ~40× the total expected gain of round 3.

Priority directions (expectation / cost):
1. **L3.3 — PQ**: a label-free profile by the key TERM×RATE × AMT_CREDIT quintile (4 columns), for LGB and CatBoost. Single LGB +0.0006 (13/15). Cost: seconds + rerunning level 1.
2. **L3.2 — CatBoost with 5 seeds** (PROF + PCAT [+ PQ]). Expected +0.0002–0.0003 for the model, +0.0001–0.0002 for the stack. Cost ~3 min/seed/fold when the machine is idle.
3. **L3.1 — drop XGB from the stack** (+0.00004, 5/5; frees time for item 2).
4. **L3.4 — report the fixed-rounds OOF** (R22) so that the final number is honest.
5. **L2.3b — penalize high rates in the term reconstruction**: the only outstanding round-2 item; expected ≈ 0 (A1's EM version was already ≈ 0) — do it to close the checklist, one screening run.

##### 6. LESSON CHECKLIST

| Code | Actionable direction / lesson | Status |
|---|---|---|
| L1.1 | Model-based EXT imputation + RATE_n, run LGB/Cat/XGB fully, rank-avg | DONE (round 1: LGB 0.76696, XGB 0.76643, Cat 0.76545; rank-avg 0.76751) |
| L1.1b | Extra terms n=6,10,18,30,42; RATE by contract type | DONE (round 1, within TERM; +0.0021 in screening) |
| L1.1c | RATE with AMT_GOODS_PRICE as the principal; EXT imputation with a second algorithm then averaging | DONE (round 2: RATEG −0.00004 (2/4); Ridge −0.00001; LGB+Ridge average +0.00015 (2/4) → dropped) |
| L1.2 | Nested KNN target mean as a feature | DONE (round 1: −0.0005…−0.0007, 1/4 → dropped) |
| L1.2b | KNN as a meta-level column / separate model | DONE (round 2: +0.00002 … 0 → dropped) |
| L1.3 | Models from a different family: LR-binning, MLP | DONE (round 1: lrbin kept, MLP dropped; round 2: lrbin_r2 0.75851, +0.00014 in the stack → kept) |
| L1.4 | Multi-seed final configuration for all 3 algorithms | DONE (round 2: LGB 3 seeds, CatBoost 2 seeds, XGB 2 seeds) |
| L1.5a | Pseudo-labels on test | DONE (round 2: soft −0.00003 (2/4); hard +0.00027 (4/4, below threshold) → dropped) |
| L1.5b | Separate models/features for weak groups | DONE (round 2, merged into L2.4) |
| R1–R12 | Round-1 rules for evaluation/allocation/combination | DONE (rounds 1–2, applied) |
| L2.1 | TE PRODUCT (α=10) as a meta-LR column, nested per R14 | DONE (round 2: on the round-1 models +0.00155 (5/5); on the round-2 models +0.00013 … +0.00002 → not used at the meta level) |
| L2.2 | Product signal at level 1: (a) TE column for the GBDT; (b) PRODUCT categorical for CatBoost | DONE (round 2: (a) lgbte_r2 0.77008, +0.00030 vs lgb_r2, kept; (b) CatBoost +0.00205 in screening, cat_r2 0.76947, kept) |
| L2.3 | Fix the term reconstruction with posterior/EM | DONE (round 2 by A1: EM −0.00001, 2/4 → dropped) |
| L2.3b | Fix the term reconstruction with a penalty/prior on rates > 35–40%/year when choosing TERM_HAT; R3 screening on the PROF set (rebuilding TERM + PROF with the new key) | NOT DONE |
| L2.4 | Weak groups at level 1: TE (INCOME×EDU×age), per-group GBDTs | DONE (round 2: TE −0.00002 (1/5) / −0.00001 (2/4); separate GBDTs worse by 0.005–0.011 within the group, in the stack −0.00010 … +0.00001 → dropped) |
| L2.5 | Meta = simple LR (R17) | DONE (round 2, applied) |
| L2.6 | Do not add information-subset models to the stack (R18) | DONE (round 2: the 8-model version +0.00009 → rejected; EBM corr 0.98 → rejected) |
| L2.7 | Repeat-client matching (uid) | NOT FEASIBLE in a useful way (only 124 duplicate rows when EXT1 must match; duplicate groups with mixed labels) |
| L2.8 | AUC-by-group table every round (R20) | DONE (round 2, table in history.md) — continue every round |
| L2.9 | Ablation of TERM_HAT, `_RES` (R19) | DONE (round 2: +0.00013 (3/4), +0.00005 (2/4) → kept as is) |
| R13–R21 | Round-2 rules | DONE (round 2, applied) |
| L3.1 | Drop xgb_r2 from the final stack (stack = LGB + CatBoost + lrbin); do not rerun XGB | NOT DONE (A2's preliminary measurement: 0.77085, +0.00004, 5/5) |
| L3.2 | CatBoost PROF + PCAT with 5 seeds (instead of 2), fed into the stack | NOT DONE (basis: 2 seeds already gave +0.00040 vs 1 seed) |
| L3.3 | PQ group: label-free train+test profile (mean EXTF_MEAN, EXT_3, DAYS_BIRTH, DAYS_EMPLOYED; < 30 rows → NaN) by the key TERM×RATE(0.5%) × AMT_CREDIT quintile, added to PROF; R3 screening (LGB) and R23 (CatBoost); if kept, run all 5 folds and stack. Build code: `a2r3/c4.py` (variable `PQ`) | NOT DONE (A2's preliminary measurement with LGB lr 0.05, 5 folds × 3 seeds: +0.00057, 13/15; harness on the stack ≈ 0) |
| L3.4 | Report the "fixed number of rounds" OOF (rounds = mean best_iter of the other 4 folds) for the final models, at least LGB, and record it in history.md (R22) | NOT DONE (A2 measured: ES optimism +0.00025 for LGB at lr 0.05) |
| L3.5 | Nested TE of PRODUCT × AMT_CREDIT quintile: meta column and LGB column | DONE (round 3 by A2: meta +0.00013 (5/5) on the full stack, below threshold; LGB column −0.00006 (2/5) → not used; the direction became L3.3) |
| L3.6 | LGB + PRODUCT categorical with strong categorical regularization | DONE (round 3 by A2: −0.00300 by default; −0.00016 / +0.00016 (2/4) when tightened → not used) |
| L3.7 | CatBoost parameter variants on PROF + PCAT (max_ctr_complexity 1, depth 4, l2 20, rsm 0.3) | DONE (round 3 by A2, screening: −0.00138/−0.00081, −0.00056, −0.00050, −0.00117 → defaults kept) |
| L3.8 | TE of combinations PRODUCT × {gender, education, income type, family, region, EXT, age, insurance, occupation, housing, EXT3_NA, employment} and TE of the Revolving credit limit at the meta level | DONE (round 3 by A2: all ≤ +0.00009 on the full stack → not used) |
| L3.9 | Neighbour features by SK_ID_CURR (application order) | NOT FEASIBLE in a useful way (A2 measured: AUC 0.495–0.501, residual correlation ≤ 0.005) |
| R22–R25 | Round-3 rules (fixed-rounds OOF; CatBoost screening ≥ 2 seeds; priority of group-signal forms; ensemble without XGB) | NOT DONE (to be applied in round 3) |

Total: **6 rows NOT DONE** — L2.3b, L3.1, L3.2, L3.3, L3.4, R22–R25.

### Agent 3 — Implementation (reconstructed from code and logs)
> **Source note:** Agent 3 of round 3 was stopped by the model's weekly usage limit (13/11 ~08:00 UTC) **before writing its own report**. This whole section was reconstructed by the main session from the code (`fe_r3.py`, `run_r3.py`, `div_r3.py`, `tabm_r3.py`, `ens_final_r3.py`, `queue1/2/3_r3.sh`) and logs (`log_r3_*.txt`) that Agent 3 left behind; the submission-writing step (`ens_final_r3.py write`) was run by the main session.

##### Inputs
- Agent 1, round 3: `STATUS: NO_NEW_METHODS` → this is the final round under stop condition 2.
- Mandatory list: (A) Agent 2's checklist — L2.3b, L3.1, L3.2, L3.3, L3.4, R22–R25; (B) Agent 1's old items — LGB DART / extra_trees / linear_tree, Null Importance, RealMLP/TabM; (C) optional path_smooth = 50.

##### Code written
- `fe_r3.py`: `load_r3()` = round-2 features + the **PQ** group (4 columns: label-free profile by the key TERM × RATE(0.5%) × AMT_CREDIT quintile, computed on train+test, groups < 30 rows → NaN) + `term_alt()` rebuilding the term with a high-rate penalty (L2.3b). No column uses TARGET.
- `run_r3.py`: trains level 1 with a **fixed number of rounds** and saves the AUC curve so that **both** numbers can be reported (L3.4/R22): (ES) predictions at the simulated early-stopping round (patience 300 on the scored fold, as in rounds 0–2) and (FIX) predictions at the number of rounds = mean ES rounds of the other 4 folds → the scored fold's labels are not used. Flags: TE, PCAT, PS50, ET, LT, DART, LR, NR, NI.
- `div_r3.py` (lrbin), `tabm_r3.py` (TabM-mini + piecewise-linear embeddings, torch CPU; early stopping on 10% held out from the fold's train part), `ens_final_r3.py` (comparison of rank-avg / nested LR stacking and writing the submission).
- Restartable queues: `queue1_r3.sh`, `queue2_r3.sh`, `queue3_r3.sh` (skip what already exists; checkpoints per fold/seed in `r3m/`).

##### Screening (LightGBM lr 0.05, several seeds × folds)
`log_r3_screen1.txt` (4 evaluations):

| Configuration | Mean AUC | vs BASE | positive |
|---|---|---|---|
| BASE (IMP+RATE+TERM+PROF, 129 columns) | 0.76804 | — | — |
| PQ | 0.76842 | +0.00038 | 3/4 |
| T_soft35 (soft rate penalty, L2.3b) | 0.76836 | +0.00032 | 3/4 |
| T_hard40 (hard penalty, L2.3b) | 0.76817 | +0.00013 | 4/4 |
| T_soft35 + PQA | 0.76856 | +0.00052 | 3/4 |
| PS50 (path_smooth 50) | 0.76818 | +0.00014 | 3/4 |
| PQ + PS50 | 0.76869 | +0.00065 | 4/4 |
| ET (extra_trees) | 0.76743 | −0.00061 | 1/4 |
| LT (linear_tree) | 0.76730 | −0.00074 | 1/4 |

`log_r3_screen2.txt` (6 other evaluations, seeds 2–4):

| Configuration | Mean AUC | vs BASE | positive |
|---|---|---|---|
| BASE | 0.76938 | — | — |
| PQ | 0.77004 | +0.00066 | 5/6 |
| T_soft35 | 0.76902 | −0.00036 | 1/6 |
| T_soft35 + PQA | 0.77021 | +0.00083 | 5/6 |
| PQ + PS50 | 0.77004 | +0.00066 | 6/6 |
| PS50 | 0.76950 | +0.00012 | 4/6 |

→ PQ kept (stably positive). The rate penalty (L2.3b) flips sign between the two batches (+0.00032 → −0.00036) → dropped; the extra gain of T_soft35+PQA lies in PQ. PS50 gives only +0.00012–0.00014 → **not included in the final model** (lgbte_r3 uses the spec `+PROF+PQ~TE`, without PS50).

`log_r3_screen3.txt`:

| Configuration | Mean AUC | vs BASE | positive |
|---|---|---|---|
| NI0 (Null Importance threshold 0, keeps 88–92/129 columns) | 0.76737 | −0.00067 | 1/4 |
| NI1 (threshold 1, keeps 33–36/129 columns) | 0.76200 | −0.00604 | 0/4 |
| DART (lr 0.1) | 0.76677 | −0.00127 | 0/4 |

→ Null Importance dropped (it even removes useful variables such as AMT_CREDIT, AMT_ANNUITY, DAYS_BIRTH).

CatBoost screening with/without PQ (2 folds × 2 seeds, lr 0.1, `log_r3_catscrA/B.txt`): A (PROF+PCAT) ES 0.76694/0.76577/0.76959/0.76877; B (PROF+PQ+PCAT) 0.76577/0.76784/0.76765/0.76882 → mean −0.00025, 2/4 → **CatBoost does not use PQ**.

##### Full 5-fold training

| Model | Spec | Seeds | OOF ES | OOF FIX | Folds (ES) |
|---|---|---|---|---|---|
| lgbte_r3 | LGB `+PROF+PQ~TE`, lr 0.01 | 0,1,2 | 0.77048 | 0.77032 | 0.7677 / 0.77138 / 0.77611 / 0.7671 / 0.77032 |
| catn_r3 | Cat `+PROF~PCAT`, lr 0.03 | 2,3,4 | 0.76968 | 0.76939 | 0.76819 / 0.77007 / 0.77414 / 0.76613 / 0.77011 |
| cat_r3 = 0.4·cat_r2 + 0.6·catn_r3 (**5 seeds**, L3.2) | — | 0–4 | 0.76975 | — | 0.76838 / 0.77027 / 0.77402 / 0.76618 / 0.77008 |
| lrbin_r3 | LR on binned features (2074 columns) | — | 0.75917 | — | 0.75817 / 0.75886 / 0.76277 / 0.75541 / 0.76078 |
| dart_r3 | LGB DART lr 0.1, 1200 rounds | 0 | 0.76842 | 0.76842 | 0.76667 / 0.76892 / 0.77357 / 0.76527 / 0.76784 |
| et_r3 | LGB extra_trees lr 0.03 | 0 | 0.76960 | 0.76932 | 0.76642 / 0.77088 / 0.77413 / 0.76694 / 0.76982 |
| lt_r3 | LGB linear_tree lr 0.03 | 0 | 0.76963 | 0.76921 | 0.76639 / 0.76989 / 0.77506 / 0.76716 / 0.76973 |
| tabm_r3 | TabM seed 0 | 0 | 0.76654 | — | 0.76501 / 0.76843 / 0.77264 / 0.7636 / 0.76991 |
| tabmB_r3 | TabM seed 1 | 1 | 0.76785 | — | 0.76761 / 0.76844 / 0.77108 / 0.76519 / 0.76874 |
| tabmC_r3 | TabM seed 2 | 2 | 0.76575 | — | 0.76545 / 0.76779 / 0.7713 / 0.76403 / 0.76794 |
| tabm3_r3 = mean of 3 TabM seeds | — | 0,1,2 | 0.76887 | — | 0.7679 / 0.76923 / 0.77358 / 0.7661 / 0.77 |

Incidents during the runs:
- Installing torch from `download.pytorch.org` was blocked by the proxy (403) → torch 2.15.0 CPU installed from PyPI into `pylib_torch` (successful), packages `tabm` + `rtdl_num_embeddings` into `pylib_tabm`.
- The first TabM run failed in `compute_bins`: a column had a single value within a fold → fixed and rerun.
- Long commands were split by fold/seed with timeouts; several interrupted runs were resumed from checkpoints.

##### Ensembling (`log_r3_ens2.txt`, nested LR stacking; baseline = round-2 stack 0.77081)

| Set | OOF | vs r2 baseline | positive |
|---|---|---|---|
| r2 without XGB (L3.1) | 0.77085 | +0.00004 | 5/5 |
| rank_core3 (lgbte_r3, cat_r3, lrbin_r3) | 0.77050 | −0.00031 | 0/5 |
| **stack_core3** | 0.77124 | +0.00044 | 5/5 |
| stack_core3 + xgb_r2 | 0.77126 | (vs core3 +0.00002, 3/5) | — |
| stack_core3 + lgbte_r2, cat_r2 | 0.77124 | (vs core3 −0.00001, 2/5) | — |
| stack_core3 + tabm_r3 (1 seed) | 0.77130 | (vs core3 +0.00012, 4/5) | — |
| stack_core3 + dart_r3 | — | (vs core3 +0.00001, 2/5) | — |
| stack_core3 + et_r3 | — | (vs core3 +0.00016, 3/5) | — |
| stack_core3 + lt_r3 | — | (vs core3 +0.00014, 4/5) | — |
| stack_core3 + tabm3_r3 | — | (vs core3 +0.00024, 4/5) | — |
| **stack_core3 + alldiv (dart, et, lt, tabm3)** | **0.77149** | +0.00070 (vs core3 +0.00026) | **5/5** |
| rank_core3 + alldiv | 0.77144 | +0.00061 | 5/5 |
| ES stack (lgbte_r3, catn_r3, lrbin_r3) vs the corresponding FIX stack | — | FIX lower than ES by ≈ 0.00021 (0/5) | — |

(The earlier `log_r3_ens1.txt` version used TabM with 1 seed: stack_core3 + alldiv = 0.77143.)

Stack coefficients with 3-seed TabM (core3 + tabm3): lgbte_r3_fix 0.339, cat_r3_fix 0.307, lrbin_r3 0.118, tabm3_r3 0.318. Rank correlation (Spearman) with stack_core3: lgbte 0.9955, cat 0.993, lrbin 0.9438, dart 0.9834, et 0.9864, lt 0.986, tabm 0.9708, tabm3 0.9792.

##### Final result
- Chosen: **nested LR stacking over lgbte_r3 + cat_r3 + lrbin_r3 + dart_r3 + et_r3 + lt_r3 + tabm3_r3**.
- OOF AUC **0.77149** (round 2: 0.77081; +0.00070, 5/5 folds). Folds: 0.76923 / 0.77219 / 0.77622 / 0.76827 / 0.77190.
- "Fixed number of rounds" OOF (L3.4): level-1 FIX is lower than ES by 0.00016–0.00042 per model; the FIX stack is lower than ES by ≈ 0.0002 → honest OOF estimated at ≈ 0.7713.
- `loop/sub_round3.csv`: 61,500 rows, in sample_submission order, no NaN; Spearman with sub_round2 = 0.9977. Copied to `com/submission.csv`.

##### Status of each code

| Code | Content | Status |
|---|---|---|
| L2.3b | Penalize high rates in the term reconstruction | DONE (round 3): T_soft35 +0.00032 (3/4) then −0.00036 (1/6); T_hard40 +0.00013 → dropped |
| L3.1 | Drop XGB from the stack | DONE: r2 +0.00004 (5/5); in r3 adding xgb_r2 +0.00002 → XGB not used |
| L3.2 | CatBoost with 5 seeds | DONE: cat_r3 0.76975 vs cat_r2 0.76947 → kept |
| L3.3 | PQ group at level 1 | DONE: LGB +0.00066 (5/6, 4/4 together with PS50); lgbte_r3 0.77048 vs lgbte_r2 0.77008 → kept for LGB; CatBoost −0.00025 (2/4) → not used |
| L3.4 | Fixed-rounds OOF | DONE: table above |
| R22 | Report OOF without early stopping | APPLIED |
| R23 | CatBoost screening with ≥ 2 seeds | APPLIED (catscrA/B 2 seeds × 2 folds) |
| R24 | Priority of group-signal forms | APPLIED (PQ measured on top of PROF) |
| R25 | Ensemble without XGB | APPLIED |
| A1 #9 | LGB DART / extra_trees / linear_tree | DONE: individually worse than lgbte (0.76842 / 0.76960 / 0.76963) but they diversify the stack → kept in the stack |
| A1 #7 | Null Importance | DONE: −0.00067 / −0.00604 → dropped |
| A1 #5 | RealMLP / TabM | DONE (TabM; RealMLP not run separately since it plays the same role as a diversifying neural network): 3-seed TabM 0.76887, +0.00024 in the stack → kept |
| A1 R3 | path_smooth = 50 | DONE: +0.00012–0.00014 alone → below threshold, not used |

No item is left NOT DONE → **stop condition 2 is met; the loop ends.**

---

# PART III — Overall conclusion

| Round | AUC (5-fold OOF) |
|---|---|
| 0 | 0.76273 |
| 1 | 0.76790 |
| 2 | 0.77081 |
| 3 | **0.77149** (honest estimate ≈ 0.7713) |

- The 0.80 target was not reached. Every agent in every round estimated the legitimate ceiling of the 33 application columns at about 0.771–0.772; going beyond it requires more data (credit history, previous loans), which is not available in this task.
- The loop stopped under condition 2 (Agent 1 reported NO_NEW_METHODS and every checklist item had been carried out).

## Parts with NO original source / still missing
1. **Round 0** had no agents; its content is the results recorded in the shared log by the main session.
2. **Agent 3 of round 3** did not write its own report (stopped by the weekly usage limit); its section was reconstructed by the main session from the code and logs it left behind.
3. **RealMLP** (A1 #5) was never run separately; only TabM was run as the neural-network representative.
4. Status update (after this document was first written): the experiment code is now archived in `code/` on the branch `claude/cool-noether-usxp6b` (`code/vong0–3`, `code/tac_nhan_1`, `code/tac_nhan_2`), and the reproducible submission notebook is in `code/nop_bai/` (`home_credit_competition.ipynb`, English version `home_credit_competition_v2.ipynb`). The intermediate data files (OOF/test predictions, trained models) remain only in the session's scratch directory and will be lost when the session ends.

## Addendum — Session 7: code archive and submission notebook (16/11 – 20/11)
- **Code archive:** 86 original experiment scripts copied from the scratch directory into `code/`, organized by round (`vong0`–`vong3`) and by agent (`tac_nhan_1`, `tac_nhan_2`), with a `code/README.md`.
- **Submission notebook** (`code/nop_bai/`): a single Colab-runnable notebook rebuilt from the round-3 pipeline — basic features → EXT imputation → implied rate and term → product profile (PROF, PQ) → LightGBM (2 seeds, nested TE PRODUCT) + CatBoost (1 seed, PRODUCT_CAT) → logistic-regression stacking → `output/submission.csv` → `code.zip`.
- **Test run (4 CPUs, ~17 minutes):** LightGBM 0.77051, CatBoost 0.76870, stacking **0.77095** (per fold 0.76849 / 0.77182 / 0.77621 / 0.76778 / 0.77079); 0.0005 below the full round-3 ensemble because TabM and the LightGBM variants are omitted; rank correlation with the round-3 submission 0.9977.
- **Competition deadline:** 20/11/2026 01:00 UTC (submission file + code zip for Honors/Outstanding); the notebook was finished after that deadline.
