# Retention Risk Desk — Telecom Customer Churn Prediction

A churn-risk assessment tool for telecom operations staff. Instead of working with a raw dataset and a model notebook, a retention agent, customer-care executive or account manager answers a short set of plain-language questions about a customer and gets back:

- the **probability that the customer will churn**,
- a **risk band** (Low / Moderate / High / Critical) with a required next step,
- a **churn / retain classification** at a tuned business threshold,
- the **factors driving the score**, and
- **recommended retention actions** that match this customer's profile.

A second mode, **Portfolio batch**, scores a whole customer list from a CSV export so a retention team lead can size and prioritise a campaign.

---

## Quick start

- **Use it:** open `index.html` in any modern browser. No install, no server.
- **Host it:** the repo is a static site. Deploy it on Vercel or GitHub Pages as-is and share the link with your team.
- **Retrain it:** see [§12](#12-project-files-and-how-to-rebuild).

---

## Contents

1. [Business context](#1-business-context)
2. [Dataset](#2-dataset)
3. [From dataset columns to staff questions](#3-from-dataset-columns-to-staff-questions)
4. [Data preparation](#4-data-preparation)
5. [Modelling](#5-modelling)
6. [Model performance](#6-model-performance)
7. [Risk bands and decision policy](#7-risk-bands-and-decision-policy)
8. [Key churn drivers](#8-key-churn-drivers)
9. [Retention playbook](#9-retention-playbook)
10. [Using the tool](#10-using-the-tool)
11. [Batch scoring file format](#11-batch-scoring-file-format)
12. [Project files and how to rebuild](#12-project-files-and-how-to-rebuild)
13. [Limitations and responsible use](#13-limitations-and-responsible-use)
14. [Next steps for production](#14-next-steps-for-production)

---

## 1. Business context

**Churn** is when a subscriber ends their service and moves to a competitor or stops using telecom services. Acquiring a new subscriber typically costs several times more than keeping an existing one, so identifying at-risk customers *before* they leave lets the retention team act while the relationship can still be saved.

The tool answers one operational question:

> *Given what we know about this customer's account today, how likely are they to leave, why, and what should we do about it?*

**Intended users**

| Role | Typical use |
|---|---|
| Retention / save-desk agent | Score a customer during an inbound call or before an outbound save call |
| Customer-care executive | Check risk when a customer raises a complaint or asks about cancellation |
| Account / relationship manager | Review key accounts ahead of renewal |
| Retention team lead / CRM analyst | Score a customer list in batch, size a campaign, assign work by band |

---

## 2. Dataset

**IBM Telco Customer Churn** (IBM sample data, widely used as the reference telecom churn dataset).

| Property | Value |
|---|---|
| Customers | 7,043 (7,032 used — see §4) |
| Columns | 21 (1 ID, 19 attributes, 1 target) |
| Target | `Churn` = Yes / No (left within the last month) |
| Churn rate | **26.6 %** |
| Currency | USD (monthly charges range ≈ $18–$119) |
| Source | `github.com/IBM/telco-customer-churn-on-icp4d` → `data/Telco-Customer-Churn.csv` |

**Original columns**

| Group | Columns |
|---|---|
| Identity | `customerID` |
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Phone | `PhoneService`, `MultipleLines` |
| Internet | `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Target | `Churn` |

---

## 3. From dataset columns to staff questions

The main usability change is rewriting machine column names and codes as questions a telecom employee can answer straight from the CRM or billing screen. Questions are grouped in the order an agent would naturally look them up, with the strongest signals first.

### Section A — Account & relationship (strongest signals)

| Question shown to staff | Dataset column | Input |
|---|---|---|
| How long has the customer been with us? (months) | `tenure` | Slider + number, 1–72 |
| What contract term is the customer on? | `Contract` | Month-to-month / 1 year / 2 years |
| What is the monthly recurring charge? (USD) | `MonthlyCharges` | Slider + number, $18–$120 |

### Section B — Services subscribed

| Question | Column | Input |
|---|---|---|
| Does the customer have a phone line with us? | `PhoneService` | Yes / No |
| More than one phone line? | `MultipleLines` | Yes / No (disabled if no phone) |
| Which internet service is on the account? | `InternetService` | Fibre optic / DSL / None |
| Online security add-on | `OnlineSecurity` | Yes / No (disabled if no internet) |
| Online backup add-on | `OnlineBackup` | Yes / No (disabled if no internet) |
| Device protection plan | `DeviceProtection` | Yes / No (disabled if no internet) |
| Premium technical support | `TechSupport` | Yes / No (disabled if no internet) |
| Streaming TV package | `StreamingTV` | Yes / No (disabled if no internet) |
| Streaming movies package | `StreamingMovies` | Yes / No (disabled if no internet) |

### Section C — Billing

| Question | Column | Input |
|---|---|---|
| Payment method on file | `PaymentMethod` | Electronic cheque (manual) / Mailed cheque / Bank transfer (auto-pay) / Credit card (auto-pay) |
| Paperless (e-bill) billing? | `PaperlessBilling` | Yes / No |

### Section D — Household profile

| Question | Column | Input |
|---|---|---|
| Is the account holder a senior citizen (65+)? | `SeniorCitizen` | Yes / No (was coded 0/1) |
| Has a partner or spouse on record? | `Partner` | Yes / No |
| Has dependents on record? | `Dependents` | Yes / No |

### Usability decisions

- **Dependent questions are disabled automatically.** If "Internet service" is *None*, the six internet add-on questions grey out and are set to *No*. This removes the dataset's awkward `"No internet service"` / `"No phone service"` values, which confuse non-technical users.
- **Codes become words.** `SeniorCitizen` 0/1 → Yes/No; `Fiber optic` → Fibre optic; `Electronic check` → Electronic cheque (manual), so auto-pay vs manual payment is obvious.
- **17 questions instead of 20 columns.** `customerID` is not a predictor; `TotalCharges` and `gender` were removed (see §4).
- **Example profiles** (new fibre month-to-month, mid-tenure DSL, long-standing two-year) let a new user see the tool working in one click.
- **Live scoring.** There is no "Submit" button; the verdict updates on every change, so agents can try "what if we moved them to a one-year contract?" during a call.

---

## 4. Data preparation

| Step | What was done | Why |
|---|---|---|
| Fix `TotalCharges` type | Converted text to numeric | Stored as text in the raw file |
| Remove 11 rows | Customers with `tenure = 0` and blank `TotalCharges` | Brand-new accounts with no billing history; 0.16 % of data |
| Drop `TotalCharges` | Not used | ≈ `tenure × MonthlyCharges` (correlation > 0.8); adds collinearity and one more question with no accuracy gain |
| Drop `gender` | Not used | Coefficient was ≈ 0 (no predictive value), and using a protected attribute in customer-treatment decisions is poor practice |
| Collapse service values | `"No internet service"` / `"No phone service"` → `"No"` | The parent `InternetService` / `PhoneService` field already carries that information |
| Recode `SeniorCitizen` | 0/1 → No/Yes | Consistency with other binary fields |
| Add `log_tenure` | `log(1 + tenure)` | Churn risk falls steeply in the first year, then flattens. The log term captures that curve |
| One-hot encode | Each categorical level vs. the most common level | Required for logistic regression |
| Standardise | Z-score every feature before fitting | Makes the regularisation fair across features and makes coefficients comparable |

---

## 5. Modelling

### Approach

- **Split:** 80 % train / 20 % test, stratified on churn, `random_state=42`.
- **Validation:** 5-fold stratified cross-validation on the training set.
- **Candidates:** regularised logistic regression (L2, C = 0.5), random forest (400 trees, min leaf 5), gradient boosting (scikit-learn defaults).
- **Primary metric:** ROC-AUC, because the business uses the probability to *rank* customers for outreach.

### Why logistic regression was deployed

1. **It was the most accurate** on both cross-validation and the held-out test set (see §6).
2. **It is fully explainable.** Every score breaks down into per-factor contributions, which powers the "What is driving this score" panel and lets agents explain the outcome.
3. **It runs anywhere.** The model is 22 coefficients and an intercept. The scaling was folded into the raw coefficients, so the tool scores customers in the browser with no server and no customer data leaving the page.

### The scoring formula

```
z = intercept + Σ coefficient_i × feature_i
P(churn) = 1 / (1 + e^(−z))
```

Features: `tenure`, `log(1+tenure)`, `MonthlyCharges`, and 0/1 indicators for each non-baseline category level. The baseline (reference) profile is: month-to-month, fibre optic, electronic cheque, paperless billing, phone service, single line, no add-ons, not senior, no partner, no dependents.

### Explaining an individual score

For each question the tool computes

```
contribution = coefficient × (customer's value − average customer's value)
```

summed over all features belonging to that question (e.g. `tenure` + `log_tenure`). Positive contributions raise risk (red bar, right of centre), negative ones lower it (green bar, left). These add up exactly to the difference between this customer's log-odds and an average customer's, so the explanation is faithful to the model rather than approximated.

**Verification:** the browser implementation was checked against scikit-learn's `predict_proba` on sample customers and matched to four decimal places.

---

## 6. Model performance

### Model comparison (ROC-AUC)

| Model | 5-fold CV AUC | Test AUC |
|---|---|---|
| **Logistic Regression (deployed)** | **0.852 ± 0.005** | **0.840** |
| Random Forest | 0.846 ± 0.004 | 0.836 |
| Gradient Boosting | 0.847 ± 0.004 | 0.836 |

An AUC of 0.84 means that if you pick one customer who churned and one who stayed at random, the model ranks the churner higher about 84 % of the time. This is in line with published results on this dataset.

### Classification at the operating threshold (test set, 1,407 customers)

| Metric | Threshold 0.36 (deployed) | Threshold 0.50 (default) |
|---|---|---|
| Accuracy | 77.8 % | 79.5 % |
| **Recall** — churners caught | **71.4 %** | 53.7 % |
| Precision — flags that were right | 56.6 % | 63.6 % |
| F1 score | **0.631** | 0.583 |

**Confusion matrix at 0.36**

|  | Predicted: Retain | Predicted: Churn |
|---|---|---|
| **Actually stayed** | 828 | 205 |
| **Actually churned** | 107 | 267 |

**Why 0.36 and not 0.50?** The threshold was chosen to maximise F1. Lowering it from 0.50 to 0.36 catches **66 more churners** (267 vs 201) at the cost of 90 extra calls to customers who would have stayed. In telecom, losing a subscriber usually costs far more than one retention call or a modest discount, so favouring recall is the right business trade-off. Section 7 shows how the bands refine this further.

> Accuracy alone is misleading here: a model that predicts "nobody churns" scores 73 % accuracy and catches no one.

---

## 7. Risk bands and decision policy

The probability is grouped into four bands, each with a defined action. Calibration on the test set shows observed churn rises steadily across the bands, so they can be used directly for prioritisation.

| Band | Probability | Share of customers | Observed churn in band | Required action |
|---|---|---|---|---|
| 🟢 **Low** | 0 – 30 % | 60.3 % | 10.3 % | Stable. Maintain standard service; no intervention |
| 🟡 **Moderate** | 30 – 55 % | 21.8 % | 39.5 % | Watch list. Add to proactive engagement next cycle |
| 🟠 **High** | 55 – 75 % | 11.7 % | 55.8 % | Likely to churn. Contact within 14 days with a targeted offer |
| 🔴 **Critical** | 75 – 100 % | 6.2 % | 85.1 % | Imminent risk. Escalate to retention team within 48 hours |

**Reading it:** 18 % of customers (High + Critical) account for a large share of all churners, so a team that can only make a limited number of calls should work Critical first, then High.

The verdict panel also shows:

- **Model classification** — *Churn* if probability ≥ 36 %, otherwise *Retain*.
- **Relative to average customer** — probability ÷ 26.6 % base rate (e.g. "2.4×").
- **Observed churn in band** — what actually happened to similar-scored customers in testing.

---

## 8. Key churn drivers

### Strongest model factors (standardised coefficients)

| Factor | Effect on risk |
|---|---|
| Longer tenure | ↓↓ Strongest protective factor |
| Two-year contract | ↓↓ |
| No internet service | ↓↓ |
| DSL (vs fibre optic) | ↓ |
| One-year contract | ↓ |
| Multiple phone lines | ↑ |
| Not on paperless billing | ↓ |
| No phone service (internet only) | ↑ |
| Online security, tech support | ↓ |
| Auto-pay or mailed cheque (vs electronic cheque) | ↓ |
| Higher monthly charge | ↑ |
| Senior citizen | ↑ |

### Churn rate by segment (full dataset)

| Segment | Churn rate |
|---|---|
| **All customers** | **26.6 %** |
| Month-to-month contract | 42.7 % |
| One-year contract | 11.3 % |
| Two-year contract | 2.8 % |
| Tenure 0–6 months | 53.3 % |
| Tenure 7–12 months | 35.9 % |
| Tenure 13–24 months | 28.7 % |
| Tenure 25–48 months | 20.4 % |
| Tenure 49–72 months | 9.5 % |
| Fibre optic internet | 41.9 % |
| DSL internet | 19.0 % |
| No internet | 7.4 % |
| Electronic cheque payment | 45.3 % |
| Credit card auto-pay | 15.3 % |
| No tech support | 31.2 % |
| With tech support | 15.2 % |
| Senior citizen | 41.7 % |
| Paperless billing | 33.6 % |

### Business interpretation

1. **Commitment is the biggest lever.** Month-to-month customers churn 15× more than two-year customers.
2. **The first six months are the danger zone.** More than half of brand-new customers leave; risk falls steadily with every year of tenure.
3. **Fibre optic customers churn heavily** despite the premium product, which points to price-to-value or reliability issues worth investigating.
4. **Manual electronic-cheque payers churn about 3× more** than auto-pay customers. Every month they make an active decision to pay.
5. **Support and security add-ons protect.** They raise perceived value and make switching harder.

---

## 9. Retention playbook

The tool recommends actions based on which factors apply to *this* customer, ranked by how much each factor is pushing their risk up. Up to four are shown.

| Trigger | Tag | Recommended action |
|---|---|---|
| Month-to-month contract | CONTRACT | Offer a 12- or 24-month plan with a loyalty discount or device upgrade |
| One-year contract | RENEWAL | Approach before the anniversary with a two-year upgrade offer |
| Tenure ≤ 12 months | ONBOARD | Early-life care: welcome check-in, confirm service meets expectations |
| Electronic cheque payment | BILLING | Offer a small monthly credit for switching to auto-pay |
| Internet, no tech support | SUPPORT | Three-month free trial of premium tech support |
| Internet, no online security | BUNDLE | Include online security in a value bundle |
| Fibre optic internet | QUALITY | Review outages, speed tests and support tickets |
| Monthly charge ≥ $80 | PRICING | Right-size the plan against actual usage, or apply a bundle discount |
| Senior citizen | CARE | Named contact or priority line; simplify the bill |
| Paperless billing raising risk | ENGAGE | App notifications with usage and value summaries |
| None of the above | MAINTAIN | Keep service quality high; consider referral or loyalty reward |

These are starting points. Each operator should map them to its actual offers, discount limits and approval rules.

---

## 10. Using the tool

### Single customer

1. Open the **Single customer** tab.
2. Answer the questions in sections A–D from the customer's account. Use the example chips to see typical profiles.
3. Read the **verdict panel** on the right (below the form on mobile):
   - large percentage = churn probability, coloured by band;
   - the marker on the bar shows where the customer sits; the dashed line is the 26.6 % average;
   - the instruction under it is the required next step for that band.
4. Check **What is driving this score** to understand *why*.
5. Use **Recommended retention actions** to plan the conversation. Change an answer (for example, switch the contract to "1 year") to see how much an offer would reduce risk.
6. Click **Copy case summary** and paste it into the CRM case notes.

**Example case summary**

```
CHURN RISK ASSESSMENT
Risk: 73.1% (High) | Classification: Churn
Profile: 8 months, Month-to-month, Fiber optic internet, $89.10/mo, Electronic check
Next step: Likely to churn. Contact within 14 days with a targeted offer.
Actions:
  1. Offer a 12- or 24-month plan
  2. Run early-life care
  3. Move to auto-pay
```

### Portfolio batch

1. Open the **Portfolio batch** tab (a 40-customer sample loads automatically).
2. Click **Upload CSV** or drag a file onto the drop zone.
3. Review the four band cards: customer count, share of the list, and monthly revenue at stake per band.
4. Click a column header to sort; the default sort is highest risk first.
5. Click **Copy scored results** and paste into Excel or Google Sheets to assign work.

### Model & method

Shows model comparison, threshold performance, band calibration and top-risk segments for managers or auditors.

---

## 11. Batch scoring file format

A CSV with a header row using the **original Telco column names**. Extra columns are ignored.

| Column | Required | Accepted values |
|---|---|---|
| `customerID` | Recommended | Any text (row number used if missing) |
| `tenure` | **Yes** | 1–72 (months) |
| `Contract` | **Yes** | `Month-to-month`, `One year`, `Two year` |
| `MonthlyCharges` | **Yes** | Number |
| `InternetService` | Recommended | `DSL`, `Fiber optic`, `No` |
| `PaymentMethod` | Recommended | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` |
| `PaperlessBilling`, `PhoneService`, `MultipleLines`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Partner`, `Dependents` | Recommended | `Yes`, `No` (also accepts `No internet service` / `No phone service`) |
| `SeniorCitizen` | Recommended | `Yes`/`No` or `1`/`0` |
| `Churn` | Optional | `Yes`/`No`; shown as "Actual" for back-testing |

Missing or unrecognised values fall back to the most common category, so check column spelling if results look off.

**Output columns (Copy scored results):** `customerID, churn_probability, risk_band, top_driver, recommended_action`.

---

## 12. Project files and how to rebuild

```
Telecom-Churn/
├── index.html                    # the tool: self-contained, open in any browser
├── README.md                     # this document
├── requirements.txt              # Python dependencies for retraining
├── data/
│   └── telco.csv                 # IBM Telco Customer Churn (raw)
├── src/
│   ├── train.py                  # cleaning, model comparison, export
│   ├── template.html             # tool UI (model injected at build)
│   └── build.py                  # injects model + metrics + sample -> index.html
└── out/
    ├── model.json                # intercept, coefficients, levels, threshold
    ├── metrics.json              # AUCs, confusion matrix, bands, segments
    └── sample_batch.csv          # 40 customers for the batch demo
```

**Requirements:** Python 3.10+, `pandas`, `numpy`, `scikit-learn`.

```bash
pip install -r requirements.txt

# 1. Train, evaluate and export the model
python src/train.py data/telco.csv out

# 2. Build the tool (writes index.html)
python src/build.py
```

To retrain on your own operator's data, keep the same column names (or map yours to them), rerun both steps, and the tool picks up the new coefficients, threshold, bands and segment rates automatically.

### `model.json` structure

```json
{
  "intercept": 1.8467,
  "coef":      { "tenure": ..., "log_tenure": ..., "Contract=Two year": ..., ... },
  "std_coef":  { ... standardised coefficients for importance ... },
  "means":     { ... feature means, used for driver explanations ... },
  "levels":    { "Contract": ["Month-to-month","One year","Two year"], ... },
  "baseline":  { "Contract": "Month-to-month", ... },
  "threshold": 0.36,
  "base_rate": 0.2658
}
```

---

## 13. Limitations and responsible use

- **Sample data.** The model is trained on IBM's sample dataset of a fictional US operator. Before using it for real decisions, retrain and validate on your own customer history.
- **Snapshot, not timeline.** The data is one point in time. It does not use call-centre complaints, network quality, usage trends, competitor offers or recent price changes, which are often strong churn signals in practice.
- **Correlation, not causation.** "Two-year contract customers churn less" partly reflects that loyal customers choose long contracts. Moving someone onto a contract helps, but not by the full amount the model implies. Validate offers with controlled tests.
- **A decision aid, not a decision.** The score supports an agent's judgement; it should not automatically deny service, change pricing, or treat customers differently on its own.
- **Fairness.** Gender was excluded. `SeniorCitizen` remains because it is predictive and its use here is to offer *more* support (the CARE action). Review this under local regulation and company policy.
- **Calibration drift.** Customer behaviour changes. Monitor actual churn by band monthly and retrain when the bands stop separating cleanly.
- **Privacy.** The web tool scores entirely in the browser; uploaded CSVs are not sent anywhere. Still follow your data-handling policy for exports that contain customer IDs.

---

## 14. Next steps for production

1. **Retrain on internal data** with 12–24 months of history and a clear churn definition (e.g. disconnected or ported out within 90 days).
2. **Add behavioural features**: complaint and ticket counts, dropped-call rate, data-usage trend, days since last plan change, late payments, competitor-area flags.
3. **Integrate with CRM** so the score and drivers appear on the customer screen and refresh nightly.
4. **Measure uplift** with A/B tests (offer vs. no offer per band) to learn which actions actually save customers and at what cost.
5. **Optimise for value**, not only probability: rank by *churn probability × customer lifetime value* so the team spends effort where the most revenue is at stake.
6. **Monitor** AUC, band calibration and save-rate monthly, with a retraining trigger.

---

*Dataset: IBM Telco Customer Churn. Model: L2-regularised logistic regression (scikit-learn 1.9). Tool: single self-contained HTML page, no server required.*
