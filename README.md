# AI Development Workflow â€” Student Assignment

This repository contains:
- Concise answers to the AI Development Workflow assignment (student dropout + hospital readmission).
- A minimal Streamlit app to train and run two example models (student dropout RandomForest, hospital readmission GradientBoosting).
- Deployment files (Dockerfile, Procfile).

## Contents
- Part 1: Predicting Student Dropout Rates (problem definition, data, model, evaluation)
- Part 2: Hospital Readmission Case Study (scope, data strategy, model, deployment)
- Part 3: Critical Thinking (ethics & bias, trade-offs)
- Part 4: Reflection & Workflow Diagram
- How to run the Streamlit app and deploy

---

## Part 1 â€” Predicting Student Dropout Rates

### 1.1 Problem Definition
- Objectives
  1. Achieve recall â‰¥ 80% on a held-out test set within one academic term for students who eventually drop out.
  2. Reduce semester dropout rate by 15% within 12 months among students who receive targeted interventions based on model output.
  3. Maintain false-positive rate â‰¤ 20% to limit unnecessary interventions.
- Key stakeholders
  - Academic advisors / student support services.
  - Registrar / institutional analytics team.
- KPI
  - Intervention-adjusted retention lift: percent decrease in dropout rate among flagged+intervened students vs matched control.

### 1.2 Data & Preprocessing
- Data sources
  1. Student Information System (demographics, enrollment, GPA, course grades, financial aid).
  2. Learning Management System logs (logins, submissions, forum activity).
- Potential bias
  - Socioeconomic bias: lower-income students may have less LMS activity or missing records, causing underestimation of their risk and unequal intervention allocation.
- Essential preprocessing steps
  1. Missing-data handling: impute and add missingness indicators (missingness can be informative).
  2. Encoding categorical variables: one-hot or target encoding for major, course codes.
  3. Scaling/normalization for numeric features: stabilize training and make features comparable.

### 1.3 Model Development
- Model choice: Random Forest
  - Reason: Tabular, mixed feature types, robust to outliers, interpretable feature importance, good for modest dataset sizes.
- Data split
  - Train 70% / Validation 15% / Test 15%; use temporal split if cohorts change over time.
- Hyperparameters to tune
  1. n_estimators â€” affects stability/variance.
  2. max_depth (or min_samples_leaf) â€” controls overfitting vs underfitting.

### 1.4 Evaluation & Deployment
- Metrics (beyond accuracy)
  - Recall (sensitivity): capture as many true at-risk students as possible.
  - Precision or F1: control false positives to keep interventions cost-effective.
- Concept drift
  - Definition: change in the relationship between inputs and labels over time (policy changes, remote learning).
  - Monitoring: track validation/test metrics in production and data-distribution measures (PSI); trigger retraining if thresholds exceeded.
- Deployment challenge & solution
  - Challenge: low-latency integration with advising dashboards.
  - Solution: nightly batch scoring + lightweight REST endpoint for on-demand inference; cache results.

---

## Part 2 â€” Hospital Readmission Case Study

### 2.1 Problem Scope
- Problem statement
  - Predict 30-day unplanned readmission risk at discharge to enable targeted interventions and reduce avoidable readmissions.
- Objectives
  1. Achieve recall â‰¥ 75% for 30-day readmissions on validation.
  2. Reduce 30-day readmission rate by a stakeholder-defined % among flagged patients receiving interventions within 6 months.
- Stakeholders
  - Clinicians / discharge planners / case managers.
  - Hospital quality improvement / administration.
  - Patients and patient advocates.

### 2.2 Data Strategy
- Data sources
  1. EHR structured data: diagnoses (ICD), meds, labs, vitals, procedures.
  2. Claims or utilization history: prior admissions, ED visits.
  3. Clinical notes/discharge summaries (NLP): SDOH mentions and discharge barriers.
- Ethical concerns
  1. Patient privacy (HIPAA): PHI exposure risks; critical because legal and harm implications.
  2. Algorithmic bias/equity: historical disparities can produce unequal recommendations and worsen outcomes.
- Preprocessing pipeline
  1. De-identify where possible and enforce role-based access.
  2. Clean and standardize codes/timestamps; normalize lab units.
  3. Impute missing values and add missingness flags.
  4. Feature engineering: number of prior admissions in last 12 months; Charlson comorbidity index; lab trend features (slopes).
  5. NLP extraction from notes: SDOH flags, discharge instruction clarity.

### 2.3 Model Development
- Model selection: Gradient Boosting (e.g., XGBoost / LightGBM)
  - Reason: high performance on tabular data, handles missing values, supports regularization and feature importance.
- Confusion matrix (TP=40, FP=15, FN=30, TN=115)
  - Precision = 40 / (40 + 15) = 40/55 â‰ˆ 0.727 (72.7%)
  - Recall = 40 / (40 + 30) = 40/70 â‰ˆ 0.571 (57.1%)
- Interpretation
  - Precision: ~73% of flagged patients actually readmit â€” important to minimize wasted resources.
  - Recall: ~57% of actual readmissions are caught â€” indicates many readmissions are missed and could need model improvement.

### 2.4 Deployment & Optimization
- Integration steps
  1. Expose model as a secure API (FHIR/REST) and/or batch scoring for discharge lists.
  2. Embed risk scores and recommended actions into EHR discharge workflows and case manager dashboards.
  3. Train staff, define SOPs, and capture feedback/labels for continuous improvement.
- HIPAA compliance
  1. Encrypt PHI at rest and in transit; ensure secure key management.
  2. Role-based access control, audit logging, and BAAs with vendors.
- Overfitting mitigation
  - Early stopping with cross-validation and L2 regularization for gradient boosting: prevents over-complex ensembles and improves generalization.

---

## Part 3 â€” Critical Thinking

### 3.1 Ethics & Bias
- Example harm from biased data
  - If historical data shows fewer admissions for underserved groups due to access barriers, model may predict lower risk for those groups and under-allocate post-discharge resources, increasing harm.
- Mitigation strategy
  - Data rebalancing and fairness-aware evaluation: stratified sampling/augmentation, monitor group-specific TPR/PPV, and apply threshold adjustments or reweighting.

### 3.2 Trade-offs
- Interpretability vs Accuracy
  - Interpretability is crucial for clinician trust, accountability, and actionable reasoning. If a black-box model is more accurate but opaque, use explainability tools (SHAP) or prefer a transparent model if performance gap is small.
- Resource constraints
  - Limited compute favors simpler models (logistic regression, small trees). Pros: lower latency/cost and easier auditing. Cons: potentially lower predictive performance; mitigate via feature engineering or model distillation.

---

## Part 4 â€” Reflection & Workflow Diagram

### 4.1 Reflection
- Most challenging part
  - Ethics and fairness: requires institutional change, high-quality subgroup data, and stakeholder engagement beyond technical metrics.
- Improvement with more resources
  - Conduct a prospective pilot/RCT with clinician-in-the-loop validation and richer SDOH data collection.

### 4.2 Workflow (flowchart text)
1. Problem definition â†’ 2. Stakeholder & feasibility review â†’ 3. Data inventory & access â†’ 4. Governance & ethics review â†’ 5. Data preprocessing & feature engineering â†’ 6. Model selection & training â†’ 7. Evaluation & fairness testing â†’ 8. Deployment planning â†’ 9. Integration & user training â†’ 10. Monitoring, drift detection & retraining â†’ 11. Continuous governance & improvement.

---

## Run the app
1. Activate venv: .venv\Scripts\activate
2. Install: pip install -r requirements.txt
3. Run: streamlit run streamlit_app.py