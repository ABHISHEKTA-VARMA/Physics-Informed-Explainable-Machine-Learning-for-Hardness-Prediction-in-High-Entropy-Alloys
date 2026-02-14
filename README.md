# Physics-Informed Explainable Machine Learning for Hardness Prediction in High-Entropy Alloys

A structured computational workflow for predicting hardness in high-entropy alloys using composition-derived descriptors, machine learning, explainability analysis, and finite element validation. The pipeline integrates data preprocessing, physics-informed feature construction, model benchmarking, and simulation input generation within a reproducible framework.

## Pipeline Overview

### Step 1 — Dataset Preparation and Composition Parsing

The raw experimental alloy dataset is cleaned, audited for missing values, and parsed to extract normalized elemental compositions. Elemental fractions are explicitly encoded, and dataset coverage statistics are recorded to ensure traceability and reproducibility.

### Step 2 — Composition-Based Baseline Modeling

Hardness prediction models are trained using only elemental fraction features. Multiple algorithms are evaluated using cross-validation to establish a reference performance level before introducing physics-based descriptors.

### Step 3 — Physics-Informed Descriptor Construction

A set of physically motivated descriptors is generated from elemental compositions using literature-based property proxies. These include atomic size statistics, thermodynamic mixing indicators, valence electron concentration, and elastic property proxies. The resulting dataset forms the primary input for descriptor-driven modeling.

### Step 4 — Descriptor-Based Modeling and Benchmarking

Machine learning models are trained using the constructed descriptors. Cross-validated evaluation is performed across multiple regressors, with hyperparameter optimization applied to XGBoost. Performance metrics and comparison plots are generated for consistent analysis.

### Step 4.1 — Dataset Metrics and Exploratory Visualization

Supplementary visual analysis is performed to understand dataset characteristics and modeling context. This includes hardness distribution plots, elemental occurrence frequency, and comparison of top-performing models using stored evaluation metrics.

### Step 5 — SHAP Explainability and Redundancy Analysis

Model interpretability is performed using SHAP (SHapley Additive exPlanations). Global feature importance rankings are computed, dependence plots are generated for key descriptors, and redundancy analysis is conducted using correlation matrices and variance inflation factors.

### Step 6 — ANSYS Input Generation

Descriptor-derived elastic proxies are used to generate physically consistent input parameter sets for finite element simulations. Contact indentation and anisotropic elasticity cases are prepared in SI units using representative elastic statistics.

### Step 6.1 — Mesh Convergence Assessment

Simulation outputs obtained at multiple mesh resolutions are analyzed to assess numerical stability. Convergence behavior is evaluated using clustered mesh grouping, consistency checks across refinement levels, and Richardson-style extrapolation. Results are visualized to verify mesh-independent responses.

### Step 7 — Descriptor Selection for Statistical Analysis

The top descriptors identified through SHAP importance are compiled into a structured dataset for downstream statistical exploration. Predictors are standardized, while the target hardness variable is retained in physical units. This dataset supports correlation studies, multicollinearity diagnostics, and factor-structure analysis.

## Data Availability

The datasets used in this study are compiled from experimental literature sources.
To maintain review integrity, raw and intermediate datasets are not included at this stage.

All processed datasets will be made publicly available upon publication.
