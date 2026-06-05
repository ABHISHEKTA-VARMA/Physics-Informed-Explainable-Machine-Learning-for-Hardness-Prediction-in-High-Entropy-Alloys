# Physics-Informed Explainable Machine Learning Framework for Hardness Prediction and Virtual Design of High-Entropy Alloys

## Introduction

High-entropy alloys (HEAs) occupy a vast compositional space, making comprehensive experimental exploration both time-consuming and resource-intensive. This repository contains the complete computational workflow developed for hardness prediction, candidate alloy discovery, and simulation-driven verification using materials informatics, machine learning, explainability analysis, and finite element modelling.

The workflow begins with dataset curation and quality control, followed by composition-based and descriptor-based machine learning. Physics-informed descriptors are constructed from elemental properties to represent compositional, thermodynamic, electronic, and elastic characteristics of alloys. Model reliability is evaluated through uncertainty quantification, applicability-domain assessment, and algorithm consistency analysis. The resulting models are subsequently used for alloy screening, candidate optimization, digital material card generation, ANSYS material card preparation, fatigue assessment, SHAP-based interpretation, and mesh convergence verification.

Each stage of the workflow is implemented as a separate module, allowing the complete methodology to be reproduced, inspected, and validated in a transparent and systematic manner.

---

## Workflow Overview

```text
Dataset Curation
        ↓
Baseline Composition-Based ML
        ↓
Universal Property Database
        ↓
Physics-Informed Descriptor Generation
        ↓
Descriptor-Based ML
        ↓
Algorithm Consistency Analysis
        ↓
Global Exploration
        ↓
Adaptive Optimization
        ↓
Digital Material Card Generation
        ↓
ANSYS Material Card Generation
        ↓
Fatigue S–N Curve Generation
        ↓
SHAP Interpretability
        ↓
Mesh Sensitivity and Convergence Analysis
```

---

## Methodology

### Step 01 — Dataset Curation

Preparation of the hardness dataset through composition standardization, duplicate removal, quality control, dataset verification, and reproducible dataset versioning.

### Step 02 — Baseline Composition-Based Machine Learning

Development of composition-based predictive models to establish reference performance and assess the predictive information contained within alloy chemistry alone.

### Step 03A — Universal Property Database

Compilation of elemental properties used throughout the study, including physical, thermodynamic, electronic, and elastic attributes required for descriptor construction.

### Step 03B — Physics-Informed Descriptor Generation

Calculation of composition-derived descriptors using statistical aggregation operators and physically meaningful elemental properties.

### Step 04A — Descriptor-Based Machine Learning

Training and evaluation of descriptor-driven machine learning models for hardness prediction, including uncertainty quantification and applicability-domain assessment.

### Step 04B — Algorithm Consistency Analysis

Evaluation of model robustness through seed-stability studies, permutation testing, consistency assessment, and reproducibility checks.

### Step 05A — Global Exploration

Systematic exploration of the alloy design space to identify promising candidate compositions satisfying predefined constraints.

### Step 05B — Adaptive Optimization

Refinement and ranking of candidate alloys through iterative screening and objective-based selection.

### Step 05C — Digital Material Card Generation

Generation of structured alloy summaries containing predicted mechanical properties, descriptor information, and candidate rankings.

### Step 06A — ANSYS Material Card Generation

Conversion of selected alloy candidates into simulation-ready material definitions suitable for finite element analysis.

### Step 06B — Fatigue S–N Curve Generation

Estimation of fatigue behaviour and generation of S–N curves for selected virtual alloy candidates.

### Step 07 — SHAP Interpretability

Investigation of descriptor contributions and model behaviour using SHAP-based global and local explainability analysis.

### Step 08 — Mesh Sensitivity and Convergence Analysis

Assessment of mesh independence for indentation, fatigue, and tensile simulations to support finite element verification.

---

## Repository Structure

```text
src/

Step_01_Dataset_Curation.py
Step_02_Composition_Baseline_ML.py

Step_03A_Elemental_Property_Database.py
Step_03B_Physics_Informed_Descriptor_Generation.py

Step_04A_Descriptor_Based_ML.py
Step_04B_Algorithm_Consistency_Analysis.py

Step_05A_Global_Exploration.py
Step_05B_Adaptive_Optimization.py
Step_05C_Digital_Material_Card_Generation.py

Step_06A_ANSYS_Material_Card_Generation.py
Step_06B_Fatigue_SN_Curve_Generation.py

Step_07_SHAP_Interpretability.py

Step_08_Mesh_Sensitivity_Analysis.py
```

---

## Reproducibility

The workflow incorporates dataset versioning, composition hashing, group-aware validation strategies, uncertainty quantification, applicability-domain assessment, and algorithm consistency analysis to support reproducibility and facilitate independent verification of results.

---

## Software Requirements

Python 3.10 or later

Required packages include:

* numpy
* pandas
* scipy
* scikit-learn
* xgboost
* shap
* matplotlib
* seaborn
* statsmodels
* joblib

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## License

This repository is distributed under the MIT License.
