Physics-Informed Explainable Machine Learning for Hardness Prediction in High-Entropy Alloys
Overview

This repository presents a reproducible framework for predicting Vickers hardness in high-entropy alloys (HEAs) by integrating machine learning with physics-based validation. The workflow combines composition-based modeling, physics-informed descriptor construction, interpretable learning, and finite element simulations to ensure both predictive accuracy and physical consistency.

Methodology

The study follows a structured multi-stage pipeline. A curated dataset is constructed from multiple sources and subjected to strict validation and normalization. Baseline models are developed using composition-only features, followed by descriptor-based modeling incorporating elastic, thermodynamic, and electronic properties. Model performance is evaluated using cross-validation, and robustness is assessed through repeated sampling and permutation testing. SHAP analysis is used to interpret feature contributions.

To ensure physical relevance, descriptor-derived properties are used to generate finite element inputs. Simulations based on Vickers and Berkovich indentation, along with anisotropic elastic analysis, are performed to validate predicted trends. Mesh convergence and statistical consistency checks are included to ensure numerical reliability.

Repository Structure

src/ – step-wise implementation (Step 1–13)
output/ – generated datasets, plots, and simulation inputs
data/ – source and processed datasets
docs/ – supporting figures and workflow materials

Reproducibility

All datasets are processed through deterministic pipelines with explicit validation. Model performance is reported using R², RMSE, and MAE, and results are supported by both statistical and physics-based validation.

License

MIT License

Citation

To be updated upon publication.
