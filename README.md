# 100 Days of Machine Learning: From Foundations to Publication-Ready Remote Sensing

A complete, executable machine-learning curriculum in **100 Jupyter notebooks**. It starts with Python and NumPy, moves through statistics,
classical ML, ensembles, unsupervised and probabilistic learning, deep learning, modern architectures and trustworthy ML/MLOps, and ends
with **seven end-to-end remote-sensing research projects**.

## How each notebook is organised
1. **Title and module**, followed by a short motivation
2. **Numbered sections**: concept, then runnable code, then interpretation of the actual output
3. **Part 2: Going Deeper**: advanced extensions
4. **Practice problems with solutions** (a few open exercises are marked as such)
5. **Key References** (textbooks and papers)

All notebooks are **self-contained**. They use synthetic data or datasets bundled with scikit-learn, so they run offline on a CPU laptop
without downloads. Every notebook in this repository has been executed, and the outputs are saved.

## Setup
```bash
conda env create -f environment.yml      # or: pip install -r requirements.txt  (Python 3.12)
conda activate ml100
jupyter lab
```
- CPU only is fine. Most notebooks run in under 2 minutes; the deep-learning and project notebooks take 2-10 minutes.
- Windows: DataLoaders use `num_workers=0`.
- Optional extras (code is included but guarded by flags, because they need accounts or installs): Google Earth Engine and STAC
  (Day 87), MLflow (Day 81), Dask (Day 84), Hugging Face `transformers` (Day 76), TorchGeo pretrained weights (Day 97).

## Module 13: remote-sensing projects
Module 13 works in a simulated but realistic Sentinel-1/2 study area (UTM 45N), modelled loosely on the Damodar valley and the
Dhanbad-Jharia coalfield. The simulation includes clouds, speckle, haze, regional gradients, mixed pixels, label noise, GPS error, and
spectral and phenological confusion. Because the **true** land cover, biomass, flood extent and susceptibility are known, every notebook
can show how far each validation method is from the truth. To use real data, run the Day 87 GEE pipeline for your own area and replace
`13_Remote_Sensing_ML_Publication_Projects/data/study_area`; the project code is written to run unchanged.

| Project | Days | Topic | Highlights |
|---|---|---|---|
| Methods | 86-88 | RS data, GEE/STAC pipeline, spatial validation | Moran's I, variograms, block/buffered/NNDM CV, AOA, Olofsson estimators |
| 1 | 89-91 | Land use / land cover | feature module, label QC, nested spatial CV of 6 models, ablation, McNemar/Holm, map + stratified area estimates, SHAP |
| 2 | 92-93 | Crop type from S1/S2 time series | Whittaker smoothing, phenology, leave-region/year-out, TempCNN/LSTM/Transformer, early-season mapping |
| 3 | 94 | Flood mapping from SAR | Lee filter, Otsu, change detection + HAND, RF, U-Net, error analysis by condition, exposure |
| 4 | 95 | Landslide susceptibility | depression filling, TWI, VIF, frequency ratio, LR/RF/XGBoost, success/prediction rates, zonation, absence sampling |
| 5 | 96 | Biomass and carbon | saturation, conformalised quantile regression, GREG model-assisted totals, carbon and CO2e |
| 6 | 97 | Scene classification | SimCLR pretraining, label efficiency, dry-season shift, Grad-CAM with sanity check |
| 7 | 98 | Land-cover change and projection | transition matrices, spurious change, sample-based change areas, Markov-CA, Figure of Merit |
| Publication | 99-100 | Reproducible package and writing | single-command pipeline, determinism check, tests, model card, Zenodo; auto-drafted results, manuscript linter |

Helper code shared by Module 13 is in [`rs_utils.py`](13_Remote_Sensing_ML_Publication_Projects/rs_utils.py).

## Syllabus

### Module 1: Foundations and Tools (Days 1-8)
*Python, NumPy, pandas, plotting, EDA, and the linear algebra and calculus behind ML*

| Day | Notebook |
|---|---|
| 1 | [What is Machine Learning](01_Foundations_and_Tools/Day_001_What_is_Machine_Learning.ipynb) |
| 2 | [Environment Setup and Reproducible Workflow](01_Foundations_and_Tools/Day_002_Environment_Setup_and_Reproducible_Workflow.ipynb) |
| 3 | [NumPy for Machine Learning](01_Foundations_and_Tools/Day_003_NumPy_for_Machine_Learning.ipynb) |
| 4 | [Pandas for Machine Learning](01_Foundations_and_Tools/Day_004_Pandas_for_Machine_Learning.ipynb) |
| 5 | [Data Visualization with Matplotlib and Seaborn](01_Foundations_and_Tools/Day_005_Data_Visualization_with_Matplotlib_and_Seaborn.ipynb) |
| 6 | [Exploratory Data Analysis](01_Foundations_and_Tools/Day_006_Exploratory_Data_Analysis.ipynb) |
| 7 | [Linear Algebra for ML](01_Foundations_and_Tools/Day_007_Linear_Algebra_for_ML.ipynb) |
| 8 | [Calculus and Automatic Differentiation](01_Foundations_and_Tools/Day_008_Calculus_and_Automatic_Differentiation.ipynb) |

### Module 2: Probability Statistics and Optimization (Days 9-14)
*Probability, statistics, hypothesis tests, information theory, optimisation, MLE/MAP*

| Day | Notebook |
|---|---|
| 9 | [Probability and Distributions](02_Probability_Statistics_and_Optimization/Day_009_Probability_and_Distributions.ipynb) |
| 10 | [Descriptive and Inferential Statistics](02_Probability_Statistics_and_Optimization/Day_010_Descriptive_and_Inferential_Statistics.ipynb) |
| 11 | [Hypothesis Testing for ML](02_Probability_Statistics_and_Optimization/Day_011_Hypothesis_Testing_for_ML.ipynb) |
| 12 | [Information Theory for ML](02_Probability_Statistics_and_Optimization/Day_012_Information_Theory_for_ML.ipynb) |
| 13 | [Optimization from Gradient Descent to Adam](02_Probability_Statistics_and_Optimization/Day_013_Optimization_from_Gradient_Descent_to_Adam.ipynb) |
| 14 | [MLE MAP and Loss Functions](02_Probability_Statistics_and_Optimization/Day_014_MLE_MAP_and_Loss_Functions.ipynb) |

### Module 3: Data Preparation and Feature Engineering (Days 15-21)
*The scikit-learn workflow, leakage, missing data, scaling and encoding, feature engineering, pipelines, imbalance*

| Day | Notebook |
|---|---|
| 15 | [The ML Workflow and scikit learn API](03_Data_Preparation_and_Feature_Engineering/Day_015_The_ML_Workflow_and_scikit_learn_API.ipynb) |
| 16 | [Train Test Split Overfitting and Data Leakage](03_Data_Preparation_and_Feature_Engineering/Day_016_Train_Test_Split_Overfitting_and_Data_Leakage.ipynb) |
| 17 | [Missing Values and Outliers](03_Data_Preparation_and_Feature_Engineering/Day_017_Missing_Values_and_Outliers.ipynb) |
| 18 | [Feature Scaling and Encoding](03_Data_Preparation_and_Feature_Engineering/Day_018_Feature_Scaling_and_Encoding.ipynb) |
| 19 | [Feature Engineering](03_Data_Preparation_and_Feature_Engineering/Day_019_Feature_Engineering.ipynb) |
| 20 | [Pipelines and ColumnTransformer](03_Data_Preparation_and_Feature_Engineering/Day_020_Pipelines_and_ColumnTransformer.ipynb) |
| 21 | [Imbalanced Data](03_Data_Preparation_and_Feature_Engineering/Day_021_Imbalanced_Data.ipynb) |

### Module 4: Regression (Days 22-26)
*OLS and its assumptions, gradient descent, regularisation, diagnostics, GLMs, robust and quantile regression*

| Day | Notebook |
|---|---|
| 22 | [Linear Regression OLS and Assumptions](04_Regression/Day_022_Linear_Regression_OLS_and_Assumptions.ipynb) |
| 23 | [Gradient Descent and Polynomial Regression](04_Regression/Day_023_Gradient_Descent_and_Polynomial_Regression.ipynb) |
| 24 | [Regularization Ridge Lasso ElasticNet](04_Regression/Day_024_Regularization_Ridge_Lasso_ElasticNet.ipynb) |
| 25 | [Regression Metrics and Diagnostics](04_Regression/Day_025_Regression_Metrics_and_Diagnostics.ipynb) |
| 26 | [GLMs Robust and Quantile Regression](04_Regression/Day_026_GLMs_Robust_and_Quantile_Regression.ipynb) |

### Module 5: Classification (Days 27-34)
*Logistic regression, metrics, calibration, kNN, Naive Bayes, LDA/QDA, SVM, trees, multiclass/multilabel*

| Day | Notebook |
|---|---|
| 27 | [Logistic Regression](05_Classification/Day_027_Logistic_Regression.ipynb) |
| 28 | [Classification Metrics](05_Classification/Day_028_Classification_Metrics.ipynb) |
| 29 | [Probability Calibration and Thresholds](05_Classification/Day_029_Probability_Calibration_and_Thresholds.ipynb) |
| 30 | [k Nearest Neighbours and Distance Metrics](05_Classification/Day_030_k_Nearest_Neighbours_and_Distance_Metrics.ipynb) |
| 31 | [Naive Bayes LDA and QDA](05_Classification/Day_031_Naive_Bayes_LDA_and_QDA.ipynb) |
| 32 | [Support Vector Machines](05_Classification/Day_032_Support_Vector_Machines.ipynb) |
| 33 | [Decision Trees](05_Classification/Day_033_Decision_Trees.ipynb) |
| 34 | [Multiclass and Multilabel Classification](05_Classification/Day_034_Multiclass_and_Multilabel_Classification.ipynb) |

### Module 6: Ensemble Learning (Days 35-40)
*Bagging, random forests, boosting, XGBoost/LightGBM/CatBoost, stacking, cross-validation strategies*

| Day | Notebook |
|---|---|
| 35 | [Ensemble Basics Voting and Bagging](06_Ensemble_Learning/Day_035_Ensemble_Basics_Voting_and_Bagging.ipynb) |
| 36 | [Random Forest](06_Ensemble_Learning/Day_036_Random_Forest.ipynb) |
| 37 | [Boosting AdaBoost and Gradient Boosting](06_Ensemble_Learning/Day_037_Boosting_AdaBoost_and_Gradient_Boosting.ipynb) |
| 38 | [XGBoost LightGBM and CatBoost](06_Ensemble_Learning/Day_038_XGBoost_LightGBM_and_CatBoost.ipynb) |
| 39 | [Stacking and Blending](06_Ensemble_Learning/Day_039_Stacking_and_Blending.ipynb) |
| 40 | [Cross Validation Strategies](06_Ensemble_Learning/Day_040_Cross_Validation_Strategies.ipynb) |

### Module 7: Model Selection Tuning and Explainability (Days 41-45)
*Hyperparameter search, learning curves, statistical model comparison, feature selection, PDP/ALE, SHAP/LIME*

| Day | Notebook |
|---|---|
| 41 | [Hyperparameter Tuning Grid Random and Bayesian](07_Model_Selection_Tuning_and_Explainability/Day_041_Hyperparameter_Tuning_Grid_Random_and_Bayesian.ipynb) |
| 42 | [Learning Curves and Statistical Model Comparison](07_Model_Selection_Tuning_and_Explainability/Day_042_Learning_Curves_and_Statistical_Model_Comparison.ipynb) |
| 43 | [Feature Selection](07_Model_Selection_Tuning_and_Explainability/Day_043_Feature_Selection.ipynb) |
| 44 | [Interpretability Permutation Importance PDP ICE ALE](07_Model_Selection_Tuning_and_Explainability/Day_044_Interpretability_Permutation_Importance_PDP_ICE_ALE.ipynb) |
| 45 | [SHAP and LIME](07_Model_Selection_Tuning_and_Explainability/Day_045_SHAP_and_LIME.ipynb) |

### Module 8: Unsupervised Learning (Days 46-52)
*k-means, hierarchical, DBSCAN/HDBSCAN, GMM/EM, PCA/SVD, t-SNE/UMAP, anomaly detection, recommenders*

| Day | Notebook |
|---|---|
| 46 | [KMeans Clustering](08_Unsupervised_Learning/Day_046_KMeans_Clustering.ipynb) |
| 47 | [Hierarchical Clustering DBSCAN and HDBSCAN](08_Unsupervised_Learning/Day_047_Hierarchical_Clustering_DBSCAN_and_HDBSCAN.ipynb) |
| 48 | [Gaussian Mixture Models and EM](08_Unsupervised_Learning/Day_048_Gaussian_Mixture_Models_and_EM.ipynb) |
| 49 | [PCA and SVD](08_Unsupervised_Learning/Day_049_PCA_and_SVD.ipynb) |
| 50 | [Manifold Learning tSNE and UMAP](08_Unsupervised_Learning/Day_050_Manifold_Learning_tSNE_and_UMAP.ipynb) |
| 51 | [Anomaly Detection](08_Unsupervised_Learning/Day_051_Anomaly_Detection.ipynb) |
| 52 | [Recommender Systems and Association Rules](08_Unsupervised_Learning/Day_052_Recommender_Systems_and_Association_Rules.ipynb) |

### Module 9: Probabilistic ML Time Series and Semi Supervised (Days 53-57)
*Bayesian ML and MCMC, Gaussian processes, ARIMA, ML forecasting, semi-supervised and active learning*

| Day | Notebook |
|---|---|
| 53 | [Bayesian Machine Learning and MCMC](09_Probabilistic_ML_Time_Series_and_Semi_Supervised/Day_053_Bayesian_Machine_Learning_and_MCMC.ipynb) |
| 54 | [Gaussian Processes](09_Probabilistic_ML_Time_Series_and_Semi_Supervised/Day_054_Gaussian_Processes.ipynb) |
| 55 | [Time Series Analysis ARIMA and Smoothing](09_Probabilistic_ML_Time_Series_and_Semi_Supervised/Day_055_Time_Series_Analysis_ARIMA_and_Smoothing.ipynb) |
| 56 | [Machine Learning for Time Series Forecasting](09_Probabilistic_ML_Time_Series_and_Semi_Supervised/Day_056_Machine_Learning_for_Time_Series_Forecasting.ipynb) |
| 57 | [Semi Supervised and Active Learning](09_Probabilistic_ML_Time_Series_and_Semi_Supervised/Day_057_Semi_Supervised_and_Active_Learning.ipynb) |

### Module 10: Deep Learning Foundations (Days 58-66)
*Networks from scratch, PyTorch, training deep nets, CNNs, transfer learning, augmentation, segmentation, RNNs*

| Day | Notebook |
|---|---|
| 58 | [Neural Networks from Scratch](10_Deep_Learning_Foundations/Day_058_Neural_Networks_from_Scratch.ipynb) |
| 59 | [PyTorch Tensors and Autograd](10_Deep_Learning_Foundations/Day_059_PyTorch_Tensors_and_Autograd.ipynb) |
| 60 | [PyTorch Training Loop Datasets and DataLoaders](10_Deep_Learning_Foundations/Day_060_PyTorch_Training_Loop_Datasets_and_DataLoaders.ipynb) |
| 61 | [Training Deep Networks Init Norm Dropout Schedulers](10_Deep_Learning_Foundations/Day_061_Training_Deep_Networks_Init_Norm_Dropout_Schedulers.ipynb) |
| 62 | [Convolutional Neural Networks](10_Deep_Learning_Foundations/Day_062_Convolutional_Neural_Networks.ipynb) |
| 63 | [CNN Architectures and Transfer Learning](10_Deep_Learning_Foundations/Day_063_CNN_Architectures_and_Transfer_Learning.ipynb) |
| 64 | [Data Augmentation and Training Tricks](10_Deep_Learning_Foundations/Day_064_Data_Augmentation_and_Training_Tricks.ipynb) |
| 65 | [Semantic Segmentation and Object Detection](10_Deep_Learning_Foundations/Day_065_Semantic_Segmentation_and_Object_Detection.ipynb) |
| 66 | [RNN LSTM and GRU](10_Deep_Learning_Foundations/Day_066_RNN_LSTM_and_GRU.ipynb) |

### Module 11: Advanced Deep Learning (Days 67-76)
*Attention and transformers, NLP, ViT, self-supervised learning, VAEs, GANs, diffusion, GNNs, RL, LLMs*

| Day | Notebook |
|---|---|
| 67 | [Attention and Transformers from Scratch](11_Advanced_Deep_Learning/Day_067_Attention_and_Transformers_from_Scratch.ipynb) |
| 68 | [NLP Text Representation and Embeddings](11_Advanced_Deep_Learning/Day_068_NLP_Text_Representation_and_Embeddings.ipynb) |
| 69 | [Transformers for NLP and Vision ViT](11_Advanced_Deep_Learning/Day_069_Transformers_for_NLP_and_Vision_ViT.ipynb) |
| 70 | [Self Supervised and Contrastive Learning](11_Advanced_Deep_Learning/Day_070_Self_Supervised_and_Contrastive_Learning.ipynb) |
| 71 | [Autoencoders and VAEs](11_Advanced_Deep_Learning/Day_071_Autoencoders_and_VAEs.ipynb) |
| 72 | [Generative Adversarial Networks](11_Advanced_Deep_Learning/Day_072_Generative_Adversarial_Networks.ipynb) |
| 73 | [Diffusion Models](11_Advanced_Deep_Learning/Day_073_Diffusion_Models.ipynb) |
| 74 | [Graph Neural Networks](11_Advanced_Deep_Learning/Day_074_Graph_Neural_Networks.ipynb) |
| 75 | [Reinforcement Learning](11_Advanced_Deep_Learning/Day_075_Reinforcement_Learning.ipynb) |
| 76 | [LLMs and Foundation Models](11_Advanced_Deep_Learning/Day_076_LLMs_and_Foundation_Models.ipynb) |

### Module 12: Trustworthy ML and MLOps (Days 77-85)
*Uncertainty and conformal prediction, fairness, domain shift, physics-informed ML, MLOps, deployment, monitoring, scaling*

| Day | Notebook |
|---|---|
| 77 | [Uncertainty Quantification and Conformal Prediction](12_Trustworthy_ML_and_MLOps/Day_077_Uncertainty_Quantification_and_Conformal_Prediction.ipynb) |
| 78 | [Fairness Bias and Responsible AI](12_Trustworthy_ML_and_MLOps/Day_078_Fairness_Bias_and_Responsible_AI.ipynb) |
| 79 | [Domain Shift and Domain Adaptation](12_Trustworthy_ML_and_MLOps/Day_079_Domain_Shift_and_Domain_Adaptation.ipynb) |
| 80 | [Physics Informed Machine Learning](12_Trustworthy_ML_and_MLOps/Day_080_Physics_Informed_Machine_Learning.ipynb) |
| 81 | [Experiment Tracking and Reproducibility](12_Trustworthy_ML_and_MLOps/Day_081_Experiment_Tracking_and_Reproducibility.ipynb) |
| 82 | [Model Deployment Saving APIs and ONNX](12_Trustworthy_ML_and_MLOps/Day_082_Model_Deployment_Saving_APIs_and_ONNX.ipynb) |
| 83 | [Model Monitoring and Data Drift](12_Trustworthy_ML_and_MLOps/Day_083_Model_Monitoring_and_Data_Drift.ipynb) |
| 84 | [Scaling ML to Big Data](12_Trustworthy_ML_and_MLOps/Day_084_Scaling_ML_to_Big_Data.ipynb) |
| 85 | [Capstone End to End ML Project and AutoML](12_Trustworthy_ML_and_MLOps/Day_085_Capstone_End_to_End_ML_Project_and_AutoML.ipynb) |

### Module 13: Remote Sensing ML Publication Projects (Days 86-100)
*RS data, Google Earth Engine, spatial validation, and seven end-to-end, publication-ready RS projects*

| Day | Notebook |
|---|---|
| 86 | [Remote Sensing Data for ML](13_Remote_Sensing_ML_Publication_Projects/Day_086_Remote_Sensing_Data_for_ML.ipynb) |
| 87 | [Google Earth Engine Data Pipeline for ML](13_Remote_Sensing_ML_Publication_Projects/Day_087_Google_Earth_Engine_Data_Pipeline_for_ML.ipynb) |
| 88 | [Geospatial ML Methodology Spatial CV and Accuracy Assessment](13_Remote_Sensing_ML_Publication_Projects/Day_088_Geospatial_ML_Methodology_Spatial_CV_and_Accuracy_Assessment.ipynb) |
| 89 | [Project1 LULC Part1 Study Area Sampling and Features](13_Remote_Sensing_ML_Publication_Projects/Day_089_Project1_LULC_Part1_Study_Area_Sampling_and_Features.ipynb) |
| 90 | [Project1 LULC Part2 Model Comparison and Spatial CV](13_Remote_Sensing_ML_Publication_Projects/Day_090_Project1_LULC_Part2_Model_Comparison_and_Spatial_CV.ipynb) |
| 91 | [Project1 LULC Part3 Mapping Area Estimation and Figures](13_Remote_Sensing_ML_Publication_Projects/Day_091_Project1_LULC_Part3_Mapping_Area_Estimation_and_Figures.ipynb) |
| 92 | [Project2 Crop Type Part1 Time Series and Phenology](13_Remote_Sensing_ML_Publication_Projects/Day_092_Project2_Crop_Type_Part1_Time_Series_and_Phenology.ipynb) |
| 93 | [Project2 Crop Type Part2 Deep Temporal Models](13_Remote_Sensing_ML_Publication_Projects/Day_093_Project2_Crop_Type_Part2_Deep_Temporal_Models.ipynb) |
| 94 | [Project3 Flood Mapping from SAR with UNet](13_Remote_Sensing_ML_Publication_Projects/Day_094_Project3_Flood_Mapping_from_SAR_with_UNet.ipynb) |
| 95 | [Project4 Landslide Susceptibility Mapping](13_Remote_Sensing_ML_Publication_Projects/Day_095_Project4_Landslide_Susceptibility_Mapping.ipynb) |
| 96 | [Project5 Biomass Regression with Uncertainty](13_Remote_Sensing_ML_Publication_Projects/Day_096_Project5_Biomass_Regression_with_Uncertainty.ipynb) |
| 97 | [Project6 Scene Classification Transfer Learning and GradCAM](13_Remote_Sensing_ML_Publication_Projects/Day_097_Project6_Scene_Classification_Transfer_Learning_and_GradCAM.ipynb) |
| 98 | [Project7 Land Cover Change and Future Projection](13_Remote_Sensing_ML_Publication_Projects/Day_098_Project7_Land_Cover_Change_and_Future_Projection.ipynb) |
| 99 | [Reproducible Research Package for Publication](13_Remote_Sensing_ML_Publication_Projects/Day_099_Reproducible_Research_Package_for_Publication.ipynb) |
| 100 | [Writing and Publishing an RS ML Paper](13_Remote_Sensing_ML_Publication_Projects/Day_100_Writing_and_Publishing_an_RS_ML_Paper.ipynb) |

## Repository layout
```
01_Foundations_and_Tools/ ... 13_Remote_Sensing_ML_Publication_Projects/   one folder per module, Day_NNN_*.ipynb inside
13_Remote_Sensing_ML_Publication_Projects/rs_utils.py                       shared RS helpers and simulators
13_Remote_Sensing_ML_Publication_Projects/project*_*/                       figures, tables and models written by the project notebooks
13_Remote_Sensing_ML_Publication_Projects/data/                             simulated rasters (regenerated by Days 86-87; git-ignored)
environment.yml, requirements.txt                                           the tested environment
```

## Suggested pace
- **1 day = 1 notebook** (1.5-3 hours including the practice problems).
- Short on time? Do Modules 1-7 and 10, then go straight to Module 13.
- For your own research, Days 88, 90, 91 and 99 are the methodological core.
