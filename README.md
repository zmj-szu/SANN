# Enhancing Malignant Transformation Predictions in Oral Potentially Malignant Disorders: A Novel Machine Learning Framework Using Real-World Data
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)

# Background
Oral potentially malignant disorders (OPMDs) carry an inherent risk of malignant transformation (MT), making early detection and appropriate management essential to prevent neoplastic progression. However, the absence of reliable predictive models has impeded the accurate identification of OPMD patients at heightened risk for MT. This study aimed to compare the predictive performance of traditional statistical methods with machine learning (ML) algorithms and to propose a novel enhanced ML framework specifically designed for accurate MT risk prediction in OPMDs.
# Installation
### 1. Clone Repository
```
git clone https://github.com/zmj-szu/SANN.git
cd SANN
```
### 2. Create Environment
Use Conda to ensure a reproducible environment:
```
conda create -n SANN_env python=3.8
conda activate SANN_env
pip install -r requirements.txt
```
### 3. Run the Example
Users can test the pipeline using the included `raw_data0126_OLC.xlsx`, which serves as a small sample dataset. Please note that this file is provided solely for demonstration purposes and does not represent the full dataset used in our experiments. 
Replacing it with your dataset following the same structure.
```
python self_attention_ann.py
```


# Methods
Data from one thousand and ninety-four OPMD patients across three tertiary health institutions, spanning the years 2004 to 2023, were retrieved for this study. We systematically collected information on demographic characteristics, lifestyle habits, underlying health conditions, clinical manifestations related to the lesions, histopathological features, and prior treatments of the enrolled patients. A customized, enhanced machine learning model, named Self Attention Artificial Neural Network (SANN), was trained, tested, and validated. Its performance was compared to a Cox proportional hazards (Cox-PH) based nomogram and other previously well-performing conventional ML algorithms.
# Findings
The AUC of the nomogram ranged from 0.880 to 0.902 for predictions spanning 3 to 20 years. In comparison, ML algorithms demonstrated superior predictive performance, with the following AUCs: SANN (0.9877), ANN (0.9788), RF (0.9672), and DeepSurv (0.9484). The customized SANN model achieved remarkable metrics, including sensitivity of 0.9656, specificity of 0.9593, accuracy of 0.9623, and precision of 0.9603. Comprehensive evaluation of the ROC, calibration curves, and DCA across the 4 algorithms revealed that the SANN exhibited the best predictive efficacy and stability, followed by ANN algorithm. External validation further confirmed the robustness and generalizability of the SANN model.
# Interpretation
This study shows that ML-based predictive models offer high sensitivity, specificity, and accuracy in identifying OPMD patients at risk of MT, outperforming traditional statistical approaches. Notably, the customized SANN algorithm demonstrated strong predictive capabilities, highlighting its potential utility in clinical practice.
