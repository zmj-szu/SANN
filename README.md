# Enhancing Malignant Transformation Predictions in Oral Potentially Malignant Disorders: A Novel Machine Learning Framework Using Real-World Data
![](https://img.shields.io/badge/python-3.10+-blue.svg)  
![](https://img.shields.io/badge/PyTorch-2.9.1-red.svg)

# Background
Oral potentially malignant disorders (OPMDs) carry an inherent risk of malignant transformation (MT), making early detection and appropriate management essential to prevent neoplastic progression. However, the absence of reliable predictive models has impeded the accurate identification of OPMD patients at heightened risk for MT. This study aimed to compare the predictive performance of traditional statistical methods with machine learning (ML) algorithms and to propose a novel enhanced ML framework specifically designed for accurate MT risk prediction in OPMDs.

# Installation
### **Git Download and Installation (if not installed)**
Before cloning the repository, please ensure that **Git** is installed on your computer. If you haven't installed Git yet, follow the steps below:

#### **Step 1: Download and Install Git**
1. **Visit the Git Official Website**:
    - Go to the [Git official website](https://git-scm.com/) to download Git.
2. **Select the Correct Version for Your OS**:
    - The website will automatically detect your operating system. Click the appropriate download button for your system.
    - If it's not automatically detected, manually select the version for **Windows**, **macOS**, or **Linux**.
3. **Install Git**:
    - After downloading, run the installer and follow the default installation prompts. Make sure to select the option to add Git to your system's path for easy access from the command line.
4. **Verify Installation**:
    - Once installed, open **Command Prompt** (Windows) or **Terminal** (macOS/Linux) and verify the installation by typing:

```plain
git --version
```

    - You should see a version number, such as `git version 2.x.x`, confirming that Git is installed.

---

### **Install Anaconda (if not installed)**
Before creating the environment, make sure **Anaconda** is installed. If not, follow these steps to install it:

#### **Step 1: Download Anaconda**
1. **Visit Anaconda's Official Website**:
    - Go to [Anaconda's official website](https://www.anaconda.com/products/distribution) to download Anaconda.
2. **Select the Correct Version for Your OS**:
    - Anaconda will automatically detect your operating system. If it doesn't, select the appropriate version for your system (Windows, macOS, or Linux).
3. **Download the Installer**:
    - For **Windows**: Click the **"Download"** button for the **Windows** version.
    - For **macOS/Linux**: Choose the appropriate version and download it.
4. **Run the Installer**:
    - After downloading, double-click the `.exe` file (for Windows) or the corresponding file for macOS/Linux to run the installer.
5. **Install Anaconda**:
    - Follow the prompts to install Anaconda. You can select the default settings unless you have specific preferences.
    - **Add Anaconda to PATH (optional)**: It is recommended to select the option to **"Add Anaconda to my PATH environment variable"** during installation. This step is optional but helpful for easy access.

#### **Step 2: Open Anaconda Prompt**
After Anaconda is installed, open **Anaconda Prompt** to continue:

#### For **Windows**:
1. **Open Anaconda Prompt**:
    - Click the **Start Menu** (Windows icon), search for **"Anaconda Prompt"**, and click on the icon when it appears.
2. **Verify Installation**:
    - To check if Anaconda was installed correctly, type the following command in the Anaconda Prompt:

```plain
conda --version
```

    - This should display the version of Anaconda installed on your system.

#### For **macOS/Linux**:
1. **Open Terminal**:
    - For **macOS**: Go to **Applications > Utilities > Terminal**.
    - For **Linux**: Press **Ctrl + Alt + T** to open Terminal.
2. **Verify Installation**:
    - To verify Anaconda is working, type the following command:

```plain
conda --version
```

---

### **Cloning the Repository**
Once Git is installed, you can proceed with cloning the repository:

1. **Clone the Repository**:

```plain
git clone https://github.com/zmj-szu/SANN.git
cd SANN
```

```
- Open **Anaconda Prompt** (or **Command Prompt**/**Terminal** depending on your system) and run the following command:
```

---

### **Create the Environment**
1. **Create the Environment**:

```plain
conda create -n SANN_env python=3.10
```

    - Run the following command in Anaconda Prompt (or Terminal) to create a new environment named `SANN_env` with Python 3.10:
2. **Activate the Environment**:

```plain
conda activate SANN_env
```

    - After the environment is created, activate it by running:
3. **Install Required Dependencies**:

```plain
pip install -r requirements.txt
```

    - With the environment activated, install the required dependencies by running:

---

### **Run the Example**
A small sample dataset, `sample_data.xlsx`, is provided to test the pipeline and understand the data structure. Please note that this sample may not contain enough data points to fully replicate the SMOTE sampling or 10-fold cross-validation procedures as described in our paper. The original dataset is not publicly available due to ethical concerns. The sample is intended primarily to demonstrate the format and structure of the data used in our study.

To run your own dataset, follow these steps:

1. Create a new directory named `data`.
2. Rename your dataset file to `raw_data0126_OLC.xlsx`.
3. Place the renamed file into the `data` directory.

Make sure your dataset follows the same structure as the provided sample to maintain compatibility with the pipeline.

```plain
python self-attention-ann.py
```

---



# Methods
Data from one thousand and ninety-four OPMD patients across three tertiary health institutions, spanning the years 2004 to 2023, were retrieved for this study. We systematically collected information on demographic characteristics, lifestyle habits, underlying health conditions, clinical manifestations related to the lesions, histopathological features, and prior treatments of the enrolled patients. A customized, enhanced machine learning model, named Self Attention Artificial Neural Network (SANN), was trained, tested, and validated. Its performance was compared to a Cox proportional hazards (Cox-PH) based nomogram and other previously well-performing conventional ML algorithms.

# Findings
The AUC of the nomogram ranged from 0.880 to 0.902 for predictions spanning 3 to 20 years. In comparison, ML algorithms demonstrated superior predictive performance, with the following AUCs: SANN (0.9877), ANN (0.9788), RF (0.9672), and DeepSurv (0.9484). The customized SANN model achieved remarkable metrics, including sensitivity of 0.9656, specificity of 0.9593, accuracy of 0.9623, and precision of 0.9603. Comprehensive evaluation of the ROC, calibration curves, and DCA across the 4 algorithms revealed that the SANN exhibited the best predictive efficacy and stability, followed by ANN algorithm. External validation further confirmed the robustness and generalizability of the SANN model.

# Interpretation
This study shows that ML-based predictive models offer high sensitivity, specificity, and accuracy in identifying OPMD patients at risk of MT, outperforming traditional statistical approaches. Notably, the customized SANN algorithm demonstrated strong predictive capabilities, highlighting its potential utility in clinical practice.

