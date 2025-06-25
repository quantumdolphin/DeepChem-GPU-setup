# 🧪 DeepChem + PyTorch + PyG + Transformers (GPU Setup)

> A fully GPU-accelerated machine learning environment tailored for computational chemistry and drug discovery. Includes DeepChem, RDKit, PyTorch (with CUDA), PyTorch Geometric, and Hugging Face Transformers — built and tested on a high-performance Linux workstation.

---

## 💻 System Configuration

* **OS:** Ubuntu 24.04.2 LTS
* **GPU:** NVIDIA GeForce RTX 4080 SUPER
* **Driver Version:** 570.133.20
* **CUDA Runtime Supported:** 12.8
* **Python Version:** 3.9.23
* **Environment Name:** `deepchem-gpu`

---

## 📦 What’s Included

This environment is built for modern deep learning applications in cheminformatics and molecular modeling. It includes:

* 🔬 **RDKit** — cheminformatics toolkit
* 🧪 **DeepChem 2.8.0** — ML for drug discovery, GNNs, multitask learning
* 🔥 **PyTorch 2.5.1 + CUDA 12.1** — deep learning with GPU acceleration
* 🧠 **PyTorch Geometric** — graph neural networks for molecular graphs
* 🧬 **Hugging Face Transformers** — SMILES/token embeddings for molecular NLP

TensorFlow was **intentionally skipped** to reduce dependency conflicts. This decision ensures that the environment remains lightweight and reproducible. TensorFlow can be added later if specific model support is required.

A reproducible `deepchem-gpu.yml` file is included for easy installation via Conda.

---

## 🛠️ Installation Instructions

### Step 1: Create the environment

```bash
conda create -n deepchem-gpu python=3.9 -y
conda activate deepchem-gpu
```

---

### Step 2: Install RDKit

```bash
conda install -c conda-forge rdkit -y
```

```python
from rdkit import Chem
print("RDKit OK")
```

---

### Step 3: Install PyTorch + CUDA 12.1

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

### Step 4: Install DeepChem

```bash
pip install deepchem
```

```python
import deepchem
print(deepchem.__version__)  # 2.8.0
```

---

### Step 5: Install PyTorch Geometric (CUDA 12.1 compatible)

```bash
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

```python
from torch_geometric.data import Data
print("PyG OK")
```

---

### Step 6: Install Hugging Face Transformers

```bash
pip install transformers
```

```python
from transformers import pipeline
print(pipeline("sentiment-analysis")("DeepChem is awesome"))
```

```python
from deepchem.models.torch_models import HuggingFaceModel
print("HuggingFaceModel is available")
```

---

## 📄 Final Environment Summary

| Component           | Status    |
| ------------------- | --------- |
| RDKit               | ✅ Working |
| PyTorch (CUDA 12.1) | ✅ Working |
| DeepChem            | ✅ v2.8.0  |
| PyTorch Geometric   | ✅ Working |
| Transformers        | ✅ Working |
| TensorFlow          | ❌ Skipped |

This environment is now suitable for:

* Graph neural networks (GNNs) for molecules
* Transformer-based SMILES models
* Fingerprint-based clustering and regression
* High-throughput featurization and multitask learning

---

## 📁 Included Files

* `deepchem-gpu.yml`: A reproducible environment definition file. You can recreate the environment with:

```bash
conda env create -f deepchem-gpu.yml
conda activate deepchem-gpu
```

---

## 📢 Usage Note for Resume / GitHub

This installation was performed and documented for sharing on GitHub to help others build GPU-accelerated cheminformatics environments with modern ML tools. It is suitable for showcasing on a technical resume or portfolio.

---

## ✅ Status

Fully functional and tested environment as of June 2025.

End of installation guide.
