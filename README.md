# AG-Cast

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

The core network architecture and training scripts will be released upon paper acceptance.

---

## 📂 Repository Structure

```text
AG-Cast/
├── agcast/
│   ├── __init__.py
│   ├── data.py          
│   └── metrics.py       
├── configs/             
│   ├── sevir.json
│   ├── meteonet.json
│   ├── shanghai.json
│   └── cikm.json
├── test.py              # Standalone evaluation script
├── requirements.txt     # Python dependency specifications
└── README.md
```

## 🛠️ Installation

We recommend using Python 3.10+ and a matching PyTorch environment. Install the required dependencies:

```
pip install -r requirements.txt
```

##  Test

```
python test.py --config configs/sevir.json --predictions /path/to/prediction.npy  --targets /path/to/target.npy --output outputs/sevir_eval --batch-size 4
```

##  Train

```
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/sevir.json --data /your/data/sevir.h5 --output outputs/sevir --batch-size 4 --workers 4 --epochs 200 --patience 15
```
