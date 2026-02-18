## Fine-tuning CodeT5-Large or CodeLlama LORA (Low-Rank Adaptation) on the TSSB-3M dataset for code-related tasks.

---

## 📁 Repository Structure

```
├── codellama-finetuned/
├── codet5-finetuned/
│   └── runs/                        ← Training logs (not included, generated on run)
├── processed_data/
│   ├── statistics.json
│   ├── test.jsonl
│   ├── train_sample.jsonl
│   ├── train.jsonl
│   └── val.jsonl
├── tssb_data/
│   └── extracted/
│       └── tssb_data_3M.zip         ← NOT included (see Dataset Setup below)
├── venv/                            ← NOT included (generated locally)
├── check.py
├── inference_CodeT5.py
├── inference_CodeLlama.py
├── preprocess_tssb.py
├── requirements.txt
└── train_codeT5.py
└── train_codeLlama.py
```

---

## 📦 Dataset Setup

The raw dataset (`tssb_data_3M.zip`) is **not included** in this repository due to GitHub's 100MB file size limit.

### Download Instructions

1. Download the TSSB-3M dataset from one of the following sources:

   | Source | Link |
   |--------|------|
   | 🤗 Hugging Face | https://huggingface.co/datasets/zirui3/TSSB-3M-instructions/tree/main |
   | GitHub | https://cedricrupb.github.io/TSSB3M/ |

2. Place the downloaded file at:
   ```
   tssb_data/tssb_data_3M.zip
   ```

3. Extract it:
   ```bash
   unzip tssb_data/tssb_data_3M.zip -d tssb_data/extracted/
   ```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/Training-TSSB3M_CodeT5-Large.git
cd Training-TSSB3M_CodeT5-Large
```

### 2. Set up the environment
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the dataset
Follow the [Dataset Setup](#-dataset-setup) instructions above.

### 4. Preprocess the data
```bash
python preprocess_tssb.py
```

### 5. Run training
```bash
python train_codeT5.py
```

### 6. Run inference
```bash
python inference.py
```

---

## 📊 Processed Data

The `processed_data/` folder contains pre-split, ready-to-use JSONL files:

| File | Description |
|------|-------------|
| `train.jsonl` | Full training set |
| `train_sample.jsonl` | Small sample for quick testing |
| `val.jsonl` | Validation set |
| `test.jsonl` | Test set |
| `statistics.json` | Dataset statistics |

---


## 🙏 Acknowledgements

- [TSSB-3M Dataset](https://github.com/cedricrupb/TSSB-3M)
- [CodeT5](https://github.com/salesforce/CodeT5)
- [CodeLlama](https://github.com/facebookresearch/codellama)


##  Training Screenshots
<img width="1061" height="159" alt="image" src="https://github.com/user-attachments/assets/f55ffd2c-6fdf-4629-9a37-890932c06703" />


## Test Case Screenshots
<img width="282" height="590" alt="image" src="https://github.com/user-attachments/assets/be67cca1-c7dd-4dca-95ba-d8e07c42ea15" />


