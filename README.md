Fine-tuning CodeT5-Large on the TSSB-3M dataset for code-related tasks.

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
├── inference.py
├── preprocess_tssb.py
├── requirements.txt
└── train_codeT5.py
```

---

## 📦 Dataset Setup

The raw dataset (`tssb_data_3M.zip`) is **not included** in this repository due to GitHub's 100MB file size limit.

### Download Instructions

1. Download the TSSB-3M dataset from one of the following sources:

   | Source | Link |
   |--------|------|
   | 🤗 Hugging Face | [https://huggingface.co/datasets/zirui3/TSSB-3M-instructions/tree/main] |
   | GitHub | [[https://cedricrupb.github.io/TSSB3M/] |

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

## 📄 License

[Add your license here]

---

## 🙏 Acknowledgements

- [TSSB-3M Dataset](https://github.com/cedricrupb/TSSB-3M)
- [CodeT5](https://github.com/salesforce/CodeT5)
- [CodeLlama](https://github.com/facebookresearch/codellama)
