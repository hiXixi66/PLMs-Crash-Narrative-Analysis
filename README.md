
# PLMS-CRASH-NARRATIVE-ANALYSIS

This project contains data processing, model fine-tuning, and evaluation scripts for **traffic crash narrative analysis**.  
It supports **large language models (PLMs)** (e.g., Qwen, LLaMA, BERT) and classic NLP models (TextRNN, FastText) for tasks such as:
- **MANCOLL classification** (manner of collision)
- **Crash Category (CCat)**
- **Crash Configuration (CConf)**
- **Crash Type (CTp)**

The project is expected to run on **GPU-enabled servers**.

---

## 📂 Project Structure

```

PLMS-CRASH-NARRATIVE-ANALYSIS/
├── analysis/
│   ├── consis-compute.py          # Compute cross-model consistency (pairwise agreement, Fleiss' kappa)
│   ├── mancoll-unknown-class.py   # Analyze the outputs of PLMs on unknown MANCOLL classes
│   ├── self-cons.py               # Self-consistency analysis for multiple runs
│   └── US-DB.txt                  # U.S. database reference
│
├── data/
│   ├── processed_data/            # Cleaned & processed Excel data
│   └── raw-data/                  # Original raw crash data
│
├── src/
│   ├── data_process/
│   │   ├── extract_excel_info.py  # Extract and clean data from the massive datset
│   │   └── NoiseTest_data_gen.py  # Generate noisy datasets for robustness testing
│   │
│   └── llm/
│       ├── bert-noft.py                   # BERT feature extraction (no fine-tuning)
│       ├── finetune_bert_crashtype.py     # Fine-tune BERT for Crash Type
│       ├── finetune_bert_mancoll.py       # Fine-tune BERT for MANCOLL classification
│       ├── finetune_bert_mancoll_noise.py # Fine-tune BERT with noisy MANCOLL labels
│       ├── llm_loader_HPC.py              # **Central config**: specify model storage paths (Qwen, LLaMA, Mistral, BERT)
│       ├── mancoll-fastText-finetune.py   # FastText baseline
│       ├── mancoll-rnn-finetune.py        # RNN baseline
│       ├── model_finetune_CCat.py         # Fine-tune LLM for Crash Category
│       ├── model_finetune_Cconf.py        # Fine-tune LLM for Crash Configuration
│       ├── model_finetune_CTp.py          # Fine-tune LLM for Crash Type
│       ├── model_finetune-withNoise.py    # Fine-tune LLM with noisy labels
│       └── model_finetune.py              # Fine-tune LLM for manner of collision
│
└── tests/
├── CrashCAT_test.py
├── CrashCONF_test.py
├── CrashType_test.py
├── CrashType_test-analysis.py
└── MANCOLL_test.py

````

---

## ⚙️ Model Storage Configuration

Edit `src/llm/llm_loader_HPC.py` to set the storage paths of your LLMs (on HPC or local server):

All training scripts import this file to load models automatically.

---

## 🚀 Running on GPU Cluster

Each main training/evaluation script has a corresponding `.sh` launcher (for SLURM or similar HPC schedulers).

Example SLURM template:

```bash
#!/bin/bash
#SBATCH --job-name=finetune_mancoll
#SBATCH --gres=gpu:1          # Number of GPUs
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mancoll_%j.out

module load cuda/11.8
source ~/envs/plms/bin/activate   # activate your Python virtual env

python src/llm/finetune_bert_mancoll.py
```

Recommended `.sh` scripts:

| Script name                   | Purpose                                                 |
| ----------------------------- | ------------------------------------------------------- |
| `train_mancoll_bert.sh`       | Fine-tune BERT for MANCOLL                              |
| `train_mancoll_bert_noise.sh` | Fine-tune BERT with noisy MANCOLL labels                |
| `train_crashcat_llm.sh`       | Fine-tune LLM for Crash Category                        |
| `train_crashconf_llm.sh`      | Fine-tune LLM for Crash Configuration                   |
| `train_crashtype_llm.sh`      | Fine-tune LLM for Crash Type                            |
| `train_llm_with_noise.sh`     | Fine-tune LLM with noisy labels                         |
| `fasttext_baseline.sh`        | Train FastText model                                    |
| `rnn_baseline.sh`             | Train RNN model                                         |
| `consistency_analysis.sh`     | Run self-consistency and cross-model agreement analysis |

---

### Example: `train_mancoll_bert.sh`

```bash
#!/bin/bash
#SBATCH --job-name=mancoll_bert
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mancoll_bert_%j.out

source ~/envs/plms/bin/activate
python src/llm/finetune_bert_mancoll.py
```


---

## 🛠️ Installation

```bash
conda create -n plms python=3.10
conda activate plms

pip install -r requirements.txt
```

`requirements.txt` includes:
`torch`, `transformers`, `datasets`, `peft`, `accelerate`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `statsmodels`, `openpyxl`.

---

## 📊 Outputs

* **Model checkpoints** → `models/`
* **Analysis reports** → `reports/`
* **Plots & heatmaps** → `reports/.../plots/`
* **Consistency metrics** → CSV files (`consistency_matrix.csv`, `sample_consistency.csv`)

