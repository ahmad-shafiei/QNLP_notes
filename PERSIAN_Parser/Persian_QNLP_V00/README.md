# Persian QNLP Pipeline

A hybrid **Quantum Natural Language Processing (QNLP)** and **PEFT + RAG** framework for Persian-language text understanding and generation.

---

## 🚀 Project Overview
This repository integrates **DisCoCat-based QNLP modeling** (using `lambeq` + `PennyLane`) with **classical transformer-based methods** enhanced via **Parameter-Efficient Fine-Tuning (PEFT)** and **Retrieval-Augmented Generation (RAG)**.

The project explores both theoretical and engineering aspects:
- **QNLP-hybrid pipeline:** quantum-enhanced semantic modeling.
- **PEFT+RAG pipeline:** efficient fine-tuning and context-grounded generation.

---

## 🧩 Folder Structure

Persian_QNLP_V00/

│
├── preprocess/ # Text preprocessing and normalization (Hazm, tokenization)

├── parser/ # PersianCatParser and grammar-based parsing

├── qnlp/ # lambeq + PennyLane QNLP circuits

├── rag_peft/ # RAG and PEFT classical pipelines

├── demo/ # Streamlit or Gradio web demo

├── notebooks/ # Experiments, results and analysis

├── results/ # Plots, metrics, and reports

├── README.md # Project documentation

└── requirements.txt # Dependencies list

---

## 🧠 3-Month Research Plan (Summary)
**Goal:** deliver a publishable comparison between QNLP-hybrid and PEFT+RAG pipelines and a working demo.

| Phase | Weeks | Focus |
|-------|--------|-------|
| Phase 1 | 0–2 | Baseline classical model (BERT/ParsBERT fine-tuning) |
| Phase 2 | 3–5 | Implement QNLP hybrid (PersianCatParser → lambeq → PennyLane) |
| Phase 3 | 4–7 | Develop PEFT + RAG pipeline |
| Phase 4 | 8–9 | Comparative evaluation and analysis |
| Phase 5 | 10–12 | Write paper, build Streamlit demo, publish repo |

---

## 🧰 Requirements
See [`requirements.txt`](requirements.txt) for detailed package list.  
Main dependencies include:
- `lambeq`, `discopy`, `pennylane`, `qiskit`
- `transformers`, `peft`, `faiss-cpu`
- `streamlit`, `gradio`, `hazm`, `numpy`, `pandas`, `matplotlib`

---

## 🧪 How to Run
```bash
# activate your environment
conda activate qnlp

# install requirements
pip install -r requirements.txt

# run tests or demo
python demo/app.py
