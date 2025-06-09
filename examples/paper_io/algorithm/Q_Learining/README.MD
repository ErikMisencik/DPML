# Deep Paper.io Agent Training

This repository provides a framework for training and evaluating reinforcement learning agents in a custom Paper.io-like environment.

## 📁 Project Navigation

Start by navigating to the working directory:

```bash
cd examples/paper_io
```

## 🛠️ Environment Setup

Install the required Conda environment using the provided `environment.yml` file:
If you want change name of conda environment you need to change name in `environment.yml` 

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate paperio
```

## 🧠 Agent Training

To begin training the agents, run:

```bash
python training2.py
```

## 📊 Agent Evaluation

To evaluate trained agents, use:

```bash
python eval.py
```

> ⚠️ **Important:** You must manually set the correct file paths to the trained agent models inside the `eval.py` script.

## 🎮 Game Environment

The core game logic and the most up-to-date environment configuration — including rewards, penalties, and game mechanics — are implemented in:

```
Paper_io_develop.py
```

Use this file as the reference for the current simulation behavior.

---