# NBA_FAS
NBA Future Analytics Stars Program

# NBA ROY Prediction — README

## Overview
This project builds a machine learning model to predict the NBA Rookie of the Year (ROY) winner for the 2025–26 season.  
The workflow includes collecting historical rookie data, engineering performance metrics, training a classification model, and generating probability-based predictions.  
This work follows the requirements described in the 2026 Future Analytics Stars Technical Challenge.

---

## Installation

### 1. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```
## 3. Build, Train, Generate
```bash
python src/build_roy_dataset.py
python src/train_roy_model.py
python src/predict_roy.py
```
