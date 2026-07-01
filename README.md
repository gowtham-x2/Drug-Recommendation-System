# Drug Recommendation System

This project predicts a likely drug class from a patient's demographic and clinical profile using a multi-class classification pipeline and a Streamlit interface.

## Features

- synthetic patient dataset generation
- categorical and numeric preprocessing pipeline
- multi-class drug prediction model
- evaluation metrics and confusion matrix report
- single patient recommendation
- batch CSV scoring

## Files

- `drug_system.py` - data generation, training, inference, and reporting
- `train_model.py` - command-line training entry point
- `app.py` - Streamlit interface

## Run

```bash
python train_model.py
streamlit run app.py
```

## Expected Batch CSV Columns

`age`, `sex`, `blood_pressure`, `cholesterol`, `na_to_k`
