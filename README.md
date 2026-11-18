# Prediction of Heart, Kidney, and Eye Complications in Diabetes Patients

Python + scikit-learn + Flask website with a local SQLite history.

## Run in VS Code
1. File > Open Folder > select `disease_predictor`
2. Terminal > New Terminal, then `python -m venv venv` and `venv\Scripts\activate` (Mac/Linux: `source venv/bin/activate`)
3. `pip install -r requirements.txt`
4. `python app.py` (or press F5 > "Run website (Flask)"), then open http://127.0.0.1:5000

`python train.py` retrains the models (Data Acquisition > Preparation > Model Selection > Evaluation, in `ml/pipeline.py`).

Features: risk gauge, values-vs-typical-range charts, lifestyle and next-step advice, prediction history (`history.db`, local only).
Not a medical device.

## Model Algorithm Comparison

The algorithm-selection program is available in `compare_models.py`.

It compares six candidate algorithms for each dataset:
- Logistic Regression
- K-Nearest Neighbours
- Support Vector Machine
- Decision Tree
- Random Forest
- Gradient Boosting

The program uses 5-fold stratified cross-validation and selects the model with the highest mean ROC-AUC. The held-out test set is not used for this selection; it is used later for final evaluation.

Run it from the project folder with:

```bash
python compare_models.py
```

The same comparison logic is also implemented in `ml/pipeline.py` inside `candidates()` and `select_model()`.
