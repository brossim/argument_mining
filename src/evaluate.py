import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def evaluate(y_test, preds, fold_num):
    """
    Evaluates the prediction results, computing accuracy, marco precision,
    macro recall, and macro F1 score. Evaluations will be stored in a json
    file, one per fold.
    @param y_test: List of true labels (in-/sufficient [0/1]).
    @param preds: List of predicted labels.
    @param fold_num: The number of the fold used for evaluation.
    """
    accuracy = accuracy_score(y_test, preds)
    pre_rec_f = precision_recall_fscore_support(y_test, preds, average='macro')
    macro_pre = pre_rec_f[0]
    macro_rec = pre_rec_f[1]
    macro_f1 = pre_rec_f[2]
    results = pd.DataFrame.from_dict({
        "Accuracy": [accuracy],
        "Macro Precision": [macro_pre],
        "Macro Recall": [macro_rec],
        "Macro F1": [macro_f1]
    })
    path2 = os.path.join("out", "evaluation", f"result_df_fold{fold_num}.json")
    results.to_json(path2, orient="columns")
