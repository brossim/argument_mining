import pandas as pd
import os
from sklearn.model_selection import ShuffleSplit
from src.get_data import get_masked_data_arranged
from src.finetune_bart import finetune
from src.argument_generation import generate_conclusions
from src.SufficiencyClassifier import SufficiencyClassifier
from src.evaluate import evaluate


def unflatten(nested_list):
    """
    Unflattens the index-arranged data. The data is arranged
    by essay index to prevent essay overlap between train and
    test set.
    @param nested_list: List of lists.
    @return: Unflattened list.
    """
    return [datapoint for ele in nested_list for datapoint in ele]


# get data
X, y = get_masked_data_arranged()
# prepare 5-fold cross validation
folds = ShuffleSplit(n_splits=5, test_size=0.2, random_state=987)

for fold_num, (train_index, test_index) in enumerate(
        folds.split(X, y), start=1):
    # unflatten data and convert X (dicts) to dataframe
    X_train = pd.DataFrame.from_dict(
        unflatten([X[i] for i in train_index]))
    X_test = pd.DataFrame.from_dict(
        unflatten([X[i] for i in test_index]))
    y_train = unflatten([y[i] for i in train_index])
    y_test = unflatten([y[i] for i in test_index])

    # finetune BART large on X_train
    finetune(fold_num, X_train)

    # fit SufficiencyClassifier on X_train
    # as X_train has already been used for finetuning, passing
    # it again to the model to regenerate the conclusions should
    # yield rather similar outputs in terms of semantics. This is
    # supposed to ensure that the classifier determines a high
    # threshold for assessing an argument as sufficient. As a consequence,
    # good generalization abilities are required of the model when
    # performing on the test set
    classifier = SufficiencyClassifier()
    queries = X_train["au_masked"].tolist()
    generated = generate_conclusions(fold_num, queries)
    conclusion_pairs = [
        (orig, gen) for orig, gen
        in zip(X_train["conclusion"].tolist(), generated)
    ]
    classifier.fit(conclusion_pairs, y_train)

    # testing SufficiencyClassifier
    # (re-)generate conclusions from test set
    queries_test = X_test["au_masked"].tolist()
    generated_test = generate_conclusions(fold_num, queries_test)
    # extend dataframe by generated conclusions and labels
    X_test["generated_conclusion"] = generated_test
    X_test["gold_label"] = y_test

    # predict
    conclusion_pairs_test = [
        (orig, gen) for orig, gen
        in zip(X_test["conclusion"].tolist(), generated_test)
    ]
    preds, th_sim = classifier.predict(conclusion_pairs_test)
    # add threshold, computed cosine similarity and predicted labels
    # to dataframe and store test results per fold
    X_test["th_sim"] = th_sim
    X_test["predicted_label"] = preds
    path = os.path.join("out", "dfs_test_with_gen",
                        f"data_fold{fold_num}.json")
    X_test.to_json(path, orient="columns")
    # evaluate predictions
    evaluate(y_test, preds, fold_num)
