from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_recall_curve
from numpy import argmax
from tqdm import tqdm


class SufficiencyClassifier:
    """
    A classifier that determines if an argument is in-/sufficient (labels: 0/1)
    using a normalized cosine similarity score based on semantic similarity
    between the argument's original conclusion and one that was generated by
    a fine-tuned BART model. If the generated conclusion is semantically
    similar enough to the original one, i.e. its normalized cosine similarity
    value is greater than or equal to the computed threshold, the argument will
    be classified as sufficient, otherwise as insufficient. The threshold is
    determined using a Precision Recall Curve, where the threshold during
    fitting is computed by maximizing the F1 score.
    """

    def __init__(self):
        self.__threshold = None
        # transformer model to get sentence embeddings from conclusions
        self.__model = SentenceTransformer("stsb-roberta-base")

    @property
    def threshold(self):
        return self.__threshold

    @threshold.setter
    def threshold(self, value):
        self.__threshold = value

    @property
    def model(self):
        return self.__model

    def fit(self, X_train, y_train):
        """
        Fits the model to the training data.
        @param X_train: A list of conclusion pairs, where each conclusion pair
        is a tuple of two strings, corresponding to the ground truth and
        generated conclusion.
        @param y_train: A list of labels corresponding to the conclusion pairs
        in X_train.
        """
        assert len(X_train) == len(y_train), \
            "Data and labels have mismatching length"
        norm_cos_scores = []
        desc = "Fitting Sufficiency Classifier"
        for conclusion_pair in tqdm(X_train, desc=desc):
            assert isinstance(conclusion_pair, tuple) and \
                   len(conclusion_pair) == 2, \
                   "Data points in the train set must be composed of tuples"
            for conclusion in conclusion_pair:
                assert isinstance(conclusion, str), \
                    "Tuples in the train set must contain two strings each"
            norm_cos_scores.append(self.__compute_cos_sim(conclusion_pair))

        # determine optimal probability threshold by maximizing the F1 score
        self.threshold = self.__find_best_threshold(y_train, norm_cos_scores)

    @staticmethod
    def __find_best_threshold(y_true, y_proba):
        """
        Determines the optimal probability threshold that maximizes the F1
        score during training.
        @param y_true: List of true labels.
        @param y_proba: List of probability scores (i.e. normalized
        cosine similarity).
        @return: Optimal threshold as float.
        """
        precisions, recalls, thresholds = precision_recall_curve(
            y_true, y_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        best_idx = argmax(f1_scores)
        return thresholds[best_idx]

    def __compute_cos_sim(self, conclusion_pair):
        """
        Computes the cosine similarity score between the ground truth and
        BART-generated conclusion. Rescales the values from [-1, 1] to [0,1] in
        order to make them ROC-compatible.
        @param conclusion_pair: Tuple containing both conclusions.
        @return: Normalized cosine similarity, as float.
        """
        orig_premise = conclusion_pair[0]
        generated_premise = conclusion_pair[1]
        # get word embeddings for both premises
        orig_embedded = self.model.encode(
            orig_premise,
            convert_to_tensor=True)
        generated_embedded = self.model.encode(
            generated_premise,
            convert_to_tensor=True
        )
        # compute cosine similarity between the premises
        cosine_score = util.cos_sim(orig_embedded, generated_embedded).item()
        # normalize cosine score to range [0,1]
        # pay attention to zero division
        try:
            norm_cos_score = (cosine_score + 1) / 2
        except ZeroDivisionError:
            # if -1 + 1 / 2
            norm_cos_score = 0
        return norm_cos_score

    def predict(self, X_test):
        """
        Makes predictions on new conclusion pairs, using the determined
        threshold from the ROC curve.
        @param X_test: A list of conclusion pairs, where each conclusion
        pair is a tuple of two strings.
        @return: A list of binary labels indicating whether the argument
        that the conclusion(s) pertain to is sufficient (1) or not (0).
        """
        assert self.threshold is not None, \
            "Model has not been fit yet"
        out_labels = []
        th_sim = []
        for conclusion_pair in tqdm(X_test, desc="Sufficiency predicting"):
            cos_sim = self.__compute_cos_sim(conclusion_pair)
            th_sim.append(f"Threshold: {self.threshold}, cos_sim: {cos_sim}")
            out_labels.append(0) if cos_sim <= self.threshold \
                else out_labels.append(1)
        return out_labels, th_sim