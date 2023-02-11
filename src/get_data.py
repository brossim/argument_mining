import json
import os
import pandas as pd


def get_raw_data():
    path = os.path.join("data", "data_raw.json")
    with open(path, "br") as f:
        data = json.load(f)
    return data


def get_masked_data_df():
    path = os.path.join("..", "data", "data_raw.json")
    with open(path, "br") as f:
        data = json.load(f)
        # extend every argument by a conclusion-masked version of itself
        for argument in data:
            arg_noisy = argument["au"].replace(
                argument["conclusion"], "<mask>")
            argument["au_masked"] = arg_noisy
    return pd.DataFrame.from_dict(data)


def get_masked_data_arranged():
    """
    Loads the raw data from a JSON file, masks the conclusion of each argument
    and stores the arguments in a list of lists, arranged by essay index.
    Each sublist corresponds to a single essay and contains one or more
    argument dictionaries.
    @return: data_arranged: a list of lists, where each sublist contains one
    or more argument dictionaries.
    """
    # there are 402 essay in the corpus, each of which contains one or
    # more arguments
    data_arranged = [[] for i in range(402)]
    labels_arranged = [[] for i in range(402)]
    path = os.path.join("data", "data_raw.json")
    with open(path, "br") as f:
        data = json.load(f)
        # extend every argument by a conclusion-masked version of itself
        for argument in data:
            arg_noisy = argument["au"].replace(
                argument["conclusion"], "<mask>"
            )
            argument["au_masked"] = arg_noisy
            # get the essay index to determine the correct list position in
            # data_arranged/labels_arranged
            num = int(argument["index"][5:8]) - 1
            # only add data pertinent to the study
            subset_dict = {key: value for key, value in argument.items()
                           if key in [
                               "index",
                               "au_masked",
                               "conclusion"
                           ]}
            data_arranged[num].append(subset_dict)
            # get respective label and store in correct sublist
            label = argument["local_sufficency"]
            # label is str; --> float --> int
            labels_arranged[num].append(int(float(label)))

    return data_arranged, labels_arranged


def get_gurcke_data_dict():
    path = os.path.join("data", "gurcke_generated.json")
    with open(path, "br") as f:
        data = json.load(f)
    return data


def get_gurcke_data_df():
    path = os.path.join("data", "gurcke_generated.json")
    with open(path, "br") as f:
        data = json.load(f)
    return pd.DataFrame.from_dict(data)
