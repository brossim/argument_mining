import os
from tqdm import tqdm
from transformers import TFAutoModelForSeq2SeqLM, BartTokenizer


def generate_conclusions(num_fold, masked_arguments):
    """
    Generates conclusions for masked arguments using a finetuned BART model.
    @param num_fold: The fold number (for the respective model stored during
    k-fold cross validation) to use for loading the respective model.
    @param masked_arguments: A list of masked arguments.
    @return: A list of generated conclusions inferred from the masked
    arguments.
    """
    path = os.path.join("out", "bart_fine", f"fold{num_fold}")
    generated = []
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model = TFAutoModelForSeq2SeqLM.from_pretrained(path)
    desc = "(Re-)Generating conclusions"

    for arg in tqdm(masked_arguments, desc=desc):
        inp = tokenizer(arg, return_tensors='tf')
        conclusion_generation = model.generate(
            inp["input_ids"],
            attention_mask=inp["attention_mask"],
            use_cache=True,
            early_stopping=False,
            # length_penalty=0.8,
            max_length=70,
            min_length=20,
            no_repeat_ngram_size=2,
            num_beams=5,
            temperature=0.7,
            repetition_penalty=1.8
        )

        conclusion = tokenizer.decode(
            conclusion_generation[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        # if model generates more than one sentence, only choose the first
        # ground truth conclusions are not longer than 1 sentence
        # remove period from beginning of generations, if existent
        if conclusion.startswith("."):
            conclusion = conclusion[1:]
        # sometimes, there's a redundant period in the first sentence,
        # mostly after the first word; remove to prevent splitting there
        if "." in conclusion[:20]:
            conclusion = conclusion.replace(".", " ", 1)
        generated.append(conclusion.split(".")[0])
    return generated

#
# from src.get_data import get_masked_data_df
# data = get_masked_data_df()
# query = data["au_masked"].tolist()[895:900]
# gen = generate_conclusions(1, query)
# print(gen)
# print(data["conclusion"].tolist()[895:900])

