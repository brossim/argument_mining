import tensorflow as tf
import os
import pickle
from transformers import TFBartForConditionalGeneration, \
    BartTokenizer


# get model, tokenizer, and data
model_checkpoint = "facebook/bart-large"
model = TFBartForConditionalGeneration.from_pretrained(model_checkpoint)
tokenizer = BartTokenizer.from_pretrained(model_checkpoint)


def finetune(num_fold, data):
    """
    Finetunes a pre-trained BART model (facebook/bart-large) on
    the regeneration of masked argument conclusions.
    @param num_fold: The current fold number in cross-validation.
    @param data: The custom data (essay arguments) to use for finetuning.
    """
    print(f"Starting finetuning fold {num_fold}/5")
    # tokenize inputs (masked arguments) and labels (conclusions)
    # and pad both to the max. length of tokenized masked arguments (=188)
    inputs = tokenizer(
        data["au_masked"].tolist(),
        max_length=188,
        return_tensors="tf",
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        data["conclusion"].tolist(),
        max_length=188,
        return_tensors="tf",
        truncation=True,
        padding="max_length"
    )

    # use learning rate scheduler to stabilize training
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.000005,
        decay_steps=10,
        alpha=0.0000005
    )

    # compile and train
    model.compile(
        optimizer=tf.optimizers.Adam(lr_schedule),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            # ignore pad token when computing loss
            from_logits=True, ignore_class=tokenizer.pad_token_id
        ),
        # metrics for training and validation
        metrics=["acc", "ce"]
    )

    # use early stopping and restoring to prevent overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.1,
        patience=2,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        [inputs.input_ids, inputs.attention_mask],
        labels.input_ids,
        verbose=1,
        batch_size=5,
        epochs=10,
        validation_split=0.15,
        callbacks=[early_stop],
        shuffle=True
    )

    # save model and history
    path = os.path.join("out", "bart_fine", f"fold{num_fold}")
    if not os.path.isdir(path):
        os.mkdir(path)
    model.save_pretrained(path)

    path2 = os.path.join("out", "histories",
                         f"history_fold{num_fold}.pkl")
    with open(path2, "wb") as f:
        pickle.dump(history.history, f)
