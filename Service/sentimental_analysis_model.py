import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import os
from transformers import BertTokenizer


def load_model():
    model_path = os.path.join(os.path.abspath(os.getcwd()), 'sentiment_analysis_model')
    print("model path here is:" + model_path)
    model = tf.keras.models.load_model(model_path)
    return model


model = load_model()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def inference_model(input_sequence):
    predicted_value = -1
    try:
        tokens = tokenizer(input_sequence, return_tensors='tf', padding=True, truncation=True)
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        token_type_ids = tokens['token_type_ids']
        predictions = model.predict(
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            })
        logits = predictions['logits']
        predicted_value = tf.argmax(logits, axis=1).numpy()[0]
    except Exception as e:
        print("Error in making prediction:" + e)
        return predicted_value
    return predicted_value
