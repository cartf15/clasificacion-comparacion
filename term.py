import numpy as np
from transformers import TFBertForSequenceClassification

# Cargar el modelo previamente entrenado
model = TFBertForSequenceClassification.from_pretrained("predictor2")

def predict_sentiment(sentence):
    # Normalización de la frase
    normalized_sentence = normalize([sentence])

    # Convertir en InputExample
    predict_example = InputExample(guid=None, text_a = normalized_sentence[0], text_b = None, label = 0)

    # Convertir en tf.data.Dataset
    predict_data = convert_examples_to_tf_dataset([predict_example], tokenizer)
    predict_data = predict_data.batch(1)

    # Predicción
    logits = model.predict(predict_data)[0]

    # Transformar los logits en probabilidades
    probs = tf.nn.softmax(logits, axis=1).numpy()

    # Retornar la clase y su probabilidad asociada
    return np.argmax(probs), np.max(probs)

sentence = 'hola'
class_id, prob = predict_sentiment(sentence)
print(f'The sentiment of the sentence is: {class_id} with a probability of {prob}')
