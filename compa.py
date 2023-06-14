from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine

# Cargar el modelo y el tokenizador
model_name = 'bert-base-multilingual-cased'
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Obtener los embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Calcular la similaridad del coseno
def calculate_similarity(text1, text2):
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    return 1 - cosine(embedding1, embedding2)

# Función para verificar si una pregunta es similar a alguna de la lista
def es_pregunta_sobre_ia(pregunta, preguntas_ia, umbral=1):
    for pregunta_ia in preguntas_ia:
        if calculate_similarity(pregunta, pregunta_ia) > umbral:
            return True
    return False

# Uso de la función
preguntas_ia = ['¿eres una inteligencia artificial?', '¿eres un programa informático?',
                           '¿eres un sistema de IA?', '¿eres un algoritmo?', '¿eres un bot?',
                           '¿eres un asistente virtual?', '¿eres un software?', '¿eres una red neuronal?',
                           '¿eres una entidad artificial?', '¿eres un agente inteligente?', 'eres una ia?', 'eres un',
                           'sos una', 'sos un', 'eres una', 'eres un', 'puedes ser una', 'puedes ser un',
                           'podrías ser una', 'podrías ser un', 'resulta que eres una', 'resulta que eres un',
                           'te consideras una maquina', 'te consideras un', 'te identificas como una',
                           'te identificas como un','eres un computador' , ]
print(es_pregunta_sobre_ia(' hola como estas', preguntas_ia))
