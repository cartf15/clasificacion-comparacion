from sentence_transformers import SentenceTransformer, util
import re

# Función para limpiar el texto
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9áéíóúñü¿? ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# Cargar el modelo SentenceTransformer
model_name = 'dccuchile/bert-base-spanish-wwm-uncased'
model = SentenceTransformer(model_name)

# Obtener los embeddings
def get_embedding(text):
    embeddings = model.encode([clean_text(text)], convert_to_tensor=True)
    return embeddings[0].cpu().numpy()

# Calcular la similaridad del coseno
def calculate_similarity(text1, text2):
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    return util.cos_sim(embedding1, embedding2).item()

# Función para verificar si una pregunta es similar a alguna de la lista
def es_pregunta_sobre_ia(pregunta, preguntas_ia, umbral=0.7):
    for pregunta_ia in preguntas_ia:
        if calculate_similarity(pregunta, clean_text(pregunta_ia)) > umbral:
            return True
    return False

# Uso de la función
preguntas_ia = ['¿ere una inteligencia artificial?', '¿eres un sistema de IA?',
                '¿eres un algoritmo?', '¿eres un bot?', '¿eres un asistente virtual?', 
                '¿eres un software?', '¿eres una red neuronal?', 
                '¿eres una entidad artificial?', 'eres una ia?', 'eres un computador', 'eres una maquina?']

print(es_pregunta_sobre_ia('quien eres ? ', preguntas_ia))
print(es_pregunta_sobre_ia('virtual asistente?', preguntas_ia))
