import fitz
import faiss
# otra opcion de index puede ser annoy (de spotify)

import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_EMB = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
TOKENIZER = AutoTokenizer.from_pretrained("google/flan-t5-large")
MODEL_LLM = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
print(TOKENIZER.model_max_length)

# Paso 1: leer el PDF


def read_pdf(path: str):
    ''' funcion para leer el contenido de un archivo pdf
    @params
    path (str)
    return (str)
    '''
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Paso 2: dividir en chunks


def divide_text(text: str, chunk_size: int, overlap: int) -> list:
    words = text.split()
    chunks = []
    # separa las palabras por grupos de chunks.
    # Ej: primera posicion con 500 palabras y asi hasta terminar
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks


# Paso 3: crear embeddings y el índice
def create_index(chunks: list) -> tuple:
    ''' crea un índice de búsqueda semántica basado en embeddings de texto'''
    emb = MODEL_EMB.encode(chunks, normalize_embeddings=True)
    # normalize_embeddings=True: normaliza los vectores para que tengan longitud 1.
    # Esto permite usar similitud coseno

    index = faiss.IndexFlatIP(emb.shape[1])
    # este índice almacena los vectores y permite buscar los más similares a una consulta

    # se añaden los embeddings al indice
    index.add(np.array(emb))

    return index, emb

# Paso 4: recuperar contexto y generar respuesta
def responder(question: str, index, chunks: list, k=3):
    '''
    question: lo que el usuario quiere saber.
    index: el índice FAISS con embeddings de los chunks.
    chunks: lista de fragmentos del documento.
    k: número de chunks relevantes que se van a recuperar.'''

    # convierto la pregunta en un vector con el modelo embedding
    q_emb = MODEL_EMB.encode([question], normalize_embeddings=True)

    # se buscan los k chunks más cercanos a la pregunta en el espacio vectorial.
    # (I son los índices de los chunks relevantes. D son las distancias (qué tan similares son).
    D, I = index.search(np.array(q_emb), k)
    print('distancias: ', D, ' indices: ', I)

    # une los chunks recuperados en un solo bloque de texto, este será el contexto con el que el modelo responderá
    context = "\n".join([chunks[i] for i in I[0]])
    
    # se crea el prompt (contexto-pregunta-respuesta)
    prompt = f"Usa el siguiente contexto para responder:\n{context}\n\nPregunta: {question}\nRespuesta:"
    
    # se tokeniza el prompt
    inputs = TOKENIZER(prompt, return_tensors="pt")
    # el modelo genera la salida, con max de 200 tokens
    outputs = MODEL_LLM.generate(**inputs, max_new_tokens=200)
    
    # se decodifica la salida. Convierte los tokens generados de vuelta a texto legible
    return TOKENIZER.decode(outputs[0], skip_special_tokens=True)
