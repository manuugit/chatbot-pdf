from chatbot_pdf import create_index, divide_text, read_pdf, responder
from fastapi import FastAPI

MODEL_MAX_TOKENS = 512
CHUNK_SIZE = int(MODEL_MAX_TOKENS * 0.1)
OVERLAP_SIZE = int(CHUNK_SIZE * 0.2)


def main():
    print("ejecutando...")
    path = './instructivo_licuadora.pdf'
    text = read_pdf(path)

    print('TAMAÑO DEL CHUNK =', CHUNK_SIZE)
    chunks = divide_text(text, CHUNK_SIZE, OVERLAP_SIZE)
    print('cantidad de chunks ', len(chunks))
    
    index, emb = create_index(chunks)
    
    return chunks, index
   


app = FastAPI()

@app.post("/ask")
def ask(question: str):
    chunks, index = main()
    respuesta = responder(question, index, chunks)
    return {"respuesta": respuesta}
