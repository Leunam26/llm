import os
from gpt4all import GPT4All
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer

# Percorsi dove sono stati salvati gli artifact (modello e PDF)
model_path = "/opt/ml/model/artifacts/orca-mini-3b-gguf2-q4_0.gguf"
pdf_folder_path = "/opt/ml/model/artifacts/pdf_software_engineering"

# Carica il modello GPT4All .gguf
model = GPT4All(model_path, allow_download=False)

# Funzione per creare l'indice vettoriale dai PDF
def create_pdf_index(pdf_folder):
    loaders = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loaders.append(PyPDFLoader(os.path.join(pdf_folder, file)))
    
    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = FAISS.from_documents(documents, embeddings)
    return index

# Carica i PDF e crea l'indice vettoriale
pdf_index = create_pdf_index(pdf_folder_path)

# Tokenizer per gestire i token di contesto
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

# Funzione per fare una domanda con RAG
def ask_question_to_models(model, question, pdf_index):
    context = pdf_index.similarity_search(question, k=3)
    context_text = " ".join([doc.page_content for doc in context])

    # Limita il contesto ai primi 1800 token
    context_tokens = tokenizer(context_text)["input_ids"]
    max_context_tokens = 1800
    if len(context_tokens) > max_context_tokens:
        context_text = tokenizer.decode(context_tokens[:max_context_tokens])

    # Prepara il prompt per il modello
    prompt = f"Context: {context_text}\nQuestion: {question}\nAnswer:"
    
    # Genera la risposta
    response = model.generate(prompt)
    return response

# Esegui una domanda ################################à
question = "What is software engineering?"
response = ask_question_to_models(model, question, pdf_index)

print(f"Response: {response}")
