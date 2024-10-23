from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer
import os
import time
import json
from gpt4all import GPT4All
import psycopg2
import json
import numpy as np
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import pandas as pd


# Set our tracking server uri for logging
#mlflow.set_tracking_uri(uri="http://localhost:5000")
mlflow.set_experiment(experiment_name='LLM Mini Orca')
mlflow.start_run(run_name='Run Mini Orca RAG')
run_id = mlflow.active_run().info.run_id
mlflow.set_tag("Training Info", "Run Orca su dataset Software_Engineering con RAG")

mlflow.log_param("model_name", "Mini Orca")
mlflow.log_param("model_type", "LLM")
mlflow.log_param("model_size", "3B")
mlflow.log_param("quantization", "q4_0")
mlflow.log_param("library", "gpt4all")

# Log specific run parameters
mlflow.log_param("number_questions", 15) 
mlflow.log_param("Dataset", "software_engineering")

import os
model_path = os.path.abspath("Modelli_gpt4all")
dataset_path = os.path.abspath("Dataset")


def create_table():
    conn = psycopg2.connect(
        host="16.171.42.195",   # Update with the EC2 instance's IPv4 address ("localhost" if local)
        database="rag_evaluation",
        user="postgres",
        password="1234"
    )
    cursor = conn.cursor()

 # Creazione tabella (se non esiste)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_rag_responses_se (
    question_id INT,
    question_text TEXT,
    answer_orca TEXT,
    orca_time FLOAT
    );
    """)
    conn.commit()
    cursor.close()
    conn.close()

# Funzione per salvare gradualmente le risposte nel database
def save_to_db(result):
    conn = psycopg2.connect(
        host="13.51.162.134",     # Update with the EC2 instance's IPv4 address ("localhost" if local)
        database="rag_evaluation",
        user="postgres",
        password="1234"
    )
    cursor = conn.cursor()

    # Inserimento dati nel database
    cursor.execute("""
        INSERT INTO model_rag_responses_se (question_id, question_text, answer_orca, orca_time)
        VALUES (%s, %s, %s, %s);
    """, (
        result["question_id"],
        result["question_text"],
        result["answer_orca"],
        result["orca_time"]
    ))

    # Esegui il commit e chiudi la connessione
    conn.commit()
    cursor.close()
    conn.close()






# Caricamento dei PDF e creazione di un indice di vettori
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

pdf_index = create_pdf_index(os.path.join(dataset_path, "pdf_software_engineering"))

# Caricamento delle domande dal file JSON
with open(os.path.join(dataset_path, "software_engineering.json")) as f:
    questions = json.load(f)

# Carica i modelli .gguf
models = {
    "Orca": GPT4All(os.path.join(model_path, "orca-mini-3b-gguf2-q4_0.gguf"), allow_download=False)
}



# Carica il modello e il tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")



def ask_question_to_models(models, question, pdf_index):
    
    context = pdf_index.similarity_search(question, k=3)
    context_text = " ".join([doc.page_content for doc in context])

    # Tokenizza il contesto per verificare la lunghezza
    context_tokens = tokenizer(context_text)["input_ids"]
    
    
    
    # Limita il numero di token totali (per esempio, 2048 - lunghezza della domanda)
    max_context_tokens = 1800  # Lascia un margine per la domanda
    if len(context_tokens) > max_context_tokens:
        # Ritaglia il contesto per non eccedere i 1800 token
        context_text = tokenizer.decode(context_tokens[:max_context_tokens])
    
    # Fai domanda a ogni modello
    responses = {}
    response_times = {}  # Dizionario per salvare i tempi di risposta
    for model_name, model in models.items():
        prompt = f"Context: {context_text}\nQuestion: {question}\nAnswer:"
        
        # Misurazione del tempo di inizio
        start_time = time.time()
        
        # Utilizza il metodo generate per ottenere la risposta
        response = model.generate(prompt)
        
        # Misurazione del tempo di fine
        end_time = time.time()
        
        # Calcola il tempo di risposta
        response_time = end_time - start_time
        response_times[model_name] = response_time
        
        # Verifica se la risposta è una stringa o un dizionario
        if isinstance(response, str):
            responses[model_name] = response
        elif isinstance(response, dict) and "choices" in response:
            responses[model_name] = response["choices"][0]["text"]
        else:
            responses[model_name] = "Error: Unexpected response format"
        
        # Stampa il progresso della risposta e il tempo impiegato
        print(f"Model: {model_name} - Question: {question[:30]}... - Response: {responses[model_name][:50]}... - Time: {response_time:.2f} seconds")
    
    return responses, response_times  # Ritorna anche i tempi di risposta



# Modifica della logica di esecuzione per salvare anche i tempi
create_table()  # Creazione della tabella nel database

#########################
# Function to save the results to a CSV file
def save_to_csv(file_path, data):
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

# List to store results locally
results_data = []
#########################

for idx, question in enumerate(questions, start=1):
    q_id = question["id"]
    q_text = question["text"]
    print(f"Processing Question ID: {q_id} - {q_text[:50]}...")  # Messaggio di inizio per ogni domanda
    
    # Ottieni le risposte dai modelli e i tempi di risposta
    responses, response_times = ask_question_to_models(models, q_text, pdf_index)
    
    # Prepara il dizionario per salvare i risultati di ogni modello
    result = {
        "question_id": q_id,
        "question_text": q_text,
        "answer_orca": responses.get("Orca", ""),
        "orca_time": response_times.get("Orca", 0)
    }
    
    # Log del tempo di risposta di Orca per ogni domanda come metrica su MLflow
    mlflow.log_metric("orca_response_time", response_times.get("Orca", 0), step=idx)

    # Aggiungi il risultato alla lista
    results_data.append(result)    

    # Salva la risposta e i tempi nel database per la domanda corrente
    save_to_db(result)
    print(f"Saved response and times for question ID {q_id} to the database.") 

# 6.3 Save the results to a CSV file
csv_file_path = os.path.join(dataset_path, "evaluation_results.csv")
save_to_csv(csv_file_path, results_data)

# Log the CSV file to MLflow as an artifact
mlflow.log_artifact(csv_file_path)

mlflow.log_artifact(os.path.join(dataset_path, "pdf_software_engineering"))

mlflow.log_artifact("question.py")


print("Salvati tutti i risultati su CSV.") 

# Define custom PythonModel class for GPT4All
class GPT4AllPythonModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load or initialize the GPT4All model here if needed
        self.model = GPT4All("/opt/ml/model/")

    def predict(self, context, model_input):
        # Expect model_input to be a DataFrame with 'context' and 'question' columns
        results = []
        for _, row in model_input.iterrows():
            context = row['context']
            question = row['question']
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            response = self.model.generate(prompt)
            results.append(response)
        return results

# Log the GPT4All model
mlflow.pyfunc.log_model(
    artifact_path="gpt4all_model",
    python_model=GPT4AllPythonModel(),
    artifacts={"model_path": os.path.join(model_path, "orca-mini-3b-gguf2-q4_0.gguf")},
    registered_model_name="GPT4All_Orca_Model",
    conda_env={
        "channels": ["defaults"],
        "dependencies": [
            "python=3.9",
            "pip",
            {
                "pip": [
                    "mlflow",
                    "gpt4all",
                    "numpy",
                    "psycopg2-binary",
                    "matplotlib",
                    "seaborn",
                    "pandas",
                    "gguf"
                ]
            }
        ],
        "name": "gpt4all_env"
    }
)

# Chiudi la run di MLflow
mlflow.end_run()