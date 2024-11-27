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
import re

# Set our tracking server uri for logging #
#mlflow.set_tracking_uri(uri="http://localhost:5000")
mlflow.set_experiment(experiment_name='Final example')
mlflow.start_run(run_name='Planets and moons - PDF')
run_id = mlflow.active_run().info.run_id
mlflow.set_tag("Training Info", "Run Orca with RAG on planets and moons PDF file")

mlflow.log_param("model_name", "Mini Orca")
mlflow.log_param("model_type", "LLM")
mlflow.log_param("model_size", "3B")
mlflow.log_param("quantization", "q4_0")
mlflow.log_param("library", "gpt4all")

# Log specific run parameters
mlflow.log_param("number_questions", 60) 
mlflow.log_param("Dataset", "Astronomy for Mere Mortals v23")

import os
model_path = os.path.abspath("Modelli_gpt4all")
dataset_path = os.path.abspath("Dataset")

# Impostare la connessione al database
def connect_to_db():
    return psycopg2.connect(
        host="13.60.51.238",  
        database="final_example",  
        user="postgres",  
        password="1234"  
    )

def create_table():
    with connect_to_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS final_pdf;")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS final_pdf (
                    id SERIAL PRIMARY KEY,
                    question TEXT,
                    answer_orca TEXT,
                    response_time_orca FLOAT
                );
            """)
            conn.commit()


# Funzione per salvare gradualmente le risposte nel database
def save_to_db(result):
    with connect_to_db() as conn:
        with conn.cursor() as cursor:
            # Inserimento dati nel database
            cursor.execute("""
                INSERT INTO model_rag_responses_se (question, answer_orca, response_time_orca)
                VALUES (%s, %s, %s);
                """, (
            result["question"],
            result["answer_orca"],
            result["response_time_orca"]
        ))
            conn.commit()


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
with open(os.path.join(dataset_path, "esempione.json")) as f:
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
        
        # Verifica se la risposta è una stringa o un dizionario
        if isinstance(response, str):
            response_text = response
        elif isinstance(response, dict) and "choices" in response:
            response_text = response["choices"][0]["text"]
        else:
            response_text = "Error: Unexpected response format"

        # Salva la risposta e il tempo nel dizionario `responses`
        responses[f"{model_name}"] = response_text
        responses[f"time_{model_name}"] = response_time

        # Stampa il progresso della risposta e il tempo impiegato
        print(f"Model: {model_name} - Question: {question[:30]}... - Response: {response_text[:50]}... - Time: {response_time:.2f} seconds")

    return responses  # Ritorna anche i tempi di risposta



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

with connect_to_db() as conn:
    with conn.cursor() as cursor:
        for question_item in questions:
            question = question_item["question"].strip()
            responses = ask_question_to_models(models, question, pdf_index)
            
            cursor.execute("""
                INSERT INTO final_pdf (question, answer_orca, response_time_orca)
                VALUES (%s, %s, %s);
            """, (question, 
                  responses.get("Orca"), responses.get("time_Orca")))
        conn.commit()
        print(f"Saved response and times for question ID to the database.") 


# Aggiungi una nuova colonna ground_truth alla tabella
with connect_to_db() as conn:
    with conn.cursor() as cursor:
        try:
            cursor.execute("ALTER TABLE final_pdf ADD COLUMN ground_truth TEXT;")
            conn.commit()
        except psycopg2.errors.DuplicateColumn:
            conn.rollback()

        # Aggiorna la tabella con i valori ground_truth
        for item in questions:
            answer_value = "; ".join(item["ground_truth"]) if isinstance(item["ground_truth"], list) else item["ground_truth"]
            cursor.execute(
                "UPDATE final_pdf SET ground_truth = %s WHERE question = %s;",
                (answer_value, item["question"].strip())
            )
        conn.commit()

# Esporta la tabella `final_pdf` in un file CSV
with connect_to_db() as conn:
    query = "SELECT * FROM final_pdf;"
    df = pd.read_sql_query(query, conn)
    csv_path = os.path.join(dataset_path, "final_pdf.csv")
    df.to_csv(csv_path, index=False)

mlflow.log_artifact(csv_path)
print(f"File CSV `final_pdf.csv` salvato come artefatto su MLflow.")

# CALCOLA F1 ed EM ################

def calculate_f1_and_exact(predicted, ground_truth):
    # Tokenizza il testo dividendo su spazi e caratteri speciali
    def tokenize(text):
        return re.findall(r'\w+', text.lower())

    pred_tokens = tokenize(predicted)
    gt_tokens = tokenize(ground_truth)

    # Calcola l'F1-score
    common_tokens = set(pred_tokens) & set(gt_tokens)
    num_common = len(common_tokens)

    if num_common == 0:
        f1 = 0.0
    else:
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gt_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)

    # Calcola l'Exact Match
    exact_match = int(predicted.strip().lower() == ground_truth.strip().lower())

    return f1, exact_match

# Calcola F1 ed Exact Match e salva i risultati

def contains_ground_truth(predicted, ground_truth):
    # Tokenizza la ground truth e la risposta
    def tokenize(text):
        return set(re.findall(r'\w+', text.lower()))
    
    gt_tokens = tokenize(ground_truth)
    pred_tokens = tokenize(predicted)
    
    # Ritorna 1 se tutti i token della ground truth sono contenuti nella risposta
    return int(gt_tokens.issubset(pred_tokens))

with connect_to_db() as conn:
    with conn.cursor() as cursor:
        # Seleziona `id` insieme a `ground_truth`, `answer_orca` e `response_time_orca`
        cursor.execute("SELECT id, ground_truth, answer_orca, response_time_orca FROM final_pdf;")
        rows = cursor.fetchall()

        total_f1 = 0
        total_response_time = 0
        total_contains_gt = 0
        count = 0

        for row in rows:
            id, ground_truth, answer_orca, response_time_orca = row
            f1_orca, em_orca = calculate_f1_and_exact(answer_orca, ground_truth)
            metric_contains_gt = contains_ground_truth(answer_orca, ground_truth)

            # Logga i valori F1, EM, e response time per ciascun id su MLflow come metriche
            mlflow.log_metric("f1 orca", f1_orca, step=id)
            mlflow.log_metric("em orca", em_orca, step=id)
            mlflow.log_metric("cgt orca", metric_contains_gt, step=id)
            mlflow.log_metric("response time", response_time_orca, step=id)

            total_f1 += f1_orca
            total_response_time += response_time_orca
            total_contains_gt += metric_contains_gt

            count += 1

            print(f"Logged F1: {f1_orca}, EM: {em_orca} for ID: {id}")

        # Calcola le medie
        avg_f1 = total_f1 / count if count > 0 else 0
        avg_response_time = total_response_time / count if count > 0 else 0

        mlflow.log_metric("total cgt", total_contains_gt)

        # Logga le medie su MLflow
        mlflow.log_metric("average f1", avg_f1)
        mlflow.log_metric("average response time", avg_response_time)

        print(f"Logged average F1: {avg_f1}, average response time: {avg_response_time}.")

print("F1 ed EM loggati su MLflow per ciascun ID, insieme a F1 medio e tempo di risposta medio.")





# Define custom PythonModel class for GPT4All
class GPT4AllPythonModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        pass       

    def predict(self, context, model_input):
        pass        

# Log the GPT4All model
mlflow.pyfunc.log_model(
    artifact_path="gpt4all_model",
    python_model=GPT4AllPythonModel(),
#    artifacts={
#        "model_path": os.path.join(model_path, "orca-mini-3b-gguf2-q4_0.gguf"),
#        "pdf_artifacts": os.path.join(dataset_path, "pdf_software_engineering"),
#        "question_script": "question.py"
#    },
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
                    "gguf",
                    "transformers",
                    "pypdf",
                    "sentence-transformers",
                    "faiss-cpu"
                ]
            }
        ],
        "name": "gpt4all_env"
    }
)

# Chiudi la run di MLflow
mlflow.end_run()