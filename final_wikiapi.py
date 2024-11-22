import time
import json
import psycopg2
from gpt4all import GPT4All
from wikipediaapi import Wikipedia

import mlflow
import pandas as pd
import re

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://localhost:5000")
mlflow.set_experiment(experiment_name='Final example')
mlflow.start_run(run_name='Planets and moons - Wiki API')
run_id = mlflow.active_run().info.run_id
mlflow.set_tag("Training Info", "Run Orca with RAG on planets and moons Wiki API")

mlflow.log_param("model_name", "Mini Orca")
mlflow.log_param("model_type", "LLM")
mlflow.log_param("model_size", "3B")
mlflow.log_param("quantization", "q4_0")
mlflow.log_param("library", "gpt4all")

# Log specific run parameters
mlflow.log_param("number_questions", 60) 
mlflow.log_param("Dataset", "Wiki API")


#ricordati di inserire nel file yaml il pacchetto wikipedia-api##################
# Configurazione dei percorsi dei modelli
import os
model_path = os.path.abspath("Modelli_gpt4all")
dataset_path = os.path.abspath("Dataset")
# 1. Carica i modelli locali
model_orca = GPT4All(os.path.join(model_path, "orca-mini-3b-gguf2-q4_0.gguf"), allow_download=False)

# Classe wrapper per i modelli LLM
class LocalLLM:
    def __init__(self, model):
        self.model = model

    def __call__(self, prompt):
        return self.model.generate(prompt)

# Inizializzazione dei modelli
orca = LocalLLM(model_orca)

# Configurazione del retriever di Wikipedia
class WikipediaRetriever:
    def __init__(self):
        self.wiki = Wikipedia(language='en', user_agent="RAG_project/1.0 (manuelnicolosi2000@gmail.com)")

    def retrieve(self, query):
        page = self.wiki.page(query)
        if page.exists():
            return page.summary
        else:
            return "No relevant information found."

retriever = WikipediaRetriever()

# Classe per la catena di RAG
class RetrievalQA:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def __call__(self, query):
        context = self.retriever.retrieve(query)
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        return self.llm(prompt)

# Inizializzazione delle catene di RAG per ogni modello
qa_orca = RetrievalQA(orca, retriever)

# Configurazione della connessione al database PostgreSQL
def connect_to_db():
    return psycopg2.connect(
        host="localhost",
        database="final_example",
        user="postgres",
        password="1234"
    )

def create_table():
    # Crea una connessione al database
    with connect_to_db() as conn:
        with conn.cursor() as cursor:
            # Elimina la tabella se esiste già
            cursor.execute("DROP TABLE IF EXISTS final_wikiapi;")
            # Creazione tabella (se non esiste)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS final_wikiapi (
                    id SERIAL PRIMARY KEY,
                    question TEXT,
                    answer_orca TEXT,
                    time_response_orca FLOAT,
                    ground_truth TEXT
                );
            """)
            conn.commit()

create_table()

# Funzione principale
def main():
    # Caricamento delle domande dal file JSON
    with open(os.path.join(dataset_path, "esempione_prova.json")) as f:
        questions = json.load(f)

    # Inizializzazione della connessione al database
    conn = connect_to_db()
    cur = conn.cursor()

    for idx, question in enumerate(questions):
        id = question["id"]
        question_text = question["question"]
        ground_truth = question.get("ground_truth", "")

        print(f"Processing question {idx + 1}/{len(questions)}: {question_text}")

        # Generazione delle risposte e misurazione dei tempi
        start_time = time.time()
        answer_orca = qa_orca(question_text)
        time_response_orca = time.time() - start_time

        # Salvataggio delle risposte nel database
        save_to_db(cur, id, question_text, ground_truth, answer_orca, time_response_orca)
        conn.commit()

        # Mostra il progresso
        print(f"Saved responses for question {idx + 1}/{len(questions)}")

    # Chiusura delle risorse
    cur.close()
    conn.close()

# Funzione to save results in database
def save_to_db(cur, id, question, ground_truth, answer_orca, time_response_orca):
    cur.execute("""
        INSERT INTO final_wikiapi (id, question, ground_truth, answer_orca, time_response_orca)
        VALUES (%s, %s, %s, %s, %s);
    """, (id, question, ground_truth, answer_orca, time_response_orca))

if __name__ == "__main__":
    main()


# Esporta la tabella `final_wikiapi` in un file CSV
with connect_to_db() as conn:
    query = "SELECT * FROM final_wikiapi;"
    df = pd.read_sql_query(query, conn)
    csv_path = os.path.join(dataset_path, "final_wikiapi.csv")
    df.to_csv(csv_path, index=False)

mlflow.log_artifact(csv_path)
print(f"File CSV `final_wikiapi.csv` salvato come artefatto su MLflow.")


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
        cursor.execute("SELECT id, ground_truth, answer_orca, time_response_orca FROM final_wikiapi;")
        rows = cursor.fetchall()

        total_f1 = 0
        total_response_time = 0
        count = 0

        for row in rows:
            id, ground_truth, answer_orca, time_response_orca = row
            f1_orca, em_orca = calculate_f1_and_exact(answer_orca, ground_truth)
            metric_contains_gt = contains_ground_truth(answer_orca, ground_truth)

            # Logga i valori F1, EM, e response time per ciascun id su MLflow come metriche
            mlflow.log_metric("f1 orca", f1_orca, step=id)
            mlflow.log_metric("em orca", em_orca, step=id)
            mlflow.log_metric("cgt orca", metric_contains_gt, step=id)
            mlflow.log_metric("response time", time_response_orca, step=id)

            total_f1 += f1_orca
            total_response_time += time_response_orca
            count += 1

            print(f"Logged F1: {f1_orca}, EM: {em_orca} for ID: {id}")

        # Calcola le medie
        avg_f1 = total_f1 / count if count > 0 else 0
        avg_response_time = total_response_time / count if count > 0 else 0

        # Logga le medie su MLflow
        mlflow.log_metric("average f1 orca", avg_f1)
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
#        "ontology_artifacts": os.path.join(dataset_path, "ontology_files"),
#        "question_script": "ontology_question.py"
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
                    "pandas",
                    "gguf"
                ]
            }
        ],
        "name": "gpt4all_env"
    }
)
