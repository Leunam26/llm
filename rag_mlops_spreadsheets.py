import pandas as pd
import json
import time
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import psycopg2
import openpyxl
from gpt4all import GPT4All
import re
import mlflow

# Set our tracking server uri for logging
#mlflow.set_tracking_uri(uri="http://localhost:5000")
mlflow.set_experiment(experiment_name='LLM Mini Orca')
mlflow.start_run(run_name='Run Mini Orca RAG - Spreadsheets')
run_id = mlflow.active_run().info.run_id
mlflow.set_tag("Training Info", "Run Orca with RAG on Champions League spreadsheets")

mlflow.log_param("model_name", "Mini Orca")
mlflow.log_param("model_type", "LLM")
mlflow.log_param("model_size", "3B")
mlflow.log_param("quantization", "q4_0")
mlflow.log_param("library", "gpt4all")

# Log specific run parameters
mlflow.log_param("number_questions", 20) 
mlflow.log_param("Dataset", "Champions League spreadsheets")


# Impostare la connessione al database
def connect_to_db():
    return psycopg2.connect(
        host="13.60.45.22",  # Inserisci nome host
        database="rag_tables_evaluation",  # Inserisci il nome del database
        user="postgres",  # Inserisci il nome utente
        password="1234"  # Inserisci la password
    )

import os
model_path = os.path.abspath("Modelli_gpt4all")
dataset_path = os.path.abspath("Dataset")

# Carica i due file CSV locali
performance_df = pd.read_excel(os.path.join(dataset_path, "xlsx_files", "UCLperformances.xlsx"))
finals_df = pd.read_excel(os.path.join(dataset_path, "xlsx_files", "UCLFinals.xlsx"))
players_df = pd.read_excel(os.path.join(dataset_path, "xlsx_files", "PlayergoalsCL.xlsx"))
coaches_df = pd.read_excel(os.path.join(dataset_path, "xlsx_files", "CoachappersCL.xlsx"))

# 1. Carica i modelli locali
model_orca = GPT4All(os.path.join(model_path, "orca-mini-3b-gguf2-q4_0.gguf"), allow_download=False)

# Configura il prompt
prompt_template = PromptTemplate(template="Domanda: {question}\nContesto: {context}\nRisposta:", input_variables=["question", "context"])

# Funzione per il retrieval dei dati
def retrieve_context(question):
    keywords = question.lower().split()  # Suddivide la domanda in parole chiave
    # Filtro per ogni file CSV in base ai campi che contengono keywords
    performance_results = performance_df[performance_df.apply(lambda row: any(keyword in row.astype(str).str.lower().values for keyword in keywords), axis=1)]
    finals_results = finals_df[finals_df.apply(lambda row: any(keyword in row.astype(str).str.lower().values for keyword in keywords), axis=1)]
    players_results = players_df[players_df.apply(lambda row: any(keyword in row.astype(str).str.lower().values for keyword in keywords), axis=1)]
    coaches_results = coaches_df[coaches_df.apply(lambda row: any(keyword in row.astype(str).str.lower().values for keyword in keywords), axis=1)]
    context = pd.concat([performance_results, finals_results, players_results, coaches_results]).to_string(index=False)
    return context if context else "No relevant information found."


# Funzione principale per generare risposte
def generate_responses(question):
    context = retrieve_context(question)
    prompt = prompt_template.format(question=question, context=context)

    responses = {}
    start_time = time.time()
    responses["orca"] = model_orca.generate(prompt)
    responses["time_orca"] = time.time() - start_time

    return responses

# Carica il file JSON con le domande
with open(os.path.join(dataset_path, "champions_league.json")) as f:
    questions = json.load(f)


def create_table():
    # Crea una connessione al database
    with connect_to_db() as conn:
        with conn.cursor() as cursor:
            # Elimina la tabella se esiste già
            cursor.execute("DROP TABLE IF EXISTS qa_results;")
            # Creazione tabella (se non esiste)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS qa_results (
                    id SERIAL PRIMARY KEY,
                    question TEXT,
                    answer_orca TEXT,
                    response_time_orca FLOAT
                );
            """)
            conn.commit()

create_table()


# Itera sulle domande e salva i risultati
for question_item in questions:
    question = question_item["text"].strip()  # Estrae il testo e rimuove eventuali spazi bianchi
    responses = generate_responses(question)

    # Salva le risposte nel database
    with connect_to_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO qa_results (question, answer_orca, response_time_orca)
                VALUES (%s, %s, %s);
            """, (question, 
                  responses.get("orca"), responses.get("time_orca")))
            conn.commit()

    print(f"Domanda: {question} - Risposte salvate.")

print("Processo completato e risultati salvati nel database PostgreSQL.")

# AGGIUNGI COLONNA ###############

# Aggiungi una nuova colonna ground_truth alla tabella
with connect_to_db() as conn:
    with conn.cursor() as cursor:
        try:
            cursor.execute("ALTER TABLE qa_results ADD COLUMN ground_truth TEXT;")
            conn.commit()
        except psycopg2.errors.DuplicateColumn:
            conn.rollback()

        # Aggiorna la tabella con i valori ground_truth
        for item in questions:
            answer_value = "; ".join(item["answer"]) if isinstance(item["answer"], list) else item["answer"]
            cursor.execute(
                "UPDATE qa_results SET ground_truth = %s WHERE question = %s;",
                (answer_value, item["text"].strip())
            )
        conn.commit()

# Esporta la tabella `qa_results` in un file CSV
with connect_to_db() as conn:
    query = "SELECT * FROM qa_results;"
    df = pd.read_sql_query(query, conn)
    csv_path = os.path.join(dataset_path, "qa_results.csv")
    df.to_csv(csv_path, index=False)

mlflow.log_artifact(csv_path)
print(f"File CSV `qa_results.csv` salvato come artefatto su MLflow.")

mlflow.log_artifact("spreadsheets_question.py")

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
        # Seleziona `id` insieme a `ground_truth` e `answer_orca`
        cursor.execute("SELECT id, ground_truth, answer_orca FROM qa_results;")
        rows = cursor.fetchall()

        for row in rows:
            id, ground_truth, answer_orca = row
            f1_orca, em_orca = calculate_f1_and_exact(answer_orca, ground_truth)
            metric_contains_gt = contains_ground_truth(answer_orca, ground_truth)
    
            # Logga i valori F1 ed EM per ciascun id su MLflow come metriche
            mlflow.log_metric("f1_orca", f1_orca, step=id)
            mlflow.log_metric("em_orca", em_orca, step=id)
            mlflow.log_metric("contains_ground_truth", metric_contains_gt, step=id)
            
            print(f"Logged F1: {f1_orca}, EM: {em_orca} for ID: {id}")

print("F1 ed EM loggati su MLflow per ciascun ID.")