from gpt4all import GPT4All
import psycopg2
import json
import time
from rdflib import Graph
import re
import mlflow
import pandas as pd

# Set our tracking server uri for logging
#mlflow.set_tracking_uri(uri="http://16.171.132.68:5000")
mlflow.set_experiment(experiment_name='Final example')
mlflow.start_run(run_name='Planets and satellites - Ontology')
run_id = mlflow.active_run().info.run_id
mlflow.set_tag("Training Info", "Run Orca with RAG on planets and moons ontology file")

mlflow.log_param("model_name", "Mini Orca")
mlflow.log_param("model_type", "LLM")
mlflow.log_param("model_size", "3B")
mlflow.log_param("quantization", "q4_0")
mlflow.log_param("library", "gpt4all")

# Log specific run parameters
mlflow.log_param("number_questions", 60) 
mlflow.log_param("Dataset", "Planets and moons")


# Impostare la connessione al database
def connect_to_db():
    return psycopg2.connect(
        host="13.51.172.225",  
        database="final_example",  
        user="postgres",  
        password="1234"  
    )

import os
model_path = os.path.abspath("Modelli_gpt4all")
dataset_path = os.path.abspath("Dataset")

# 1. Load the local model
model_orca = GPT4All(os.path.join(model_path, "orca-mini-3b-gguf2-q4_0.gguf"), allow_download=False)

# Carica il file JSON con le domande
with open(os.path.join(dataset_path, "esempione.json")) as f:
    questions = json.load(f)

# Load RDF data and set up retrieval function
def load_rdf_data(rdf_file_path):
    graph = Graph()
    graph.parse(rdf_file_path, format="xml")  # Adjust format if necessary
    return graph


def retrieve_context(graph, question):
    # Customize the SPARQL query to extract relevant info based on the question
    # Example: find triples that match certain keywords
    query = """
    SELECT ?subject ?predicate ?object
    WHERE {
        ?subject ?predicate ?object .
        FILTER(CONTAINS(LCASE(STR(?object)), LCASE("{question_text}")))
    }
    """.replace("{question_text}", question)
    results = graph.query(query)
    context = " ".join([str(row.object) for row in results])
    return context


# Connect to RDF file
rdf_graph_1 = load_rdf_data(os.path.join(dataset_path, "ontology_files", "planets.rdf"))
rdf_graph_2 = load_rdf_data(os.path.join(dataset_path, "ontology_files", "satellites.rdf"))
rdf_graph = rdf_graph_1 + rdf_graph_2

def create_table():
    # Crea una connessione al database
    with connect_to_db() as conn:
        with conn.cursor() as cursor:
            # Elimina la tabella se esiste già
            cursor.execute("DROP TABLE IF EXISTS final_ontology;")
            # Creazione tabella (se non esiste)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS final_ontology (
                    id SERIAL PRIMARY KEY,
                    question_text TEXT,
                    ground_truth TEXT,
                    answer_orca TEXT,
                    response_time_orca FLOAT
                );
            """)
            conn.commit()

create_table()

# Define function to query models and save results
def process_questions(questions, graph):
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # SQL command to insert data
    insert_query = """
    INSERT INTO final_ontology (id, question_text, ground_truth, answer_orca, response_time_orca)
    VALUES (%s, %s, %s, %s, %s)
    """
    
    for item in questions:
        question_id = item["id"]
        question_text = item["question"]
        ground_truth = item["ground_truth"]
        
        # Retrieve context from RDF
        context = retrieve_context(graph, question_text)
        input_text = f"Question: {question_text}\nContext: {context}"
        
        # Query each model and record response times
        start_time = time.time()
        answer_orca = model_orca.generate(input_text)
        time_response_orca = time.time() - start_time
        
        # Save results to database
        cursor.execute(insert_query, (question_id, question_text, ground_truth, answer_orca, time_response_orca))
        
    
    conn.commit()
    cursor.close()
    conn.close()

# Process all questions
process_questions(questions, rdf_graph)
print(f"Saved response and times for question ID {id} to the database.")
print("Processing complete, results saved to database.")

# Esporta la tabella `qa_results` in un file CSV
with connect_to_db() as conn:
    query = "SELECT * FROM final_ontology;"
    df = pd.read_sql_query(query, conn)
    csv_path = os.path.join(dataset_path, "qa_results.csv")
    df.to_csv(csv_path, index=False)

mlflow.log_artifact(csv_path)
print(f"File CSV `final_ontology.csv` salvato come artefatto su MLflow.")


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
        cursor.execute("SELECT id, ground_truth, answer_orca, response_time_orca FROM final_ontology;")
        rows = cursor.fetchall()

        total_f1 = 0
        total_response_time = 0
        count = 0

        for row in rows:
            id, ground_truth, answer_orca, time_response_orca = row
            f1_orca, em_orca = calculate_f1_and_exact(answer_orca, ground_truth)
            metric_contains_gt = contains_ground_truth(answer_orca, ground_truth)
    
            # Logga i valori F1 ed EM per ciascun id su MLflow come metriche
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
        mlflow.log_metric("average f1", avg_f1)
        mlflow.log_metric("average response time", avg_response_time)

        print(f"Logged average F1: {avg_f1}, average response time: {avg_response_time}.")

print("F1 ed EM loggati su MLflow per ciascun ID.")


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
                    "gguf",
                    "rdflib"
                ]
            }
        ],
        "name": "gpt4all_env"
    }
)