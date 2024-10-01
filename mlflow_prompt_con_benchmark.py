from gpt4all import GPT4All
import json
import numpy as np
import psycopg2
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import pandas as pd

# Set our tracking server uri for logging
#mlflow.set_tracking_uri(uri="http://localhost:5000")
mlflow.set_experiment(experiment_name='LLM Mini Orca')
mlflow.start_run(run_name='Run Mini Orca')
run_id = mlflow.active_run().info.run_id
mlflow.set_tag("Training Info", "Run Orca su dataset SQuAD")

mlflow.log_param("model_name", "Mini Orca")
mlflow.log_param("model_type", "LLM")
mlflow.log_param("model_size", "3B")
mlflow.log_param("quantization", "q4_0")
mlflow.log_param("library", "gpt4all")
    
# Log specific run parameters
mlflow.log_param("number_questions", 10) 
mlflow.log_param("dataset", "SQuAD V1.1")

# Set up the database connection
def connect_to_db():
    return psycopg2.connect(
        host="13.53.41.168",  # Update with the EC2 instance's IPv4 address ("localhost" if local)
        database="llm_evaluation",  # Enter the database name
        user="postgres",  # Enter the username
        password="1234"  # Enter the password
    )

import os
model_path = os.path.abspath("Modelli_gpt4all")
dataset_path = os.path.abspath("Dataset")

# Use os.path.join to concatenate paths
model_orca = GPT4All(os.path.join(model_path, "orca-mini-3b-gguf2-q4_0.gguf"), allow_download=False)

# 1. Load the local model
#model_orca = GPT4All("Modelli_gpt4all/orca-mini-3b-gguf2-q4_0.gguf", allow_download=False)


# 2. Load the SQuAD dataset
with open(os.path.join(dataset_path, "dev-v1.1.json")) as f: #versione 1.1 del database
    squad_data = json.load(f)

os.path.join(model_path, "orca-mini-3b-gguf2-q4_0.gguf")

# with open("Dataset/dev-v1.1.json") as f: #versione 1.1 del database
#    squad_data = json.load(f)

# 3. Extract the first 10 questions
squad_subset = []
for entry in squad_data['data']:
    for paragraph in entry['paragraphs']:
        for qa in paragraph['qas']:
            squad_subset.append({
                'context': paragraph['context'],
                'question': qa['question'],
                'answer': qa['answers'][0]['text']
            })
            if len(squad_subset) >=10:
                break
        if len(squad_subset) >= 10:
            break
    if len(squad_subset) >= 10:
        break

# 4. Function to ask questions to the models using gpt4all
def ask_question_gpt4all(model, question, context):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    response = model.generate(prompt)
    return response

# 5. Metrics for calculating Exact Match (EM) and F1
def exact_match(pred, truth):
    return int(pred.strip().lower() == truth.strip().lower())

def f1(pred, truth):
    pred_tokens = pred.split()
    truth_tokens = truth.split()
    common = set(pred_tokens) & set(truth_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall)

# 6. Function to save the results to the database
def save_to_db(cur, question, context, truth, answer_orca, em_orca, f1_orca):
    cur.execute("""
        INSERT INTO evaluation_results (question, context, ground_truth, answer_orca, em_orca, f1_orca)
        VALUES (%s, %s, %s, %s, %s, %s);
    """, (question, context, truth, answer_orca, em_orca, f1_orca))

#########
# 6.1 Function to save the results to a CSV file
def save_to_csv(file_path, data):
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

# List to store results locally
results_data = []
#########

# 7. Database connection
conn = connect_to_db()
cur = conn.cursor()

# Define custom PythonModel class for GPT4All
class GPT4AllPythonModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Carica il modello dal file python_model.pkl
        model_path = "/opt/ml/model/"
        
        try:
            self.model = mlflow.pyfunc.load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")

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
    
    def query_llm(self, context, question):
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        response = self.model.generate(prompt)
        return response


# Lists to store EM and F1 values for each example
f1_scores = []
em_scores = []

try:
    # Iterate over the 10 questions of the local benchmark
    for idx, example in enumerate(squad_subset, 1):
        question = example['question']
        context = example['context']
        truth = example['answer']

        # Get responses from the models
        answer_orca = ask_question_gpt4all(model_orca, question, context)

        # Compare the responses with the provided ground truth
        em_orca = exact_match(answer_orca, truth)
        f1_orca = f1(answer_orca, truth)

        # Aggiungi i valori di EM e F1 alle liste
        em_scores.append(em_orca)
        f1_scores.append(f1_orca)

        # Log metrics to MLflow
        mlflow.log_metric("exact_match", em_orca, step=idx)
        mlflow.log_metric("f1_score", f1_orca, step=idx)

        # Save the results to the database
        save_to_db(cur, question, context, truth, answer_orca, em_orca, f1_orca)

        # Append data to the results for the CSV/JSON file
        results_data.append({
            'question': question,
            'context': context,
            'ground_truth': truth,
            'answer_orca': answer_orca,
            'em_orca': em_orca,
            'f1_orca': f1_orca
        })

        # Confirm the insertion after each iteration
        conn.commit()

        # Print progress
        print(f"Salvati {idx} risultati su 10.")

    # Calculate the mean EM and F1
    mean_em = statistics.mean(em_scores)
    mean_f1 = statistics.mean(f1_scores)

    # Log the mean EM and F1 to MLflow
    mlflow.log_metric("mean_exact_match", mean_em)
    mlflow.log_metric("mean_f1_score", mean_f1)

    # 6.3 Save the results to a CSV file
    csv_file_path = os.path.join(dataset_path, "evaluation_results.csv")
    save_to_csv(csv_file_path, results_data)

    # Log the CSV file to MLflow as an artifact
    mlflow.log_artifact(csv_file_path)

    print("Dati salvati con successo nel database.")

except Exception as e:
    print(f"Errore durante il salvataggio dei dati: {e}")
    conn.rollback()

finally:
    # Close the connection to the Postgres database
    cur.close()
    conn.close()

    # Log the GPT4All model as an MLflow PythonModel
    mlflow.pyfunc.log_model(
        artifact_path="gpt4all_model",
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
