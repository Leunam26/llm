from gpt4all import GPT4All
import json
import numpy as np
import psycopg2
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

# Set our tracking server uri for logging
#mlflow.set_tracking_uri(uri="http://localhost:5000")
mlflow.set_experiment(experiment_name='LLM Mini Orca')
mlflow.start_run(run_name='Run Mini Orca')
run_id = mlflow.active_run().info.run_id
mlflow.set_tag("Training Info", "Run Orca su dataset SQuAD")

# Impostare la connessione al database
def connect_to_db():
    return psycopg2.connect(
        host="localhost",  # Cambia se necessario
        database="llm_evaluation",  # Inserisci il nome del database
        user="postgres",  # Inserisci il nome utente
        password="1234"  # Inserisci la password
    )

import os
model_path = os.path.abspath("Modelli_gpt4all")
dataset_path = os.path.abspath("Dataset")

# Usa os.path.join per unire i percorsi
model_orca = GPT4All(os.path.join(model_path, "orca-mini-3b-gguf2-q4_0.gguf"), allow_download=False)


# 1. Carica il modello locale
#model_orca = GPT4All("Modelli_gpt4all/orca-mini-3b-gguf2-q4_0.gguf", allow_download=False)


# 2. Carica il benchmark SQuAD locale
with open(os.path.join(dataset_path, "dev-v1.1.json")) as f: #versione 1.1 del database
    squad_data = json.load(f)

os.path.join(model_path, "orca-mini-3b-gguf2-q4_0.gguf")

#with open("Dataset/dev-v1.1.json") as f: #versione 1.1 del database
#    squad_data = json.load(f)

# 3. Estrai le prime 300 domande
squad_subset = []
for entry in squad_data['data']:
    for paragraph in entry['paragraphs']:
        for qa in paragraph['qas']:
            squad_subset.append({
                'context': paragraph['context'],
                'question': qa['question'],
                'answer': qa['answers'][0]['text']
            })
            if len(squad_subset) >= 10:
                break
        if len(squad_subset) >= 10:
            break
    if len(squad_subset) >= 10:
        break

# 4. Funzione per fare domande ai modelli usando gpt4all
def ask_question_gpt4all(model, question, context):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    response = model.generate(prompt)
    return response

# 5. Metriche per il calcolo dell'Exact Match (EM) e F1
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

# 6. Funzione per salvare i risultati nel database
def save_to_db(cur, question, context, truth, answer_orca, em_orca, f1_orca):
    cur.execute("""
        INSERT INTO evaluation_results (question, context, ground_truth, answer_orca, em_orca, f1_orca)
        VALUES (%s, %s, %s, %s, %s, %s);
    """, (question, context, truth, answer_orca, em_orca, f1_orca))

# 7. Connessione al database
conn = connect_to_db()
cur = conn.cursor()

# Define custom PythonModel class for GPT4All
class GPT4AllPythonModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load or initialize the GPT4All model here if needed
        self.model = GPT4All("C:/Users/mnico/Documents/GitHub/llm/Modelli_gpt4all/orca-mini-3b-gguf2-q4_0.gguf")

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

try:
    # Itera sulle 300 domande del benchmark locale
    for idx, example in enumerate(squad_subset, 1):
        question = example['question']
        context = example['context']
        truth = example['answer']

        # Ottieni risposte dai modelli
        answer_orca = ask_question_gpt4all(model_orca, question, context)

        # Confronta le risposte con la verità fornita (truth)
        em_orca = exact_match(answer_orca, truth)
        f1_orca = f1(answer_orca, truth)

        # Log delle metriche su MLflow
        mlflow.log_metric("exact_match", em_orca, step=idx)
        mlflow.log_metric("f1_score", f1_orca, step=idx)

        # Salva i risultati nel database
        save_to_db(cur, question, context, truth, answer_orca, em_orca, f1_orca)

        # Conferma l'inserimento dopo ogni iterazione
        conn.commit()

        # Stampa progressiva
        print(f"Salvati {idx} risultati su 10.")

    print("Dati salvati con successo nel database.")

except Exception as e:
    print(f"Errore durante il salvataggio dei dati: {e}")
    conn.rollback()

finally:
    # Chiudi la connessione
    cur.close()
    conn.close()

    # Log the GPT4All model as an MLflow PythonModel
    mlflow.pyfunc.log_model(
        artifact_path="gpt4all_model",
        python_model=GPT4AllPythonModel(),
        registered_model_name="GPT4All_Orca_Model"
    )

    # Chiudi la run di MLflow
    mlflow.end_run()
