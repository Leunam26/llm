import os
import pandas as pd
import json
from gpt4all import GPT4All
from langchain.prompts import PromptTemplate

# Imposta i percorsi
model_path = os.path.abspath("Modelli_gpt4all")
dataset_path = os.path.abspath("Dataset")

# Carica i dataset
performance_df = pd.read_excel(os.path.join(dataset_path, "xlsx_files", "UCLperformances.xlsx"))
finals_df = pd.read_excel(os.path.join(dataset_path, "xlsx_files", "UCLFinals.xlsx"))
players_df = pd.read_excel(os.path.join(dataset_path, "xlsx_files", "PlayergoalsCL.xlsx"))
coaches_df = pd.read_excel(os.path.join(dataset_path, "xlsx_files", "CoachappersCL.xlsx"))

# Carica il modello
model_orca = GPT4All(os.path.join(model_path, "orca-mini-3b-gguf2-q4_0.gguf"), allow_download=False)

# Definisce il prompt template
prompt_template = PromptTemplate(template="Domanda: {question}\nContesto: {context}\nRisposta:", input_variables=["question", "context"])

def retrieve_context(question):
    """Funzione per il retrieval del contesto dal dataset."""
    keywords = question.lower().split()
    # Filtro per ogni dataset in base ai campi contenenti le keywords
    performance_results = performance_df[performance_df.apply(lambda row: any(keyword in row.astype(str).str.lower().values for keyword in keywords), axis=1)]
    finals_results = finals_df[finals_df.apply(lambda row: any(keyword in row.astype(str).str.lower().values for keyword in keywords), axis=1)]
    players_results = players_df[players_df.apply(lambda row: any(keyword in row.astype(str).str.lower().values for keyword in keywords), axis=1)]
    coaches_results = coaches_df[coaches_df.apply(lambda row: any(keyword in row.astype(str).str.lower().values for keyword in keywords), axis=1)]
    # Combina i risultati
    context = pd.concat([performance_results, finals_results, players_results, coaches_results]).to_string(index=False)
    return context if context else "No relevant information found."

def query_model(question):
    """Funzione per porre una domanda al modello."""
    context = retrieve_context(question)
    prompt = prompt_template.format(question=question, context=context)
    response = model_orca.generate(prompt)
    return response

if __name__ == "__main__":
    # Sostituisci questa variabile `question` con la tua domanda
    question = "Come si chiama il calciatore che ha segnato più goal in Champions League?"
    
    # Esegui la query e stampa la risposta
    response = query_model(question)
    print(f"Domanda: {question}\nRisposta del modello: {response}")
