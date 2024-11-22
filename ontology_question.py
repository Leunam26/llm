from gpt4all import GPT4All
import time
from rdflib import Graph
import os
import re

# Percorsi dove sono stati salvati gli artifact (modello e PDF)
model_path = "/opt/ml/model/artifacts/orca-mini-3b-gguf2-q4_0.gguf"
dataset_path = "/opt/ml/model/artifacts/ontology_files"

# Carica il modello GPT4All .gguf
model_orca = GPT4All(model_path, allow_download=False)

# Funzione per caricare i dati RDF e configurare il retrieval
def load_rdf_data(rdf_file_path):
    graph = Graph()
    graph.parse(rdf_file_path, format="xml")  # Adatta il formato se necessario
    return graph

def retrieve_context(graph, question):
    # Personalizza la query SPARQL per estrarre informazioni rilevanti basate sulla domanda
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

# Connetti il grafo RDF
rdf_graph = load_rdf_data(os.path.join(dataset_path, "owlapi.xrdf"))

# Loop per accettare input dell'utente e fare domande al modello
print("Inserisci la tua domanda (digita 'ctrl + c' per terminare):")
while True:
    question_text = input("Domanda: ")
    if question_text.lower() == "exit":
        break
    
    # Recupera il contesto dall'RDF basato sulla domanda
    context = retrieve_context(rdf_graph, question_text)
    input_text = f"Question: {question_text}\nContext: {context}"
    
    answer_orca = model_orca.generate(input_text)
    
    # Stampa la risposta e il tempo impiegato
    print(f"Risposta del modello: {answer_orca}")
