import os
from gpt4all import GPT4All
from wikipediaapi import Wikipedia

# Configurazione dei percorsi dei modelli
model_path = os.path.abspath("Modelli_gpt4all")
model_orca = GPT4All(os.path.join(model_path, "orca-mini-3b-gguf2-q4_0.gguf"), allow_download=False)

# Classe wrapper per i modelli LLM
class LocalLLM:
    def __init__(self, model):
        self.model = model

    def __call__(self, prompt):
        return self.model.generate(prompt)

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

qa_orca = RetrievalQA(orca, retriever)

# Funzione per interagire con il modello
def main():
    while True:
        query = input("Enter your question (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        answer = qa_orca(query)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
