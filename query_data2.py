import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
#from populate_database import vectordb
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are an expert insurance policy assistant.Ise the following clauses to answer the question

Clauses:
{context}

Question:
{question}

Provide a concise , direct answer in plain text without much additional information
Respond only with the relevant data and context.
if information not found in the provided context, reply: "Information not available in the document"

respond ONLY with the answer text

"""


def main():
    # Create CLI.
    # parser = argparse.ArgumentParser()
    # parser.add_argument("query_text", type=str, help="The query text.")
    # args = parser.parse_args()
    # query_text = args.query_text
    # query_rag(query_text)

    while True:
        try:
            print("\n\n------------------------------------------")
            
            query = input("Enter Questions(q to stop): ")
            if query == "q":
                break
            
            process_query(query) 
        except Exception as e:
            print(f"\nError: {e}\n")

def process_query(query_text:str):
    
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory = CHROMA_PATH, embedding_function = embedding_function)

    #similarity search in Db
    similarity_search = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in similarity_search])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    process_response(prompt,similarity_search)

def process_response(prompt,similarity_search)-> str:

    model = Ollama(model="llama3.2:1b")
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in similarity_search]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(response_text)
    #return response_text




if __name__ == "__main__":
    main()
    


