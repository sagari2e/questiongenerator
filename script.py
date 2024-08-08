import logging
import sys
import os
from llama_index.core import Settings
from llama_index.core.service_context import set_global_service_context
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from huggingface_hub import login
try:
  from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader, StorageContext, load_index_from_storage
except ImportError:
  from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader, StorageContext, load_index_from_storage
 
# Logging configuration
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# HuggingFace login
login("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")

PERSIST_DIR = "./chroma_db"

def remove_folder(folder_path):
    try:
        if os.path.exists(folder_path):
            # Remove the directory and all its contents
            shutil.rmtree(folder_path)
            print(f"Folder '{folder_path}' has been removed successfully.")
        else:
            print(f"Folder '{folder_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred while removing the folder: {e}")

def load_data():
    global index, chat_engine, memory
    reader = SimpleDirectoryReader(input_dir="./files", recursive=True)
    docs = reader.load_data()

    llm = HuggingFaceInferenceAPI(
        temperature=0.0, num_output=2048, model_name="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    model_name = "BAAI/bge-large-en"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embed_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    service_context = ServiceContext.from_defaults(
        chunk_size=1000,
        chunk_overlap=100,
        embed_model=embed_model,
        llm=llm
    )
    set_global_service_context(service_context)

    Settings.llm = llm
    Settings.embed_model = embed_model

    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

    from llama_index.core.memory import ChatMemoryBuffer
    memory = ChatMemoryBuffer.from_defaults(token_limit=16392)
    chat_engine = index.as_chat_engine(chat_mode="condense_question", memory=memory, llm=Settings.llm, verbose=True)

def ask_question(question):
    global chat_engine
    response = chat_engine.chat(question)
    return response.response

def save_response_to_file(response, filename):
    with open(filename+".txt", "w") as file:
        file.write(response + "\n")

def ask_main(question, filename):
    remove_folder(PERSIST_DIR)

    load_data()
    # while True:
    #     question = input("Enter your question (or type 'exit' to quit): ")
    #     if question.lower() == 'exit':
    #         break
    response = ask_question(question)
    print("Response:", response)
    save_response_to_file(response, filename)
    print("Response saved to chat_response.txt")

if __name__ == "__main__":
    main()
