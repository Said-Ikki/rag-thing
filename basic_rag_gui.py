# Created by Said Ikki

import documents
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader , TextLoader, PyPDFLoader , DirectoryLoader, UnstructuredFileLoader
from langchain_core import documents

# for more loaders, go here:
# https://api.python.langchain.com/en/latest/community_api_reference.html#module-langchain_community.document_loaders

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community import embeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough



# Load in LLM
print("loading LLM")
llm = Ollama(model = "llama3")
# add a document, we will use an online article as a dummy for now
# we'll see if we can extend it later
#loader = WebBaseLoader(
   # web_path="https://en.wikipedia.org/wiki/Joker_(character)"
#)
txt_loader = TextLoader(
    file_path="C:/Local Documents/pdf/example_document.txt"
)
pdf_loader = PyPDFLoader(
    file_path="C:/Local Documents/pdf/alice.pdf"
)
# put it into the collection of documents to look through
#docs1 =  loader.load()
docs2 = txt_loader.load()
docs3 = pdf_loader.load()
#docs = loader.load()
# documents need to be cut into smaller pieces
# text_splitter sets the parameters of the cutting,
# modify to see if it works better some other time
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    add_start_index = True
)

docs = DirectoryLoader(
    path="C:/Local Documents/pdf",
    loader_cls=UnstructuredFileLoader,
)
print('load docs')
# actually cut up the docs
all_splits = text_splitter.split_documents(docs.load())
#all_splits.append( text_splitter.split_documents(docs2) )
#all_splits.append( text_splitter.split_documents(docs3) )

# embeddings? idk
embedding = embeddings.OllamaEmbeddings(
    model="nomic-embed-text"
)
print('set up vectore store')
# hold documents in vector store
vectorstore = Chroma.from_documents(
    documents = all_splits,
    embedding = embedding
)
print('set up retriever')
# retriever part of the RAG
retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":6}
)

# chat history prompt for the AI
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
print("contextualize")
# format every prompt along with a chat history and the above prompt
contextualize_q_prompt= ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# chains? idk
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

# Q&A prompt for LLM
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following information to provide accurate answers but act as if you knew this information innately.\
You can use the information provided to help answer the question, but you are not limited to it \
If unsure, simply state that you don't know.\
{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# format docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# make 'context'
def contextualized_question(input: dict):
    if input.get("chat_history"):
        q = contextualize_q_chain
        return q
    else:
        return input["question"]
print( "initialize RAG chain")
rag_chain = (
    RunnablePassthrough.assign(
        context = contextualized_question | retriever | format_docs
    )
    | qa_prompt
    | llm
)

inp = ""
chat_history = []

print("")
print("enter prompts")
'''
while input != "end":
    inp = input("")
    if inp == "end" :
        break
    print("Please Wait while the Magic Pixie living in your computer answers your question")
    ai_msg = rag_chain.invoke(
        {
            "question": inp,
            "chat_history": chat_history
        }
    )
    print(ai_msg)
    chat_history.extend(
        [
            HumanMessage(content=inp), ai_msg
        ]
    )
'''
from flask import Flask, render_template, request
from flask_socketio import SocketIO, send

print("Setting Up Web Server")

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
chat_histories = {}

@socketio.on('connect')
def test_connect():
    print("User Connected! " + request.sid)
    chat_histories[request.sid] = []

@socketio.on('message')
def handle_message(message):
    print("Recieved Message: " + message + " from SID == " + request.sid)
    ai_msg = rag_chain.invoke(
        {
            "question": message,
            "chat_history": chat_histories[request.sid]
        }
    )
    send(ai_msg)
    chat_histories[request.sid].extend(
        [
            HumanMessage(content=message), ai_msg
        ]
    )

@app.route("/")
def index():
    #return render_template("index.html")
    return render_template("gpt-clone.html")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=2000, allow_unsafe_werkzeug=True)
