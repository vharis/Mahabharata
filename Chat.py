from langchain_ollama import OllamaEmbeddings

from langchain_community.vectorstores import FAISS 
# from langchain_community.docstore.in_memory import InMemoryDocstore
import streamlit as st
from langchain_ollama import ChatOllama


from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough 
from langchain_core.prompts import ChatPromptTemplate
# from opik.integrations.langchain import OpikTracer
from langchain_community.chat_message_histories import SQLChatMessageHistory
# import opik
# import os

# embeddings = OllamaEmbeddings(model='nomic-embed-text')

embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url='http://localhost:11434')

db_name = "Mahabhararta_KMGanguli"
vector_store = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)
question = "Type your question?"
docs = vector_store.search(query=question, k=5, search_type="similarity")

# retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs = {'k': 3})

# os.environ["OPIK_BASE_URL"] = "http://localhost:5173/api"

# opik.configure(use_local=True)

# opik_tracer = OpikTracer(tags=["langchain", "ollama"])

prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Answer in bullet points. Make sure your answer is relevant to the question and it is answered from the context only.
    Question: {question}
    Answer:
"""

prompt = ChatPromptTemplate.from_template(prompt)



st.title("FIRST Chatbot v0.2")
#st.write("I'm your financial assistant. I can help you with your financial queries.")
# user_id = st.text_input("Enter your user id", "vharis")

def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# if st.button("Start New Conversation"):
#     st.session_state.chat_history = []
#     # history = get_session_history(user_id)
#     history = get_session_history("user_id")
#     history.clear()


for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

llm = ChatOllama(model='granite3.1-dense:latest', base_url='http://localhost:11434')
# .with_config({"callbacks": [opik_tracer]})

rag_chain = prompt | llm | StrOutputParser()



runnable_with_history = RunnableWithMessageHistory(rag_chain, get_session_history, 
                                                   input_messages_key='question', 
                                                   history_messages_key='history')

def chat_with_llm(session_id, input):
    for output in runnable_with_history.stream({'question': input}, config={'configurable': {'session_id': session_id}}):
        yield output


query = st.chat_input("Ask your FIRST question?")
# st.write(prompt)

if query:
    st.session_state.chat_history.append({'role': 'user', 'content': query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        response = st.write_stream(chat_with_llm("user_id", query))

    st.session_state.chat_history.append({'role': 'assistant', 'content': response})