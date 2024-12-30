from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import SQLChatMessageHistory

#ssh -i "/Users/vharis/Downloads/Downloads/ollama16gb.pem" ec2-user@ec2-54-83-95-47.compute-1.amazonaws.com
embeddings = OllamaEmbeddings(
    model='nomic-embed-text', base_url='http://localhost:11434')

db_name = "Mahabhararta_KMGanguli"
vector_store = FAISS.load_local(
    db_name, embeddings, allow_dangerous_deserialization=True)
question = "Type your question?"
docs = vector_store.search(query=question, k=5, search_type="similarity")
context = "mahabharata"
prompt = """
    You are an assistant for question-answering tasks and acts as an expert of Indian epic Mahabharata. Make sure your answers are 5 or 7 bullet points only and keep the answers concise and provide the source of the answer. 
    
    If the question is out side of "Mahabharata" scope, simply say "I don't know". Do not hallucinate.
    
    Question: {question}
    Answer:
"""

prompt = ChatPromptTemplate.from_template(prompt)
st.title("MAHABHARATA Chatbot v0.2")

def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

llm = ChatOllama(model='granite3.1-dense:latest',
                 base_url='http://localhost:11434')

rag_chain = prompt | llm | StrOutputParser()


runnable_with_history = RunnableWithMessageHistory(rag_chain, get_session_history,
                                                   input_messages_key='question',
                                                   history_messages_key='history')
def chat_with_llm(session_id, input):
    for output in runnable_with_history.stream({'question': input}, config={'configurable': {'session_id': session_id}}):
        yield output

query = st.chat_input("Ask your MAHABHARATA question?")

if query:
    st.session_state.chat_history.append({'role': 'user', 'content': query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        response = st.write_stream(chat_with_llm("user_id", query))

    st.session_state.chat_history.append(
        {'role': 'assistant', 'content': response})
