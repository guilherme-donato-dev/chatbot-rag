import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# LangChain Imports
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain_huggingface import HuggingFaceEmbeddings

# Configuração inicial
load_dotenv()
st.set_page_config(page_title='Chat RAG Inteligente', page_icon='🏈')

PERSIST_DIRECTORY = 'db'

@st.cache_resource
def get_vectorstore():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)

def process_and_add_pdf(file, vector_store):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        if chunks:
            vector_store.add_documents(chunks)
            return True
    except Exception as e:
        st.error(f"Erro ao processar PDF: {e}")
        return False
    finally:
        os.remove(temp_file_path)

def get_context_retriever_chain(vector_store, llm):
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Dada a conversa acima, gere uma query de busca para encontrar informações relevantes para a conversa.")
    ])

    return create_history_aware_retriever(llm, retriever, prompt)

def get_conversational_rag_chain(retriever_chain, llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um assistente que responde perguntas com base EXCLUSIVAMENTE no contexto fornecido abaixo.
        O contexto foi extraído de documentos enviados pelo usuário.
        Se a resposta estiver no contexto, responda com base nele.
        Se não estiver, diga que não encontrou a informação nos documentos.
        NÃO diga que não consegue acessar arquivos — o conteúdo já foi extraído e está no contexto abaixo.

        Contexto:
        {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        output_parser=StrOutputParser()
    )

    return create_retrieval_chain(
        retriever=retriever_chain,
        combine_docs_chain=stuff_documents_chain
    )

# --- INTERFACE GRÁFICA ---

st.header('Chat com seus Documentos (RAG) 🏈')

vector_store = get_vectorstore()

with st.sidebar:
    st.header('Configurações')
    uploaded_files = st.file_uploader("Upload de PDFs", type=['pdf'], accept_multiple_files=True)

    if uploaded_files:
        if st.button("Processar Arquivos"):
            with st.spinner("Processando e indexando..."):
                for pdf in uploaded_files:
                    process_and_add_pdf(pdf, vector_store)
                st.success("Processamento concluído!")

    model_choice = st.selectbox("Modelo", ['gemini-2.5-flash', 'gemini-2.5-pro'])

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if isinstance(message, dict):
        with st.chat_message(message['role']):
            st.write(message['content'])
    else:
        role = "user" if message.type == "human" else "ai"
        with st.chat_message(role):
            st.write(message.content)

user_query = st.chat_input("Digite sua pergunta aqui...")

if user_query:
    st.chat_message("user").write(user_query)

    llm = ChatGoogleGenerativeAI(model=model_choice)  # ← Trocado

    history_aware_retriever = get_context_retriever_chain(vector_store, llm)
    rag_chain = get_conversational_rag_chain(history_aware_retriever, llm)

    with st.spinner("Pensando..."):
        response = rag_chain.invoke({
            "input": user_query,
            "chat_history": st.session_state.chat_history
        })

        answer = response['answer']

        st.chat_message("ai").write(answer)

        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=answer))