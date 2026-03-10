import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# LangChain Imports
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser   

# Configuração inicial
load_dotenv()
st.set_page_config(page_title='Chat RAG Inteligente', page_icon='🏈')

# Definição de constantes
PERSIST_DIRECTORY = 'db'

# Cache para evitar recarregar o modelo/banco a cada interação
@st.cache_resource
def get_vectorstore():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(PERSIST_DIRECTORY):
        return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
    return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)

def process_and_add_pdf(file, vector_store):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        
        # Dividir texto em chunks (Aumentei overlap para não perder contexto entre quebras)
        from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    
    # Prompt que instrui a IA a reformular a pergunta baseada no histórico
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Dada a conversa acima, gere uma query de busca para encontrar informações relevantes para a conversa.")
    ])
    
    return create_history_aware_retriever(llm, retriever, prompt)

def get_conversational_rag_chain(retriever_chain, llm):
    # --- DEBUG: Mostra no terminal quem está chegando vazio ---
    print(f"DEBUG CHECK -> LLM: {type(llm)}")
    
    if llm is None:
        raise ValueError("ERRO CRÍTICO: A variável 'llm' é None. Verifique sua API Key.")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Responda às perguntas do usuário com base no contexto abaixo:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    # --- CORREÇÃO: Passando output_parser explicitamente ---
    # O erro NoneType muitas vezes acontece porque o LangChain 0.3 
    # tenta adivinhar o parser e falha. Aqui forçamos ele.
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

# Inicializa o VectorStore (com cache)
vector_store = get_vectorstore()

# Sidebar
with st.sidebar:
    st.header('Configurações')
    uploaded_files = st.file_uploader("Upload de PDFs", type=['pdf'], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Processar Arquivos"):
            with st.spinner("Processando e indexando..."):
                for pdf in uploaded_files:
                    process_and_add_pdf(pdf, vector_store)
                st.success("Processamento concluído!")

    model_choice = st.selectbox("Modelo", ['gpt-3.5-turbo', 'gpt-4o'])

# Inicializa Histórico
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Exibe mensagens anteriores
for message in st.session_state.chat_history:
    if isinstance(message, dict): # Formato simples para exibição
         with st.chat_message(message['role']):
            st.write(message['content'])
    else: # Formato LangChain (HumanMessage/AIMessage)
        role = "user" if message.type == "human" else "ai"
        with st.chat_message(role):
            st.write(message.content)

# Input do Usuário
user_query = st.chat_input("Digite sua pergunta aqui...")

if user_query:
    # Mostra pergunta do usuário
    st.chat_message("user").write(user_query)
    
    llm = ChatOpenAI(model=model_choice)
    
    # 1. Cria retriever que entende histórico
    history_aware_retriever = get_context_retriever_chain(vector_store, llm)
    
    # 2. Cria a chain final de resposta
    rag_chain = get_conversational_rag_chain(history_aware_retriever, llm)
    
    with st.spinner("Pensando..."):
        # A mágica acontece aqui: passamos o chat_history para a chain
        response = rag_chain.invoke({
            "input": user_query,
            "chat_history": st.session_state.chat_history
        })
        
        answer = response['answer']
        
        # Mostra resposta
        st.chat_message("ai").write(answer)
        
        # Atualiza histórico (usando objetos do LangChain para manter o contexto correto)
        from langchain_core.messages import HumanMessage, AIMessage
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=answer))
