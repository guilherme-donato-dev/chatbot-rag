# 🤖 Chatbot RAG Inteligente

> Converse com seus documentos PDF usando Inteligência Artificial com memória de contexto e recuperação semântica de informações.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=for-the-badge)

---

## 📋 Sobre o Projeto

Este projeto é um **chatbot com RAG (Retrieval-Augmented Generation)** — uma técnica de IA que permite que o modelo responda perguntas com base em documentos reais enviados pelo usuário, em vez de depender apenas do seu conhecimento pré-treinado.

O fluxo funciona assim:

1. O usuário faz upload de um ou mais PDFs
2. O sistema quebra os documentos em chunks e os transforma em vetores semânticos usando embeddings do HuggingFace
3. Esses vetores são armazenados em um banco de dados vetorial (ChromaDB)
4. Quando o usuário faz uma pergunta, o sistema busca os chunks mais relevantes e os envia como contexto para o Google Gemini
5. O modelo responde com base **exclusivamente** no conteúdo dos documentos enviados

---

## ✨ Funcionalidades

- 📄 **Upload de múltiplos PDFs** simultaneamente
- 🧠 **RAG com memória de conversa** — o chatbot lembra do histórico da conversa ao formular buscas
- 🔍 **Transparência do contexto** — é possível visualizar exatamente quais trechos dos documentos foram usados para gerar a resposta
- 🤖 **Seleção de modelo** — suporte a `gemini-2.5-flash` e `gemini-2.5-pro`
- 💾 **Persistência do banco vetorial** — documentos indexados ficam disponíveis entre sessões
- 🚫 **Respostas honestas** — o modelo informa quando a resposta não está nos documentos enviados

---

## 🛠️ Tecnologias Utilizadas

| Tecnologia | Função |
|---|---|
| **Streamlit** | Interface web interativa |
| **LangChain** | Orquestração das chains de RAG |
| **Google Gemini API** | Modelo de linguagem (LLM) |
| **HuggingFace Embeddings** | Geração de vetores semânticos (`all-MiniLM-L6-v2`) |
| **ChromaDB** | Banco de dados vetorial para armazenar os chunks |
| **PyPDF** | Leitura e extração de texto dos PDFs |
| **python-dotenv** | Gerenciamento de variáveis de ambiente |

---

## 🏗️ Arquitetura RAG

```
PDF Upload
    │
    ▼
PyPDFLoader ──► RecursiveCharacterTextSplitter
                        │
                        ▼ (chunks de 1000 chars, overlap 200)
              HuggingFaceEmbeddings
              (all-MiniLM-L6-v2)
                        │
                        ▼
                   ChromaDB (persistido em ./db)
                        │
         ───────────────┴───────────────
         │                             │
    Pergunta do                  Histórico da
      usuário                     conversa
         │                             │
         └──────────────┬──────────────┘
                        ▼
          history_aware_retriever
          (reformula a query com contexto)
                        │
                        ▼
              Top-4 chunks mais relevantes
                        │
                        ▼
              Google Gemini (LLM)
                        │
                        ▼
                   Resposta final
```

---

## 🚀 Como Rodar Localmente

### Pré-requisitos

- Python 3.10+
- Uma chave de API do [Google AI Studio](https://aistudio.google.com/)

### 1. Clone o repositório

```bash
git clone https://github.com/guilherme-donato-dev/chatbot-rag.git
cd chatbot-rag
```

### 2. Crie e ative um ambiente virtual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

> ⚠️ A instalação do `sentence-transformers` pode demorar alguns minutos na primeira vez, pois faz o download do modelo de embeddings.

### 4. Configure as variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
GOOGLE_API_KEY=sua_chave_aqui
```

### 5. Execute a aplicação

```bash
streamlit run app.py
```

A aplicação estará disponível em `http://localhost:8501`

---

## 📖 Como Usar

1. **Faça o upload** de um ou mais PDFs pelo painel lateral (sidebar)
2. Clique em **"Processar Arquivos"** e aguarde a indexação
3. Escolha o **modelo Gemini** desejado (flash é mais rápido, pro é mais preciso)
4. **Digite sua pergunta** no campo de chat
5. Opcionalmente, expanda **"Ver contexto recuperado"** para ver os trechos do documento usados na resposta

---

## 📁 Estrutura do Projeto

```
chatbot-rag/
├── app.py              # Código principal da aplicação
├── requirements.txt    # Dependências do projeto
├── Procfile            # Configuração para deploy (Streamlit Cloud)
├── packages.txt        # Dependências de sistema (para deploy)
├── .gitignore
├── .env                # Variáveis de ambiente (NÃO versionar)
└── db/                 # Banco vetorial ChromaDB (gerado automaticamente)
```

---

## 🔧 Decisões Técnicas

**Por que HuggingFace para embeddings e não a API do Google?**
O modelo `all-MiniLM-L6-v2` roda localmente, sem custo por requisição e sem latência de rede. Para gerar embeddings de centenas de chunks de PDF, isso é muito mais eficiente e econômico do que usar uma API externa.

**Por que ChromaDB com persistência local?**
A persistência em disco evita reprocessar os PDFs a cada reinicialização da aplicação. Os documentos ficam indexados permanentemente na pasta `./db`.

**Por que `history_aware_retriever`?**
Sem ele, perguntas de acompanhamento como *"e quanto ao segundo ponto?"* seriam buscadas literalmente no banco vetorial, sem contexto. O `history_aware_retriever` usa o histórico da conversa para reformular a query antes da busca, tornando o chatbot muito mais preciso em conversas longas.

---

## 🌐 Deploy

Este projeto está configurado para deploy no **Streamlit Community Cloud**. O `Procfile` e o `packages.txt` contêm as configurações necessárias para o ambiente de produção.

> **Nota:** Em ambientes de deploy com sistema de arquivos efêmero (como Streamlit Cloud), o banco ChromaDB é recriado a cada reinicialização. Para persistência real em produção, recomenda-se migrar para um banco vetorial gerenciado como **Pinecone** ou **Qdrant Cloud**.

---

## 👨‍💻 Autor

**Guilherme Donato**

[![GitHub](https://img.shields.io/badge/GitHub-guilherme--donato--dev-181717?style=flat-square&logo=github)](https://github.com/guilherme-donato-dev)
