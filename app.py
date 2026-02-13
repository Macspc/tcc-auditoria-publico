import streamlit as st
import os
import tempfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="IA Auditoria Municipal", layout="wide")

# Esconder menus do Streamlit
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- CARREGAR CHAVES (SECRETS) ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
else:
    st.error("Erro: Chaves de API não encontradas. Configure os Secrets no Streamlit Cloud.")
    st.stop()

# --- FUNÇÕES DO SISTEMA (BACKEND) ---

@st.cache_resource
def get_vectorstore():
    """Conecta ao Pinecone e retorna o banco vetorial"""
    
    # --- AQUI ESTAVA O ERRO! AGORA ESTÁ CORRIGIDO ---
    # Usamos o nome exato que o diagnóstico encontrou
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    index_name = "tcc-auditoria" 
    
    vectorstore = PineconeVectorStore(
        index_name=index_name, 
        embedding=embeddings
    )
    return vectorstore

def process_pdf(uploaded_file):
    """Lê o PDF, quebra em pedaços e salva no Pinecone"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()

        # Quebra o texto em pedaços de 1000 caracteres
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        vectorstore = get_vectorstore()
        vectorstore.add_documents(splits)
        
        os.remove(tmp_file_path)
        return True, f"Sucesso! {len(splits)} trechos processados e indexados."
    except Exception as e:
        return False, str(e)

def get_resposta(pergunta, perfil):
    """Gera a resposta usando Google Gemini"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-Pro", temperature=0.3)
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    if perfil == "server":
        system_prompt = (
            "Você é um Auditor Assistente especializado em legislação municipal. "
            "Responda à pergunta do funcionário público baseando-se EXCLUSIVAMENTE no contexto fornecido. "
            "Cite o nome da Lei, o Artigo e o Parágrafo sempre que possível. "
            "Se a informação não estiver no contexto, afirme que não consta na base de dados. "
            "Contexto Legal:\n{context}"
        )
    else: # Perfil Cidadão
        system_prompt = (
            "Você é um Assistente Virtual da Prefeitura. "
            "Explique a resposta de forma simples para um cidadão leigo. "
            "Evite termos jurídicos complexos. "
            "Use o contexto abaixo como base. "
            "Contexto:\n{context}"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": pergunta})
    return response["answer"]

# --- INTERFACE (FRONTEND) ---

query_params = st.query_params
modo = query_params.get("mode", "cidadao")

if modo == "server":
    st.info("🔓 Modo Servidor Público - Acesso Completo")
    
    with st.expander("📂 Alimentar Base de Conhecimento (Upload PDF)"):
        uploaded_file = st.file_uploader("Escolha uma Lei ou Edital", type="pdf")
        if uploaded_file and st.button("Processar Documento"):
            with st.spinner("Processando Inteligência Artificial..."):
                sucesso, msg = process_pdf(uploaded_file)
                if sucesso:
                    st.success(msg)
                    st.balloons()
                else:
                    st.error(f"Erro: {msg}")
    
    st.divider()
    st.subheader("💬 Chat de Auditoria Técnica")

else:
    st.success("👋 Olá! Sou o Assistente Virtual da Prefeitura.")
    st.subheader("💬 Tire suas dúvidas sobre leis municipais")

# --- CHATBOT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Digite sua pergunta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando legislação..."):
            try:
                resposta = get_resposta(prompt, modo)
                st.markdown(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})
            except Exception as e:
                st.error(f"Erro ao gerar resposta: {e}")

