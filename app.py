import streamlit as st
import os
import time
import tempfile
import hashlib
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

# --- CARREGAR CHAVES ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
else:
    st.error("Erro: Chaves de API não encontradas.")
    st.stop()

# --- FUNÇÕES DO SISTEMA ---

@st.cache_resource
def get_vectorstore():
    """Conecta ao Pinecone usando o modelo CORRETO para sua conta"""
    # IMPORTANTE: Usando o modelo que seu diagnóstico descobriu
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    index_name = "tcc-auditoria" 
    
    vectorstore = PineconeVectorStore(
        index_name=index_name, 
        embedding=embeddings
    )
    return vectorstore

def process_pdf(uploaded_file):
    """Processa PDF com Verificação de Duplicidade (Hash MD5)"""
    try:
        # 1. Lê o arquivo para calcular o HASH (DNA do arquivo)
        file_content = uploaded_file.read()
        file_hash = hashlib.md5(file_content).hexdigest()
        
        # Reseta o ponteiro do arquivo para o início (pois acabamos de ler tudo)
        uploaded_file.seek(0)

        # 2. Verifica se esse Hash já existe no Pinecone
        vectorstore = get_vectorstore()
        
        # Faz uma busca "dummy" filtrando apenas por esse Hash
        # Se retornar algo, é porque o arquivo já está lá
        try:
            results = vectorstore.similarity_search(
                "teste", 
                k=1, 
                filter={"file_hash": file_hash}
            )
            if results:
                return False, "⚠️ Este documento JÁ FOI processado anteriormente! Não é necessário enviar novamente."
        except:
            # Se der erro na busca (ex: index vazio), apenas ignora e continua
            pass

        # 3. Se não existe, cria arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        # 4. Carrega e Divide
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        
        # 5. Adiciona o Hash nos Metadados de cada pedacinho
        for split in splits:
            split.metadata["file_hash"] = file_hash
            split.metadata["file_name"] = uploaded_file.name

        total_chunks = len(splits)
        st.write(f"📄 Documento novo detectado! Gerou {total_chunks} partes.")

        # 6. Envio com Rate Limit (Anti-Erro 429)
        batch_size = 5 
        progress_bar = st.progress(0, text="Iniciando processamento...")

        for i in range(0, total_chunks, batch_size):
            batch = splits[i : i + batch_size]
            sucesso_lote = False
            tentativas = 0
            
            while not sucesso_lote and tentativas < 3:
                try:
                    vectorstore.add_documents(batch)
                    sucesso_lote = True
                except Exception as e:
                    erro_msg = str(e)
                    if "429" in erro_msg:
                        tentativas += 1
                        tempo_espera = 20 * tentativas
                        st.warning(f"⏳ Cota excedida. Pausa de {tempo_espera}s...")
                        time.sleep(tempo_espera)
                    else:
                        raise e

            progresso = min((i + batch_size) / total_chunks, 1.0)
            progress_bar.progress(progresso, text=f"Processando parte {min(i+batch_size, total_chunks)} de {total_chunks}...")
            time.sleep(2) 

        os.remove(tmp_file_path)
        progress_bar.empty()
        return True, f"Sucesso! Documento '{uploaded_file.name}' indexado."

    except Exception as e:
        return False, str(e)

# ... (MANTENHA OS IMPORTS E AS FUNÇÕES process_pdf, get_vectorstore, ETC IGUAIS) ...

def get_resposta(pergunta, modo):
    """Define a personalidade da IA baseada no nível de acesso"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # PERSONALIDADE 1: CIDADÃO (Simples e Didático)
    if modo == "cidadao":
        system_prompt = (
            "Você é um Assistente Virtual da Prefeitura, focado em ajudar o cidadão comum. "
            "Use linguagem simples, evite termos técnicos e explique os direitos de forma clara. "
            "Baseie-se no contexto abaixo:\n{context}"
        )
    
    # PERSONALIDADE 2 e 3: TÉCNICA (Para Admin e Funcionário)
    else: 
        system_prompt = (
            "Você é um Auditor Sênior de Conformidade Legal. "
            "Sua resposta deve ser técnica, formal e precisa. "
            "CITE SEMPRE: O nome da Lei, o número do Artigo e o Parágrafo. "
            "Se a informação não estiver no contexto, diga 'Não consta nos autos'. "
            "Contexto:\n{context}"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
    return chain.invoke({"input": pergunta})["answer"]

# --- INTERFACE PRINCIPAL ---
query_params = st.query_params
modo = query_params.get("mode", "cidadao") # Padrão é cidadão

# 1. MODO ADMINISTRADOR (Acesso Total)
if modo == "admin":
    st.info("🔒 Painel de Controle - Administrador do Sistema")
    
    with st.expander("📂 Upload de Documentos (Acesso Exclusivo)", expanded=True):
        uploaded_file = st.file_uploader("Adicionar Lei/Edital ao Banco", type="pdf")
        if uploaded_file and st.button("Processar Documento"):
            with st.spinner("Indexando..."):
                sucesso, msg = process_pdf(uploaded_file)
                if sucesso: st.success(msg)
                else: st.error(msg)
    st.divider()
    st.subheader("💬 Chat Técnico (Modo Auditor)")

# 2. MODO FUNCIONÁRIO (Sem Upload, Chat Técnico)
elif modo == "funcionario":
    st.info("👤 Acesso Servidor Público - Consulta Técnica")
    st.warning("⚠️ Seu perfil permite apenas consulta. Para inserir documentos, contate o Administrador.")
    st.subheader("💬 Chat Técnico (Modo Auditor)")

# 3. MODO CIDADÃO (Chat Simples)
else:
    st.success("👋 Bem-vindo ao Portal da Transparência!")
    st.subheader("💬 Tire suas dúvidas")

# --- CHATBOT (Comum a todos) ---
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
        with st.spinner("Analisando..."):
            resp = get_resposta(prompt, modo)
            st.markdown(resp)
            st.session_state.messages.append({"role": "assistant", "content": resp})

