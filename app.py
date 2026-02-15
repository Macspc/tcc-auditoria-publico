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

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="IA Auditoria Municipal", layout="wide", page_icon="🏛️")

# Esconde menu padrão do Streamlit para parecer um app profissional
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stAlert {margin-top: 10px;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. CARREGAMENTO DE SEGREDOS (CHAVES API) ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
else:
    st.error("❌ ERRO CRÍTICO: Chaves de API não configuradas nos Secrets!")
    st.stop()

# --- 3. FUNÇÕES DE BACKEND (O CÉREBRO) ---

@st.cache_resource
def get_vectorstore():
    """Conecta ao Pinecone. Ajuste o modelo se necessário."""
    # Modelo atualizado para contas pagas/novas. 
    # Gera vetores de 768 dimensões por padrão.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    index_name = "tcc-auditoria" 
    
    vectorstore = PineconeVectorStore(
        index_name=index_name, 
        embedding=embeddings
    )
    return vectorstore

def calculate_md5(file_content):
    """Gera a 'Impressão Digital' do arquivo"""
    return hashlib.md5(file_content).hexdigest()

def process_pdf(uploaded_file):
    """Processa PDF com: Diagnóstico de Leitura + Anti-Duplicidade + Anti-Erro 429"""
    try:
        # A. Verifica Duplicidade (Hashing)
        file_content = uploaded_file.read()
        file_hash = calculate_md5(file_content)
        uploaded_file.seek(0) # Reseta ponteiro

        vectorstore = get_vectorstore()
        
        # Tenta buscar se o hash já existe (Isso evita gastar dinheiro reprocessando)
        try:
            exists = vectorstore.similarity_search("teste", k=1, filter={"file_hash": file_hash})
            if exists:
                return False, "⚠️ Este documento JÁ FOI processado anteriormente! O sistema bloqueou a duplicidade."
        except:
            pass # Se o index for novo, a busca falha, segue o jogo.

        # B. Cria Arquivo Temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        # C. Carrega e DIAGNOSTICA (Para resolver problema do Diário Oficial)
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()

        if not docs:
            return False, "❌ O PDF está vazio ou corrompido."
        
        # --- DIAGNÓSTICO DE LEITURA ---
        primeira_pag = docs[0].page_content
        chars_lidos = len(primeira_pag)
        st.info(f"🔍 Diagnóstico: Li {chars_lidos} caracteres na 1ª página.")
        
        if chars_lidos < 100:
            st.warning("⚠️ ALERTA: Pouco texto detectado! Se for um Diário Oficial ESCANEADO (Imagem), a IA não consegue ler. Use um OCR antes.")
            with st.expander("👀 Ver o que o robô conseguiu ler"):
                st.write(primeira_pag)
        # ------------------------------

        # D. Quebra em Pedaços (Chunks)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        
        # Adiciona Metadados (Hash e Nome)
        for split in splits:
            split.metadata["file_hash"] = file_hash
            split.metadata["source"] = uploaded_file.name

        total_chunks = len(splits)
        st.write(f"📄 Processando {total_chunks} fragmentos de texto...")

        # E. Envio Seguro (Rate Limiting)
        batch_size = 5 
        progress_bar = st.progress(0, text="Iniciando upload para o Cérebro Digital...")

        for i in range(0, total_chunks, batch_size):
            batch = splits[i : i + batch_size]
            
            sucesso_lote = False
            tentativas = 0
            
            while not sucesso_lote and tentativas < 5:
                try:
                    vectorstore.add_documents(batch)
                    sucesso_lote = True
                except Exception as e:
                    erro = str(e)
                    if "429" in erro: # Cota excedida ou Rate Limit
                        tentativas += 1
                        tempo = 10 * tentativas
                        st.toast(f"⏳ O Google pediu calma... Esperando {tempo}s.")
                        time.sleep(tempo)
                    else:
                        st.error(f"Erro fatal no lote {i}: {erro}")
                        return False, str(e)

            # Atualiza barra
            progresso = min((i + batch_size) / total_chunks, 1.0)
            progress_bar.progress(progresso, text=f"Indexando parte {min(i+batch_size, total_chunks)} de {total_chunks}...")
            time.sleep(1) # Pausa de cortesia

        os.remove(tmp_file_path)
        progress_bar.empty()
        return True, f"✅ Sucesso! Documento '{uploaded_file.name}' blindado no banco de dados."

    except Exception as e:
        return False, f"Erro Geral: {str(e)}"

def get_resposta(pergunta, modo):
    """Gera resposta com RAG e mostra Debug"""
    # Modelo de Chat (Inteligente)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-Pro", temperature=0.3)
    vectorstore = get_vectorstore()
    
    # 1. Busca Contexto (Recuperação)
    docs_encontrados = vectorstore.similarity_search(pergunta, k=5)
    
    # --- DEBUG VISUAL (RAIO-X) ---
    with st.expander("🕵️ [AUDITORIA] O que a IA leu para responder? (Debug)", expanded=False):
        if not docs_encontrados:
            st.warning("⚠️ O banco retornou ZERO documentos parecidos. Verifique o upload.")
        for i, doc in enumerate(docs_encontrados):
            st.markdown(f"**📄 Trecho {i+1} (Fonte: {doc.metadata.get('source', 'Desconhecido')})**")
            st.caption(f"...{doc.page_content[:400]}...") # Mostra 400 chars
            st.divider()
    # -----------------------------

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 2. Define Personalidade (Prompt)
    if modo == "cidadao":
        system_prompt = (
            "Você é um Assistente Virtual da Prefeitura, amigável e didático. "
            "Seu objetivo é explicar leis complexas em linguagem simples para o cidadão. "
            "Use OBRIGATORIAMENTE o contexto abaixo. Se a resposta não estiver lá, diga que não sabe. "
            "Contexto:\n{context}"
        )
    else: # Admin ou Funcionario
        system_prompt = (
            "Você é um Auditor Assistente Sênior. "
            "Responda de forma técnica, citando Artigos, Parágrafos e Leis. "
            "Baseie-se ESTRITAMENTE no contexto fornecido. "
            "Se o contexto for insuficiente, informe 'Dados insuficientes nos autos'. "
            "Contexto:\n{context}"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
    
    # 3. Gera Resposta
    return chain.invoke({"input": pergunta})["answer"]

# --- 4. INTERFACE GRÁFICA (FRONTEND) ---

# Captura o modo via URL (enviado pelo PHP)
query_params = st.query_params
modo = query_params.get("mode", "cidadao")

# CABEÇALHO DINÂMICO
if modo == "admin":
    st.info("🔒 MODO ADMINISTRADOR - Acesso Total")
    # Apenas Admin vê upload
    with st.expander("📂 Alimentar Base de Dados (Upload PDF)", expanded=True):
        uploaded_file = st.file_uploader("Escolha Lei ou Edital (PDF)", type="pdf")
        if uploaded_file and st.button("Processar Documento"):
            with st.spinner("Analisando integridade e indexando..."):
                sucesso, msg = process_pdf(uploaded_file)
                if sucesso:
                    st.success(msg)
                    st.balloons()
                else:
                    st.error(msg)
                    
elif modo == "funcionario":
    st.info("👤 MODO SERVIDOR PÚBLICO - Consulta Técnica")
    st.warning("⚠️ Perfil de Consulta: Upload desabilitado.")

else: # Cidadão
    st.success("👋 Olá! Bem-vindo ao Portal da Transparência.")
    st.markdown("**Como posso ajudar você a entender as leis municipais hoje?**")

st.divider()

# ÁREA DE CHAT
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input do usuário
if prompt := st.chat_input("Digite sua dúvida sobre legislação..."):
    # 1. Guarda msg do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Gera resposta da IA
    with st.chat_message("assistant"):
        with st.spinner("Consultando base legal..."):
            try:
                resposta = get_resposta(prompt, modo)
                st.markdown(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})
            except Exception as e:
                st.error(f"Erro ao processar: {e}")

