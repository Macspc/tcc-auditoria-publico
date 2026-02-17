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

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stAlert {margin-top: 10px;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. CARREGAMENTO DE SEGREDOS ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
else:
    st.error("❌ ERRO: Chaves de API não configuradas!")
    st.stop()

# --- 3. FUNÇÕES DE BACKEND ---

@st.cache_resource
def get_vectorstore():
    """Conecta ao Pinecone usando o modelo que SUA conta possui"""
    # NOME EXATO QUE APARECEU NO SEU DIAGNÓSTICO:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    index_name = "tcc-auditoria" 
    
    vectorstore = PineconeVectorStore(
        index_name=index_name, 
        embedding=embeddings
    )
    return vectorstore

def calculate_md5(file_content):
    return hashlib.md5(file_content).hexdigest()

def process_pdf(uploaded_file):
    try:
        # A. Verifica Duplicidade
        file_content = uploaded_file.read()
        file_hash = calculate_md5(file_content)
        uploaded_file.seek(0)

        vectorstore = get_vectorstore()
        
        try:
            exists = vectorstore.similarity_search("teste", k=1, filter={"file_hash": file_hash})
            if exists:
                return False, "⚠️ Documento já processado anteriormente (Duplicidade barrada)."
        except:
            pass 

        # B. Cria Arquivo Temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        # C. Carrega e Diagnostica
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()

        if not docs:
            return False, "❌ PDF vazio ou ilegível."
        
        primeira_pag = docs[0].page_content
        chars = len(primeira_pag)
        st.info(f"🔍 Diagnóstico: Li {chars} caracteres na 1ª página.")
        
        if chars < 100:
            st.warning("⚠️ ALERTA: Texto insuficiente! Pode ser imagem escaneada. Use OCR.")

        # D. Chunking e Upload
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        for split in splits:
            split.metadata["file_hash"] = file_hash
            split.metadata["source"] = uploaded_file.name

        total = len(splits)
        st.write(f"📄 Processando {total} partes...")

        batch_size = 5 
        progress = st.progress(0, text="Enviando...")

        for i in range(0, total, batch_size):
            batch = splits[i : i + batch_size]
            sucesso_lote = False
            tentativas = 0
            while not sucesso_lote and tentativas < 5:
                try:
                    vectorstore.add_documents(batch)
                    sucesso_lote = True
                except Exception as e:
                    if "429" in str(e):
                        tentativas += 1
                        time.sleep(5 * tentativas)
                    else:
                        return False, str(e)
            
            progress.progress(min((i + batch_size) / total, 1.0))
            time.sleep(1)

        os.remove(tmp_file_path)
        progress.empty()
        return True, "✅ Sucesso! Documento indexado."

    except Exception as e:
        return False, str(e)



def get_resposta(pergunta, modo):
    """Gera resposta com RAG em MODO ESTRITO (Sem Alucinação)"""
    
    # 1. Configura o Modelo (Temperatura 0 = Criatividade Zero, Foco Total)
    # Usamos temperatura 0.0 para garantir que ela não invente nada.
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.0)
    
    vectorstore = get_vectorstore()
    
    # 2. Busca Contexto (Aumentamos para 7 trechos para garantir mais contexto)
    docs_encontrados = vectorstore.similarity_search(pergunta, k=7)
    
    # --- DEBUG VISUAL (RAIO-X) ---
    with st.expander("🕵️ [AUDITORIA] Fontes Recuperadas (O que a IA leu)", expanded=False):
        if not docs_encontrados:
            st.error("⚠️ O banco retornou ZERO documentos. A IA não terá base para responder.")
        for i, doc in enumerate(docs_encontrados):
            source = doc.metadata.get('source', 'Desconhecido')
            st.markdown(f"**📄 Trecho {i+1} (Fonte: {source})**")
            st.caption(f"...{doc.page_content.replace(chr(10), ' ')[:300]}...") # Remove quebras de linha para visualizar melhor
            st.divider()
    # -----------------------------

    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

    # 3. PROMPTS BLINDADOS (AQUI ESTÁ A MÁGICA)
    
    if modo == "cidadao":
        system_prompt = (
            "Você é um Assistente Oficial da Prefeitura. "
            "Sua única fonte de verdade são os documentos fornecidos abaixo no 'Contexto'. "
            "INSTRUÇÕES RÍGIDAS:\n"
            "1. Responda SOMENTE com base no contexto.\n"
            "2. Se a resposta não estiver no contexto, diga: 'Desculpe, essa informação não consta nos documentos oficiais disponíveis.'\n"
            "3. NÃO use seu conhecimento externo (internet/treino). "
            "4. Seja educado e claro.\n\n"
            "CONTEXTO OFICIAL:\n{context}"
        )
    else: # Admin ou Funcionario
        system_prompt = (
            "Você é um Auditor de Conformidade Legal. "
            "Sua tarefa é extrair informações EXATAS dos documentos fornecidos. "
            "REGRAS DE OURO:\n"
            "1. IGNORE todo seu conhecimento prévio. Use APENAS o contexto abaixo.\n"
            "2. Se o contexto diz 'O céu é verde', você responde 'O céu é verde'. Fidelidade total ao texto.\n"
            "3. Cite a fonte (Artigo, Parágrafo, Cláusula) sempre que possível.\n"
            "4. Se a informação não estiver explícita, responda: 'DADO NÃO ENCONTRADO NOS AUTOS'.\n"
            "5. Não invente, não deduza e não arredonde valores.\n\n"
            "CONTEXTO DOS AUTOS:\n{context}"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
    
    return chain.invoke({"input": pergunta})["answer"]








# --- 4. INTERFACE ---
query_params = st.query_params
modo = query_params.get("mode", "cidadao")

if modo == "admin":
    st.info("🔒 ADMIN - Upload Liberado")
    with st.expander("📂 Upload PDF", expanded=True):
        uploaded_file = st.file_uploader("Arquivo", type="pdf")
        if uploaded_file and st.button("Processar"):
            with st.spinner("Indexando..."):
                sucesso, msg = process_pdf(uploaded_file)
                if sucesso: st.success(msg)
                else: st.error(msg)
elif modo == "funcionario":
    st.info("👤 SERVIDOR - Consulta Técnica")
else:
    st.success("👋 Portal da Transparência")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Digite sua dúvida..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando..."):
            try:
                resposta = get_resposta(prompt, modo)
                st.markdown(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})
            except Exception as e:
                st.error(f"Erro: {e}")

