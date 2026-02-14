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

def get_resposta(pergunta, perfil):
    """Gera a resposta e MOSTRA O DEBUG"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    vectorstore = get_vectorstore()
    
    # --- ÁREA DE DEBUG (RAIO-X) ---
    # Busca os documentos antes de passar para a IA
    docs_encontrados = vectorstore.similarity_search(pergunta, k=4)
    
    with st.expander("🕵️ [DEBUG] O que encontrei no Pinecone:", expanded=False):
        if not docs_encontrados:
            st.error("❌ NENHUM DOCUMENTO ENCONTRADO PARA ESSA PERGUNTA!")
            st.write("Dica: Verifique se o upload foi concluído.")
        else:
            st.success(f"✅ Encontrei {len(docs_encontrados)} trechos relevantes.")
            for i, doc in enumerate(docs_encontrados):
                st.markdown(f"**Trecho {i+1}:**")
                st.caption(doc.page_content[:300] + "...") # Mostra os primeiros 300 caracteres
                st.divider()
    # ---------------------------------

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    if perfil == "server":
        system_prompt = (
            "Você é um Auditor Assistente. Responda estritamente com base no contexto abaixo. "
            "Se a resposta não estiver no texto, diga 'Não consta nos documentos carregados'. "
            "Cite artigos e leis se possível. "
            "\n\nContexto:\n{context}"
        )
    else:
        system_prompt = (
            "Você é um assistente da prefeitura. Explique de forma simples com base no texto abaixo. "
            "\n\nContexto:\n{context}"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": pergunta})
    return response["answer"]

# --- INTERFACE ---
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

st.subheader("💬 Chat de Auditoria")

# Inicializa histórico
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input do usuário
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
                st.error(f"Erro: {e}")

