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
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone, ServerlessSpec  # IMPORTANTE: Import correto
import uuid

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="IA Auditoria Municipal - Consulta Avançada", layout="wide", page_icon="🏛️")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stAlert {margin-top: 10px;}
    .source-box {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        border-left: 4px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CARREGAMENTO DE SEGREDOS COM VALIDAÇÃO ---
if "GOOGLE_API_KEY" not in st.secrets or "PINECONE_API_KEY" not in st.secrets:
    st.error("❌ ERRO: Chaves de API não configuradas no secrets.toml!")
    st.stop()

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

# Configurações do Pinecone
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = st.secrets.get("PINECONE_ENVIRONMENT", "gcp-starter")  # Ajuste conforme seu ambiente
INDEX_NAME = "tcc-auditoria"

# --- 3. INICIALIZAÇÃO CORRETA DO PINECONE ---
@st.cache_resource
def init_pinecone():
    """Inicializa o cliente Pinecone corretamente"""
    try:
        # Inicializa o cliente Pinecone (versão mais recente)
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Lista índices existentes
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        # Verifica se o índice existe, se não, cria
        if INDEX_NAME not in existing_indexes:
            st.info(f"🔄 Índice '{INDEX_NAME}' não encontrado. Criando...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=768,  # Dimensão do embedding Gemini
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"  # Ajuste conforme necessário
                )
            )
            # Aguarda a criação do índice
            time.sleep(10)
            st.success(f"✅ Índice '{INDEX_NAME}' criado com sucesso!")
        
        return pc
    except Exception as e:
        st.error(f"❌ Erro ao inicializar Pinecone: {str(e)}")
        return None

@st.cache_resource
def get_vectorstore():
    """Conecta ao Pinecone com configurações otimizadas"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-gecko-multilingual-001",  # Suporte multilíngue
            task_type="retrieval_query"
        )
        
        # Conecta ao índice existente
        vectorstore = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            pinecone_api_key=PINECONE_API_KEY  # Importante: passar a API key
        )
        
        return vectorstore
    except Exception as e:
        st.error(f"❌ Erro ao conectar ao vectorstore: {str(e)}")
        return None

# --- 4. FUNÇÃO DE PROCESSAMENTO DE PDF CORRIGIDA ---
def process_pdf_otimizado(uploaded_file):
    """Processamento otimizado de PDFs com melhor tratamento de erros"""
    tmp_file_path = None
    try:
        # Validação inicial
        if uploaded_file is None:
            return False, "❌ Nenhum arquivo fornecido."
        
        # Lê conteúdo
        file_content = uploaded_file.read()
        if len(file_content) == 0:
            return False, "❌ Arquivo vazio."
        
        # Calcula hash
        file_hash = hashlib.md5(file_content).hexdigest()
        uploaded_file.seek(0)
        
        # Obtém vectorstore
        vectorstore = get_vectorstore()
        if vectorstore is None:
            return False, "❌ Não foi possível conectar ao banco de dados."
        
        # Verificação de duplicidade
        try:
            # Usa busca por similaridade com filtro
            existing = vectorstore.similarity_search(
                "dummy query",
                k=1,
                filter={"file_hash": {"$eq": file_hash}}
            )
            if existing:
                return False, "⚠️ Documento já processado anteriormente."
        except Exception as e:
            # Se falhar na verificação, continua (pode ser que não haja documentos ainda)
            st.warning(f"⚠️ Não foi possível verificar duplicidade: {str(e)}")
        
        # Cria arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        # Carrega PDF
        try:
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
        except Exception as e:
            return False, f"❌ Erro ao ler PDF: {str(e)}"
        
        if not docs:
            return False, "❌ PDF vazio ou sem texto extraível."
        
        # Diagnóstico de conteúdo
        total_chars = sum(len(doc.page_content) for doc in docs)
        if total_chars < 100:
            st.warning("⚠️ ALERTA: Pouco texto extraído! Pode ser imagem escaneada. Considere usar OCR.")
        
        # Text splitter otimizado
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Divide documentos
        splits = text_splitter.split_documents(docs)
        
        # Prepara documentos com metadados enriquecidos
        documentos_para_adicionar = []
        for i, split in enumerate(splits):
            # Cria um ID único para cada chunk
            chunk_id = str(uuid.uuid4())
            
            # Enriquece metadados
            split.metadata.update({
                "file_hash": file_hash,
                "source": uploaded_file.name,
                "chunk_index": i,
                "total_chunks": len(splits),
                "doc_type": "PDF",
                "id": chunk_id
            })
            
            # Adiciona conteúdo para garantir que não está vazio
            if split.page_content and len(split.page_content.strip()) > 0:
                documentos_para_adicionar.append(split)
        
        if not documentos_para_adicionar:
            return False, "❌ Nenhum conteúdo válido extraído do PDF."
        
        total = len(documentos_para_adicionar)
        st.write(f"📄 Processando {total} partes...")
        
        # Upload em lote com progresso
        batch_size = 10
        progress_bar = st.progress(0, text="Enviando para o Pinecone...")
        
        for i in range(0, total, batch_size):
            batch = documentos_para_adicionar[i:i + batch_size]
            
            # Tenta enviar com retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    vectorstore.add_documents(batch)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        st.warning(f"⚠️ Erro no envio, tentando novamente em {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise e
            
            # Atualiza progresso
            progress = min((i + batch_size) / total, 1.0)
            progress_bar.progress(progress, text=f"Enviando... {int(progress * 100)}%")
        
        progress_bar.empty()
        
        # Limpeza
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        
        return True, f"✅ Sucesso! {total} partes indexadas no Pinecone."
        
    except Exception as e:
        # Limpeza em caso de erro
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
            except:
                pass
        return False, f"❌ Erro durante o processamento: {str(e)}"

# --- 5. FUNÇÕES DE BUSCA CORRIGIDAS ---
def search_with_metadata(pergunta, k=7):
    """Busca com scoring e filtro - APENAS PDFs"""
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return []
    
    try:
        # Busca semântica com filtro
        docs = vectorstore.similarity_search_with_score(
            pergunta,
            k=k,
            filter={"doc_type": {"$eq": "PDF"}}  # Sintaxe correta para filtros
        )
        
        # Filtra por relevância
        relevant_docs = []
        for doc, score in docs:
            # Quanto menor o score, mais relevante (distância cosseno)
            if score < 0.8:  # Ajuste conforme necessidade
                relevant_docs.append((doc, score))
        
        return relevant_docs
    except Exception as e:
        st.error(f"Erro na busca: {str(e)}")
        return []

def get_pdf_only_retriever(k=7):
    """Retriever configurado para PDFs"""
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return None
    
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k,
                "filter": {"doc_type": {"$eq": "PDF"}}
            }
        )
        return retriever
    except Exception as e:
        st.error(f"Erro ao criar retriever: {str(e)}")
        return None

# --- 6. INTERFACE DO USUÁRIO ---
def main():
    # Inicializa Pinecone
    pc = init_pinecone()
    if pc is None:
        st.error("❌ Não foi possível inicializar o Pinecone. Verifique suas credenciais.")
        return
    
    query_params = st.query_params
    modo = query_params.get("mode", "cidadao")
    
    # Sidebar
    with st.sidebar:
        st.title("🏛️ Painel de Controle")
        
        if modo == "admin":
            st.success("🔒 MODO ADMINISTRADOR")
            st.markdown("---")
            st.subheader("📤 Upload de Documentos")
            
            uploaded_file = st.file_uploader("Selecione o PDF", type="pdf")
            if uploaded_file and st.button("🚀 Processar Documento", use_container_width=True):
                with st.spinner("Processando documento..."):
                    sucesso, msg = process_pdf_otimizado(uploaded_file)
                    if sucesso:
                        st.success(msg)
                        st.balloons()
                    else:
                        st.error(msg)
            
            st.markdown("---")
            st.subheader("📊 Estatísticas")
            st.metric("Documentos Indexados", "Aguardando...")
            st.metric("Status", "Conectado" if pc else "Desconectado")
            st.metric("Índice", INDEX_NAME)
    
    # Área principal
    st.title("🤖 Assistente Virtual da Prefeitura")
    st.caption("Consultas baseadas exclusivamente em documentos PDF oficiais")
    
    # Histórico
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input do usuário
    if prompt := st.chat_input("Digite sua dúvida sobre os documentos municipais..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Consultando base documental..."):
                try:
                    # Busca documentos
                    docs_com_scores = search_with_metadata(prompt, k=5)
                    
                    if docs_com_scores:
                        resposta = f"Encontrei {len(docs_com_scores)} trechos relevantes nos documentos PDF."
                        
                        # Exibe os trechos encontrados
                        for i, (doc, score) in enumerate(docs_com_scores):
                            fonte = doc.metadata.get('source', 'Fonte desconhecida')
                            trecho = doc.page_content[:300] + "..."
                            
                            with st.expander(f"📄 Trecho {i+1} - {fonte} (relevância: {score:.4f})"):
                                st.markdown(f"**Conteúdo:**\n{trecho}")
                        
                        # Resposta simples
                        st.markdown("Para uma análise mais detalhada, estou preparando uma resposta personalizada...")
                        
                        # Aqui você pode adicionar a geração de resposta com LLM
                        
                    else:
                        st.warning("Nenhum documento PDF relevante encontrado para sua consulta.")
                        resposta = "Não encontrei documentos PDF relacionados à sua pergunta."
                    
                    # Salva no histórico
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": resposta
                    })
                    
                except Exception as e:
                    st.error(f"Erro na consulta: {str(e)}")

if __name__ == "__main__":
    main()

