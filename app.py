import streamlit as st
import os
import time
import tempfile
import hashlib
import uuid

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone, ServerlessSpec

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="IA Auditoria Municipal - Consulta Avançada", 
                   layout="wide", 
                   page_icon="🏛️")

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
INDEX_NAME = "tcc-auditoria"

# --- 3. INICIALIZAÇÃO CORRETA DO PINECONE ---
@st.cache_resource
def init_pinecone():
    """Inicializa o cliente Pinecone corretamente"""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if INDEX_NAME not in existing_indexes:
            st.info(f"🔄 Índice '{INDEX_NAME}' não encontrado. Criando...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=768,  # Dimensão do embedding Gemini
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"  # Ajuste conforme seu ambiente Pinecone
                )
            )
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
        # Nota: models/embedding-001 é o nome padrão correto para embeddings textuais do Gemini
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            task_type="retrieval_query"  # Otimizado para consulta
        )
        
        vectorstore = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            pinecone_api_key=PINECONE_API_KEY
        )
        return vectorstore
    except Exception as e:
        st.error(f"❌ Erro ao conectar ao vectorstore: {str(e)}")
        return None

@st.cache_resource
def get_llm():
    """Inicializa o modelo de linguagem Gemini"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        temperature=0.1,
        max_retries=1  # <-- ADICIONE ISTO AQUI
    )


# --- 4. PROCESSAMENTO DE PDF ---
def process_pdf_otimizado(uploaded_file):
    """Processamento otimizado de PDFs"""
    tmp_file_path = None
    try:
        if uploaded_file is None:
            return False, "❌ Nenhum arquivo fornecido."
        
        file_content = uploaded_file.read()
        if len(file_content) == 0:
            return False, "❌ Arquivo vazio."
        
        file_hash = hashlib.md5(file_content).hexdigest()
        uploaded_file.seek(0)
        
        vectorstore = get_vectorstore()
        if vectorstore is None:
            return False, "❌ Não foi possível conectar ao banco de dados."
        
        try:
            existing = vectorstore.similarity_search(
                "dummy query",
                k=1,
                filter={"file_hash": {"$eq": file_hash}}
            )
            if existing:
                return False, "⚠️ Documento já processado anteriormente."
        except Exception as e:
            st.warning(f"⚠️ Não foi possível verificar duplicidade: {str(e)}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        try:
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
        except Exception as e:
            return False, f"❌ Erro ao ler PDF: {str(e)}"
        
        if not docs:
            return False, "❌ PDF vazio ou sem texto extraível."
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        
        documentos_para_adicionar = []
        for i, split in enumerate(splits):
            chunk_id = str(uuid.uuid4())
            split.metadata.update({
                "file_hash": file_hash,
                "source": uploaded_file.name,
                "chunk_index": i,
                "total_chunks": len(splits),
                "doc_type": "PDF",
                "id": chunk_id
            })
            if split.page_content and len(split.page_content.strip()) > 0:
                documentos_para_adicionar.append(split)
        
        if not documentos_para_adicionar:
            return False, "❌ Nenhum conteúdo válido extraído do PDF."
        
        total = len(documentos_para_adicionar)
        progress_bar = st.progress(0, text="Enviando para o Pinecone...")
        
        batch_size = 10
        for i in range(0, total, batch_size):
            batch = documentos_para_adicionar[i:i + batch_size]
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    vectorstore.add_documents(batch)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        raise e
            progress = min((i + batch_size) / total, 1.0)
            progress_bar.progress(progress, text=f"Enviando... {int(progress * 100)}%")
        
        progress_bar.empty()
        
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
            
        return True, f"✅ Sucesso! {total} partes indexadas no Pinecone."
        
    except Exception as e:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
            except:
                pass
        return False, f"❌ Erro durante o processamento: {str(e)}"

# --- 5. INTERFACE DO USUÁRIO ---
def main():
    pc = init_pinecone()
    if pc is None:
        st.error("❌ Não foi possível inicializar o Pinecone. Verifique suas credenciais.")
        return
    
    # ==========================
    # LOGIN VIA PHP (IFRAME)
    # ==========================
    params = st.query_params
    logado = params.get("logado", "false").lower() == "true"
    nivel = params.get("nivel", "cidadao").lower()

    modo = nivel if logado else "cidadao"

    # ==========================
    # SIDEBAR
    # ==========================
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
            st.metric("Status", "Conectado" if pc else "Desconectado")
            st.metric("Índice", INDEX_NAME)
    
    # Área principal
    st.title("🤖 Assistente Virtual da Prefeitura")
    st.caption("Consultas baseadas exclusivamente em documentos oficiais (RAG + Pinecone)")
    
    # Histórico de Chat
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
            with st.spinner("🔍 Consultando base documental e gerando resposta..."):
                try:
                    vectorstore = get_vectorstore()
                    llm = get_llm()
                    
                    if vectorstore and llm:
                        # 1. Configurar o Retriever do Pinecone
                        retriever = vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 5, "filter": {"doc_type": "PDF"}}
                        )
                        
                        # 2. Criar o Prompt que OBRIGA a IA a usar o Pinecone
                        if modo == "cidadao":
                            system_prompt = (
                                "Você é um assistente virtual da Prefeitura que ajuda cidadãos comuns. "
                                "Responda de forma simples, clara e didática, evitando linguagem técnica. "
                                "Baseie-se EXCLUSIVAMENTE nos documentos fornecidos.\n\n"
                                "REGRAS:\n"
                                "- Explique como se estivesse falando com alguém leigo\n"
                                "- NÃO citar artigo ou número de lei, a menos que seja extremamente necessário\n"
                                "- Use linguagem acessível\n"
                                "- Se não houver resposta nos documentos, diga exatamente:\n"
                                "'Desculpe, não encontrei informações sobre isso nos documentos oficiais anexados.'\n\n"
                                "Contexto:\n{context}"
                            )
                        else:  # admin ou funcionario
                            system_prompt = (
                                "Você é um assistente técnico da Auditoria Municipal. "
                                "Responda de forma objetiva, precisa e técnica.\n\n"
                                "REGRAS:\n"
                                "- Baseie-se EXCLUSIVAMENTE nos documentos fornecidos\n"
                                "- Sempre que possível, cite explicitamente:\n"
                                "  • Nome da lei/decreto\n"
                                "  • Número do artigo\n"
                                "- Seja direto e sucinto\n"
                                "- NÃO invente informações\n"
                                "- Se não houver resposta nos documentos, diga exatamente:\n"
                                "'Desculpe, não encontrei informações sobre isso nos documentos oficiais anexados.'\n\n"
                                "Contexto:\n{context}"
                            )
                        
                        prompt_template = ChatPromptTemplate.from_messages([
                            ("system", system_prompt),
                            ("human", "{input}")
                        ])
                        
                        # 3. Montar a Cadeia RAG
                        question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                        
                        # 4. Executar a consulta
                        response = rag_chain.invoke({"input": prompt})
                        
                        answer = response["answer"]
                        source_docs = response["context"]
                        
                        # Exibe a resposta formulada pela IA
                        st.markdown(answer)
                        
                        # Exibe as fontes de onde ela tirou a informação
                        if source_docs:
                            st.markdown("---")
                            st.markdown("📚 **Trechos Consultados:**")
                            for i, doc in enumerate(source_docs):
                                fonte = doc.metadata.get('source', 'Fonte desconhecida')
                                with st.expander(f"📄 Fonte {i+1} - {fonte}"):
                                    st.markdown(f"*{doc.page_content}*")
                        else:
                            st.warning("⚠️ Nenhum documento PDF relevante foi encontrado no banco de dados para esta consulta.")
                        
                        # Salva no histórico
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        st.error("Erro interno: Falha ao carregar banco de dados vetorial ou modelo LLM.")
                        
                except Exception as e:
                    st.error(f"Erro durante a geração da resposta: {str(e)}")

if __name__ == "__main__":
    main()
