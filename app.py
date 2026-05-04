import streamlit as st
import os
import time
import tempfile
import hashlib
import uuid
import re
import random

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone, ServerlessSpec

# --- 1. CONFIGURAÇÃO DA PÁGINA (OTIMIZADA PARA IFRAME) ---
st.set_page_config(
    page_title="IA Auditoria Municipal",
    layout="wide",
    page_icon="🏛️",
    initial_sidebar_state="collapsed"  # Sidebar fechada por padrão no iframe
)

# CSS otimizado para integração com iframe PHP
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Ajustes para iframe */
    .stApp {
        margin-top: -80px;  /* Compensa header do Streamlit */
    }
    
    .stAlert {
        margin-top: 10px;
    }
    
    .source-box {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        border-left: 4px solid #4CAF50;
    }
    
    /* Cores por perfil (matching PHP) */
    .admin-theme {
        --primary-color: #2c3e50;
    }
    .funcionario-theme {
        --primary-color: #2980b9;
    }
    .cidadao-theme {
        --primary-color: #ecf0f1;
    }
    
    /* Responsividade para iframe */
    @media (max-width: 768px) {
        .stApp {
            margin-top: -60px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DETECÇÃO DO MODO VIA QUERY PARAMS (ENVIADO PELO PHP) ---
def detectar_modo():
    """
    Detecta o modo de acesso baseado nos query params enviados pelo PHP
    Modos:
    - admin: Acesso total (upload + chat técnico + referências)
    - funcionario: Chat técnico com referências
    - cidadao: Chat simples e casual (padrão)
    """
    query_params = st.query_params
    
    # Modo enviado pelo PHP via iframe
    modo = query_params.get("mode", "cidadao")
    
    # Verificar se está embedado (dentro do iframe do PHP)
    embed = query_params.get("embed", "false")
    
    # Mapear modos válidos
    modos_validos = {
        "admin": {
            "nome": "Administrador",
            "nivel": "admin",
            "icone": "🔒",
            "cor": "#2c3e50"
        },
        "funcionario": {
            "nome": "Servidor",
            "nivel": "funcionario", 
            "icone": "🔐",
            "cor": "#2980b9"
        },
        "cidadao": {
            "nome": "Cidadão",
            "nivel": "cidadao",
            "icone": "👤",
            "cor": "#ecf0f1"
        }
    }
    
    if modo in modos_validos:
        return modos_validos[modo], embed == "true"
    else:
        # Fallback para cidadão
        return modos_validos["cidadao"], embed == "true"

# --- 3. CARREGAMENTO DE SEGREDOS ---
if "GOOGLE_API_KEY" not in st.secrets or "PINECONE_API_KEY" not in st.secrets:
    st.error("❌ Configuração necessária. Entre em contato com o administrador do sistema.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = "tcc-auditoria"

# --- 4. INICIALIZAÇÃO DO PINECONE (CACHE) ---
@st.cache_resource
def init_pinecone():
    """Inicializa o cliente Pinecone"""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if INDEX_NAME not in existing_indexes:
            pc.create_index(
                name=INDEX_NAME,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            time.sleep(10)
        
        return pc
    except Exception as e:
        st.error(f"❌ Erro de conexão com banco de dados: {str(e)}")
        return None

@st.cache_resource
def get_vectorstore():
    """Conecta ao Pinecone"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            task_type="retrieval_query"
        )
        
        vectorstore = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            pinecone_api_key=PINECONE_API_KEY
        )
        return vectorstore
    except Exception as e:
        st.error(f"❌ Erro ao conectar: {str(e)}")
        return None

@st.cache_resource
def get_llm():
    """Inicializa o modelo Gemini"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.1,
        max_retries=1
    )

# --- 5. PROCESSAMENTO DE PDF (APENAS ADMIN) ---
def process_pdf_otimizado(uploaded_file):
    """Processa PDF e indexa no Pinecone"""
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
            return False, "❌ Banco de dados indisponível."
        
        # Verificar duplicidade
        try:
            existing = vectorstore.similarity_search(
                "dummy query",
                k=1,
                filter={"file_hash": {"$eq": file_hash}}
            )
            if existing:
                return False, "⚠️ Documento já existe na base."
        except:
            pass
        
        # Salvar temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        # Carregar PDF
        try:
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
        except Exception as e:
            return False, f"❌ Erro ao ler PDF: {str(e)}"
        
        if not docs:
            return False, "❌ PDF sem conteúdo extraível."
        
        # Dividir em chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        
        # Preparar documentos
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
            return False, "❌ Nenhum conteúdo válido extraído."
        
        # Upload em lotes
        total = len(documentos_para_adicionar)
        progress_bar = st.progress(0, text="Indexando no banco de dados...")
        
        batch_size = 10
        for i in range(0, total, batch_size):
            batch = documentos_para_adicionar[i:i + batch_size]
            for attempt in range(3):
                try:
                    vectorstore.add_documents(batch)
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        raise e
            progress = min((i + batch_size) / total, 1.0)
            progress_bar.progress(progress, text=f"Processando... {int(progress * 100)}%")
        
        progress_bar.empty()
        
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
            
        return True, f"✅ Sucesso! {total} trechos indexados."
        
    except Exception as e:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
            except:
                pass
        return False, f"❌ Erro: {str(e)}"

# --- 6. SISTEMA DE PROMPTS POR PERFIL ---
def get_system_prompt(perfil):
    """
    Retorna o prompt adequado baseado no perfil do usuário
    Integrado com os níveis do PHP:
    - admin: Acesso total
    - funcionario: Técnico com referências
    - cidadao: Simples e casual
    """
    
    prompts = {
        "admin": (
            "Você é um ASSISTENTE JURÍDICO ESPECIALIZADO da Auditoria Municipal.\n"
            "Respondendo a um ADMINISTRADOR com acesso TOTAL ao sistema.\n\n"
            "REGRAS DE RESPOSTA:\n"
            "1. Seja TÉCNICO, PRECISO e DETALHADO\n"
            "2. Sempre cite a LEI, DECRETO, ARTIGO, PARÁGRAFO e INCISO exato\n"
            "3. Inclua o TRECHO LITERAL do documento quando relevante\n"
            "4. Formate a resposta com:\n"
            "   - Resposta técnica completa\n"
            "   - Fundamentação legal detalhada\n"
            "   - Trecho exato do documento\n"
            "5. Se não encontrar nos documentos, informe exatamente quais fontes faltam\n\n"
            "Contexto recuperado dos documentos oficiais:\n{context}"
        ),
        
        "funcionario": (
            "Você é um ASSISTENTE TÉCNICO da Prefeitura Municipal.\n"
            "Respondendo a um SERVIDOR PÚBLICO autorizado.\n\n"
            "REGRAS DE RESPOSTA:\n"
            "1. Seja PRECISO e FUNDAMENTADO\n"
            "2. Indique a LEI, DECRETO e ARTIGO correspondente\n"
            "3. Destaque os pontos principais da legislação\n"
            "4. Se não encontrar, oriente onde buscar a informação\n\n"
            "Contexto recuperado dos documentos oficiais:\n{context}"
        ),
        
        "cidadao": (
            "Você é um ASSISTENTE VIRTUAL AMIGÁVEL da Prefeitura Municipal.\n"
            "Respondendo a um CIDADÃO comum.\n\n"
            "REGRAS DE RESPOSTA:\n"
            "1. Use linguagem SIMPLES, CLARA e ACESSÍVEL\n"
            "2. Explique como se estivesse conversando com um amigo\n"
            "3. NÃO mencione números de leis ou artigos técnicos\n"
            "4. Use exemplos do dia a dia quando possível\n"
            "5. Se a informação não estiver disponível, explique como proceder\n"
            "6. Sempre termine com informações de contato úteis\n\n"
            "Contexto recuperado dos documentos oficiais:\n{context}"
        )
    }
    
    return prompts.get(perfil, prompts["cidadao"])

def formatar_resposta_cidadao(resposta_tecnica):
    """Adapta resposta técnica para linguagem cidadã"""
    
    # Dicionário de termos técnicos → linguagem simples
    termos = {
        "in verbis": "conforme está escrito",
        "data venia": "com o devido respeito",
        "a priori": "em princípio",
        "a posteriori": "depois",
        "ad referendum": "para aprovação",
        "caput": "parte principal do artigo",
        "parágrafo único": "parte única",
        "inciso": "item",
        "alínea": "subitem",
        "ementa": "resumo",
        "jurisprudência": "decisões anteriores da justiça",
        "trânsito em julgado": "decisão final, sem possibilidade de recurso",
        "ex officio": "por obrigação do cargo",
        "pro labore": "pelo trabalho",
        "ad hoc": "para este fim específico",
        "sine die": "sem data definida",
        "sub judice": "sob julgamento",
        " erga omnes": "vale para todos",
    }
    
    resposta_simples = resposta_tecnica
    for termo, substituto in termos.items():
        resposta_simples = resposta_simples.replace(termo, substituto)
    
    # Introduções amigáveis
    introducoes = [
        "Vou te explicar de um jeito bem simples: ",
        "Olha só, é o seguinte: ",
        "Deixa eu te contar como funciona: ",
        "É mais simples do que parece: ",
        "Vamos por partes, de forma bem clara: ",
    ]
    
    resposta_final = random.choice(introducoes) + "\n\n" + resposta_simples
    
    # Adiciona canais de ajuda
    resposta_final += (
        "\n\n---"
        "\n📞 **Canais de Atendimento:**"
        "\n• Telefone: (XX) XXXX-XXXX"
        "\n• Email: contato@macspc.com.br"
        "\n• Site: www.macspc.com.br"
        "\n• Presencial: Rua X, 123 - Centro"
        "\n• Horário: Seg-Sex, 8h às 17h"
    )
    
    return resposta_final

def extrair_referencias(source_docs):
    """Extrai referências legais dos documentos"""
    referencias = set()
    
    padroes = [
        r'(Lei|Decreto|Portaria|Resolução|Instrução Normativa|Medida Provisória|Emenda Constitucional)\s*(?:Federal|Estadual|Municipal)?\s*(?:n[º°]\.?\s*)?(\d+[./]?\d*)\s*(?:de\s*)?(?:(\d{1,2})\s*de\s*(\w+)\s*de\s*(\d{4}))?',
        r'[Aa]rt(?:igo)?\.?\s*(\d+[°º]?)\s*(?:[,.]?\s*(§\s*\d+|parágrafo\s+(?:único|\d+)))?',
        r'[Ii]nciso\s+([IVXLC]+|[a-z])',
        r'[Aa]línea\s+([a-z])',
    ]
    
    for doc in source_docs:
        for padrao in padroes:
            encontrados = re.findall(padrao, doc.page_content, re.IGNORECASE)
            for encontrado in encontrados:
                if isinstance(encontrado, tuple):
                    referencias.add(" ".join(filter(None, encontrados)))
                else:
                    referencias.add(encontrado)
    
    return list(referencias)[:10]  # Limita a 10 referências

# --- 7. INTERFACE PRINCIPAL ---
def main():
    # Detectar modo e se está embedado
    perfil_info, is_embed = detectar_modo()
    modo = perfil_info["nivel"]
    
    # Inicializar Pinecone
    pc = init_pinecone()
    if pc is None:
        st.error("❌ Sistema temporariamente indisponível.")
        if modo == "cidadao":
            st.info("📞 Por favor, tente novamente mais tarde ou entre em contato pelo telefone (XX) XXXX-XXXX.")
        return
    
    # --- SIDEBAR (ADAPTATIVA POR PERFIL) ---
    with st.sidebar:
        st.title(f"{perfil_info['icone']} Painel")
        
        if modo == "admin":
            st.success("🔒 MODO ADMINISTRADOR")
            st.caption("Acesso total ao sistema")
            
            st.markdown("---")
            st.subheader("📤 Upload de Documentos")
            
            uploaded_file = st.file_uploader(
                "Selecionar PDF:",
                type="pdf",
                help="Apenas PDFs com texto extraível"
            )
            
            if uploaded_file and st.button("🚀 Processar Documento", use_container_width=True):
                with st.spinner("Processando..."):
                    sucesso, msg = process_pdf_otimizado(uploaded_file)
                    if sucesso:
                        st.success(msg)
                        st.balloons()
                        # Limpar cache para forçar reindexação
                        st.cache_resource.clear()
                    else:
                        st.error(msg)
            
            st.markdown("---")
            st.subheader("📊 Estatísticas")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Status", "Conectado ✅")
            with col2:
                st.metric("Índice", INDEX_NAME)
            
        elif modo == "funcionario":
            st.info("🔐 MODO SERVIDOR")
            st.caption("Acesso técnico autorizado")
            st.markdown("---")
            st.markdown("""
            **Recursos Disponíveis:**
            - ✅ Consulta à legislação
            - ✅ Referências legais
            - ✅ Artigos e parágrafos
            - ✅ Documentos oficiais
            """)
            
        else:  # cidadao
            st.info("👤 PORTAL DO CIDADÃO")
            st.markdown("---")
            st.subheader("📱 Canais de Atendimento")
            st.markdown("""
            📞 **(XX) XXXX-XXXX**
            📧 **contato@macspc.com.br**
            🌐 **www.macspc.com.br**
            🏢 **Rua X, 123 - Centro**
            
            ⏰ Seg-Sex, 8h às 17h
            """)
            
            st.markdown("---")
            st.subheader("🔐 Área do Servidor")
            st.caption("Acesso exclusivo para funcionários")
            st.markdown("*Faça login no portal para acesso técnico*")
    
    # --- ÁREA PRINCIPAL ---
    # Título adaptativo
    titulos = {
        "admin": "🤖 Assistente Técnico - Administração",
        "funcionario": "🤖 Assistente Técnico - Servidor",
        "cidadao": "💬 Assistente Virtual da Prefeitura"
    }
    
    subtitulos = {
        "admin": "Consultas detalhadas com referências legais completas e upload de documentos",
        "funcionario": "Consultas fundamentadas na legislação municipal vigente",
        "cidadao": "Tire suas dúvidas de forma simples e rápida!"
    }
    
    st.title(titulos.get(modo, titulos["cidadao"]))
    st.caption(subtitulos.get(modo, subtitulos["cidadao"]))
    
    # --- HISTÓRICO DE CHAT ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Mostrar histórico
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # --- INPUT DO USUÁRIO ---
    placeholders = {
        "admin": "Digite sua consulta técnica detalhada...",
        "funcionario": "Pergunte sobre leis, decretos e procedimentos...",
        "cidadao": "Como posso ajudar? Pergunte sobre seus direitos e serviços..."
    }
    
    if prompt := st.chat_input(placeholders.get(modo, "Digite sua dúvida...")):
        # Adicionar pergunta ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Gerar resposta
        with st.chat_message("assistant"):
            with st.spinner("🔍 Consultando documentos oficiais..."):
                try:
                    vectorstore = get_vectorstore()
                    llm = get_llm()
                    
                    if vectorstore and llm:
                        # Configurar retriever
                        k_docs = 3 if modo == "cidadao" else 8  # Mais docs para técnicos
                        
                        retriever = vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={
                                "k": k_docs,
                                "filter": {"doc_type": "PDF"}
                            }
                        )
                        
                        # Criar prompt específico
                        system_prompt = get_system_prompt(modo)
                        
                        prompt_template = ChatPromptTemplate.from_messages([
                            ("system", system_prompt),
                            ("human", "{input}")
                        ])
                        
                        # Criar cadeia RAG
                        question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                        
                        # Executar consulta
                        response = rag_chain.invoke({"input": prompt})
                        
                        answer = response["answer"]
                        source_docs = response["context"]
                        
                        # --- FORMATAR RESPOSTA POR PERFIL ---
                        if modo == "cidadao":
                            # Resposta simplificada
                            answer = formatar_resposta_cidadao(answer)
                            st.markdown(answer)
                            
                            if source_docs:
                                st.markdown("---")
                                st.info("💡 Resposta baseada em documentos oficiais da Prefeitura Municipal.")
                        
                        elif modo == "funcionario":
                            # Resposta técnica
                            st.markdown("### 📋 Resposta Técnica")
                            st.markdown(answer)
                            
                            # Referências legais
                            referencias = extrair_referencias(source_docs)
                            if referencias:
                                st.markdown("---")
                                st.markdown("### ⚖️ Fundamentação Legal")
                                for ref in referencias[:5]:
                                    st.markdown(f"📌 {ref}")
                            
                            # Fontes resumidas
                            if source_docs:
                                st.markdown("---")
                                st.markdown("### 📚 Documentos Consultados")
                                for i, doc in enumerate(source_docs[:3]):
                                    with st.expander(f"📄 {doc.metadata.get('source', 'Documento')} - Trecho {i+1}"):
                                        st.text(doc.page_content[:400] + "...")
                        
                        else:  # admin
                            # Resposta completa
                            st.markdown("### 📋 Análise Técnica Completa")
                            st.markdown(answer)
                            
                            # Todas as referências
                            referencias = extrair_referencias(source_docs)
                            if referencias:
                                st.markdown("---")
                                st.markdown("### ⚖️ Referências Legais Completas")
                                for ref in referencias:
                                    st.markdown(f"📌 {ref}")
                            
                            # Fontes completas
                            if source_docs:
                                st.markdown("---")
                                st.markdown("### 📚 Fontes Documentais")
                                for i, doc in enumerate(source_docs):
                                    with st.expander(f"📄 {doc.metadata.get('source', 'Desconhecido')} (Chunk {doc.metadata.get('chunk_index', '?')}/{doc.metadata.get('total_chunks', '?')})"):
                                        st.text(doc.page_content)
                        
                        # Salvar no histórico
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer
                        })
                        
                    else:
                        st.error("❌ Sistema temporariamente indisponível.")
                        if modo == "cidadao":
                            st.info("📞 Entre em contato pelo telefone (XX) XXXX-XXXX.")
                        
                except Exception as e:
                    if modo in ["admin", "funcionario"]:
                        st.error(f"❌ Erro técnico: {str(e)}")
                    else:
                        st.error("❌ Desculpe, ocorreu um erro. Por favor, tente novamente.")
                        st.info("📞 Se o problema persistir, ligue para (XX) XXXX-XXXX.")

if __name__ == "__main__":
    main()
