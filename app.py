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
st.set_page_config(
    page_title="IA Auditoria Municipal - Consulta Avançada", 
    layout="wide", 
    page_icon="🏛️"
)

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

# --- 2. SISTEMA DE AUTENTICAÇÃO SIMPLIFICADO ---
def verificar_autenticacao():
    """Verifica se usuário está autenticado como servidor ou admin"""
    if "autenticado" not in st.session_state:
        st.session_state.autenticado = False
        st.session_state.perfil = None
    
    # Verificar query params para login
    query_params = st.query_params
    modo = query_params.get("mode", "cidadao")
    
    if modo in ["servidor", "admin"]:
        # Em produção, aqui teria uma autenticação real (OAuth, banco de dados, etc.)
        if "token" in query_params:
            token_valido = validar_token(query_params["token"])
            if token_valido:
                st.session_state.autenticado = True
                st.session_state.perfil = modo
            else:
                st.session_state.autenticado = False
                st.session_state.perfil = "cidadao"
        else:
            st.session_state.autenticado = False
            st.session_state.perfil = "cidadao"
    else:
        st.session_state.autenticado = False
        st.session_state.perfil = "cidadao"
    
    return st.session_state.autenticado, st.session_state.perfil

def validar_token(token):
    """Valida token de acesso (simplificado para exemplo)"""
    # Em produção: validar JWT, consultar banco, etc.
    tokens_validos = st.secrets.get("TOKENS_AUTORIZADOS", "").split(",")
    return token in tokens_validos

# --- 3. CARREGAMENTO DE SEGREDOS COM VALIDAÇÃO ---
if "GOOGLE_API_KEY" not in st.secrets or "PINECONE_API_KEY" not in st.secrets:
    st.error("❌ ERRO: Chaves de API não configuradas no secrets.toml!")
    st.info("💰 Para contratar a versão PRO com acesso completo, entre em contato: (XX) XXXX-XXXX")
    st.stop()

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

# Configurações do Pinecone
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = "tcc-auditoria"

# --- 4. INICIALIZAÇÃO CORRETA DO PINECONE ---
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
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
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
        st.error(f"❌ Erro ao conectar ao vectorstore: {str(e)}")
        return None

@st.cache_resource
def get_llm():
    """Inicializa o modelo de linguagem Gemini"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        temperature=0.1,
        max_retries=1
    )

# --- 5. PROCESSAMENTO DE PDF (Apenas Admin) ---
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

# --- 6. SISTEMA DE PROMPTS DIFERENCIADOS ---
def get_system_prompt(perfil):
    """Retorna o prompt adequado baseado no perfil do usuário"""
    
    if perfil == "admin":
        return (
            "Você é um assistente jurídico especializado da Auditoria Municipal, "
            "respondendo a um ADMINISTRADOR DO SISTEMA com acesso total.\n"
            "Forneça respostas TÉCNICAS, PRECISAS e DETALHADAS.\n"
            "Sempre cite a LEI, DECRETO, ARTIGO e PARÁGRAFO exato de onde a informação foi extraída.\n"
            "Formato de resposta:\n"
            "1. Resposta técnica completa\n"
            "2. Fundamentação legal detalhada (Lei nº X/Ano, Art. Y, §Z)\n"
            "3. Trecho exato do documento consultado\n"
            "Contexto recuperado dos documentos oficiais:\n{context}"
        )
    
    elif perfil == "servidor":
        return (
            "Você é um assistente técnico da Prefeitura Municipal, "
            "respondendo a um SERVIDOR PÚBLICO autorizado.\n"
            "Forneça respostas PRECISAS e FUNDAMENTADAS.\n"
            "Indique a lei, decreto e artigo correspondente.\n"
            "Contexto recuperado dos documentos oficiais:\n{context}"
        )
    
    else:  # cidadão
        return (
            "Você é um assistente virtual amigável da Prefeitura Municipal, "
            "respondendo a um CIDADÃO.\n"
            "Use uma linguagem SIMPLES, CLARA e ACESSÍVEL.\n"
            "Explique de forma casual e educativa, como se estivesse conversando.\n"
            "Não mencione números de lei ou artigos, apenas explique o conceito.\n"
            "Se a informação não estiver disponível, oriente onde o cidadão pode buscar ajuda.\n"
            "Contexto recuperado dos documentos oficiais:\n{context}"
        )

def formatar_resposta_cidadao(resposta_tecnica, source_docs):
    """Adapta resposta técnica para linguagem cidadã"""
    
    # Remove termos muito técnicos
    termos_tecnicos = {
        "in verbis": "conforme",
        "data venia": "com o devido respeito",
        "a priori": "em princípio",
        "a posteriori": "depois",
        "ad referendum": "para aprovação",
        "caput": "parte principal",
        "parágrafo único": "parte única",
        "inciso": "item",
        "alínea": "subitem",
    }
    
    resposta_simples = resposta_tecnica
    for termo, substituto in termos_tecnicos.items():
        resposta_simples = resposta_simples.replace(termo, substituto)
    
    # Adiciona tom mais casual
    introducoes = [
        "Então, vou te explicar de um jeito simples: ",
        "Olha só, é o seguinte: ",
        "Deixa eu te contar: ",
    ]
    
    import random
    resposta_final = random.choice(introducoes) + resposta_simples
    
    # Adiciona informação de ajuda
    resposta_final += (
        "\n\n📞 Se precisar de mais ajuda, você pode:"
        "\n• Ligar para o telefone da Prefeitura: (XX) XXXX-XXXX"
        "\n• Ir pessoalmente ao setor de atendimento ao cidadão"
        "\n• Acessar o site: www.macspc.com.br"
    )
    
    return resposta_final

def extrair_referencias(source_docs):
    """Extrai referências legais dos documentos fonte"""
    referencias = []
    
    import re
    for doc in source_docs:
        # Busca padrões de lei/decreto/artigo
        padrao_lei = r'(Lei|Decreto|Portaria|Resolução|Instrução Normativa)\s+(?:Federal|Estadual|Municipal)?\s*(?:n[º°]\.?\s*)?(\d+[./]?\d*)\s*(?:de\s*)?(?:(\d{1,2})\s*de\s*(\w+)\s*de\s*(\d{4}))?'
        padrao_artigo = r'[Aa]rt(?:igo)?\.?\s*(\d+[°º]?)\s*(?:[,.]?\s*(§\s*\d+|parágrafo\s+(?:único|\d+)))?'
        
        leis_encontradas = re.findall(padrao_lei, doc.page_content)
        artigos_encontrados = re.findall(padrao_artigo, doc.page_content)
        
        for lei in leis_encontradas:
            referencias.append(f"{lei[0]} nº {lei[1]}")
        
        for art in artigos_encontrados:
            referencias.append(f"Art. {art[0]} {art[1] if art[1] else ''}")
    
    return list(set(referencias))  # Remove duplicatas

# --- 7. INTERFACE DO USUÁRIO ---
def main():
    # Verificar autenticação
    autenticado, perfil = verificar_autenticacao()
    
    pc = init_pinecone()
    if pc is None:
        st.error("❌ Não foi possível inicializar o Pinecone. Verifique suas credenciais.")
        st.info("💡 Dica: Entre em contato com o suporte técnico pelo telefone (XX) XXXX-XXXX")
        return
    
    # Sidebar
    with st.sidebar:
        st.title("🏛️ Painel de Controle")
        
        if perfil == "admin":
            st.success("🔒 MODO ADMINISTRADOR")
            st.info("Acesso total às funcionalidades do sistema")
            
            st.markdown("---")
            st.subheader("📤 Upload de Documentos")
            
            uploaded_file = st.file_uploader(
                "Selecione o PDF para processar", 
                type="pdf",
                help="Apenas arquivos PDF com texto extraível"
            )
            
            if uploaded_file and st.button("🚀 Processar Documento", use_container_width=True):
                with st.spinner("Processando documento..."):
                    sucesso, msg = process_pdf_otimizado(uploaded_file)
                    if sucesso:
                        st.success(msg)
                        st.balloons()
                    else:
                        st.error(msg)
            
            st.markdown("---")
            st.subheader("📊 Estatísticas do Sistema")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Status", "Conectado ✅" if pc else "Desconectado ❌")
            with col2:
                st.metric("Índice", INDEX_NAME)
            
            # Botão de logout
            if st.button("🔓 Sair do Modo Admin", use_container_width=True):
                st.query_params.clear()
                st.rerun()
        
        elif perfil == "servidor":
            st.info("🔐 MODO SERVIDOR")
            st.success("Acesso autorizado à base documental")
            
            st.markdown("---")
            st.subheader("📚 Acesso à Legislação")
            st.markdown("""
            ✅ Consulta completa
            ✅ Referências legais
            ✅ Artigos e parágrafos
            """)
            
            if st.button("🔓 Sair", use_container_width=True):
                st.query_params.clear()
                st.rerun()
        
        else:
            st.info("👤 MODO CIDADÃO")
            st.markdown("---")
            st.subheader("📱 Canais de Atendimento")
            st.markdown("""
            📞 **Telefone:** (12) 9999-9999  
            📧 **Email:** contato@macspc.com.br  
            🌐 **Site:** www.macspc.com.br  
            🏢 **Presencial:** Rua X, 123 - Centro  
            
            ⏰ **Horário:** Seg-Sex, 8h às 17h
            """)
            
            # Opção de login
            st.markdown("---")
            st.markdown("### 🔐 Área Restrita")
            st.markdown("Para servidores e administradores:")
            
            codigo_acesso = st.text_input(
                "Código de acesso:", 
                type="password",
                placeholder="Digite seu código"
            )
            
            if st.button("Entrar", use_container_width=True):
                if codigo_acesso in st.secrets.get("TOKENS_AUTORIZADOS", "").split(","):
                    st.query_params["mode"] = "servidor"
                    st.query_params["token"] = codigo_acesso
                    st.rerun()
                else:
                    st.error("❌ Código inválido")
    
    # Área principal
    if perfil == "admin":
        st.title("🤖 Assistente Técnico - Modo Administrador")
        st.caption("Consultas detalhadas com referências legais completas")
    elif perfil == "servidor":
        st.title("🤖 Assistente Técnico - Modo Servidor")
        st.caption("Consultas fundamentadas na legislação municipal")
    else:
        st.title("💬 Assistente Virtual da Prefeitura")
        st.caption("Tire suas dúvidas sobre serviços e documentos municipais de forma simples!")
    
    # Histórico de Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input do usuário
    placeholder_text = {
        "admin": "Digite sua consulta técnica detalhada...",
        "servidor": "Pergunte sobre leis, decretos e procedimentos...",
        "cidadao": "Como posso ajudar? Pergunte sobre seus direitos, serviços, documentos..."
    }
    
    if prompt := st.chat_input(placeholder_text.get(perfil, "Digite sua dúvida...")):
        # Adiciona pergunta ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Consultando documentos oficiais..."):
                try:
                    vectorstore = get_vectorstore()
                    llm = get_llm()
                    
                    if vectorstore and llm:
                        # Configurar o Retriever do Pinecone
                        retriever = vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={
                                "k": 5 if perfil == "cidadao" else 10,  # Mais documentos para usuários logados
                                "filter": {"doc_type": "PDF"}
                            }
                        )
                        
                        # Criar o Prompt baseado no perfil
                        system_prompt = get_system_prompt(perfil)
                        
                        prompt_template = ChatPromptTemplate.from_messages([
                            ("system", system_prompt),
                            ("human", "{input}")
                        ])
                        
                        # Montar a Cadeia RAG
                        question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                        
                        # Executar a consulta
                        response = rag_chain.invoke({"input": prompt})
                        
                        answer = response["answer"]
                        source_docs = response["context"]
                        
                        # Formatar resposta baseado no perfil
                        if perfil == "cidadao":
                            # Simplifica para cidadão
                            answer = formatar_resposta_cidadao(answer, source_docs)
                            st.markdown(answer)
                            
                            # Apenas indica que tem fontes, sem detalhes técnicos
                            if source_docs:
                                st.markdown("---")
                                st.info("💡 Esta resposta foi baseada em documentos oficiais da Prefeitura Municipal.")
                        
                        elif perfil == "servidor":
                            # Resposta técnica com referências
                            st.markdown("### 📋 Resposta Técnica:")
                            st.markdown(answer)
                            
                            # Mostrar referências legais
                            referencias = extrair_referencias(source_docs)
                            if referencias:
                                st.markdown("---")
                                st.markdown("### ⚖️ Fundamentação Legal:")
                                for ref in referencias[:5]:  # Limita a 5 referências
                                    st.markdown(f"📌 {ref}")
                            
                            # Mostrar fontes
                            if source_docs:
                                st.markdown("---")
                                st.markdown("### 📚 Documentos Consultados:")
                                for i, doc in enumerate(source_docs[:3]):
                                    with st.expander(f"📄 Fonte {i+1} - {doc.metadata.get('source', 'Documento')}"):
                                        st.text(doc.page_content[:300] + "...")
                        
                        else:  # admin
                            # Resposta completa com tudo
                            st.markdown("### 📋 Análise Técnica Completa:")
                            st.markdown(answer)
                            
                            # Referências detalhadas
                            referencias = extrair_referencias(source_docs)
                            if referencias:
                                st.markdown("---")
                                st.markdown("### ⚖️ Referências Legais:")
                                for ref in referencias:
                                    st.markdown(f"📌 {ref}")
                            
                            # Fontes completas
                            if source_docs:
                                st.markdown("---")
                                st.markdown("### 📚 Fontes Documentais:")
                                for i, doc in enumerate(source_docs):
                                    with st.expander(f"📄 Documento {i+1} - {doc.metadata.get('source', 'Desconhecido')}"):
                                        st.text(doc.page_content)
                                        st.caption(f"Chunk: {doc.metadata.get('chunk_index')}/{doc.metadata.get('total_chunks')}")
                        
                        # Salva no histórico
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer
                        })
                    else:
                        st.error("❌ Erro interno: Sistema temporariamente indisponível.")
                        if perfil == "cidadao":
                            st.info("Por favor, tente novamente mais tarde ou entre em contato pelo telefone (XX) XXXX-XXXX.")
                        
                except Exception as e:
                    if perfil == "admin" or perfil == "servidor":
                        st.error(f"❌ Erro técnico: {str(e)}")
                    else:
                        st.error("❌ Desculpe, ocorreu um erro. Por favor, tente novamente ou entre em contato com a Prefeitura.")
                        st.info("📞 Telefone: (12) 99999-9999")

if __name__ == "__main__":
    main()
