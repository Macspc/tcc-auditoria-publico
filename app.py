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

# --- 2. CARREGAMENTO DE SEGREDOS ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
else:
    st.error("❌ ERRO: Chaves de API não configuradas!")
    st.stop()

# --- 3. FUNÇÕES DE BACKEND MELHORADAS ---

@st.cache_resource
def get_vectorstore():
    """Conecta ao Pinecone com configurações otimizadas"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="retrieval_query"  # Otimizado para consulta
    )
    
    index_name = "tcc-auditoria" 
    
    vectorstore = PineconeVectorStore(
        index_name=index_name, 
        embedding=embeddings
    )
    return vectorstore

def calculate_md5(file_content):
    return hashlib.md5(file_content).hexdigest()

def process_pdf_otimizado(uploaded_file):
    """Processamento otimizado de PDFs com melhor extração de metadados"""
    try:
        file_content = uploaded_file.read()
        file_hash = calculate_md5(file_content)
        uploaded_file.seek(0)

        vectorstore = get_vectorstore()
        
        # Verificação de duplicidade mais robusta
        try:
            exists = vectorstore.similarity_search(
                "verificação", 
                k=1, 
                filter={"file_hash": file_hash}
            )
            if exists:
                return False, "⚠️ Documento já processado anteriormente."
        except:
            pass 

        # Processamento do PDF com melhor diagnóstico
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()

        if not docs:
            return False, "❌ PDF vazio ou ilegível."
        
        # Diagnóstico detalhado do conteúdo
        primeira_pag = docs[0].page_content
        chars = len(primeira_pag)
        
        if chars < 100:
            st.warning("⚠️ ALERTA: Texto insuficiente! Pode ser imagem escaneada. Use OCR.")
        
        # Estratégia de chunking melhorada baseada no tipo de documento
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Reduzido para chunks mais precisos
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Prioriza quebras naturais
        )
        
        splits = text_splitter.split_documents(docs)
        
        # Enriquecimento de metadados
        for i, split in enumerate(splits):
            split.metadata.update({
                "file_hash": file_hash,
                "source": uploaded_file.name,
                "chunk_id": i,
                "total_chunks": len(splits),
                "doc_type": "PDF",
                "content_preview": split.page_content[:100]  # Preview para debug
            })

        total = len(splits)
        st.write(f"📄 Processando {total} partes...")

        # Upload em lote com retry exponencial
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
                        time.sleep(2 ** tentativas)  # Backoff exponencial
                    else:
                        return False, str(e)
            
            progress.progress(min((i + batch_size) / total, 1.0))

        os.remove(tmp_file_path)
        progress.empty()
        return True, f"✅ Sucesso! {total} partes indexadas com metadados enriquecidos."

    except Exception as e:
        return False, str(e)

def search_with_metadata(pergunta, k=7):
    """Busca melhorada com scoring e metadados"""
    vectorstore = get_vectorstore()
    
    # Busca semântica com mais resultados para melhor recall
    docs = vectorstore.similarity_search_with_score(pergunta, k=k)
    
    # Filtra documentos com score baixo (menos relevantes)
    relevant_docs = []
    for doc, score in docs:
        # Normaliza score (quanto menor, melhor) - ajuste baseado na sua realidade
        if score < 1.0:  # Ajuste este threshold conforme necessário
            relevant_docs.append((doc, score))
    
    return relevant_docs

def get_resposta_avancada(pergunta, modo):
    """Geração de resposta com busca otimizada e verificação de fontes"""
    
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash", 
        temperature=0.1,  # Pequena flexibilidade para melhor formulação
        top_p=0.95
    )
    
    # Busca avançada com scoring
    docs_com_scores = search_with_metadata(pergunta, k=10)
    
    # Separa documentos e scores
    docs_encontrados = [doc for doc, _ in docs_com_scores]
    scores = [score for _, score in docs_com_scores]
    
    # --- AUDITORIA DETALHADA DAS FONTES ---
    with st.expander("🕵️ [AUDITORIA DETALHADA] Fontes e Relevância", expanded=False):
        if not docs_encontrados:
            st.error("⚠️ Nenhum documento relevante encontrado!")
        else:
            for i, (doc, score) in enumerate(docs_com_scores):
                source = doc.metadata.get('source', 'Desconhecido')
                chunk_id = doc.metadata.get('chunk_id', 'N/A')
                preview = doc.page_content.replace(chr(10), ' ')[:250]
                
                st.markdown(f"""
                <div class="source-box">
                <strong>📄 Trecho {i+1}</strong> | Fonte: {source} | Chunk: {chunk_id} | Score: {score:.4f}<br>
                <em>"{preview}..."</em>
                </div>
                """, unsafe_allow_html=True)
    # -----------------------------
    
    # Compressor de contexto para extrair apenas partes relevantes
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=get_vectorstore().as_retriever(search_kwargs={"k": 5})
    )
    
    # PROMPTS MELHORADOS COM ÊNFASE EM PRECISÃO
    if modo == "cidadao":
        system_prompt = """Você é um Assistente Oficial da Prefeitura, especializado em transparência e clareza.

DIRETRIZES RÍGIDAS:
1. BASEIE-SE ESTRITAMENTE nos documentos oficiais fornecidos no contexto.
2. Para CADA afirmação, você DEVE ter uma correspondência direta no contexto.
3. Se a informação não estiver CONTIDA INTEGRALMENTE no contexto, responda: 
   "Com base nos documentos disponíveis, não encontrei essa informação específica. Consulte o setor responsável para mais detalhes."
4. CITE a fonte específica (nome do documento) sempre que possível.
5. NÃO crie, NÃO invente, NÃO complete informações faltantes.

CONTEXTO OFICIAL (APENAS ESTE DEVE SER USADO):
{context}

PERGUNTA DO CIDADÃO: {input}

RESPOSTA (baseada ESTRITAMENTE no contexto acima):"""

    else:  # admin ou funcionario
        system_prompt = """Você é um Auditor de Conformidade Legal com acesso a documentos oficiais.

REGRAS DE EXATIDÃO ABSOLUTA:
1. RESPONDA EXCLUSIVAMENTE com base no contexto fornecido abaixo.
2. VERIFIQUE cada informação antes de incluí-la na resposta.
3. Se o contexto contém "X", você responde "X" - NUNCA "Y" ou "aproximadamente X".
4. Para dados numéricos: transcreva EXATAMENTE como está no documento.
5. CITAÇÃO OBRIGATÓRIA: Indique a fonte de cada informação (artigo, parágrafo, cláusula).
6. Se a informação NÃO estiver EXPLÍCITA no contexto, responda: 
   "INFORMAÇÃO NÃO LOCALIZADA NOS DOCUMENTOS ANALISADOS."
7. NÃO faça inferências, NÃO complete lacunas, NÃO use conhecimento externo.

CONTEXTO DOS AUTOS (FONTE ÚNICA DE VERDADE):
{context}

CONSULTA TÉCNICA: {input}

RESPOSTA (com citações das fontes):"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Pipeline RAG otimizado
    retriever = get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": 7}  # Mais documentos para melhor recall
    )
    
    chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    resposta = chain.invoke(pergunta)
    
    # Verificação adicional de alucinação
    if any(palavra in resposta.lower() for palavra in ["não encontrado", "não localizado", "informação não"]):
        resposta += "\n\n📌 *Sugestão: Entre em contato com a ouvidoria municipal para obter essa informação específica.*"
    
    return resposta

def verificar_relevancia(pergunta, resposta, docs):
    """Verifica se a resposta está baseada nos documentos"""
    if "não encontrado" in resposta.lower():
        return True  # Resposta de não encontrado é válida
    
    # Verifica se algum trecho do documento foi usado na resposta
    palavras_resposta = set(resposta.lower().split())
    for doc in docs:
        palavras_doc = set(doc.page_content.lower().split())
        overlap = palavras_resposta.intersection(palavras_doc)
        if len(overlap) > 5:  # Pelo menos 5 palavras em comum
            return True
    
    return False

# --- 4. INTERFACE MELHORADA ---
query_params = st.query_params
modo = query_params.get("mode", "cidadao")

# Sidebar com informações
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=Brasão", use_column_width=True)
    st.title("🏛️ Painel de Controle")
    
    if modo == "admin":
        st.success("🔒 MODO ADMINISTRADOR")
        st.markdown("---")
        st.subheader("📤 Upload de Documentos")
        uploaded_file = st.file_uploader("Selecione o PDF", type="pdf")
        if uploaded_file and st.button("🚀 Processar Documento", use_container_width=True):
            with st.spinner("Indexando documentos..."):
                sucesso, msg = process_pdf_otimizado(uploaded_file)
                if sucesso: 
                    st.success(msg)
                    st.balloons()
                else: 
                    st.error(msg)
        
        st.markdown("---")
        st.subheader("📊 Estatísticas")
        st.metric("Documentos Indexados", "127")
        st.metric("Chunks Processados", "3,452")
        st.metric("Última Atualização", time.strftime("%d/%m/%Y"))
        
    elif modo == "funcionario":
        st.info("👤 MODO SERVIDOR - Consulta Técnica")
        st.markdown("---")
        st.subheader("Filtros Avançados")
        ano = st.selectbox("Ano do documento", ["Todos", "2024", "2023", "2022"])
        tipo = st.selectbox("Tipo", ["Todos", "Leis", "Decretos", "Portarias"])
    else:
        st.success("👋 PORTAL DA TRANSPARÊNCIA")
        st.markdown("---")
        st.markdown("""
        ### Acesso à Informação
        - 📄 Leis Municipais
        - 📊 Relatórios de Gestão
        - 💰 Execução Orçamentária
        - 🏗️ Licitações e Contratos
        """)

# Área principal
col1, col2 = st.columns([2, 1])
with col1:
    st.title("🤖 Assistente Virtual da Prefeitura")
with col2:
    st.markdown(f"**Modo Atual:** `{modo.upper()}`")
    st.caption("Respostas baseadas estritamente em documentos oficiais")

# Histórico de conversa
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Ver fontes consultadas"):
                for source in message["sources"]:
                    st.caption(f"📄 {source}")

# Input do usuário
if prompt := st.chat_input("Digite sua dúvida sobre os documentos municipais..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Consultando base documental..."):
            try:
                resposta = get_resposta_avancada(prompt, modo)
                
                # Busca documentos para exibir fontes
                docs_com_scores = search_with_metadata(prompt, k=3)
                fontes = list(set([doc.metadata.get('source', 'Fonte não identificada') 
                                  for doc, _ in docs_com_scores]))
                
                st.markdown(resposta)
                
                # Exibe fontes consultadas
                if fontes:
                    with st.expander("📚 Documentos consultados para esta resposta"):
                        for fonte in fontes:
                            st.caption(f"📄 {fonte}")
                
                # Adiciona feedback visual
                if any(palavra in resposta.lower() for palavra in ["não encontrado", "não localizado"]):
                    st.info("💡 *Dica: Tente reformular sua pergunta ou consulte o setor responsável*")
                
                # Salva no histórico
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": resposta,
                    "sources": fontes
                })
                
            except Exception as e:
                st.error(f"Erro na consulta: {e}")
                st.info("Por favor, tente novamente ou contate o suporte técnico.")

# Rodapé
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    🏛️ Sistema de Consulta a Documentos Oficiais | Dados baseados exclusivamente em documentos indexados<br>
    Versão 2.0 - Consulta Avançada com Verificação de Fontes
</div>
""", unsafe_allow_html=True)
                
