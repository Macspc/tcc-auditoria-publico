import streamlit as st
import google.generativeai as genai
import os

st.set_page_config(page_title="Diagnóstico Google", layout="wide")
st.title("🕵️ Diagnóstico de Modelos Disponíveis")

# Pega a chave dos segredos
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
else:
    st.error("Chave API não encontrada nos Secrets!")
    st.stop()

try:
    st.info("Consultando os servidores do Google... Aguarde.")
    
    # Lista todos os modelos disponíveis para SUA conta
    models = list(genai.list_models())
    
    st.subheader("✅ Modelos de EMBEDDING (Para o Banco de Dados)")
    embed_models = [m for m in models if 'embedContent' in m.supported_generation_methods]
    
    if embed_models:
        for m in embed_models:
            st.success(f"NOME EXATO: {m.name}")
            st.json(m.to_dict()) # Mostra detalhes técnicos
    else:
        st.error("❌ Nenhum modelo de Embedding encontrado! Sua chave pode estar limitada.")

    st.divider()

    st.subheader("✅ Modelos de CHAT (Para a Resposta)")
    chat_models = [m for m in models if 'generateContent' in m.supported_generation_methods]
    
    if chat_models:
        for m in chat_models:
            st.info(f"NOME EXATO: {m.name}")
    else:
        st.error("❌ Nenhum modelo de Chat encontrado!")

except Exception as e:
    st.error(f"Erro fatal ao conectar: {str(e)}")
