import streamlit as st
import google.generativeai as genai
import os

st.set_page_config(page_title="Diagnóstico Rápido", layout="wide")
st.title("🕵️ Lista de Modelos Disponíveis")

# Configura a chave
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("Chave API não encontrada!")
    st.stop()

try:
    st.info("Conectando ao Google...")
    
    # Lista os modelos
    models = list(genai.list_models())
    
    st.subheader("🔹 Modelos de EMBEDDING (Para o Banco de Dados)")
    encontrou_embed = False
    for m in models:
        # Verifica se serve para embedding
        if 'embedContent' in m.supported_generation_methods:
            st.success(f"NOME: {m.name}")
            encontrou_embed = True
            
    if not encontrou_embed:
        st.error("❌ Nenhum modelo de Embedding encontrado nesta conta.")

    st.divider()

    st.subheader("🔸 Modelos de CHAT (Para a Resposta)")
    for m in models:
        # Verifica se serve para gerar texto
        if 'generateContent' in m.supported_generation_methods:
            st.info(f"NOME: {m.name}")

except Exception as e:
    st.error(f"Erro fatal: {str(e)}")
