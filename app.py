import streamlit as st
import google.generativeai as genai
import os

st.title("🕵️ Diagnóstico de Modelos Google")

# Pega a chave dos Secrets
api_key = st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    st.error("Chave não encontrada nos Secrets!")
    st.stop()

genai.configure(api_key=api_key)

st.write("Conectando ao Google... Buscando modelos disponíveis para sua chave...")

try:
    modelos = list(genai.list_models())
    encontrou_embedding = False
    
    st.subheader("Lista de Modelos Disponíveis:")
    
    for m in modelos:
        # Mostra todos, mas destaca os de Embedding
        if 'embedContent' in m.supported_generation_methods:
            st.success(f"✅ MODELO DE EMBEDDING ENCONTRADO: {m.name}")
            encontrou_embedding = True
        else:
            st.text(f"Modelo (Outros): {m.name}")

    if not encontrou_embedding:
        st.error("❌ Nenhum modelo de embedding encontrado. Verifique se a API 'Generative Language' está ativada no Google Cloud.")

except Exception as e:
    st.error(f"Erro ao conectar: {e}")
