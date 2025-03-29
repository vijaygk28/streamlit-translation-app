import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the AI Model
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Create Prompt Template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# Define Output Parser
parser = StrOutputParser()

# Create Translation Chain
chain = prompt_template | model | parser

# Streamlit UI
st.set_page_config(page_title="AI Translator", page_icon="🌍")
st.title("🌍 AI Translator using Groq")
st.write("Enter text and specify a language to get an AI-powered translation.")

# User Input
text = st.text_area("Enter text to translate:")
language = st.text_input("Enter target language:")

if st.button("Translate"):
    if text.strip() and language.strip():
        with st.spinner("Translating..."):
            response = chain.invoke({"language": language, "text": text})
            st.success("Translation completed!")
            st.text_area("Translated Text:", response, height=100)
    else:
        st.warning("Please enter both text and target language.")
