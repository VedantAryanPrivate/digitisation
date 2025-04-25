import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MATHPIX_APP_ID = os.getenv("MATHPIX_APP_ID")
MATHPIX_APP_KEY = os.getenv("MATHPIX_APP_KEY")

client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
    st.stop()

st.title("Your Application")
st.write("This is a placeholder for your application code.")
