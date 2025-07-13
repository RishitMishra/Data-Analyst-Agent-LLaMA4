"""
🧠 Data Analyst Agent
Submitted by: 
- Rishit Mishra (B.Tech CSE, BBDITM)
- Email: rishitmishra05@gmail.com
- Contact: +91 8318802901

Overview:
An intelligent, Streamlit-based data analyst agent powered by LLaMA-4 (Together.ai).
- Accepts: CSV, Excel, PDF, TXT, DOCX, JPG, PNG
- Auto-analyzes file contents in plain English
- Supports natural language follow-up questions
- Automatically shows visualizations when prompted
- Designed for non-technical users (no code output)

━━━━━━━━━━━━━━━━━━━━━
🛠️ Setup Instructions:

1. 🔧 Install all required libraries:
   pip install streamlit pandas matplotlib seaborn pytesseract python-docx pymupdf openai pillow

2. 🔑 Add your Together.ai API key:
   Replace the placeholder in this line:
   os.environ["OPENAI_API_KEY"] = "your_together_api_key_here"

3. 🔍 Install Tesseract OCR (for image support):
   Windows: https://github.com/UB-Mannheim/tesseract/wiki
   After installing, set the correct path:
   pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

4. ▶️ Run the app:
   streamlit run main.py

━━━━━━━━━━━━━━━━━━━━━
💬 How to Use:

1. Upload a file (CSV, Excel, image, or text).
2. Agent will automatically:
   - Acknowledge the upload
   - Analyze its contents
   - Provide a summary
3. Ask follow-up questions like:
   - "What is the average age?"
   - "Which department earns the most?"
4. To visualize data, simply say:
   - "Visualize the salary distribution"
   - "Show a heatmap"

✨ The model will:
- Generate plots
- Add visual explanations to the chat
- Remember uploaded content for follow-ups

━━━━━━━━━━━━━━━━━━━━━
Notes:
- No "visualize" button needed — the model handles it via prompts.
- Suitable for non-technical users (no Python code shown).
- Internet is required for LLaMA-4 API via Together.ai.

✅ Project is complete, functional, and submission-ready.
"""


import pandas as pd
import mimetypes
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
from openai import OpenAI
import os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


# Replace with your Together API key
os.environ["OPENAI_API_KEY"] = "your_together_api_key_here"

client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.environ["OPENAI_API_KEY"]
)

pytesseract.pytesseract.tesseract_cmd = "tesseract_path_here"  # Update with your Tesseract-OCR path

chat_history = []

def read_file(file_path):
    file_type, _ = mimetypes.guess_type(file_path)

    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)

    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)

    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    elif file_path.endswith('.docx'):
        import docx
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)

    elif file_path.endswith('.pdf'):
        import fitz  # pymupdf
        doc = fitz.open(file_path)
        return "\n".join(page.get_text() for page in doc)

    elif file_path.endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)

    else:
        return "Unsupported file type"
    
def visualize_data(df):

    numeric_cols = df.select_dtypes(include='number').columns
    categorical_cols = df.select_dtypes(include='object').columns
    description = ""

    if len(numeric_cols) > 0:
        st.subheader("📊 Histograms")
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(5, 3))
            df[col].hist(ax=ax, bins=10)
            ax.set_title(f"Histogram of {col}")
            st.pyplot(fig)
            description += f"The column '{col}' shows distribution of values.\n"

    if len(numeric_cols) >= 2:
        st.subheader("📈 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        description += "The correlation heatmap shows how numeric columns relate to each other.\n"

    if len(categorical_cols) > 0:
        st.subheader("📋 Categorical Value Counts")
        for col in categorical_cols:
            fig, ax = plt.subplots(figsize=(5, 3))
            df[col].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f"Distribution: {col}")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
            description += f"The column '{col}' shows frequency of each category.\n"

    return description


def ask_llama(prompt, context=""):
    global chat_history

    # Add the latest user message (with context if needed)
    chat_history.append({"role": "user", "content": context + "\n\n" + prompt})

    response = client.chat.completions.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[
            {"role": "system", "content": (
                "You are a helpful data analyst assistant. "
                "Only use plain, easy-to-understand language. "
                "Do not include any code, technical formulas, or programming examples. "
                "Your users are non-technical, so keep responses conversational, brief, and insightful."
            )}
        ] + chat_history,
        temperature=0.3
    )

    reply = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": reply})

    return reply


# Streamlit app setup


st.set_page_config(page_title="Data Analyst Agent", layout="wide")
st.title("🧠 Chat-based Data Analyst Agent")

# --- Session State Setup ---
if "context_parts" not in st.session_state:
    st.session_state.context_parts = []
if "context_files" not in st.session_state:
    st.session_state.context_files = set()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_dataframes" not in st.session_state:
    st.session_state.uploaded_dataframes = {}  # filename -> DataFrame
if "uploaded_texts" not in st.session_state:
    st.session_state.uploaded_texts = {}       # filename -> text
if "uploaded_image_names" not in st.session_state:
    st.session_state.uploaded_image_names = set()

# --- Display Chat History ---
for q, r in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(r)

# --- Upload Section ---
st.markdown("### 📁 Upload a File")
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "txt", "docx", "pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    filename = uploaded_file.name
    with open(filename, "wb") as f:
        f.write(uploaded_file.read())

    with st.chat_message("user"):
        st.markdown(f"📎 You uploaded **{filename}**")

    if filename not in st.session_state.context_files:
        content = read_file(filename)

        if isinstance(content, pd.DataFrame):
            st.session_state.uploaded_dataframes[filename] = content
            st.session_state.context_parts.append(content.to_csv(index=False))
            st.dataframe(content)
        else:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")) and not content.strip():
                content = "[No readable text found in image]"
            st.session_state.uploaded_texts[filename] = content
            st.session_state.context_parts.append(content)
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                st.session_state.uploaded_image_names.add(filename)

        st.session_state.context_files.add(filename)

        # Auto query after upload
        auto_query = f"Please analyze the contents of the uploaded file '{filename}'."

        with st.chat_message("user"):
            st.markdown(auto_query)

        with st.spinner("Analyzing..."):
            full_context = "\n\n".join(st.session_state.context_parts)
            response = ask_llama(auto_query, context=full_context)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.chat_history.append((auto_query, response))

    else:
        with st.chat_message("assistant"):
            st.markdown("This file is already uploaded. You can now ask follow-up questions.")

# --- Ask a Question ---
query = st.chat_input("Ask a question based on uploaded files...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    # Check for visualization trigger words
    trigger_keywords = ["visualize", "plot", "show graph", "chart", "see trends", "bar graph", "distribution", "heatmap"]
    trigger_visual = any(k in query.lower() for k in trigger_keywords)

    full_context = "\n\n".join(st.session_state.context_parts)

    if trigger_visual:
        if st.session_state.uploaded_dataframes:
            latest_df = list(st.session_state.uploaded_dataframes.values())[-1]
            latest_name = list(st.session_state.uploaded_dataframes.keys())[-1]
            explanation = visualize_data(latest_df)
            st.session_state.context_parts.append(f"Visualization Summary for {latest_name}:\n{explanation}")
            full_context += "\n\n" + explanation

            # Auto visual summary question
            followup_prompt = f"Explain the visualizations generated from the dataset in '{latest_name}'."
            with st.chat_message("user"):
                st.markdown(followup_prompt)
            with st.spinner("Analyzing visualizations..."):
                followup_response = ask_llama(followup_prompt, context=full_context)
            with st.chat_message("assistant"):
                st.markdown(followup_response)
            st.session_state.chat_history.append((followup_prompt, followup_response))

    # Main response to user’s question
    with st.spinner("Thinking..."):
        response = ask_llama(query, context=full_context)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.chat_history.append((query, response))
# End of Streamlit app setup