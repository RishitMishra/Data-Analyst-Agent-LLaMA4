# 🧠 Data Analyst Agent

An interactive, intelligent **Data Analyst Agent** built with **Streamlit**, powered by the **LLaMA-4 Maverick model (via Together API)**. This assistant lets you upload data files, get instant insights, ask follow-up questions, and even generate visualizations via simple prompts.

---

## 📦 Features

- Supports diverse file formats:
  - CSV, Excel, PDF, TXT, DOCX, PNG, JPG
- Auto-analysis of uploaded content
- Conversational Q&A with LLaMA-4 Maverick
- Automatic data visualizations via prompts (no buttons required)
- Remembers context across multiple uploads
- Optimized for **non-technical users** (no code shown)

---

## 🚀 Setup Instructions

### 1. **Clone the Repository:**

```
git clone https://github.com/your-username/Data-Analyst-Agent.git
cd Data-Analyst-Agent
```

### 2. **Create and Activate a Virtual Environment:**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

### 4. **Install Tesseract OCR:**

- **Windows:** Download & install from: [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

- **Linux (Ubuntu/Debian):**

```bash
sudo apt-get install tesseract-ocr
```

- **macOS:**

```bash
brew install tesseract
```

Update the Tesseract path in the script:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
```

### 5. **Add Your Together API Key:**

Replace the placeholder in the code:

```python
os.environ["OPENAI_API_KEY"] = "your_together_api_key_here"
```

---

## ▶️ Running the App

```bash
streamlit run data_analyst_agent.py
```

Then open your browser at: [http://localhost:8501](http://localhost:8501)

---

## 💡 How to Use

1. Upload a file (CSV, Excel, image, text, etc.).
2. The agent automatically analyzes the file & responds.
3. Ask natural language questions like:
   - "What is the average salary?"
   - "Show a heatmap of correlations."
4. Visualizations will be generated if prompts like "visualize" or "plot" are detected.
5. Context is preserved for follow-up analysis across multiple files.

---

## 📝 Notes

- Internet connection is required for LLaMA-4 API via Together.ai.
- Ensure Tesseract is installed for OCR functionality on images.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🙌 Author

- **Rishit Mishra**
- Email: [rishitmishra05@gmail.com](mailto\:rishitmishra05@gmail.com)
- Contact: +91 8318802901

---

> For feedback, contributions, or issues, please open an issue or PR on this repository.

