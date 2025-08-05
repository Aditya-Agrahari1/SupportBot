
# Food Delivery Support Chatbot

This project is a support chatbot for a food delivery app. It uses a PDF document as its knowledge base to answer user questions.

## 🛠 Tech Stack

- **Language**: Python
- **UI Framework**: Streamlit
- **LLM Orchestration**: LangChain
- **Language Model**: Google Gemini
- **Vector Store**: ChromaDB (In-Memory)

## How to Run

### 1. Install Dependencies

Make sure you have Python installed, then run:

```
pip install -r requirements.txt
```

### 2. Set Up Your API Key

Create a `.env` file in the root directory and add your Google API key like this:

```
GOOGLE_API_KEY="AIza..."
```

### 3. Run the App

Ensure your `faq.pdf` file is in the same folder as the app, then run:

```
streamlit run app.py
```

This will launch the app in your default web browser.

---

📄 **Note**: This chatbot will use the content of `faq.pdf` as its knowledge base. Make sure your questions align with the content inside.

