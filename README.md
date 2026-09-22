# Data Scientist BOT

A Streamlit tutor for data science, machine learning, generative AI, agentic AI, and Python.

## Features

- Chat with conversation memory and a small built-in knowledge base
- Image, text, and PDF attachments
- Mock interviews with scoring and a model answer
- Assignment generator with PDF download and AI grading
- Code generation and review
- Web research and fact-checking

## Run locally

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project folder:

```
GROQ_API_KEY=your_key_from_https://console.groq.com
```

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open [share.streamlit.io](https://share.streamlit.io) and choose **Create app**.
3. Repository: `Hisingh11/data-science-tutor`
4. Branch: `main`
5. Main file: `app.py`
6. In **Advanced settings → Secrets**, paste:

```toml
GROQ_API_KEY = "your_key_from_https://console.groq.com"
```

Accounts and chat history use a local SQLite file. On Streamlit Cloud that file resets when the app restarts, so treat accounts as temporary.

## Notes

- Chat uses Groq models `openai/gpt-oss-120b` and `openai/gpt-oss-20b`.
- Image attachments use `qwen/qwen3.8-27b`.
- The code assistant explains and reviews code. It does not execute code on the server.
