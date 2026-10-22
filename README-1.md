# 🤖 AI Chat Assistant (LLM APIs & Application Development)

A Streamlit chat app that connects to LLM APIs (Groq / Google Gemini) via the OpenAI-compatible SDK.

## Features
- Secure API key handling (env variables / Streamlit secrets)
- Multi-turn chat history with system, user and assistant messages
- Custom instructions with presets
- Adjustable temperature, max tokens and context window (token optimisation)
- Structured JSON responses (answer, summary, key points, confidence)
- Error handling: invalid key, rate limit, timeout, network, server errors (with retry/backoff)
- Token usage and estimated cost tracking, chat export to JSON

## Run locally
```bash
pip install -r requirements.txt
cp .env.example .env     # add your key
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push this repo to GitHub.
2. share.streamlit.io -> New app -> select repo, main file `app.py`.
3. Advanced settings -> Secrets -> paste `GROQ_API_KEY = "..."`.
4. Deploy and copy the live URL.

## Tech
Python, Streamlit, OpenAI SDK, python-dotenv
