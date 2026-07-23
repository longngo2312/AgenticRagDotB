"""
DAY 3 — Terminal Q&A for testing the RAG backbone without the web UI.

Usage:
  python scripts/chat_cli.py

Lets you type questions and see:
  - Retrieved chunks (debug mode)
  - Generated answer with citations
  - Confidence score
"""
# TODO D3: import retriever, LLM (Gemini), prompts
# TODO D3: simple REPL loop:
#   while True:
#       question = input("You: ")
#       docs = retrieve(question)
#       answer = generate(question, docs)
#       print("Agent:", answer)
#       print("Sources:", [d['doc_url'] for d in docs])
