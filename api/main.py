"""DAY 6 — FastAPI server with WebSocket streaming."""
# TODO D6: FastAPI app with CORS
# TODO D6: POST /chat — single-turn, returns full response (testing)
# TODO D6: WS  /ws/{session_id} — streaming WebSocket
#   - receive: {question: str}
#   - stream tokens back as they arrive from LLM
#   - send citation cards as final message: {type: 'sources', sources: [...]}
# TODO D6: GET /health — liveness check
