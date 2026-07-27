"""DAY 5 — LangGraph node functions (each reads/writes AgentState)."""
# TODO D5: def guardrails_node(state)   — check input length, scope, prompt injection
def guardrails_node(state):
    pass 
def router_node(state):
    pass 
def retrieve_node(state):
    pass
def grace_node(state):
    pass
def faithfulness_node(state):
    pass
def clarity_node(state):
    pass
def handoff_node(state):
    pass
# TODO D5: def router_node(state)       — LLM decides: retrieve | clarify | handoff
# TODO D5: def retrieve_node(state)     — call search_docs tool, populate state['docs']
# TODO D5: def grade_node(state)        — LLM grades context sufficiency → route
# TODO D5: def generate_node(state)     — LLM generates answer with citations
# TODO D5: def faithfulness_node(state) — self-check answer vs docs → confidence score
# TODO D5: def clarify_node(state)      — ask user for clarification
# TODO D5: def handoff_node(state)      — trigger human handoff
