"""DAY 5 — LangGraph graph wiring."""
# TODO D5: build StateGraph(AgentState)
#   nodes: guardrails → router → [retrieve → grade → generate → faithfulness]
#                                 [clarify]
#                                 [handoff]
#   conditional edges:
#     router   → retrieve | clarify | handoff
#     grade    → generate (sufficient) | retrieve (loop, max 3 attempts) | handoff
#     faithful → END (pass) | handoff (fail)
#   compile → app = graph.compile()
