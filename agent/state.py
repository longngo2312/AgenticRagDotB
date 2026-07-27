"""DAY 5 — LangGraph AgentState definition."""
# TODO D5: from typing import TypedDict, Annotated
#          class AgentState(TypedDict):
#            messages:   list          # full conversation history
#            query:      str           # rewritten standalone query
#            docs:       list          # retrieved + reranked parent chunks
#            attempts:   int           # self-correction loop counter
#            confidence: float         # faithfulness / grounding score
#            route:      str           # 'retrieve' | 'clarify' | 'handoff' | 'answer'
from typing import TypedDict, Annotated
class AgentState(TypedDict):
    messages:   list          # full conversation history
    query:      str           # rewritten standalone query
    docs:       list          # retrieved + reranked parent chunks
    attempts:   int           # self-correction loop counter
    confidence: float         # faithfulness / grounding score
    route:      str           # 'retrieve' | 'clarify' | 'handoff' | 'answer'
