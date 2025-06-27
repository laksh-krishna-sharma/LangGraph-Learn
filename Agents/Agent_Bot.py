from typing import List, TypedDict
from langchain_core.messages import HumanMessage
from langchain_ollama.llms import OllamaLLM
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]=os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

class AgentState(TypedDict):
    message: List[HumanMessage]

llm=OllamaLLM(model="gemma3:1b")

def process(state: AgentState) -> AgentState:
    """Process the state by appending a new message."""
    response = llm.invoke(state['message'])
    print(f"\nAI: {response}")
    return state

graph = StateGraph(AgentState)

graph.add_node('process', process)
graph.add_edge(START, 'process')
graph.add_edge('process', END)

agent = graph.compile()

user_input = input("You: ")

while user_input.lower() != 'exit()':
    agent.invoke({'message': [HumanMessage(content=user_input)]})
    user_input = input("You: ")