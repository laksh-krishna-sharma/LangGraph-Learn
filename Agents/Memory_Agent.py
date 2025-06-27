from typing import List, TypedDict, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama.llms import OllamaLLM
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]=os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

class AgentState(TypedDict):
    message: List[Union[HumanMessage, AIMessage]]

llm=OllamaLLM(model="gemma3:1b")

def process(state: AgentState) -> AgentState:
    """Process the state by appending a new message."""
    response = llm.invoke(state['message'])
    state['message'].append(AIMessage(content=response))
    print(f"\nAI: {response}")
    return state

graph = StateGraph(AgentState)

graph.add_node('process', process)
graph.add_edge(START, 'process')
graph.add_edge('process', END)

agent = graph.compile()

conversation_history = []

user_input = input("You: ")

while user_input.lower() != 'exit()':
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({'message': conversation_history})
    conversation_history = result['message']

    user_input = input("You: ")

with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n\n")
    
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")