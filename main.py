import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


load_dotenv()


llm = ChatOpenAI(model_name="gpt-5", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Welcome to the LangChain Agent!")
    res = agent.invoke({"messages": [HumanMessage(content="search for a full stack developer job in tel aviv")]})
    print("Agent Response:", res) 

if __name__ == "__main__":
    main()
