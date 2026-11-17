import os
from typing import  List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for the agent's response"""

    answer: str = Field(description="The content of the agent's response")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used by the agent"
    )


llm = ChatOpenAI(model_name="gpt-5", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Welcome to the LangChain Agent!")
    res = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="search for a full stack developer job in tel aviv"
                )
            ]
        }
    )
    print("Agent Response:", res)


if __name__ == "__main__":
    main()
