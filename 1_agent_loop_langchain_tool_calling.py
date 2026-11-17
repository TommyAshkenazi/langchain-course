# from _typeshed import OpenBinaryMode
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "ollama:qwen3:1.7b"

# Tools


@tool
def get_product_price(product_name: str) -> str:
    """Get the price of a product."""
    """Look up the price of a product in the catalog and return it."""
    print(f"Looking up price for product: {product_name}")
    prices = {
        "laptop": "$999",
        "smartphone": "$699",
        "headphones": "$199",
    }
    return prices.get(product_name.lower(), "Product not found")


@tool
def apply_discount(price: float, tier: str) -> float:
    """Apply a membership discount to a numeric price and return the discounted price.
    Available tiers: bronze, silver, gold, platinum.
    """
    print(f"Applying {tier} discount to price: {price}")
    discounts = {
        "bronze": 0.05,
        "silver": 0.10,
        "gold": 0.15,
        "platinum": 0.20,
    }
    discount = discounts.get(tier.lower(), 0)
    return price * (1 - discount)


# agent loop


@traceable(name="Langchain Agent Loop with Tool Calling")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(model=MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Running agent with question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=""
            "You are a helpful assistant that can answer questions about product prices and apply discounts. "
            "STRICT RULES - you must follow: "
            "1. NEVER make up prices. If you don't know the price, say 'I don't know the price of that product.' "
            "2. ALWAYS use the get_product_price tool to look up prices. "
            "3. ALWAYS use the apply_discount tool (with a numeric price and a tier) to apply discounts. "
            "4. NEVER make up discount tiers. If you don't know the discount tier, ask the user for their membership tier (bronze, silver, gold, platinum)."
        ),
        HumanMessage(content=question),
    ]

    for it in range(MAX_ITERATIONS):
        print(
            f"---------------Iteration {it + 1}/{MAX_ITERATIONS}--------------------------------"
        )

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        #    if no tool, this is the final answer
        if not tool_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        #    process only the First tool call
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_input = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"Tool Call: {tool_name} with input: {tool_input}")

        tool_to_use = tools_dict.get(tool_name)
        if not tool_to_use:
            print(f"Error: Tool {tool_name} not found.")
            return f"Error: Tool {tool_name} not found."

        observation = tool_to_use.invoke(tool_input)

        print(f"Observation: {observation}")
        messages.append(ai_message)
        messages.append(
            ToolMessage(
                content=str(observation),
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            )
        )

    return "Reached the maximum number of iterations without a final answer."


if __name__ == "__main__":
    print("Welcome to the Product Price Agent! (.bind_tools)")
    result = run_agent("What is the price of a laptop after applying a gold discount?")
