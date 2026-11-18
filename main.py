import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# --- Configuration ---
LLM_MODEL = "gpt-5"
RETRIEVER_K = 3
QUERY = "What is pinecone in machine learning?"
SEPARATOR = "=" * 70

# --- Components ---
print("Loading vector store...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

vectorstore = PineconeVectorStore(
    index_name=os.environ.get("PINECONE_INDEX_NAME"),
    embedding=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions about the content of a document. "
            "Use ONLY the following context to answer. If the answer is not in the context, say you don't know.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


# --- Helpers ---
def format_docs(docs) -> str:
    """Format the retrieved documents into a single string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def print_section(title: str) -> None:
    """Print a titled section header."""
    print(f"\n{SEPARATOR}")
    print(title)
    print(SEPARATOR)


def run_raw_llm(query: str) -> str:
    """Ask the LLM directly, without retrieval (no RAG)."""
    result = llm.invoke([HumanMessage(content=query)])
    return result.content


def retrieval_chain_without_lcel(query: str) -> str:
    """RAG done step by step, without LCEL."""
    # Step 1: retrieve relevant docs
    docs = retriever.invoke(query)

    # Step 2: format docs into a context string
    context = format_docs(docs)

    # Step 3: fill the prompt with the context and the question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: invoke the LLM with the formatted messages
    response = llm.invoke(messages)

    # Step 5: return the answer text
    return response.content


def create_retrieval_chain_with_lcel():
    """Build a RAG chain with LCEL. Invoke it with the question string."""
    retrieval_chain = (
        {
            # Both keys receive the same input (the question string) and run in parallel
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt_template  # receives {"context": ..., "question": ...}
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


# --- Entry point ---
def main() -> None:
    print("Asking a question...")

    print_section("Option 1: raw LLM (no RAG)")
    answer = run_raw_llm(QUERY)
    print("Answer:\n")
    print(answer)

    print_section("Option 2: RAG (retrieval without LCEL)")
    answer = retrieval_chain_without_lcel(QUERY)
    print("Answer:\n")
    print(answer)

    print_section("Option 3: RAG with lcel")
    chain = create_retrieval_chain_with_lcel()
    answer = chain.invoke(QUERY)
    print("Answer:\n")
    print(answer)



if __name__ == "__main__":
    main()
