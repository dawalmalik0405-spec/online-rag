from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os 
from retriever import search_query


load_dotenv()

nvidia_key = (os.getenv("NVIDIA_API_KEY") or "").strip().strip('"').strip("'")

llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    api_key=nvidia_key,
    temperature=0.2
)



rephrase_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Given a conversation history between a user and an assistant, and a follow-up question, rewrite the follow-up question into a clear, standalone search query that can be understood without the conversation history.

Rules:
1. Do NOT answer the question.
2. Only return the standalone rephrased question.
3. If the question is already standalone or needs no context from history, return the question unchanged."""
    ),
    (
        "human",
        """Conversation History:
{chat_history}

Follow-up Question: {question}

Standalone Search Query:"""
    )
])


def condense_query_if_needed(query: str, formatted_history: str = "") -> str:
    """
    Rewrites follow-up questions into standalone search queries for vector retrieval
    when prior conversation history exists.
    """
    if not formatted_history or not formatted_history.strip():
        return query

    try:
        messages = rephrase_prompt.format_messages(
            chat_history=formatted_history,
            question=query
        )
        response = llm.invoke(messages)
        rewritten = response.content.strip()
        if rewritten and len(rewritten) > 3:
            return rewritten
    except Exception:
        pass

    return query


def llm_core(query: str, formatted_history: str = "", filename: str = None):
    """
    Executes document-grounded RAG query against the specified document.
    Returns a dict with 'answer' and 'sources'.
    """
    if not filename:
        return {
            "answer": "Please select a document first.",
            "sources": []
        }

    # Rewrite follow-up question into standalone query for accurate vector retrieval
    search_text = condense_query_if_needed(query, formatted_history)
    chunks = search_query(query=search_text, filename=filename)


    if not chunks:
        return {
            "answer": "I could not find any relevant information in this document to answer your question.",
            "sources": []
        }

    context = "\n\n---\n\n".join([chunk["content"] for chunk in chunks])

    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful, document-grounded AI assistant.

Strict rules you must follow:
1. Answer ONLY using the facts present in the provided Context.
2. Do NOT use outside knowledge or make assumptions beyond what is explicitly stated.
3. If the answer cannot be found in the context, say: "I don't know based on the provided document."
4. Be factual, concise, and directly address the user's question.
5. You may format your answer with clear markdown (such as bullet points, numbered lists, bold text, or code blocks) for readability.
6. Greet the user politely when they greet you."""
        ),
        (
            "human",
            """Context:
{context}

Conversation so far:
{chat_history}

Question:
{question}"""
        )
    ])

    messages = prompt_template.format_messages(
        context=context,
        chat_history=formatted_history,
        question=query
    )

    try:
        response = llm.invoke(messages)
        answer_text = response.content
    except Exception as e:
        answer_text = f"An error occurred while communicating with the AI model: {str(e)}"

    sources = [
        {
            "chunk_index": c["metadata"].get("chunk_index", 0),
            "similarity": c["similarity"],
            "snippet": c["content"][:220] + "..." if len(c["content"]) > 220 else c["content"]
        }
        for c in chunks
    ]

    return {
        "answer": answer_text,
        "sources": sources
    }


def llm_stream(query: str, formatted_history: str = "", filename: str = None):
    """
    Generator that yields token-by-token streaming events:
    - Yields sources first
    - Streams tokens as they generate
    - Yields completion event with full text for history persistence
    """
    if not filename:
        yield {"type": "error", "data": "Please select a document first."}
        return

    # Rewrite follow-up query with conversation context for accurate retrieval
    search_text = condense_query_if_needed(query, formatted_history)
    chunks = search_query(query=search_text, filename=filename)


    sources = [
        {
            "chunk_index": c["metadata"].get("chunk_index", 0),
            "similarity": c["similarity"],
            "snippet": c["content"][:220] + "..." if len(c["content"]) > 220 else c["content"]
        }
        for c in chunks
    ]

    yield {"type": "sources", "data": sources}

    if not chunks:
        not_found = "I could not find any relevant information in this document to answer your question."
        yield {"type": "token", "data": not_found}
        yield {"type": "done", "full_answer": not_found}
        return

    context = "\n\n---\n\n".join([chunk["content"] for chunk in chunks])

    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful, document-grounded AI assistant.

Strict rules you must follow:
1. Answer ONLY using the facts present in the provided Context.
2. Do NOT use outside knowledge or make assumptions beyond what is explicitly stated.
3. If the answer cannot be found in the context, say: "I don't know based on the provided document."
4. Be factual, concise, and directly address the user's question.
5. You may format your answer with clear markdown (such as bullet points, numbered lists, bold text, or code blocks) for readability.
6. Greet the user politely when they greet you."""
        ),
        (
            "human",
            """Context:
{context}

Conversation so far:
{chat_history}

Question:
{question}"""
        )
    ])

    messages = prompt_template.format_messages(
        context=context,
        chat_history=formatted_history,
        question=query
    )

    full_text = ""
    try:
        for chunk in llm.stream(messages):
            token = chunk.content
            if token:
                full_text += token
                yield {"type": "token", "data": token}
    except Exception as e:
        err_msg = f"\n\n[Error from AI model: {str(e)}]"
        full_text += err_msg
        yield {"type": "token", "data": err_msg}

    yield {"type": "done", "full_answer": full_text}





# result = llm_core()
# print("Response from LLM:")
# print(result)


# def chat_with_llm():
    
#     while True:
#         query = input("You: ").strip()
#         if query.lower() in {"exit", "quit"}:
#             # print("Exiting chat.")
#             break
        
#         chat_history.add_user_message(query)
#         formatted_history = "\n".join(
#             f"{msg.type.upper()}: {msg.content}"
#             for msg in chat_history.messages
#         )
#         answer = llm_core(query, formatted_history)
#         # print(f"AI: {answer}")
#         chat_history.add_ai_message(answer)

    
        
# if __name__ == "__main__":
#     chat_with_llm()

    




