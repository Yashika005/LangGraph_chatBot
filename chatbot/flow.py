from typing import TypedDict, List
from langgraph.graph import StateGraph
from chatbot.llm import get_llm, generate_response
from chatbot.vectordb import get_vectordb
from chatbot.rag_strategies import (
    multi_query_rag_rewrite,
    multi_query_rag_retrieve,
    hierarchical_rag_retrieve,
    graph_rag_retrieve,
    hybrid_rag_retrieve,
    verified_rag_generate,
    agentic_rag_generate,
    modular_rag_retrieve,
    modular_rag_generate,
)

class ChatState(TypedDict):
    question: str
    rewritten_query: str
    context: List[str]
    sources: List[dict]
    answer: str
    memory: List[dict]
    rag_strategy: str  # NEW: Strategy selector

def rewrite_query(state: ChatState):
    """Standard query rewriting"""
    strategy = state.get("rag_strategy", "standard")
    
    # Multi-Query RAG uses its own rewriting
    if strategy == "multi_query":
        return multi_query_rag_rewrite(state)
    
    llm = get_llm()

    memory_context = ""
    if state.get("memory"):
        memory_context = "Previous conversation:\n"
        for msg in state["memory"][-6:]:
            memory_context += f"{msg['role'].capitalize()}: {msg['content']}\n"

    prompt = f"""
You are a search query optimizer for a vector database.

Your task:
- Rewrite the user question into a short, precise search query
- Use previous conversation ONLY if needed
- Focus on keywords and entities
- DO NOT answer the question
- Return ONLY the search query

{memory_context}

User question:
{state['question']}

Search query:
"""

    rewritten = generate_response(llm, prompt).strip()

    print(f"🔍 Rewritten search query: {rewritten}")

    return {"rewritten_query": rewritten}


def retrieve(state: ChatState):
    """Route to appropriate retrieval strategy"""
    strategy = state.get("rag_strategy", "standard")
    
    print(f"📊 Using RAG strategy: {strategy}")
    
    if strategy == "multi_query":
        return multi_query_rag_retrieve(state)
    elif strategy == "hierarchical":
        return hierarchical_rag_retrieve(state)
    elif strategy == "graph":
        return graph_rag_retrieve(state)
    elif strategy == "hybrid":
        return hybrid_rag_retrieve(state)
    elif strategy == "modular":
        return modular_rag_retrieve(state)
    else:  # standard or memory-augmented
        vectordb = get_vectordb()
        search_query = state.get("rewritten_query") or state["question"]
        docs = vectordb.similarity_search(search_query, k=4)

        contexts = [doc.page_content for doc in docs] if docs else []
        sources = [doc.metadata for doc in docs] if docs else []
        
        source_files = set()
        for source in sources:
            if source and "source_file" in source:
                source_files.add(source["source_file"])
        
        print(f"📚 Retrieved {len(contexts)} context chunks from {len(source_files)} document(s): {list(source_files)}")
        
        return {"context": contexts, "sources": sources}

def generate(state: ChatState):
    """Route to appropriate generation strategy"""
    strategy = state.get("rag_strategy", "standard")
    
    if strategy == "verified":
        return verified_rag_generate(state)
    elif strategy == "agentic":
        return agentic_rag_generate(state)
    elif strategy == "modular":
        return modular_rag_generate(state)
    else:  # standard, multi_query, hierarchical, graph, hybrid, memory-augmented
        llm = get_llm()
        
        if not state["context"]:
            return {
                "answer": "I couldn't find any relevant information in the documents to answer your question.",
                "memory": state.get("memory", [])
            }
        
        context_text = "\n\n".join(state["context"])
        
        # Add document source info
        source_info = ""
        if state.get("sources"):
            unique_sources = {}
            for source in state["sources"]:
                if source and "source_file" in source:
                    file_name = source["source_file"]
                    page = source.get("page", "N/A")
                    if file_name not in unique_sources:
                        unique_sources[file_name] = []
                    unique_sources[file_name].append(str(page))
            
            source_info = "Information retrieved from the following sources:\n"
            for file_name, pages in unique_sources.items():
                source_info += f"- {file_name} (pages: {', '.join(set(pages))})\n"
            source_info += "\n"
        
        # Build conversation history
        memory_context = ""
        if state.get("memory"):
            memory_context = "Previous conversation:\n"
            for msg in state["memory"][-6:]:
                memory_context += f"{msg['role'].capitalize()}: {msg['content']}\n"
            memory_context += "\n"
        
        prompt = f"""You are a helpful assistant. 
Answer the question strictly using the context below. If the context doesn't contain the answer, say so.

{memory_context}{source_info}Context:
{context_text}

Question: 
{state['question']}

Answer:"""
        
        try:
            answer = generate_response(llm, prompt)
            
            # Update memory
            updated_memory = state.get("memory", [])
            updated_memory.append({"role": "user", "content": state["question"]})
            updated_memory.append({"role": "assistant", "content": answer})
            
            if len(updated_memory) > 20:
                updated_memory = updated_memory[-20:]
            
            # Add citations
            if state.get("sources"):
                citations = []
                for source in state["sources"]:
                    if source and "source_file" in source:
                        citation = source["source_file"]
                        if "page" in source:
                            citation += f" (page {source['page']})"
                        citations.append(citation)
                
                if citations:
                    unique_citations = []
                    seen = set()
                    for citation in citations:
                        if citation not in seen:
                            seen.add(citation)
                            unique_citations.append(citation)
                    
                    if unique_citations:
                        citation_text = "\n\n**Sources:** " + ", ".join(unique_citations)
                        answer += citation_text
            
            print(f"✅ Generated answer: {answer[:100]}...")
            return {"answer": answer, "memory": updated_memory}
        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            return {
                "answer": f"Error generating response: {str(e)}",
                "memory": state.get("memory", [])
            }

def build_graph(rag_strategy="standard"):
    """Build graph with specified RAG strategy"""
    graph = StateGraph(ChatState)

    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.set_entry_point("rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "generate")

    return graph.compile()