from typing import List, Dict, TypedDict
from chatbot.llm import get_llm, generate_response
from chatbot.vectordb import get_vectordb
import re

class ChatState(TypedDict):
    question: str
    rewritten_query: str
    context: List[str]
    sources: List[dict]
    answer: str
    memory: List[dict]
    rag_strategy: str

# ==================== MULTI-QUERY RAG ====================
def multi_query_rag_rewrite(state: ChatState):
    """Generate multiple query variations for better retrieval"""
    llm = get_llm()
    
    prompt = f"""Generate 3 different versions of this search query to retrieve relevant documents from a vector database.
Each version should capture different aspects of the question.

Original question: {state['question']}

Return ONLY the 3 queries, one per line, without numbering or explanation."""
    
    queries_text = generate_response(llm, prompt).strip()
    queries = [q.strip() for q in queries_text.split('\n') if q.strip()]
    
    # Add original query
    queries.insert(0, state['question'])
    
    print(f"🔍 Multi-Query RAG: Generated {len(queries)} query variations")
    return {"rewritten_query": " | ".join(queries[:4])}  # Store all queries

def multi_query_rag_retrieve(state: ChatState):
    """Retrieve using multiple query variations and merge results"""
    vectordb = get_vectordb()
    queries = state["rewritten_query"].split(" | ")
    
    all_docs = []
    seen_content = set()
    
    for query in queries:
        docs = vectordb.similarity_search(query, k=3)
        for doc in docs:
            # Avoid duplicates
            if doc.page_content not in seen_content:
                all_docs.append(doc)
                seen_content.add(doc.page_content)
    
    # Limit to top results
    all_docs = all_docs[:6]
    
    contexts = [doc.page_content for doc in all_docs]
    sources = [doc.metadata for doc in all_docs]
    
    print(f"📚 Multi-Query RAG: Retrieved {len(contexts)} unique chunks")
    return {"context": contexts, "sources": sources}

# ==================== HIERARCHICAL RAG ====================
def hierarchical_rag_retrieve(state: ChatState):
    """Two-stage retrieval: first get diverse documents, then focused chunks"""
    vectordb = get_vectordb()
    search_query = state.get("rewritten_query") or state["question"]
    
    # Stage 1: Get diverse document chunks (broader search)
    broad_docs = vectordb.similarity_search(search_query, k=10)
    
    # Stage 2: Re-rank and filter to most relevant
    llm = get_llm()
    
    # Create summaries of each chunk
    chunk_summaries = []
    for i, doc in enumerate(broad_docs[:6]):
        summary = doc.page_content[:200] + "..."
        chunk_summaries.append(f"{i+1}. {summary}")
    
    # Ask LLM to rank relevance
    ranking_prompt = f"""Given the question: "{state['question']}"

Rank these document chunks by relevance (most relevant first).
Return ONLY the numbers in order, comma-separated (e.g., "3,1,5,2,4,6").

Chunks:
{chr(10).join(chunk_summaries)}

Rankings:"""
    
    try:
        rankings_text = generate_response(llm, ranking_prompt).strip()
        rankings = [int(x.strip()) - 1 for x in rankings_text.split(',') if x.strip().isdigit()]
        
        # Reorder documents based on rankings
        reranked_docs = [broad_docs[i] for i in rankings if i < len(broad_docs)][:4]
    except:
        reranked_docs = broad_docs[:4]
    
    contexts = [doc.page_content for doc in reranked_docs]
    sources = [doc.metadata for doc in reranked_docs]
    
    print(f"🌲 Hierarchical RAG: Retrieved and re-ranked {len(contexts)} chunks")
    return {"context": contexts, "sources": sources}

# ==================== GRAPH RAG ====================
def graph_rag_retrieve(state: ChatState):
    """Build relationships between chunks and traverse the graph"""
    vectordb = get_vectordb()
    search_query = state.get("rewritten_query") or state["question"]
    
    # Initial retrieval
    initial_docs = vectordb.similarity_search(search_query, k=3)
    
    # Expand with related chunks (same document, nearby pages)
    expanded_docs = list(initial_docs)
    
    for doc in initial_docs:
        metadata = doc.metadata
        source_file = metadata.get("source_file")
        page = metadata.get("page")
        
        if source_file and page:
            # Find neighboring chunks
            all_data = vectordb.get()
            if all_data and all_data.get("metadatas"):
                for i, meta in enumerate(all_data["metadatas"]):
                    if meta and meta.get("source_file") == source_file:
                        neighbor_page = meta.get("page")
                        # Add adjacent pages
                        if neighbor_page and abs(int(neighbor_page) - int(page)) <= 1:
                            neighbor_doc = type("Doc", (), {
                                "page_content": all_data["documents"][i],
                                "metadata": meta
                            })()
                            expanded_docs.append(neighbor_doc)
    
    # Remove duplicates and limit
    seen = set()
    unique_docs = []
    for doc in expanded_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)
    
    unique_docs = unique_docs[:5]
    
    contexts = [doc.page_content for doc in unique_docs]
    sources = [doc.metadata for doc in unique_docs]
    
    print(f"🕸️ Graph RAG: Retrieved {len(contexts)} chunks with graph expansion")
    return {"context": contexts, "sources": sources}

# ==================== HYBRID RAG ====================
def hybrid_rag_retrieve(state: ChatState):
    """Combine semantic search with keyword matching"""
    vectordb = get_vectordb()
    search_query = state.get("rewritten_query") or state["question"]
    
    # Semantic search
    semantic_docs = vectordb.similarity_search(search_query, k=4)
    
    # Keyword search (BM25-like)
    keywords = extract_keywords(search_query)
    all_data = vectordb.get()
    
    keyword_matches = []
    if all_data and all_data.get("documents"):
        for i, doc_content in enumerate(all_data["documents"]):
            score = sum(1 for kw in keywords if kw.lower() in doc_content.lower())
            if score > 0:
                metadata = all_data["metadatas"][i] if all_data.get("metadatas") else {}
                keyword_matches.append((score, doc_content, metadata))
    
    # Sort by keyword match score
    keyword_matches.sort(reverse=True, key=lambda x: x[0])
    keyword_docs = [
        type("Doc", (), {"page_content": content, "metadata": meta})()
        for _, content, meta in keyword_matches[:3]
    ]
    
    # Merge results
    combined_docs = semantic_docs + keyword_docs
    seen = set()
    unique_docs = []
    for doc in combined_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)
    
    unique_docs = unique_docs[:5]
    
    contexts = [doc.page_content for doc in unique_docs]
    sources = [doc.metadata for doc in unique_docs]
    
    print(f"🔀 Hybrid RAG: Retrieved {len(contexts)} chunks (semantic + keyword)")
    return {"context": contexts, "sources": sources}

def extract_keywords(text: str) -> List[str]:
    """Extract important keywords from text"""
    # Remove common stop words and extract meaningful terms
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were'}
    words = re.findall(r'\w+', text.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 3]
    return keywords[:5]  # Top 5 keywords

# ==================== VERIFIED RAG (Anti-Hallucination) ====================
def verified_rag_generate(state: ChatState):
    """Generate answer with verification to prevent hallucinations"""
    llm = get_llm()
    
    if not state["context"]:
        return {
            "answer": "I couldn't find any relevant information in the documents to answer your question.",
            "memory": state.get("memory", [])
        }
    
    context_text = "\n\n".join(state["context"])
    
    # Step 1: Generate initial answer
    initial_prompt = f"""Answer the question using ONLY the context provided. If the answer is not in the context, say "I don't have enough information."

Context:
{context_text}

Question: {state['question']}

Answer:"""
    
    initial_answer = generate_response(llm, initial_prompt)
    
    # Step 2: Verify the answer against context
    verification_prompt = f"""You are a fact-checker. Verify if the answer is supported by the context.

Context:
{context_text}

Question: {state['question']}
Answer: {initial_answer}

Is this answer fully supported by the context? Reply with:
- "VERIFIED" if fully supported
- "PARTIALLY VERIFIED" if mostly supported but has minor unsupported details
- "NOT VERIFIED" if it contains information not in the context

Then provide a corrected answer if needed.

Format:
Status: [VERIFIED/PARTIALLY VERIFIED/NOT VERIFIED]
Corrected Answer: [corrected answer if needed, or "No correction needed"]"""
    
    verification_result = generate_response(llm, verification_prompt)
    
    # Parse verification
    final_answer = initial_answer
    if "NOT VERIFIED" in verification_result or "PARTIALLY VERIFIED" in verification_result:
        if "Corrected Answer:" in verification_result:
            corrected = verification_result.split("Corrected Answer:")[1].strip()
            if corrected and "No correction needed" not in corrected:
                final_answer = corrected
    
    # Add verification badge
    if "VERIFIED" in verification_result:
        final_answer = "✅ **[Verified Answer]**\n\n" + final_answer
    elif "PARTIALLY VERIFIED" in verification_result:
        final_answer = "⚠️ **[Partially Verified]**\n\n" + final_answer
    
    # Update memory
    updated_memory = state.get("memory", [])
    updated_memory.append({"role": "user", "content": state["question"]})
    updated_memory.append({"role": "assistant", "content": final_answer})
    
    if len(updated_memory) > 20:
        updated_memory = updated_memory[-20:]
    
    # Add sources
    if state.get("sources"):
        citations = []
        seen = set()
        for source in state["sources"]:
            if source and "source_file" in source:
                citation = source["source_file"]
                if "page" in source:
                    citation += f" (page {source['page']})"
                if citation not in seen:
                    seen.add(citation)
                    citations.append(citation)
        
        if citations:
            final_answer += "\n\n**Sources:** " + ", ".join(citations)
    
    print(f"✅ Verified RAG: Answer verified and generated")
    return {"answer": final_answer, "memory": updated_memory}

# ==================== AGENTIC RAG ====================
def agentic_rag_generate(state: ChatState):
    """Agent-based RAG with tool selection"""
    llm = get_llm()
    
    if not state["context"]:
        return {
            "answer": "I couldn't find any relevant information in the documents to answer your question.",
            "memory": state.get("memory", [])
        }
    
    context_text = "\n\n".join(state["context"])
    
    # Step 1: Determine if we need additional tools
    tool_selection_prompt = f"""You are an AI agent. Determine if you need any additional tools to answer this question.

Question: {state['question']}

Available tools:
1. SUMMARIZE - Summarize long context
2. CALCULATE - Perform calculations
3. COMPARE - Compare multiple items
4. DIRECT_ANSWER - Answer directly from context

Select ONE tool and explain why. Format: "TOOL: [tool_name] - [reason]"

Selection:"""
    
    tool_selection = generate_response(llm, tool_selection_prompt)
    
    # Step 2: Use selected tool
    if "SUMMARIZE" in tool_selection:
        answer_prompt = f"""Summarize the key information from the context that answers the question.

Context:
{context_text}

Question: {state['question']}

Summary:"""
    elif "COMPARE" in tool_selection:
        answer_prompt = f"""Compare the relevant items/concepts from the context to answer the question.

Context:
{context_text}

Question: {state['question']}

Comparison:"""
    elif "CALCULATE" in tool_selection:
        answer_prompt = f"""Extract numerical information and perform necessary calculations to answer the question.

Context:
{context_text}

Question: {state['question']}

Calculation and Answer:"""
    else:  # DIRECT_ANSWER
        answer_prompt = f"""Answer the question directly using the context.

Context:
{context_text}

Question: {state['question']}

Answer:"""
    
    answer = generate_response(llm, answer_prompt)
    answer = f"🤖 **[Agentic RAG - Tool: {tool_selection.split('TOOL:')[1].split('-')[0].strip() if 'TOOL:' in tool_selection else 'DIRECT_ANSWER'}]**\n\n{answer}"
    
    # Update memory
    updated_memory = state.get("memory", [])
    updated_memory.append({"role": "user", "content": state["question"]})
    updated_memory.append({"role": "assistant", "content": answer})
    
    if len(updated_memory) > 20:
        updated_memory = updated_memory[-20:]
    
    # Add sources
    if state.get("sources"):
        citations = []
        seen = set()
        for source in state["sources"]:
            if source and "source_file" in source:
                citation = source["source_file"]
                if "page" in source:
                    citation += f" (page {source['page']})"
                if citation not in seen:
                    seen.add(citation)
                    citations.append(citation)
        
        if citations:
            answer += "\n\n**Sources:** " + ", ".join(citations)
    
    print(f"🤖 Agentic RAG: Used tool-based generation")
    return {"answer": answer, "memory": updated_memory}

# ==================== MODULAR RAG (Production) ====================
def modular_rag_retrieve(state: ChatState):
    """Production-ready modular retrieval with fallbacks"""
    vectordb = get_vectordb()
    search_query = state.get("rewritten_query") or state["question"]
    
    contexts = []
    sources = []
    
    try:
        # Primary: Semantic search
        docs = vectordb.similarity_search(search_query, k=4)
        
        if docs:
            contexts = [doc.page_content for doc in docs]
            sources = [doc.metadata for doc in docs]
        else:
            # Fallback 1: Broader search with relaxed query
            print(" Primary search returned no results, trying fallback...")
            keywords = extract_keywords(search_query)
            if keywords:
                fallback_query = " ".join(keywords[:3])
                docs = vectordb.similarity_search(fallback_query, k=4)
                contexts = [doc.page_content for doc in docs]
                sources = [doc.metadata for doc in docs]
        
        if not contexts:
            # Fallback 2: Get any documents
            print(" Fallback search failed, retrieving sample documents...")
            all_data = vectordb.get()
            if all_data and all_data.get("documents"):
                contexts = all_data["documents"][:4]
                sources = all_data["metadatas"][:4] if all_data.get("metadatas") else []
    
    except Exception as e:
        print(f" Retrieval error: {e}, using empty context")
    
    print(f"🔧 Modular RAG: Retrieved {len(contexts)} chunks with fallback support")
    return {"context": contexts, "sources": sources}

def modular_rag_generate(state: ChatState):
    """Production-ready generation with error handling"""
    llm = get_llm()
    
    context_text = "\n\n".join(state["context"]) if state["context"] else "No context available"
    
    # Build memory context
    memory_context = ""
    if state.get("memory"):
        memory_context = "Previous conversation:\n"
        for msg in state["memory"][-6:]:
            memory_context += f"{msg['role'].capitalize()}: {msg['content'][:100]}...\n"
        memory_context += "\n"
    
    prompt = f"""{memory_context}You are a helpful assistant. Answer the question using the context below.

Context:
{context_text}

Question: {state['question']}

Answer:"""
    
    try:
        answer = generate_response(llm, prompt)
    except Exception as e:
        answer = f"I encountered an error while generating the response: {str(e)}"
    
    # Update memory
    updated_memory = state.get("memory", [])
    updated_memory.append({"role": "user", "content": state["question"]})
    updated_memory.append({"role": "assistant", "content": answer})
    
    if len(updated_memory) > 20:
        updated_memory = updated_memory[-20:]
    
    # Add sources
    if state.get("sources"):
        citations = []
        seen = set()
        for source in state["sources"]:
            if source and "source_file" in source:
                citation = source["source_file"]
                if "page" in source:
                    citation += f" (page {source['page']})"
                if citation not in seen:
                    seen.add(citation)
                    citations.append(citation)
        
        if citations:
            answer += "\n\n**Sources:** " + ", ".join(citations)
    
    print(f"🔧 Modular RAG: Generated answer with error handling")
    return {"answer": answer, "memory": updated_memory}