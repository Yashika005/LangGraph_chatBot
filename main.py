import streamlit as st
from dotenv import load_dotenv
from chatbot.ingest import ingest_pdfs_if_needed
from chatbot.flow import build_graph
from chatbot.vectordb import get_vectordb

load_dotenv()

st.set_page_config(page_title="LangGraph RAG Chatbot", layout="wide")
st.title("🤖 LangGraph RAG Chatbot")

# ==================== SIDEBAR ====================
st.sidebar.header("⚙️ RAG Strategy Selector")

rag_strategy = st.sidebar.selectbox(
    "Choose RAG Strategy:",
    [
        "standard",
        "multi_query",
        "hierarchical",
        "graph",
        "memory_augmented",
        "agentic",
        "hybrid",
        "modular",
        "verified"
    ],
    format_func=lambda x: {
        "standard": "📝 Standard RAG",
        "multi_query": "🔍 Multi-Query RAG",
        "hierarchical": "🌲 Hierarchical RAG",
        "graph": "🕸️ Graph RAG",
        "memory_augmented": "🧠 Memory-Augmented RAG",
        "agentic": "🤖 Agentic RAG",
        "hybrid": "🔀 Hybrid RAG",
        "modular": "🔧 Modular RAG",
        "verified": "✅ Verified RAG"
    }[x]
)

# Strategy descriptions
strategy_descriptions = {
    "standard": "Basic RAG with query rewriting and semantic search",
    "multi_query": "Generates multiple query variations for comprehensive retrieval",
    "hierarchical": "Two-stage retrieval with re-ranking for better relevance",
    "graph": "Explores relationships between documents and chunks",
    "memory_augmented": "Enhanced conversation memory and context awareness",
    "agentic": "Tool-based RAG with dynamic strategy selection",
    "hybrid": "Combines semantic search with keyword matching",
    "modular": "Production-ready with fallbacks and error handling",
    "verified": "Anti-hallucination with fact-checking verification"
}

st.sidebar.info(f"**ℹ️ {strategy_descriptions[rag_strategy]}**")

st.sidebar.header("📊 Document Statistics")

with st.spinner("Checking PDFs..."):
    ingest_pdfs_if_needed()

vectordb = get_vectordb()
stats = vectordb.get_stats()
st.sidebar.write(f"**Total chunks:** {stats['total_chunks']}")
st.sidebar.write(f"**Documents indexed:** {len(stats['sources'])}")

if stats['sources']:
    with st.sidebar.expander("View Document Details"):
        for doc, count in stats['sources'].items():
            st.write(f"📄 {doc}: {count} chunks")

# Build or rebuild graph when strategy changes
if "current_strategy" not in st.session_state or st.session_state.current_strategy != rag_strategy:
    st.session_state.graph = build_graph(rag_strategy)
    st.session_state.current_strategy = rag_strategy
    st.sidebar.success(f"✅ Loaded: {strategy_descriptions[rag_strategy]}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.sidebar.button("🗑️ Clear Conversation"):
    st.session_state.chat_history = []
    st.rerun()

# ==================== CHAT INTERFACE ====================
for q, a in st.session_state.chat_history:
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)

question = st.chat_input("Ask something about the PDFs")

if question:
    st.chat_message("user").write(question)
    
    with st.spinner(f"🔄 Processing with {rag_strategy.replace('_', ' ').title()} strategy..."):
        try:
            # Build memory from chat history
            memory = []
            for i, (q, a) in enumerate(st.session_state.chat_history[-5:]):
                memory.append({"role": "user", "content": q})
                memory.append({"role": "assistant", "content": a})
            
            # Invoke graph with strategy
            result = st.session_state.graph.invoke({
                "question": question,
                "context": [],
                "answer": "",
                "memory": memory,
                "rag_strategy": rag_strategy
            })
            
            answer = result.get("answer", "No answer generated")
            
            st.chat_message("assistant").write(answer)
            
            # Save to history
            st.session_state.chat_history.append((question, answer))
            
            # Debug info in sidebar
            with st.sidebar.expander("🔍 Debug Info"):
                st.write(f"**Strategy Used:** {rag_strategy}")
                st.write(f"**Memory Size:** {len(memory)} messages")
                st.write(f"**Retrieved Chunks:** {len(result.get('context', []))}")
                if result.get("rewritten_query"):
                    st.write(f"**Rewritten Query:** {result['rewritten_query'][:100]}...")
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.error(error_msg)
            st.session_state.chat_history.append((question, error_msg))