# Week 2 — Building Agent Capabilities

**Presenters:** Yashika & Jaswantha
**Date:** 20-Aug-2026
**Theme:** Retrieval & Knowledge → RAG Deep Dive → Memory Systems → Tools & Integrations → Frameworks → Hands-on RAG/PDF Agent

---

# 1. Overview & Setup

## 1.1 Overview

## Goal

By the end of this session, participants will understand:

- What embeddings are and how similarity search works
- What a Vector Database is and why agents need one
- What RAG (Retrieval-Augmented Generation) is and its 9 major variants
- How to choose the right RAG pattern for a given problem
- What memory systems are (conversation, short-term, long-term, semantic)
- How agents integrate with APIs, web search, databases, and files
- LangChain basics, LangGraph fundamentals, and MCP (Model Context Protocol)
- How to build a PDF Question Answering Agent and a RAG Agent with a Vector DB (FAISS/Chroma)

---

## 1.2 Prerequisites & Setup

Before starting the session, ensure you have:

- Completed Week 1 (LLMs, Prompting, Agents, Tool Calling)
- Python 3.10+ installed
- API keys set up (`GEMINI_API_KEY` or `GORQ_API_KEY`)
- `langchain`, `langgraph`, and a vector store (`faiss-cpu` or `chromadb`) installed

---



# 2. Retrieval & Knowledge

## 2.0 What is Retrieval?

### Definition

Retrieval is the process of finding the most relevant information from an external data source based on a user's query.

Instead of expecting the AI model to remember everything, retrieval allows it to search a knowledge source and fetch only the relevant information before generating an answer.

### Example

Suppose you ask:

```text
"What is the leave policy for employees?"
```

**Without Retrieval:** The AI guesses from its training.

**With Retrieval:** It searches your company's HR document, finds the Leave Policy section, and answers using that information.

---

## 2.1 Why Retrieval Matters

An LLM only knows what it saw during training plus whatever fits in its context window. To answer questions about private, large, or fast-changing data, we need a way to **fetch** the relevant information first.

```text
Traditional LLM
User Question → LLM (trained knowledge only) → Answer

Retrieval-Augmented LLM
User Question → Search Knowledge Base → Relevant Chunks → LLM → Answer
```

---

## 2.2 Embeddings (Recap + Applied)

An embedding converts text into a numerical vector that captures semantic meaning.

```text
"How do I reset my password?"
            ↓
   Embedding Model
            ↓
[0.02, -0.31, 0.88, ...]
```

Key property: **semantically similar text produces vectors that are close together**, even if the wording is different.

```text
"reset my password"   ≈  "forgot password" ≈  "change login credentials"
```

This is what makes semantic search possible — matching by **meaning**, not exact keywords.

### Types of Embeddings

| Type of Embedding | What it means | Example | What happens / Why it is useful |
|---|---|---|---|
| Word Embedding | Converts an individual word into a vector. | "king" → [0.21, 0.45, -0.12, ...] | Words with related meanings have similar vectors. For example, "king" and "queen" will be closer than "king" and "banana". |
| Sentence Embedding | Converts an entire sentence into one vector representing its meaning. | "How do I reset my password?" → [0.02, -0.31, 0.88, ...] | "I forgot my password" can produce a vector close to the first sentence even though the words are different. Commonly used for semantic search and RAG. |
| Document Embedding | Converts a large piece of text/document into a vector. | A PDF about company policies → [0.14, -0.52, 0.76, ...] | Represents the overall content of the document, useful for comparing documents. In RAG, documents are usually split into smaller chunks first rather than embedding an entire long PDF as one vector. |
| Contextual Embedding | Creates a representation of a word/token based on the surrounding words. | "I deposited money in the bank." vs "I sat beside the river bank." | The word "bank" has different meanings in the two sentences, so its representation changes according to context. Important for understanding language. |
| Image Embedding | Converts an image into a numerical vector. | Photo of a dog → [0.31, -0.12, 0.87, ...] | Similar images produce similar vectors. Photos of dogs can be closer together than a dog photo and a car photo. Used for image search and similarity. |
| Multimodal Embedding | Represents different data types, such as text and images, in a shared vector space. | "golden retriever" text ≈ golden retriever image vector | Because the text and image vectors are close, you can search for an image using text. Enables text-to-image search and multimodal retrieval. |
| Code Embedding | Converts source code into a vector representing its meaning/function. | `def reset_password(user): ...` → [0.18, -0.52, 0.91, ...] | Code performing similar tasks can have similar embeddings. Useful for code search, code retrieval, and coding assistants. |
|

### Embedding Models

Embedding Model = Tool that converts text into meaningful vectors.

| Model Type | Example Model | Use |
|---|---|---|
| SentenceTransformer | `all-MiniLM-L6-v2` | Local, free, fast, and good for learning embeddings. |
| OpenAI Embeddings | `text-embedding-3-small` | High-quality API-based embeddings. |
| Cohere Embeddings | `embed-english-v3` | Alternative API-based embedding model. |
| HuggingFace Models | BGE / E5 models | Open-source embedding models for different use cases. |

---

## 2.3 Vector Databases

A Vector Database stores embeddings (plus their original text/metadata) and allows fast similarity search over millions of vectors.

```text
Documents
   ↓
Chunking
   ↓
Embedding Model
   ↓
Vectors
   ↓
Vector Database (FAISS / Chroma / Pinecone / Weaviate)
```

Common vector databases:

| Vector DB | Notes |
|---|---|
| FAISS | Local, fast, no server required, great for prototypes |
| Chroma | Local/embedded, easy Python API, persistent storage |
| Pinecone | Managed cloud vector DB |
| Weaviate | Open-source, supports hybrid search |
| pgvector | Vector search inside PostgreSQL |

---

## 2.4 Similarity Search

When a user asks a question, we:

```text
1. Embed the user query
2. Compare query vector to all stored vectors
3. Return the top-k closest matches
```

Similarity is typically measured using one of the following metrics:

| Metric | Idea | Formula | Usage |
|---|---|---|---|
| Cosine Similarity | Compares vector direction using angle. | $\cos(\theta)=\dfrac{\mathbf{A}\cdot\mathbf{B}}{\lVert\mathbf{A}\rVert\lVert\mathbf{B}\rVert}$ | Most used for text embeddings and semantic search. |
| Euclidean Distance | Measures straight-line distance between vectors. | $d(\mathbf{A},\mathbf{B})=\sqrt{\sum_{i=1}^{n}(A_i-B_i)^2}$ | Useful for spatial/numerical data; less preferred for semantic text. |
| Dot Product | Measures similarity using multiplication of corresponding components; affected by vector magnitude. | $\mathbf{A}\cdot\mathbf{B}=\sum_{i=1}^{n}A_iB_i$ | Used in recommendations, ranking, and vector search. |
| Manhattan Distance | Sums the absolute differences between vector components. | $d(\mathbf{A},\mathbf{B})=\sum_{i=1}^{n}\lvert A_i-B_i\rvert$ | Useful when differences along each dimension matter individually (e.g. grid-like data). |

```text
Query Vector
     ↓
Compare against Vector DB
     ↓
Top-k Nearest Neighbors
     ↓
Return matching chunks
```

---

# 3. RAG (Retrieval-Augmented Generation)

## 3.0 Chunking in RAG

Chunking means breaking a large document into smaller pieces called chunks before creating embeddings.

Instead of:

```text
Document → 1 Embedding
```

we do:

```text
Document → Chunks → Embeddings → Vector DB
```

### Why do we need chunking?

Suppose a document has 100 pages. If you create one embedding for the entire document, the embedding may represent the overall meaning, but it may not capture the specific information needed for a particular question.

Instead:

```text
Document
  │
  ├── Chunk 1 → Embedding 1
  ├── Chunk 2 → Embedding 2
  ├── Chunk 3 → Embedding 3
  ├── Chunk 4 → Embedding 4
  └── Chunk 5 → Embedding 5
```

When the user asks a question, RAG retrieves the most relevant chunks.

| Chunking Type | How it works | Advantages | Disadvantages | Best Use |
|---|---|---|---|---|
| Fixed-Size | Splits text into fixed token/character sizes. | Simple, fast, easy to implement | Can cut sentences/ideas; may lose context | Basic/general RAG |
| Recursive | Splits hierarchically: section → paragraph → sentence → smaller chunks. | Preserves document structure; good balance of quality and simplicity | More complex than fixed; boundaries aren't always semantically perfect |  General-purpose RAG |
| Semantic | Uses embeddings/similarity to group text with similar meaning. | Better semantic coherence; adapts to topic changes | More computationally expensive; slower; depends on embedding quality | Complex documents, meaning-focused retrieval |
| Late Chunking | Processes the full document first, then creates chunk embeddings from contextual token embeddings. | Preserves broader context; reduces information loss at chunk boundaries | Requires suitable long-context embedding model; more memory/compute; more complex |  Long documents where context across chunks matters |

---

## 3.1 What is RAG?

```text
RAG = Search + LLM
```

Instead of an LLM answering purely from memory:

- We search documents
- We retrieve relevant chunks
- We give those chunks to the LLM
- The LLM answers only from that data

```text
User Question
     ↓
Retrieve Relevant Chunks (Vector DB)
     ↓
Chunks + Question → LLM Prompt
     ↓
LLM Answer (grounded in retrieved data)
```

---

## 3.2 Types of RAG

There isn't just one way to build RAG. Depending on data size, latency budget, accuracy needs, and conversation style, different RAG patterns are used.

### 3.2.1 Standard RAG

A single user query is used for retrieval. The system finds the top-k matching chunks using embeddings and sends them to the LLM, which answers based only on those chunks.

```text
User Question → Vector Search (top-k) → LLM → Answer
```

![Standard RAG flow](Document/media/Standard_Rag.png)

**Advantages:** very simple, fast, cheap, easy to debug, best for beginners.
**Disadvantages:** sensitive to wording, may miss important chunks, weak on vague questions.

**Prefer when:** dataset is small/medium, questions are clear, low latency/cost matters, prototyping.
**Avoid when:** questions are ambiguous, documents are large/deep, high trust or complex reasoning is required.

**Example:** "What is OAuth?" → only chunks closely matching "OAuth" are retrieved.

---

### 3.2.2 Multi-Query RAG

Multiple versions of the user's question are generated, each searched separately, and results are merged.

> "Let's search the same thing in different ways"

```text
LLM generates 3-4 query variants → each searches Vector DB → merge → dedupe → LLM
```

![Multi-Query RAG flow](Document/media/MultiQuery.png)

**Advantages:** better document coverage, handles vague questions, higher recall.
**Disadvantages:** more LLM calls, slightly slower, higher cost.

**Prefer when:** questions are vague/short, recall matters more than speed, varied terminology.
**Avoid when:** low latency/budget is critical, questions are already well-defined.

**Example:** "Explain hashing" → generates "hashing in computer science", "hash functions explanation", "data hashing algorithm".

---

### 3.2.3 Hierarchical RAG

A two-stage retrieval approach: broad search first, then re-rank and select only the top few chunks.

> First skim all books → then read only the best pages

```text
Stage 1: Retrieve many chunks (k=10)
Stage 2: LLM ranks chunks → select top 3-4
```

![Hierarchical RAG flow](Document/media/HierarchialRag.png)

**Advantages:** very relevant results, reduces noise, good for large document sets.
**Disadvantages:** extra LLM call, slower, costlier.

**Prefer when:** large/complex document sets, precision > speed.
**Avoid when:** low latency is essential, small well-organized dataset.

---

### 3.2.4 Graph RAG

Uses relationships between document chunks. If one chunk is relevant, connected chunks (nearby pages/sections) are also retrieved.

```text
Retrieve initial chunks → check metadata (source, page) → fetch neighboring pages → merge
```

![Graph RAG flow](Document/media/Graph%20RAG.png)

**Advantages:** maintains continuity, great for PDFs/books/manuals, reduces missing context.
**Disadvantages:** more complex logic, may add redundant chunks.

**Prefer when:** documents are highly structured (books, manuals, legal texts) and continuity matters.
**Avoid when:** documents are unstructured or you lack reliable metadata.

**Example:** relevant chunk from Page 10 → also retrieve Page 9 and Page 11.

---

### 3.2.5 Memory-Augmented RAG

Uses past conversation history during retrieval and generation for context-aware, human-like responses.

```text
Previous Q&A stored in memory → last N messages added to prompt → retrieval + generation consider history
```

![Memory-Augmented RAG flow](Document/media/Memory-Augmented.png)

**Advantages:** great for follow-up questions, context-aware answers, better UX.
**Disadvantages:** memory growth must be managed, can add noise if not trimmed.

**Prefer when:** building conversational agents/chatbots with multi-turn dialogue.
**Avoid when:** conversations are strictly single-turn or history may hold sensitive data.

**Example:** User asks "What is JWT?" then later "How is it different from OAuth?" → system recalls the earlier JWT answer.

---

### 3.2.6 Agentic RAG

The LLM behaves like an agent — it first decides what action is needed (summarize, compare, calculate, or answer directly), then executes the right retrieval/processing step.

```text
LLM selects a tool → executes tool-specific prompt → generates structured output
```

![Agentic RAG flow](Document/media/Agentic%20RAG.png)

**Advantages:** powerful reasoning, handles complex tasks, flexible logic.
**Disadvantages:** harder to debug, higher cost, prompt-sensitive.

**Prefer when:** multi-step reasoning or dynamic tool selection is needed.
**Avoid when:** questions are simple/factual and low latency/cost is critical.

**Example:** "Compare OAuth and JWT" → selects a COMPARE tool and produces a structured comparison.

---

### 3.2.7 Hybrid RAG

Combines semantic search (embeddings) with keyword-based retrieval, then merges results.

```text
Semantic similarity search + Keyword scoring → merge results → remove duplicates
```

![Hybrid RAG flow](Document/media/Hybrid%20RAG.png)

**Advantages:** handles exact terms, better for technical docs, higher accuracy.
**Disadvantages:** more computation, more logic to manage.

**Prefer when:** queries include exact names/codes/technical terms, jargon-heavy documents.
**Avoid when:** documents are general/non-technical and simplicity is preferred.

**Example:** "MD5 hashing" → keyword search matches "MD5" exactly, semantic search adds general hashing content.

---

### 3.2.8 Modular RAG (Production RAG)

Production-ready approach with multiple fallback strategies — if one retrieval method fails, the system tries another.

```text
Semantic search → if empty → fallback query → if still empty → general documents → safe generation
```

![Modular RAG flow](Document/media/Modular%20RAG.png)

**Advantages:** very robust, never fails silently, production-safe.
**Disadvantages:** more code, harder to maintain.

**Prefer when:** building production-grade, high-uptime applications with unpredictable data sources.
**Avoid when:** rapid prototyping with a small, static, reliable dataset.

**Example:** No relevant chunks for "What is quantum encryption?" → fallback to a broader search and respond: "While specific details aren't available, here's a general overview…"

---

### 3.2.9 Verified RAG (Anti-Hallucination)

Validates the generated answer against retrieved context before returning it, correcting it if needed.

```text
Generate initial answer → verify against context → correct if needed → add verification label
```

![Verified RAG flow](Document/media/Verified%20RAG.png)

**Advantages:** high trust, great for legal/medical/finance, reduces hallucination.
**Disadvantages:** slow, double LLM calls, expensive.

**Prefer when:** accuracy and traceability are paramount (regulated/high-stakes domains).
**Avoid when:** low latency/cost is a priority or the use case is low-risk.

**Example:** "What is the capital of France?" → "Verified — the answer 'Paris' is directly supported by the retrieved documents."

---

## 3.3 Choosing the Right RAG Pattern

| RAG Type | Speed | Cost | Best For |
|---|---|---|---|
| Standard | Fast | Low | Prototypes, clear questions |
| Multi-Query | Medium | Medium | Vague/short questions |
| Hierarchical | Medium | Medium | Large document sets |
| Graph | Medium | Medium | Structured docs (books, manuals) |
| Memory-Augmented | Medium | Medium | Conversational agents |
| Agentic | Slow | High | Complex, multi-step reasoning |
| Hybrid | Medium | Medium | Technical/jargon-heavy content |
| Modular | Medium | Medium-High | Production reliability |
| Verified | Slow | High | High-trust/regulated domains |

---

# 4. Memory Systems

## 4.1 Why Agents Need Memory

Without memory, every request is stateless — the agent forgets everything after each response. Memory allows an agent to behave consistently across a conversation and across sessions.

```text
No Memory
User → LLM → Answer   (forgets everything immediately)

With Memory
User → LLM (+ Memory) → Answer → Memory Updated
```

---

## 4.2 Conversation Memory

Stores the ongoing back-and-forth of the current session so the agent has continuity within a single conversation.

```text
Message 1
Message 2
Message 3
   ↓
Included in next LLM call as context
```

### Example

A user is planning a trip and the conversation continues over multiple messages.

```text
┌──────────────────────────┐
│ User: "I want to visit    │
│ Goa this December."      │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Agent remembers the      │
│ current conversation     │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ User: "What places        │
│ should I visit?"         │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ LLM receives:             │
│ "User is visiting Goa    │
│ in December" +           │
│ "What places..."         │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Agent: "Since you're      │
│ visiting Goa in           │
│ December, you can visit  │
│ Baga, Fort Aguada..."     │
└──────────────────────────┘
```

> **Simple idea:** Conversation Memory = Remembering what was said earlier in the SAME conversation.

---

## 4.3 Short-Term Memory

Holds recent context only — typically the last few turns — and is discarded once the session ends or the window is exceeded.

```text
Last N messages → kept in context window
Older messages → dropped or summarized
```

### Example

A chatbot only keeps the last 3 messages.

```text
                Conversation
                      │
                      ↓
┌────────────────────────────────────┐
│ Message 1: "My name is Rahul."      │
│ Message 2: "I am learning Python."  │
│ Message 3: "I built a chatbot."     │
│ Message 4: "It uses LangGraph."     │
│ Message 5: "How can I improve it?"  │
└───────────────────┬────────────────┘
                    ↓
             Keep Last 3
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
   Message 3              Message 4
   Message 4              Message 5
   Message 5
          │
          ↓
┌──────────────────────────┐
│       LLM Context        │
│                          │
│ "I built a chatbot."     │
│ "It uses LangGraph."     │
│ "How can I improve it?"  │
└────────────┬─────────────┘
             ↓
        LLM generates
          response
```

The older messages are dropped:

```text
Message 1 ──X──> Dropped
Message 2 ──X──> Dropped
```

> **Simple idea:** Short-Term Memory = Remember only recent information. Analogy: short-term memory is like a classroom whiteboard — you can see what was recently written, but old information may be erased when the board gets full.

---

## 4.4 Long-Term Memory

Persists information across sessions (user preferences, past interactions, facts learned) — usually stored externally (database or vector store) and retrieved when relevant.

```text
Session 1 → facts saved
Session 2 → facts retrieved and reused
```

### Example

The user tells the agent something in January, and the agent remembers it in March.

```text
               SESSION 1
                    │
                    ↓
┌──────────────────────────────┐
│ User: "I prefer Python over  │
│ Java for AI projects."       │
└──────────────┬───────────────┘
               ↓
        Extract important
             fact
               ↓
┌──────────────────────────────┐
│ "User prefers Python for     │
│ AI projects."                │
└──────────────┬───────────────┘
               ↓
       ┌─────────────────┐
       │ Long-Term        │
       │ Memory / DB      │
       └────────┬────────┘
                │ Stored
                ↓
          SESSION 2
       Several weeks later
                │
                ↓
┌──────────────────────────────┐
│ User: "Suggest a language    │
│ for my new AI project."      │
└──────────────┬───────────────┘
               ↓
        Retrieve memory
               ↓
┌──────────────────────────────┐
│ "User prefers Python for     │
│ AI projects."                │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ Agent: "I'd recommend         │
│ Python, especially since     │
│ you prefer Python for AI."    │
└──────────────────────────────┘
```

> **Simple idea:** Long-Term Memory = Remember useful information even after the conversation/session ends. Think of it like: Conversation ends → Memory is saved → New conversation starts → Memory is retrieved.

---

## 4.5 Semantic Memory

Stores knowledge as embeddings so it can be retrieved by meaning rather than exact match — effectively a memory store built the same way as a RAG knowledge base.

```text
"User prefers dark mode"
        ↓
   Embedding
        ↓
Stored in Vector Memory
        ↓
Retrieved later when relevant, regardless of exact wording
```

### Example

The user says: "I really enjoy working with Python." The system doesn't just store the exact sentence — it converts the information into an embedding.

```text
┌────────────────────────────────┐
│ User:                          │
│ "I really enjoy working        │
│  with Python."                 │
└───────────────┬────────────────┘
                ↓
          Extract meaning
                ↓
┌────────────────────────────────┐
│ Meaning:                       │
│ "User likes/prefers Python."   │
└───────────────┬────────────────┘
                ↓
          Create Embedding
                ↓
        [0.21, -0.74, 0.35,
         0.82, -0.11, ...]
                ↓
┌────────────────────────────────┐
│       Vector Database          │
│                                │
│ Python preference → Vector     │
│ Java experience   → Vector     │
│ AI interest        → Vector    │
└───────────────┬────────────────┘
                │ Later...
                ↓
┌────────────────────────────────┐
│ User:                          │
│ "Which language should I use   │
│  for my machine learning app?" │
└───────────────┬────────────────┘
                ↓
         Convert query
          into embedding
                ↓
       Similarity Search
                ↓
┌────────────────────────────────┐
│ Most relevant memory:          │
│ "User likes/prefers Python."   │
└───────────────┬────────────────┘
                ↓
┌────────────────────────────────┐
│ LLM + Retrieved Memory         │
└───────────────┬────────────────┘
                ↓
┌────────────────────────────────┐
│ Agent:                         │
│ "Python would be a good choice │
│ for your ML application."      │
└────────────────────────────────┘
```

So instead of the user repeating "I like Python", the system searches memory based on meaning:

```text
"Which language should I use for ML?"
                ↓
        Semantic Search
                ↓
"User likes working with Python"
                ↓
             LLM
                ↓
      "Python is a good choice."
```

> **Simple idea:** Semantic Memory = Remember information by its MEANING, not just exact words.

---

## 4.6 Memory Systems Comparison

| Memory Type | Scope | Storage | What does it remember? | Example |
|---|---|---|---|---|
| Conversation | Current session | In-memory / context | Current conversation | "We were talking about Goa." |
| Short-term | Recent turns | In-memory / context window | Recent messages (last 3–5) | Last 3–5 messages |
| Long-term | Across sessions | Database / vector store | Important facts across sessions | "User prefers Python." |
| Semantic | Meaning-based | Vector store | Facts searchable by meaning | "Which language should I use?" → retrieves Python preference |

---


---

# 6. Frameworks

## 6.1 LangChain Basics

LangChain provides building blocks for LLM applications: prompts, models, messages, output parsers, chains, tools, and agents.

```text
LLM
 ↓
LangChain
 ↓
Prompts → Models → Messages → Output Parsers → Chains → Tools → Agents
```

Core ideas:

- **Prompt Templates** — reusable, parameterized prompts
- **Chains** — sequences of calls (prompt → LLM → parser)
- **Retrievers** — wrap a vector store for RAG
- **Agents** — LLM + tools + decision loop

---

## 6.2 LangGraph Fundamentals

LangGraph models an agent's logic as a **graph of nodes and edges** instead of a single chain, enabling loops, branching, and stateful multi-step workflows.

```text
        ┌──────────┐
        │  START   │
        └────┬─────┘
             ↓
        ┌──────────┐
        │  Node A  │  (e.g., retrieve)
        └────┬─────┘
             ↓
        ┌──────────┐
        │  Node B  │  (e.g., generate)
        └────┬─────┘
             ↓
        Condition?
         /      \
       Yes       No
        │         │
   loop back    END
```

Why LangGraph over a simple chain:

- Supports cycles (retry, re-plan, reflect)
- Explicit state passed between nodes
- Easier to visualize and debug complex agent flows

---

## 6.3 MCP (Model Context Protocol)

MCP is a standardized protocol that lets LLM applications connect to external tools, data sources, and services in a consistent way — instead of writing custom integration code for every tool.

```text
LLM Application
      ↓
MCP Client
      ↓
MCP Server (exposes tools/resources/prompts)
      ↓
External System (files, APIs, databases)
```

Benefits:

- Standard interface between agents and tools
- Reusable servers across different LLM apps
- Clear separation between "the agent" and "the tools it can use"

### What is MCP?

**MCP is a standard way for an AI application to connect and interact with external tools and data.**

Think of MCP as a **bridge between the AI and tools that the AI cannot directly access.**

### Example: Employee Database

Suppose we have an AI assistant and a private company database containing employee information.

The user asks:

> **"Who is working on Project A?"**

The AI cannot directly access the private database. MCP provides a standard connection between the AI and the database.

### Flow

```text
User
  ↓
"Who is working on Project A?"
  ↓
AI / LLM
  ↓
MCP Client
  ↓
MCP Server
  ↓
Employee Database
  ↓
"Rahul and Amit"
  ↓
AI / LLM
  ↓
"Rahul and Amit are working on Project A."
```

### What does MCP do?

- Connects AI applications with external tools and data
- Provides a **standard communication method**
- Allows AI to use tools without building a completely different integration for every tool

### Examples of External Systems

- 🗄️ Databases
- 🌐 APIs
- 📁 File systems
- 📧 Email services
- 🐙 GitHub
- 🔧 Other business tools

### Easy Definition

> **MCP is like a universal connector that allows AI applications to communicate with and use external tools and data.**

### Remember

**AI = Brain 🧠**
**MCP = Connector 🔌**
**External Tool = Capability 🛠️**

---

# 8. Summary & Wrap-up

## 8.1 Key Concepts Summary

At the end of Week 2, participants should remember:

```text
Embeddings
  ↓
Vector Database
  ↓
Similarity Search
  ↓
RAG (Standard → Multi-Query → Hierarchical → Graph → Memory-Augmented
     → Agentic → Hybrid → Modular → Verified)
  ↓
Memory (Conversation / Short-term / Long-term / Semantic)
  ↓

LangChain / LangGraph / MCP
  
```

---







## 8.7 Final Takeaways 

### 8.7.1 Final Takeaways

### 1. Embeddings

Embeddings convert text into vectors that capture meaning, enabling semantic search.

### 2. Vector Databases

Vector databases (FAISS/Chroma) store embeddings and support fast similarity search.

### 3. Similarity Search

Similarity search ranks stored vectors by closeness (cosine similarity, dot product, or Euclidean distance) to a query vector.

### 4. RAG

RAG retrieves relevant chunks and injects them into the prompt so the LLM answers using grounded facts — with variants (Standard, Multi-Query, Hierarchical, Graph, Memory-Augmented, Agentic, Hybrid, Modular, Verified) suited to different needs.

### 5. Memory

Agents need a mix of conversation, short-term, long-term, and semantic memory depending on the use case.

### 6. Tools & Integrations

APIs, web search, databases, and file operations let agents act on and retrieve real-world data.

### 7. LangChain

LangChain provides composable building blocks: prompts, models, parsers, chains, retrievers, memory.

### 8. LangGraph

LangGraph turns agent logic into a stateful graph, enabling loops, branching, and multi-step workflows.

### 9. MCP

MCP is a standard protocol that connects agents to external tools and data sources consistently.



---


---

### 8.7.3 End-of-Session Checklist

Participants should be able to explain:

- [ ] What is an embedding?
- [ ] What is a vector database?
- [ ] FAISS vs Chroma — key differences
- [ ] How does similarity search work?
- [ ] What is RAG and why does it reduce hallucination?
- [ ] What are the 9 RAG variants and when to use each?
- [ ] Conversation memory vs short-term memory
- [ ] Long-term memory vs semantic memory
- [ ] How do agents call APIs and web search tools?
- [ ] How do agents access databases and files safely?
- [ ] What is LangChain and LCEL?
- [ ] What is LangGraph and how does it differ from a linear chain?
- [ ] What is MCP and why does it matter?
- [ ] How to build a PDF Question Answering Agent
- [ ] How to build a RAG Agent with FAISS or Chroma

---

### 8.7.4 One-Sentence Summary

> **Embeddings let us represent meaning, vector databases let us search by meaning, RAG lets an LLM answer using retrieved facts, memory lets an agent remember what matters, and LangChain/LangGraph/MCP give us the frameworks to wire retrieval, memory, and tools into a real, controllable agent.**

---

# Week 2 Complete

**Presenters:** Yashika & Jaswantha
**Date:** 20-Aug-2026

### Next Step

**Week 3 → Agentic Workflows & Multi-Agent Systems**
**Presenters:** Aditya & Sumit

```text
Agent Design Patterns
  ├── ReAct
  ├── Plan-and-Execute
  ├── Reflection
  ├── Self-Correction
  └── Tree of Thoughts
        ↓
Multi-Agent Systems
  ├── Supervisor Agent
  ├── Worker Agents
  ├── Agent Communication
  └── Agent Orchestration
        ↓
Frameworks
  ├── LangGraph Advanced
  ├── CrewAI
  └── AutoGen
        ↓
Evaluation
  ├── Agent Benchmarking
  ├── Prompt Evaluation
  └── Tracing and Debugging
```

