# test_retrieval.py
from chatbot.vectordb import get_vectordb

vectordb = get_vectordb()

# Test query
query = "AWS services"
print(f"Query: {query}")

# Check stats
stats = vectordb.get_stats()
print(f"\nDatabase Stats:")
print(f"Total chunks: {stats['total_chunks']}")
for doc, count in stats['sources'].items():
    print(f"  {doc}: {count} chunks")

# Test retrieval
docs = vectordb.similarity_search(query, k=4)
print(f"\nRetrieved {len(docs)} chunks:")
for i, doc in enumerate(docs):
    source = doc.metadata.get('source_file', 'unknown')
    page = doc.metadata.get('page', 'N/A')
    print(f"{i+1}. {source} (page {page}): {doc.page_content[:100]}...")