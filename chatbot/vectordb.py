import chromadb
from chatbot.embed import EmbeddingFunction

PERSIST_DIR = "chroma_db"

class VectorDB: 
    def __init__(self):
        self.client = chromadb.PersistentClient(path=PERSIST_DIR)
        self.embedding_fn = EmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="pdfs",
            embedding_function=self.embedding_fn
        )

    def similarity_search(self, query, k=4):
        query_embedding = self.embedding_fn([query])
        
        initial_k = min(k * 3, 20)  # Get more results initially
        results_data = self.collection.query(
            query_embeddings=query_embedding,
            n_results=initial_k,
            include=["documents", "metadatas"]
        )
        
        results = []
        selected_sources = set()
        
        if results_data and results_data['documents']:
            for i, content in enumerate(results_data['documents'][0]):
                metadata = results_data['metadatas'][0][i] if results_data['metadatas'] and i < len(results_data['metadatas'][0]) else {}
                source_file = metadata.get("source_file", "unknown")
                
                
                if source_file not in selected_sources or len(selected_sources) < k:
                    doc = type("Doc", (), {
                        "page_content": content,
                        "metadata": metadata
                    })()
                    results.append(doc)
                    selected_sources.add(source_file)
                
                if len(results) >= k:
                    break
        
            if len(results) < k:
                for i, content in enumerate(results_data['documents'][0]):
                    if len(results) >= k:
                        break
                    metadata = results_data['metadatas'][0][i] if results_data['metadatas'] and i < len(results_data['metadatas'][0]) else {}
                    

                    already_included = False
                    for existing_doc in results:
                        if existing_doc.metadata == metadata and existing_doc.page_content == content:
                            already_included = True
                            break
                    
                    if not already_included:
                        doc = type("Doc", (), {
                            "page_content": content,
                            "metadata": metadata
                        })()
                        results.append(doc)
        
        return results

    def get(self):
        return self.collection.get()

    def add_documents(self, chunks):
        docs = [c.page_content for c in chunks]
        
        ids = []
        source_counts = {}
        for chunk in chunks:
            source = chunk.metadata.get("source_file", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
            ids.append(f"{source}_{source_counts[source]}")
        
        metadatas = [c.metadata for c in chunks]  
        
        batch_size = 100
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]  
            self.collection.add(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_metadatas 
            )
        return True
    
    def delete_by_ids(self, ids):
        """Delete documents by their IDs"""
        if ids:
            self.collection.delete(ids=ids)
    
    def clear(self):
        """Clear all documents from the collection"""
        self.client.delete_collection(name="pdfs")
        self.collection = self.client.get_or_create_collection(
            name="pdfs",
            embedding_function=self.embedding_fn
        )
    
    
    def get_stats(self):
        """Get statistics about the stored documents"""
        data = self.collection.get()
        stats = {
            "total_chunks": len(data["ids"]) if data.get("ids") else 0,
            "sources": {}
        }
        
        if data.get("metadatas"):
            for metadata in data["metadatas"]:
                if metadata and "source_file" in metadata:
                    source = metadata["source_file"]
                    stats["sources"][source] = stats["sources"].get(source, 0) + 1
        
        return stats

def get_vectordb():
    return VectorDB()