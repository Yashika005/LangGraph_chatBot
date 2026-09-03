# test_chunks.py - Works with your CURRENT ingest.py
import os
import sys

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot.vectordb import get_vectordb

def analyze_current_chunking():
    """Analyze the chunking quality in your current database"""
    
    print("=== Analyzing Current Chunking Quality ===")
    
    vectordb = get_vectordb()
    
    # Get all data from the database
    data = vectordb.get()
    
    if not data or not data.get("documents"):
        print("❌ No data found in the database.")
        print("Please run the main application first to ingest PDFs.")
        return
    
    documents = data["documents"]
    metadatas = data.get("metadatas", [])
    
    print(f"✅ Found {len(documents)} chunks in database")
    
    # Analyze chunk sizes
    chunk_sizes = []
    for doc in documents:
        chunk_sizes.append(len(doc))
    
    if chunk_sizes:
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        max_size = max(chunk_sizes)
        min_size = min(chunk_sizes)
        
        print(f"\n📊 Chunk Size Analysis:")
        print(f"  Average: {avg_size:.0f} characters")
        print(f"  Minimum: {min_size} characters")
        print(f"  Maximum: {max_size} characters")
        
        # Distribution
        size_ranges = {
            "Tiny (0-250)": 0,
            "Small (251-500)": 0,
            "Medium (501-1000)": 0,
            "Large (1001-1500)": 0,
            "Very Large (1500+)": 0
        }
        
        for size in chunk_sizes:
            if size <= 250:
                size_ranges["Tiny (0-250)"] += 1
            elif size <= 500:
                size_ranges["Small (251-500)"] += 1
            elif size <= 1000:
                size_ranges["Medium (501-1000)"] += 1
            elif size <= 1500:
                size_ranges["Large (1001-1500)"] += 1
            else:
                size_ranges["Very Large (1500+)"] += 1
        
        print(f"\n📈 Size Distribution:")
        for range_name, count in size_ranges.items():
            percentage = (count / len(chunk_sizes)) * 100
            print(f"  {range_name}: {count} chunks ({percentage:.1f}%)")
        
        # Document statistics
        print(f"\n📄 Document Statistics:")
        if metadatas:
            source_counts = {}
            for metadata in metadatas:
                if metadata and "source_file" in metadata:
                    source = metadata["source_file"]
                    source_counts[source] = source_counts.get(source, 0) + 1
            
            for source, count in source_counts.items():
                avg_for_source = sum([len(d) for d, m in zip(documents, metadatas) 
                                     if m and m.get("source_file") == source]) / count
                print(f"  {source}: {count} chunks (avg: {avg_for_source:.0f} chars)")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if avg_size > 1200:
            print("  ⚠️  Average chunk size > 1200 chars - Consider reducing chunk_size to 800-1000")
        elif avg_size < 400:
            print("  ⚠️  Average chunk size < 400 chars - Consider increasing chunk_size to 800-1000")
        else:
            print("  ✅ Current chunk size (1000 chars) looks appropriate")
        
        if max_size > 2000:
            print("  ⚠️  Some chunks > 2000 chars - Recursive splitter may be cutting incorrectly")
        
        # Check overlap
        print(f"\n🔄 Overlap Analysis:")
        print(f"  Current overlap: 200 characters (20% of 1000)")
        if avg_size < 1000:
            actual_overlap_percentage = (200 / avg_size) * 100 if avg_size > 0 else 0
            print(f"  Actual overlap relative to avg chunk: {actual_overlap_percentage:.1f}%")
    
    # Show sample chunks
    print(f"\n🔍 Sample Chunks (first 3):")
    for i in range(min(3, len(documents))):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Size: {len(documents[i])} characters")
        if i < len(metadatas) and metadatas[i]:
            print(f"Source: {metadatas[i].get('source_file', 'unknown')}")
            print(f"Page: {metadatas[i].get('page', 'unknown')}")
        print("Content preview:")
        print(documents[i][:200] + "..." if len(documents[i]) > 200 else documents[i])

if __name__ == "__main__":
    analyze_current_chunking()