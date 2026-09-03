import os
import time
import shutil
from chatbot.ingest import ingest_pdfs_if_needed

def clear_vector_db():
    """Delete the vector database AND folder hash to start fresh"""
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
        print("  Vector DB cleared")

    if os.path.exists("folder_hash.txt"):
        os.remove("folder_hash.txt")
        print("  Folder hash cleared\n")

def test_initial_ingestion():
    """Test:  First time ingesting all PDFs"""

    print("TEST 1: Initial Ingestion (All PDFs)")

    
    clear_vector_db()
    
    start_time = time.time()
    ingest_pdfs_if_needed()
    end_time = time.time()
    
    elapsed = end_time - start_time
    print(f"\n Time taken: {elapsed:.2f} seconds")
    return elapsed

def test_no_changes():
    """Test: Running again with no changes (should be instant)"""
   
    print("TEST 2: Re-run with NO Changes (Should Skip)")

    
    start_time = time.time()
    ingest_pdfs_if_needed()
    end_time = time.time()
    
    elapsed = end_time - start_time
    print(f"\n  Time taken: {elapsed:.2f} seconds")
    print(f"=" * 60)
    return elapsed

def test_add_new_pdf():
    """Test: Add a new PDF (should only process new one)"""
  
    print("TEST 3: Add New PDF (Only Process New File)")

    
    # Create a dummy PDF (just copy an existing one)
    pdf_files = [f for f in os.listdir("pdfs") if f.endswith(".pdf")]
    if pdf_files:
        source = os.path.join("pdfs", pdf_files[0])
        destination = os.path.join("pdfs", "test_new_file.pdf")
        
        # Only copy if it doesn't exist
        if not os.path.exists(destination):
            shutil.copy(source, destination)
            print(f" Added new PDF: test_new_file.pdf")
    
    start_time = time.time()
    ingest_pdfs_if_needed()
    end_time = time.time()
    
    elapsed = end_time - start_time
    print(f"\n Time taken: {elapsed:.2f} seconds")

    
    # Clean up
    test_file = os.path.join("pdfs", "test_new_file.pdf")
    if os.path.exists(test_file):
        os.remove(test_file)
        print(" Cleaned up test file")
    
    return elapsed

def test_modify_pdf():
    """Test: Simulate modifying a PDF"""
   
    print("TEST 4: Modify Existing PDF (Re-process One File)")

    
    pdf_files = [f for f in os.listdir("pdfs") if f.endswith(".pdf")]
    if pdf_files: 
        # "Modify" by appending a byte (simulates content change)
        target_file = os.path.join("pdfs", pdf_files[0])
        
        # Read original content
        with open(target_file, 'rb') as f:
            original_content = f.read()
        
        # Append a space (modifies the file)
        with open(target_file, 'ab') as f:
            f.write(b' ')
        
        print(f" Modified:  {pdf_files[0]}")
        
        start_time = time.time()
        ingest_pdfs_if_needed()
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"\n Time taken: {elapsed:.2f} seconds")
        
        # Restore original file
        with open(target_file, 'wb') as f:
            f.write(original_content)
        print("Restored original file")
        
   
    return elapsed

def run_all_tests():
    """Run complete test suite"""
    
    print("PERFORMANCE TEST:  Incremental Ingestion")
    
    
    # Count PDFs
    pdf_count = len([f for f in os.listdir("pdfs") if f.endswith(".pdf")])
   
    
    results = {}
    
    # Run tests
    results['initial'] = test_initial_ingestion()
    results['no_changes'] = test_no_changes()
    results['add_new'] = test_add_new_pdf()
    results['modify'] = test_modify_pdf()
    
    # Summary
   
    print("RESULTS SUMMARY")

    print(f"1. Initial Ingestion (all PDFs):     {results['initial']:.2f}s")
    print(f"2. No Changes (skip all):            {results['no_changes']:.2f}s")
    print(f"3. Add New PDF (process 1 only):     {results['add_new']:.2f}s")
    print(f"4. Modify PDF (re-process 1 only):   {results['modify']:.2f}s")
    
    print("\n Performance Improvement:")
    if results['initial'] > 0:
        speedup = results['initial'] / results['no_changes']
        print(f"   No changes: {speedup:.1f}x FASTER than re-ingesting everything!")
        
        if results['add_new'] < results['initial']:
            speedup2 = results['initial'] / results['add_new']
            print(f"   Add new PDF: {speedup2:.1f}x FASTER than re-ingesting everything!")
    
 

if __name__ == "__main__":
    # Check if PDFs exist
    if not os.path.exists("pdfs") or not os.listdir("pdfs"):
        print(" Error: No PDFs found in 'pdfs/' folder")
        print("   Please add some PDF files before running tests")
    else:
        run_all_tests()

        