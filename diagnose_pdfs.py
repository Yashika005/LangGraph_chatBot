# diagnose_pdfs.py
import os
from langchain_community.document_loaders import PyPDFLoader

def check_pdf_content():
    pdf_dir = "pdfs"
    
    if not os.path.exists(pdf_dir):
        print(f" Directory '{pdf_dir}' not found!")
        return
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f" No PDF files found in '{pdf_dir}'")
        return
    
    print(f"Found {len(pdf_files)} PDF(s):")
    
    for pdf_file in pdf_files:
        file_path = os.path.join(pdf_dir, pdf_file)
        print(f"\n{'='*60}")
        print(f" Analyzing: {pdf_file}")
        print(f"{'='*60}")
        
        try:
            # Load PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            print(f" Pages: {len(documents)}")
            
            # Check first few pages for text content
            for i, doc in enumerate(documents[:3]):  # Check first 3 pages
                content = doc.page_content.strip()
                print(f"\n Page {i+1}:")
                print(f"   Length: {len(content)} characters")
                print(f"   Preview: {content[:200]}..." if content else "     NO TEXT CONTENT!")
                
                # Check if it's mostly empty (scanned PDF)
                if len(content) < 50:
                    print("     WARNING: Very little text - may be scanned/image PDF")
            
            # Check total text
            total_text = sum(len(doc.page_content.strip()) for doc in documents)
            print(f"\n Total text in PDF: {total_text:,} characters")
            
            if total_text < 100:
                print(" WARNING: PDF appears to have very little text!")
                print("   This could be a scanned/image PDF. Consider using OCR.")
            
        except Exception as e:
            print(f" Error loading {pdf_file}: {e}")

if __name__ == "__main__":
    check_pdf_content()