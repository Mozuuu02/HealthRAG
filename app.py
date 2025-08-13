import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import networkx as nx
import PyPDF2
import time
import nltk
from nltk.tokenize import sent_tokenize

# ===== PDF helper functions =====
def extract_text_from_pdf(pdf_path):
    import re
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
    return text

def chunk_text(text, sentences_per_chunk=5):
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunks.append(" ".join(sentences[i:i + sentences_per_chunk]))
    return chunks

# ===== Streamlit UI =====
st.title("HealthRAG")

pdf_path = "disease-handbook-complete.pdf"  # replace with your PDF filename

# Show progress while processing
progress_text = "Processing..."
my_bar = st.progress(0, text=progress_text)

# Step 1: Extract PDF text
time.sleep(0.1)
pdf_text = extract_text_from_pdf(pdf_path)
my_bar.progress(20, text=progress_text)

# Step 2: Chunk the text
pdf_chunks = chunk_text(pdf_text)
my_bar.progress(50, text=progress_text)

# Step 3: Setup vector search and store embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.get_or_create_collection(name="knowledge_base")
collection.delete(where={"_all_": True})

all_docs = pdf_chunks
embeddings = model.encode(all_docs).tolist()
for i, doc in enumerate(all_docs):
    collection.add(documents=[doc], embeddings=[embeddings[i]], ids=[str(i)])
my_bar.progress(90, text=progress_text)

time.sleep(0.2)  # small delay for UX
my_bar.progress(100, text="Done!")
time.sleep(0.5)
my_bar.empty()

# ===== Setup Knowledge Graph =====
G = nx.Graph()
G.add_edge("Diabetes", "Insulin", relation="treated_by")
G.add_edge("Diabetes", "High Blood Sugar", relation="symptom")
G.add_edge("High Blood Sugar", "Organ Damage", relation="causes")

# ===== Hybrid Retrieval Function =====
def hybrid_search(query):
    q_emb = model.encode([query]).tolist()[0]
    vector_results = collection.query(query_embeddings=[q_emb], n_results=3)
    vector_docs = vector_results["documents"][0]

    related_entities = []
    for node in G.nodes:
        if query.lower() in node.lower():
            related_entities.extend(list(G.neighbors(node)))

    return vector_docs + related_entities

# ===== Query UI =====
query = st.text_input("Ask a question:")
if query:
    results = hybrid_search(query)
    st.write("### Results:")
    for r in results:
        st.write("-", r)
