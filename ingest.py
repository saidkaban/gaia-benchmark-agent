# ingest.py
import os
import ast
import pandas as pd
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client

# 1) Load env
load_dotenv()
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

# 2) Init embeddings + vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents_langchain",
)

# 3) Load your CSV
df = pd.read_csv("supabase_docs.csv")

# 4) Build Documents
docs = []
for _, row in df.iterrows():
    content = row["content"]
    # parse the metadata string into a dict
    metadata = ast.literal_eval(row["metadata"])
    docs.append(Document(page_content=content, metadata=metadata))

# 5) Upsert them (will compute embeddings)
vector_store.add_documents(docs)
print(f"✅ Inserted {len(docs)} documents into Supabase.")
