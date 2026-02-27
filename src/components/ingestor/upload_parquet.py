import os
import json
import time
import argparse
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def run_upload():
    # --- 1. Argument Parsing for Generality ---
    parser = argparse.ArgumentParser(description="Upload Parquet data to Qdrant")
    parser.add_argument("--file", type=str, default="data/data.parquet", 
                        help="Path to the parquet file (relative to root)")
    parser.add_argument("--collection", type=str, default="my_collection", 
                        help="Name of the Qdrant collection")
    parser.add_argument("--vector_size", type=int, default=1024, 
                        help="Size of the embedding vectors (e.g., 1024 for BGE-large)")
    
    args = parser.parse_args()

    # --- 2. Dynamic Connection Logic ---
    # When running manually on host, use localhost. Inside Docker, use qdrant.
    qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    
    print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}...")
    client = QdrantClient(host=qdrant_host, port=qdrant_port, prefer_grpc=False)

    # --- 3. Load Data ---
    if not os.path.exists(args.file):
        print(f"❌ Error: File not found at {args.file}")
        return

    print(f"Reading {args.file}...")
    df = pd.read_parquet(args.file)

    # --- 4. Collection Setup ---
    if not client.collection_exists(args.collection):
        print(f"Creating collection: {args.collection}...")
        client.recreate_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=args.vector_size, distance=Distance.COSINE),
        )
    else:
        print(f"Collection {args.collection} already exists. Upserting new data...")

    # --- 5. Data Transformation ---
    print("Preparing points...")
    points = []
    for i in range(len(df)):
        try:
            payload_data = df.payload.iloc[i]
            
            # Robust metadata handling
            metadata = payload_data.get('metadata', {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            points.append(
                PointStruct(
                    id=i,
                    vector=df.vector.iloc[i].tolist(),
                    payload={
                        "text": payload_data.get('text', ""),
                        "metadata": metadata
                    }
                )
            )
        except Exception as e:
            print(f"⚠️ Skipping row {i} due to error: {e}")

    # --- 6. Batch Upload ---
    BATCH_SIZE = 100
    print(f"Starting upload of {len(points)} points to '{args.collection}'...")
    
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i : i + BATCH_SIZE]
        client.upsert(
            collection_name=args.collection,
            points=batch
        )
        if i % 500 == 0:
            print(f"  ...pushed {i} points")
        time.sleep(0.05)

    print(f"✅ Successfully pushed {len(points)} points to Qdrant collection '{args.collection}'!")

if __name__ == "__main__":
    run_upload()