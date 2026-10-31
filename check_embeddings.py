import sqlite3
import numpy as np

conn = sqlite3.connect("minigpt.db")
cursor = conn.cursor()

cursor.execute("""
    SELECT content, embedding
    FROM document_chunks
    LIMIT 1
""")

row = cursor.fetchone()

if row:
    content, embedding = row

    vector = np.frombuffer(embedding, dtype=np.float32)

    print("Document content:", content)
    print("Vector dimensions:", len(vector))
    print("Vector shape:", vector.shape)
    print("All values finite:", np.isfinite(vector).all())
    print("Vector norm:", np.linalg.norm(vector))
    print("First 10 values:", vector[:10])

    if len(vector) == 384 and np.isfinite(vector).all():
        print("\nSUCCESS: Valid semantic embedding!")
    else:
        print("\nERROR: Embedding validation failed.")

else:
    print("No embedding records found.")

conn.close()