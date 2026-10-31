import sqlite3

conn = sqlite3.connect("minigpt.db")
cursor = conn.cursor()

cursor.execute("""
SELECT
    d.filename,
    COUNT(dc.id) AS total_chunks
FROM documents d
JOIN document_chunks dc
    ON d.id = dc.document_id
GROUP BY d.id, d.filename;
""")

rows = cursor.fetchall()

print("\nStored documents and embeddings:\n")

for row in rows:
    print(f"Filename: {row[0]}")
    print(f"Total chunks: {row[1]}")
    print("-" * 40)

conn.close()