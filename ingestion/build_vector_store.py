'''

Purpose: Convert cleaned chunks into embeddings and store them in ChromaDB.

FUNCTION build_vector_store():

    LOAD chunks from processed JSONL

    CONNECT to ChromaDB persistent collection

    FOR each chunk:

        embedding = CALL Ollama embedding model
                    WITH chunk.text

        UPSERT into ChromaDB:
            id = chunk.chunk_id
            embedding = embedding
            document = chunk.text
            metadata = chunk.metadata

    SAVE index statistics
    RETURN number of indexed chunks


FUNCTION verify_index():

    EXECUTE sample academic search query

    CHECK returned document IDs
    CHECK page metadata
    CHECK retrieved text
    REPORT missing or malformed results



Use stable IDs and upsert, rather than blindly adding every chunk each time. If a document changes, replace its old chunks and remove obsolete ones.

Chroma provides a persistent local client, collection upserts, queries and metadata filtering
'''