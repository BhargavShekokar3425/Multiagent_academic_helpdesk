'''
Purpose: Split long documents into smaller sections suitable for retrieval.

FUNCTION create_chunks(document):

    IDENTIFY major headings
    IDENTIFY section boundaries

    FOR each section:

        IF section is short:
            CREATE one chunk

        ELSE:
            SPLIT section into smaller chunks
            USE configured chunk size and overlap

        FOR each chunk:

            ASSIGN stable chunk_id

            ATTACH:
                document_id
                document_title
                source_url
                page_number
                section_heading
                effective_date
                academic_program

            APPEND chunk

    RETURN chunks


FUNCTION process_all_documents():

    LOAD cleaned documents

    FOR each document:
        chunks = create_chunks(document)

    SAVE all chunks as JSONL
'''