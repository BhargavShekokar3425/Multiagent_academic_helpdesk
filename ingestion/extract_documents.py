'''
Purpose: Read PDFs and extract text while preserving page and source information.

Inputs: data/document_manifest.csv, local PDF files.

Output: Extracted text records in data/processed/

FUNCTION extract_documents():

    manifest = READ document_manifest.csv

    FOR each document IN manifest:

        IF document is not verified:
            SKIP

        IF local PDF does not exist:
            LOG missing document
            CONTINUE

        OPEN PDF using PyMuPDF

        FOR each page:
            EXTRACT text
            EXTRACT page number

            IF text is empty or suspicious:
                FLAG as possible scanned page

            CREATE page_record:
                document_id
                page_number
                text
                document_metadata

            APPEND page_record

        SAVE page records to JSON

    RETURN extraction summary
'''