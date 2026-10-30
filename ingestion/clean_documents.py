'''
Purpose: Remove extraction noise without losing academic rules.

For example, a PDF may repeat the institute name, page number and publication footer on every page.

FUNCTION clean_documents():

    LOAD extracted JSON documents

    FOR each document:
        FOR each page:

            NORMALIZE whitespace
            REMOVE repeated page headers where safe
            REMOVE duplicate blank lines
            FIX broken words caused by PDF extraction
            PRESERVE headings and section numbers
            PRESERVE lists, percentages and exceptions
            PRESERVE table boundaries

            IF text appears malformed:
                FLAG page for manual review

        SAVE cleaned document

    RETURN cleaning statistics
'''