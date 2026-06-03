# Debug Mode Rules

## Silent Failure Points
- **Chunk prefix mismatch**: First chunk of lesson uses `"Lesson X content:"` but subsequent chunks use `"Course Y Lesson X content:"` ([`backend/document_processor.py:186`](../backend/document_processor.py:186) vs `:234`) — queries may miss results due to inconsistent prefix pattern
- **ChromaDB empty results**: If no courses loaded, search returns empty results silently — check `backend/chroma_db/` exists and has data
- **Missing `.env`**: No `ANTHROPIC_API_KEY` set causes Claude API call to fail with generic error

## Data Verification
- Check ChromaDB collections via `course_catalog.get()` and `course_content.get()`
- Course titles used as document IDs — duplicate titles silently ignored (existing courses skipped in [`rag_system.py:87`](../backend/rag_system.py:87))
- Session state is in-memory only (`Dict[str, List[Message]]`) — lost on server restart

## Environment
- Default port: 8000 (make sure not already in use)
- Python 3.13+ required (`.python-version` at project root)
