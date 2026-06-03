# Architect Mode Rules

## Architectural Constraints
- **Stateless sessions**: `SessionManager` stores conversations in-memory (`Dict`) — no persistence across restarts
- **ChromaDB as single source of truth**: Both course metadata and content live in ChromaDB; no SQL/cache layer
- **Claude drives search logic**: The AI model decides when to call `search_course_content` — the system doesn't pre-fetch or hybrid search
- **Tool calling is single-turn**: Only one round of tool execution supported ([`backend/ai_generator.py:83-84`](../backend/ai_generator.py:83-84)) — no multi-step tool chains

## Hidden Coupling
- `tools` parameter passed to Claude uses `tool_choice: "auto"` ([`backend/ai_generator.py:77`](../backend/ai_generator.py:77))
- Course title is both the display name AND the ChromaDB document ID — changing a title would orphan existing data
- Lesson metadata serialized as JSON string in a single metadata field (ChromaDB limitation for nested data)

## Performance Notes
- `sentence-transformers` model (`all-MiniLM-L6-v2`) loads on startup — first request is slow
- Documents loaded synchronously at startup in `@app.on_event("startup")` ([`backend/app.py:88`](../backend/app.py:88))
- ChromaDB uses `PersistentClient` — writes are synchronous, no batching
