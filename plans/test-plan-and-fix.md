# RAG Chatbot Debugging Test Plan

## 1. Problem Analysis

User reports RAG chatbot returns "query failed" for any content-related questions.

### 1.1 Key Code Trace

Request pipeline:
```
POST /api/query → rag_system.query() → ai_generator.generate_response() 
→ LLM tool call → ToolManager.execute_tool("search_course_content", ...)
→ CourseSearchTool.execute(query, ...) → VectorStore.search(query, ...)
→ chroma_db.query(n_results=search_limit, ...)
```

### 1.2 Found Critical Bug

**Bug 1 (Root Cause): `MAX_RESULTS = 0`** — config.py line 23

ChromaDB’s `n_results` parameter must be positive. Passing 0 causes ChromaDB to throw an exception.

**Bug 2 (Potential): Inconsistent chunk prefix** — document_processor.py:186 vs :234

First chunk uses "Lesson X content:" but subsequent chunks use "Course Y Lesson X content:".

## 2. Test Architecture

### Test Files
```
backend/
 └── tests/
     ├── __init__.py
     ├── conftest.py              # pytest fixtures
     ├── test_search_tools.py     # Test 1: CourseSearchTool.execute()
     ├── test_ai_generator.py     # Test 2: AIGenerator tool calling
     └── test_rag_integration.py  # Test 3: End-to-end RAG flow
```

## 3. Fix

### Fix 1 (Critical): Fix `MAX_RESULTS` default — config.py:23
```python
# Before
MAX_RESULTS: int = 0

# After
MAX_RESULTS: int = 5
```

### Fix 2 (Recommended): Chunk prefix consistency

First chunk prefix changed from "Lesson X content:" to "Course Y Lesson X content:"

## 4. Execution Plan

```mermaid
flowchart TD
    A[Create backend/tests/ dir] ────> B[Create conftest.py fixtures]
    B ────> C[Create test_search_tools.py]
    B ────> D[Create test_ai_generator.py]
    B ────> E[Create test_rag_integration.py]
    C & D & E ────> F[Install pytest deps]
    F ────> G[Run all tests]
    G ────> H{Test 3.7 fails?}
    H ────>|Yes| I[Confirm MAX_RESULTS=0 root cause]
    H ────>|No| J[Further diagnosis]
    I ────> K[Fix config.py MAX_RESULTS=5]
    K ────> L[Rerun tests to verify]
    J ────> L
    L ────> M[Output test report]
```
