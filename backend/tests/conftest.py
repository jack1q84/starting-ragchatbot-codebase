"""Shared fixtures and mock data for all test suites"""

from unittest.mock import MagicMock, Mock, patch
import pytest
import sys
import os

# Ensure backend directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


# ── Mock SearchResults ──────────────────────────────────────────────

@pytest.fixture
def mock_search_results_valid():
    """Return valid SearchResults with sample course content"""
    return SearchResults(
        documents=[
            "Computer use allows models to look at a screen and generate mouse clicks or keystrokes.",
            "The API supports multi-modal requests where you can analyze images.",
        ],
        metadata=[
            {
                "course_title": "Building Towards Computer Use with Anthropic",
                "lesson_number": 0,
                "chunk_index": 0,
            },
            {
                "course_title": "Building Towards Computer Use with Anthropic",
                "lesson_number": 1,
                "chunk_index": 1,
            },
        ],
        distances=[0.15, 0.25],
    )


@pytest.fixture
def mock_search_results_empty():
    """Return empty SearchResults (no documents found)"""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def mock_search_results_error():
    """Return SearchResults with an error"""
    return SearchResults.empty("Search error: something went wrong")


# ── Mock VectorStore ────────────────────────────────────────────────

@pytest.fixture
def mock_vector_store(mock_search_results_valid):
    """Create a mock VectorStore with controlled search behavior"""
    store = MagicMock()
    store.search.return_value = mock_search_results_valid
    store._resolve_course_name.return_value = "Building Towards Computer Use with Anthropic"
    store.get_lesson_link.return_value = "https://example.com/lesson/0"
    store.get_course_link.return_value = "https://example.com/course"
    store.max_results = 5
    return store


@pytest.fixture
def mock_vector_store_no_results(mock_search_results_empty):
    """VectorStore that returns empty results"""
    store = MagicMock()
    store.search.return_value = mock_search_results_empty
    store._resolve_course_name.return_value = None
    return store


@pytest.fixture
def mock_vector_store_error(mock_search_results_error):
    """VectorStore that returns an error"""
    store = MagicMock()
    store.search.return_value = mock_search_results_error
    return store


@pytest.fixture
def mock_vector_store_max_results_zero():
    """VectorStore with max_results = 0 — simulates the known bug"""
    store = MagicMock()
    store.max_results = 0
    # Simulate the real search method to show the bug
    def search_side_effect(query, course_name=None, lesson_number=None, limit=None):
        search_limit = limit if limit is not None else 0
        if search_limit <= 0:
            return SearchResults.empty("Search error: n_results must be > 0")
        return SearchResults(
            documents=["test"],
            metadata=[{"course_title": "Test", "lesson_number": 1, "chunk_index": 0}],
            distances=[0.1],
        )
    store.search.side_effect = search_side_effect
    store._resolve_course_name.return_value = "Test Course"
    return store


# ── CourseSearchTool fixtures ───────────────────────────────────────

@pytest.fixture
def search_tool(mock_vector_store):
    """Create a CourseSearchTool with a mock vector store"""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def search_tool_no_results(mock_vector_store_no_results):
    """CourseSearchTool that returns no results"""
    return CourseSearchTool(mock_vector_store_no_results)


@pytest.fixture
def search_tool_error(mock_vector_store_error):
    """CourseSearchTool that simulates a search error"""
    return CourseSearchTool(mock_vector_store_error)


@pytest.fixture
def search_tool_max_results_zero(mock_vector_store_max_results_zero):
    """CourseSearchTool with MAX_RESULTS=0 — bug scenario"""
    return CourseSearchTool(mock_vector_store_max_results_zero)


# ── ToolManager fixtures ────────────────────────────────────────────

@pytest.fixture
def tool_manager(search_tool):
    """Create a ToolManager with CourseSearchTool registered"""
    mgr = ToolManager()
    mgr.register_tool(search_tool)
    return mgr


# ── Mock OpenAI / AIGenerator fixtures ──────────────────────────────

@pytest.fixture
def mock_openai_chat_completion():
    """Create a mock OpenAI chat.completions.create response (no tool calls)"""
    mock_message = MagicMock()
    mock_message.content = "Computer use allows AI models to interact with screens."
    mock_message.tool_calls = None
    def _model_dump():
        return {"role": "assistant", "content": mock_message.content, "tool_calls": None}
    mock_message.model_dump = _model_dump

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.fixture
def mock_openai_tool_call_response():
    """Create a mock OpenAI response with a tool call request (with model_dump support)"""
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_12345"
    mock_tool_call.function.name = "search_course_content"
    mock_tool_call.function.arguments = '{"query": "computer use", "course_name": "Building Towards Computer Use"}'

    mock_message = MagicMock()
    mock_message.content = None
    mock_message.tool_calls = [mock_tool_call]
    def _model_dump():
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                }
                for tc in mock_message.tool_calls
            ]
        }
    mock_message.model_dump = _model_dump

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "tool_calls"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.fixture
def mock_openai_final_response():
    """Mock for the final API call after tool execution (with model_dump support)"""
    mock_message = MagicMock()
    mock_message.content = "Based on the course content, computer use lets AI models interact with computer screens by generating mouse clicks and keystrokes."
    mock_message.tool_calls = None
    def _model_dump():
        return {"role": "assistant", "content": mock_message.content, "tool_calls": None}
    mock_message.model_dump = _model_dump

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


# ── Sequential tool calling fixtures ────────────────────────────────

@pytest.fixture
def outline_tool(mock_vector_store):
    """Create a CourseOutlineTool with a mock vector store"""
    from search_tools import CourseOutlineTool
    return CourseOutlineTool(mock_vector_store)


@pytest.fixture
def tool_manager_with_both_tools(search_tool, outline_tool):
    """ToolManager with both CourseSearchTool and CourseOutlineTool registered"""
    mgr = ToolManager()
    mgr.register_tool(search_tool)
    mgr.register_tool(outline_tool)
    return mgr


@pytest.fixture
def mock_openai_tool_call_response_v2():
    """
    Second tool_calls response (different function name and arguments).
    Used to simulate Round 2 where Claude calls a different tool.
    """
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_67890"
    mock_tool_call.function.name = "search_course_content"
    mock_tool_call.function.arguments = '{"query": "Prompt Compression", "course_name": "Course Y"}'

    mock_message = MagicMock()
    mock_message.content = None
    mock_message.tool_calls = [mock_tool_call]
    def _model_dump():
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                }
                for tc in mock_message.tool_calls
            ]
        }
    mock_message.model_dump = _model_dump

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "tool_calls"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.fixture
def mock_two_round_responses(mock_openai_tool_call_response, mock_openai_tool_call_response_v2, mock_openai_final_response):
    """
    Three sequential responses simulating:
    Round 1: tool_calls (get_course_outline)
    Round 2: tool_calls (search_course_content)
    Final:   stop (final answer)
    """
    round1 = mock_openai_tool_call_response
    round2 = mock_openai_tool_call_response_v2

    # Override Round 1 to use get_course_outline
    mock_tool_call_outline = MagicMock()
    mock_tool_call_outline.id = "call_outline_001"
    mock_tool_call_outline.function.name = "get_course_outline"
    mock_tool_call_outline.function.arguments = '{"course_title": "Course X"}'
    
    round1.choices[0].message.tool_calls = [mock_tool_call_outline]
    round1.choices[0].message.content = None
    # Update model_dump for the outline call
    def _model_dump_outline():
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_outline_001", "function": {"name": "get_course_outline", "arguments": '{"course_title": "Course X"}'}}
            ]
        }
    round1.choices[0].message.model_dump = _model_dump_outline

    return [round1, round2, mock_openai_final_response]


@pytest.fixture
def mock_second_round_no_tool(mock_openai_tool_call_response, mock_openai_final_response):
    """
    Round 1 calls a tool, Round 2 stops immediately (no more tool calls).
    Used to test early termination.
    """
    return [mock_openai_tool_call_response, mock_openai_final_response]


# ── Config fixture ──────────────────────────────────────────────────

@pytest.fixture
def test_config():
    """Create a test configuration with mocked API values"""
    from config import Config
    cfg = Config()
    cfg.LLM_API_KEY = "test-key"
    cfg.LLM_BASE_URL = "https://api.deepseek.com"
    cfg.LLM_MODEL = "deepseek-chat"
    cfg.CHROMA_PATH = "./test_chroma_db"
    cfg.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    cfg.MAX_RESULTS = 5
    cfg.CHUNK_SIZE = 800
    cfg.CHUNK_OVERLAP = 100
    cfg.MAX_HISTORY = 2
    return cfg


@pytest.fixture
def test_config_max_results_zero(test_config):
    """Config with MAX_RESULTS=0 — bug scenario"""
    test_config.MAX_RESULTS = 0
    return test_config
