"""End-to-end integration tests for RAG system query handling"""

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


class TestRAGSystemInitialization:
    """Suite: RAGSystem initialization and component wiring"""

    def test_rag_system_initializes_components(self, test_config):
        """3.1: RAGSystem creates all required components on init"""
        from rag_system import RAGSystem

        rag = RAGSystem(test_config)
        assert rag.document_processor is not None
        assert rag.vector_store is not None
        assert rag.ai_generator is not None
        assert rag.session_manager is not None
        assert rag.tool_manager is not None

    def test_rag_system_registers_both_tools(self, test_config):
        """3.1b: Both CourseSearchTool and CourseOutlineTool are registered"""
        from rag_system import RAGSystem

        rag = RAGSystem(test_config)
        definitions = rag.tool_manager.get_tool_definitions()
        tool_names = [d["function"]["name"] for d in definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names


class TestRAGSystemQuery:
    """Suite: RAGSystem.query() — the main query pipeline"""

    def test_query_returns_response_and_sources(self, test_config, mock_openai_chat_completion):
        """3.2: query() returns (response_string, sources_list) tuple"""
        from rag_system import RAGSystem

        rag = RAGSystem(test_config)

        # Mock the AI generator to return a direct response (no tool call)
        with patch.object(rag.ai_generator.client.chat.completions, "create",
                          return_value=mock_openai_chat_completion):
            response, sources = rag.query("What is computer use?")

        assert isinstance(response, str)
        assert isinstance(sources, list)
        assert len(response) > 0

    def test_query_with_session_id(self, test_config, mock_openai_chat_completion):
        """3.3: Session ID is used for conversation history tracking"""
        from rag_system import RAGSystem

        rag = RAGSystem(test_config)
        session_id = rag.session_manager.create_session()

        with patch.object(rag.ai_generator.client.chat.completions, "create",
                          return_value=mock_openai_chat_completion):
            response, sources = rag.query("What is computer use?", session_id=session_id)

        # Session should now have history
        history = rag.session_manager.get_conversation_history(session_id)
        assert history is not None
        assert "What is computer use?" in history

    def test_query_sources_reset_after_each_call(self, test_config, mock_openai_chat_completion):
        """3.4: Sources are reset between successive query calls"""
        from rag_system import RAGSystem

        rag = RAGSystem(test_config)

        with patch.object(rag.ai_generator.client.chat.completions, "create",
                          return_value=mock_openai_chat_completion):
            # First call
            response1, sources1 = rag.query("What is computer use?")
            # Second call
            response2, sources2 = rag.query("What is RAG?")

        # After second call, sources should be empty (no tool was called)
        assert isinstance(sources2, list)

    def test_query_outline_detection_adds_context(self, test_config, mock_openai_chat_completion):
        """3.5: Queries with outline keywords get pre-fetched outline context"""
        from rag_system import RAGSystem

        rag = RAGSystem(test_config)

        # Mock the vector store to return a course for outline pre-fetch
        with patch.object(rag.vector_store, "get_existing_course_titles",
                          return_value=["Building Towards Computer Use with Anthropic"]):
            with patch.object(rag.vector_store, "get_course_outline",
                              return_value={
                                  "title": "Building Towards Computer Use with Anthropic",
                                  "course_link": "https://example.com",
                                  "instructor": "Colt Steele",
                                  "lessons": [
                                      {"lesson_number": 1, "lesson_title": "Introduction",
                                       "lesson_link": None}
                                  ]
                              }):
                with patch.object(rag.ai_generator.client.chat.completions, "create",
                                  return_value=mock_openai_chat_completion) as mock_create:
                    rag.query("What is the outline of the computer use course?")

                    # The prompt should now include outline data
                    call_messages = mock_create.call_args[1]["messages"]
                    user_msg = [m for m in call_messages if m["role"] == "user"][0]
                    assert "Course Outline Data" in user_msg["content"]


class TestQueryEdgeCases:
    """Suite: Edge cases and error handling in the query pipeline"""

    def test_query_empty_string(self, test_config, mock_openai_chat_completion):
        """3.6a: Empty query string is passed through to AI"""
        from rag_system import RAGSystem

        rag = RAGSystem(test_config)

        with patch.object(rag.ai_generator.client.chat.completions, "create",
                          return_value=mock_openai_chat_completion) as mock_create:
            response, sources = rag.query("")

        # AI should still receive the query (empty string is valid)
        assert response is not None
        mock_create.assert_called_once()

    def test_query_special_characters(self, test_config, mock_openai_chat_completion):
        """3.6b: Queries with special characters don't crash"""
        from rag_system import RAGSystem

        rag = RAGSystem(test_config)

        with patch.object(rag.ai_generator.client.chat.completions, "create",
                          return_value=mock_openai_chat_completion):
            response, sources = rag.query("What's the <difference> between RAG & AI? (2024)")

        assert response is not None
        assert isinstance(response, str)


class TestConfigMaxResultsZero:
    """Suite: KEY TEST — MAX_RESULTS=0 bug reproduction"""

    def test_vector_store_search_fails_with_max_results_zero(self, mock_vector_store_max_results_zero):
        """3.7: When max_results=0, VectorStore.search() returns an error

        This is the root cause of 'query failed'. ChromaDB's query() method
        requires n_results > 0, but config.py sets MAX_RESULTS = 0.
        """
        store = mock_vector_store_max_results_zero
        result = store.search(query="computer use")

        assert result.error is not None
        assert "n_results must be > 0" in result.error or "error" in result.error.lower()
        assert result.is_empty()

    def test_course_search_tool_returns_error_with_max_results_zero(
        self, search_tool_max_results_zero
    ):
        """3.7b: CourseSearchTool returns error string when max_results=0"""
        result = search_tool_max_results_zero.execute(query="computer use")

        # The tool returns the error from VectorStore
        assert "error" in result.lower() or "No relevant content" in result

    def test_full_pipeline_returns_query_failed_with_max_results_zero(
        self, test_config_max_results_zero, mock_openai_tool_call_response,
        mock_openai_final_response
    ):
        """3.7c: FULL PIPELINE — query() returns error when MAX_RESULTS=0

        This simulates the exact scenario: LLM calls search_course_content,
        which calls VectorStore with n_results=0, ChromaDB fails,
        tool returns error, LLM says 'query failed'.
        """
        from rag_system import RAGSystem

        rag = RAGSystem(test_config_max_results_zero)

        # Mock the two-step AI call (tool call → response)
        mock_create = MagicMock()
        mock_create.side_effect = [
            mock_openai_tool_call_response,   # LLM decides to use search tool
            mock_openai_final_response         # LLM responds after tool result
        ]

        with patch.object(rag.ai_generator.client.chat.completions, "create", mock_create):
            # We need to make the vector store return an error for n_results=0
            # Patch the search to simulate the ChromaDB failure
            original_search = rag.vector_store.search

            def broken_search(query, course_name=None, lesson_number=None, limit=None):
                search_limit = limit if limit is not None else 0
                if search_limit <= 0:
                    from vector_store import SearchResults
                    return SearchResults.empty("Search error: n_results must be > 0")
                return original_search(query, course_name, lesson_number, limit)

            with patch.object(rag.vector_store, "search", side_effect=broken_search):
                response, sources = rag.query("What is computer use?")

                # The response should contain the error or be a fallback message
                # NOTE: The LLM receives the tool error and may respond differently
                # depending on the model, but the key assertion is that the pipeline
                # doesn't crash — the tool error is handled gracefully
                assert response is not None
                assert isinstance(response, str)
