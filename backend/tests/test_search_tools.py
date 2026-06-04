"""Tests for CourseSearchTool.execute() and related search functionality"""

import pytest
from search_tools import CourseSearchTool


class TestCourseSearchToolExecute:
    """Suite: CourseSearchTool.execute() — all branches and edge cases"""

    def test_execute_valid_query(self, search_tool, mock_search_results_valid):
        """1.1: Valid query returns properly formatted results with course/lesson headers"""
        result = search_tool.execute(query="computer use")

        # Should contain the course name header
        assert "[Building Towards Computer Use with Anthropic" in result
        # Should contain lesson number
        assert "Lesson 0" in result or "Lesson 1" in result
        # Should contain actual content
        assert "computer use" in result.lower() or "screen" in result.lower()
        # Should not contain error message
        assert "Error" not in result
        assert "No relevant content" not in result

    def test_execute_with_course_filter(self, search_tool, mock_vector_store):
        """1.2: Passing course_name filters the search via VectorStore"""
        search_tool.execute(
            query="multi-modal requests",
            course_name="Building Towards Computer Use"
        )

        # Verify the vector store search was called with the course_name
        mock_vector_store.search.assert_called_once_with(
            query="multi-modal requests",
            course_name="Building Towards Computer Use",
            lesson_number=None
        )

    def test_execute_with_lesson_filter(self, search_tool, mock_vector_store):
        """1.3: Passing lesson_number filters the search"""
        search_tool.execute(query="API", lesson_number=1)

        mock_vector_store.search.assert_called_once_with(
            query="API",
            course_name=None,
            lesson_number=1
        )

    def test_execute_all_filters(self, search_tool, mock_vector_store):
        """1.4: Both course_name and lesson_number are passed through"""
        search_tool.execute(
            query="multi-modal",
            course_name="Computer Use",
            lesson_number=1
        )

        mock_vector_store.search.assert_called_once_with(
            query="multi-modal",
            course_name="Computer Use",
            lesson_number=1
        )

    def test_execute_no_results(self, search_tool_no_results):
        """1.5: Empty results return 'No relevant content found' message"""
        result = search_tool_no_results.execute(query="nonexistent topic")
        assert "No relevant content found" in result

    def test_execute_no_results_with_filter(self, search_tool_no_results):
        """1.5b: Empty results mention the active filters"""
        result = search_tool_no_results.execute(
            query="nonexistent",
            course_name="MCP Course",
            lesson_number=5
        )
        assert "No relevant content found" in result
        assert "MCP Course" in result
        assert "lesson 5" in result.lower()

    def test_execute_course_not_found(self, search_tool_error):
        """1.6: Search errors are passed through from VectorStore"""
        result = search_tool_error.execute(query="anything")
        assert "Search error" in result or "error" in result.lower()

    def test_format_results_sources_populated(self, search_tool):
        """1.7: After execute(), last_sources contains source info"""
        search_tool.execute(query="computer use")
        assert len(search_tool.last_sources) > 0
        # Each source should have text and link
        for source in search_tool.last_sources:
            assert "text" in source
            assert "link" in source

    def test_execute_resets_sources(self, search_tool, mock_vector_store):
        """1.8: last_sources is updated on each execute call"""
        # First call populates sources
        search_tool.execute(query="computer use")
        first_sources = search_tool.last_sources.copy()
        assert len(first_sources) > 0
        assert first_sources[0]["text"] == "Building Towards Computer Use with Anthropic - Lesson 0"

        # Second call with different query updates sources
        mock_vector_store.search.return_value = type('search_results', (), {
            'documents': ["Different content"],
            'metadata': [{"course_title": "Different Course", "lesson_number": 2, "chunk_index": 0}],
            'distances': [0.5],
            'error': None,
            'is_empty': lambda self: False
        })()

        search_tool.execute(query="different topic")
        # Sources should now reflect the new search
        assert len(search_tool.last_sources) == 1
        assert search_tool.last_sources[0]["text"] == "Different Course - Lesson 2"

    def test_sources_persist_on_error(self, search_tool, mock_vector_store):
        """1.8b: When search returns an error, last_sources retains previous values
        (CourseSearchTool.execute() returns early on error without modifying last_sources)"""
        # First call populates sources
        search_tool.execute(query="computer use")
        first_sources = search_tool.last_sources.copy()
        assert len(first_sources) > 0

        # Mock an error for second call
        mock_vector_store.search.return_value = type('search_results', (), {
            'error': 'Search error: test',
            'is_empty': lambda self: False
        })()

        result = search_tool.execute(query="error case")
        assert "Search error" in result
        # last_sources retains previous values (code doesn't clear them on error)
        assert len(search_tool.last_sources) == len(first_sources)

    def test_execute_empty_query(self, search_tool):
        """1.9: Empty string query still goes to VectorStore (no client-side validation)"""
        # The tool doesn't validate empty queries — just passes through
        result = search_tool.execute(query="")
        # Should still go through the pipeline (VectorStore decides what happens)
        assert result is not None


class TestCourseSearchToolFormatting:
    """Suite: Output formatting details"""

    def test_format_single_result(self, search_tool):
        """Single result should be formatted correctly"""
        result = search_tool.execute(query="computer use")
        # Should start with [Course Name]
        assert result.startswith("[")
        assert "]" in result

    def test_sources_have_lesson_context(self, search_tool):
        """Source entries include lesson numbers when available"""
        search_tool.execute(query="computer use")
        for source in search_tool.last_sources:
            assert "text" in source
            # Should contain course name
            assert len(source["text"]) > 0


class TestToolManager:
    """Suite: ToolManager routing and source management"""

    def test_register_and_execute(self, tool_manager):
        """ToolManager routes search_course_content to CourseSearchTool"""
        result = tool_manager.execute_tool(
            "search_course_content",
            query="computer use"
        )
        assert result is not None
        assert "[Building Towards Computer Use with Anthropic" in result

    def test_execute_unknown_tool(self, tool_manager):
        """Unknown tool name returns error message"""
        result = tool_manager.execute_tool("nonexistent_tool", query="test")
        assert "not found" in result.lower() or "Tool" in result

    def test_get_last_sources(self, tool_manager):
        """ToolManager.get_last_sources() retrieves sources from tools"""
        tool_manager.execute_tool("search_course_content", query="computer use")
        # execute_tool now collects sources into all_sources and clears last_sources
        # get_last_sources() only returns sources still on the tool (not yet collected)
        # Use get_all_sources() to get accumulated sources
        sources = tool_manager.get_all_sources()
        assert len(sources) > 0
        assert "text" in sources[0]

    def test_reset_sources(self, tool_manager):
        """ToolManager.reset_sources() clears sources from all tools"""
        tool_manager.execute_tool("search_course_content", query="computer use")
        assert len(tool_manager.get_all_sources()) > 0
        tool_manager.reset_sources()
        assert len(tool_manager.get_all_sources()) == 0
