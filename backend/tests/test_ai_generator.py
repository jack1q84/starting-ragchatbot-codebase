"""Tests for AIGenerator tool calling and ToolManager integration

Tests verify external behavior (API calls made, tools executed, results returned)
rather than internal state details.
"""

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


class TestAIGeneratorToolDefinitions:
    """Suite: AIGenerator correctly sets up and uses tool definitions"""

    def test_system_prompt_mentions_search_tool(self):
        """2.1: SYSTEM_PROMPT contains reference to search_course_content"""
        from ai_generator import AIGenerator
        assert "search_course_content" in AIGenerator.SYSTEM_PROMPT

    def test_system_prompt_mentions_outline_tool(self):
        """2.1b: SYSTEM_PROMPT contains reference to get_course_outline"""
        from ai_generator import AIGenerator
        assert "get_course_outline" in AIGenerator.SYSTEM_PROMPT

    def test_system_prompt_allows_multi_step(self):
        """2.1c: SYSTEM_PROMPT allows multi-step reasoning (removed 'one tool call' limit)"""
        from ai_generator import AIGenerator
        # The old restriction should be gone
        assert "One tool call per query maximum" not in AIGenerator.SYSTEM_PROMPT
        # New multi-step guidance should be present
        assert "Multi-step reasoning" in AIGenerator.SYSTEM_PROMPT
        assert "2 rounds" in AIGenerator.SYSTEM_PROMPT

    def test_course_search_tool_definition_structure(self, search_tool):
        """2.2: Tool definition has correct OpenAI-compatible structure"""
        definition = search_tool.get_tool_definition()
        assert definition["type"] == "function"
        assert definition["function"]["name"] == "search_course_content"
        assert "parameters" in definition["function"]
        assert "query" in definition["function"]["parameters"]["properties"]
        assert definition["function"]["parameters"]["required"] == ["query"]

    def test_course_outline_tool_definition_structure(self, mock_vector_store):
        """2.2b: CourseOutlineTool has correct structure"""
        from search_tools import CourseOutlineTool
        tool = CourseOutlineTool(mock_vector_store)
        definition = tool.get_tool_definition()
        assert definition["type"] == "function"
        assert definition["function"]["name"] == "get_course_outline"
        assert "course_title" in definition["function"]["parameters"]["properties"]

    def test_tool_choice_is_auto(self, mock_openai_chat_completion):
        """2.3: When tools are provided, tool_choice is set to 'auto'"""
        from ai_generator import AIGenerator

        generator = AIGenerator(api_key="test-key", base_url="https://test.com", model="test-model")

        with patch.object(generator.client.chat.completions, "create",
                          return_value=mock_openai_chat_completion) as mock_create:
            generator.generate_response(
                query="test query",
                tools=[{"type": "function", "function": {"name": "test_tool"}}],
                tool_manager=None
            )

            call_kwargs = mock_create.call_args[1]
            assert "tools" in call_kwargs
            assert call_kwargs["tool_choice"] == "auto"

    def test_generate_response_no_tools(self, mock_openai_chat_completion):
        """2.4: Without tools, response is returned directly from the LLM"""
        from ai_generator import AIGenerator

        generator = AIGenerator(api_key="test-key", base_url="https://test.com", model="test-model")

        with patch.object(generator.client.chat.completions, "create",
                          return_value=mock_openai_chat_completion) as mock_create:
            response = generator.generate_response(query="Hello")

            assert response == "Computer use allows AI models to interact with screens."
            mock_create.assert_called_once()
            # No tools param when not provided
            assert "tools" not in mock_create.call_args[1]

    def test_generate_response_without_tool_manager_no_loop(self, mock_openai_tool_call_response):
        """2.4b: Even with tools, if tool_manager is None, don't enter tool loop"""
        from ai_generator import AIGenerator

        generator = AIGenerator(api_key="test-key", base_url="https://test.com", model="test-model")

        with patch.object(generator.client.chat.completions, "create",
                          return_value=mock_openai_tool_call_response) as mock_create:
            response = generator.generate_response(
                query="test",
                tools=[{"type": "function", "function": {"name": "test_tool"}}],
                tool_manager=None
            )

            # Without tool_manager, the tool_calls finish_reason would normally
            # cause a loop — but since tool_manager is None, we check that
            # the code doesn't crash (the generate_response checks both conditions)
            # Note: with tool_manager=None, the code falls through to return content
            # which may be None for tool_calls responses
            assert response is None or isinstance(response, str)
            mock_create.assert_called_once()


class TestSequentialToolLoop:
    """Suite: Sequential tool calling round control and termination conditions"""

    def test_sequential_two_rounds(
        self,
        mock_two_round_responses,
        tool_manager_with_both_tools
    ):
        """3.1: Two sequential tool calls — both tools are executed in order

        API calls: 3 total (Round 1 tool_calls → Round 2 tool_calls → Final stop)
        Tool execs: 2 total (get_course_outline → search_course_content)
        """
        from ai_generator import AIGenerator

        generator = AIGenerator(api_key="test-key", base_url="https://test.com",
                                model="test-model", max_tool_rounds=2)

        mock_create = MagicMock()
        mock_create.side_effect = mock_two_round_responses

        with patch.object(generator.client.chat.completions, "create", mock_create):
            with patch.object(tool_manager_with_both_tools, "execute_tool",
                              wraps=tool_manager_with_both_tools.execute_tool) as spy:
                response = generator.generate_response(
                    query="Find a course that covers the same topic as Course X Lesson 4",
                    tools=tool_manager_with_both_tools.get_tool_definitions(),
                    tool_manager=tool_manager_with_both_tools,
                )

                # execute_tool called twice (two rounds)
                assert spy.call_count == 2

                # First call: get_course_outline
                call_1_args = spy.call_args_list[0]
                assert call_1_args[0][0] == "get_course_outline"
                assert "course_title" in call_1_args[1]

                # Second call: search_course_content
                call_2_args = spy.call_args_list[1]
                assert call_2_args[0][0] == "search_course_content"
                assert "query" in call_2_args[1]

                # Final response is returned
                assert response is not None
                assert isinstance(response, str)
                assert len(response) > 0

        # Total API calls: 3 (initial + round1 + final)
        assert mock_create.call_count == 3

    def test_early_termination_no_tool_call(
        self,
        mock_openai_chat_completion,
        tool_manager
    ):
        """3.2: LLM returns stop on first call — no tool loop entered"""
        from ai_generator import AIGenerator

        generator = AIGenerator(api_key="test-key", base_url="https://test.com",
                                model="test-model")

        with patch.object(generator.client.chat.completions, "create",
                          return_value=mock_openai_chat_completion) as mock_create:
            response = generator.generate_response(
                query="What is the capital of France?",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager,
            )

            # Direct response returned
            assert "computer use" in response.lower()

            # Only 1 API call
            mock_create.assert_called_once()

            # Tools were provided in the call
            assert "tools" in mock_create.call_args[1]

    def test_early_termination_second_round(
        self,
        mock_second_round_no_tool,
        tool_manager
    ):
        """3.3: Round 1 calls tool, Round 2 stops — no further rounds"""
        from ai_generator import AIGenerator

        generator = AIGenerator(api_key="test-key", base_url="https://test.com",
                                model="test-model", max_tool_rounds=2)

        mock_create = MagicMock()
        mock_create.side_effect = mock_second_round_no_tool

        with patch.object(generator.client.chat.completions, "create", mock_create):
            with patch.object(tool_manager, "execute_tool",
                              wraps=tool_manager.execute_tool) as spy:
                response = generator.generate_response(
                    query="What is computer use?",
                    tools=tool_manager.get_tool_definitions(),
                    tool_manager=tool_manager,
                )

                # execute_tool called once (only round 1)
                spy.assert_called_once()

                # Response returned (from round 2 stop)
                assert response is not None
                assert isinstance(response, str)

        # API calls: 2 (initial + round 1 → stop)
        assert mock_create.call_count == 2

    def test_max_rounds_enforced(
        self,
        mock_openai_tool_call_response,
        mock_openai_tool_call_response_v2,
        mock_openai_final_response,
        tool_manager_with_both_tools
    ):
        """3.4: Even if Claude wants more tools at round 2, it's forced to stop"""
        from ai_generator import AIGenerator

        generator = AIGenerator(api_key="test-key", base_url="https://test.com",
                                model="test-model", max_tool_rounds=2)

        mock_create = MagicMock()
        mock_create.side_effect = [
            mock_openai_tool_call_response,       # Round 1: tool_calls
            mock_openai_tool_call_response_v2,    # Round 2: still wants tools → but tools removed!
            mock_openai_final_response,           # Final: no tools, forced answer
        ]

        with patch.object(generator.client.chat.completions, "create", mock_create):
            with patch.object(tool_manager_with_both_tools, "execute_tool",
                              wraps=tool_manager_with_both_tools.execute_tool) as spy:
                response = generator.generate_response(
                    query="Complex query",
                    tools=tool_manager_with_both_tools.get_tool_definitions(),
                    tool_manager=tool_manager_with_both_tools,
                )

                # execute_tool called twice
                assert spy.call_count == 2

                # Round 2 API call (index 1) still has tools (not final round)
                call_2_kwargs = mock_create.call_args_list[1][1]
                assert "tools" in call_2_kwargs
                assert call_2_kwargs["tool_choice"] == "auto"

                # Final API call (index 2) has NO tools
                call_3_kwargs = mock_create.call_args_list[2][1]
                assert "tools" not in call_3_kwargs

                assert response is not None
                assert isinstance(response, str)

        # API calls: 3
        assert mock_create.call_count == 3

    def test_tool_execution_error(
        self,
        mock_openai_tool_call_response,
        mock_openai_final_response,
        tool_manager
    ):
        """3.5: Tool execution exception is caught gracefully, error passed to LLM"""
        from ai_generator import AIGenerator

        generator = AIGenerator(api_key="test-key", base_url="https://test.com",
                                model="test-model")

        mock_create = MagicMock()
        mock_create.side_effect = [
            mock_openai_tool_call_response,   # Round 1: tool_calls
            mock_openai_final_response,       # Final: response
        ]

        with patch.object(generator.client.chat.completions, "create", mock_create):
            with patch.object(tool_manager, "execute_tool",
                              side_effect=ValueError("ChromaDB connection failed")):
                # Exception should NOT propagate to caller
                response = generator.generate_response(
                    query="What is computer use?",
                    tools=tool_manager.get_tool_definitions(),
                    tool_manager=tool_manager,
                )

                # Graceful handling: response returned
                assert response is not None
                assert isinstance(response, str)

        # API calls still happened
        assert mock_create.call_count == 2


class TestSequentialAPIParams:
    """Suite: API parameter correctness across sequential rounds"""

    def test_tools_remain_in_nonfinal_rounds(
        self,
        mock_openai_tool_call_response,
        mock_openai_tool_call_response_v2,
        tool_manager_with_both_tools
    ):
        """3.6: Non-final rounds include tools parameter with tool_choice='auto'"""
        from ai_generator import AIGenerator

        generator = AIGenerator(api_key="test-key", base_url="https://test.com",
                                model="test-model", max_tool_rounds=3)

        # Need a final stop response
        mock_final = MagicMock()
        mock_final.choices[0].message.content = "Final answer"
        mock_final.choices[0].message.tool_calls = None
        mock_final.choices[0].finish_reason = "stop"
        def _final_dump():
            return {"role": "assistant", "content": "Final answer", "tool_calls": None}
        mock_final.choices[0].message.model_dump = _final_dump

        mock_create = MagicMock()
        mock_create.side_effect = [
            mock_openai_tool_call_response,       # Round 1
            mock_openai_tool_call_response_v2,    # Round 2 (non-final, should have tools)
            mock_final,                           # Round 3 (final, no tools)
        ]

        with patch.object(generator.client.chat.completions, "create", mock_create):
            generator.generate_response(
                query="test",
                tools=tool_manager_with_both_tools.get_tool_definitions(),
                tool_manager=tool_manager_with_both_tools,
            )

            # Round 1 (initial call) has tools
            call_1 = mock_create.call_args_list[0][1]
            assert "tools" in call_1

            # Round 2 (non-final loop iteration) has tools
            call_2 = mock_create.call_args_list[1][1]
            assert "tools" in call_2
            assert call_2["tool_choice"] == "auto"

    def test_final_round_removes_tools(
        self,
        mock_openai_tool_call_response,
        mock_openai_tool_call_response_v2,
        tool_manager_with_both_tools
    ):
        """3.7: Final round (max_tool_rounds reached) removes tools from API params"""
        from ai_generator import AIGenerator

        generator = AIGenerator(api_key="test-key", base_url="https://test.com",
                                model="test-model", max_tool_rounds=2)

        mock_final = MagicMock()
        mock_final.choices[0].message.content = "Here is my final answer."
        mock_final.choices[0].message.tool_calls = None
        mock_final.choices[0].finish_reason = "stop"
        def _final_dump():
            return {"role": "assistant", "content": "Here is my final answer.", "tool_calls": None}
        mock_final.choices[0].message.model_dump = _final_dump

        mock_create = MagicMock()
        mock_create.side_effect = [
            mock_openai_tool_call_response,       # Round 1
            mock_openai_tool_call_response_v2,    # Round 2 (tools removed by code)
            mock_final,                           # Final stop
        ]

        with patch.object(generator.client.chat.completions, "create", mock_create):
            response = generator.generate_response(
                query="test",
                tools=tool_manager_with_both_tools.get_tool_definitions(),
                tool_manager=tool_manager_with_both_tools,
            )

            # Round 2 call (the one that received tool_calls_v2) has NO tools
            # because max_tool_rounds=2 means round 1 (index 0) is last with tools
            # Wait — let me re-check: max_tool_rounds=2 means:
            #   round_num=0 (non-final) → has tools
            #   round_num=1 (final) → no tools
            # 
            # But round_num=1 is the SECOND iteration in the loop.
            # So the API call at index 1 is the one WITH tools (round_num=0),
            # and call at index 2 (round_num=1, final) has no tools.
            
            # Actually, let me trace more carefully:
            # generate_response makes initial call → call_args_list[0]
            # _sequential_tool_loop round 0 (non-final, max-1=1, is_last=False) → call_args_list[1]
            # _sequential_tool_loop round 1 (final, is_last=True) → call_args_list[2]
            
            # Wait, the mock side_effect is [round1, round2, final]
            # generate_response's first api call: mock[0] = round1 (tool_calls response)
            # loop round 0: is_last? round_num(0) == max_tool_rounds-1(1)? No → has tools
            #   api call: mock[1] = round2 (another tool_calls)
            # loop round 1: is_last? Yes → no tools
            #   api call: mock[2] = final (stop)
            
            # So call_args_list:
            # [0] = initial call (has tools)
            # [1] = loop round 0 (has tools — non-final)
            # [2] = loop round 1 (NO tools — final)
            
            # Round 2 call (index 1) has tools
            call_2 = mock_create.call_args_list[1][1]
            assert "tools" in call_2  # non-final, has tools

            # Final call (index 2) has NO tools
            call_3 = mock_create.call_args_list[2][1]
            assert "tools" not in call_3

            assert response == "Here is my final answer."

    def test_message_context_preserved(
        self,
        mock_two_round_responses,
        tool_manager_with_both_tools
    ):
        """3.8: Message history grows across rounds and system/user messages remain unchanged"""
        from ai_generator import AIGenerator

        generator = AIGenerator(api_key="test-key", base_url="https://test.com",
                                model="test-model")

        mock_create = MagicMock()
        mock_create.side_effect = mock_two_round_responses

        with patch.object(generator.client.chat.completions, "create", mock_create):
            generator.generate_response(
                query="test query",
                tools=tool_manager_with_both_tools.get_tool_definitions(),
                tool_manager=tool_manager_with_both_tools,
            )

            # Total API calls = 3 (initial + round 1 + round 2)
            assert mock_create.call_count == 3

            # The last API call's messages should include all conversation history
            final_msgs = mock_create.call_args_list[-1][1]["messages"]

            # system + user messages are always preserved unchanged
            assert final_msgs[0]["role"] == "system"
            assert final_msgs[1]["role"] == "user"
            assert final_msgs[1]["content"] == "test query"

            # At minimum: system + user + (assistant+tool) * 2 rounds = 6
            # (may be more depending on round count)
            assert len(final_msgs) >= 6

            # Last message should always be a tool result (role: tool)
            # from the latest round of tool execution
            assert final_msgs[-1]["role"] == "tool"

            # Messages alternate: system, user, then pairs of assistant+tool
            tool_roles = [m["role"] for m in final_msgs[2:]]
            # Should have at least one assistant+tool pair
            assert "assistant" in tool_roles
            assert "tool" in tool_roles


class TestToolManagerIntegration:
    """Suite: ToolManager internal routing and sources"""

    def test_tool_manager_execute_search_tool(self, search_tool):
        """2.7: ToolManager correctly routes to CourseSearchTool"""
        from search_tools import ToolManager
        mgr = ToolManager()
        mgr.register_tool(search_tool)

        result = mgr.execute_tool("search_course_content", query="test")
        assert result is not None
        assert "Building Towards Computer Use" in result

    def test_tool_manager_get_last_sources(self, tool_manager):
        """2.7b: get_last_sources returns sources from the last search"""
        tool_manager.execute_tool("search_course_content", query="test")
        sources = tool_manager.get_last_sources()
        assert isinstance(sources, list)
        if sources:
            assert "text" in sources[0]

    def test_get_tool_definitions_format(self, tool_manager):
        """2.8: get_tool_definitions returns list of OpenAI-compatible definitions"""
        definitions = tool_manager.get_tool_definitions()
        assert isinstance(definitions, list)
        assert len(definitions) >= 1

        for defn in definitions:
            assert defn["type"] == "function"
            assert "function" in defn
            assert defn["function"]["name"] is not None

    def test_tool_manager_accumulates_sources_across_rounds(
        self,
        tool_manager_with_both_tools
    ):
        """4.1: ToolManager accumulates sources across multi-round tool calls"""
        # Round 1: search → produces sources
        tool_manager_with_both_tools.execute_tool(
            "search_course_content", query="computer use"
        )
        assert len(tool_manager_with_both_tools.all_sources) > 0

        first_count = len(tool_manager_with_both_tools.all_sources)

        # Round 2: outline → no sources (CourseOutlineTool doesn't track them)
        tool_manager_with_both_tools.execute_tool(
            "get_course_outline", course_title="Test Course"
        )
        # all_sources should be unchanged (outline produces no sources)
        assert len(tool_manager_with_both_tools.all_sources) == first_count

        # Round 3: search again → adds more sources
        tool_manager_with_both_tools.execute_tool(
            "search_course_content", query="prompt compression"
        )
        assert len(tool_manager_with_both_tools.all_sources) >= 1

        # Verify get_all_sources() returns the accumulated list
        all_sources = tool_manager_with_both_tools.get_all_sources()
        assert len(all_sources) == len(tool_manager_with_both_tools.all_sources)

    def test_tool_manager_reset_sources_clears_accumulated(
        self,
        tool_manager_with_both_tools
    ):
        """4.2: reset_sources clears both accumulated and per-tool sources"""
        tool_manager_with_both_tools.execute_tool(
            "search_course_content", query="computer use"
        )
        assert len(tool_manager_with_both_tools.get_all_sources()) > 0

        tool_manager_with_both_tools.reset_sources()
        assert len(tool_manager_with_both_tools.get_all_sources()) == 0
