import json
from openai import OpenAI
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with LLM API (OpenAI-compatible, e.g. DeepSeek) for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to two tools for course information.

Available Tools:
1. **search_course_content** — Searches within course material for detailed content on specific topics. Use this for questions about what a lesson covers, specific concepts, or technical details.

2. **get_course_outline** — Retrieves the full outline of a course including the course title, course link, instructor, and complete list of lessons with numbers and titles. Use this for questions about course structure, syllabus, or what a course covers overall.

Tool Usage Guidelines:
- **For outline/syllabus/structure questions**: You MUST call `get_course_outline` with the course title. Do NOT use `search_course_content` for this.
- **For specific lesson content questions**: Call `search_course_content` with relevant search terms and optional filters.
- **Multi-step reasoning**: When a question requires multiple pieces of information (e.g., first find a course outline, then search for related content), you may call tools sequentially across multiple rounds. Each round builds on the previous results.
- **Plan before calling**: If you need information from one tool to formulate parameters for another, call them one at a time across rounds.
- **Maximum tool rounds**: You have up to 2 rounds of tool calls to gather the information you need.
- Synthesize tool results into accurate, fact-based responses.
- If a tool yields no results, state this clearly without offering alternatives.
- CRITICAL: When asked about a course outline, syllabus, or lesson list, ALWAYS use get_course_outline — it returns the complete lesson list from metadata.

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Use the appropriate tool first, then answer based on the results
- **No meta-commentary**:
  - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
  - Do not mention "based on the search results" or "I used the tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, base_url: str, model: str, max_tool_rounds: int = 2):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional sequential tool usage and conversation context.
        
        Supports up to `max_tool_rounds` rounds of sequential tool calling,
        allowing Claude to reason across multiple tool results for complex queries.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use (OpenAI format)
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        # Build messages list
        messages = []
        
        # System prompt as first message
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        messages.append({"role": "system", "content": system_content})
        
        # User query
        messages.append({"role": "user", "content": query})
        
        # Prepare API call parameters
        api_params = {
            **self.base_params,
            "messages": messages
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"
        
        # Get response from LLM
        response = self.client.chat.completions.create(**api_params)
        
        # Enter sequential tool loop if needed
        if response.choices[0].finish_reason == "tool_calls" and tool_manager:
            return self._sequential_tool_loop(response, messages, tools, tool_manager)
        
        # Return direct response
        return response.choices[0].message.content
    
    def _sequential_tool_loop(self, initial_response, messages: List[Dict[str, Any]], 
                                tools: Optional[List], tool_manager) -> str:
        """
        Execute sequential tool calls across up to max_tool_rounds rounds.
        
        Each round:
        1. Appends the assistant message (with tool_calls) to the conversation
        2. Executes all tool calls and appends results
        3. Calls the LLM with tools retained (except on the final round)
        
        Args:
            initial_response: The response containing the first tool call request
            messages: Current message list (system + user)
            tools: Available tool definitions (included in non-final rounds)
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after all tool rounds
        """
        current_response = initial_response
        current_messages = messages.copy()
        
        for round_num in range(self.max_tool_rounds):
            assistant_msg = current_response.choices[0].message
            
            # Termination condition (b): Claude chose not to call tools
            if current_response.choices[0].finish_reason != "tool_calls":
                return assistant_msg.content
            
            # Append assistant message (use model_dump for safe serialization)
            current_messages.append(assistant_msg.model_dump())
            
            # Execute all tool calls from this round
            for tool_call in assistant_msg.tool_calls:
                try:
                    tool_result = tool_manager.execute_tool(
                        tool_call.function.name,
                        **json.loads(tool_call.function.arguments)
                    )
                except Exception as e:
                    # Termination condition (c): graceful error handling
                    tool_result = f"Tool execution error: {str(e)}"
                
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
            
            # Build next round API parameters
            round_params = {
                **self.base_params,
                "messages": current_messages
            }
            
            # Non-final rounds keep tools; final round removes them
            is_last_round = (round_num == self.max_tool_rounds - 1)
            if not is_last_round:
                round_params["tools"] = tools
                round_params["tool_choice"] = "auto"
            # Final round: no tools → LLM must produce a direct answer
            
            # Get next response from LLM
            current_response = self.client.chat.completions.create(**round_params)
        
        # Termination condition (a): max rounds reached
        return current_response.choices[0].message.content or "I have completed my analysis."
