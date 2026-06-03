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
- **One tool call per query maximum** — do not call both tools.
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
    
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        
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
        Generate AI response with optional tool usage and conversation context.
        
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
        
        # Handle tool execution if needed
        if response.choices[0].finish_reason == "tool_calls" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response.choices[0].message.content
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool call requests
            base_params: Base API parameters (with messages)
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Get the assistant message with tool calls
        assistant_message = initial_response.choices[0].message
        
        # Add assistant's tool call response
        messages.append(assistant_message)
        
        # Execute all tool calls and collect results
        tool_call_id = None
        for tool_call in assistant_message.tool_calls:
            tool_call_id = tool_call.id
            tool_result = tool_manager.execute_tool(
                tool_call.function.name,
                **json.loads(tool_call.function.arguments)
            )
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages
        }
        
        # Get final response
        final_response = self.client.chat.completions.create(**final_params)
        return final_response.choices[0].message.content