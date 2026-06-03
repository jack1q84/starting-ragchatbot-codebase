import json
from openai import OpenAI
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with LLM API (OpenAI-compatible, e.g. DeepSeek) for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


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