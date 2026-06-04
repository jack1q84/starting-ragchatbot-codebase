from typing import Dict, Any, Optional, List, Protocol
from abc import ABC, abstractmethod
from vector_store import VectorStore, SearchResults
import json


class Tool(ABC):
    """Abstract base class for all tools"""
    
    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return OpenAI-compatible tool definition for this tool"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return OpenAI-compatible tool definition for this tool"""
        return {
            "type": "function",
            "function": {
                "name": "search_course_content",
                "description": "Search within lesson content for specific topics, concepts, or technical details. NOT for course outlines, syllabi, or lesson listings — use get_course_outline for that.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string", 
                            "description": "What to search for in the course content"
                        },
                        "course_name": {
                            "type": "string",
                            "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                        },
                        "lesson_number": {
                            "type": "integer",
                            "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    def execute(self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None) -> str:
        """
        Execute the search tool with given parameters.
        
        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter
            
        Returns:
            Formatted search results or error message
        """
        
        # Use the vector store's unified search interface
        results = self.store.search(
            query=query,
            course_name=course_name,
            lesson_number=lesson_number
        )
        
        # Handle errors
        if results.error:
            return results.error
        
        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."
        
        # Format and return results
        return self._format_results(results)
    
    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI
        
        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')
            
            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"
            
            # Look up lesson link from the course catalog
            link = None
            if course_title and lesson_num is not None:
                link = self.store.get_lesson_link(course_title, lesson_num)
            if not link and course_title:
                link = self.store.get_course_link(course_title)
            
            # Track source for the UI (with invisible link data)
            source_text = course_title
            if lesson_num is not None:
                source_text += f" - Lesson {lesson_num}"
            
            sources.append({
                "text": source_text,
                "link": link or ""
            })
            
            formatted.append(f"{header}\n{doc}")
        
        # Store sources for retrieval
        self.last_sources = sources
        
        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for retrieving course outline (title, link, and complete lesson list)"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return OpenAI-compatible tool definition for this tool"""
        return {
            "type": "function",
            "function": {
                "name": "get_course_outline",
                "description": "Get the full outline of a course including course title, course link, and complete lesson list with lesson numbers and titles. Use this for questions about course structure, syllabus, or what a course covers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "course_title": {
                            "type": "string",
                            "description": "The full or partial course title to look up (e.g. 'MCP: Build Rich-Context AI Apps with Anthropic')"
                        }
                    },
                    "required": ["course_title"]
                }
            }
        }
    
    def execute(self, course_title: str) -> str:
        """
        Execute the course outline tool.
        
        Args:
            course_title: The course title to look up
            
        Returns:
            Formatted course outline or error message
        """
        # First resolve the course name via semantic search
        resolved_title = self.store._resolve_course_name(course_title)
        if not resolved_title:
            return f"No course found matching '{course_title}'."
        
        # Get the full outline
        outline = self.store.get_course_outline(resolved_title)
        if not outline:
            return f"Could not retrieve outline for '{resolved_title}'."
        
        # Format the outline
        lines = []
        lines.append(f"Course Title: {outline['title']}")
        if outline.get('course_link'):
            lines.append(f"Course Link: {outline['course_link']}")
        if outline.get('instructor'):
            lines.append(f"Instructor: {outline['instructor']}")
        lines.append("")
        lines.append("Lessons:")
        for lesson in outline['lessons']:
            lesson_line = f"  Lesson {lesson['lesson_number']}: {lesson['lesson_title']}"
            if lesson.get('lesson_link'):
                lesson_line += f" ({lesson['lesson_link']})"
            lines.append(lesson_line)
        
        return "\n".join(lines)


class ToolManager:
    """Manages available tools for the AI"""
    
    def __init__(self):
        self.tools = {}
        self.all_sources = []  # Accumulated sources across multi-round tool calls
    
    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("function", {}).get("name") if "function" in tool_def else tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    
    def get_tool_definitions(self) -> list:
        """Get all tool definitions for OpenAI-compatible tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name and auto-collect sources across rounds"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"
        
        result = self.tools[tool_name].execute(**kwargs)
        
        # Auto-collect sources from tools that track them
        if hasattr(self.tools[tool_name], 'last_sources'):
            if self.tools[tool_name].last_sources:
                self.all_sources.extend(self.tools[tool_name].last_sources)
                # Clear to prevent double-counting on next execute
                self.tools[tool_name].last_sources = []
        
        return result
    
    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources') and tool.last_sources:
                return tool.last_sources
        return []

    def get_all_sources(self) -> list:
        """Get all accumulated sources across multi-round tool calls"""
        return self.all_sources

    def reset_sources(self):
        """Reset all sources (accumulated and per-tool)"""
        self.all_sources = []
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = []