from typing import List, Tuple, Optional, Dict
import os
from document_processor import DocumentProcessor
from vector_store import VectorStore
from ai_generator import AIGenerator
from session_manager import SessionManager
from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool
from models import Course, Lesson, CourseChunk

class RAGSystem:
    """Main orchestrator for the Retrieval-Augmented Generation system"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize core components
        self.document_processor = DocumentProcessor(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        self.vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        self.ai_generator = AIGenerator(config.LLM_API_KEY, config.LLM_BASE_URL, config.LLM_MODEL)
        self.session_manager = SessionManager(config.MAX_HISTORY)
        
        # Initialize search tools
        self.tool_manager = ToolManager()
        self.search_tool = CourseSearchTool(self.vector_store)
        self.tool_manager.register_tool(self.search_tool)
        self.outline_tool = CourseOutlineTool(self.vector_store)
        self.tool_manager.register_tool(self.outline_tool)
    
    def add_course_document(self, file_path: str) -> Tuple[Course, int]:
        """
        Add a single course document to the knowledge base.
        
        Args:
            file_path: Path to the course document
            
        Returns:
            Tuple of (Course object, number of chunks created)
        """
        try:
            # Process the document
            course, course_chunks = self.document_processor.process_course_document(file_path)
            
            # Add course metadata to vector store for semantic search
            self.vector_store.add_course_metadata(course)
            
            # Add course content chunks to vector store
            self.vector_store.add_course_content(course_chunks)
            
            return course, len(course_chunks)
        except Exception as e:
            print(f"Error processing course document {file_path}: {e}")
            return None, 0
    
    def add_course_folder(self, folder_path: str, clear_existing: bool = False) -> Tuple[int, int]:
        """
        Add all course documents from a folder.
        
        Args:
            folder_path: Path to folder containing course documents
            clear_existing: Whether to clear existing data first
            
        Returns:
            Tuple of (total courses added, total chunks created)
        """
        total_courses = 0
        total_chunks = 0
        
        # Clear existing data if requested
        if clear_existing:
            print("Clearing existing data for fresh rebuild...")
            self.vector_store.clear_all_data()
        
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist")
            return 0, 0
        
        # Get existing course titles to avoid re-processing
        existing_course_titles = set(self.vector_store.get_existing_course_titles())
        
        # Process each file in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(('.pdf', '.docx', '.txt')):
                try:
                    # Check if this course might already exist
                    # We'll process the document to get the course ID, but only add if new
                    course, course_chunks = self.document_processor.process_course_document(file_path)
                    
                    if course and course.title not in existing_course_titles:
                        # This is a new course - add it to the vector store
                        self.vector_store.add_course_metadata(course)
                        self.vector_store.add_course_content(course_chunks)
                        total_courses += 1
                        total_chunks += len(course_chunks)
                        print(f"Added new course: {course.title} ({len(course_chunks)} chunks)")
                        existing_course_titles.add(course.title)
                    elif course:
                        print(f"Course already exists: {course.title} - skipping")
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
        
        return total_courses, total_chunks
    
    def query(self, query: str, session_id: Optional[str] = None) -> Tuple[str, List[str]]:
        """
        Process a user query using the RAG system with tool-based search.
        
        Args:
            query: User's question
            session_id: Optional session ID for conversation context
            
        Returns:
            Tuple of (response, sources list - empty for tool-based approach)
        """
        # Auto-detect outline queries and pre-fetch course outline data
        outline_keywords = ['outline', 'syllabus', 'lesson list', 'list all lesson',
                           'what does.*cover', 'structure of', 'lessons in',
                           'course overview', 'topics covered', 'curriculum']
        outline_context = ""
        import re
        STOP_WORDS = {'the', 'a', 'an', 'of', 'in', 'to', 'for', 'with', 'on', 'at',
                      'is', 'are', 'was', 'were', 'be', 'been', 'being', 'do', 'does',
                      'did', 'will', 'would', 'can', 'could', 'may', 'might', 'shall',
                      'should', 'has', 'have', 'had', 'not', 'no', 'nor', 'this', 'that',
                      'these', 'those', 'it', 'its', 'what', 'which', 'who', 'whom',
                      'and', 'but', 'or', 'as', 'by', 'from', 'about', 'all', 'each',
                      'every', 'some', 'any', 'both', 'more', 'most', 'other', 'than',
                      'so', 'if', 'into', 'over', 'very', 'just', 'also', 'how'}
        for keyword in outline_keywords:
            if re.search(keyword, query, re.IGNORECASE):
                # Try to extract course name from query
                course_titles = self.vector_store.get_existing_course_titles()
                matched_course = None
                
                # Strategy 1: Check if any course title appears as substring in query
                query_lower = query.lower()
                for title in course_titles:
                    if title.lower() in query_lower:
                        matched_course = title
                        break
                
                # Strategy 2: Check for significant word overlap (excluding stop words)
                if not matched_course:
                    query_significant = {w for w in query_lower.split() if w not in STOP_WORDS and len(w) > 2}
                    best_match = None
                    best_count = 0
                    for title in course_titles:
                        title_words = {w for w in title.lower().split() if w not in STOP_WORDS and len(w) > 2}
                        overlap = query_significant & title_words
                        if len(overlap) > best_count:
                            best_count = len(overlap)
                            best_match = title
                    if best_count > 0:
                        matched_course = best_match
                
                # Strategy 3: Fallback by checking for specific terms (MCP, Chroma, etc.)
                if not matched_course:
                    specific_terms = ['mcp', 'chroma', 'computer use', 'prompt compression',
                                     'rag', 'retrieval', 'anthropic', 'deeplearning']
                    for term in specific_terms:
                        if term in query_lower:
                            for t in course_titles:
                                if term in t.lower():
                                    matched_course = t
                                    break
                            if matched_course:
                                break
                
                if matched_course:
                    outline = self.vector_store.get_course_outline(matched_course)
                    if outline:
                        lines = [f"Course Title: {outline['title']}"]
                        if outline.get('course_link'):
                            lines.append(f"Course Link: {outline['course_link']}")
                        if outline.get('instructor'):
                            lines.append(f"Instructor: {outline['instructor']}")
                        lines.append("Lessons:")
                        for lesson in outline['lessons']:
                            lines.append(f"  Lesson {lesson['lesson_number']}: {lesson['lesson_title']}")
                        outline_context = "\n".join(lines)
                break  # Process only first matched keyword
        
        # Create prompt for the AI
        if outline_context:
            prompt = f"""Answer this question about course materials using the provided course outline data.
DO NOT search for this information - it is already provided below.

Question: {query}

Course Outline Data:
{outline_context}"""
        else:
            prompt = f"""Answer this question about course materials: {query}"""
        
        # Get conversation history if session exists
        history = None
        if session_id:
            history = self.session_manager.get_conversation_history(session_id)
        
        # Generate response using AI with tools
        response = self.ai_generator.generate_response(
            query=prompt,
            conversation_history=history,
            tools=self.tool_manager.get_tool_definitions(),
            tool_manager=self.tool_manager
        )
        
        # Get all accumulated sources from multi-round tool calls
        sources = self.tool_manager.get_all_sources()

        # Reset sources after retrieving them
        self.tool_manager.reset_sources()
        
        # Update conversation history
        if session_id:
            self.session_manager.add_exchange(session_id, query, response)
        
        # Return response with sources from tool searches
        return response, sources
    
    def get_course_analytics(self) -> Dict:
        """Get analytics about the course catalog"""
        return {
            "total_courses": self.vector_store.get_course_count(),
            "course_titles": self.vector_store.get_existing_course_titles()
        }