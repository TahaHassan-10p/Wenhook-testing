from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, Form, File
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.database.models import User, Project, TestCase, AIProvider, AIAnalysis, TestSuite, TestPlan, Milestone, TestRun, Section, SectionTestCase
from app.schemas.extended import (
    AIProviderCreate, AIProviderUpdate, AIProvider as AIProviderSchema,
    AIAnalysisCreate, AIAnalysis as AIAnalysisSchema
)
from app.auth.deps import get_current_active_user, require_permission
from app.core.permissions import Permission
from pydantic import BaseModel
import tempfile
import os
import logging
import json

# Import our new AI services
from app.services.zip_processor import ZipProcessor
from app.services.project_import_processor import AIProjectProcessor, AIProvider as AIProviderEnum, AIProviderConfig
from app.services.code_analysis import CodeAnalysisService, CodeAnalysisResult
from app.services.test_generator import TestCaseGenerator, TestGenerationConfig, GeneratedTestStructure
from app.services.rate_limiter import RateLimiter, AIProvider as RLAIProvider, get_rate_limiter
from app.services.token_manager import TokenManager, AIProvider as TMAIProvider, get_token_manager
from app.services.pricing_manager import PricingManager
from app.services.chunk_processor import EnhancedChunkProcessor
from app.services.analytics import analytics

logger = logging.getLogger(__name__)

router = APIRouter()

# Load mock responses
MOCK_MODE = False  # Set to False to use actual LLM calls

def load_mock_response(filename: str) -> dict:
    """Load mock response from JSON file"""
    try:
        file_path = os.path.join(os.path.dirname(__file__), "..", filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load mock response from {filename}: {e}")
        return None

# Request/Response Models for new AI endpoints
class ProjectFromZipRequest(BaseModel):
    project_name: str
    ai_provider: str  # "openai", "anthropic", "gemini", "vertex", "openrouter", "llama", "anthropic_vertex"
    ai_model: str
    api_key: str
    minimum_test_cases: Optional[int] = 10
    include_edge_cases: bool = True
    include_negative_tests: bool = True
    include_security_tests: bool = True

class TokenEstimationRequest(BaseModel):
    ai_provider: str
    ai_model: str

class CodeAnalysisRequest(BaseModel):
    ai_provider: str
    ai_model: str
    api_key: str
    minimum_test_cases: Optional[int] = 10

class TestGenerationRequest(BaseModel):
    requirements: str
    project_name: Optional[str] = None
    ai_provider: str
    ai_model: str
    api_key: str
    minimum_test_cases: Optional[int] = 10
    maximum_test_cases: Optional[int] = 50
    include_edge_cases: bool = True
    include_negative_tests: bool = True
    include_security_tests: bool = True
    priority_distribution: Optional[Dict[str, float]] = None
    type_distribution: Optional[Dict[str, float]] = None

class ProjectImportResponse(BaseModel):
    success: bool
    project_id: Optional[int] = None
    analysis_results: Optional[Dict[str, Any]] = None
    generation_results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_entities: Dict[str, int] = {}

# New AI-Powered Endpoints

@router.post("/test-form-data")
async def test_form_data(
    zip_file: UploadFile = File(...),
    request_data: str = Form(...),
    current_user: User = Depends(get_current_active_user)
):
    """
    Test endpoint to verify form data parsing
    """
    import json
    
    try:
        # Parse request data
        request_dict = json.loads(request_data)
        
        return {
            "success": True,
            "zip_file": {
                "filename": zip_file.filename,
                "content_type": zip_file.content_type,
                "size": zip_file.size
            },
            "request_data": request_dict,
            "parsed_successfully": True
        }
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": "JSON decode error",
            "details": str(e),
            "raw_request_data": request_data
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@router.post("/debug-zip-contents")
async def debug_zip_contents(
    zip_file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    """
    Debug endpoint to check ZIP file contents and extraction
    """
    try:
        # Validate file type
        if not zip_file.filename or not zip_file.filename.endswith('.zip'):
            return {
                "error": "File must be a ZIP archive",
                "filename": zip_file.filename
            }
        
        # Read ZIP file content
        content = await zip_file.read()
        
        # Process ZIP file
        zip_processor = ZipProcessor()
        processed_project = await zip_processor.process_zip_file(content, 'gpt-4')
        
        return {
            "filename": zip_file.filename,
            "file_size_bytes": len(content),
            "total_files_found": processed_project.file_count,
            "supported_files": [
                {
                    "path": f.file_path,
                    "size": f.size,
                    "extension": f.extension,
                    "content_preview": f.content[:200] + "..." if len(f.content) > 200 else f.content
                }
                for f in processed_project.supported_files[:5]  # Show first 5 files
            ],
            "unsupported_files": processed_project.unsupported_files[:10],  # Show first 10
            "total_content_length": len(processed_project.merged_content),
            "estimated_tokens": processed_project.token_count
        }
        
    except Exception as e:
        logger.error(f"Debug ZIP processing failed: {e}")
        return {
            "error": str(e),
            "filename": zip_file.filename if zip_file else "unknown"
        }

@router.post("/estimate-zip-tokens")
async def estimate_zip_tokens(
    zip_file: UploadFile = File(...),
    request_data: str = Form(...),
    current_user: User = Depends(get_current_active_user)
):
    """
    Estimate token requirements for ZIP file analysis without performing AI analysis.
    Similar to TestWorthy Pro's token estimation feature.
    """
    import json
    
    try:
        # Parse request data
        request_dict = json.loads(request_data)
        token_request = TokenEstimationRequest(**request_dict)
        
        # Validate file type
        if not zip_file.filename or not zip_file.filename.endswith('.zip'):
            return {
                "success": False,
                "error_message": "File must be a ZIP archive",
                "filename": zip_file.filename,
                "estimated_tokens": 0,
                "file_count": 0
            }
        
        # Read ZIP file content
        content = await zip_file.read()
        
        # Process ZIP file to get content for token calculation
        zip_processor = ZipProcessor()
        processed_project = await zip_processor.process_zip_file(content, token_request.ai_model)
        
        # Calculate actual token count using TokenManager
        token_manager = get_token_manager()
        estimated_input_tokens = token_manager.count_tokens(
            processed_project.merged_content, 
            token_request.ai_model
        )
        
        # Get model configuration for output token limits
        model_config = token_manager.MODEL_CONFIGS.get(token_request.ai_model)
        max_output_tokens = model_config.max_output_tokens if model_config else 8192
        
        # Validate token budget
        validation = token_manager.validate_request(
            estimated_input_tokens, 
            token_request.ai_model, 
            estimated_output_tokens=max_output_tokens
        )
        
        # Calculate estimated cost (if available in model config)
        estimated_cost = None
        warnings = []
        
        if model_config:
            input_cost = (estimated_input_tokens / 1000) * model_config.cost_per_input_token
            output_cost = (max_output_tokens / 1000) * model_config.cost_per_output_token
            estimated_cost = round(input_cost + output_cost, 6)
        
        # Add warnings for large files or token counts
        if estimated_input_tokens > 100000:
            warnings.append("Large token count may result in longer processing times")
        
        if processed_project.file_count > 100:
            warnings.append("Many files detected - processing may take extra time")
            
        if len(processed_project.unsupported_files) > 0:
            warnings.append(f"{len(processed_project.unsupported_files)} unsupported files were skipped")
        
        # Return response format matching frontend expectations
        return {
            "success": True,
            "estimated_tokens": estimated_input_tokens,
            "file_count": processed_project.file_count,
            "estimated_cost": estimated_cost,
            "warnings": warnings if warnings else None,
            "filename": zip_file.filename,
            # Additional details for debugging/logging
            "file_analysis": {
                "total_files": processed_project.file_count,
                "supported_files": len(processed_project.supported_files),
                "unsupported_files": len(processed_project.unsupported_files),
                "total_size_bytes": processed_project.total_size,
                "merged_content_length": len(processed_project.merged_content)
            },
            "token_estimation": {
                "estimated_input_tokens": estimated_input_tokens,
                "max_output_tokens": max_output_tokens,
                "total_estimated_tokens": estimated_input_tokens + max_output_tokens,
                "model": token_request.ai_model,
                "provider": token_request.ai_provider
            },
            "budget_validation": {
                "valid": validation['valid'],
                "message": validation.get('message', 'Token budget OK'),
                "estimated_cost_usd": estimated_cost
            }
        }
        
    except json.JSONDecodeError:
        return {
            "success": False,
            "error_message": "Invalid JSON in request_data",
            "filename": zip_file.filename if zip_file else "unknown",
            "estimated_tokens": 0,
            "file_count": 0
        }
    except Exception as e:
        logger.error(f"Token estimation failed: {e}")
        return {
            "success": False,
            "error_message": str(e),
            "filename": zip_file.filename if zip_file else "unknown",
            "estimated_tokens": 0,
            "file_count": 0
        }

@router.post("/estimate-tokens")
async def estimate_tokens(
    request_data: dict,
    current_user: User = Depends(get_current_active_user)
):
    """
    Estimate tokens for direct text input from frontend ZIP parsing.
    This endpoint accepts compiled file contents and returns accurate token counts.
    """
    try:
        # Extract request parameters
        text_content = request_data.get("text_content")
        ai_provider = request_data.get("ai_provider")
        ai_model = request_data.get("ai_model")
        file_count = request_data.get("file_count")
        total_size_kb = request_data.get("total_size_kb")
        target_test_cases = request_data.get("target_test_cases")
        project_name = request_data.get("project_name")

        logger.info(f"Token estimation request: provider={ai_provider}, model={ai_model}, "
                   f"content_size={total_size_kb}KB, files={file_count}")

        # Validate required parameters
        if not text_content:
            return {
                "success": False,
                "error_message": "No text content provided for token estimation",
                "estimated_tokens": 0,
                "file_count": file_count or 0
            }

        if not ai_provider:
            return {
                "success": False,
                "error_message": "AI provider is required",
                "estimated_tokens": 0,
                "file_count": file_count or 0
            }

        if not ai_model:
            return {
                "success": False,
                "error_message": "AI model is required",
                "estimated_tokens": 0,
                "file_count": file_count or 0
            }

        # Map model names to their tokenizer type (for accurate token counting)
        # Vertex AI can use models from different families, so we detect by model name
        def get_tokenizer_provider(provider: str, model: str) -> str:
            """Get the provider type for token counting based on model name"""
            model_lower = model.lower()
            
            # Vertex AI can use various models - detect by model name
            if provider == "vertex":
                if "gpt" in model_lower or "o1" in model_lower:
                    return "openai"
                elif "claude" in model_lower:
                    return "anthropic"
                elif "gemini" in model_lower:
                    return "gemini"
                elif "llama" in model_lower:
                    return "llama"
                else:
                    # Default to gemini for vertex if unclear
                    return "gemini"
            elif provider == "llama":
                return "llama"
            
            # OpenRouter can also use various models
            elif provider == "openrouter":
                if "gpt" in model_lower or "o1" in model_lower:
                    return "openai"
                elif "claude" in model_lower:
                    return "anthropic"
                elif "gemini" in model_lower:
                    return "gemini"
                else:
                    return "openai"  # Default for openrouter
            
            # Direct providers
            return provider
        
        # Validate provider is supported
        if ai_provider not in ["openai", "anthropic", "gemini", "vertex", "openrouter", "llama", "anthropic_vertex"]:
            return {
                "success": False,
                "error_message": f"Unsupported AI provider: {ai_provider}",
                "estimated_tokens": 0,
                "file_count": file_count or 0
            }

        if file_count is None:
            return {
                "success": False,
                "error_message": "File count is required",
                "estimated_tokens": 0,
                "file_count": 0
            }

        if total_size_kb is None:
            return {
                "success": False,
                "error_message": "Total size in KB is required",
                "estimated_tokens": 0,
                "file_count": file_count
            }

        if target_test_cases is None:
            return {
                "success": False,
                "error_message": "Target test cases count is required",
                "estimated_tokens": 0,
                "file_count": file_count
            }

        if not project_name:
            return {
                "success": False,
                "error_message": "Project name is required",
                "estimated_tokens": 0,
                "file_count": file_count
            }

        # Get token manager for accurate token counting
        token_manager = get_token_manager()
        
        # Get the correct tokenizer type based on model
        tokenizer_provider = get_tokenizer_provider(ai_provider, ai_model)
        
        # Count tokens in the compiled text content using the appropriate tokenizer
        input_tokens = token_manager.count_tokens(text_content, ai_model)
        
        # Calculate additional tokens for AI processing
        # Base system prompt (instructions for test generation)
        system_prompt_tokens = 300
        
        # Output tokens estimation based on target test cases
        # Assume ~150 tokens per test case (including structure, assertions, etc.)
        estimated_output_tokens = target_test_cases * 150
        
        # Add overhead for JSON structure, metadata, etc.
        response_overhead_tokens = 200
        
        total_estimated_tokens = input_tokens + system_prompt_tokens + estimated_output_tokens + response_overhead_tokens
        
        # Calculate cost estimation using PricingManager
        pricing_manager = PricingManager()
        input_cost, output_cost, total_estimated_cost = pricing_manager.calculate_cost(
            input_tokens + system_prompt_tokens,
            estimated_output_tokens + response_overhead_tokens,
            ai_provider,
            ai_model
        )
        
        # Estimate processing time (rough approximation)
        # Base time + time per 1000 tokens
        base_processing_time = 30  # seconds
        time_per_1k_tokens = 2  # seconds per 1000 tokens
        estimated_processing_seconds = base_processing_time + (total_estimated_tokens * time_per_1k_tokens / 1000)
        estimated_processing_minutes = max(1, round(estimated_processing_seconds / 60))
        
        # Generate warnings based on analysis
        warnings = []
        
        # Model context window warnings
        model_context_limits = {
            # OpenAI models
            "gpt-4": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4.1-mini": 128000,
            "gpt-3.5-turbo": 16385,
            "o1": 200000,
            "o1-mini": 128000,
            # Anthropic Claude models (direct and Vertex AI)
            "claude-3-5-sonnet-v2@20241022": 200000,
            "claude-3-5-sonnet@20240620": 200000,
            "claude-3-5-haiku@20241022": 200000,
            "claude-3-opus@20240229": 200000,
            "claude-3-sonnet@20240229": 200000,
            "claude-3-haiku@20240307": 200000,
            "claude-opus-4.5": 200000,
            "claude-sonnet-4.5": 200000,
            "claude-haiku-4.5": 200000,
            "claude-opus-4.1": 200000,
            "claude-opus-4": 200000,
            "claude-sonnet-4": 200000,
            "claude-3.5-haiku": 200000,
            "claude-3-haiku": 200000,
            "claude-3-haiku": 200000,
            "claude-3-sonnet": 200000,
            "claude-3-opus": 200000,
            "claude-3-5-sonnet": 200000,
            "claude-3-5-haiku": 200000,
            # Google Gemini models (direct and Vertex AI) - Official Google specs
            "gemini-3-pro": 1048576,
            "gemini-3-pro-preview": 1048576,
            "gemini-3-flash": 1048576,
            "gemini-3-flash-preview": 1048576,
            "gemini-2.5-pro": 1048576,
            "gemini-2.5-flash": 1000000,
            "gemini-2.5-flash-lite": 1000000,
            "gemini-pro": 32760,
            "gemini-1.5-pro": 2000000,
            "gemini-1.5-flash": 1000000,
            "gemini-2.0-flash": 1000000,
            "gemini-1.5-pro-002": 2000000,
            "gemini-1.5-flash-002": 1000000,
            # Llama 4 models via Vertex AI  
            "publishers/meta/models/llama-4-maverick-17b-128e-instruct-maas": 419430,  # 419K tokens
            "publishers/meta/models/llama-4-scout-17b-16e-instruct-maas": 1310720,     # 1.31M tokens
        }
        
        context_limit = model_context_limits.get(ai_model, 128000)
        if total_estimated_tokens > context_limit:
            warnings.append(f"Content exceeds model context window ({context_limit:,} tokens). Consider breaking into chunks.")
        elif total_estimated_tokens > context_limit * 0.8:
            warnings.append(f"Content is near model context limit ({context_limit:,} tokens). Monitor for truncation.")
        
        # Cost warnings
        if total_estimated_cost > 10.0:
            warnings.append(f"High estimated cost: ${total_estimated_cost:.2f}. Consider using a smaller model.")
        elif total_estimated_cost > 5.0:
            warnings.append(f"Moderate cost estimation: ${total_estimated_cost:.2f}")
        
        # Processing time warnings
        if estimated_processing_minutes > 30:
            warnings.append(f"Long processing time estimated: {estimated_processing_minutes} minutes")
        
        # File count warnings
        if file_count > 100:
            warnings.append(f"Large number of files ({file_count}). Processing may take longer.")
        
        # Content size warnings
        if total_size_kb > 1000:  # > 1MB
            warnings.append(f"Large codebase ({total_size_kb}KB). Consider processing in batches.")

        return {
            "success": True,
            "estimated_tokens": total_estimated_tokens,
            "input_tokens": input_tokens,
            "output_tokens": estimated_output_tokens + response_overhead_tokens,
            "system_tokens": system_prompt_tokens,
            "estimated_cost": round(total_estimated_cost, 4),
            "input_cost": round(input_cost, 4),
            "output_cost": round(output_cost, 4),
            "estimated_processing_time_minutes": estimated_processing_minutes,
            "file_count": file_count,
            "total_size_kb": total_size_kb,
            "target_test_cases": target_test_cases,
            "model_info": {
                "provider": ai_provider,
                "model": ai_model,
                "context_limit": context_limit,
                "utilization_percent": round((total_estimated_tokens / context_limit) * 100, 1)
            },
            "analysis": {
                "content_length": len(text_content),
                "lines_of_code": text_content.count('\n'),
                "files_analyzed": file_count,
                "average_tokens_per_file": round(input_tokens / max(1, file_count), 1)
            },
            "warnings": warnings if warnings else None,
            "breakdown": {
                "input_content": input_tokens,
                "system_prompt": system_prompt_tokens,
                "estimated_output": estimated_output_tokens,
                "response_overhead": response_overhead_tokens,
                "total": total_estimated_tokens
            }
        }
        
    except Exception as e:
        logger.error(f"Token estimation failed: {e}")
        return {
            "success": False,
            "error_message": str(e),
            "estimated_tokens": 0,
            "file_count": request_data.get("file_count", 0)
        }

@router.post("/test-chunking")
async def test_chunking(
    current_user: User = Depends(get_current_active_user)
):
    """
    Test endpoint to verify chunking functionality with sample data
    """
    try:
        logger.info("Testing chunking functionality")
        
        # Sample code content for testing
        sample_content = """// File: main.py
def calculate_fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib_sequence = [0, 1]
    for i in range(2, n):
        fib_sequence.append(fib_sequence[i-1] + fib_sequence[i-2])
    
    return fib_sequence

def main():
    n = 10
    result = calculate_fibonacci(n)
    # Fibonacci test function

if __name__ == "__main__":
    main()

// File: utils.py
def validate_input(value):
    if not isinstance(value, int):
        raise ValueError("Input must be an integer")
    if value < 0:
        raise ValueError("Input must be non-negative")
    return True

def format_output(sequence):
    return ", ".join(map(str, sequence))

class FibonacciCalculator:
    def __init__(self):
        self.cache = {}
    
    def calculate(self, n):
        if n in self.cache:
            return self.cache[n]
        
        if n <= 0:
            result = []
        elif n == 1:
            result = [0]
        elif n == 2:
            result = [0, 1]
        else:
            result = self.calculate(n-1)
            if len(result) >= 2:
                result.append(result[-1] + result[-2])
        
        self.cache[n] = result
        return result
"""
        
        # Test chunk processor
        chunk_processor = EnhancedChunkProcessor()
        
        # Test with small model for guaranteed chunking
        test_model = "gemini-2.5-flash"
        
        # Force chunking by using small available tokens
        original_context_limits = chunk_processor.context_limits.copy()
        chunk_processor.context_limits[test_model] = 1000  # Force chunking
        
        chunk_results = chunk_processor.prepare_chunks_with_overlap(
            content=sample_content,
            model=test_model,
            overlap_percentage=0.2
        )
        
        # Restore original limits
        chunk_processor.context_limits = original_context_limits
        
        # Format response
        chunks = []
        for chunk_result in chunk_results:
            chunks.append({
                "chunk_id": chunk_result.chunk_id,
                "token_count": chunk_result.token_count,
                "start_line": chunk_result.start_line,
                "end_line": chunk_result.end_line,
                "overlap_start_lines": chunk_result.overlap_start,
                "overlap_end_lines": chunk_result.overlap_end,
                "content_preview": chunk_result.content[:100] + "..." if len(chunk_result.content) > 100 else chunk_result.content
            })
        
        return {
            "success": True,
            "test_type": "chunking_functionality",
            "sample_content_length": len(sample_content),
            "total_chunks": len(chunks),
            "chunks": chunks,
            "model_tested": test_model,
            "overlap_percentage": 20,
            "message": "Chunking functionality test completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Chunking test failed: {e}", exc_info=True)
        return {
            "success": False,
            "error_message": str(e),
            "test_type": "chunking_functionality"
        }

@router.post("/prepare-chunks")
async def prepare_chunks(
    request_data: dict,
    current_user: User = Depends(get_current_active_user)
):
    """
    Prepare content for chunked processing with 20% overlap and 80% context window usage.
    Returns chunks or indicates if direct processing is possible.
    """
    try:
        logger.info(f"Preparing chunks for user {current_user.email}")
        
        # Extract request parameters
        text_content = request_data.get("text_content")
        ai_provider = request_data.get("ai_provider")
        ai_model = request_data.get("ai_model")
        project_name = request_data.get("project_name")
        
        logger.info(f"Request params: provider={ai_provider}, model={ai_model}, project={project_name}, content_length={len(text_content) if text_content else 0}")
        
        # Validate required parameters
        if not text_content:
            return {"success": False, "error_message": "Text content is required"}
        if not ai_provider or not ai_model:
            return {"success": False, "error_message": "AI provider and model are required"}
        if not project_name:
            return {"success": False, "error_message": "Project name is required"}
        
        # Initialize enhanced chunk processor
        logger.info("Initializing enhanced chunk processor")
        chunk_processor = EnhancedChunkProcessor()
        
        # Get token count for original content
        logger.info(f"Counting tokens for content with model {ai_model}")
        total_tokens = chunk_processor.token_manager.count_tokens(text_content, ai_model)
        
        # Get context limit for model (already includes 80% buffer)
        context_limit = chunk_processor.context_limits.get(ai_model, 80000)
        system_prompt_tokens = 1000
        response_tokens = 8000
        available_tokens = context_limit - system_prompt_tokens - response_tokens
        
        logger.info(f"Token analysis: total={total_tokens}, limit={context_limit}, available={available_tokens}")
        
        # Check if chunking is needed
        needs_chunking = total_tokens > available_tokens
        
        if not needs_chunking:
            # Track chunking preparation
            analytics.track_event(
                "ai_chunk_preparation_completed",
                user_id=current_user.id,
                properties={
                    "project_name": project_name,
                    "ai_provider": ai_provider,
                    "ai_model": ai_model,
                    "needs_chunking": False,
                    "content_tokens": total_tokens,
                    "available_tokens": available_tokens
                }
            )
            
            return {
                "success": True,
                "needs_chunking": False,
                "content_tokens": total_tokens,
                "available_tokens": available_tokens,
                "context_limit": context_limit,
                "chunks": [],
                "overlap_percentage": 0,
                "context_buffer_percentage": 20,
                "recommendation": "Content fits in context window. Direct processing recommended."
            }
        
        # Create chunks with 20% overlap
        logger.info(f"Creating chunks with 20% overlap for {ai_model}")
        chunk_results = chunk_processor.prepare_chunks_with_overlap(
            content=text_content,
            model=ai_model,
            overlap_percentage=0.2
        )
        
        logger.info(f"Created {len(chunk_results)} chunks")
        
        # Convert chunk results to API response format
        chunks = []
        for chunk_result in chunk_results:
            chunk_data = {
                "chunk_id": chunk_result.chunk_id,
                "content": chunk_result.content,
                "token_count": chunk_result.token_count,
                "start_line": chunk_result.start_line,
                "end_line": chunk_result.end_line,
                "line_count": chunk_result.end_line - chunk_result.start_line + 1,
                "content_preview": chunk_result.content[:200] + "..." if len(chunk_result.content) > 200 else chunk_result.content
            }
            
            # Add overlap information if present
            if chunk_result.overlap_start is not None:
                chunk_data["overlap_start_lines"] = chunk_result.overlap_start
            if chunk_result.overlap_end is not None:
                chunk_data["overlap_end_lines"] = chunk_result.overlap_end
            
            chunks.append(chunk_data)
            logger.debug(f"Chunk {chunk_result.chunk_id}: {chunk_result.token_count} tokens, lines {chunk_result.start_line}-{chunk_result.end_line}")
        
        logger.info(f"Successfully prepared {len(chunks)} chunks for processing")
        
        # Track chunking preparation
        analytics.track_event(
            "ai_chunk_preparation_completed",
            user_id=current_user.id,
            properties={
                "project_name": project_name,
                "ai_provider": ai_provider,
                "ai_model": ai_model,
                "needs_chunking": True,
                "content_tokens": total_tokens,
                "available_tokens": available_tokens,
                "total_chunks": len(chunks),
                "overlap_percentage": 20
            }
        )
        
        return {
            "success": True,
            "needs_chunking": True,
            "content_tokens": total_tokens,
            "available_tokens": available_tokens,
            "context_limit": context_limit,
            "total_chunks": len(chunks),
            "chunks": chunks,
            "overlap_percentage": 20,
            "context_buffer_percentage": 20,
            "recommendation": f"Content requires {len(chunks)} chunks with 20% overlap for processing.",
            "processing_info": {
                "model": ai_model,
                "provider": ai_provider,
                "chunks_created": len(chunks),
                "total_content_lines": len(text_content.split('\n')),
                "average_chunk_tokens": sum(chunk["token_count"] for chunk in chunks) // len(chunks) if chunks else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Enhanced chunk preparation failed: {e}", exc_info=True)
        return {
            "success": False, 
            "error_message": str(e),
            "error_type": type(e).__name__,
            "debugging_info": {
                "model": ai_model,
                "provider": ai_provider,
                "content_length": len(text_content) if text_content else 0
            }
        }

@router.post("/process-chunk")
async def process_chunk(
    request_data: dict,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    token_manager: TokenManager = Depends(get_token_manager)
):
    """
    Process a single chunk using any supported AI provider to generate test cases.
    Supports: OpenAI, Gemini, Vertex AI, Anthropic, and OpenRouter.
    Enforces per-user token quota limits before processing.
    """
    try:
        chunk_content = request_data.get("chunk_content")
        chunk_id = request_data.get("chunk_id")
        total_chunks = request_data.get("total_chunks")
        ai_provider = request_data.get("ai_provider")
        ai_model = request_data.get("ai_model")
        api_key = request_data.get("api_key")
        project_name = request_data.get("project_name")
        project_id = request_data.get("project_id")  # Get project_id for usage tracking
        start_line = request_data.get("start_line", 0)
        end_line = request_data.get("end_line", 0)
        minimum_test_cases = request_data.get("minimum_test_cases", 10)
        
        # Validate required parameters
        if not chunk_content:
            return {"success": False, "error_message": "Chunk content is required"}
        if not ai_provider or not ai_model:
            return {"success": False, "error_message": "AI provider and model are required"}
        if not project_name:
            return {"success": False, "error_message": "Project name is required"}
        

        
        # Handle API key from environment if placeholder is sent
        # api_key_source = "user"  # Default to user-provided
        if not api_key or api_key == "CONFIGURED_API_KEY":
            api_key_source = "server"
            from app.core.config import settings
            
            # Map provider to environment variable
            if ai_provider.lower() == "gemini":
                api_key = settings.GEMINI_API_KEY
            elif ai_provider.lower() == "openai":
                api_key = getattr(settings, "OPENAI_API_KEY", "")
            elif ai_provider.lower() == "anthropic":
                api_key = getattr(settings, "ANTHROPIC_API_KEY", "")
            elif ai_provider.lower() == "vertex":
                # Vertex AI uses service account
                if not settings.GOOGLE_APPLICATION_CREDENTIALS or not settings.GCP_PROJECT_ID:
                    return {
                        "success": False,
                        "error_message": "Vertex AI requires GOOGLE_APPLICATION_CREDENTIALS and GCP_PROJECT_ID"
                    }
                api_key = "vertex-configured"  # Placeholder
            elif ai_provider.lower() == "openrouter":
                api_key = getattr(settings, "OPENROUTER_API_KEY", "")
            elif ai_provider.lower() == "llama":
                # Llama uses Vertex AI infrastructure
                if not settings.GOOGLE_APPLICATION_CREDENTIALS or not settings.GCP_PROJECT_ID:
                    return {
                        "success": False,
                        "error_message": "Llama via Vertex AI requires GOOGLE_APPLICATION_CREDENTIALS and GCP_PROJECT_ID"
                    }
                api_key = "llama-vertex-configured"  # Placeholder
            elif ai_provider.lower() == "anthropic_vertex":
                if not settings.GOOGLE_APPLICATION_CREDENTIALS or not settings.GCP_PROJECT_ID:
                    return {
                        "success": False,
                        "error_message": "Anthropic via Vertex AI requires GOOGLE_APPLICATION_CREDENTIALS and GCP_PROJECT_ID"
                    }
                api_key = "anthropic-vertex-configured"  # Placeholder
            elif ai_provider.lower() == "llama_openrouter":
                api_key = getattr(settings, "OPENROUTER_API_KEY", "")
            
            if not api_key:
                return {
                    "success": False, 
                    "error_message": f"API key for {ai_provider} not configured in environment"
                }
            
            logger.info(f"Using configured API key from environment for {ai_provider}")
        
        # Validate provider
        supported_providers = ["openai", "gemini", "vertex", "anthropic", "openrouter", "llama", "anthropic_vertex", "llama_openrouter"]
        if ai_provider.lower() not in supported_providers:
            return {
                "success": False, 
                "error_message": f"Unsupported AI provider: {ai_provider}. Supported: {', '.join(supported_providers)}"
            }
        
        # Estimate token usage and check quota (added 2026-01-01)
        estimated_input_tokens = len(chunk_content) // 4  # Rough estimate: 1 token ≈ 4 chars
        max_output_tokens = minimum_test_cases * 500  # Estimate: ~500 tokens per test case
        total_estimated_tokens = estimated_input_tokens + max_output_tokens
        
        from app.services.user_quota_manager import check_user_quota
        quota_check = await check_user_quota(
            user_id=current_user.id,
            estimated_tokens=total_estimated_tokens,
            db=db,
            token_manager=token_manager
        )
        
        if not quota_check['allowed']:
            return {
                "success": False,
                "error_message": f"Token quota exceeded. Monthly remaining: {quota_check['remaining_monthly']:,}, "
                                f"Purchased balance: {quota_check['purchased_balance']:,}. "
                                f"Required: {total_estimated_tokens:,} tokens. "
                                f"Quota resets on 1st of next month."
            }
        
        # Return mock data if MOCK_MODE is enabled
        if MOCK_MODE:
            return {
                "success": True,
                "chunk_id": 1,
                "total_chunks": 1,
                "test_cases": [
                    {
                        "title": "Verify relsenderreceiverid retrieval with direct match",
                        "description": "Ensure the system retrieves the correct relsenderreceiverid when a direct match exists for sender and receiver IDs in the hierarchy data.",
                        "preconditions": "Hierarchy data file exists with a direct match for the specified sender and receiver IDs.",
                        "steps": [
                            "Provide valid sender ID",
                            "Provide valid receiver ID",
                            "Specify the path to the hierarchy data file",
                            "Execute the relsenderreceiverid retrieval process"
                        ],
                        "expected_results": "The system returns the correct relsenderreceiverid associated with the provided sender and receiver IDs.",
                        "priority": "high",
                        "type": "functional",
                        "functional_area": "Hierarchy Resolution",
                        "chunk_id": 1,
                        "total_chunks": 1,
                        "source_lines": "0-145"
                    },
                    {
                        "title": "Validate relsenderreceiverid retrieval through parent-child traversal",
                        "description": "Verify the system correctly traverses the hierarchy to find the relsenderreceiverid when a direct match is not found, but a parent-child relationship exists.",
                        "preconditions": "Hierarchy data file exists with a parent-child relationship between sender and receiver IDs, but no direct match.",
                        "steps": [
                            "Provide valid sender ID",
                            "Provide valid receiver ID",
                            "Specify the path to the hierarchy data file",
                            "Execute the relsenderreceiverid retrieval process"
                        ],
                        "expected_results": "The system returns the correct relsenderreceiverid associated with the parent sender and receiver IDs.",
                        "priority": "high",
                        "type": "functional",
                        "functional_area": "Hierarchy Resolution",
                        "chunk_id": 1,
                        "total_chunks": 1,
                        "source_lines": "0-145"
                    },
                    {
                        "title": "Ensure 'all dealers/senders' match is found when no other match exists",
                        "description": "Confirm that the system correctly identifies and returns the relsenderreceiverid when the sender ID is 'all dealers/senders' and matches the receiver ID.",
                        "preconditions": "Hierarchy data file exists with an entry for 'all dealers/senders' and the specified receiver ID, and no other matching sender ID.",
                        "steps": [
                            "Provide 'all dealers/senders' as the sender ID",
                            "Provide valid receiver ID",
                            "Specify the path to the hierarchy data file",
                            "Execute the relsenderreceiverid retrieval process"
                        ],
                        "expected_results": "The system returns the relsenderreceiverid associated with 'all dealers/senders' and the provided receiver ID.",
                        "priority": "medium",
                        "type": "functional",
                        "functional_area": "Hierarchy Resolution",
                        "chunk_id": 1,
                        "total_chunks": 1,
                        "source_lines": "0-145"
                    },
                    {
                        "title": "Verify no relsenderreceiverid is returned when no match is found",
                        "description": "Ensure the system returns None when no direct match, parent-child relationship, or 'all dealers/senders' match is found in the hierarchy data.",
                        "preconditions": "Hierarchy data file exists, but contains no matching sender ID, receiver ID, parent-child relationship, or 'all dealers/senders' entry.",
                        "steps": [
                            "Provide a non-existent sender ID",
                            "Provide a non-existent receiver ID",
                            "Specify the path to the hierarchy data file",
                            "Execute the relsenderreceiverid retrieval process"
                        ],
                        "expected_results": "The system returns None, indicating no matching relsenderreceiverid was found.",
                        "priority": "high",
                        "type": "functional",
                        "functional_area": "Hierarchy Resolution",
                        "chunk_id": 1,
                        "total_chunks": 1,
                        "source_lines": "0-145"
                    },
                    {
                        "title": "Validate error handling for invalid hierarchy data file",
                        "description": "Verify the system handles errors gracefully when provided with an invalid hierarchy data file (e.g., incorrect JSON format).",
                        "preconditions": "An invalid JSON file exists.",
                        "steps": [
                            "Provide a valid sender ID",
                            "Provide a valid receiver ID",
                            "Specify the path to the invalid JSON file",
                            "Execute the relsenderreceiverid retrieval process"
                        ],
                        "expected_results": "The system catches the exception, prints an error message, and returns None.",
                        "priority": "high",
                        "type": "functional",
                        "functional_area": "Hierarchy Resolution",
                        "chunk_id": 1,
                        "total_chunks": 1,
                        "source_lines": "0-145"
                    },
                    {
                        "title": "Ensure correct relsenderreceiverid is returned when parent sender is 0",
                        "description": "Verify the system correctly handles cases where the 'parent sender / Dealer' value is 0, indicating no parent.",
                        "preconditions": "Hierarchy data file exists with a sender ID that has a 'parent sender / Dealer' value of 0.",
                        "steps": [
                            "Provide the sender ID with a parent sender of 0",
                            "Provide a valid receiver ID",
                            "Specify the path to the hierarchy data file",
                            "Execute the relsenderreceiverid retrieval process"
                        ],
                        "expected_results": "The system returns the relsenderreceiverid associated with the provided sender and receiver IDs, without attempting to traverse to a non-existent parent.",
                        "priority": "medium",
                        "type": "functional",
                        "functional_area": "Hierarchy Resolution",
                        "chunk_id": 1,
                        "total_chunks": 1,
                        "source_lines": "0-145"
                    },
                    {
                        "title": "Validate that hierarchy traversal stops when a sender is visited twice",
                        "description": "Ensure that the hierarchy traversal algorithm correctly detects and prevents infinite loops by stopping when a sender ID is visited more than once.",
                        "preconditions": "Hierarchy data file exists with a circular parent-child relationship that would cause an infinite loop if not detected.",
                        "steps": [
                            "Provide a sender ID that is part of a circular parent-child relationship",
                            "Provide a valid receiver ID",
                            "Specify the path to the hierarchy data file",
                            "Execute the relsenderreceiverid retrieval process"
                        ],
                        "expected_results": "The system returns None, indicating that no matching relsenderreceiverid was found and the traversal stopped to prevent an infinite loop.",
                        "priority": "medium",
                        "type": "functional",
                        "functional_area": "Hierarchy Resolution",
                        "chunk_id": 1,
                        "total_chunks": 1,
                        "source_lines": "0-145"
                    },
                    {
                        "title": "Verify relsenderreceiverid retrieval with numeric sender and receiver IDs",
                        "description": "Ensure the system correctly retrieves the relsenderreceiverid when sender and receiver IDs are numeric strings.",
                        "preconditions": "Hierarchy data file exists with numeric sender and receiver IDs.",
                        "steps": [
                            "Provide a numeric sender ID",
                            "Provide a numeric receiver ID",
                            "Specify the path to the hierarchy data file",
                            "Execute the relsenderreceiverid retrieval process"
                        ],
                        "expected_results": "The system returns the correct relsenderreceiverid associated with the provided numeric sender and receiver IDs.",
                        "priority": "medium",
                        "type": "functional",
                        "functional_area": "Hierarchy Resolution",
                        "chunk_id": 1,
                        "total_chunks": 1,
                        "source_lines": "0-145"
                    },
                    {
                        "title": "Validate handling of missing 'parent sender / Dealer' field",
                        "description": "Verify the system handles cases where the 'parent sender / Dealer' field is missing from a record in the hierarchy data.",
                        "preconditions": "Hierarchy data file exists with a record where the 'parent sender / Dealer' field is missing.",
                        "steps": [
                            "Provide the sender ID from the record with the missing 'parent sender / Dealer' field",
                            "Provide a valid receiver ID",
                            "Specify the path to the hierarchy data file",
                            "Execute the relsenderreceiverid retrieval process"
                        ],
                        "expected_results": "The system returns None if no direct match is found, and does not attempt to traverse to a non-existent parent.",
                        "priority": "medium",
                        "type": "functional",
                        "functional_area": "Hierarchy Resolution",
                        "chunk_id": 1,
                        "total_chunks": 1,
                        "source_lines": "0-145"
                    },
                    {
                        "title": "Ensure correct relsenderreceiverid is returned when result is null in direct match",
                        "description": "Verify the system correctly traverses the hierarchy to find the relsenderreceiverid when a direct match exists, but the relsenderreceiverid is null.",
                        "preconditions": "Hierarchy data file exists with a direct match for the specified sender and receiver IDs, but the relsenderreceiverid is null.",
                        "steps": [
                            "Provide valid sender ID",
                            "Provide valid receiver ID",
                            "Specify the path to the hierarchy data file",
                            "Execute the relsenderreceiverid retrieval process"
                        ],
                        "expected_results": "The system traverses the hierarchy and returns the correct relsenderreceiverid associated with the parent sender and receiver IDs, or None if no match is found.",
                        "priority": "medium",
                        "type": "functional",
                        "functional_area": "Hierarchy Resolution",
                        "chunk_id": 1,
                        "total_chunks": 1,
                        "source_lines": "0-145"
                    }
                ],
                "chunk_summary": "Generated 10 test cases from chunk 1",
                "processing_time_seconds": 14.3157377243042,
                "error_message": None,
                "token_count": 1662,
                "provider": "gemini",
                "model": "gemini-2.0-flash",
                "api_key_source": "server"
            }
        
        # Initialize enhanced chunk processor
        chunk_processor = EnhancedChunkProcessor()
        
        # Create a chunk result object for processing
        from ..services.chunk_processor import ChunkResult
        chunk_obj = ChunkResult(
            chunk_id=chunk_id,
            content=chunk_content,
            token_count=chunk_processor.token_manager.count_tokens(chunk_content, ai_model),
            start_line=start_line,
            end_line=end_line
        )
        
        # Process the chunk with the specified AI provider
        logger.info(f"Processing chunk {chunk_id}/{total_chunks} with {ai_provider}/{ai_model}")
        
        result = await chunk_processor.process_chunk_with_ai(
            chunk=chunk_obj,
            total_chunks=total_chunks,
            project_name=project_name,
            provider=ai_provider,
            model=ai_model,
            api_key=api_key,
            minimum_test_cases=minimum_test_cases
        )
        
        # Track chunk processing
        analytics.track_event(
            "ai_chunk_processing_completed",
            user_id=current_user.id,
            properties={
                "project_name": project_name,
                "ai_provider": ai_provider,
                "ai_model": ai_model,
                "chunk_id": chunk_id,
                "total_chunks": total_chunks,
                "success": result.success,
                "test_cases_generated": len(result.test_cases) if result.test_cases else 0,
                "processing_time_seconds": result.processing_time,
                "token_count": chunk_obj.token_count,
                "api_key_source": api_key_source
            }
        )
        
        # Record token usage to user's quota (added 2026-01-01)
        if result.success:
            from app.services.user_quota_manager import record_user_usage
            # Use actual token counts from result (from Vertex AI response)
            estimated_input_tokens = chunk_obj.token_count
            actual_input_tokens = result.input_tokens if result.input_tokens > 0 else chunk_obj.token_count
            actual_output_tokens = result.output_tokens if result.output_tokens > 0 else (len(str(result.test_cases)) // 4 if result.test_cases else max_output_tokens // 2)
            
            # Debug logging for token tracking
            logger.info(f"🔢 TOKEN TRACKING DEBUG - process-chunk:")
            logger.info(f"   Estimated Input: {estimated_input_tokens:,} tokens")
            logger.info(f"   Actual Input:    {actual_input_tokens:,} tokens")
            logger.info(f"   Actual Output:   {actual_output_tokens:,} tokens")
            logger.info(f"   Total:           {actual_input_tokens + actual_output_tokens:,} tokens")
            print("=" * 80)
            print(f"🔢 TOKEN TRACKING - process-chunk")
            print(f"   Estimated Input: {estimated_input_tokens:,} tokens")
            print(f"   Actual Input:    {actual_input_tokens:,} tokens")
            print(f"   Actual Output:   {actual_output_tokens:,} tokens")
            print(f"   Total Recorded:  {actual_input_tokens + actual_output_tokens:,} tokens")
            print("=" * 80)
            
            await record_user_usage(
                user_id=current_user.id,
                project_id=project_id,
                operation_type="chunk_processing",
                input_tokens=actual_input_tokens,
                output_tokens=actual_output_tokens,
                model_name=ai_model,
                db=db,
                token_manager=token_manager
            )
        
        return {
            "success": result.success,
            "chunk_id": chunk_id,
            "total_chunks": total_chunks,
            "test_cases": result.test_cases,
            "chunk_summary": result.chunk_summary,
            "processing_time_seconds": result.processing_time,
            "error_message": result.error_message,
            "token_count": chunk_obj.token_count,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.total_tokens,
            "provider": ai_provider,
            "model": ai_model,
            "api_key_source": api_key_source
        }
        
    except Exception as e:
        logger.error(f"Chunk processing failed: {e}", exc_info=True)
        return {
            "success": False, 
            "error_message": str(e), 
            "chunk_id": request_data.get("chunk_id")
        }

@router.post("/aggregate-chunks")
async def aggregate_chunks(
    request_data: dict,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    token_manager: TokenManager = Depends(get_token_manager)
):
    """
    Aggregate results from all processed chunks using enhanced aggregation logic with LLM deduplication.
    Enforces per-user token quota limits before processing.
    """
    try:
        chunk_results = request_data.get("chunk_results", [])
        project_name = request_data.get("project_name")
        project_id = request_data.get("project_id")  # Get project_id for usage tracking
        ai_provider = request_data.get("ai_provider", "gemini")
        ai_model = request_data.get("ai_model", "gemini-2.5-flash")
        api_key = request_data.get("api_key", "")
        use_deduplication = request_data.get("use_deduplication", True)
        
        logger.info(f"Aggregate request received - Provider: {ai_provider}, Model: {ai_model}, API Key provided: {bool(api_key)}")
        
        if not chunk_results:
            return {"success": False, "error_message": "Chunk results are required"}
        if not project_name:
            return {"success": False, "error_message": "Project name is required"}
        
        # Estimate token usage and check quota (added 2026-01-01)
        all_chunks_text = " ".join([str(chunk) for chunk in chunk_results])
        estimated_input_tokens = len(all_chunks_text) // 4
        max_output_tokens = len(chunk_results) * 1000  # Estimate: ~1000 tokens for aggregation output
        total_estimated_tokens = estimated_input_tokens + max_output_tokens
        
        from app.services.user_quota_manager import check_user_quota
        quota_check = await check_user_quota(
            user_id=current_user.id,
            estimated_tokens=total_estimated_tokens,
            db=db,
            token_manager=token_manager
        )
        
        if not quota_check['allowed']:
            return {
                "success": False,
                "error_message": f"Token quota exceeded. Monthly remaining: {quota_check['remaining_monthly']:,}, "
                                f"Purchased balance: {quota_check['purchased_balance']:,}. "
                                f"Required: {total_estimated_tokens:,} tokens. "
                                f"Quota resets on 1st of next month."
            }
        
        # Return mock data if MOCK_MODE is enabled
        if MOCK_MODE:
            return {
                "success": True,
                "project_structure": {
                    "milestones": [
                        {
                            "name": "Hierarchy Resolution - Initial Development",
                            "status": "completed"
                        },
                        {
                            "name": "Hierarchy Resolution - Testing Phase",
                            "status": "in_progress"
                        },
                        {
                            "name": "Hierarchy Resolution - Release Candidate",
                            "status": "planned"
                        }
                    ],
                    "test_plans": [
                        {
                            "name": "Functional Test Plan",
                            "type": "functional",
                            "description": "Covers all functional aspects of the hierarchy resolution logic.",
                            "test_case_count": 9
                        },
                        {
                            "name": "Error Handling Test Plan",
                            "type": "error_handling",
                            "description": "Focuses on validating error handling and edge cases.",
                            "test_case_count": 1
                        }
                    ],
                    "test_suites": [
                        {
                            "name": "Hierarchy Resolution Functionality",
                            "description": "Test suite for validating the core functionality of relsenderreceiverid retrieval.",
                            "test_count": 9,
                            "sections": [
                                {
                                    "name": "Direct Match",
                                    "suite_name": "Hierarchy Resolution Functionality",
                                    "description": "Tests for direct matches between sender and receiver IDs.",
                                    "display_order": 0,
                                    "parent_section_name": None,
                                    "depth": 0,
                                    "test_case_titles": [
                                        "Verify relsenderreceiverid retrieval with direct match",
                                        "Verify relsenderreceiverid retrieval with numeric sender and receiver IDs"
                                    ]
                                },
                                {
                                    "name": "Hierarchy Traversal",
                                    "suite_name": "Hierarchy Resolution Functionality",
                                    "description": "Tests for relsenderreceiverid retrieval through parent-child traversal.",
                                    "display_order": 1,
                                    "parent_section_name": None,
                                    "depth": 0,
                                    "test_case_titles": [
                                        "Validate relsenderreceiverid retrieval through parent-child traversal",
                                        "Ensure correct relsenderreceiverid is returned when parent sender is 0",
                                        "Validate that hierarchy traversal stops when a sender is visited twice",
                                        "Validate handling of missing 'parent sender / Dealer' field",
                                        "Ensure correct relsenderreceiverid is returned when result is None in direct match"
                                    ]
                                },
                                {
                                    "name": "\"All Dealers/Senders\" Match",
                                    "suite_name": "Hierarchy Resolution Functionality",
                                    "description": "Tests for the 'all dealers/senders' matching logic.",
                                    "display_order": 2,
                                    "parent_section_name": None,
                                    "depth": 0,
                                    "test_case_titles": [
                                        "Ensure 'all dealers/senders' match is found when no other match exists"
                                    ]
                                },
                                {
                                    "name": "No Match",
                                    "suite_name": "Hierarchy Resolution Functionality",
                                    "description": "Tests for scenarios where no matching relsenderreceiverid is found.",
                                    "display_order": 3,
                                    "parent_section_name": None,
                                    "depth": 0,
                                    "test_case_titles": [
                                        "Verify no relsenderreceiverid is returned when no match is found"
                                    ]
                                }
                            ]
                        },
                        {
                            "name": "Error Handling",
                            "description": "Test suite for validating error handling scenarios.",
                            "test_count": 1,
                            "sections": [
                                {
                                    "name": "Invalid Data File",
                                    "suite_name": "Error Handling",
                                    "description": "Tests for handling invalid hierarchy data files.",
                                    "display_order": 0,
                                    "parent_section_name": None,
                                    "depth": 0,
                                    "test_case_titles": [
                                        "Validate error handling for invalid hierarchy data file"
                                    ]
                                }
                            ]
                        }
                    ]
                },
                "statistics": {
                    "total_chunks_processed": 1,
                    "successful_chunks": 1,
                    "failed_chunks": 0,
                    "total_test_cases": 10,
                    "total_test_suites": 2,
                    "total_sections": 5,
                    "total_test_plans": 2,
                    "total_milestones": 3,
                    "total_processing_time_seconds": 14.3157377243042,
                    "priority_distribution": {
                        "high": 4,
                        "medium": 6
                    },
                    "type_distribution": {
                        "functional": 10
                    }
                },
                "processing_summary": {
                    "chunks_with_errors": [],
                    "avg_processing_time_per_chunk": 14.3157377243042,
                    "overlap_percentage": 20,
                    "context_buffer_percentage": 20,
                    "deduplication_applied": True,
                    "test_cases_removed": 0,
                    "deduplication_summary": "No test cases were removed as all test different aspects of the hierarchy resolution functionality."
                },
                "provider": "gemini",
                "model": "gemini-2.0-flash",
                "api_key_source": "server"
            }
        
        # Initialize enhanced chunk processor
        chunk_processor = EnhancedChunkProcessor()
        
        # Convert chunk results to ProcessingResult objects
        from ..services.chunk_processor import ProcessingResult
        processing_results = []
        
        for chunk_result in chunk_results:
            result = ProcessingResult(
                success=chunk_result.get("success", False),
                test_cases=chunk_result.get("test_cases", []),
                chunk_summary=chunk_result.get("chunk_summary", ""),
                processing_time=chunk_result.get("processing_time_seconds", 0),
                error_message=chunk_result.get("error_message")
            )
            processing_results.append(result)
        
        # Handle API key from environment if placeholder is sent
        api_key_source = "user"  # Default to user-provided
        if not api_key or api_key == "CONFIGURED_API_KEY":
            api_key_source = "server"
            from app.core.config import settings
            
            # Map provider to environment variable
            if ai_provider.lower() == "gemini":
                api_key = settings.GEMINI_API_KEY
            elif ai_provider.lower() == "openai":
                api_key = getattr(settings, "OPENAI_API_KEY", "")
            elif ai_provider.lower() == "anthropic":
                api_key = getattr(settings, "ANTHROPIC_API_KEY", "")
            elif ai_provider.lower() == "vertex":
                # Vertex AI uses service account
                if not settings.GOOGLE_APPLICATION_CREDENTIALS or not settings.GCP_PROJECT_ID:
                    api_key = None
                else:
                    api_key = "vertex-configured"  # Placeholder
            elif ai_provider.lower() == "openrouter":
                api_key = getattr(settings, "OPENROUTER_API_KEY", "")
            elif ai_provider.lower() == "llama":
                # Llama uses Vertex AI infrastructure
                if not settings.GOOGLE_APPLICATION_CREDENTIALS or not settings.GCP_PROJECT_ID:
                    api_key = None
                else:
                    api_key = "llama-vertex-configured"  # Placeholder
            
            if not api_key:
                logger.warning(f"API key for {ai_provider} not configured, deduplication will be skipped")
                use_deduplication = False
            else:
                logger.info(f"Using configured API key from environment for {ai_provider}")
        
        # Aggregate using enhanced logic with optional LLM deduplication
        aggregated_result = await chunk_processor.aggregate_chunk_results(
            chunk_results=processing_results,
            project_name=project_name,
            use_llm_deduplication=use_deduplication,
            provider=ai_provider,
            model=ai_model,
            api_key=api_key
        )
        
        # Record actual token usage (added 2026-01-01)
        actual_input_tokens = aggregated_result.get("input_tokens", estimated_input_tokens)
        actual_output_tokens = aggregated_result.get("output_tokens", max_output_tokens)
        
        # Debug logging for token tracking
        logger.info(f"🔢 TOKEN TRACKING DEBUG - aggregate-chunks:")
        logger.info(f"   Estimated Input: {estimated_input_tokens:,} tokens")
        logger.info(f"   Actual Input:    {actual_input_tokens:,} tokens")
        logger.info(f"   Actual Output:   {actual_output_tokens:,} tokens")
        logger.info(f"   Total:           {actual_input_tokens + actual_output_tokens:,} tokens")
        print("=" * 80)
        print(f"🔢 TOKEN TRACKING - aggregate-chunks")
        print(f"   Estimated Input: {estimated_input_tokens:,} tokens")
        print(f"   Actual Input:    {actual_input_tokens:,} tokens")
        print(f"   Actual Output:   {actual_output_tokens:,} tokens")
        print(f"   Total Recorded:  {actual_input_tokens + actual_output_tokens:,} tokens")
        print("=" * 80)
        
        from app.services.user_quota_manager import record_user_usage
        await record_user_usage(
            user_id=current_user.id,
            project_id=project_id,
            operation_type="aggregate_chunks",
            input_tokens=actual_input_tokens,
            output_tokens=actual_output_tokens,
            model_name=ai_model,
            db=db,
            token_manager=token_manager
        )
        
        # Add provider, model, and api_key_source to response
        aggregated_result["provider"] = ai_provider
        aggregated_result["model"] = ai_model
        aggregated_result["api_key_source"] = api_key_source
        
        # Track chunk aggregation
        analytics.track_event(
            "ai_chunk_aggregation_completed",
            user_id=current_user.id,
            properties={
                "project_name": project_name,
                "ai_provider": ai_provider,
                "ai_model": ai_model,
                "chunks_aggregated": len(chunk_results),
                "total_test_cases": aggregated_result.get("total_test_cases", 0),
                "duplicates_removed": aggregated_result.get("duplicates_removed", 0),
                "use_deduplication": use_deduplication,
                "api_key_source": api_key_source
            }
        )
        
        return aggregated_result
        
    except Exception as e:
        logger.error(f"Enhanced chunk aggregation failed: {e}")
        return {"success": False, "error_message": str(e)}

@router.post("/import-project-from-zip", response_model=ProjectImportResponse)
async def import_project_from_zip(
    zip_file: UploadFile = File(...),
    request_data: str = Form(...),  # JSON string of ProjectFromZipRequest
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(require_permission(Permission.CREATE)),
    db: Session = Depends(get_db)
):
    """
    Complete project creation from ZIP file using AI analysis
    This endpoint combines all AI services for comprehensive project import
    """
    import json
    
    try:
        # Parse request data
        request_dict = json.loads(request_data)
        project_request = ProjectFromZipRequest(**request_dict)
        
        # Validate file type
        if not zip_file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="File must be a ZIP archive")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
            content = await zip_file.read()
            temp_file.write(content)
            temp_zip_path = temp_file.name
        
        try:
            # Initialize services
            rate_limiter = await get_rate_limiter()
            token_manager = get_token_manager()
            
            # Create AI provider config
            ai_provider_enum = AIProviderEnum(project_request.ai_provider.lower())
            
            # Get model-specific token limits from token manager
            token_manager_instance = get_token_manager()
            model_config = token_manager_instance.get_model_config(project_request.ai_model)
            max_output_tokens = model_config.max_output_tokens if model_config else 8192
            
            ai_config = AIProviderConfig(
                provider=ai_provider_enum,
                api_key=project_request.api_key,
                model=project_request.ai_model,
                context_window=128000,  # Default context window
                temperature=0.7,
                max_tokens=max_output_tokens  # Use model-specific limit
            )
            
            # Process ZIP file to get actual content for token calculation
            zip_processor = ZipProcessor()
            processed_project = await zip_processor.process_zip_file(content, project_request.ai_model)
            
            # Calculate actual token count using TokenManager
            estimated_tokens = token_manager.count_tokens(
                processed_project.merged_content, 
                project_request.ai_model
            )
            
            # Validate token budget for the request
            validation = token_manager.validate_request(
                estimated_tokens, 
                project_request.ai_model, 
                estimated_output_tokens=max_output_tokens
            )
            
            if not validation['valid']:
                raise HTTPException(
                    status_code=429, 
                    detail=f"Token budget exceeded: {validation['message']} (Estimated: {estimated_tokens:,} tokens)"
                )
            
            # Check user's token quota before proceeding (added 2026-01-01)
            from app.services.user_quota_manager import check_user_quota
            quota_check = await check_user_quota(
                user_id=current_user.id,
                estimated_tokens=estimated_tokens + max_output_tokens,  # Include expected output
                db=db,
                token_manager=token_manager
            )
            
            if not quota_check['allowed']:
                return {
                    "success": False,
                    "error_message": f"Token quota exceeded. Monthly remaining: {quota_check['remaining_monthly']:,}, "
                                    f"Purchased balance: {quota_check['purchased_balance']:,}. "
                                    f"Required: {estimated_tokens + max_output_tokens:,} tokens. "
                                    f"Quota resets on 1st of next month."
                }
            
            # Perform AI analysis using already processed project
            code_analysis_service = CodeAnalysisService()
            
            # Create analysis config
            from app.services.code_analysis import AnalysisConfig
            analysis_config = AnalysisConfig(
                ai_config=ai_config,
                minimum_target_use_cases=project_request.minimum_test_cases
            )
            
            # Perform comprehensive analysis using processed project content
            analysis_result = await code_analysis_service.analyze_code_text(
                processed_project.merged_content,
                analysis_config,
                f"{zip_file.filename} (processed ZIP with {processed_project.file_count} files)"
            )
            
            if not analysis_result.success:
                raise HTTPException(
                    status_code=400,
                    detail=f"Analysis failed: {analysis_result.error_message}"
                )
            
            # Create project in database
            project = Project(
                name=project_request.project_name,
                description=f"AI-generated project from {zip_file.filename}",
                created_by_id=current_user.id,
                ai_generated=True,
                ai_aggregate_response=analysis_result.project_structure
            )
            db.add(project)
            db.commit()
            db.refresh(project)
            
            # Create project entities from analysis
            created_entities = await _create_project_entities_from_analysis(
                project.id,
                analysis_result.project_structure,
                current_user.id,
                db
            )
            
            # Record token usage to user's quota (updated 2026-01-01)
            from app.services.user_quota_manager import record_user_usage
            actual_input_tokens = analysis_result.analysis_metadata.get('input_tokens', estimated_tokens)
            actual_output_tokens = analysis_result.analysis_metadata.get('output_tokens', max_output_tokens // 2)
            
            await record_user_usage(
                user_id=current_user.id,
                project_id=project.id,
                operation_type="project_import",
                input_tokens=actual_input_tokens,
                output_tokens=actual_output_tokens,
                model_name=project_request.ai_model,
                db=db,
                token_manager=token_manager
            )
            
            return ProjectImportResponse(
                success=True,
                project_id=project.id,
                analysis_results=analysis_result.analysis_metadata,
                generation_results=analysis_result.project_structure.__dict__ if analysis_result.project_structure else None,
                created_entities=created_entities
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_zip_path):
                os.unlink(temp_zip_path)
                
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Project import failed: {e}\nFull traceback:\n{error_details}")
        return ProjectImportResponse(
            success=False,
            error_message=f"{str(e)}\n\nTraceback:\n{error_details}"
        )

@router.post("/analyze-code-from-zip")
async def analyze_code_from_zip(
    zip_file: UploadFile = File(...),
    request_data: str = Form(...),  # JSON string of CodeAnalysisRequest
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Analyze code from ZIP file without creating a full project
    Returns analysis results and suggestions
    """
    import json
    
    try:
        # Parse request data
        request_dict = json.loads(request_data)
        analysis_request = CodeAnalysisRequest(**request_dict)
        
        # Validate file type
        if not zip_file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="File must be a ZIP archive")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
            content = await zip_file.read()
            temp_file.write(content)
            temp_zip_path = temp_file.name
        
        try:
            # Create AI provider config
            ai_provider_enum = AIProviderEnum(analysis_request.ai_provider.lower())
            ai_config = AIProviderConfig(
                provider=ai_provider_enum,
                api_key=analysis_request.api_key,
                model=analysis_request.ai_model,
                context_window=128000,  # Default context window
                temperature=0.7,
                max_tokens=4096
            )
            
            # Read ZIP file content
            with open(temp_zip_path, 'rb') as f:
                zip_content = f.read()
            
            # Create analysis config  
            from app.services.code_analysis import AnalysisConfig
            analysis_config = AnalysisConfig(
                ai_config=ai_config,
                minimum_target_use_cases=analysis_request.minimum_test_cases
            )
            
            # Perform analysis
            code_analysis_service = CodeAnalysisService()
            analysis_result = await code_analysis_service.analyze_zip_file(
                zip_content,
                analysis_config
            )
            
            if not analysis_result.success:
                raise HTTPException(
                    status_code=400,
                    detail=f"Analysis failed: {analysis_result.error_message}"
                )
            
            # Return analysis without creating project
            return {
                "success": True,
                "processed_project": analysis_result.processed_project.__dict__ if analysis_result.processed_project else None,
                "project_structure": analysis_result.project_structure.__dict__ if analysis_result.project_structure else None,
                "analysis_metadata": analysis_result.analysis_metadata,
                "processing_time": analysis_result.processing_time
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_zip_path):
                os.unlink(temp_zip_path)
                
    except Exception as e:
        logger.error(f"Code analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-test-structure", response_model=Dict[str, Any])
async def generate_test_structure_from_requirements(
    request: TestGenerationRequest,
    current_user: User = Depends(require_permission(Permission.CREATE)),
    db: Session = Depends(get_db)
):
    """
    Generate comprehensive test structure from requirements using AI
    Creates milestones, test plans, test suites, test cases, and test runs
    """
    try:
        # Create AI provider config
        ai_provider_enum = AIProviderEnum(request.ai_provider.lower())
        ai_config = AIProviderConfig(
            provider=ai_provider_enum,
            api_key=request.api_key,
            model=request.ai_model,
            context_window=128000,  # Default context window
            temperature=0.7,
            max_tokens=4096
        )
        
        # Create test generation config
        test_config = TestGenerationConfig(
            ai_config=ai_config,
            minimum_test_cases=request.minimum_test_cases,
            maximum_test_cases=request.maximum_test_cases,
            include_edge_cases=request.include_edge_cases,
            include_negative_tests=request.include_negative_tests,
            include_security_tests=request.include_security_tests,
            priority_distribution=request.priority_distribution,
            type_distribution=request.type_distribution
        )
        
        # Validate token limits
        token_manager = get_token_manager()
        estimated_tokens = len(request.requirements) * 2  # Conservative estimate
        validation = token_manager.validate_request(
            estimated_tokens,
            request.ai_model,
            estimated_output_tokens=15000
        )
        
        if not validation['valid']:
            raise HTTPException(
                status_code=429,
                detail=f"Request exceeds limits: {validation['reason']}"
            )
        
        # Generate test structure
        generation_result = await TestCaseGenerator.generate_complete_project_structure(
            request.requirements,
            test_config,
            request.project_name,
            is_code_analysis=False
        )
        
        if not generation_result.success:
            raise HTTPException(
                status_code=400,
                detail=f"Test generation failed: {generation_result.error_message}"
            )
        
        # Record token usage (estimate based on response size)
        output_tokens = len(str(generation_result.__dict__)) // 4  # Rough estimate
        token_manager.record_usage(
            input_tokens=estimated_tokens,
            output_tokens=output_tokens,
            model_name=request.ai_model
        )
        
        return {
            "success": True,
            "milestones": [m.__dict__ for m in generation_result.milestones],
            "test_plans": [tp.__dict__ for tp in generation_result.test_plans],
            "test_suites": [ts.__dict__ for ts in generation_result.test_suites],
            "test_cases": [tc.__dict__ for tc in generation_result.test_cases],
            "test_runs": [tr.__dict__ for tr in generation_result.test_runs],
            "metadata": generation_result.generation_metadata
        }
        
    except Exception as e:
        logger.error(f"Test structure generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rate-limit-status")
async def get_rate_limit_status(
    current_user: User = Depends(get_current_active_user)
):
    """Get current rate limiting status for all AI providers"""
    try:
        rate_limiter = await get_rate_limiter()
        return rate_limiter.get_all_status()
    except Exception as e:
        logger.error(f"Failed to get rate limit status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/token-usage-status")
async def get_token_usage_status(
    provider: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Get current token usage status"""
    try:
        token_manager = get_token_manager()
        
        if provider:
            provider_enum = TMAIProvider(provider.lower())
            return token_manager.get_usage_stats(provider_enum)
        else:
            return {
                provider.value: token_manager.get_usage_stats(provider)
                for provider in TMAIProvider
            }
            
    except Exception as e:
        logger.error(f"Failed to get token usage status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/supported-models")
async def get_supported_models(
    provider: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Get list of supported AI models"""
    try:
        token_manager = get_token_manager()
        
        if provider:
            provider_enum = TMAIProvider(provider.lower())
            models = token_manager.list_available_models(provider_enum)
            return {
                "provider": provider,
                "models": [
                    {
                        "name": model,
                        "details": token_manager.get_model_info(model)
                    }
                    for model in models
                ]
            }
        else:
            all_models = {}
            for provider_enum in TMAIProvider:
                models = token_manager.list_available_models(provider_enum)
                all_models[provider_enum.value] = [
                    {
                        "name": model,
                        "details": token_manager.get_model_info(model)
                    }
                    for model in models
                ]
            return all_models
            
    except Exception as e:
        logger.error(f"Failed to get supported models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _convert_aggregate_format_to_dataclass(project_structure_dict: Dict) -> Dict:
    """
    Convert aggregate response format (with suite_name references) to dataclass format (with suite_id)
    This modifies sections in place to use suite_id and parent_id instead of name references.
    Also maps test cases to their sections based on test_case_titles in each section.
    """
    from app.services.project_import_processor import Section as SectionDataclass
    
    logger.info(f"Converting aggregate format - found {len(project_structure_dict.get('sections', []))} sections and {len(project_structure_dict.get('test_cases', []))} test cases")
    
    # If no sections, return as-is
    if 'sections' not in project_structure_dict:
        logger.warning("No sections found in project structure, returning as-is")
        return project_structure_dict
    
    # Already converted if sections are SectionDataclass instances
    if project_structure_dict.get('sections') and isinstance(project_structure_dict['sections'][0], SectionDataclass):
        return project_structure_dict
    
    # Create mapping of suite names to generated IDs
    suite_name_to_id = {}
    if 'test_suites' in project_structure_dict:
        for idx, suite in enumerate(project_structure_dict.get('test_suites', [])):
            suite_id = f"suite-{idx+1}"
            suite_name_to_id[suite.get('name', '')] = suite_id
    
    # Convert sections from name-based references to ID-based
    # Also build mapping of test case title → section ID
    converted_sections = []
    section_name_to_id = {}
    test_case_title_to_section_id = {}
    
    for idx, section in enumerate(project_structure_dict.get('sections', [])):
        section_id = f"section-{idx+1}"
        section_name = section.get('name', '')
        section_name_to_id[section_name] = section_id
        
        suite_name = section.get('suite_name', '')
        suite_id = suite_name_to_id.get(suite_name, 'suite-1')
        
        parent_section_name = section.get('parent_section_name')
        parent_id = section_name_to_id.get(parent_section_name) if parent_section_name else None
        
        # Map test cases to this section
        for test_case_title in section.get('test_case_titles', []):
            test_case_title_to_section_id[test_case_title] = section_id
        
        converted_sections.append(SectionDataclass(
            id=section_id,
            name=section_name,
            suite_id=suite_id,
            description=section.get('description'),
            display_order=section.get('display_order', 0),
            parent_id=parent_id,
            depth=section.get('depth', 0)
        ))
    
    # Build mapping of section_id → suite_id
    section_id_to_suite_id = {}
    for section in converted_sections:
        section_id_to_suite_id[section.id] = section.suite_id
    
    logger.info(f"Built mappings - {len(test_case_title_to_section_id)} test case titles mapped to sections, {len(section_id_to_suite_id)} sections mapped to suites")
    
    # Update test cases with section_id and suite_id based on their title
    updated_test_cases = []
    for test_case in project_structure_dict.get('test_cases', []):
        test_case_dict = test_case if isinstance(test_case, dict) else test_case.__dict__
        test_case_title = test_case_dict.get('title', '')
        section_id = test_case_title_to_section_id.get(test_case_title)
        
        # Get suite_id from the section this test case belongs to
        suite_id = section_id_to_suite_id.get(section_id, 'suite-1') if section_id else test_case_dict.get('suite_id', 'suite-1')
        
        # Create updated test case dict with section_id and suite_id
        updated_case = test_case_dict.copy()
        updated_case['section_id'] = section_id
        updated_case['suite_id'] = suite_id
        updated_test_cases.append(updated_case)
        
        if not section_id:
            logger.warning(f"Test case '{test_case_title}' not found in any section's test_case_titles")
    
    logger.info(f"Updated {len(updated_test_cases)} test cases with section_id and suite_id mappings")
    
    # Return modified dict with converted sections and updated test cases
    result = project_structure_dict.copy()
    result['sections'] = converted_sections
    result['test_cases'] = updated_test_cases
    return result

async def _create_project_entities_from_analysis(
    project_id: int,
    project_structure: Any,
    created_by_id: int,
    db: Session
) -> Dict[str, int]:
    """
    Helper function to create database entities from AI analysis results
    """
    # Convert aggregate format to dataclass format if needed
    project_structure = _convert_aggregate_format_to_dataclass(project_structure)
    
    # Handle both dict and dataclass formats
    def get_attr(obj, key, default=[]):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
    
    created_entities = {
        "milestones": 0,
        "test_plans": 0,
        "test_suites": 0,
        "sections": 0,
        "test_cases": 0,
        "test_runs": 0
    }
    
    try:
        # Create milestones
        milestone_map = {}
        for milestone_data in get_attr(project_structure, 'milestones'):
            milestone = Milestone(
                name=milestone_data.name,
                description=milestone_data.description,
                due_date=milestone_data.due_date,
                status=milestone_data.status,
                project_id=project_id,
                created_by_id=created_by_id
            )
            db.add(milestone)
            db.commit()
            db.refresh(milestone)
            milestone_map[milestone_data.id] = milestone.id
            created_entities["milestones"] += 1
        
        # Create test suites
        suite_map = {}
        for suite_data in get_attr(project_structure, 'test_suites'):
            test_suite = TestSuite(
                name=suite_data.name if hasattr(suite_data, 'name') else suite_data.get('name'),
                description=suite_data.description if hasattr(suite_data, 'description') else suite_data.get('description'),
                project_id=project_id,
                created_by_id=created_by_id
            )
            db.add(test_suite)
            db.commit()
            db.refresh(test_suite)
            suite_id_key = suite_data.id if hasattr(suite_data, 'id') else suite_data.get('id')
            suite_map[suite_id_key] = test_suite.id
            created_entities["test_suites"] += 1
        
        # Create sections within test suites
        section_map = {}
        for section_data in get_attr(project_structure, 'sections'):
            suite_id_key = section_data.suite_id if hasattr(section_data, 'suite_id') else section_data.get('suite_id')
            suite_id = suite_map.get(suite_id_key)
            
            parent_id_key = section_data.parent_id if hasattr(section_data, 'parent_id') else section_data.get('parent_id')
            parent_id = section_map.get(parent_id_key) if parent_id_key else None
            
            section = Section(
                name=section_data.name if hasattr(section_data, 'name') else section_data.get('name'),
                suite_id=suite_id,
                description=section_data.description if hasattr(section_data, 'description') else section_data.get('description'),
                display_order=section_data.display_order if hasattr(section_data, 'display_order') else section_data.get('display_order', 0),
                parent_id=parent_id,
                depth=section_data.depth if hasattr(section_data, 'depth') else section_data.get('depth', 0)
            )
            db.add(section)
            db.commit()
            db.refresh(section)
            section_id_key = section_data.id if hasattr(section_data, 'id') else section_data.get('id')
            section_map[section_id_key] = section.id
            created_entities["sections"] += 1
        
        # Create test plans
        plan_map = {}
        for plan_data in get_attr(project_structure, 'test_plans'):
            milestone_id_key = plan_data.milestone_id if hasattr(plan_data, 'milestone_id') else plan_data.get('milestone_id')
            milestone_id = milestone_map.get(milestone_id_key)
            test_plan = TestPlan(
                name=plan_data.name if hasattr(plan_data, 'name') else plan_data.get('name'),
                description=plan_data.description if hasattr(plan_data, 'description') else plan_data.get('description'),
                milestone_id=milestone_id,
                project_id=project_id,
                created_by_id=created_by_id
            )
            db.add(test_plan)
            db.commit()
            db.refresh(test_plan)
            plan_id_key = plan_data.id if hasattr(plan_data, 'id') else plan_data.get('id')
            plan_map[plan_id_key] = test_plan.id
            created_entities["test_plans"] += 1
        
        # Create test cases and link to sections
        for case_data in get_attr(project_structure, 'test_cases'):
            suite_id_key = case_data.suite_id if hasattr(case_data, 'suite_id') else case_data.get('suite_id')
            suite_id = suite_map.get(suite_id_key)
            
            section_id_key = case_data.section_id if hasattr(case_data, 'section_id') else case_data.get('section_id')
            section_id = section_map.get(section_id_key) if section_id_key else None
            
            test_case = TestCase(
                title=case_data.title if hasattr(case_data, 'title') else case_data.get('title'),
                description=case_data.description if hasattr(case_data, 'description') else case_data.get('description'),
                preconditions=case_data.preconditions if hasattr(case_data, 'preconditions') else case_data.get('preconditions'),
                steps=case_data.steps if hasattr(case_data, 'steps') else case_data.get('steps'),
                expected_results=case_data.expected_results if hasattr(case_data, 'expected_results') else case_data.get('expected_results'),
                priority=case_data.priority if hasattr(case_data, 'priority') else case_data.get('priority'),
                suite_id=suite_id,
                created_by_id=created_by_id
            )
            db.add(test_case)
            db.commit()
            db.refresh(test_case)
            
            # Link test case to section via many-to-many relationship
            if section_id:
                section_test_case = SectionTestCase(
                    section_id=section_id,
                    test_case_id=test_case.id
                )
                db.add(section_test_case)
            
            created_entities["test_cases"] += 1
        
        # Create test runs
        for run_data in get_attr(project_structure, 'test_runs'):
            plan_id_key = run_data.plan_id if hasattr(run_data, 'plan_id') else run_data.get('plan_id')
            plan_id = plan_map.get(plan_id_key)
            test_run = TestRun(
                name=run_data.name if hasattr(run_data, 'name') else run_data.get('name'),
                description=run_data.description if hasattr(run_data, 'description') else run_data.get('description'),
                test_plan_id=plan_id,
                project_id=project_id,
                created_by_id=created_by_id,
                status="not_started"
            )
            db.add(test_run)
            created_entities["test_runs"] += 1
        
        # Commit all remaining changes
        db.commit()
        
        return created_entities
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create project entities: {e}")
        raise

# Original AI Provider Management endpoints (keeping existing functionality)
@router.post("/providers/", response_model=AIProviderSchema)
async def create_ai_provider(
    provider_data: AIProviderCreate,
    current_user: User = Depends(require_permission(Permission.CONFIGURE_SETTINGS)),
    db: Session = Depends(get_db)
):
    """Create a new AI provider configuration"""
    
    # Check if provider name already exists
    existing_provider = db.query(AIProvider).filter(
        AIProvider.name == provider_data.name
    ).first()
    
    if existing_provider:
        raise HTTPException(
            status_code=400,
            detail=f"AI provider '{provider_data.name}' already exists"
        )
    
    ai_provider = AIProvider(
        name=provider_data.name,
        provider_type=provider_data.provider_type,
        api_key=provider_data.api_key,
        api_url=provider_data.api_url,
        model_name=provider_data.model_name,
        max_tokens=provider_data.max_tokens,
        temperature=provider_data.temperature,
        is_active=provider_data.is_active,
        settings=provider_data.settings,
        created_by_id=current_user.id
    )
    
    db.add(ai_provider)
    db.commit()
    db.refresh(ai_provider)
    
    return ai_provider

@router.get("/providers/", response_model=List[AIProviderSchema])
async def get_ai_providers(
    active_only: bool = False,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all AI provider configurations"""
    
    query = db.query(AIProvider)
    
    if active_only:
        query = query.filter(AIProvider.is_active == True)
    
    providers = query.order_by(AIProvider.created_at).all()
    
    # Mask API keys in response
    for provider in providers:
        if provider.api_key:
            provider.api_key = "***" + provider.api_key[-4:] if len(provider.api_key) > 4 else "***"
    
    return providers

@router.get("/providers/{provider_id}", response_model=AIProviderSchema)
async def get_ai_provider(
    provider_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get a specific AI provider configuration"""
    
    provider = db.query(AIProvider).filter(AIProvider.id == provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="AI provider not found")
    
    # Mask API key in response
    if provider.api_key:
        provider.api_key = "***" + provider.api_key[-4:] if len(provider.api_key) > 4 else "***"
    
    return provider

@router.put("/providers/{provider_id}", response_model=AIProviderSchema)
async def update_ai_provider(
    provider_id: int,
    provider_data: AIProviderUpdate,
    current_user: User = Depends(require_permission(Permission.CONFIGURE_SETTINGS)),
    db: Session = Depends(get_db)
):
    """Update an AI provider configuration"""
    
    provider = db.query(AIProvider).filter(AIProvider.id == provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="AI provider not found")
    
    # Check if new name conflicts with existing providers (if name is being changed)
    if provider_data.name and provider_data.name != provider.name:
        existing_provider = db.query(AIProvider).filter(
            AIProvider.name == provider_data.name,
            AIProvider.id != provider_id
        ).first()
        
        if existing_provider:
            raise HTTPException(
                status_code=400,
                detail=f"AI provider '{provider_data.name}' already exists"
            )
    
    # Update fields
    for key, value in provider_data.dict(exclude_unset=True).items():
        setattr(provider, key, value)
    
    provider.updated_at = db.func.now()
    
    db.commit()
    db.refresh(provider)
    
    # Mask API key in response
    if provider.api_key:
        provider.api_key = "***" + provider.api_key[-4:] if len(provider.api_key) > 4 else "***"
    
    return provider

@router.delete("/providers/{provider_id}")
async def delete_ai_provider(
    provider_id: int,
    current_user: User = Depends(require_permission(Permission.CONFIGURE_SETTINGS)),
    db: Session = Depends(get_db)
):
    """Delete an AI provider configuration"""
    
    provider = db.query(AIProvider).filter(AIProvider.id == provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="AI provider not found")
    
    # Check if provider is being used in any analyses
    analysis_count = db.query(AIAnalysis).filter(AIAnalysis.provider_id == provider_id).count()
    if analysis_count > 0:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete provider: {analysis_count} AI analyses are using this provider"
        )
    
    db.delete(provider)
    db.commit()
    
    return {"message": "AI provider deleted successfully"}

# AI Analysis Management
@router.post("/analysis/", response_model=AIAnalysisSchema)
async def create_ai_analysis(
    analysis_data: AIAnalysisCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission(Permission.CREATE)),
    db: Session = Depends(get_db)
):
    """Create a new AI analysis request"""
    
    # Validate project exists
    project = db.query(Project).filter(Project.id == analysis_data.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Validate provider exists and is active
    provider = db.query(AIProvider).filter(
        AIProvider.id == analysis_data.provider_id,
        AIProvider.is_active == True
    ).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Active AI provider not found")
    
    ai_analysis = AIAnalysis(
        project_id=analysis_data.project_id,
        provider_id=analysis_data.provider_id,
        analysis_type=analysis_data.analysis_type,
        input_data=analysis_data.input_data,
        status="pending",
        created_by_id=current_user.id
    )
    
    db.add(ai_analysis)
    db.commit()
    db.refresh(ai_analysis)
    
    # Queue background task to process the analysis
    background_tasks.add_task(process_ai_analysis, ai_analysis.id, db)
    
    return ai_analysis

@router.get("/analysis/", response_model=List[AIAnalysisSchema])
async def get_ai_analyses(
    project_id: int = None,
    status: str = None,
    analysis_type: str = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get AI analyses with optional filtering"""
    
    query = db.query(AIAnalysis)
    
    if project_id:
        query = query.filter(AIAnalysis.project_id == project_id)
    
    if status:
        query = query.filter(AIAnalysis.status == status)
    
    if analysis_type:
        query = query.filter(AIAnalysis.analysis_type == analysis_type)
    
    analyses = query.order_by(AIAnalysis.created_at.desc()).offset(skip).limit(limit).all()
    return analyses

@router.get("/analysis/{analysis_id}", response_model=AIAnalysisSchema)
async def get_ai_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get a specific AI analysis"""
    
    analysis = db.query(AIAnalysis).filter(AIAnalysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="AI analysis not found")
    
    return analysis

@router.post("/analysis/{analysis_id}/retry")
async def retry_ai_analysis(
    analysis_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission(Permission.UPDATE)),
    db: Session = Depends(get_db)
):
    """Retry a failed AI analysis"""
    
    analysis = db.query(AIAnalysis).filter(AIAnalysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="AI analysis not found")
    
    if analysis.status not in ["failed", "error"]:
        raise HTTPException(
            status_code=400,
            detail="Can only retry failed analyses"
        )
    
    # Reset analysis status
    analysis.status = "pending"
    analysis.error_message = None
    analysis.retry_count = (analysis.retry_count or 0) + 1
    analysis.updated_at = db.func.now()
    
    db.commit()
    
    # Queue background task to reprocess the analysis
    background_tasks.add_task(process_ai_analysis, analysis_id, db)
    
    return {"message": "AI analysis queued for retry"}

@router.post("/generate-test-cases")
async def generate_test_cases_from_requirements(
    project_id: int,
    requirements: str,
    provider_id: int,
    test_suite_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission(Permission.CREATE)),
    db: Session = Depends(get_db)
):
    """Generate test cases from requirements using AI"""
    
    # Validate inputs
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    provider = db.query(AIProvider).filter(
        AIProvider.id == provider_id,
        AIProvider.is_active == True
    ).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Active AI provider not found")
    
    # Create analysis record
    analysis_data = {
        "requirements": requirements,
        "test_suite_id": test_suite_id,
        "generation_params": {
            "max_test_cases": 20,
            "include_edge_cases": True,
            "include_negative_tests": True
        }
    }
    
    ai_analysis = AIAnalysis(
        project_id=project_id,
        provider_id=provider_id,
        analysis_type="test_case_generation",
        input_data=analysis_data,
        status="pending",
        created_by_id=current_user.id
    )
    
    db.add(ai_analysis)
    db.commit()
    db.refresh(ai_analysis)
    
    # Queue background task
    background_tasks.add_task(generate_test_cases_task, ai_analysis.id, db)
    
    return {
        "analysis_id": ai_analysis.id,
        "message": "Test case generation started",
        "status": "pending"
    }

@router.post("/analyze-code")
async def analyze_code_quality(
    project_id: int,
    code_content: str,
    file_path: str,
    provider_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission(Permission.CREATE)),
    db: Session = Depends(get_db)
):
    """Analyze code quality and suggest test improvements"""
    
    # Validate inputs
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    provider = db.query(AIProvider).filter(
        AIProvider.id == provider_id,
        AIProvider.is_active == True
    ).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Active AI provider not found")
    
    # Create analysis record
    analysis_data = {
        "code_content": code_content,
        "file_path": file_path,
        "analysis_types": [
            "code_quality",
            "test_coverage_suggestions",
            "security_analysis",
            "performance_analysis"
        ]
    }
    
    ai_analysis = AIAnalysis(
        project_id=project_id,
        provider_id=provider_id,
        analysis_type="code_analysis",
        input_data=analysis_data,
        status="pending",
        created_by_id=current_user.id
    )
    
    db.add(ai_analysis)
    db.commit()
    db.refresh(ai_analysis)
    
    # Queue background task
    background_tasks.add_task(analyze_code_task, ai_analysis.id, db)
    
    return {
        "analysis_id": ai_analysis.id,
        "message": "Code analysis started",
        "status": "pending"
    }

# Background task functions (these would implement actual AI integration)
async def process_ai_analysis(analysis_id: int, db: Session):
    """Process AI analysis in background"""
    # This is a placeholder - implement actual AI processing logic
    pass

async def generate_test_cases_task(analysis_id: int, db: Session):
    """Generate test cases using AI in background"""
    # This is a placeholder - implement actual AI test case generation
    pass

async def analyze_code_task(analysis_id: int, db: Session):
    """Analyze code using AI in background"""
    # This is a placeholder - implement actual AI code analysis
    pass

@router.post("/heal-test-case")
async def heal_test_case(
    request_data: dict,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    token_manager: TokenManager = Depends(get_token_manager)
):
    """
    Heal/improve a test case using AI based on user feedback.
    Returns the healed test case and updated aggregate response with diff information.
    Enforces per-user token quota limits before processing.
    """
    try:
        # Extract request parameters
        project_id = request_data.get("project_id")
        test_case_id = request_data.get("test_case_id")
        user_request = request_data.get("user_request")
        ai_provider = request_data.get("ai_provider", "gemini")
        ai_model = request_data.get("ai_model", "gemini-2.0-flash")
        api_key = request_data.get("api_key", "")
        max_tokens = request_data.get("max_tokens", 16000)
        
        # LOG HEALING REQUEST DETAILS
        logger.info(f"{'='*80}")
        logger.info(f"🔧 HEALING REQUEST:")
        logger.info(f"   Project ID: {project_id}")
        logger.info(f"   Test Case ID: {test_case_id}")
        logger.info(f"   AI Provider: {ai_provider}")
        logger.info(f"   AI Model: {ai_model}")
        logger.info(f"   Max Tokens: {max_tokens}")
        logger.info(f"   User Request: {user_request[:100]}...")
        logger.info(f"{'='*80}")
        
        # Validate required parameters
        if not project_id:
            return {"success": False, "error_message": "Project ID is required"}
        if not test_case_id:
            return {"success": False, "error_message": "Test case ID is required"}
        if not user_request or not user_request.strip():
            return {"success": False, "error_message": "User request/prompt is required"}
        
        # Get project from database
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return {"success": False, "error_message": "Project not found"}
        
        # Check if project was AI-generated (has aggregate response)
        if not project.ai_generated or not project.ai_aggregate_response:
            return {
                "success": False, 
                "error_message": "This project was not created with AI. Healing is only available for AI-generated projects."
            }
        
        # Get test case from database
        test_case = db.query(TestCase).filter(
            TestCase.id == test_case_id,
            TestCase.project_id == project_id
        ).first()
        
        if not test_case:
            return {"success": False, "error_message": "Test case not found"}
        
        # Handle API key from environment if placeholder is sent
        if not api_key or api_key == "CONFIGURED_API_KEY":
            from app.core.config import settings
            
            if ai_provider.lower() == "gemini":
                api_key = settings.GEMINI_API_KEY
            elif ai_provider.lower() == "openai":
                api_key = getattr(settings, "OPENAI_API_KEY", "")
            elif ai_provider.lower() == "anthropic":
                api_key = getattr(settings, "ANTHROPIC_API_KEY", "")
            elif ai_provider.lower() == "vertex":
                # Vertex AI uses service account
                if not settings.GOOGLE_APPLICATION_CREDENTIALS or not settings.GCP_PROJECT_ID:
                    return {
                        "success": False,
                        "error_message": "Vertex AI requires GOOGLE_APPLICATION_CREDENTIALS and GCP_PROJECT_ID"
                    }
                api_key = "vertex-configured"  # Placeholder
            elif ai_provider.lower() == "openrouter":
                api_key = getattr(settings, "OPENROUTER_API_KEY", "")
            elif ai_provider.lower() == "llama":
                # Llama uses Vertex AI infrastructure
                if not settings.GOOGLE_APPLICATION_CREDENTIALS or not settings.GCP_PROJECT_ID:
                    return {
                        "success": False,
                        "error_message": "Llama via Vertex AI requires GOOGLE_APPLICATION_CREDENTIALS and GCP_PROJECT_ID"
                    }
                api_key = "llama-vertex-configured"  # Placeholder
            
            if not api_key:
                return {
                    "success": False,
                    "error_message": f"API key for {ai_provider} not configured"
                }
        
        # Prepare test case JSON
        test_case_dict = {
            "id": test_case.id,
            "case_number": test_case.case_number,
            "title": test_case.title,
            "description": test_case.description,
            "preconditions": test_case.preconditions,
            "steps": test_case.steps,
            "expected_results": test_case.expected_results,
            "priority": test_case.priority.value if test_case.priority else "medium",
            "type": test_case.type.value if test_case.type else "functional",
            "functional_area": getattr(test_case, 'functional_area', 'General')
        }
        
        # Load healing prompt from configuration
        from app.services.chunk_processor import AI_PROMPTS
        healing_prompts = AI_PROMPTS.get('test_case_healing', {})
        combined_prompt_template = healing_prompts.get('combined_prompt', '')
        
        if not combined_prompt_template:
            return {"success": False, "error_message": "Healing prompt not configured"}
        
        # Format prompt with actual values
        combined_prompt = combined_prompt_template.format(
            project_name=project.name,
            user_request=user_request,
            test_case_json=json.dumps(test_case_dict, indent=2),
            aggregate_response=json.dumps(project.ai_aggregate_response, indent=2)
        )
        
        # Estimate token usage and check quota (added 2026-01-01)
        estimated_input_tokens = len(combined_prompt) // 4
        estimated_output_tokens = max_tokens
        total_estimated_tokens = estimated_input_tokens + estimated_output_tokens
        
        from app.services.user_quota_manager import check_user_quota
        quota_check = await check_user_quota(
            user_id=current_user.id,
            estimated_tokens=total_estimated_tokens,
            db=db,
            token_manager=token_manager
        )
        
        if not quota_check['allowed']:
            return {
                "success": False,
                "error_message": f"Token quota exceeded. Monthly remaining: {quota_check['remaining_monthly']:,}, "
                                f"Purchased balance: {quota_check['purchased_balance']:,}. "
                                f"Required: {total_estimated_tokens:,} tokens. "
                                f"Quota resets on 1st of next month."
            }
        
        logger.info(f"Healing test case {test_case_id} for project {project_id} with {ai_provider}/{ai_model}")
        
        # Use AI client to generate healed response with retry logic
        from app.services.unified_ai_client import UnifiedAIClient
        from app.services.chunk_processor import retry_with_timeout, MAX_RETRIES, RETRY_TIMEOUT
        
        async def _generate_healing():
            logger.info(f"🤖 Calling {ai_provider} API with model {ai_model}...")
            
            ai_client = UnifiedAIClient(
                provider=ai_provider,
                model=ai_model,
                api_key=api_key,
                temperature=0.1,
                max_tokens=max_tokens
            )
            
            system_prompt = healing_prompts.get('system_prompt', '')
            response_text = await ai_client.generate_content(system_prompt, combined_prompt)
            
            # Capture token usage from AI client
            usage = ai_client.last_usage_metadata
            token_usage = {
                'input_tokens': usage.get('input_tokens', 0),
                'output_tokens': usage.get('output_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            }
            logger.info(f"📊 Token usage: Input={token_usage['input_tokens']}, Output={token_usage['output_tokens']}, Total={token_usage['total_tokens']}")
            
            # LOG AI RESPONSE
            logger.info(f"AI Response received. Length: {len(response_text)} characters")
            logger.info(f"Response preview: {response_text[:500]}...")
            
            # Clean and parse JSON response
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            # Parse JSON
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            
            healing_result = json.loads(response_text)
            # Add token usage to the result
            healing_result['input_tokens'] = token_usage['input_tokens']
            healing_result['output_tokens'] = token_usage['output_tokens']
            healing_result['total_tokens'] = token_usage['total_tokens']
            return healing_result
        
        # Execute with retry logic (120 second timeout per attempt for healing - AI needs time)
        try:
            healing_result = await retry_with_timeout(
                _generate_healing,
                max_retries=MAX_RETRIES,
                timeout=120.0  # 2 minute timeout for healing (AI responses can be slow)
            )
        except Exception as e:
            logger.error(f"Healing failed after retries: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            return {
                "success": False,
                "error_message": f"AI healing failed: {str(e)[:200]}"
            }
        
        # Extract results
        healed_test_case = healing_result.get('healed_test_case', {})
        aggregate_response_updates = healing_result.get('aggregate_response_updates', {})
        changes_summary = healing_result.get('changes_summary', '')
        affected_test_cases = healing_result.get('affected_test_cases', [])
        
        if not healed_test_case:
            return {"success": False, "error_message": "AI did not return healed test case"}
        
        # LOG what AI returned
        logger.info(f"🔍 AI Response Structure:")
        logger.info(f"   - healed_test_case: {bool(healed_test_case)}")
        logger.info(f"   - aggregate_response_updates: {aggregate_response_updates}")
        logger.info(f"   - updated_aggregate (fallback): {'updated_aggregate' in healing_result}")
        
        # Merge aggregate updates with original aggregate
        # Check if LLM returned new format (aggregate_response_updates) or old format (updated_aggregate)
        has_updates = aggregate_response_updates and (
            aggregate_response_updates.get('updated_test_cases') or 
            aggregate_response_updates.get('new_test_cases') or 
            aggregate_response_updates.get('deleted_test_case_titles')
        )
        
        if has_updates:
            # New format: merge partial updates with original
            updated_aggregate = _merge_aggregate_updates(
                project.ai_aggregate_response,
                aggregate_response_updates
            )
            logger.info(f"📦 Merged partial updates: {len(aggregate_response_updates.get('updated_test_cases', []))} updated, {len(aggregate_response_updates.get('new_test_cases', []))} new, {len(aggregate_response_updates.get('deleted_test_case_titles', []))} deleted")
        else:
            # Old format fallback: use full updated_aggregate if provided, otherwise copy original
            if 'updated_aggregate' in healing_result:
                updated_aggregate = healing_result['updated_aggregate']
                logger.info(f"⚠️  Using old format (full updated_aggregate)")
            else:
                # AI didn't provide aggregate updates - just update the one test case in the original
                import copy
                updated_aggregate = copy.deepcopy(project.ai_aggregate_response)
                
                logger.info(f"🔍 Looking for test case to update:")
                logger.info(f"   - DB Test Case ID: {test_case.id}")
                logger.info(f"   - DB Test Case Title: {test_case.title}")
                logger.info(f"   - Healed Test Case Title: {healed_test_case.get('title')}")
                
                # Debug the aggregate structure first
                logger.info(f"   - Aggregate structure debug:")
                logger.info(f"     - Has updated_aggregate: {updated_aggregate is not None}")
                if updated_aggregate:
                    logger.info(f"     - Aggregate keys: {list(updated_aggregate.keys())}")
                    test_suites = updated_aggregate.get('test_suites', [])
                    logger.info(f"     - Number of test_suites: {len(test_suites)}")
                    if test_suites:
                        for i, suite in enumerate(test_suites[:2]):  # Show first 2 suites
                            suite_name = suite.get('name', 'Unnamed')
                            test_cases = suite.get('test_cases', [])
                            logger.info(f"     - Suite {i+1}: '{suite_name}' has {len(test_cases)} test cases")
                            if test_cases:
                                # Show first test case structure
                                first_case = test_cases[0]
                                logger.info(f"       - First case keys: {list(first_case.keys())}")
                else:
                    logger.warning(f"     - updated_aggregate is None!")
                
                # Find and update the test case in the aggregate
                test_suites = updated_aggregate.get('test_suites', []) if updated_aggregate else []
                found = False
                total_cases_checked = 0
                
                # Handle empty or malformed aggregate structure
                if not updated_aggregate:
                    logger.warning(f"   ⚠️  Aggregate is None - cannot update")
                elif not test_suites:
                    logger.warning(f"   ⚠️  No test_suites found in aggregate")
                    # Try to find if there's a different structure
                    alt_suites = updated_aggregate.get('suites') or updated_aggregate.get('test_suite') or []
                    if alt_suites:
                        logger.info(f"   🔍 Found alternative structure with {len(alt_suites)} suites")
                        test_suites = alt_suites
                        # Normalize structure
                        updated_aggregate['test_suites'] = alt_suites
                else:
                    logger.info(f"   🔍 Processing {len(test_suites)} test suites...")
                
                for suite_idx, suite in enumerate(test_suites):
                    suite_name = suite.get('name', f'Suite_{suite_idx+1}')
                    suite_test_cases = suite.get('test_cases', [])
                    
                    # Handle different possible structures
                    if not suite_test_cases:
                        # Try alternative keys
                        suite_test_cases = (suite.get('cases') or 
                                          suite.get('testCases') or 
                                          suite.get('tests') or [])
                        if suite_test_cases:
                            suite['test_cases'] = suite_test_cases  # Normalize
                    
                    logger.info(f"   - Suite '{suite_name}': {len(suite_test_cases)} test cases")
                    
                    for i, tc in enumerate(suite_test_cases):
                        total_cases_checked += 1
                        
                        # Log first few test cases to see structure
                        if i < 3:
                            logger.info(f"   - Checking TC #{total_cases_checked}: title='{tc.get('title')}', id={tc.get('id')}, case_number={tc.get('case_number')}")
                        
                        # Improved matching strategies with more flexible comparison
                        title_match = (
                            tc.get('title') == test_case.title or 
                            tc.get('title') == healed_test_case.get('title') or
                            (tc.get('title') and test_case.title and tc.get('title').strip() == test_case.title.strip())
                        )
                        id_match = tc.get('id') == test_case.id
                        case_num_match = tc.get('case_number') == test_case.case_number
                        
                        # Also try matching by position if case_number is sequential
                        position_match = False
                        if not (title_match or id_match or case_num_match):
                            # If case number is close (within 2) and title is similar, consider it a match
                            if (tc.get('case_number') and test_case.case_number and 
                                abs(tc.get('case_number') - test_case.case_number) <= 2 and
                                tc.get('title') and test_case.title and
                                len(tc.get('title')) > 10 and len(test_case.title) > 10):
                                # Check if titles have significant overlap (fuzzy matching)
                                title1_words = set(tc.get('title').lower().split())
                                title2_words = set(test_case.title.lower().split())
                                overlap = len(title1_words.intersection(title2_words))
                                if overlap >= min(3, len(title1_words) // 2, len(title2_words) // 2):
                                    position_match = True
                        
                        if title_match or id_match or case_num_match or position_match:
                            logger.info(f"   ✅ Found match in suite '{suite_name}': title_match={title_match}, id_match={id_match}, case_num_match={case_num_match}, position_match={position_match}")
                            
                            # Update with healed version
                            suite_test_cases[i].update({
                                'id': test_case.id,  # Ensure ID is set correctly
                                'case_number': test_case.case_number,  # Ensure case number is set correctly
                                'title': healed_test_case.get('title', tc.get('title')),
                                'description': healed_test_case.get('description', tc.get('description')),
                                'preconditions': healed_test_case.get('preconditions', tc.get('preconditions')),
                                'steps': healed_test_case.get('steps', tc.get('steps')),
                                'expected_results': healed_test_case.get('expected_results', tc.get('expected_results')),
                                'priority': healed_test_case.get('priority', tc.get('priority')),
                                'type': healed_test_case.get('type', tc.get('type')),
                                'functional_area': healed_test_case.get('functional_area', tc.get('functional_area'))
                            })
                            logger.info(f"   ✏️  Updated test case '{tc.get('title')}' in suite '{suite_name}'")
                            logger.info(f"   📝 New description: {healed_test_case.get('description', 'N/A')[:100]}")
                            found = True
                            break
                    if found:
                        break
                
                if not found:
                    logger.warning(f"   ⚠️  Could not find test case to update in aggregate! Searched {total_cases_checked} cases.")
                    logger.warning(f"   📋 Target: ID={test_case.id}, case_number={test_case.case_number}, title='{test_case.title}'")
                    
                    # Initialize aggregate structure if it's missing or malformed
                    if not updated_aggregate:
                        logger.info(f"   🔧 Creating new aggregate structure")
                        updated_aggregate = {'test_suites': []}
                    
                    if not updated_aggregate.get('test_suites'):
                        logger.info(f"   🔧 Creating test_suites array in aggregate")
                        updated_aggregate['test_suites'] = []
                    
                    # Try to add the test case to the first available suite or create a new one
                    test_suites = updated_aggregate['test_suites']
                    target_suite = None
                    
                    if test_suites and len(test_suites) > 0:
                        # Use the first suite
                        target_suite = test_suites[0]
                    else:
                        # Create a new suite
                        target_suite = {
                            'name': 'General Test Cases',
                            'description': 'Test cases not found in original aggregate',
                            'test_cases': []
                        }
                        test_suites.append(target_suite)
                        logger.info(f"   ➕ Created new suite: '{target_suite['name']}'")
                    
                    # Add the healed test case to this suite
                    new_case = {
                        'id': test_case.id,
                        'case_number': test_case.case_number,
                        'title': healed_test_case.get('title', test_case.title),
                        'description': healed_test_case.get('description', test_case.description),
                        'preconditions': healed_test_case.get('preconditions', test_case.preconditions),
                        'steps': healed_test_case.get('steps', test_case.steps),
                        'expected_results': healed_test_case.get('expected_results', test_case.expected_results),
                        'priority': healed_test_case.get('priority', test_case.priority.value if test_case.priority else 'MEDIUM'),
                        'type': healed_test_case.get('type', test_case.type.value if test_case.type else 'FUNCTIONAL'),
                        'functional_area': healed_test_case.get('functional_area', '')
                    }
                    
                    if 'test_cases' not in target_suite:
                        target_suite['test_cases'] = []
                    target_suite['test_cases'].append(new_case)
                    
                    logger.info(f"   ➕ Added test case as new entry to suite '{target_suite.get('name')}'")
                    found = True
                
                if not found:
                    logger.warning(f"   ⚠️  Could not find test case to update in aggregate!")
                else:
                    logger.info(f"   ✅ Successfully updated test case in aggregate")
        
        logger.info(f"✅ Healing completed successfully. Summary: {changes_summary}")
        
        # Record actual token usage (added 2026-01-01)
        actual_input_tokens = healing_result.get("input_tokens", estimated_input_tokens)
        actual_output_tokens = healing_result.get("output_tokens", estimated_output_tokens)
        
        # Debug logging for token tracking
        logger.info(f"🔢 TOKEN TRACKING DEBUG - heal-test-case:")
        logger.info(f"   Estimated Input: {estimated_input_tokens:,} tokens")
        logger.info(f"   Actual Input:    {actual_input_tokens:,} tokens")
        logger.info(f"   Actual Output:   {actual_output_tokens:,} tokens")
        logger.info(f"   Total:           {actual_input_tokens + actual_output_tokens:,} tokens")
        print("=" * 80)
        print(f"🔢 TOKEN TRACKING - heal-test-case")
        print(f"   Estimated Input: {estimated_input_tokens:,} tokens")
        print(f"   Actual Input:    {actual_input_tokens:,} tokens")
        print(f"   Actual Output:   {actual_output_tokens:,} tokens")
        print(f"   Total Recorded:  {actual_input_tokens + actual_output_tokens:,} tokens")
        print("=" * 80)
        
        from app.services.user_quota_manager import record_user_usage
        await record_user_usage(
            user_id=current_user.id,
            project_id=project_id,
            operation_type="heal_test_case",
            input_tokens=actual_input_tokens,
            output_tokens=actual_output_tokens,
            model_name=ai_model,
            db=db,
            token_manager=token_manager
        )
        
        # Return healing results (don't save yet - user will review diff)
        return {
            "success": True,
            "healed_test_case": healed_test_case,
            "original_test_case": test_case_dict,
            "updated_aggregate": updated_aggregate,
            "original_aggregate": project.ai_aggregate_response,
            "changes_summary": changes_summary,
            "affected_test_cases": affected_test_cases,
            "provider": ai_provider,
            "model": ai_model
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse healing response: {e}")
        return {
            "success": False,
            "error_message": f"Failed to parse AI response: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Test case healing failed: {e}", exc_info=True)
        return {
            "success": False,
            "error_message": str(e)
        }

def _merge_aggregate_updates(original_aggregate: dict, updates: dict) -> dict:
    """
    Merge partial aggregate updates with the original aggregate response.
    
    Args:
        original_aggregate: Complete original aggregate response (flat structure)
        updates: Partial updates containing only affected test cases
        
    Returns:
        Complete aggregate with updates merged in
    """
    import copy
    
    # Deep copy original to avoid mutation
    merged = copy.deepcopy(original_aggregate)
    
    # Handle empty updates
    if not updates:
        return merged
    
    # Extract update lists
    updated_test_cases = updates.get('updated_test_cases', [])
    new_test_cases = updates.get('new_test_cases', [])
    deleted_titles = updates.get('deleted_test_case_titles', [])
    
    # Work with flat structure (data is stored at root level, not nested in project_structure)
    # Handle both flat and nested structures for backward compatibility
    if 'project_structure' in merged:
        # Legacy nested structure
        logger.info("📦 Processing nested structure (legacy format)")
        test_suites = merged['project_structure'].get('test_suites', [])
    else:
        # Current flat structure
        logger.info("📦 Processing flat structure (current format)")
        test_suites = merged.get('test_suites', [])
    
    # Track which updated test cases were applied
    applied_updates = set()
    
    # Process updates for each test suite
    for suite in test_suites:
        suite_test_cases = suite.get('test_cases', [])
        
        # Apply deletions
        if deleted_titles:
            original_count = len(suite_test_cases)
            suite_test_cases = [
                tc for tc in suite_test_cases 
                if tc.get('title') not in deleted_titles
            ]
            deleted_count = original_count - len(suite_test_cases)
            if deleted_count > 0:
                logger.info(f"   Deleted {deleted_count} test case(s) from suite '{suite.get('name', 'Unknown')}'")
        
        # Apply updates (match by title, ID, or case number)
        for updated_tc in updated_test_cases:
            updated_title = updated_tc.get('title')
            updated_id = updated_tc.get('id')
            updated_case_number = updated_tc.get('case_number')
            
            found = False
            for i, existing_tc in enumerate(suite_test_cases):
                # Try multiple matching strategies
                title_match = existing_tc.get('title') == updated_title
                id_match = updated_id and existing_tc.get('id') == updated_id
                case_num_match = updated_case_number and existing_tc.get('case_number') == updated_case_number
                
                # Fuzzy title matching for slight variations
                fuzzy_title_match = False
                if not title_match and existing_tc.get('title') and updated_title:
                    existing_words = set(existing_tc.get('title').lower().split())
                    updated_words = set(updated_title.lower().split())
                    overlap = len(existing_words.intersection(updated_words))
                    # If significant overlap (more than half of the smaller title)
                    min_words = min(len(existing_words), len(updated_words))
                    if min_words > 0 and overlap >= min_words * 0.7:
                        fuzzy_title_match = True
                
                if title_match or id_match or case_num_match or fuzzy_title_match:
                    # Replace with updated version, preserving important fields
                    updated_case = updated_tc.copy()
                    # Ensure we preserve the ID and case_number from the existing case if not in update
                    if not updated_case.get('id') and existing_tc.get('id'):
                        updated_case['id'] = existing_tc['id']
                    if not updated_case.get('case_number') and existing_tc.get('case_number'):
                        updated_case['case_number'] = existing_tc['case_number']
                    
                    suite_test_cases[i] = updated_case
                    applied_updates.add(updated_title)
                    found = True
                    match_type = []
                    if title_match: match_type.append("title")
                    if id_match: match_type.append("id")  
                    if case_num_match: match_type.append("case_num")
                    if fuzzy_title_match: match_type.append("fuzzy_title")
                    logger.info(f"   Updated test case '{updated_title}' in suite '{suite.get('name', 'Unknown')}' (matched by: {', '.join(match_type)})")
                    break
            
            if not found:
                # Test case not found in this suite - might need to be added
                logger.info(f"   Test case '{updated_title}' not found in suite '{suite.get('name', 'Unknown')}'")
                
                # Add it as a new case to the first suite if we can't find it anywhere
                if suite == test_suites[0]:  # Only add to first suite to avoid duplicates
                    suite_test_cases.append(updated_tc)
                    applied_updates.add(updated_title)
                    logger.info(f"   Added missing test case '{updated_title}' to suite '{suite.get('name', 'Unknown')}'")
                    found = True
        
        # Apply new test cases (append to appropriate suite based on functional_area)
        for new_tc in new_test_cases:
            new_functional_area = new_tc.get('functional_area', '')
            suite_name = suite.get('name', '')
            new_title = new_tc.get('title', 'Unknown')
            
            # Check if this test case belongs to this suite
            if new_functional_area in suite_name or suite_name in new_functional_area:
                suite_test_cases.append(new_tc)
                logger.info(f"   Added new test case '{new_title}' to suite '{suite_name}'")
        
        # Update suite test cases
        suite['test_cases'] = suite_test_cases
        suite['test_count'] = len(suite_test_cases)
    
    # Log any updates that weren't applied (might indicate mismatch)
    unapplied = set(tc.get('title') for tc in updated_test_cases) - applied_updates
    if unapplied:
        logger.warning(f"   ⚠️  Some updates were not applied (test cases not found): {unapplied}")
    
    # Update the structure based on format
    if 'project_structure' in merged:
        # Legacy nested structure
        merged['project_structure']['test_suites'] = test_suites
    else:
        # Current flat structure
        merged['test_suites'] = test_suites
    
    return merged


@router.post("/save-healed-test-case")
async def save_healed_test_case(
    request_data: dict,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Save the healed test case and updated aggregate response after user review.
    """
    try:
        project_id = request_data.get("project_id")
        test_case_id = request_data.get("test_case_id")
        healed_test_case = request_data.get("healed_test_case")
        updated_aggregate = request_data.get("updated_aggregate")
        
        logger.info(f"{'='*80}")
        logger.info(f"💾 SAVING HEALED TEST CASE:")
        logger.info(f"   Project ID: {project_id}")
        logger.info(f"   Test Case ID: {test_case_id}")
        logger.info(f"   Has healed data: {healed_test_case is not None}")
        logger.info(f"   Has updated aggregate: {updated_aggregate is not None}")
        logger.info(f"{'='*80}")
        
        # Validate
        if not all([project_id, test_case_id, healed_test_case]):
            return {"success": False, "error_message": "Missing required fields"}
        
        # Get project and test case
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return {"success": False, "error_message": "Project not found"}
        
        test_case = db.query(TestCase).filter(
            TestCase.id == test_case_id,
            TestCase.project_id == project_id
        ).first()
        
        if not test_case:
            return {"success": False, "error_message": "Test case not found"}
        
        # Log original values
        logger.info(f"📋 ORIGINAL TEST CASE:")
        logger.info(f"   Title: {test_case.title}")
        logger.info(f"   Steps: {test_case.steps}")
        
        # Create test case version BEFORE making changes
        from app.database.models import TestCaseVersion
        
        # Get current max version number for this test case
        max_tc_version = db.query(TestCaseVersion).filter(
            TestCaseVersion.case_id == test_case_id
        ).order_by(TestCaseVersion.version_number.desc()).first()
        
        next_tc_version = (max_tc_version.version_number + 1) if max_tc_version else 1
        
        logger.info(f"📝 CREATING TEST CASE VERSION {next_tc_version}")
        
        # Save current test case state as version
        tc_version = TestCaseVersion(
            case_id=test_case_id,
            version_number=next_tc_version,
            title=test_case.title,
            description=test_case.description,
            preconditions=test_case.preconditions,
            steps=test_case.steps,
            expected_results=test_case.expected_results,
            priority=test_case.priority,
            type=test_case.type,
            created_by_id=current_user.id
        )
        db.add(tc_version)
        logger.info(f"   ✅ Test case version {next_tc_version} created")
        
        # Update test case with healed data
        # Handle steps - convert array to newline-separated string if needed
        healed_steps = healed_test_case.get("steps")
        if healed_steps:
            if isinstance(healed_steps, list):
                # Convert array to numbered string format
                test_case.steps = '\n'.join([f"{i+1}. {step}" for i, step in enumerate(healed_steps)])
            else:
                test_case.steps = healed_steps
        
        test_case.title = healed_test_case.get("title", test_case.title)
        test_case.description = healed_test_case.get("description", test_case.description)
        test_case.preconditions = healed_test_case.get("preconditions", test_case.preconditions)
        test_case.expected_results = healed_test_case.get("expected_results", test_case.expected_results)
        
        # Update enum fields if provided
        if healed_test_case.get("priority"):
            from app.database.models import TestCasePriority
            test_case.priority = TestCasePriority(healed_test_case["priority"])
        
        if healed_test_case.get("type"):
            from app.database.models import TestCaseType
            test_case.type = TestCaseType(healed_test_case["type"])
        
        # Log updated values
        logger.info(f"✨ UPDATED TEST CASE:")
        logger.info(f"   Title: {test_case.title}")
        logger.info(f"   Description: {test_case.description[:100] if test_case.description else 'None'}...")
        logger.info(f"   Steps: {test_case.steps[:200] if test_case.steps else 'None'}...")
        logger.info(f"   Expected Results: {test_case.expected_results[:100] if test_case.expected_results else 'None'}...")
        logger.info(f"   Priority: {test_case.priority}")
        logger.info(f"   Type: {test_case.type}")
        
        # Update aggregate response if provided
        version_number = None
        if updated_aggregate:
            from app.database.models import ProjectAggregateVersion
            
            # Get current max version number
            max_version = db.query(ProjectAggregateVersion).filter(
                ProjectAggregateVersion.project_id == project_id
            ).order_by(ProjectAggregateVersion.version_number.desc()).first()
            
            next_version = (max_version.version_number + 1) if max_version else 1
            version_number = next_version
            
            logger.info(f"📦 CREATING AGGREGATE VERSION {next_version}")
            
            # Save current aggregate as version (before updating)
            if project.ai_aggregate_response:
                version = ProjectAggregateVersion(
                    project_id=project_id,
                    version_number=next_version,
                    aggregate_data=project.ai_aggregate_response,
                    created_by_id=current_user.id,
                    description=f"Auto-saved before healing test case #{test_case.case_number}: {test_case.title[:50]}"
                )
                db.add(version)
                logger.info(f"   ✅ Version {next_version} created with description: {version.description}")
            
            # Update to new aggregate
            project.ai_aggregate_response = updated_aggregate
            logger.info(f"   ✅ Project aggregate updated")
            
            # Create new version for the healed aggregate
            next_version += 1
            healed_version = ProjectAggregateVersion(
                project_id=project_id,
                version_number=next_version,
                aggregate_data=updated_aggregate,
                created_by_id=current_user.id,
                description=f"After healing test case #{test_case.case_number}: {test_case.title[:50]}"
            )
            db.add(healed_version)
            version_number = next_version
            logger.info(f"   ✅ Version {next_version} created for healed aggregate")
        
        # Flush changes to database before commit
        db.flush()
        
        # Verify test case was updated
        logger.info(f"🔍 VERIFYING TEST CASE BEFORE COMMIT:")
        logger.info(f"   Title: {test_case.title}")
        logger.info(f"   Steps length: {len(test_case.steps) if test_case.steps else 0}")
        
        # Create activity log for the healing action
        from app.database.models import TestCaseActivity, ActivityType
        
        activity = TestCaseActivity(
            test_case_id=test_case_id,
            activity_type=ActivityType.UPDATED,
            description=f"Test case healed using AI - Version {next_tc_version} created",
            activity_data={
                "action": "ai_healing",
                "version_created": next_tc_version,
                "aggregate_version": version_number,
                "healed_fields": list(healed_test_case.keys()) if healed_test_case else []
            },
            created_by_id=current_user.id
        )
        db.add(activity)
        
        # Commit all changes
        db.commit()
        db.refresh(test_case)
        db.refresh(project)
        
        # Verify after commit
        logger.info(f"🔍 VERIFYING TEST CASE AFTER COMMIT:")
        logger.info(f"   Title: {test_case.title}")
        logger.info(f"   Steps length: {len(test_case.steps) if test_case.steps else 0}")
        
        logger.info(f"✅ Successfully saved healed test case {test_case_id}")
        logger.info(f"   Created test case version: {next_tc_version}")
        logger.info(f"   Created activity log entry")
        if version_number:
            logger.info(f"   Created aggregate version: {version_number}")
        
        return {
            "success": True,
            "message": "Test case healed and saved successfully",
            "test_case_id": test_case.id,
            "test_case_version": next_tc_version,
            "aggregate_version": version_number,
            "activity_logged": True
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save healed test case: {e}", exc_info=True)
        return {"success": False, "error_message": str(e)}

@router.post("/projects/{project_id}/restore-aggregate-version/{version_number}")
async def restore_aggregate_version(
    project_id: int,
    version_number: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Restore aggregate to a previous version.
    """
    try:
        from app.database.models import ProjectAggregateVersion
        
        # Get the project
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return {"success": False, "error_message": "Project not found"}
        
        # Get the version to restore
        version = db.query(ProjectAggregateVersion).filter(
            ProjectAggregateVersion.project_id == project_id,
            ProjectAggregateVersion.version_number == version_number
        ).first()
        
        if not version:
            return {"success": False, "error_message": f"Version {version_number} not found"}
        
        logger.info(f"🔄 RESTORING AGGREGATE TO VERSION {version_number}")
        logger.info(f"   Project: {project.name}")
        logger.info(f"   Description: {version.description}")
        
        # Create a new version with current state before restoring
        max_version = db.query(ProjectAggregateVersion).filter(
            ProjectAggregateVersion.project_id == project_id
        ).order_by(ProjectAggregateVersion.version_number.desc()).first()
        
        next_version = (max_version.version_number + 1) if max_version else 1
        
        backup_version = ProjectAggregateVersion(
            project_id=project_id,
            version_number=next_version,
            aggregate_data=project.ai_aggregate_response,
            created_by_id=current_user.id,
            description=f"Auto-backup before restoring to version {version_number}"
        )
        db.add(backup_version)
        logger.info(f"   ✅ Created backup version {next_version}")
        
        # Restore the aggregate
        project.ai_aggregate_response = version.aggregate_data
        
        db.commit()
        db.refresh(project)
        
        logger.info(f"✅ Successfully restored aggregate to version {version_number}")
        
        return {
            "success": True,
            "message": f"Aggregate restored to version {version_number}",
            "restored_version": version_number,
            "backup_version": next_version
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to restore aggregate version: {e}", exc_info=True)
        return {"success": False, "error_message": str(e)}

@router.get("/projects/{project_id}/aggregate-versions")
async def get_aggregate_versions(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get all aggregate versions for a project.
    """
    from app.database.models import ProjectAggregateVersion
    
    logger.info("="*80)
    logger.info(f"🔍 AGGREGATE VERSIONS REQUEST")
    logger.info(f"   Project ID: {project_id} (type: {type(project_id)})")
    logger.info(f"   User: {current_user.name} (ID: {current_user.id}, Email: {current_user.email})")
    logger.info(f"   Database session: {db}")
    logger.info("="*80)
    
    try:
        # Test database connection
        logger.info("📊 Testing database connection...")
        test_query = db.query(Project).count()
        logger.info(f"   Total projects in DB: {test_query}")
        
        # Check if this specific project exists
        logger.info(f"🔍 Looking for project {project_id}...")
        project = db.query(Project).filter(Project.id == project_id).first()
        
        if project:
            logger.info(f"   ✅ Project found: {project.name}")
            logger.info(f"   AI Generated: {project.ai_generated}")
            logger.info(f"   Has aggregate: {project.ai_aggregate_response is not None}")
        else:
            logger.error(f"   ❌ Project {project_id} NOT FOUND")
            all_projects = db.query(Project.id, Project.name).all()
            logger.error(f"   Available project IDs: {[p.id for p in all_projects]}")
            
        # Get versions
        logger.info(f"🔍 Querying ProjectAggregateVersion table...")
        versions = db.query(ProjectAggregateVersion).filter(
            ProjectAggregateVersion.project_id == project_id
        ).order_by(ProjectAggregateVersion.version_number.desc()).all()
        
        logger.info(f"   Found {len(versions)} versions")
        
        # Get current aggregate from project
        current_aggregate = project.ai_aggregate_response if project else None
        
        # Determine current version number (highest version number)
        current_version_number = versions[0].version_number if versions else None
        
        # Manually serialize to avoid schema issues
        versions_data = []
        for v in versions:
            version_dict = {
                "id": v.id,
                "project_id": v.project_id,
                "version_number": v.version_number,
                "aggregate_data": v.aggregate_data,
                "created_at": v.created_at.isoformat() if v.created_at else None,
                "created_by_id": v.created_by_id,
                "description": v.description
            }
            
            # Add created_by user info if available
            if v.created_by:
                version_dict["created_by"] = {
                    "id": v.created_by.id,
                    "username": v.created_by.name,  # Use 'name' field as username
                    "email": v.created_by.email
                }
            
            versions_data.append(version_dict)
        
        return {
            "success": True,
            "versions": versions_data,
            "current_aggregate": current_aggregate,
            "current_version_number": current_version_number
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get aggregate versions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/projects/{project_id}/aggregate-versions")
async def create_aggregate_version(
    project_id: int,
    description: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a new version snapshot of the current aggregate response.
    """
    from app.database.models import ProjectAggregateVersion
    
    try:
        # Get project
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if not project.ai_aggregate_response:
            raise HTTPException(status_code=400, detail="Project has no aggregate response to version")
        
        # Get next version number
        max_version = db.query(ProjectAggregateVersion).filter(
            ProjectAggregateVersion.project_id == project_id
        ).order_by(ProjectAggregateVersion.version_number.desc()).first()
        
        next_version = (max_version.version_number + 1) if max_version else 1
        
        # Create version
        new_version = ProjectAggregateVersion(
            project_id=project_id,
            version_number=next_version,
            aggregate_data=project.ai_aggregate_response,
            created_by_id=current_user.id,
            description=description or f"Version {next_version}"
        )
        
        db.add(new_version)
        db.commit()
        db.refresh(new_version)
        
        logger.info(f"Created aggregate version {next_version} for project {project_id}")
        
        return {
            "success": True,
            "message": f"Version {next_version} created successfully",
            "version": {
                "id": new_version.id,
                "version_number": new_version.version_number,
                "created_at": new_version.created_at.isoformat(),
                "description": new_version.description
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create aggregate version: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/projects/{project_id}/aggregate-versions/{version_id}/revert")
async def revert_to_aggregate_version(
    project_id: int,
    version_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Revert the project's aggregate response to a specific version.
    This will update the current aggregate and optionally sync entities.
    """
    from app.database.models import ProjectAggregateVersion
    
    try:
        # Get project
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Get version to revert to
        version = db.query(ProjectAggregateVersion).filter(
            ProjectAggregateVersion.id == version_id,
            ProjectAggregateVersion.project_id == project_id
        ).first()
        
        if not version:
            raise HTTPException(status_code=404, detail="Version not found")
        
        # Save current aggregate as new version before reverting
        max_version = db.query(ProjectAggregateVersion).filter(
            ProjectAggregateVersion.project_id == project_id
        ).order_by(ProjectAggregateVersion.version_number.desc()).first()
        
        next_version = (max_version.version_number + 1) if max_version else 1
        
        # Create backup of current state
        if project.ai_aggregate_response:
            backup_version = ProjectAggregateVersion(
                project_id=project_id,
                version_number=next_version,
                aggregate_data=project.ai_aggregate_response,
                created_by_id=current_user.id,
                description=f"Auto-backup before reverting to version {version.version_number}"
            )
            db.add(backup_version)
        
        # Revert to selected version
        project.ai_aggregate_response = version.aggregate_data
        db.commit()
        
        logger.info(f"Reverted project {project_id} aggregate to version {version.version_number}")
        
        return {
            "success": True,
            "message": f"Reverted to version {version.version_number}",
            "reverted_to": {
                "version_number": version.version_number,
                "created_at": version.created_at.isoformat(),
                "description": version.description
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to revert aggregate version: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
