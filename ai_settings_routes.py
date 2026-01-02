"""
AI Settings API Routes
Manages user-specific AI provider configurations and API keys
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.database.models import User, AIProvider, AIProviderConfiguration
from app.auth.deps import get_current_active_user

router = APIRouter()

@router.get("/providers")
async def get_ai_providers(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get all AI provider configurations for the current user"""
    
    # Get user's provider configurations
    configs = db.query(AIProviderConfiguration).filter(
        AIProviderConfiguration.user_id == current_user.id
    ).all()
    
    # Return list of providers with configuration status
    provider_id_map = {"openai": 1, "anthropic": 2, "gemini": 3, "vertex": 4, "openrouter": 5, "llama": 6, "anthropic_vertex": 7}
    providers = []
    for provider in AIProvider:
        config = next((c for c in configs if c.provider_type == provider.value), None)
        providers.append({
            "id": provider_id_map.get(provider.value, 0),
            "provider_type": provider.value,
            "name": provider.value.capitalize(),
            "is_configured": bool(config and config.api_key),
            "is_active": config.is_active if config else False,
            "model_name": config.model_name if config else None,
        })
    
    return {"providers": providers}

@router.get("/global-settings")
async def get_user_ai_settings(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get user's AI settings (kept as 'global-settings' for backwards compatibility)
    Returns user's default provider and model preferences
    """
    
    # Get user's provider configurations
    configs = db.query(AIProviderConfiguration).filter(
        AIProviderConfiguration.user_id == current_user.id,
        AIProviderConfiguration.is_active == True
    ).all()
    
    # Find first active provider (prefer one with API key, but accept without)
    default_config = next((c for c in configs if c.api_key), None)
    if not default_config and configs:
        # No config with API key, use first active config
        default_config = configs[0]
    
    if default_config:
        # Map provider type to ID for backwards compatibility
        provider_id_map = {"openai": 1, "anthropic": 2, "gemini": 3, "vertex": 4, "openrouter": 5, "llama": 6, "anthropic_vertex": 7}
        provider_id = provider_id_map.get(default_config.provider_type, 3)
        model_name = default_config.model_name or get_default_model(default_config.provider_type)
        temperature = default_config.temperature if default_config.temperature is not None else 0.7
        max_context_window = default_config.max_context_window if default_config.max_context_window is not None else 1000000
        
        print(f"📖 Loading AI settings for user {current_user.id}:")
        print(f"   Config ID: {default_config.id}")
        print(f"   Provider: {default_config.provider_type} (returning ID: {provider_id})")
        print(f"   Model from DB: {default_config.model_name}")
        print(f"   Model returning: {model_name}")
    else:
        # No provider configured, use gemini defaults
        provider_id = 3
        model_name = "gemini-2.0-flash"
        temperature = 0.7
        max_context_window = 1000000
    
    return {
        "default_provider_id": provider_id,
        "default_model_name": model_name,
        "temperature": temperature,
        "max_context_window": max_context_window,
        "auto_generate_test_cases": True,
        "auto_analyze_code": False,
        "max_concurrent_analyses": 3,
    }

@router.put("/global-settings")
async def update_user_ai_settings(
    settings: Dict[str, Any],
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Update user's AI settings (kept as 'global-settings' for backwards compatibility)
    Updates user's default provider preference
    """
    
    # Map provider ID to provider type
    id_to_provider = {1: "openai", 2: "anthropic", 3: "gemini", 4: "vertex", 5: "openrouter", 6: "llama", 7: "anthropic_vertex"}
    
    # Get provider type - either from provided ID or from current active config
    provider_type = None
    if "default_provider_id" in settings and settings["default_provider_id"] is not None:
        provider_type = id_to_provider.get(settings["default_provider_id"], "gemini")
    else:
        # No provider ID provided, find current active config
        active_config = db.query(AIProviderConfiguration).filter(
            AIProviderConfiguration.user_id == current_user.id,
            AIProviderConfiguration.is_active == True
        ).first()
        if active_config:
            provider_type = active_config.provider_type
        else:
            provider_type = "gemini"  # Default fallback
    
    model_name = settings.get("default_model_name")
    temperature = settings.get("temperature", 0.7)
    max_context_window = settings.get("max_context_window", 1000000)
    
    print(f"🔧 Saving AI settings:")
    print(f"   Provider: {provider_type} (ID: {settings.get('default_provider_id', 'N/A')})")
    print(f"   Model: {model_name}")
    print(f"   Temperature: {temperature}")
    print(f"   Max Context: {max_context_window}")
    
    # First, deactivate all existing configurations for this user
    db.query(AIProviderConfiguration).filter(
        AIProviderConfiguration.user_id == current_user.id
    ).update({"is_active": False})
    
    # Get or create provider configuration for the selected provider
    config = db.query(AIProviderConfiguration).filter(
        AIProviderConfiguration.user_id == current_user.id,
        AIProviderConfiguration.provider_type == provider_type
    ).first()
    
    if config:
        if model_name is not None:
            config.model_name = model_name
        config.temperature = temperature
        config.max_context_window = max_context_window
        config.is_active = True
    else:
        # Create new configuration (without API key, user needs to add it separately)
        config = AIProviderConfiguration(
            user_id=current_user.id,
            provider_type=provider_type,
            model_name=model_name,
            temperature=temperature,
            max_context_window=max_context_window,
            is_active=True
        )
        db.add(config)
    
    db.commit()
    db.refresh(config)
    
    print(f"✅ Saved config ID {config.id}: {config.provider_type} / {config.model_name}")
    
    return {"message": "Settings updated successfully"}

@router.post("/configure-provider")
async def configure_provider(
    provider_data: Dict[str, Any],
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Configure API key and settings for a specific AI provider
    
    Request body:
    {
        "provider_type": "openai|anthropic|gemini|vertex|openrouter|llama|anthropic_vertex",
        "api_key": "your-api-key",
        "model_name": "model-name" (optional)
    }
    """
    
    provider_type = provider_data.get("provider_type")
    api_key = provider_data.get("api_key")
    
    if not provider_type or not api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="provider_type and api_key are required"
        )
    
    # Validate provider type
    try:
        AIProvider(provider_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid provider type: {provider_type}"
        )
    
    # Get or create provider configuration
    config = db.query(AIProviderConfiguration).filter(
        AIProviderConfiguration.user_id == current_user.id,
        AIProviderConfiguration.provider_type == provider_type
    ).first()
    
    if config:
        config.api_key = api_key
        config.model_name = provider_data.get("model_name", config.model_name)
        config.is_active = True
    else:
        config = AIProviderConfiguration(
            user_id=current_user.id,
            provider_type=provider_type,
            api_key=api_key,
            model_name=provider_data.get("model_name"),
            is_active=True
        )
        db.add(config)
    
    db.commit()
    
    return {"message": f"{provider_type.capitalize()} provider configured successfully"}

def get_default_model(provider_type: str) -> str:
    """Get default model name for a provider"""
    defaults = {
        "openai": "gpt-4o",
        "anthropic": "claude-3.5-sonnet",
        "gemini": "gemini-2.0-flash",
        "vertex": "gemini-2.5-flash",  # Use standard flash model (works without special permissions)
        "openrouter": "openai/gpt-4o",
        "llama": "publishers/meta/models/llama-4-maverick-17b-128e-instruct-maas",  # Default Llama model via Vertex AI
        "anthropic_vertex": "claude-sonnet-4-5@20250929"  # Default Claude 4.5 Sonnet model via Vertex AI
    }
    return defaults.get(provider_type, "gemini-2.0-flash")

@router.get("/model-capabilities")
async def get_model_capabilities(
    provider_id: int = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get capabilities of AI models"""
    
    # Mock data for all available providers
    all_capabilities = [
        {
            "provider_id": 1,
            "provider_name": "OpenAI",
            "model_name": "gpt-4",
            "capabilities": {
                "test_case_generation": True,
                "code_analysis": True,
                "requirement_analysis": True,
                "bug_prediction": True,
                "performance_analysis": True,
                "security_analysis": True,
                "natural_language_processing": True,
                "code_completion": True,
                "max_tokens": 2000,
                "supports_streaming": True,
                "supports_function_calling": True
            },
            "limitations": {
                "max_code_length": 50000,
                "rate_limit_per_minute": 60,
                "supported_languages": ["python", "javascript", "java", "csharp", "typescript", "go", "rust"]
            }
        },
        {
            "provider_id": 2,
            "provider_name": "Anthropic",
            "model_name": "claude-3-sonnet",
            "capabilities": {
                "test_case_generation": True,
                "code_analysis": True,
                "requirement_analysis": True,
                "bug_prediction": False,
                "performance_analysis": True,
                "security_analysis": True,
                "natural_language_processing": True,
                "code_completion": False,
                "max_tokens": 1500,
                "supports_streaming": True,
                "supports_function_calling": False
            },
            "limitations": {
                "max_code_length": 40000,
                "rate_limit_per_minute": 50,
                "supported_languages": ["python", "javascript", "java", "csharp", "typescript"]
            }
        },
        {
            "provider_id": 3,
            "provider_name": "Google Gemini",
            "model_name": "gemini-pro",
            "capabilities": {
                "test_case_generation": True,
                "code_analysis": True,
                "requirement_analysis": True,
                "bug_prediction": True,
                "performance_analysis": True,
                "security_analysis": False,
                "natural_language_processing": True,
                "code_completion": False,
                "max_tokens": 1000,
                "supports_streaming": False,
                "supports_function_calling": False
            },
            "limitations": {
                "max_code_length": 30000,
                "rate_limit_per_minute": 40,
                "supported_languages": ["python", "javascript", "java", "typescript"]
            }
        },
        {
            "provider_id": 7,
            "provider_name": "Anthropic via Vertex",
            "model_name": "claude-3-5-sonnet@20241022",
            "capabilities": {
                "test_case_generation": True,
                "code_analysis": True,
                "requirement_analysis": True,
                "bug_prediction": True,
                "performance_analysis": True,
                "security_analysis": True,
                "natural_language_processing": True,
                "code_completion": True,
                "max_tokens": 8000,
                "supports_streaming": True,
                "supports_function_calling": True
            },
            "limitations": {
                "max_code_length": 200000,
                "rate_limit_per_minute": 60,
                "supported_languages": ["python", "javascript", "java", "csharp", "typescript", "go", "rust", "cpp", "php", "ruby"]
            }
        }
    ]
    
    # Filter by provider_id if specified
    if provider_id:
        capabilities = [cap for cap in all_capabilities if cap["provider_id"] == provider_id]
        if not capabilities:
            raise HTTPException(status_code=404, detail="Provider not found")
    else:
        capabilities = all_capabilities
    
    return {"model_capabilities": capabilities}

@router.post("/test-provider/{provider_id}")
async def test_ai_provider_connection(
    provider_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Test connection to an AI provider"""
    
    # Mock provider data for testing
    provider_names = {1: "OpenAI", 2: "Anthropic", 3: "Google Gemini", 4: "Vertex AI", 5: "OpenRouter", 6: "Llama", 7: "Anthropic via Vertex", 8: "Llama via OpenRouter"}
    
    if provider_id not in provider_names:
        raise HTTPException(status_code=404, detail="AI provider not found")
    
    # Simulate a successful connection test
    test_results = {
        "provider_id": provider_id,
        "provider_name": provider_names[provider_id],
        "connection_status": "success",
        "response_time_ms": 250 + (provider_id * 50),
        "api_version": "v1",
        "available_models": [
            "gpt-4" if provider_id == 1 else 
            "claude-3-sonnet" if provider_id == 2 else 
            "gemini-pro" if provider_id == 3 else 
            "claude-3-5-sonnet@20241022" if provider_id == 7 else 
            "gemini-pro"
        ],
        "rate_limit_remaining": 999 - (provider_id * 100),
        "test_timestamp": datetime.utcnow().isoformat()
    }
    
    return test_results

@router.get("/prompt-templates")
async def get_prompt_templates(
    analysis_type: str = None,
    current_user: User = Depends(get_current_active_user)
):
    """Get AI prompt templates for different analysis types"""
    
    templates = {
        "test_case_generation": {
            "name": "Test Case Generation",
            "description": "Generate test cases from requirements",
            "template": """
Generate comprehensive test cases for the following requirements:

Requirements: {requirements}

Please generate test cases that include:
1. Positive test scenarios
2. Negative test scenarios  
3. Edge cases
4. Boundary value tests

Format each test case with:
- Title
- Description
- Preconditions
- Test Steps
- Expected Results
- Priority (High/Medium/Low)
- Type (Functional/Integration/System/etc.)
"""
        },
        "code_analysis": {
            "name": "Code Analysis",
            "description": "Analyze code quality and suggest improvements",
            "template": """
Analyze the following code and provide insights on:

Code:
{code}

Please analyze for:
1. Code quality and maintainability
2. Potential bugs and issues
3. Performance considerations
4. Security vulnerabilities
5. Suggested test cases
6. Refactoring opportunities

Provide specific recommendations with code examples where applicable.
"""
        },
        "bug_prediction": {
            "name": "Bug Prediction",
            "description": "Predict potential bugs in code",
            "template": """
Analyze the following code for potential bugs and issues:

Code:
{code}

Focus on:
1. Logic errors
2. Memory leaks
3. Race conditions
4. Input validation issues
5. Error handling problems
6. Performance bottlenecks

Provide specific bug predictions with severity levels and suggested fixes.
"""
        }
    }
    
    if analysis_type:
        if analysis_type not in templates:
            raise HTTPException(status_code=404, detail="Template not found")
        return {"template": templates[analysis_type]}
    
    return {"templates": templates}