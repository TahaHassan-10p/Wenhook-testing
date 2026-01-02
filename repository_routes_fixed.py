from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.database.models import User, Project, RepositoryConfig, PRAnalysis, TestCaseDelta, IntegrationLog, TestCase
from app.schemas.extended import (
    RepositoryConfigCreate, RepositoryConfigUpdate, RepositoryConfig as RepositoryConfigSchema,
    PRAnalysisCreate, PRAnalysis as PRAnalysisSchema, TestCaseDeltaSchema, IntegrationLogSchema
)
from app.services.github_integration_service import GitHubIntegrationService
from app.auth.deps import get_current_active_user, require_permission
from app.core.permissions import Permission
import json
import hmac
import hashlib

router = APIRouter()

# Simplified Repository Management (for frontend integration)
@router.post("/", response_model=RepositoryConfigSchema)
async def create_repository(
    config_data: RepositoryConfigCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new repository configuration (simplified endpoint for frontend)"""
    
    # Check if repository URL already exists for this project
    existing_config = db.query(RepositoryConfig).filter(
        RepositoryConfig.project_id == config_data.project_id,
        RepositoryConfig.repository_url == config_data.repository_url
    ).first()
    
    if existing_config:
        raise HTTPException(
            status_code=400,
            detail=f"Repository configuration already exists for this URL"
        )
    
    # Validate project exists
    project = db.query(Project).filter(Project.id == config_data.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Extract repository name from URL if not provided
    repository_name = config_data.repository_name
    if not repository_name and config_data.repository_url:
        if 'github.com' in config_data.repository_url:
            # Extract from GitHub URL
            parts = config_data.repository_url.rstrip('/').split('/')
            if len(parts) >= 2:
                repository_name = f"{parts[-2]}/{parts[-1].replace('.git', '')}"
    
    repo_config = RepositoryConfig(
        project_id=config_data.project_id,
        repository_url=config_data.repository_url,
        repository_name=repository_name,
        provider=config_data.provider,
        access_token=config_data.access_token,
        webhook_secret=config_data.webhook_secret,
        branch_patterns=config_data.branch_patterns,
        auto_analyze_prs=config_data.auto_analyze_prs,
        auto_create_test_cases=config_data.auto_create_test_cases,
        settings=config_data.settings,
        is_active=config_data.is_active,
        created_by_id=current_user.id
    )
    
    db.add(repo_config)
    db.commit()
    db.refresh(repo_config)
    
    return repo_config

@router.get("/", response_model=List[RepositoryConfigSchema])
async def get_repositories(
    project_id: int = None,
    active_only: bool = False,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get repository configurations (simplified endpoint for frontend)"""
    
    query = db.query(RepositoryConfig)
    
    if project_id:
        query = query.filter(RepositoryConfig.project_id == project_id)
    
    if active_only:
        query = query.filter(RepositoryConfig.is_active == True)
    
    configs = query.order_by(RepositoryConfig.created_at).all()
    
    # Mask sensitive data in response
    for config in configs:
        if config.access_token:
            config.access_token = "***" + config.access_token[-4:] if len(config.access_token) > 4 else "***"
        if config.webhook_secret:
            config.webhook_secret = "***masked***"
    
    return configs

# Repository Configuration Management
@router.post("/configs/", response_model=RepositoryConfigSchema)
async def create_repository_config(
    config_data: RepositoryConfigCreate,
    current_user: User = Depends(require_permission(Permission.CONFIGURE_SETTINGS)),
    db: Session = Depends(get_db)
):
    """Create a new repository configuration"""
    
    # Check if repository URL already exists for this project
    existing_config = db.query(RepositoryConfig).filter(
        RepositoryConfig.project_id == config_data.project_id,
        RepositoryConfig.repository_url == config_data.repository_url
    ).first()
    
    if existing_config:
        raise HTTPException(
            status_code=400,
            detail=f"Repository configuration already exists for this URL"
        )
    
    # Validate project exists
    project = db.query(Project).filter(Project.id == config_data.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    repo_config = RepositoryConfig(
        project_id=config_data.project_id,
        repository_url=config_data.repository_url,
        provider=config_data.provider,
        access_token=config_data.access_token,
        webhook_secret=config_data.webhook_secret,
        branch_patterns=config_data.branch_patterns,
        auto_analyze_prs=config_data.auto_analyze_prs,
        auto_create_test_cases=config_data.auto_create_test_cases,
        settings=config_data.settings,
        is_active=config_data.is_active,
        created_by_id=current_user.id
    )
    
    db.add(repo_config)
    db.commit()
    db.refresh(repo_config)
    
    return repo_config

@router.get("/configs/", response_model=List[RepositoryConfigSchema])
async def get_repository_configs(
    project_id: int = None,
    active_only: bool = False,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get repository configurations with optional filtering"""
    
    query = db.query(RepositoryConfig)
    
    if project_id:
        query = query.filter(RepositoryConfig.project_id == project_id)
    
    if active_only:
        query = query.filter(RepositoryConfig.is_active == True)
    
    configs = query.order_by(RepositoryConfig.created_at).all()
    
    # Mask sensitive data in response
    for config in configs:
        if config.access_token:
            config.access_token = "***" + config.access_token[-4:] if len(config.access_token) > 4 else "***"
        if config.webhook_secret:
            config.webhook_secret = "***masked***"
    
    return configs

@router.get("/configs/{config_id}", response_model=RepositoryConfigSchema)
async def get_repository_config(
    config_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get a specific repository configuration"""
    
    config = db.query(RepositoryConfig).filter(RepositoryConfig.id == config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Repository configuration not found")
    
    # Mask sensitive data in response
    if config.access_token:
        config.access_token = "***" + config.access_token[-4:] if len(config.access_token) > 4 else "***"
    if config.webhook_secret:
        config.webhook_secret = "***masked***"
    
    return config

@router.put("/configs/{config_id}", response_model=RepositoryConfigSchema)
async def update_repository_config(
    config_id: int,
    config_data: RepositoryConfigUpdate,
    current_user: User = Depends(require_permission(Permission.CONFIGURE_SETTINGS)),
    db: Session = Depends(get_db)
):
    """Update a repository configuration"""
    
    config = db.query(RepositoryConfig).filter(RepositoryConfig.id == config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Repository configuration not found")
    
    # Update fields
    for key, value in config_data.dict(exclude_unset=True).items():
        setattr(config, key, value)
    
    config.updated_at = db.func.now()
    
    db.commit()
    db.refresh(config)
    
    # Mask sensitive data in response
    if config.access_token:
        config.access_token = "***" + config.access_token[-4:] if len(config.access_token) > 4 else "***"
    if config.webhook_secret:
        config.webhook_secret = "***masked***"
    
    return config

@router.delete("/configs/{config_id}")
async def delete_repository_config(
    config_id: int,
    current_user: User = Depends(require_permission(Permission.CONFIGURE_SETTINGS)),
    db: Session = Depends(get_db)
):
    """Delete a repository configuration"""
    
    config = db.query(RepositoryConfig).filter(RepositoryConfig.id == config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Repository configuration not found")
    
    db.delete(config)
    db.commit()
    
    return {"message": "Repository configuration deleted successfully"}

# Integration Logs
@router.get("/integration-logs", response_model=List[IntegrationLogSchema])
async def get_integration_logs(
    repository_config_id: int = None,
    event_type: str = None,
    success: bool = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get integration logs with optional filtering"""
    
    query = db.query(IntegrationLog)
    
    if repository_config_id:
        query = query.filter(IntegrationLog.repository_config_id == repository_config_id)
    
    if event_type:
        query = query.filter(IntegrationLog.event_type == event_type)
    
    if success is not None:
        query = query.filter(IntegrationLog.success == success)
    
    logs = query.order_by(IntegrationLog.created_at.desc()).offset(skip).limit(limit).all()
    return logs