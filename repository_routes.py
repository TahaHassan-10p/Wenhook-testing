from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from app.database.session import get_db
from app.database.models import User, Project, RepositoryConfig, PRAnalysis, TestCaseDelta, IntegrationLog, TestCase, TestSuite, TestCaseVersion, TestCaseActivity, ActivityType, Section
from app.schemas.extended import (
    RepositoryConfigCreate, RepositoryConfigUpdate, RepositoryConfig as RepositoryConfigSchema,
    PRAnalysisCreate, PRAnalysis as PRAnalysisSchema, TestCaseDeltaSchema, IntegrationLogSchema
)
from app.services.github_integration_service import GitHubIntegrationService
from app.auth.deps import get_current_active_user, require_permission
from app.core.permissions import Permission
from app.core.encryption import encrypt_access_token, decrypt_access_token, mask_token_for_display

import json
import hmac
import hashlib
import secrets
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Simplified Repository Management (for frontend integration)
@router.get("", response_model=List[RepositoryConfigSchema])
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
    
    # Mask sensitive data in response using proper encryption
    for config in configs:
        if config.access_token:
            try:
                # Decrypt token for masking display only
                decrypted_token = decrypt_access_token(config.access_token)
                config.access_token = mask_token_for_display(decrypted_token) if decrypted_token else "***corrupted***"
            except Exception as e:
                # Handle corrupted tokens gracefully
                logger.warning(f"Failed to decrypt access token for config {config.id}: {e}")
                config.access_token = "***corrupted***"
        if config.webhook_secret:
            config.webhook_secret = "***masked***"
    
    return configs

@router.delete("/{config_id}")
async def delete_repository_config(
    config_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a repository configuration"""
    
    repo_config = db.query(RepositoryConfig).filter(
        RepositoryConfig.id == config_id
    ).first()
    
    if not repo_config:
        raise HTTPException(status_code=404, detail="Repository configuration not found")
    
    # Store repository info for response
    repository_info = {
        "id": repo_config.id,
        "repository_url": repo_config.repository_url,
        "repository_name": repo_config.repository_name
    }
    
    # Delete the repository configuration (cascade will handle related records)
    db.delete(repo_config)
    db.commit()
    
    return {
        "message": "Repository configuration deleted successfully",
        "deleted_repository": repository_info
    }

# Repository Configuration Management
@router.post("/configs/", response_model=RepositoryConfigSchema)
async def create_repository_config(
    config_data: RepositoryConfigCreate,
    current_user: User = Depends(get_current_active_user),
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
    
    # Extract repository name from URL if not provided
    repository_name = config_data.repository_name
    if not repository_name and config_data.repository_url:
        if 'github.com' in config_data.repository_url:
            # Extract from GitHub URL
            parts = config_data.repository_url.rstrip('/').split('/')
            if len(parts) >= 2:
                repository_name = f"{parts[-2]}/{parts[-1].replace('.git', '')}"
        elif 'gitlab.com' in config_data.repository_url:
            # Extract from GitLab URL
            parts = config_data.repository_url.rstrip('/').split('/')
            if len(parts) >= 2:
                repository_name = f"{parts[-2]}/{parts[-1].replace('.git', '')}"
    
    # Auto-generate webhook secret if not provided
    webhook_secret = config_data.webhook_secret
    if not webhook_secret:
        webhook_secret = secrets.token_hex(32)  # Generate 64-character hex string
    
    # Generate webhook URL using ngrok
    from app.core.config import settings as app_settings
    webhook_url = f"{app_settings.NGROK_URL}/api/v1/webhooks/{config_data.provider}/{config_data.project_id}"
    
    # Store webhook URL in settings
    repo_settings = config_data.settings or {}
    repo_settings['webhook_url'] = webhook_url
    
    repo_config = RepositoryConfig(
        project_id=config_data.project_id,
        repository_url=config_data.repository_url,
        repository_name=repository_name,
        provider=config_data.provider,
        access_token=encrypt_access_token(config_data.access_token) if config_data.access_token else None,
        webhook_secret=webhook_secret,
        branch_patterns=config_data.branch_patterns,
        auto_analyze_prs=config_data.auto_analyze_prs,
        auto_create_test_cases=config_data.auto_create_test_cases,
        settings=repo_settings,
        is_active=config_data.is_active,
        created_by_id=current_user.id
    )
    
    db.add(repo_config)
    db.commit()
    db.refresh(repo_config)
    
    return repo_config
    
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

@router.get("/{config_id}/webhook-details")
async def get_webhook_details_full(
    config_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get full webhook details including unmasked secret for setup"""
    
    config = db.query(RepositoryConfig).filter(RepositoryConfig.id == config_id).first()
    
    if not config:
        raise HTTPException(status_code=404, detail="Repository configuration not found")
    
    # Extract repository name from URL if not set
    repository_name = config.repository_name
    if not repository_name and config.repository_url:
        if 'github.com' in config.repository_url:
            parts = config.repository_url.rstrip('/').split('/')
            if len(parts) >= 2:
                repository_name = f"{parts[-2]}/{parts[-1].replace('.git', '')}"
                # Update the database
                config.repository_name = repository_name
                db.commit()
    
    # Generate webhook URL dynamically using current NGROK_URL setting
    from app.core.config import settings as app_settings
    webhook_url = f"{app_settings.NGROK_URL}/api/v1/webhooks/{config.provider}/{config.project_id}"
    
    return {
        "id": config.id,
        "repository_name": repository_name or "Unknown Repository",
        "repository_url": config.repository_url,
        "webhook_secret": config.webhook_secret,
        "webhook_url": webhook_url,
        "provider": config.provider.value if hasattr(config.provider, 'value') else config.provider
    }

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
    
    # Update fields with proper encryption for access_token
    for key, value in config_data.dict(exclude_unset=True).items():
        if key == 'access_token' and value:
            # Encrypt access token before storing
            setattr(config, key, encrypt_access_token(value))
        else:
            setattr(config, key, value)
    
    config.updated_at = func.now()
    
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
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a repository configuration and all related data"""
    
    config = db.query(RepositoryConfig).filter(RepositoryConfig.id == config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Repository configuration not found")
    
    # Check if user has permission to delete this config (must be creator or admin)
    if config.created_by_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to delete this repository configuration")
    
    try:
        # Delete related records in correct order to avoid foreign key constraints
        logger.info(f"Starting deletion of repository config {config_id}")
        
        # 1. Delete TestCaseDeltas (they reference PRAnalysis)
        try:
            deltas_deleted = 0
            pr_analysis_ids = [analysis.id for analysis in db.query(PRAnalysis).filter(PRAnalysis.repository_config_id == config_id).all()]
            if pr_analysis_ids:
                deltas = db.query(TestCaseDelta).filter(TestCaseDelta.pr_analysis_id.in_(pr_analysis_ids)).all()
                for delta in deltas:
                    db.delete(delta)
                    deltas_deleted += 1
            logger.info(f"Deleted {deltas_deleted} test case deltas")
        except Exception as e:
            logger.warning(f"Error deleting test case deltas: {e}")
        
        # 2. Delete PRAnalyses
        try:
            analyses_deleted = 0
            pr_analyses = db.query(PRAnalysis).filter(PRAnalysis.repository_config_id == config_id).all()
            for analysis in pr_analyses:
                db.delete(analysis)
                analyses_deleted += 1
            logger.info(f"Deleted {analyses_deleted} PR analyses")
        except Exception as e:
            logger.warning(f"Error deleting PR analyses: {e}")
        
        # 3. Delete IntegrationLogs
        try:
            logs_deleted = 0
            logs = db.query(IntegrationLog).filter(IntegrationLog.repository_config_id == config_id).all()
            for log in logs:
                db.delete(log)
                logs_deleted += 1
            logger.info(f"Deleted {logs_deleted} integration logs")
        except Exception as e:
            logger.warning(f"Error deleting integration logs: {e}")
        
        # 4. Finally delete the RepositoryConfig
        db.delete(config)
        
        db.commit()
        
        logger.info(f"Successfully deleted repository config {config_id} and all related data")
        return {"message": "Repository configuration and all related data deleted successfully"}
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting repository config {config_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete repository configuration: {str(e)}")

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

@router.post("/pr-analyses/{analysis_id}/trigger")
async def trigger_pr_analysis(
    analysis_id: int,
    current_user: User = Depends(require_permission(Permission.CREATE)),
    db: Session = Depends(get_db)
):
    """Manually trigger AI analysis for a PR"""
    
    pr_analysis = db.query(PRAnalysis).filter(PRAnalysis.id == analysis_id).first()
    if not pr_analysis:
        raise HTTPException(status_code=404, detail="PR analysis not found")
    
    if pr_analysis.status == "analyzing":
        raise HTTPException(status_code=400, detail="Analysis already in progress")
    
    try:
        # Reset status and trigger analysis
        pr_analysis.status = "analyzing"
        db.commit()
        
        # TODO: Add background task to run AI analysis
        # For now, just simulate the trigger
        return {
            "message": "PR analysis triggered successfully",
            "analysis_id": analysis_id,
            "status": pr_analysis.status
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger analysis: {str(e)}")
    
    # PR Analysis endpoints
@router.get("/pr-analysis/", response_model=List[PRAnalysisSchema])
async def get_pr_analyses(
    project_id: int = None,
    repository_config_id: int = None,
    status: str = None,
    analysis_type: str = None,  # 'pr', 'push', or None for both
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all code analyses (PR and push events) with optional filtering"""
    
    query = db.query(PRAnalysis)
    
    # Filter by project if specified
    if project_id:
        query = query.join(RepositoryConfig).filter(RepositoryConfig.project_id == project_id)
    
    if repository_config_id:
        query = query.filter(PRAnalysis.repository_config_id == repository_config_id)
    
    if status:
        query = query.filter(PRAnalysis.status == status)
    
    # Filter by analysis type
    if analysis_type == "pr":
        query = query.filter(PRAnalysis.pr_number > 0)
    elif analysis_type == "push":
        query = query.filter(PRAnalysis.pr_number == 0)
    # If analysis_type is None, include both PR and push analyses (pr_number >= 0)
    
    analyses = query.order_by(PRAnalysis.created_at.desc()).offset(skip).limit(limit).all()
    return analyses

@router.get("/pr-analyses/{analysis_id}/status")
async def get_analysis_status(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get the current status of an analysis with progress information"""
    
    pr_analysis = db.query(PRAnalysis).filter(PRAnalysis.id == analysis_id).first()
    if not pr_analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Count test deltas if analysis is completed
    delta_count = 0
    if pr_analysis.status == "completed" and pr_analysis.ai_analysis_results:
        deltas = pr_analysis.ai_analysis_results.get('testCaseDeltas', [])
        delta_count = len(deltas)
    
    return {
        "id": analysis_id,
        "status": pr_analysis.status,
        "pr_number": pr_analysis.pr_number,
        "analysis_type": "push" if pr_analysis.pr_number == 0 else "pr",
        "completed_at": pr_analysis.processed_at,
        "created_at": pr_analysis.created_at,
        "has_results": pr_analysis.ai_analysis_results is not None,
        "delta_count": delta_count,
        "summary": pr_analysis.ai_analysis_results.get("summary") if pr_analysis.ai_analysis_results else None
    }

@router.get("/project/{project_id}/last-analysis-update")
async def get_last_analysis_update(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get timestamp of the most recent completed analysis for a project"""
    
    latest_analysis = db.query(PRAnalysis).join(RepositoryConfig).filter(
        RepositoryConfig.project_id == project_id,
        PRAnalysis.status == "completed"
    ).order_by(PRAnalysis.processed_at.desc()).first()
    
    if latest_analysis and latest_analysis.processed_at:
        return {"last_update": latest_analysis.processed_at.isoformat()}
    
    return {"last_update": None}

@router.get("/pr-analyses/{analysis_id}/deltas", response_model=List[TestCaseDeltaSchema])
async def get_pr_analysis_deltas(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get test case deltas for a specific PR analysis"""
    
    pr_analysis = db.query(PRAnalysis).filter(PRAnalysis.id == analysis_id).first()
    if not pr_analysis:
        raise HTTPException(status_code=404, detail="PR analysis not found")
    
    deltas = db.query(TestCaseDelta).filter(
        TestCaseDelta.pull_request_analysis_id == analysis_id
    ).order_by(TestCaseDelta.created_at.desc()).all()
    
    return deltas

@router.post("/test-case-deltas/{delta_id}/apply")
async def apply_test_case_delta(
    delta_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Apply a test case delta recommendation - creates or updates a test case"""
    
    # Get the delta
    delta = db.query(TestCaseDelta).filter(TestCaseDelta.id == delta_id).first()
    if not delta:
        raise HTTPException(status_code=404, detail="Test case delta not found")
    
    if delta.is_applied:
        raise HTTPException(status_code=400, detail="Delta has already been applied")
    
    try:
        if delta.delta_type == "create":
            # Create new test case
            if not delta.suggested_suite_id:
                raise HTTPException(status_code=400, detail="Suite ID required for creating new test case")
            
            # Verify suite exists and get project_id
            test_suite = db.query(TestSuite).filter(TestSuite.id == delta.suggested_suite_id).first()
            if not test_suite:
                raise HTTPException(status_code=400, detail="Suggested test suite not found")
            
            # Generate next case number for the project
            max_case_number = db.query(func.max(TestCase.case_number)).filter(
                TestCase.project_id == test_suite.project_id
            ).scalar()
            next_case_number = (max_case_number or 0) + 1
            
            # Create test case with all required fields
            new_test_case = TestCase(
                case_number=next_case_number,
                title=delta.suggested_title or "AI Generated Test Case",
                description="Generated from PR analysis",
                preconditions=delta.suggested_preconditions,
                steps=delta.suggested_steps,
                expected_results=delta.suggested_expected_results,
                priority=delta.suggested_priority.upper() if delta.suggested_priority else "MEDIUM",
                type=delta.suggested_type.upper() if delta.suggested_type else "FUNCTIONAL",
                suite_id=delta.suggested_suite_id,
                project_id=test_suite.project_id,
                created_by_id=current_user.id,
                reference_ids=[],
                attachments=[]
            )
            db.add(new_test_case)
            db.flush()  # Get the ID
            
            # Extract section ID from reasoning field and assign to section
            suggested_section_id = None
            if delta.reasoning:
                import re
                section_match = re.search(r'\[Suggested Section:.*?\(ID: (\d+)\)\]', delta.reasoning)
                if section_match:
                    suggested_section_id = int(section_match.group(1))
                    section = db.query(Section).filter(Section.id == suggested_section_id).first()
                    if section:
                        new_test_case.sections.append(section)
                        logger.info(f"Assigned test case {new_test_case.id} to section {section.name} (ID: {suggested_section_id})")
            
            # Create initial version for the new test case (consistent with regular creation)
            version = TestCaseVersion(
                case_id=new_test_case.id,
                version_number=1,
                title=new_test_case.title,
                description=new_test_case.description,
                preconditions=new_test_case.preconditions,
                steps=new_test_case.steps,
                expected_results=new_test_case.expected_results,
                priority=new_test_case.priority,
                type=new_test_case.type,
                created_by_id=current_user.id
            )
            db.add(version)
            
            # Get the PR analysis to access repository info
            pr_analysis = db.query(PRAnalysis).filter(PRAnalysis.id == delta.pull_request_analysis_id).first()
            repository_name = pr_analysis.repository_config.repository_name if pr_analysis and pr_analysis.repository_config else "repository"
            pr_number = pr_analysis.pr_number if pr_analysis else 0
            
            # Create activity log for webhook-created test case
            activity = TestCaseActivity(
                test_case_id=new_test_case.id,
                activity_type=ActivityType.CREATED,
                description=f"Test case created via AI recommendation from {repository_name} (PR #{pr_number})",
                activity_data={
                    "action": "webhook_recommendation_applied",
                    "delta_id": delta_id,
                    "pr_analysis_id": delta.pull_request_analysis_id
                },
                created_by_id=current_user.id
            )
            db.add(activity)
            
            # Update delta to reference the new test case
            delta.test_case_id = new_test_case.id
            
        elif delta.delta_type == "update":
            # Update existing test case
            if not delta.test_case_id:
                raise HTTPException(status_code=400, detail="Test case ID required for update")
            
            test_case = db.query(TestCase).filter(TestCase.id == delta.test_case_id).first()
            if not test_case:
                raise HTTPException(status_code=400, detail="Test case to update not found")
            
            # Track changes for versioning
            content_changed = False
            old_values = {}
            
            # Apply suggested changes and track what changed
            if delta.suggested_title and delta.suggested_title != test_case.title:
                old_values['title'] = test_case.title
                test_case.title = delta.suggested_title
                content_changed = True
            if delta.suggested_preconditions and delta.suggested_preconditions != test_case.preconditions:
                old_values['preconditions'] = test_case.preconditions
                test_case.preconditions = delta.suggested_preconditions
                content_changed = True
            if delta.suggested_steps and delta.suggested_steps != test_case.steps:
                old_values['steps'] = test_case.steps
                test_case.steps = delta.suggested_steps
                content_changed = True
            if delta.suggested_expected_results and delta.suggested_expected_results != test_case.expected_results:
                old_values['expected_results'] = test_case.expected_results
                test_case.expected_results = delta.suggested_expected_results
                content_changed = True
            if delta.suggested_priority and delta.suggested_priority != test_case.priority:
                old_values['priority'] = test_case.priority
                test_case.priority = delta.suggested_priority
                content_changed = True
            if delta.suggested_type and delta.suggested_type != test_case.type:
                old_values['type'] = test_case.type
                test_case.type = delta.suggested_type
                content_changed = True
            
            test_case.updated_at = func.now()
            
            # Create new version if content changed (consistent with regular updates)
            if content_changed:
                latest_version = db.query(TestCaseVersion).filter(
                    TestCaseVersion.case_id == delta.test_case_id
                ).order_by(TestCaseVersion.version_number.desc()).first()
                
                new_version_number = (latest_version.version_number + 1) if latest_version else 1
                
                version = TestCaseVersion(
                    case_id=test_case.id,
                    version_number=new_version_number,
                    title=test_case.title,
                    description=test_case.description,
                    preconditions=test_case.preconditions,
                    steps=test_case.steps,
                    expected_results=test_case.expected_results,
                    priority=test_case.priority,
                    type=test_case.type,
                    created_by_id=current_user.id
                )
                db.add(version)
        
            # Get the PR analysis to access repository info
            pr_analysis = db.query(PRAnalysis).filter(PRAnalysis.id == delta.pull_request_analysis_id).first()
            repository_name = pr_analysis.repository_config.repository_name if pr_analysis and pr_analysis.repository_config else "repository"
            pr_number = pr_analysis.pr_number if pr_analysis else 0
            
            # Create activity log for the webhook-applied update
            activity = TestCaseActivity(
                test_case_id=test_case.id,
                activity_type=ActivityType.UPDATED,
                description=f"Test case updated via AI recommendation from {repository_name} (PR #{pr_number}){' - Version created' if content_changed else ''}",
                activity_data={
                    "action": "webhook_recommendation_applied",
                    "delta_id": delta_id,
                    "content_changed": content_changed,
                    "updated_fields": list(old_values.keys()) if old_values else [],
                    "old_values": old_values if old_values else {},
                    "pr_analysis_id": delta.pull_request_analysis_id
                },
                created_by_id=current_user.id
            )
            db.add(activity)
            
        # Mark delta as applied
        from datetime import datetime
        delta.is_applied = True
        delta.applied_at = datetime.utcnow()
        delta.applied_by_id = current_user.id
        
        db.commit()
        
        return {
            "message": "Test case delta applied successfully",
            "delta_id": delta_id,
            "test_case_id": delta.test_case_id,
            "action_taken": delta.delta_type
        }
        
    except Exception as e:
        db.rollback()
        import traceback
        error_details = traceback.format_exc()
        print(f"Apply delta error: {error_details}")  # Log to console
        raise HTTPException(status_code=500, detail=f"Failed to apply delta: {str(e)}")


@router.post("/test-case-deltas/bulk-apply")
async def bulk_apply_test_case_deltas(
    delta_ids: List[int],
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Apply multiple test case deltas in bulk"""
    
    if not delta_ids:
        raise HTTPException(status_code=400, detail="No delta IDs provided")
    
    # Get all deltas
    deltas = db.query(TestCaseDelta).filter(TestCaseDelta.id.in_(delta_ids)).all()
    
    if len(deltas) != len(delta_ids):
        found_ids = [d.id for d in deltas]
        missing_ids = [id for id in delta_ids if id not in found_ids]
        raise HTTPException(
            status_code=404, 
            detail=f"Test case deltas not found: {missing_ids}"
        )
    
    results = {"applied": [], "skipped": [], "failed": []}
    
    for delta in deltas:
        try:
            if delta.is_applied:
                results["skipped"].append({
                    "delta_id": delta.id,
                    "reason": "Already applied"
                })
                continue
            
            # Apply the delta (same logic as single apply)
            if delta.delta_type == "create":
                if not delta.suggested_suite_id:
                    results["failed"].append({
                        "delta_id": delta.id,
                        "reason": "Suite ID required for creating new test case"
                    })
                    continue
                
                test_suite = db.query(TestSuite).filter(TestSuite.id == delta.suggested_suite_id).first()
                if not test_suite:
                    results["failed"].append({
                        "delta_id": delta.id,
                        "reason": "Suggested test suite not found"
                    })
                    continue
                
                # Generate next case number for the project
                max_case_number = db.query(func.max(TestCase.case_number)).filter(
                    TestCase.project_id == test_suite.project_id
                ).scalar()
                next_case_number = (max_case_number or 0) + 1
                
                new_test_case = TestCase(
                    case_number=next_case_number,
                    title=delta.suggested_title or "AI Generated Test Case",
                    description="Generated from PR analysis",
                    preconditions=delta.suggested_preconditions,
                    steps=delta.suggested_steps,
                    expected_results=delta.suggested_expected_results,
                    priority=delta.suggested_priority.upper() if delta.suggested_priority else "MEDIUM",
                    type=delta.suggested_type.upper() if delta.suggested_type else "FUNCTIONAL",
                    suite_id=delta.suggested_suite_id,
                    project_id=test_suite.project_id,
                    created_by_id=current_user.id,
                    reference_ids=[],
                    attachments=[]
                )
                db.add(new_test_case)
                db.flush()
                delta.test_case_id = new_test_case.id
                
            elif delta.delta_type == "update":
                if not delta.test_case_id:
                    results["failed"].append({
                        "delta_id": delta.id,
                        "reason": "Test case ID required for update"
                    })
                    continue
                
                test_case = db.query(TestCase).filter(TestCase.id == delta.test_case_id).first()
                if not test_case:
                    results["failed"].append({
                        "delta_id": delta.id,
                        "reason": "Test case to update not found"
                    })
                    continue
                
                # Apply changes
                if delta.suggested_title:
                    test_case.title = delta.suggested_title
                if delta.suggested_preconditions:
                    test_case.preconditions = delta.suggested_preconditions
                if delta.suggested_steps:
                    test_case.steps = delta.suggested_steps
                if delta.suggested_expected_results:
                    test_case.expected_results = delta.suggested_expected_results
                if delta.suggested_priority:
                    test_case.priority = delta.suggested_priority
                if delta.suggested_type:
                    test_case.type = delta.suggested_type
                
                test_case.updated_at = func.now()
            
            # Mark as applied
            from datetime import datetime
            delta.is_applied = True
            delta.applied_at = datetime.utcnow()
            delta.applied_by_id = current_user.id
            
            results["applied"].append({
                "delta_id": delta.id,
                "test_case_id": delta.test_case_id,
                "action_taken": delta.delta_type
            })
            
        except Exception as e:
            results["failed"].append({
                "delta_id": delta.id,
                "reason": str(e)
            })
    
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to commit bulk apply: {str(e)}")
    
    return {
        "message": f"Bulk apply completed: {len(results['applied'])} applied, {len(results['skipped'])} skipped, {len(results['failed'])} failed",
        "results": results
    }


@router.put("/test-case-deltas/{delta_id}/dismiss")
async def dismiss_test_case_delta(
    delta_id: int,
    reason: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Dismiss a test case delta recommendation"""
    
    delta = db.query(TestCaseDelta).filter(TestCaseDelta.id == delta_id).first()
    if not delta:
        raise HTTPException(status_code=404, detail="Test case delta not found")
    
    if delta.is_applied:
        raise HTTPException(status_code=400, detail="Cannot dismiss an already applied delta")
    
    # Add dismissed status to delta (we can use confidence = -1 to mark as dismissed)
    delta.confidence = -1  # Mark as dismissed
    delta.reasoning = f"DISMISSED: {reason or 'No reason provided'}"
    
    # Log the dismissal
    from datetime import datetime
    log_entry = IntegrationLog(
        pull_request_analysis_id=delta.pull_request_analysis_id,
        event_type="delta_dismissed",
        event_data={
            "delta_id": delta_id,
            "reason": reason,
            "dismissed_by": current_user.id
        },
        success=True,
        created_at=datetime.utcnow()
    )
    db.add(log_entry)
    
    db.commit()
    
    return {
        "message": "Test case delta dismissed successfully",
        "delta_id": delta_id,
        "reason": reason
    }