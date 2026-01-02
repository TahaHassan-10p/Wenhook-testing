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

# PR Analysis Management
@router.post("/pr-analysis/", response_model=PRAnalysisSchema)
async def create_pr_analysis(
    analysis_data: PRAnalysisCreate,
    current_user: User = Depends(require_permission(Permission.CREATE)),
    db: Session = Depends(get_db)
):
    """Create a new PR analysis"""
    
    # Validate repository config exists
    repo_config = db.query(RepositoryConfig).filter(RepositoryConfig.id == analysis_data.repository_config_id).first()
    if not repo_config:
        raise HTTPException(status_code=404, detail="Repository configuration not found")
    
    pr_analysis = PRAnalysis(
        repository_config_id=analysis_data.repository_config_id,
        pr_number=analysis_data.pr_number,
        pr_title=analysis_data.pr_title,
        pr_description=analysis_data.pr_description,
        pr_url=analysis_data.pr_url,
        source_branch=analysis_data.source_branch,
        target_branch=analysis_data.target_branch,
        author=analysis_data.author,
        status="pending",
        created_by_id=current_user.id
    )
    
    db.add(pr_analysis)
    db.commit()
    db.refresh(pr_analysis)
    
    return pr_analysis

@router.get("/pr-analysis/", response_model=List[PRAnalysisSchema])
async def get_pr_analyses(
    repository_config_id: int = None,
    status: str = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get PR analyses with optional filtering"""
    
    query = db.query(PRAnalysis)
    
    if repository_config_id:
        query = query.filter(PRAnalysis.repository_config_id == repository_config_id)
    
    if status:
        query = query.filter(PRAnalysis.status == status)
    
    analyses = query.order_by(PRAnalysis.created_at.desc()).offset(skip).limit(limit).all()
    return analyses

@router.get("/pr-analysis/{analysis_id}", response_model=PRAnalysisSchema)
async def get_pr_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get a specific PR analysis"""
    
    analysis = db.query(PRAnalysis).filter(PRAnalysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="PR analysis not found")
    
    return analysis

# Webhook Endpoints
@router.post("/webhooks/github/{config_id}")
async def github_webhook(
    config_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """Handle GitHub webhook events"""
    
    # Get repository config
    repo_config = db.query(RepositoryConfig).filter(
        RepositoryConfig.id == config_id,
        RepositoryConfig.provider == "github",
        RepositoryConfig.is_active == True
    ).first()
    
    if not repo_config:
        raise HTTPException(status_code=404, detail="Repository configuration not found")
    
    # Verify webhook signature
    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")
    
    if not verify_github_signature(body, signature, repo_config.webhook_secret):
        raise HTTPException(status_code=403, detail="Invalid webhook signature")
    
    # Parse webhook payload
    payload = await request.json()
    event_type = request.headers.get("X-GitHub-Event")
    
    # Handle different event types
    if event_type == "pull_request":
        return await handle_github_pr_event(payload, repo_config, db)
    elif event_type == "push":
        return await handle_github_push_event(payload, repo_config, db)
    else:
        return {"message": f"Event type '{event_type}' not handled"}

@router.post("/webhooks/gitlab/{config_id}")
async def gitlab_webhook(
    config_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """Handle GitLab webhook events"""
    
    # Get repository config
    repo_config = db.query(RepositoryConfig).filter(
        RepositoryConfig.id == config_id,
        RepositoryConfig.provider == "gitlab",
        RepositoryConfig.is_active == True
    ).first()
    
    if not repo_config:
        raise HTTPException(status_code=404, detail="Repository configuration not found")
    
    # Verify webhook token
    token = request.headers.get("X-Gitlab-Token")
    if token != repo_config.webhook_secret:
        raise HTTPException(status_code=403, detail="Invalid webhook token")
    
    # Parse webhook payload
    payload = await request.json()
    event_type = request.headers.get("X-Gitlab-Event")
    
    # Handle different event types
    if event_type == "Merge Request Hook":
        return await handle_gitlab_mr_event(payload, repo_config, db)
    elif event_type == "Push Hook":
        return await handle_gitlab_push_event(payload, repo_config, db)
    else:
        return {"message": f"Event type '{event_type}' not handled"}

# Webhook Helper Functions
def verify_github_signature(body: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub webhook signature"""
    if not signature or not signature.startswith("sha256="):
        return False
    
    expected_signature = "sha256=" + hmac.new(
        secret.encode(),
        body,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected_signature)

async def handle_github_pr_event(payload: Dict[str, Any], repo_config: RepositoryConfig, db: Session):
    """Handle GitHub pull request events"""
    
    action = payload.get("action")
    pr_data = payload.get("pull_request", {})
    
    if action in ["opened", "synchronize", "reopened"] and repo_config.auto_analyze_prs:
        # Create or update PR analysis
        existing_analysis = db.query(PRAnalysis).filter(
            PRAnalysis.repository_config_id == repo_config.id,
            PRAnalysis.pr_number == pr_data.get("number")
        ).first()
        
        if existing_analysis:
            # Update existing analysis
            existing_analysis.status = "pending"
            existing_analysis.pr_title = pr_data.get("title")
            existing_analysis.pr_description = pr_data.get("body")
            existing_analysis.updated_at = db.func.now()
        else:
            # Create new analysis
            pr_analysis = PRAnalysis(
                repository_config_id=repo_config.id,
                pr_number=pr_data.get("number"),
                pr_title=pr_data.get("title"),
                pr_description=pr_data.get("body"),
                pr_url=pr_data.get("html_url"),
                source_branch=pr_data.get("head", {}).get("ref"),
                target_branch=pr_data.get("base", {}).get("ref"),
                author=pr_data.get("user", {}).get("login"),
                status="pending"
            )
            db.add(pr_analysis)
        
        db.commit()
        
        # TODO: Queue background task for PR analysis
        
    return {"message": "GitHub PR event processed successfully"}

async def handle_github_push_event(payload: Dict[str, Any], repo_config: RepositoryConfig, db: Session):
    """Handle GitHub push events"""
    
    # Extract push information
    ref = payload.get("ref", "")
    branch = ref.replace("refs/heads/", "") if ref.startswith("refs/heads/") else ref
    
    # Check if branch matches patterns
    if repo_config.branch_patterns:
        # TODO: Implement branch pattern matching
        pass
    
    return {"message": "GitHub push event processed successfully"}

async def handle_gitlab_mr_event(payload: Dict[str, Any], repo_config: RepositoryConfig, db: Session):
    """Handle GitLab merge request events"""
    
    action = payload.get("object_attributes", {}).get("action")
    mr_data = payload.get("object_attributes", {})
    
    if action in ["open", "update", "reopen"] and repo_config.auto_analyze_prs:
        # Create or update MR analysis
        existing_analysis = db.query(PRAnalysis).filter(
            PRAnalysis.repository_config_id == repo_config.id,
            PRAnalysis.pr_number == mr_data.get("iid")
        ).first()
        
        if existing_analysis:
            # Update existing analysis
            existing_analysis.status = "pending"
            existing_analysis.pr_title = mr_data.get("title")
            existing_analysis.pr_description = mr_data.get("description")
            existing_analysis.updated_at = db.func.now()
        else:
            # Create new analysis
            pr_analysis = PRAnalysis(
                repository_config_id=repo_config.id,
                pr_number=mr_data.get("iid"),
                pr_title=mr_data.get("title"),
                pr_description=mr_data.get("description"),
                pr_url=mr_data.get("url"),
                source_branch=mr_data.get("source_branch"),
                target_branch=mr_data.get("target_branch"),
                author=payload.get("user", {}).get("username"),
                status="pending"
            )
            db.add(pr_analysis)
        
        db.commit()
        
        # TODO: Queue background task for MR analysis
        
    return {"message": "GitLab MR event processed successfully"}

async def handle_gitlab_push_event(payload: Dict[str, Any], repo_config: RepositoryConfig, db: Session):
    """Handle GitLab push events"""
    
    # Extract push information
    ref = payload.get("ref", "")
    branch = ref.replace("refs/heads/", "") if ref.startswith("refs/heads/") else ref
    
    # Check if branch matches patterns
    if repo_config.branch_patterns:
        # TODO: Implement branch pattern matching
        pass
    
    return {"message": "GitLab push event processed successfully"}

# Test Case Delta Management
@router.get("/pr-analyses/{analysis_id}/deltas", response_model=List[TestCaseDeltaSchema])
async def get_test_deltas(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get test case deltas for a PR analysis"""
    
    pr_analysis = db.query(PRAnalysis).filter(PRAnalysis.id == analysis_id).first()
    if not pr_analysis:
        raise HTTPException(status_code=404, detail="PR analysis not found")
    
    deltas = db.query(TestCaseDelta).filter(
        TestCaseDelta.pull_request_analysis_id == analysis_id
    ).all()
    
    return deltas

@router.post("/test-deltas/{delta_id}/apply")
async def apply_test_delta(
    delta_id: int,
    current_user: User = Depends(require_permission(Permission.CREATE)),
    db: Session = Depends(get_db)
):
    """Apply a test case delta (create/update test case)"""
    
    delta = db.query(TestCaseDelta).filter(TestCaseDelta.id == delta_id).first()
    if not delta:
        raise HTTPException(status_code=404, detail="Test delta not found")
    
    if delta.is_applied:
        raise HTTPException(status_code=400, detail="Delta already applied")
    
    try:
        if delta.delta_type == "create":
            # Create new test case
            new_test_case = TestCase(
                title=delta.suggested_title,
                preconditions=delta.suggested_preconditions,
                steps=delta.suggested_steps,
                expected_results=delta.suggested_expected_results,
                priority=delta.suggested_priority,
                type=delta.suggested_type,
                suite_id=delta.suggested_suite_id,
                created_by_id=current_user.id
            )
            db.add(new_test_case)
            db.flush()
            
            # Update delta to reference the new test case
            delta.test_case_id = new_test_case.id
            
        elif delta.delta_type == "update" and delta.test_case_id:
            # Update existing test case
            test_case = db.query(TestCase).filter(TestCase.id == delta.test_case_id).first()
            if test_case:
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
        
        # Mark delta as applied
        delta.is_applied = True
        delta.applied_at = db.func.now()
        delta.applied_by_id = current_user.id
        
        db.commit()
        
        return {
            "message": "Test delta applied successfully",
            "delta_id": delta_id,
            "test_case_id": delta.test_case_id
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to apply test delta: {str(e)}")

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
        pr_analysis.status = "pending"
        db.commit()
        
        # Run AI analysis
        success = await GitHubIntegrationService.analyze_pull_request(pr_analysis, db)
        
        if success:
            return {
                "message": "PR analysis triggered successfully",
                "analysis_id": analysis_id,
                "status": pr_analysis.status
            }
        else:
            raise HTTPException(status_code=500, detail="Analysis failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger analysis: {str(e)}")

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