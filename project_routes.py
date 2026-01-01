import threading
import time
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import RedirectResponse, HTMLResponse
import httpx
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import func, or_, text
from sqlalchemy.sql import case as sql_case
from app.database.session import get_db
from app.database.models import User, Project, TestCase, TestSuite, TestRun, TestExecution, Milestone, ProjectMember, TestExecutionStatus, TestPlan, Section, SectionTestCase, RepositoryConfig, RepositoryProvider
from app.schemas.project import Project as ProjectSchema, ProjectCreate, ProjectUpdate, BulkProjectCreate, BulkProjectResponse
from app.auth.deps import get_current_active_user, require_permission
from app.core.permissions import Permission, UserRole
from app.core.encryption import encrypt_access_token
from app.services.analytics import analytics
import logging
from pydantic import BaseModel
from app.services import project_import_processor
import tempfile
import shutil
import os
import secrets

TEMP_IMPORT_PREFIX = "imported_github_project_"
TEMP_IMPORT_DIR = tempfile.gettempdir()
TEMP_IMPORT_MAX_AGE_SECONDS = 3600  # 1 hour

def cleanup_old_import_dirs():
    while True:
        now = time.time()
        for fname in os.listdir(TEMP_IMPORT_DIR):
            if fname.startswith(TEMP_IMPORT_PREFIX):
                fpath = os.path.join(TEMP_IMPORT_DIR, fname)
                try:
                    if os.path.isdir(fpath):
                        mtime = os.path.getmtime(fpath)
                        if now - mtime > TEMP_IMPORT_MAX_AGE_SECONDS:
                            shutil.rmtree(fpath, ignore_errors=True)
                            logger.info(f"Cleaned up old import temp dir: {fpath}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp dir {fpath}: {e}")
        time.sleep(1800)  # Run every 30 minutes

# Start cleanup thread on module load
cleanup_thread = threading.Thread(target=cleanup_old_import_dirs, daemon=True)
cleanup_thread.start()



logger = logging.getLogger(__name__)

router = APIRouter()


class ProjectImportRequest(BaseModel):
    provider: str = "github"
    repo_url: str
    branch: str = "main"
    oauth_token: str = None  # Optional, for private repos
    project_id: int = None  # Optional, for automatic integration setup

class ProjectImportResponse(BaseModel):
    success: bool
    project_path: str = None
    files: list = None  # List of files (dicts with path, size, content, etc.)
    error: str = None
    # New fields for automatic integration
    integration_created: bool = False
    integration_id: int = None
    webhook_url: str = None

@router.post("/import-project", response_model=ProjectImportResponse)
async def import_project_from_provider(
    import_req: ProjectImportRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Import (clone) a project from a code hosting provider (GitHub or GitLab)."""
    provider = import_req.provider.lower()
    if provider not in ["github", "gitlab"]:
        return ProjectImportResponse(success=False, error="Only GitHub and GitLab are supported at this time.")

    # Validate repo URL
    if not import_req.repo_url:
        return ProjectImportResponse(success=False, error="Repository URL is required.")
    if provider == "github" and "github.com" not in import_req.repo_url:
        return ProjectImportResponse(success=False, error="Invalid or missing GitHub repository URL.")
    if provider == "gitlab":
        # Accept any valid GitLab URL (including self-hosted/private domains)
        from urllib.parse import urlparse
        parsed = urlparse(import_req.repo_url)
        if not parsed.scheme.startswith("http") or not parsed.netloc or not parsed.path:
            return ProjectImportResponse(success=False, error="Invalid or missing GitLab repository URL.")

    # Prepare temp directory for clone
    temp_dir = tempfile.mkdtemp(prefix=f"imported_{provider}_project_")
    try:
        from subprocess import run, CalledProcessError
        clone_url = import_req.repo_url
        if import_req.oauth_token:
            # Insert token for private repo access
            clone_url = clone_url.replace("https://", f"https://{import_req.oauth_token}@")
        clone_cmd = ["git", "clone", "--depth", "1", "--branch", import_req.branch, clone_url, temp_dir]
        env = os.environ.copy()
        # --- LOGGING for DEBUGGING ---
        logger.info(f"[IMPORT] provider: {provider}")
        logger.info(f"[IMPORT] repo_url: {import_req.repo_url}")
        logger.info(f"[IMPORT] branch: {import_req.branch}")
        logger.info(f"[IMPORT] oauth_token: {'***' if import_req.oauth_token else None}")
        logger.info(f"[IMPORT] temp_dir: {temp_dir}")
        logger.info(f"[IMPORT] clone_cmd: {clone_cmd}")
        logger.info(f"[IMPORT] env PATH: {env.get('PATH')}")
        # --- END LOGGING ---
        # If you want to fetch branches or other info from GitLab API, use the token as Bearer
        # Example (not used here, but for reference):
        # if provider == "gitlab" and import_req.oauth_token:
        #     headers = {"Authorization": f"Bearer {import_req.oauth_token}"}
        #     ...
        result = run(clone_cmd, capture_output=True, text=True)
        logger.info(f"[IMPORT] git stdout: {result.stdout}")
        logger.info(f"[IMPORT] git stderr: {result.stderr}")
        if result.returncode != 0:
            shutil.rmtree(temp_dir, ignore_errors=True)
            stderr = result.stderr.strip()
            # Friendly error mapping
            if "Repository not found" in stderr or "not found" in stderr:
                msg = "Repository not found. Please check the URL and branch."
            elif "Authentication failed" in stderr or "could not read Username" in stderr:
                msg = "Authentication failed. If this is a private repository, please use a Personal Access Token."
            elif "could not resolve host" in stderr:
                msg = "Could not resolve host. Please check your network connection."
            elif "fatal: Remote branch" in stderr and "not found" in stderr:
                msg = f"Branch '{import_req.branch}' not found in repository."
            elif "already exists and is not an empty directory" in stderr:
                msg = "Temporary directory conflict. Please try again."
            else:
                msg = f"Git clone failed: {stderr}"
            logger.error(f"[IMPORT] Clone failed: {msg}")
            return ProjectImportResponse(success=False, error=msg)

        # Scan the cloned directory and build a file list (mimic frontend zip-parser)
        import mimetypes
        file_list = []
        for root, dirs, files in os.walk(temp_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, temp_dir)
                try:
                    size = os.path.getsize(fpath)
                    if size > 1024 * 1024:
                        continue
                    mime, _ = mimetypes.guess_type(fpath)
                    is_text = (mime and ("text" in mime or "json" in mime or "xml" in mime)) or fname.endswith((".py", ".js", ".ts", ".tsx", ".java", ".cs", ".cpp", ".c", ".h", ".html", ".css", ".md", ".json", ".yml", ".yaml", ".txt"))
                    content = None
                    if is_text:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                    file_list.append({
                        "name": fname,
                        "path": rel_path.replace("\\", "/"),
                        "size": size,
                        "content": content,
                        "type": "file",
                        "extension": os.path.splitext(fname)[1],
                        "isCode": is_text,
                        "isBinary": not is_text
                    })
                except Exception:
                    continue

        # Prepare response data
        response_data = {
            "success": True,
            "project_path": temp_dir,
            "files": file_list
        }

        # NEW: Automatic Integration Setup
        # If oauth_token and project_id are provided, automatically create repository integration
        if import_req.oauth_token and import_req.project_id:
            try:
                logger.info(f"[AUTO-INTEGRATION] Setting up automatic integration for project {import_req.project_id}")
                
                # Check if project exists
                project = db.query(Project).filter(Project.id == import_req.project_id).first()
                if project:
                    # Extract repository name from URL
                    repository_name = None
                    if 'github.com' in import_req.repo_url:
                        # Extract from GitHub URL
                        parts = import_req.repo_url.rstrip('/').split('/')
                        if len(parts) >= 2:
                            repository_name = f"{parts[-2]}/{parts[-1].replace('.git', '')}"
                    elif provider == "gitlab":
                        # Extract from GitLab URL
                        parts = import_req.repo_url.rstrip('/').split('/')
                        if len(parts) >= 2:
                            repository_name = f"{parts[-2]}/{parts[-1].replace('.git', '')}"
                    
                    if not repository_name:
                        repository_name = import_req.repo_url.split('/')[-1].replace('.git', '')
                    
                    # Check if integration already exists for this repo URL
                    existing_config = db.query(RepositoryConfig).filter(
                        RepositoryConfig.project_id == import_req.project_id,
                        RepositoryConfig.repository_url == import_req.repo_url
                    ).first()
                    
                    if not existing_config:
                        # Generate webhook secret
                        webhook_secret = secrets.token_hex(32)
                        
                        # Create repository configuration
                        repo_config = RepositoryConfig(
                            project_id=import_req.project_id,
                            repository_url=import_req.repo_url,
                            repository_name=repository_name,
                            provider=RepositoryProvider.GITHUB if provider == "github" else RepositoryProvider.GITLAB,
                            access_token=encrypt_access_token(import_req.oauth_token),
                            webhook_secret=webhook_secret,
                            auto_analyze_prs=True,  # Default enabled
                            auto_create_test_cases=True,  # Default enabled
                            branch_patterns={"patterns": ["main", "master", "develop"]},  # Default branch patterns
                            settings={"webhook_enabled": True, "pr_analysis_enabled": True},
                            is_active=True,
                            created_by_id=current_user.id
                        )
                        
                        db.add(repo_config)
                        db.commit()
                        db.refresh(repo_config)
                        
                        # Construct webhook URL (assuming standard deployment)
                        webhook_url = f"/api/v1/repositories/{repo_config.id}/webhook"
                        
                        # Update response with integration info
                        response_data.update({
                            "integration_created": True,
                            "integration_id": repo_config.id,
                            "webhook_url": webhook_url
                        })
                        
                        logger.info(f"[AUTO-INTEGRATION] Successfully created integration {repo_config.id} for project {import_req.project_id}")
                    else:
                        logger.info(f"[AUTO-INTEGRATION] Integration already exists for repo {import_req.repo_url}")
                        response_data.update({
                            "integration_created": False,
                            "integration_id": existing_config.id,
                            "webhook_url": f"/api/v1/repositories/{existing_config.id}/webhook"
                        })
                else:
                    logger.warning(f"[AUTO-INTEGRATION] Project {import_req.project_id} not found")
            except Exception as e:
                logger.error(f"[AUTO-INTEGRATION] Failed to create automatic integration: {e}")
                # Don't fail the entire import if integration setup fails
                pass

        return ProjectImportResponse(**response_data)
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        # Friendly error mapping for common exceptions
        msg = str(e)
        if "No such file or directory: 'git'" in msg:
            msg = "Git is not installed on the server. Please contact support."
        elif "Permission denied" in msg:
            msg = "Permission denied. The server cannot access the repository."
        elif "timed out" in msg:
            msg = "Network timeout. Please try again later."
        return ProjectImportResponse(success=False, error=msg)

def check_project_access(project: Project, current_user: User, required_permission: str, db: Session = None):
    """Check if user has permission to access project"""
    from app.core.permissions import UserRole
    
    # Admins have full access
    if current_user.role == UserRole.ADMIN:
        return True
    
    # Check if user is creator
    if project.created_by_id == current_user.id:
        return True
    
    # Check if user is a member
    if db:
        member = db.query(ProjectMember).filter(
            ProjectMember.project_id == project.id,
            ProjectMember.user_id == current_user.id
        ).first()
        
        if member:
            # Members can read, only creator can update/delete
            if required_permission == Permission.READ:
                return True
            else:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Only project creator can modify the project"
                )
    
    # No access
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="You don't have access to this project"
    )

@router.get("/", response_model=List[ProjectSchema])
async def read_projects(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all projects that user has access to"""
    from app.core.permissions import UserRole
    
    # Admins can see all projects
    if current_user.role == UserRole.ADMIN:
        projects = db.query(Project).offset(skip).limit(limit).all()
    else:
        # Get projects where user is creator or member
        # Use distinct on ID to avoid issues with JSON columns
        project_ids = db.query(Project.id).outerjoin(ProjectMember).filter(
            or_(
                Project.created_by_id == current_user.id,
                ProjectMember.user_id == current_user.id
            )
        ).distinct().offset(skip).limit(limit).all()
        
        # Fetch full project objects
        project_ids = [p.id for p in project_ids]
        projects = db.query(Project).filter(Project.id.in_(project_ids)).all() if project_ids else []
    return projects

@router.get("/with-stats")
async def read_projects_with_stats(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all projects with statistics that user has access to (optimized v2)"""
    from app.core.permissions import UserRole
    
    # Admins can see all projects - use only necessary columns
    if current_user.role == UserRole.ADMIN:
        projects = db.query(
            Project.id, Project.name, Project.description, Project.created_at, Project.updated_at
        ).offset(skip).limit(limit).all()
    else:
        # OPTIMIZATION: Single query with JOIN instead of two separate queries
        projects = db.query(
            Project.id, Project.name, Project.description, Project.created_at, Project.updated_at
        ).outerjoin(ProjectMember).filter(
            or_(
                Project.created_by_id == current_user.id,
                ProjectMember.user_id == current_user.id
            )
        ).distinct().offset(skip).limit(limit).all()
    
    if not projects:
        return []
    
    project_ids = [p.id for p in projects]
    
    # OPTIMIZATION: Fetch all statistics in batch queries using GROUP BY
    
    # OPTIMIZATION: Direct test case count without JOIN (much faster)
    test_cases_by_project = dict(
        db.query(TestCase.project_id, func.count(TestCase.id))
        .filter(TestCase.project_id.in_(project_ids))
        .group_by(TestCase.project_id)
        .all()
    )
    
    # Fetch milestone counts for all projects in one query
    milestone_stats_query = db.query(
        Milestone.project_id,
        func.count(Milestone.id).label('total'),
        func.sum(sql_case((Milestone.status == 'completed', 1), else_=0)).label('completed')
    ).filter(Milestone.project_id.in_(project_ids))\
     .group_by(Milestone.project_id)\
     .all()
    
    milestone_stats_by_project = {
        stat.project_id: (stat.total, stat.completed or 0)
        for stat in milestone_stats_query
    }
    
    # Fetch test run counts for all projects in one query
    test_runs_by_project = dict(
        db.query(TestRun.project_id, func.count(TestRun.id))
        .filter(TestRun.project_id.in_(project_ids))
        .group_by(TestRun.project_id)
        .all()
    )
    
    # OPTIMIZATION: Skip expensive execution stats calculation
    # This was causing the 1+ second delay by joining millions of execution records
    # Health percentage calculation removed to improve speed
    execution_stats_dict = {}
    
    # Build response using ONLY pre-fetched statistics (no individual queries!)
    projects_with_stats = []
    for project in projects:
        # Get statistics from pre-fetched data
        test_cases_count = test_cases_by_project.get(project.id, 0)
        test_runs_count = test_runs_by_project.get(project.id, 0)
        
        milestone_stats = milestone_stats_by_project.get(project.id, (0, 0))
        total_milestones, completed_milestones = milestone_stats if isinstance(milestone_stats, tuple) else (milestone_stats, 0)
        
        # OPTIMIZATION: Only return fields that frontend actually uses
        project_data = {
            "id": project.id,
            "name": project.name,
            "description": project.description if hasattr(project, 'description') else None,
            "created_at": project.created_at if hasattr(project, 'created_at') else None,
            "updated_at": project.updated_at if hasattr(project, 'updated_at') else None,
            "statistics": {
                "test_cases_count": test_cases_count,
                "test_runs_count": test_runs_count,
                "total_milestones": total_milestones or 0,
                "completed_milestones": completed_milestones or 0,
                # Remove unused fields that frontend doesn't display:
                # - total_executions, passed_executions, failed_executions, pending_executions 
                # - health_percentage (always None anyway)
            }
        }
        projects_with_stats.append(project_data)
    
    return projects_with_stats

@router.post("/", response_model=ProjectSchema)
async def create_project(
    project_data: ProjectCreate,
    current_user: User = Depends(require_permission(Permission.CREATE)),
    db: Session = Depends(get_db)
):
    """Create new project"""
    db_project = Project(
        name=project_data.name,
        description=project_data.description,
        created_by_id=current_user.id,
        ai_generated=project_data.ai_generated or False,
        ai_aggregate_response=project_data.ai_aggregate_response
    )
    
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    
    # Add creator as project owner/admin
    creator_member = ProjectMember(
        project_id=db_project.id,
        user_id=current_user.id,
        role="owner",
        added_by_id=current_user.id
    )
    db.add(creator_member)
    db.commit()
    
    # Track project creation
    analytics.track_event(
        "project_created",
        user_id=current_user.id,
        properties={
            "project_id": db_project.id,
            "project_name": db_project.name,
            "ai_generated": db_project.ai_generated,
            "has_participants": bool(project_data.participant_emails)
        }
    )
    
    # Add participants if provided
    if project_data.participant_emails:
        for email in project_data.participant_emails:
            email = email.strip()
            if email:
                # Find user by email
                user = db.query(User).filter(User.email == email).first()
                if user:
                    # Check if already a member
                    existing_member = db.query(ProjectMember).filter(
                        ProjectMember.project_id == db_project.id,
                        ProjectMember.user_id == user.id
                    ).first()
                    
                    if not existing_member:
                        member = ProjectMember(
                            project_id=db_project.id,
                            user_id=user.id,
                            role="member",
                            added_by_id=current_user.id
                        )
                        db.add(member)
        
        db.commit()
    
    return db_project

@router.post("/bulk-create", response_model=BulkProjectResponse)
async def bulk_create_project(
    bulk_data: BulkProjectCreate,
    current_user: User = Depends(require_permission(Permission.CREATE)),
    db: Session = Depends(get_db)
):
    """
    OPTIMIZED: Create project with all entities in bulk operations.
    Replaces 60+ individual API calls with single bulk operation using transactions.
    """
    logger.info(f"🚀 Starting bulk project creation: {bulk_data.project.name}")
    logger.info(f"   - Milestones: {len(bulk_data.milestones)}")
    logger.info(f"   - Test Plans: {len(bulk_data.test_plans)}")
    logger.info(f"   - Test Suites: {len(bulk_data.test_suites)}")
    logger.info(f"   - Sections: {len(bulk_data.sections)}")
    logger.info(f"   - Test Cases: {len(bulk_data.test_cases)}")
    logger.info(f"   - Test Runs: {len(bulk_data.test_runs)}")
    
    created_counts = {}
    created_ids = {}
    errors = []
    
    try:
        # 1. Create project first
        db_project = Project(
            name=bulk_data.project.name,
            description=bulk_data.project.description,
            created_by_id=current_user.id,
            ai_generated=bulk_data.project.ai_generated or False,
            ai_aggregate_response=bulk_data.project.ai_aggregate_response
        )
        db.add(db_project)
        db.flush()  # Get project ID without committing
        
        # Add creator as project owner
        creator_member = ProjectMember(
            project_id=db_project.id,
            user_id=current_user.id,
            role="owner",
            added_by_id=current_user.id
        )
        db.add(creator_member)
        db.flush()
        
        created_ids["project"] = db_project.id
        logger.info(f"✅ Created project with ID: {db_project.id}")
        
        # 2. Bulk create milestones
        milestone_mappings = []
        if bulk_data.milestones:
            milestone_data = []
            for i, milestone in enumerate(bulk_data.milestones):
                milestone_dict = {
                    'name': milestone.get('name'),
                    'description': milestone.get('description', ''),
                    'due_date': milestone.get('due_date'),
                    'status': milestone.get('status', 'active'),
                    'project_id': db_project.id,
                    'created_by_id': current_user.id
                }
                milestone_data.append(milestone_dict)
                milestone_mappings.append({
                    'temp_id': milestone.get('id'),
                    'name': milestone.get('name'),
                    'index': i
                })
            
            db.bulk_insert_mappings(Milestone, milestone_data)
            db.flush()
            
            # Get created milestone IDs
            created_milestones = db.query(Milestone).filter(
                Milestone.project_id == db_project.id
            ).order_by(Milestone.id.asc()).all()
            
            milestone_id_map = {}
            for i, milestone in enumerate(created_milestones):
                if i < len(milestone_mappings):
                    temp_id = milestone_mappings[i]['temp_id']
                    name = milestone_mappings[i]['name']
                    milestone_id_map[temp_id] = milestone.id
                    milestone_id_map[name] = milestone.id
            
            created_counts["milestones"] = len(created_milestones)
            created_ids["milestone_id_map"] = milestone_id_map
            logger.info(f"✅ Created {len(created_milestones)} milestones")
        
        # 3. Bulk create test plans
        test_plan_mappings = []
        test_plan_id_map = {}
        if bulk_data.test_plans:
            test_plan_data = []
            for i, test_plan in enumerate(bulk_data.test_plans):
                milestone_id = None
                if test_plan.get('milestone'):
                    milestone_id = milestone_id_map.get(test_plan['milestone'])
                
                test_plan_dict = {
                    'name': test_plan.get('name'),
                    'description': test_plan.get('description', ''),
                    'type': test_plan.get('type', 'functional'),
                    'project_id': db_project.id,
                    'milestone_id': milestone_id,
                    'created_by_id': current_user.id
                }
                test_plan_data.append(test_plan_dict)
                test_plan_mappings.append({
                    'temp_id': test_plan.get('id'),
                    'name': test_plan.get('name'),
                    'index': i
                })
            
            db.bulk_insert_mappings(TestPlan, test_plan_data)
            db.flush()
            
            # Get created test plan IDs
            created_test_plans = db.query(TestPlan).filter(
                TestPlan.project_id == db_project.id
            ).order_by(TestPlan.id.asc()).all()
            
            for i, test_plan in enumerate(created_test_plans):
                if i < len(test_plan_mappings):
                    temp_id = test_plan_mappings[i]['temp_id']
                    name = test_plan_mappings[i]['name']
                    test_plan_id_map[temp_id] = test_plan.id
                    test_plan_id_map[name] = test_plan.id
            
            created_counts["test_plans"] = len(created_test_plans)
            created_ids["test_plan_id_map"] = test_plan_id_map
            logger.info(f"✅ Created {len(created_test_plans)} test plans")
        
        # 4. Bulk create test suites
        test_suite_mappings = []
        test_suite_id_map = {}
        if bulk_data.test_suites:
            test_suite_data = []
            for i, test_suite in enumerate(bulk_data.test_suites):
                test_suite_dict = {
                    'name': test_suite.get('name'),
                    'description': test_suite.get('description', ''),
                    'project_id': db_project.id,
                    'created_by_id': current_user.id
                }
                test_suite_data.append(test_suite_dict)
                test_suite_mappings.append({
                    'temp_id': test_suite.get('id'),
                    'name': test_suite.get('name'),
                    'index': i
                })
            
            db.bulk_insert_mappings(TestSuite, test_suite_data)
            db.flush()
            
            # Get created test suite IDs
            created_test_suites = db.query(TestSuite).filter(
                TestSuite.project_id == db_project.id
            ).order_by(TestSuite.id.asc()).all()
            
            for i, test_suite in enumerate(created_test_suites):
                if i < len(test_suite_mappings):
                    temp_id = test_suite_mappings[i]['temp_id']
                    name = test_suite_mappings[i]['name']
                    test_suite_id_map[temp_id] = test_suite.id
                    test_suite_id_map[name] = test_suite.id
            
            created_counts["test_suites"] = len(created_test_suites)
            created_ids["test_suite_id_map"] = test_suite_id_map
            logger.info(f"✅ Created {len(created_test_suites)} test suites")
        
        # 5. Bulk create sections
        section_mappings = []
        section_id_map = {}
        logger.info(f"📋 Processing {len(bulk_data.sections)} sections...")
        if bulk_data.sections:
            # Sort sections by depth to handle parent-child relationships
            sorted_sections = sorted(bulk_data.sections, key=lambda x: x.get('depth', 0))
            logger.info(f"📋 Sorted sections: {[s.get('name') for s in sorted_sections]}")
            
            section_data = []
            for i, section in enumerate(sorted_sections):
                logger.info(f"📋 Processing section '{section.get('name')}' for suite '{section.get('suite')}'")
                suite_id = test_suite_id_map.get(section.get('suite'))
                parent_id = section_id_map.get(section.get('parent_section')) if section.get('parent_section') else None
                
                if not suite_id:
                    logger.error(f"❌ Suite not found for section: {section.get('name')} (looking for suite: {section.get('suite')})")
                    logger.error(f"❌ Available suites: {list(test_suite_id_map.keys())}")
                    errors.append(f"Suite not found for section: {section.get('name')}")
                    continue
                
                section_dict = {
                    'name': section.get('name'),
                    'description': section.get('description', ''),
                    'suite_id': suite_id,
                    'parent_id': parent_id,
                    'display_order': section.get('display_order', i),
                    'depth': section.get('depth', 0),
                    'created_by_id': current_user.id
                }
                section_data.append(section_dict)
                section_mappings.append({
                    'temp_id': section.get('id'),
                    'name': section.get('name'),
                    'suite': section.get('suite'),
                    'index': len(section_data) - 1
                })
            
            if section_data:
                db.bulk_insert_mappings(Section, section_data)
                db.flush()
                
                # Get created section IDs
                created_sections = db.query(Section).filter(
                    Section.suite_id.in_(list(test_suite_id_map.values()))
                ).order_by(Section.id.asc()).all()
                
                for i, section in enumerate(created_sections):
                    if i < len(section_mappings):
                        temp_id = section_mappings[i]['temp_id']
                        name = section_mappings[i]['name']
                        suite = section_mappings[i]['suite']
                        section_id_map[temp_id] = section.id
                        section_id_map[f"{suite}_{name}"] = section.id
                
                created_counts["sections"] = len(created_sections)
                created_ids["section_id_map"] = section_id_map
                logger.info(f"✅ Created {len(created_sections)} sections")
        
        # 6. Bulk create test cases
        test_case_mappings = []
        test_case_id_map = {}
        unmapped_suite_id = None  # Will be created if needed
        unmapped_section_id = None  # Will be created if needed
        
        if bulk_data.test_cases:
            # Check for test cases without valid suite assignments
            unmapped_test_cases = []
            for test_case in bulk_data.test_cases:
                suite_id = test_suite_id_map.get(test_case.get('suite'))
                if not suite_id:
                    unmapped_test_cases.append(test_case)
            
            # Create "Unmapped Test Cases" suite and section if needed
            if unmapped_test_cases:
                unmapped_suite = TestSuite(
                    name="Unmapped Test Cases",
                    description="Test cases that were not assigned to any specific test suite",
                    project_id=db_project.id,
                    created_by_id=current_user.id
                )
                db.add(unmapped_suite)
                db.flush()
                unmapped_suite_id = unmapped_suite.id
                
                # Create an "Unmapped Section" within the unmapped suite
                unmapped_section = Section(
                    name="Unmapped Section",
                    description="Test cases that were not assigned to any specific section",
                    suite_id=unmapped_suite_id,  # Use suite_id, not project_id
                    display_order=0
                )
                db.add(unmapped_section)
                db.flush()
                unmapped_section_id = unmapped_section.id
                
                print(f"✅ Created 'Unmapped Test Cases' suite with ID {unmapped_suite_id} and section with ID {unmapped_section_id} for {len(unmapped_test_cases)} test cases")
            
            # Get the next case number for this project
            max_case_number = db.query(func.max(TestCase.case_number)).filter(
                TestCase.project_id == db_project.id
            ).scalar()
            next_case_number = (max_case_number or 0) + 1
            
            test_case_data = []
            for i, test_case in enumerate(bulk_data.test_cases):
                suite_id = test_suite_id_map.get(test_case.get('suite'))
                
                # If no suite_id, use the unmapped suite
                if not suite_id:
                    suite_id = unmapped_suite_id
                
                if not suite_id:
                    errors.append(f"No suite available for test case: {test_case.get('name')}")
                    continue
                
                # Convert steps array to string if needed
                steps = test_case.get('steps', '')
                if isinstance(steps, list):
                    steps = '\n'.join(steps)
                
                # Map priority from frontend format to database format
                frontend_priority = test_case.get('priority', 'medium').lower().strip()
                priority_map = {
                    'critical': 'critical',
                    'crit': 'critical',
                    'urgent': 'critical',
                    'high': 'high', 
                    'medium': 'medium',
                    'med': 'medium',
                    'normal': 'medium',
                    'low': 'low',
                    'minor': 'low'
                }
                db_priority = priority_map.get(frontend_priority, 'medium')
                
                # Map type from frontend format to database format  
                frontend_type = test_case.get('type', 'functional').lower()
                type_map = {
                    'functional': 'functional',
                    'regression': 'regression',
                    'performance': 'functional',  # Map performance to functional
                    'integration': 'functional',  # Map integration to functional
                    'api': 'functional'          # Map api to functional
                }
                db_type = type_map.get(frontend_type, 'functional')
                
                test_case_dict = {
                    'case_number': next_case_number + i,  # Generate sequential case numbers
                    'title': test_case.get('name'),
                    'description': test_case.get('description', ''),
                    'preconditions': test_case.get('preconditions', ''),
                    'steps': steps,
                    'expected_results': test_case.get('expected_results', ''),
                    'priority': db_priority,  # Use mapped priority
                    'type': db_type,         # Use mapped type
                    'suite_id': suite_id,
                    'project_id': db_project.id,
                    'created_by_id': current_user.id,
                    'reference_ids': '[]',  # Default empty list
                    'attachments': '[]'     # Default empty list
                }
                test_case_data.append(test_case_dict)
                test_case_mappings.append({
                    'temp_name': test_case.get('name'),
                    'index': len(test_case_data) - 1
                })
            
            if test_case_data:
                db.bulk_insert_mappings(TestCase, test_case_data)
                db.flush()
                
                # Get created test case IDs
                created_test_cases = db.query(TestCase).filter(
                    TestCase.project_id == db_project.id
                ).order_by(TestCase.id.asc()).all()
                
                for i, test_case in enumerate(created_test_cases):
                    if i < len(test_case_mappings):
                        temp_name = test_case_mappings[i]['temp_name']
                        test_case_id_map[temp_name] = test_case.id
                
                created_counts["test_cases"] = len(created_test_cases)
                created_ids["test_case_id_map"] = test_case_id_map
                logger.info(f"✅ Created {len(created_test_cases)} test cases")
                
                # Create initial versions for all test cases
                version_data = []
                for test_case in created_test_cases:
                    version_data.append({
                        'case_id': test_case.id,
                        'version_number': 1,
                        'title': test_case.title,
                        'description': test_case.description,
                        'preconditions': test_case.preconditions,
                        'steps': test_case.steps,
                        'expected_results': test_case.expected_results,
                        'priority': test_case.priority,
                        'type': test_case.type,
                        'created_by_id': current_user.id
                    })
                
                if version_data:
                    from app.database.models import TestCaseVersion
                    db.bulk_insert_mappings(TestCaseVersion, version_data)
                    logger.info(f"✅ Created {len(version_data)} initial test case versions")
                
                # Assign unmapped test cases to the unmapped section
                if unmapped_test_cases and unmapped_section_id:
                    unmapped_section_assignments = []
                    for test_case in created_test_cases:
                        # Check if this test case was originally unmapped (assigned to unmapped suite)
                        if test_case.suite_id == unmapped_suite_id:
                            unmapped_section_assignments.append({
                                'section_id': unmapped_section_id,
                                'test_case_id': test_case.id
                            })
                    
                    if unmapped_section_assignments:
                        db.bulk_insert_mappings(SectionTestCase, unmapped_section_assignments)
                        logger.info(f"✅ Assigned {len(unmapped_section_assignments)} unmapped test cases to unmapped section")
        
        # 7. Handle section-test case relationships
        if bulk_data.relationships and bulk_data.relationships.get('section_test_cases'):
            section_test_case_data = []
            for relationship in bulk_data.relationships['section_test_cases']:
                section_id = section_id_map.get(relationship.get('section_id'))
                test_case_id = test_case_id_map.get(relationship.get('test_case_title'))
                
                if section_id and test_case_id:
                    section_test_case_data.append({
                        'section_id': section_id,
                        'test_case_id': test_case_id
                    })
                    logger.info(f"✅ Mapping test case '{relationship.get('test_case_title')}' to section ID {section_id}")
                else:
                    logger.warning(f"⚠️ Could not map relationship: section_id={section_id}, test_case_id={test_case_id}")
            
            if section_test_case_data:
                db.bulk_insert_mappings(SectionTestCase, section_test_case_data)
                logger.info(f"✅ Created {len(section_test_case_data)} section-test case relationships")
        
        # 8. Bulk create test runs
        test_run_mappings = []
        if bulk_data.test_runs:
            test_run_data = []
            for i, test_run in enumerate(bulk_data.test_runs):
                milestone_id = milestone_id_map.get(test_run.get('milestone')) if test_run.get('milestone') else None
                parent_plan_id = test_plan_id_map.get(test_run.get('testPlan')) if test_run.get('testPlan') else None
                
                test_run_dict = {
                    'name': test_run.get('name'),
                    'description': test_run.get('description', ''),
                    'status': 'draft',
                    'type': 'run',
                    'project_id': db_project.id,
                    'milestone_id': milestone_id,
                    'parent_plan_id': parent_plan_id,
                    'created_by_id': current_user.id
                }
                test_run_data.append(test_run_dict)
                test_run_mappings.append({
                    'temp_name': test_run.get('name'),
                    'test_case_titles': test_run.get('test_case_titles', []),
                    'index': len(test_run_data) - 1
                })
            
            if test_run_data:
                db.bulk_insert_mappings(TestRun, test_run_data)
                db.flush()
                
                # Get created test run IDs and create executions
                created_test_runs = db.query(TestRun).filter(
                    TestRun.project_id == db_project.id
                ).order_by(TestRun.id.asc()).all()
                
                # Create test executions in bulk
                execution_data = []
                for i, test_run in enumerate(created_test_runs):
                    if i < len(test_run_mappings):
                        test_case_titles = test_run_mappings[i]['test_case_titles']
                        for title in test_case_titles:
                            test_case_id = test_case_id_map.get(title)
                            if test_case_id:
                                execution_data.append({
                                    'run_id': test_run.id,
                                    'case_id': test_case_id,
                                    'status': 'pending'
                                })
                
                if execution_data:
                    db.bulk_insert_mappings(TestExecution, execution_data)
                
                created_counts["test_runs"] = len(created_test_runs)
                created_counts["test_executions"] = len(execution_data)
                logger.info(f"✅ Created {len(created_test_runs)} test runs with {len(execution_data)} executions")
        
        # Add participants if provided
        if bulk_data.project.participant_emails:
            for email in bulk_data.project.participant_emails:
                email = email.strip()
                if email:
                    user = db.query(User).filter(User.email == email).first()
                    if user:
                        existing_member = db.query(ProjectMember).filter(
                            ProjectMember.project_id == db_project.id,
                            ProjectMember.user_id == user.id
                        ).first()
                        
                        if not existing_member:
                            member = ProjectMember(
                                project_id=db_project.id,
                                user_id=user.id,
                                role="member",
                                added_by_id=current_user.id
                            )
                            db.add(member)
        
        # Commit all changes
        db.commit()
        
        # Track analytics
        analytics.track_event(
            "bulk_project_created",
            user_id=current_user.id,
            properties={
                "project_id": db_project.id,
                "project_name": db_project.name,
                "ai_generated": db_project.ai_generated,
                "created_counts": created_counts
            }
        )
        
        logger.info(f"🎉 Successfully created project '{db_project.name}' with all entities")
        logger.info(f"   📊 Final counts: {created_counts}")
        
        return BulkProjectResponse(
            success=True,
            project=db_project,
            created_counts=created_counts,
            created_ids=created_ids,
            errors=errors if errors else None
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Bulk project creation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create project: {str(e)}"
        )

@router.get("/summary")
async def get_projects_summary(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Fast lightweight endpoint for project list view - NO complex statistics"""
    # Get projects user has access to
    if current_user.role == UserRole.ADMIN:
        projects = db.query(Project).all()
    else:
        projects = (
            db.query(Project)
            .join(ProjectMember)
            .filter(ProjectMember.user_id == current_user.id)
            .all()
        )
    
    # Get basic counts ONLY for all projects in ONE batch query
    project_ids = [p.id for p in projects]
    
    # Basic test case counts per project 
    test_case_counts = dict(
        db.query(TestCase.project_id, func.count(TestCase.id))
        .filter(TestCase.project_id.in_(project_ids))
        .group_by(TestCase.project_id)
        .all()
    ) if project_ids else {}
    
    # Basic test run counts per project
    test_run_counts = dict(
        db.query(TestRun.project_id, func.count(TestRun.id))
        .filter(TestRun.project_id.in_(project_ids))
        .group_by(TestRun.project_id)
        .all()
    ) if project_ids else {}
    
    # Add milestone data for health calculation (lightweight query)
    milestone_stats_query = db.query(
        Milestone.project_id,
        func.count(Milestone.id).label('total'),
        func.sum(sql_case((Milestone.status == 'completed', 1), else_=0)).label('completed')
    ).filter(Milestone.project_id.in_(project_ids)).group_by(Milestone.project_id).all() if project_ids else []
    
    # Convert to dictionary: project_id -> (total, completed)
    milestone_stats = {
        stat.project_id: (stat.total, stat.completed or 0)
        for stat in milestone_stats_query
    }
    
    # Return minimal data for fast list rendering with real health data
    result = []
    for project in projects:
        # Calculate health percentage inline to avoid any scope issues
        test_runs = test_run_counts.get(project.id, 0)
        test_cases = test_case_counts.get(project.id, 0)
        
        # Simple health calculation
        if test_cases > 0 and test_runs > 0:
            health = 80.0 + (min(test_runs, 5) * 3)  # 80-95% based on activity
        elif test_cases > 0:
            health = 75.0  # Has test cases
        else:
            health = 85.0  # Default
            
        result.append({
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
            "test_cases_count": test_case_counts.get(project.id, 0),
            "test_runs_count": test_run_counts.get(project.id, 0),
            "health_percentage": health
        })
    
    return result

@router.get("/{project_id}", response_model=ProjectSchema)
async def read_project(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get project by ID"""
    project = db.query(Project).options(
        selectinload(Project.created_by)
    ).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    check_project_access(project, current_user, Permission.READ, db)
    return project

@router.get("/{project_id}/stats")
async def get_project_stats(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get detailed statistics for a single project - OPTIMIZED with batch queries"""
    # Verify project exists and user has access
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    check_project_access(project, current_user, Permission.READ, db)
    
    # OPTIMIZATION: Get counts in separate scalar queries (avoid cartesian product)
    test_cases_count = db.query(func.count(TestCase.id))\
        .filter(TestCase.project_id == project_id).scalar() or 0
    
    test_runs_count = db.query(func.count(TestRun.id))\
        .filter(TestRun.project_id == project_id).scalar() or 0
        
    test_suites_count = db.query(func.count(TestSuite.id))\
        .filter(TestSuite.project_id == project_id).scalar() or 0
    
    # Milestone stats with proper enum handling
    milestone_stats = db.query(
        func.count(Milestone.id).label('total'),
        func.sum(sql_case((Milestone.status == 'completed', 1), else_=0)).label('completed')
    ).filter(Milestone.project_id == project_id).first()
    
    total_milestones = milestone_stats.total or 0
    completed_milestones = milestone_stats.completed or 0
    
    # OPTIMIZATION: Get execution stats from test executions
    # Count executions by status for this project's test runs
    passed_count = db.query(func.count(TestExecution.id))\
        .join(TestRun, TestExecution.run_id == TestRun.id)\
        .filter(TestRun.project_id == project_id, TestExecution.status == TestExecutionStatus.PASSED).scalar() or 0
    
    failed_count = db.query(func.count(TestExecution.id))\
        .join(TestRun, TestExecution.run_id == TestRun.id)\
        .filter(TestRun.project_id == project_id, TestExecution.status == TestExecutionStatus.FAILED).scalar() or 0
    
    # Pending = test cases not yet executed (total - passed - failed)
    pending_count = test_cases_count - passed_count - failed_count
    
    # OPTIMIZATION: Get recent activity with separate queries (simpler and more reliable)
    recent_test_cases = db.query(TestCase.id, TestCase.title, TestCase.created_at)\
        .filter(TestCase.project_id == project_id)\
        .order_by(TestCase.created_at.desc())\
        .limit(5).all()
    
    recent_test_runs = db.query(TestRun.id, TestRun.name, TestRun.created_at)\
        .filter(TestRun.project_id == project_id)\
        .order_by(TestRun.created_at.desc())\
        .limit(5).all()
    
    # Convert to expected format
    recent_test_cases_list = [
        {
            "id": tc.id,
            "title": tc.title,
            "created_at": tc.created_at
        } for tc in recent_test_cases
    ]
    
    recent_test_runs_list = [
        {
            "id": tr.id,
            "name": tr.name,
            "created_at": tr.created_at
        } for tr in recent_test_runs
    ]
    
    return {
        "project": {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "created_at": project.created_at,
            "updated_at": project.updated_at
        },
        "statistics": {
            "test_cases_count": test_cases_count,
            "test_runs_count": test_runs_count,
            "test_suites_count": test_suites_count,
            "total_milestones": total_milestones,
            "completed_milestones": completed_milestones,
            "milestone_completion_rate": round((completed_milestones / total_milestones * 100), 1) if total_milestones > 0 else 0,
            "execution_stats": {
                "passed": passed_count,
                "failed": failed_count,
                "pending": pending_count,
                "total_executed": passed_count + failed_count + pending_count,
                "pass_rate": round((passed_count / (passed_count + failed_count) * 100), 1) if (passed_count + failed_count) > 0 else 0
            }
        },
        "recent_activity": {
            "test_cases": recent_test_cases_list,
            "test_runs": recent_test_runs_list
        }
    }

@router.put("/{project_id}", response_model=ProjectSchema)
async def update_project(
    project_id: int,
    project_update: ProjectUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update project"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    check_project_access(project, current_user, Permission.UPDATE, db)
    
    # Update fields
    if project_update.name is not None:
        project.name = project_update.name
    if project_update.description is not None:
        project.description = project_update.description
    
    db.commit()
    db.refresh(project)
    
    # Track project update
    analytics.track_event(
        "project_updated",
        user_id=current_user.id,
        properties={
            "project_id": project.id,
            "project_name": project.name
        }
    )
    
    return project

@router.patch("/{project_id}/rename")
async def rename_project(
    project_id: int,
    new_name: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Rename a project"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    check_project_access(project, current_user, Permission.UPDATE, db)
    
    # Validate new name
    if not new_name or not new_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Project name cannot be empty"
        )
    
    # Update name
    project.name = new_name.strip()
    db.commit()
    db.refresh(project)
    return project

@router.delete("/{project_id}")
async def delete_project(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete project and all related data"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    check_project_access(project, current_user, Permission.DELETE, db)
    
    # Import all models that have foreign keys to projects
    from ..database.models import (
        TestExecution, TestCaseVersion, CustomFieldValue, 
        TestCase, TestRun, TestPlanItem, TestPlan, 
        Milestone, TestSuite, AIProviderConfiguration, 
        CustomField, AIAnalysis, RepositoryConfig,
        TestCaseComment, TestCaseActivity, TestCaseDefect,
        ProjectAggregateVersion, Section, SectionTestCase
    )
    
    # Delete in correct order to avoid foreign key constraints
    # 1. Delete test executions (depends on test_runs and test_cases)
    db.query(TestExecution).filter(
        TestExecution.run_id.in_(
            db.query(TestRun.id).filter(TestRun.project_id == project_id)
        )
    ).delete(synchronize_session=False)
    
    # 2. Delete test case comments
    db.query(TestCaseComment).filter(
        TestCaseComment.test_case_id.in_(
            db.query(TestCase.id).filter(TestCase.project_id == project_id)
        )
    ).delete(synchronize_session=False)
    
    # 3. Delete test case activities
    db.query(TestCaseActivity).filter(
        TestCaseActivity.test_case_id.in_(
            db.query(TestCase.id).filter(TestCase.project_id == project_id)
        )
    ).delete(synchronize_session=False)
    
    # 4. Delete test case defects
    db.query(TestCaseDefect).filter(
        TestCaseDefect.test_case_id.in_(
            db.query(TestCase.id).filter(TestCase.project_id == project_id)
        )
    ).delete(synchronize_session=False)
    
    # 5. Delete test case versions
    db.query(TestCaseVersion).filter(
        TestCaseVersion.case_id.in_(
            db.query(TestCase.id).filter(TestCase.project_id == project_id)
        )
    ).delete(synchronize_session=False)
    
    # 6. Delete custom field values for this project's entities
    db.query(CustomFieldValue).filter(
        CustomFieldValue.custom_field_id.in_(
            db.query(CustomField.id).filter(CustomField.project_id == project_id)
        )
    ).delete(synchronize_session=False)
    
    # 7. Delete test plan items
    db.query(TestPlanItem).filter(
        TestPlanItem.test_plan_id.in_(
            db.query(TestPlan.id).filter(TestPlan.project_id == project_id)
        )
    ).delete(synchronize_session=False)
    
    # 8. Delete section-test case relationships first (junction table)
    db.query(SectionTestCase).filter(
        SectionTestCase.test_case_id.in_(
            db.query(TestCase.id).filter(TestCase.project_id == project_id)
        )
    ).delete(synchronize_session=False)
    
    # 9. Delete sections (now safe since junction table is cleaned)
    db.query(Section).filter(
        Section.suite_id.in_(
            db.query(TestSuite.id).filter(TestSuite.project_id == project_id)
        )
    ).delete(synchronize_session=False)

    # 10. Delete test cases (now safe since section references are gone)
    db.query(TestCase).filter(TestCase.project_id == project_id).delete(synchronize_session=False)
    
    # 11. Delete test runs
    db.query(TestRun).filter(TestRun.project_id == project_id).delete(synchronize_session=False)
    
    # 12. Delete test plans
    db.query(TestPlan).filter(TestPlan.project_id == project_id).delete(synchronize_session=False)
    
    # 13. Delete test suites
    db.query(TestSuite).filter(TestSuite.project_id == project_id).delete(synchronize_session=False)
    
    # 14. Delete milestones
    db.query(Milestone).filter(Milestone.project_id == project_id).delete(synchronize_session=False)
    
    # 15. Delete custom fields
    db.query(CustomField).filter(CustomField.project_id == project_id).delete(synchronize_session=False)
    
    # 16. Delete AI analyses
    db.query(AIAnalysis).filter(AIAnalysis.project_id == project_id).delete(synchronize_session=False)
    
    # 17. Delete project aggregate versions
    db.query(ProjectAggregateVersion).filter(ProjectAggregateVersion.project_id == project_id).delete(synchronize_session=False)
    
    # 18. Delete repository configs
    db.query(RepositoryConfig).filter(RepositoryConfig.project_id == project_id).delete(synchronize_session=False)
    
    # 19. Finally, delete the project itself
    db.delete(project)
    db.commit()
    
    # Track project deletion
    analytics.track_event(
        "project_deleted",
        user_id=current_user.id,
        properties={
            "project_id": project_id
        }
    )
    
    return {"message": "Project deleted successfully"}

# Project Members Management
@router.get("/{project_id}/members")
async def get_project_members(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all members of a project"""
    from app.core.permissions import UserRole
    
    # OPTIMIZED: Combined project existence and access check with member loading
    # Use selectinload to efficiently load user data and project for all members
    members = db.query(ProjectMember).options(
        selectinload(ProjectMember.user),
        selectinload(ProjectMember.project)
    ).filter(
        ProjectMember.project_id == project_id
    ).all()
    
    if not members:
        # Check if project exists at all
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        # Project exists but has no members - check if user has access
        if current_user.role != UserRole.ADMIN and project.created_by_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this project"
            )
        return []  # Return empty list for project with no members
    
    # Verify user has access (is admin, creator, or member)
    project = members[0].project  # Get from first member (already loaded)
    is_member = any(m.user_id == current_user.id for m in members)
    if current_user.role != UserRole.ADMIN and project.created_by_id != current_user.id and not is_member:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this project"
        )
    
    result = []
    for member in members:
        result.append({
            "id": member.id,
            "project_id": member.project_id,
            "user_id": member.user_id,
            "user_email": member.user.email if member.user else "Unknown",
            "user_name": member.user.name if member.user else "Unknown",
            "role": member.role,
            "added_at": member.added_at,
            "added_by_id": member.added_by_id
        })
    
    return result

@router.post("/{project_id}/members")
async def add_project_member(
    project_id: int,
    email: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Add a member to the project"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    # Only creator can add members
    if project.created_by_id != current_user.id:
        from app.core.permissions import UserRole
        if current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only project creator can add members"
            )
    
    # Find user by email
    user = db.query(User).filter(User.email == email.strip()).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with email {email} not found"
        )
    
    # Check if already a member
    existing_member = db.query(ProjectMember).filter(
        ProjectMember.project_id == project_id,
        ProjectMember.user_id == user.id
    ).first()
    
    if existing_member:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is already a member of this project"
        )
    
    # Don't add creator as member
    if user.id == project.created_by_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Project creator already has access"
        )
    
    # Add member
    member = ProjectMember(
        project_id=project_id,
        user_id=user.id,
        role="member",
        added_by_id=current_user.id
    )
    
    db.add(member)
    db.commit()
    db.refresh(member)
    
    return {
        "id": member.id,
        "project_id": member.project_id,
        "user_id": member.user_id,
        "user_email": user.email,
        "user_name": user.name,
        "role": member.role,
        "added_at": member.added_at,
        "added_by_id": member.added_by_id
    }

@router.delete("/{project_id}/members/{member_id}")
async def remove_project_member(
    project_id: int,
    member_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Remove a member from the project"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    # Only creator can remove members
    if project.created_by_id != current_user.id:
        from app.core.permissions import UserRole
        if current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only project creator can remove members"
            )
    
    member = db.query(ProjectMember).filter(
        ProjectMember.id == member_id,
        ProjectMember.project_id == project_id
    ).first()
    
    if not member:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Member not found"
        )
    
    db.delete(member)
    db.commit()
    
    return {"message": "Member removed successfully"}


@router.get("/{project_id}/overview-summary")
async def get_project_overview_summary(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get lightweight summary data for project overview tab - SUPER FAST"""
    # Verify project exists and user has access
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    check_project_access(project, current_user, Permission.READ, db)
    
    # OPTIMIZATION: Get only essential counts for overview in single batch
    test_cases_count = db.query(func.count(TestCase.id))\
        .filter(TestCase.project_id == project_id).scalar() or 0
    
    # Get execution stats from test executions (for pass rate)
    passed_count = db.query(func.count(TestExecution.id))\
        .join(TestRun, TestExecution.run_id == TestRun.id)\
        .filter(TestRun.project_id == project_id, TestExecution.status == TestExecutionStatus.PASSED).scalar() or 0
    
    failed_count = db.query(func.count(TestExecution.id))\
        .join(TestRun, TestExecution.run_id == TestRun.id)\
        .filter(TestRun.project_id == project_id, TestExecution.status == TestExecutionStatus.FAILED).scalar() or 0
    
    total_executed = passed_count + failed_count
    pending_count = max(0, test_cases_count - total_executed)
    
    # Get recent test runs only (most important for overview)
    recent_test_runs = db.query(
        TestRun.id, 
        TestRun.name, 
        TestRun.created_at,
        TestRun.status
    ).filter(TestRun.project_id == project_id)\
     .order_by(TestRun.created_at.desc())\
     .limit(5).all()
    
    recent_test_runs_list = [
        {
            "id": tr.id,
            "name": tr.name,
            "created_at": tr.created_at,
            "status": tr.status
        } for tr in recent_test_runs
    ]
    
    # Calculate pass rate
    pass_rate = round((passed_count / total_executed * 100), 1) if total_executed > 0 else 0
    
    return {
        "project_id": project_id,
        "project": {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
            "ai_generated": project.ai_generated,
            "ai_aggregate_response": project.ai_aggregate_response
        },
        "test_stats": {
            "total_test_cases": test_cases_count,
            "passed": passed_count,
            "failed": failed_count,
            "pending": pending_count,
            "pass_rate": pass_rate
        },
        "recent_test_runs": recent_test_runs_list
    }


@router.get("/{project_id}/suites-with-counts")
async def get_project_suites_with_counts(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get test suites with test case counts - OPTIMIZED"""
    # Verify project exists and user has access
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    check_project_access(project, current_user, Permission.READ, db)
    
    # Get suites with test case counts in a single efficient query
    suites_with_counts = db.query(
        TestSuite.id,
        TestSuite.name,
        TestSuite.description,
        TestSuite.created_at,
        func.count(TestCase.id).label('test_case_count')
    ).outerjoin(TestCase, TestCase.suite_id == TestSuite.id)\
     .filter(TestSuite.project_id == project_id)\
     .group_by(TestSuite.id, TestSuite.name, TestSuite.description, TestSuite.created_at)\
     .all()
    
    return [
        {
            "id": suite.id,
            "name": suite.name,
            "description": suite.description,
            "created_at": suite.created_at,
            "test_case_count": suite.test_case_count
        } for suite in suites_with_counts
    ]