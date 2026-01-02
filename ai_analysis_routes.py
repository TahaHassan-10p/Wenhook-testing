"""
Manual AI Analysis API Routes
Allows users to trigger AI analysis on-demand for commits, PRs, or branches
"""
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database.session import get_db
from app.database.models import RepositoryConfig, PRAnalysis
from app.auth.deps import get_current_active_user, require_permission
from app.core.permissions import Permission
from app.services.github_integration_service import GitHubIntegrationService, GitHubAPIClient
from app.core.encryption import decrypt_access_token
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
router = APIRouter(tags=["AI Analysis"])


class ManualAnalysisRequest(BaseModel):
    repository_config_id: int
    analysis_type: str  # "push", "pr", "branch"
    target_ref: str  # commit SHA, PR number, or branch name
    title: Optional[str] = None
    description: Optional[str] = None


class LatestCommitRequest(BaseModel):
    repository_config_id: int
    branch: str = "main"


@router.post("/trigger-latest-commit")
async def trigger_latest_commit_analysis(
    request: LatestCommitRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Trigger AI analysis for the latest commit on a branch (simplified endpoint)"""
    
    try:
        # Get repository config
        repo_config = db.query(RepositoryConfig).filter(
            RepositoryConfig.id == request.repository_config_id
        ).first()
        
        if not repo_config:
            raise HTTPException(status_code=404, detail="Repository configuration not found")
        
        # Extract repository info from URL
        repo_url_parts = repo_config.repository_url.replace("https://github.com/", "").split("/")
        if len(repo_url_parts) < 2:
            raise HTTPException(status_code=400, detail="Invalid repository URL format")
        
        repo_owner = repo_url_parts[0]
        repo_name = repo_url_parts[1]
        
        # Get GitHub access token (pass encrypted token, let GitHubAPIClient decrypt it)
        github_client = GitHubAPIClient(repo_config.access_token)
        
        # Get the latest commit data from the specified branch
        try:
            # Get latest commit SHA from branch
            branch_data = await github_client.get_branch(repo_owner, repo_name, request.branch)
            latest_commit_sha = branch_data.get("commit", {}).get("sha")
            
            if not latest_commit_sha:
                raise HTTPException(status_code=404, detail=f"Could not find latest commit on branch {request.branch}")
            
            # Get commit details
            commit_data = await github_client.get_commit(repo_owner, repo_name, latest_commit_sha)
            
            # Extract commit information
            commit_message = commit_data.get("commit", {}).get("message", "No commit message")
            commit_author = commit_data.get("commit", {}).get("author", {}).get("name", "Unknown")
            files_changed = commit_data.get("files", [])
            
            # Calculate stats
            lines_added = sum(file.get("additions", 0) for file in files_changed)
            lines_removed = sum(file.get("deletions", 0) for file in files_changed)
            
            # Generate diff content
            diff_content = "\n".join([
                f"--- {file.get('filename', 'unknown')}" + 
                (f"\n+++ {file.get('filename', 'unknown')}" if file.get("patch") else "") +
                (f"\n{file.get('patch', '')}" if file.get("patch") else "\n(Binary or large file)")
                for file in files_changed
            ])
            
        except Exception as github_error:
            logger.error(f"Failed to fetch commit data from GitHub: {github_error}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch commit data: {str(github_error)}")
        
        # Create analysis record with actual commit data
        pr_analysis = PRAnalysis(
            repository_config_id=repo_config.id,
            pr_number=0,  # 0 indicates manual analysis
            pr_title=f"Manual Analysis - {commit_message[:100]}",
            pr_description=f"Manual AI analysis of commit {latest_commit_sha[:8]} on {request.branch} branch",
            pr_url=f"{repo_config.repository_url}/commit/{latest_commit_sha}",
            source_branch=request.branch,
            target_branch=request.branch,
            author=commit_author,
            event_type="manual_latest_commit",
            status="pending",
            diff_content=diff_content,
            files_changed=[file.get("filename") for file in files_changed],
            lines_added=lines_added,
            lines_removed=lines_removed
        )
        
        db.add(pr_analysis)
        db.commit()
        db.refresh(pr_analysis)
        
        # Trigger background AI analysis
        background_tasks.add_task(
            GitHubIntegrationService.analyze_pull_request,
            pr_analysis,
            db
        )
        
        logger.info(f"Manual AI analysis triggered for latest {request.branch} commit")
        
        return {
            "message": "AI analysis triggered for latest commit",
            "analysis_id": pr_analysis.id,
            "branch": request.branch,
            "status": "pending",
            "note": "Analysis will fetch latest commit data from GitHub automatically"
        }
        
    except Exception as e:
        logger.error(f"Latest commit analysis trigger failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger AI analysis")


@router.post("/trigger")
async def trigger_manual_analysis(
    request: ManualAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Trigger manual AI analysis for a specific commit, PR, or branch"""
    
    try:
        # Get repository config
        repo_config = db.query(RepositoryConfig).filter(
            RepositoryConfig.id == request.repository_config_id
        ).first()
        
        if not repo_config:
            raise HTTPException(status_code=404, detail="Repository configuration not found")
        
        # Check permissions (user should have access to the project)
        # Add your permission check logic here based on your auth system
        
        # Create GitHub API client
        github_client = GitHubAPIClient(repo_config.access_token)
        
        # Extract repo owner and name from URL
        repo_url_parts = repo_config.repository_url.replace("https://github.com/", "").split("/")
        if len(repo_url_parts) < 2:
            raise HTTPException(status_code=400, detail="Invalid repository URL format")
        
        repo_owner = repo_url_parts[0]
        repo_name = repo_url_parts[1]
        
        # Handle different analysis types
        if request.analysis_type == "pr":
            # Analyze specific pull request
            pr_number = int(request.target_ref)
            
            # Fetch PR data from GitHub
            pr_diff = await github_client.get_pull_request_diff(repo_owner, repo_name, pr_number)
            pr_files = await github_client.get_pull_request_files(repo_owner, repo_name, pr_number)
            
            # Create PR analysis record
            pr_analysis = PRAnalysis(
                repository_config_id=repo_config.id,
                pr_number=pr_number,
                pr_title=request.title or f"Manual Analysis - PR #{pr_number}",
                pr_description=request.description or "Manual AI analysis triggered by user",
                pr_url=f"{repo_config.repository_url}/pull/{pr_number}",
                source_branch="unknown",  # Would need additional API call to get this
                target_branch="unknown",
                author=current_user.name,
                event_type="manual_pr_analysis",
                status="pending",
                diff_content=pr_diff,
                files_changed=[file.get("filename") for file in pr_files],
                lines_added=sum(file.get("additions", 0) for file in pr_files),
                lines_removed=sum(file.get("deletions", 0) for file in pr_files)
            )
            
        elif request.analysis_type == "push" or request.analysis_type == "branch":
            # Analyze specific commit or branch
            commit_sha = request.target_ref
            
            # For branch analysis, get the latest commit
            if request.analysis_type == "branch":
                # Would need additional GitHub API call to get latest commit SHA
                # For now, assume target_ref is already a commit SHA
                pass
            
            # Create push-style analysis record
            pr_analysis = PRAnalysis(
                repository_config_id=repo_config.id,
                pr_number=0,  # 0 indicates manual analysis, not PR
                pr_title=request.title or f"Manual Analysis - {request.analysis_type.title()}",
                pr_description=request.description or f"Manual AI analysis of {request.analysis_type}: {request.target_ref}",
                pr_url=f"{repo_config.repository_url}/commit/{commit_sha}",
                source_branch=request.target_ref if request.analysis_type == "branch" else "unknown",
                target_branch=request.target_ref if request.analysis_type == "branch" else "unknown",
                author=current_user.name,
                event_type=f"manual_{request.analysis_type}_analysis",
                status="pending"
            )
            
        else:
            raise HTTPException(status_code=400, detail="Invalid analysis_type. Must be 'push', 'pr', or 'branch'")
        
        # Save analysis record
        db.add(pr_analysis)
        db.commit()
        db.refresh(pr_analysis)
        
        # Trigger background AI analysis
        background_tasks.add_task(
            GitHubIntegrationService.analyze_pull_request,
            pr_analysis,
            db
        )
        
        logger.info(f"Manual AI analysis triggered for {request.analysis_type}: {request.target_ref}")
        
        return {
            "message": "AI analysis triggered successfully",
            "analysis_id": pr_analysis.id,
            "analysis_type": request.analysis_type,
            "target_ref": request.target_ref,
            "status": "pending"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid target reference: {e}")
    except Exception as e:
        logger.error(f"Manual analysis trigger failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger AI analysis")


@router.post("/trigger-analysis/{analysis_id}")
async def trigger_analysis_for_record(
    analysis_id: int,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Trigger batch AI analysis for commits up to the selected record"""
    
    try:
        # Get the selected analysis record
        selected_analysis = db.query(PRAnalysis).filter(PRAnalysis.id == analysis_id).first()
        
        if not selected_analysis:
            raise HTTPException(status_code=404, detail="Analysis record not found")
        
        # Check if already analyzed
        if selected_analysis.status == 'completed':
            return {
                "message": "This commit has already been analyzed",
                "analysis_id": analysis_id,
                "status": "already_completed",
                "analyzed_at": selected_analysis.processed_at
            }
        
        # Get repository config
        repo_config = db.query(RepositoryConfig).filter(
            RepositoryConfig.id == selected_analysis.repository_config_id
        ).first()
        
        if not repo_config:
            raise HTTPException(status_code=404, detail="Repository configuration not found")
        
        # Find the last completed commit for this repo/branch
        last_completed = db.query(PRAnalysis).filter(
            PRAnalysis.repository_config_id == repo_config.id,
            PRAnalysis.source_branch == selected_analysis.source_branch,
            PRAnalysis.status == 'completed'
        ).order_by(PRAnalysis.id.desc()).first()
        
        # Get all pending commits in the range to analyze
        commits_to_analyze = db.query(PRAnalysis).filter(
            PRAnalysis.repository_config_id == repo_config.id,
            PRAnalysis.source_branch == selected_analysis.source_branch,
            PRAnalysis.status == 'pending',
            PRAnalysis.id > (last_completed.id if last_completed else 0),
            PRAnalysis.id <= analysis_id
        ).order_by(PRAnalysis.id.asc()).all()
        
        if not commits_to_analyze:
            return {
                "message": "No pending commits found to analyze",
                "analysis_id": analysis_id,
                "status": "no_pending_commits"
            }
        
        logger.info(f"Starting batch analysis for {len(commits_to_analyze)} commits (IDs: {[c.id for c in commits_to_analyze]})")
        
        # All-or-nothing transaction
        try:
            # Mark all commits as 'analyzing'
            for commit in commits_to_analyze:
                commit.status = "analyzing"
            db.commit()
            
            # For batch analysis, we'll analyze the selected commit (which contains cumulative diff)
            # The webhook already populated it with cumulative diff data
            success = await GitHubIntegrationService.analyze_pull_request(selected_analysis, db)
            
            if success:
                # Mark all commits in the batch as completed
                current_time = datetime.utcnow()
                for commit in commits_to_analyze:
                    commit.status = "completed"
                    commit.processed_at = current_time
                
                db.commit()
                
                return {
                    "message": f"Batch AI analysis completed successfully for {len(commits_to_analyze)} commits",
                    "analysis_id": analysis_id,
                    "status": "completed",
                    "commits_analyzed": len(commits_to_analyze),
                    "commit_ids": [c.id for c in commits_to_analyze]
                }
            else:
                # AI analysis failed - mark all as failed
                for commit in commits_to_analyze:
                    commit.status = "failed"
                db.commit()
                
                raise HTTPException(status_code=500, detail="AI analysis failed for the batch")
                
        except Exception as analysis_error:
            # Rollback all status changes on any failure
            logger.error(f"Batch analysis failed: {analysis_error}")
            for commit in commits_to_analyze:
                commit.status = "pending"  # Reset to pending
            db.commit()
            
            raise HTTPException(
                status_code=500, 
                detail=f"Batch analysis failed: {str(analysis_error)}. All commits remain pending."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger batch analysis for record {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger batch analysis")


@router.get("/status/{analysis_id}")
async def get_analysis_status(
    analysis_id: int,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get the status of a manual AI analysis"""
    
    analysis = db.query(PRAnalysis).filter(PRAnalysis.id == analysis_id).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return {
        "analysis_id": analysis.id,
        "status": analysis.status,
        "pr_title": analysis.pr_title,
        "created_at": analysis.created_at,
        "processed_at": analysis.processed_at,
        "event_type": analysis.event_type
    }