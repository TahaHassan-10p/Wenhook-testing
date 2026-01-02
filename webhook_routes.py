from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.database.models import User, Project, RepositoryConfig, PRAnalysis, IntegrationLog
from app.auth.deps import get_current_active_user, require_permission
from app.core.permissions import Permission
from app.services.github_integration_service import GitHubIntegrationService, GitHubAPIClient
from app.services.diff_calculation_service import DiffCalculationService
from app.core.encryption import decrypt_access_token
import json
import hmac
import httpx
import asyncio
import hashlib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

async def fetch_github_commit_diff(commit_sha: str, repo_url: str, access_token: str) -> Dict[str, Any]:
    """Fetch commit diff data from GitHub API"""
    
    try:
        # Extract owner/repo from URL
        repo_url_clean = repo_url.replace('https://github.com/', '').replace('.git', '')
        parts = repo_url_clean.split('/')
        if len(parts) < 2:
            logger.error(f"Invalid GitHub repo URL: {repo_url}")
            return {"files": [], "total_additions": 0, "total_deletions": 0}
            
        owner = parts[0]
        repo = parts[1]
        
        # Decrypt access token
        decrypted_token = decrypt_access_token(access_token)
        if not decrypted_token:
            logger.error("Failed to decrypt GitHub access token")
            return {"files": [], "total_additions": 0, "total_deletions": 0}
        
        headers = {
            'Authorization': f'token {decrypted_token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'TestWorthy-Pro'
        }
        
        # Get commit details with diff
        commit_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(commit_url, headers=headers)
            
            if response.status_code == 200:
                commit_data = response.json()
                files = commit_data.get('files', [])
                
                total_additions = sum(file.get('additions', 0) for file in files)
                total_deletions = sum(file.get('deletions', 0) for file in files)
                
                logger.info(f"GitHub API: {commit_sha[:8]} - {len(files)} files, +{total_additions} -{total_deletions}")
                
                return {
                    "files": files,
                    "total_additions": total_additions,
                    "total_deletions": total_deletions,
                    "diff_content": commit_data.get('commit', {}).get('message', '')
                }
            else:
                logger.error(f"GitHub API error: {response.status_code} - {response.text}")
                return {"files": [], "total_additions": 0, "total_deletions": 0}
                
    except Exception as e:
        logger.error(f"Error fetching GitHub commit diff: {e}")
        return {"files": [], "total_additions": 0, "total_deletions": 0}

async def fetch_github_cumulative_diff(commit_sha: str, branch: str, repo_url: str, access_token: str, repo_config: RepositoryConfig, db: Session) -> Dict[str, Any]:
    """Fetch cumulative diff data from last analyzed commit up to specific commit using GitHub Compare API"""
    
    try:
        # Extract owner/repo from URL
        repo_url_clean = repo_url.replace('https://github.com/', '').replace('.git', '')
        parts = repo_url_clean.split('/')
        if len(parts) < 2:
            logger.error(f"Invalid GitHub repo URL: {repo_url}")
            return {"files": [], "total_additions": 0, "total_deletions": 0}
            
        owner = parts[0]
        repo = parts[1]
        
        # Find the last analyzed commit for this repository/branch
        last_analyzed = db.query(PRAnalysis).filter(
            PRAnalysis.repository_config_id == repo_config.id,
            PRAnalysis.source_branch == branch,
            PRAnalysis.status == 'completed'
        ).order_by(PRAnalysis.processed_at.desc()).first()
        
        # Determine base reference
        if last_analyzed and last_analyzed.pr_url:
            # Extract commit SHA from the last analyzed commit's URL
            base_ref = None
            if '/commit/' in last_analyzed.pr_url:
                base_ref = last_analyzed.pr_url.split('/commit/')[-1]
            elif 'github.com' in last_analyzed.pr_url:
                # Try to extract from URL structure
                base_ref = last_analyzed.pr_url.split('/')[-1]
            
            if not base_ref:
                base_ref = "main"  # Fallback
            
            logger.info(f"Using last analyzed commit as base: {base_ref[:8] if len(base_ref) > 8 else base_ref}")
        else:
            # No previous analysis, compare against main branch
            base_ref = "main"
            logger.info(f"No previous analysis found, comparing against main")
        
        # Decrypt access token
        decrypted_token = decrypt_access_token(access_token)
        if not decrypted_token:
            logger.error("Failed to decrypt GitHub access token")
            return {"files": [], "total_additions": 0, "total_deletions": 0}
        
        headers = {
            'Authorization': f'token {decrypted_token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'TestWorthy-Pro'
        }
        
        # Use GitHub Compare API to get incremental changes
        compare_url = f"https://api.github.com/repos/{owner}/{repo}/compare/{base_ref}...{commit_sha}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(compare_url, headers=headers)
            
            if response.status_code == 200:
                compare_data = response.json()
                files = compare_data.get('files', [])
                
                total_additions = sum(file.get('additions', 0) for file in files)
                total_deletions = sum(file.get('deletions', 0) for file in files)
                
                commits_in_range = len(compare_data.get('commits', []))
                
                logger.info(f"GitHub Compare API: {base_ref}...{commit_sha[:8]} - {len(files)} files, +{total_additions} -{total_deletions}, {commits_in_range} commits")
                
                # Get commit message from the specific commit
                commit_message = ""
                commits = compare_data.get('commits', [])
                if commits:
                    # Get the latest commit message
                    latest_commit = commits[-1]
                    commit_message = latest_commit.get('commit', {}).get('message', '')
                
                return {
                    "files": files,
                    "total_additions": total_additions,
                    "total_deletions": total_deletions,
                    "diff_content": commit_message,
                    "commits_analyzed": commits_in_range,
                    "base_ref": base_ref,
                    "is_incremental": last_analyzed is not None
                }
            else:
                logger.error(f"GitHub Compare API error: {response.status_code} - {response.text}")
                # Fallback to single commit diff if compare fails
                return await fetch_github_commit_diff(commit_sha, repo_url, access_token)
                
    except Exception as e:
        logger.error(f"Error fetching GitHub cumulative diff: {e}")
        # Fallback to single commit diff
        return await fetch_github_commit_diff(commit_sha, repo_url, access_token)

async def fetch_gitlab_mr_diff(project_id: int, mr_iid: int, gitlab_url: str, access_token: str) -> Dict[str, Any]:
    """Fetch MR diff data from GitLab API"""
    
    try:
        # Decrypt access token  
        decrypted_token = decrypt_access_token(access_token)
        if not decrypted_token:
            logger.error("Failed to decrypt GitLab access token")
            return {"files": [], "total_additions": 0, "total_deletions": 0}
        
        headers = {
            'Authorization': f'Bearer {decrypted_token}',
            'Content-Type': 'application/json'
        }
        
        # Get MR changes from GitLab API
        # Extract GitLab domain from URL
        if gitlab_url.startswith('https://'):
            gitlab_domain = gitlab_url.split('/')[2]  # e.g., git.10pearls.com
        else:
            gitlab_domain = "gitlab.com"  # fallback
            
        changes_url = f"https://{gitlab_domain}/api/v4/projects/{project_id}/merge_requests/{mr_iid}/changes"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(changes_url, headers=headers)
            
            if response.status_code == 200:
                mr_data = response.json()
                changes = mr_data.get('changes', [])
                
                total_additions = 0
                total_deletions = 0
                files = []
                
                for change in changes:
                    # GitLab provides diff text, we need to count lines
                    diff = change.get('diff', '')
                    additions = diff.count('\n+') - diff.count('\n+++')  # Exclude header lines
                    deletions = diff.count('\n-') - diff.count('\n---')   # Exclude header lines
                    
                    additions = max(0, additions)  # Ensure non-negative
                    deletions = max(0, deletions)  # Ensure non-negative
                    
                    total_additions += additions
                    total_deletions += deletions
                    
                    files.append({
                        'new_path': change.get('new_path', ''),
                        'old_path': change.get('old_path', ''),  
                        'new_file': change.get('new_file', False),
                        'deleted_file': change.get('deleted_file', False),
                        'additions': additions,
                        'deletions': deletions
                    })
                
                logger.info(f"GitLab API: MR {mr_iid} - {len(files)} files, +{total_additions} -{total_deletions}")
                
                return {
                    "files": files,
                    "total_additions": total_additions, 
                    "total_deletions": total_deletions,
                    "diff_content": mr_data.get('description', '')
                }
            else:
                logger.error(f"GitLab API error: {response.status_code} - {response.text}")
                return {"files": [], "total_additions": 0, "total_deletions": 0}
                
    except Exception as e:
        logger.error(f"Error fetching GitLab MR diff: {e}")
        return {"files": [], "total_additions": 0, "total_deletions": 0}

@router.post("/github/{config_id}")
async def github_webhook(
    config_id: int,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Enhanced GitHub webhook handler with AI analysis"""
    
    try:
        # Get repository config
        repo_config = db.query(RepositoryConfig).filter(RepositoryConfig.id == config_id).first()
        if not repo_config or not repo_config.is_active:
            raise HTTPException(status_code=404, detail="Repository configuration not found or inactive")
        
        # Get request body and headers
        body = await request.body()
        signature = request.headers.get("x-hub-signature-256", "")
        event_type = request.headers.get("x-github-event", "")
        
        # Verify webhook signature if secret is configured
        if repo_config.webhook_secret:
            if not verify_github_signature(body, signature, repo_config.webhook_secret):
                raise HTTPException(status_code=403, detail="Invalid webhook signature")
        
        # Parse payload
        payload = json.loads(body.decode("utf-8"))
        
        # Log webhook event
        log_entry = IntegrationLog(
            repository_config_id=config_id,
            event_type=f"webhook_{event_type}",
            event_data={
                "action": payload.get("action"),
                "sender": payload.get("sender", {}).get("login")
            },
            success=True
        )
        db.add(log_entry)
        
        # Handle pull request events
        if event_type == "pull_request":
            result = await handle_github_pr_event_basic(payload, repo_config, db)
            db.commit()
            return result
        
        # Handle other events
        elif event_type == "push":
            return await handle_github_push_event_basic(payload, repo_config, db)
        
        else:
            logger.info(f"Ignoring unsupported GitHub event: {event_type}")
            return {"message": f"Event {event_type} ignored"}
    
    except Exception as e:
        logger.error(f"GitHub webhook error: {e}")
        
        # Log error
        error_log = IntegrationLog(
            repository_config_id=config_id,
            event_type="webhook_error",
            event_data={"error": str(e)},
            success=False,
            error_message=str(e)
        )
        db.add(error_log)
        db.commit()
        
        raise HTTPException(status_code=500, detail="Webhook processing failed")

@router.post("/gitlab/{config_id}")
async def gitlab_webhook(
    config_id: int,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """GitLab webhook handler"""
    
    try:
        logger.info(f"Processing GitLab webhook for config_id: {config_id}")
        
        # Get repository config
        repo_config = db.query(RepositoryConfig).filter(RepositoryConfig.id == config_id).first()
        if not repo_config or not repo_config.is_active:
            logger.error(f"Repository config not found or inactive: {config_id}")
            raise HTTPException(status_code=404, detail="Repository configuration not found or inactive")
        
        logger.info(f"Found active repo config: {repo_config.provider}")
        
        # Get request body and headers
        body = await request.body()
        signature = request.headers.get("X-Gitlab-Token", "")
        event_type = request.headers.get("X-Gitlab-Event", "")
        
        logger.info(f"GitLab webhook - Event type: {event_type}")
        logger.info(f"All headers: {dict(request.headers)}")
        logger.info(f"Signature header found: {bool(signature)}")
        logger.info(f"Expected secret length: {len(repo_config.webhook_secret) if repo_config.webhook_secret else 0}")
        logger.info(f"Received signature length: {len(signature)}")
        
        # Verify webhook signature using HMAC SHA-256 if secret is configured
        if repo_config.webhook_secret:
            if not verify_gitlab_signature(body, signature, repo_config.webhook_secret):
                logger.error("GitLab webhook signature verification failed")
                raise HTTPException(status_code=403, detail="Invalid webhook signature")
            else:
                logger.info("GitLab webhook signature verified successfully")
        else:
            logger.info("No webhook secret configured - skipping signature verification")
        
        # Parse payload
        try:
            payload = json.loads(body.decode("utf-8"))
            logger.info(f"Parsed payload object_kind: {payload.get('object_kind')}")
        except Exception as parse_error:
            logger.error(f"Failed to parse JSON payload: {parse_error}")
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
        
        # Log webhook event
        log_entry = IntegrationLog(
            repository_config_id=config_id,
            event_type=f"webhook_{event_type}",
            event_data={
                "object_kind": payload.get("object_kind"),
                "user": payload.get("user_username", payload.get("user_name", "unknown"))
            },
            success=True
        )
        db.add(log_entry)
        db.commit()  # Commit the log entry first
        
        # Handle merge request events
        if payload.get("object_kind") == "merge_request":
            logger.info("Processing merge request event")
            result = await handle_gitlab_mr_event(payload, repo_config, db, background_tasks)
            return result
            
        # Handle push events
        elif payload.get("object_kind") == "push":
            logger.info("Processing push event")
            result = await handle_gitlab_push_event(payload, repo_config, db, background_tasks)
            return result

        else:
            logger.info(f"Ignoring unsupported GitLab event: {payload.get('object_kind')}")
            db.commit()
            return {"message": f"Event {payload.get('object_kind')} ignored"}
            
    except Exception as e:
        logger.error(f"GitLab webhook error: {e}")
        error_log = IntegrationLog(
            repository_config_id=config_id,
            event_type="webhook_error",
            event_data={"error": str(e)},
            success=False,
            error_message=str(e)
        )
        db.add(error_log)
        db.commit()
        
        raise HTTPException(status_code=500, detail="Webhook processing failed")

def verify_github_signature(body: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub webhook signature"""
    if not signature or not signature.startswith("sha256="):
        return False
    
    expected_signature = "sha256=" + hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected_signature)


def verify_gitlab_signature(body: bytes, token: str, secret: str) -> bool:
    """Verify GitLab webhook signature using simple token comparison"""
    if not token or not secret:
        return False
    
    # GitLab sends raw secret token in X-Gitlab-Token header
    if hmac.compare_digest(token, secret):
        logger.info("GitLab webhook verified using token comparison")
        return True
    
    return False


async def run_ai_analysis_for_push(analysis_id: int):
    """Background task to run AI analysis for push events"""
    from app.database.session import SessionLocal
    
    db = SessionLocal()
    try:
        push_analysis = db.query(PRAnalysis).filter(PRAnalysis.id == analysis_id).first()
        if push_analysis:
            logger.info(f"Starting background AI analysis for push analysis {analysis_id}")
            success = await GitHubIntegrationService.analyze_pull_request(push_analysis, db)
            logger.info(f"Background AI analysis completed: {'success' if success else 'failed'}")
        else:
            logger.error(f"Push analysis {analysis_id} not found for background processing")
    except Exception as analysis_error:
        logger.error(f"Background AI analysis failed: {analysis_error}")
        if push_analysis:
            push_analysis.status = "failed"
            db.commit()
    finally:
        db.close()


async def handle_github_pr_event_enhanced(
    payload: Dict[str, Any], 
    repo_config: RepositoryConfig, 
    db: Session,
    background_tasks: BackgroundTasks
):
    """Enhanced GitHub PR event handler with hybrid approach for diff calculation"""
    
    action = payload.get("action")
    pr_data = payload.get("pull_request", {})
    
    if action in ["opened", "synchronize", "reopened"]:
        # Extract repository info
        repo_info = payload.get("repository", {})
        repo_owner = repo_info.get("owner", {}).get("login")
        repo_name = repo_info.get("name")
        
        # Extract display metadata from payload (for immediate frontend display)
        display_metadata = DiffCalculationService.extract_display_metadata_from_payload(
            payload, "pull_request"
        )
        
        # Create or update PR analysis
        existing_analysis = db.query(PRAnalysis).filter(
            PRAnalysis.repository_config_id == repo_config.id,
            PRAnalysis.pr_number == pr_data.get("number")
        ).first()
        
        if existing_analysis:
            # Update existing analysis with display metadata
            existing_analysis.status = "pending"
            existing_analysis.pr_title = pr_data.get("title")
            existing_analysis.pr_description = pr_data.get("body")
            existing_analysis.source_branch = pr_data.get("head", {}).get("ref")
            existing_analysis.target_branch = pr_data.get("base", {}).get("ref")
            existing_analysis.author = pr_data.get("user", {}).get("login")
            existing_analysis.event_type = f"pr_{action}"
            existing_analysis.updated_at = datetime.utcnow()
            
            # Store display metadata (not detailed diff - calculated at analysis time)
            existing_analysis.commit_sha = display_metadata["commit_sha"]
            existing_analysis.files_changed = display_metadata["files_changed"]
            existing_analysis.lines_added = display_metadata["lines_added"]
            existing_analysis.lines_removed = display_metadata["lines_removed"]
            # diff_content will be calculated at analysis time
            
            pr_analysis = existing_analysis
        else:
            # Create new analysis with display metadata
            pr_analysis = PRAnalysis(
                repository_config_id=repo_config.id,
                pr_number=pr_data.get("number"),
                pr_title=pr_data.get("title"),
                pr_description=pr_data.get("body"),
                pr_url=pr_data.get("html_url"),
                source_branch=pr_data.get("head", {}).get("ref"),
                target_branch=pr_data.get("base", {}).get("ref"),
                author=pr_data.get("user", {}).get("login"),
                event_type=f"pr_{action}",
                status="pending",
                # Store display metadata immediately
                commit_sha=display_metadata["commit_sha"],
                files_changed=display_metadata["files_changed"],
                lines_added=display_metadata["lines_added"],
                lines_removed=display_metadata["lines_removed"]
                # diff_content will be calculated at analysis time
            )
            db.add(pr_analysis)
        
        db.commit()
        db.refresh(pr_analysis)
        
        # Trigger AI analysis in background if auto-analysis is enabled
        if repo_config.auto_analyze_prs:
            background_tasks.add_task(
                GitHubIntegrationService.analyze_pull_request, 
                pr_analysis, 
                db
            )
            
            logger.info(f"GitHub PR {pr_data.get('number')}: Stored display metadata, AI analysis queued")
        
        return {
            "message": "GitHub PR event processed successfully",
            "analysis_id": pr_analysis.id,
            "status": pr_analysis.status
        }
    
    return {"message": f"GitHub PR action '{action}' ignored"}

async def handle_gitlab_mr_event(
    payload: Dict[str, Any], 
    repo_config: RepositoryConfig, 
    db: Session,
    background_tasks: BackgroundTasks
):
    """GitLab MR event handler"""
    
    mr_attributes = payload.get("object_attributes", {})
    action = mr_attributes.get("action")
    
    if action in ["open", "update", "reopen"]:
        # Create or update PR analysis
        existing_analysis = db.query(PRAnalysis).filter(
            PRAnalysis.repository_config_id == repo_config.id,
            PRAnalysis.pr_number == mr_attributes.get("iid")
        ).first()
        
        if existing_analysis:
            # Update existing analysis
            existing_analysis.status = "pending"
            existing_analysis.pr_title = mr_attributes.get("title")
            existing_analysis.pr_description = mr_attributes.get("description")
            existing_analysis.source_branch = mr_attributes.get("source_branch")
            existing_analysis.target_branch = mr_attributes.get("target_branch")
            existing_analysis.author = payload.get("user", {}).get("username", payload.get("user_username", payload.get("user_name", "unknown")))
            existing_analysis.event_type = f"mr_{action}"
            existing_analysis.updated_at = datetime.utcnow()
            
            # Store display metadata using hybrid approach
            display_metadata = DiffCalculationService.extract_display_metadata_from_payload(
                payload, "gitlab_merge_request"
            )
            existing_analysis.commit_sha = display_metadata["commit_sha"]
            existing_analysis.files_changed = display_metadata["files_changed"]
            existing_analysis.lines_added = display_metadata["lines_added"]
            existing_analysis.lines_removed = display_metadata["lines_removed"]
            
            pr_analysis = existing_analysis
        else:
            # Extract display metadata from payload
            display_metadata = DiffCalculationService.extract_display_metadata_from_payload(
                payload, "gitlab_merge_request"
            )
            
            # Create new analysis with display metadata only
            pr_analysis = PRAnalysis(
                repository_config_id=repo_config.id,
                pr_number=mr_attributes.get("iid"),
                pr_title=mr_attributes.get("title"),
                pr_description=mr_attributes.get("description"),
                pr_url=mr_attributes.get("url"),
                source_branch=mr_attributes.get("source_branch"),
                target_branch=mr_attributes.get("target_branch"),
                author=payload.get("user", {}).get("username", payload.get("user_username", payload.get("user_name", "unknown"))),
                event_type=f"mr_{action}",
                status="pending",
                # Store display metadata immediately
                commit_sha=display_metadata["commit_sha"],
                files_changed=display_metadata["files_changed"],
                lines_added=display_metadata["lines_added"],
                lines_removed=display_metadata["lines_removed"]
            )
            db.add(pr_analysis)
        
        db.commit()
        db.refresh(pr_analysis)
        
        logger.info(f"GitLab MR {mr_attributes.get('iid')}: Stored display metadata using hybrid approach")
        
        # Trigger AI analysis in background (detailed diff calculated at analysis time)
        if background_tasks and repo_config.auto_analyze_prs:
            background_tasks.add_task(GitHubIntegrationService.analyze_pull_request, pr_analysis, db)
        
        return {
            "message": "GitLab MR event processed successfully",
            "analysis_id": pr_analysis.id,
            "status": pr_analysis.status
        }
    
    return {"message": f"GitLab MR action '{action}' ignored"}

async def handle_github_push_event_basic(
    payload: Dict[str, Any], 
    repo_config: RepositoryConfig, 
    db: Session
):
    """Handle GitHub push events - create PR analysis records for commits"""
    
    commits = payload.get("commits", [])
    branch = payload.get("ref", "").replace("refs/heads/", "")
    repository_info = payload.get("repository", {})
    
    # Log push event
    log_entry = IntegrationLog(
        repository_config_id=repo_config.id,
        event_type="push_event",
        event_data={
            "branch": branch,
            "commits_count": len(commits),
            "pusher": payload.get("pusher", {}).get("name"),
            "auto_analysis": False
        },
        success=True
    )
    db.add(log_entry)
    
    # Create PRAnalysis records for each commit with display metadata only
    new_analyses = []
    for commit in commits:
        commit_sha = commit.get("id", "")
        commit_message = commit.get("message", "")
        commit_author = commit.get("author", {}).get("name", "Unknown")
        commit_url = commit.get("url", "")
        
        # Check if this commit already has an analysis record
        existing = db.query(PRAnalysis).filter(
            PRAnalysis.repository_config_id == repo_config.id,
            PRAnalysis.pr_url.contains(commit_sha)
        ).first()
        
        if not existing:
            # First, extract basic info from webhook payload for immediate display
            payload_metadata = DiffCalculationService.extract_display_metadata_from_payload(
                {"commits": [commit]}, "push"
            )
            
            # Try to get more accurate counts from GitHub API if possible
            # Create temporary PR analysis to use with stats calculation
            temp_analysis = PRAnalysis(
                repository_config_id=repo_config.id,
                pr_number=0,
                commit_sha=commit_sha
            )
            
            try:
                # Attempt GitHub API call for more accurate stats
                api_metadata = await DiffCalculationService.calculate_webhook_display_metadata(
                    temp_analysis, db
                )
                # Use API data if it has actual numbers, otherwise fall back to payload data
                if api_metadata.get("lines_added", 0) > 0 or api_metadata.get("lines_removed", 0) > 0:
                    display_metadata = api_metadata
                else:
                    display_metadata = payload_metadata
                    logger.info(f"🔍 GitHub API returned 0 changes, using payload data: {payload_metadata}")
            except Exception as e:
                logger.warning(f"🔍 GitHub API call failed, using payload data: {e}")
                display_metadata = payload_metadata
            
            # Create new analysis record with best available stats
            commit_analysis = PRAnalysis(
                repository_config_id=repo_config.id,
                pr_number=0,  # 0 indicates it's a commit, not a PR
                pr_title=f"Push to {branch}: {commit_message.split(chr(10))[0][:100]}",
                pr_description=f"Commit {commit_sha[:8]} pushed to {branch}",
                pr_url=commit_url,
                author=commit_author,
                source_branch=branch,
                target_branch=branch,
                event_type='webhook_push',
                status='pending',
                # Store best available metadata immediately (for frontend display)
                commit_sha=commit_sha,
                files_changed=display_metadata.get("files_changed", []),
                lines_added=display_metadata.get("lines_added", 0),
                lines_removed=display_metadata.get("lines_removed", 0),
                # diff_content will be calculated at analysis time
                analysis_results={"message": "Commit received via webhook - ready for AI analysis"}
            )
            
            db.add(commit_analysis)
            new_analyses.append({
                "sha": commit_sha[:8],
                "message": commit_message.split('\n')[0],
                "author": commit_author,
                "files": len(display_metadata.get("files_changed", [])),
                "additions": display_metadata.get("lines_added", 0),  # Accurate from GitHub API
                "deletions": display_metadata.get("lines_removed", 0)   # Accurate from GitHub API
            })
    
    db.commit()
    
    return {
        "message": "Push event processed", 
        "branch": branch, 
        "commits": len(commits),
        "new_analyses_created": len(new_analyses),
        "note": "Commit analysis records created - ready for AI analysis"
    }


async def handle_github_pr_event_basic(
    payload: Dict[str, Any], 
    repo_config: RepositoryConfig, 
    db: Session
):
    """Handle GitHub PR events - create PR analysis records"""
    
    action = payload.get("action")
    pr_data = payload.get("pull_request", {})
    pr_number = pr_data.get("number")
    
    # Log PR event
    log_entry = IntegrationLog(
        repository_config_id=repo_config.id,
        event_type=f"pr_{action}",
        event_data={
            "pr_number": pr_number,
            "action": action,
            "author": pr_data.get("user", {}).get("login"),
            "auto_analysis": False
        },
        success=True
    )
    db.add(log_entry)
    
    # Create or update PRAnalysis record for relevant actions
    if action in ["opened", "synchronize", "reopened"]:
        # Check if PR analysis already exists
        existing_analysis = db.query(PRAnalysis).filter(
            PRAnalysis.repository_config_id == repo_config.id,
            PRAnalysis.pr_number == pr_number
        ).first()
        
        if existing_analysis:
            # Update existing analysis with display metadata
            existing_analysis.status = "pending"
            existing_analysis.pr_title = pr_data.get("title")
            existing_analysis.pr_description = pr_data.get("body")
            existing_analysis.source_branch = pr_data.get("head", {}).get("ref")
            existing_analysis.target_branch = pr_data.get("base", {}).get("ref")
            existing_analysis.author = pr_data.get("user", {}).get("login")
            existing_analysis.event_type = f"webhook_pr_{action}"
            existing_analysis.updated_at = datetime.utcnow()
            
            # Store display metadata using hybrid approach
            display_metadata = DiffCalculationService.extract_display_metadata_from_payload(
                payload, "pull_request"
            )
            existing_analysis.commit_sha = display_metadata["commit_sha"]
            existing_analysis.files_changed = display_metadata["files_changed"]
            existing_analysis.lines_added = display_metadata["lines_added"]
            existing_analysis.lines_removed = display_metadata["lines_removed"]
            
            pr_analysis = existing_analysis
        else:
            # Extract display metadata from payload
            display_metadata = DiffCalculationService.extract_display_metadata_from_payload(
                payload, "pull_request"
            )
            
            # Create new analysis with display metadata only
            pr_analysis = PRAnalysis(
                repository_config_id=repo_config.id,
                pr_number=pr_number,
                pr_title=pr_data.get("title"),
                pr_description=pr_data.get("body"),
                pr_url=pr_data.get("html_url"),
                author=pr_data.get("user", {}).get("login"),
                source_branch=pr_data.get("head", {}).get("ref"),
                target_branch=pr_data.get("base", {}).get("ref"),
                event_type=f"webhook_pr_{action}",
                status="pending",
                # Store display metadata immediately
                commit_sha=display_metadata["commit_sha"],
                files_changed=display_metadata["files_changed"],
                lines_added=display_metadata["lines_added"],
                lines_removed=display_metadata["lines_removed"],
                analysis_results={"message": "PR received via webhook - ready for AI analysis"}
            )
            
            db.add(pr_analysis)
        
        db.commit()
        
        return {
            "message": f"PR {action} event processed", 
            "pr_number": pr_number,
            "analysis_id": pr_analysis.id,
            "note": "PR analysis record created with display metadata - detailed diff calculated at analysis time"
        }
    
    db.commit()
    
    return {
        "message": f"PR {action} event logged", 
        "pr_number": pr_number,
        "note": f"Action '{action}' does not trigger analysis record creation"
    }


def _should_analyze_push(repo_config: RepositoryConfig, branch: str, commits: List[Dict]) -> bool:
    """Determine if a push should trigger AI analysis"""
    
    # Check branch patterns
    if repo_config.branch_patterns:
        patterns = repo_config.branch_patterns.get("patterns", [])
        if patterns and branch not in patterns:
            return False
    
    # Skip if no commits or only merge commits
    if not commits or all(commit.get("message", "").startswith("Merge") for commit in commits):
        return False
    
    # Check for significant changes (files modified)
    total_changes = sum(len(commit.get("added", [])) + len(commit.get("modified", [])) + len(commit.get("removed", [])) for commit in commits)
    
    # If no file changes detected in payload, still analyze if commits exist
    # (GitHub webhook payload doesn't always include file change details)
    if total_changes == 0:
        # Analyze if we have non-merge commits (assume they have changes)
        return len(commits) > 0
    
    return total_changes > 0


def _generate_push_description(commits: List[Dict]) -> str:
    """Generate a description from commit messages"""
    
    if not commits:
        return "No commits found"
    
    if len(commits) == 1:
        return f"Single commit: {commits[0].get('message', 'No message')}"
    
    messages = [commit.get("message", "No message") for commit in commits[:5]]  # Limit to first 5
    description = f"{len(commits)} commits:\n" + "\n".join(f"- {msg}" for msg in messages)
    
    if len(commits) > 5:
        description += f"\n... and {len(commits) - 5} more commits"
    
    return description


async def analyze_push_with_commit_data(
    push_analysis_id: int,
    payload: Dict[str, Any],
    access_token: str
):
    """Background task to analyze push events with commit data"""
    
    from app.database.session import SessionLocal
    db = SessionLocal()
    
    try:
        push_analysis = db.query(PRAnalysis).filter(PRAnalysis.id == push_analysis_id).first()
        if not push_analysis:
            logger.error(f"Push analysis {push_analysis_id} not found")
            return
        
        # Get repository config to access repo URL
        repo_config = db.query(RepositoryConfig).filter(RepositoryConfig.id == push_analysis.repository_config_id).first()
        if not repo_config:
            logger.error(f"Repository config not found for push analysis {push_analysis_id}")
            return
        
        # Extract commit data and get real diff data from GitHub API
        commits = payload.get("commits", [])
        files_changed = set()
        total_additions = 0
        total_deletions = 0
        all_diff_content = []
        
        # Process each commit with GitHub API calls
        for commit in commits:
            commit_sha = commit.get("id")
            if commit_sha and access_token:
                # Fetch real diff data from GitHub API
                diff_data = await fetch_github_commit_diff(commit_sha, repo_config.repository_url, access_token)
                
                # Update totals with real data
                total_additions += diff_data.get("total_additions", 0)
                total_deletions += diff_data.get("total_deletions", 0)
                
                # Extract file names from API response
                for file_data in diff_data.get("files", []):
                    filename = file_data.get("filename", "")
                    if filename:
                        files_changed.add(filename)
                
                # Add commit message to diff content
                if diff_data.get("diff_content"):
                    all_diff_content.append(f"Commit {commit_sha[:8]}: {diff_data['diff_content']}")
                    
            # Fallback to webhook data if API call fails
            else:
                files_changed.update(commit.get("added", []))
                files_changed.update(commit.get("modified", []))  
                files_changed.update(commit.get("removed", []))
                all_diff_content.append(f"Commit {commit.get('id', 'unknown')[:8]}: {commit.get('message', 'No message')}")
        
        # Combine all diff content
        diff_content = "\n".join(all_diff_content) if all_diff_content else _generate_push_description(commits)
        
        # Update analysis with push data
        push_analysis.diff_content = diff_content
        push_analysis.files_changed = list(files_changed)
        push_analysis.lines_added = total_additions
        push_analysis.lines_removed = total_deletions
        
        db.commit()
        
        # Run AI analysis using the existing service
        success = await GitHubIntegrationService.analyze_pull_request(push_analysis, db)
        
        logger.info(f"Push analysis completed for branch {push_analysis.source_branch}: {'success' if success else 'failed'}")
        
    except Exception as e:
        logger.error(f"Background push analysis failed: {e}")
        
        if push_analysis:
            push_analysis.status = "failed"
            db.commit()
    finally:
        db.close()


async def handle_gitlab_push_event(
    payload: Dict[str, Any], 
    repo_config: RepositoryConfig, 
    db: Session,
    background_tasks: BackgroundTasks = None
):
    """Handle GitLab push events with AI analysis"""
    
    commits = payload.get("commits", [])
    branch = payload.get("ref", "").replace("refs/heads/", "")
    
    # Log push event
    log_entry = IntegrationLog(
        repository_config_id=repo_config.id,
        event_type="push_event",
        event_data={
            "branch": branch,
            "commits_count": len(commits),
            "pusher": payload.get("user_username", payload.get("user_name", "unknown"))
        },
        success=True
    )
    db.add(log_entry)
    
    # Check if this push should trigger AI analysis
    should_analyze = _should_analyze_push(repo_config, branch, commits)
    
    if should_analyze and repo_config.auto_analyze_prs:
        # Create a push-based analysis (similar to PR analysis but for direct pushes)
        push_analysis = PRAnalysis(
            repository_config_id=repo_config.id,
            pr_number=0,  # 0 indicates push event, not PR
            pr_title=f"Push to {branch}",
            pr_description=_generate_push_description(commits),
            pr_url=payload.get("project", {}).get("web_url", payload.get("repository", {}).get("homepage", "")),
            source_branch=branch,
            target_branch=branch,
            author=payload.get("user_username", payload.get("user_name", "unknown")),
            event_type="push",
            status="pending"
        )
        db.add(push_analysis)
        db.commit()
        db.refresh(push_analysis)
        
        # Extract commit data for analysis
        files_changed = set()
        total_file_additions = 0
        total_file_deletions = 0
        
        for commit in commits:
            # Add file names to set
            added_files = commit.get("added", [])
            modified_files = commit.get("modified", [])
            removed_files = commit.get("removed", [])
            
            files_changed.update(added_files)
            files_changed.update(modified_files)
            files_changed.update(removed_files)
            
            # Count files (not ideal, but GitLab webhook doesn't provide line counts)
            total_file_additions += len(added_files) + len(modified_files)
            total_file_deletions += len(removed_files)
        
        # Update analysis with push data
        push_analysis.diff_content = _generate_commit_diff_summary(commits)
        push_analysis.files_changed = list(files_changed)
        # Note: GitLab webhook doesn't provide actual line counts, using file counts as approximation
        push_analysis.lines_added = total_file_additions
        push_analysis.lines_removed = total_file_deletions
        
        db.commit()
        
        # Run AI analysis in background to avoid webhook timeout
        if background_tasks:
            background_tasks.add_task(run_ai_analysis_for_push, push_analysis.id)
            logger.info(f"AI analysis queued for push analysis {push_analysis.id}")
        
        log_entry.event_data["analysis_triggered"] = True
        log_entry.event_data["analysis_id"] = push_analysis.id
    
    db.commit()
    
    return {
        "message": "GitLab push event processed",
        "branch": branch,
        "commits": len(commits),
        "analysis_triggered": should_analyze and repo_config.auto_analyze_prs
    }


def _generate_commit_diff_summary(commits: List[Dict]) -> str:
    """Generate a diff summary from commit messages and changes"""
    
    if not commits:
        return "No commits found"
    
    summary = f"Summary of {len(commits)} commit(s):\n\n"
    
    for commit in commits:
        summary += f"Commit: {commit.get('message', 'No message')}\n"
        
        added = commit.get('added', [])
        modified = commit.get('modified', [])
        removed = commit.get('removed', [])
        
        if added:
            summary += f"  Added: {', '.join(added)}\n"
        if modified:
            summary += f"  Modified: {', '.join(modified)}\n"
        if removed:
            summary += f"  Removed: {', '.join(removed)}\n"
        summary += "\n"
    
    return summary


async def analyze_gitlab_push_data(
    push_analysis_id: int,
    access_token: str
):
    """Background task to analyze GitLab push events"""
    
    from app.database.session import SessionLocal
    db = SessionLocal()
    
    try:
        push_analysis = db.query(PRAnalysis).filter(PRAnalysis.id == push_analysis_id).first()
        if not push_analysis:
            logger.error(f"Push analysis {push_analysis_id} not found")
            return
        
        # Run AI analysis using the existing service
        success = await GitHubIntegrationService.analyze_pull_request(push_analysis, db)
        
        logger.info(f"GitLab push analysis completed for branch {push_analysis.source_branch}: {'success' if success else 'failed'}")
        
    except Exception as e:
        logger.error(f"Background GitLab push analysis failed: {e}")
        
        if push_analysis:
            push_analysis.status = "failed"
            db.commit()
    finally:
        db.close()


async def analyze_pr_with_github_data(
    pr_analysis_id: int,
    repo_owner: str,
    repo_name: str,
    access_token: str
):
    """Background task to fetch GitHub data and run AI analysis"""
    
    # Get a new database session for background task
    from app.database.session import SessionLocal
    db = SessionLocal()
    
    try:
        pr_analysis = db.query(PRAnalysis).filter(PRAnalysis.id == pr_analysis_id).first()
        if not pr_analysis:
            logger.error(f"PR analysis {pr_analysis_id} not found")
            return
        
        # Initialize GitHub API client
        github_client = GitHubAPIClient(access_token)
        
        # Fetch PR diff
        diff_content = await github_client.get_pull_request_diff(
            repo_owner, repo_name, pr_analysis.pr_number
        )
        
        # Fetch changed files
        files_data = await github_client.get_pull_request_files(
            repo_owner, repo_name, pr_analysis.pr_number
        )
        
        # Extract file statistics
        files_changed = [file_data["filename"] for file_data in files_data]
        lines_added = sum(file_data.get("additions", 0) for file_data in files_data)
        lines_removed = sum(file_data.get("deletions", 0) for file_data in files_data)
        
        # Update PR analysis with fetched data
        pr_analysis.diff_content = diff_content
        pr_analysis.files_changed = files_changed
        pr_analysis.lines_added = lines_added
        pr_analysis.lines_removed = lines_removed
        
        db.commit()
        
        # Run AI analysis
        success = await GitHubIntegrationService.analyze_pull_request(pr_analysis, db)
        
        logger.info(f"PR analysis completed for #{pr_analysis.pr_number}: {'success' if success else 'failed'}")
        
    except Exception as e:
        logger.error(f"Background PR analysis failed: {e}")
        
        # Update status to failed
        if pr_analysis:
            pr_analysis.status = "failed"
            db.commit()
    finally:
        db.close()

@router.get("/")
async def list_webhooks(
    project_id: int = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all configured webhooks"""
    
    query = db.query(RepositoryConfig)
    
    if project_id:
        query = query.filter(RepositoryConfig.project_id == project_id)
    
    configs = query.all()
    
    webhooks = []
    for config in configs:
        webhook_info = {
            "id": config.id,
            "project_id": config.project_id,
            "repository_url": config.repository_url,
                "provider": config.provider,
                "webhook_url": f"/api/v1/webhooks/{config.provider}/{config.id}",
            "is_active": config.is_active,
            "auto_analyze_prs": config.auto_analyze_prs,
            "auto_create_test_cases": config.auto_create_test_cases,
            "branch_patterns": config.branch_patterns,
                "events": ["push", "pull_request"] if config.provider == "github" else ["push", "merge_request"],
            "created_at": config.created_at
        }
        webhooks.append(webhook_info)
    
    return {"webhooks": webhooks}

@router.get("/{webhook_id}")
async def get_webhook_details(
    webhook_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get detailed webhook information"""
    
    config = db.query(RepositoryConfig).filter(RepositoryConfig.id == webhook_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    webhook_details = {
        "id": config.id,
        "project_id": config.project_id,
        "repository_url": config.repository_url,
        "repository_type": config.repository_type,
        "webhook_url": f"/api/v1/repositories/webhooks/{config.repository_type}/{config.id}",
        "is_active": config.is_active,
        "auto_analyze_prs": config.auto_analyze_prs,
        "auto_create_test_cases": config.auto_create_test_cases,
        "branch_patterns": config.branch_patterns,
        "settings": config.settings,
        "events": ["push", "pull_request"] if config.repository_type == "github" else ["push", "merge_request"],
        "setup_instructions": get_webhook_setup_instructions(config),
        "created_at": config.created_at,
        "updated_at": config.updated_at
    }
    
    return webhook_details

@router.post("/{webhook_id}/test")
async def test_webhook(
    webhook_id: int,
    test_payload: Dict[str, Any] = None,
    current_user: User = Depends(require_permission(Permission.CONFIGURE_SETTINGS)),
    db: Session = Depends(get_db)
):
    """Test a webhook configuration"""
    
    config = db.query(RepositoryConfig).filter(RepositoryConfig.id == webhook_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    # Create a test payload if none provided
    if not test_payload:
        test_payload = create_test_payload(config.provider)
    
    # Simulate processing the webhook
    try:
        result = {
            "webhook_id": webhook_id,
            "test_status": "success",
            "repository_type": config.provider,
            "test_timestamp": datetime.utcnow().isoformat(),
            "payload_processed": True,
            "actions_triggered": []
        }
        
        # Simulate actions that would be triggered
        if config.auto_analyze_prs and "pull_request" in str(test_payload):
            result["actions_triggered"].append("pr_analysis_queued")
        
        if config.auto_create_test_cases:
            result["actions_triggered"].append("test_case_generation_queued")
        
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Webhook test failed: {str(e)}"
        )

@router.get("/{webhook_id}/logs")
async def get_webhook_logs(
    webhook_id: int,
    limit: int = 50,
    skip: int = 0,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get webhook execution logs"""
    
    config = db.query(RepositoryConfig).filter(RepositoryConfig.id == webhook_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    # In a real implementation, you would fetch from a webhook logs table
    # For now, return sample logs
    sample_logs = [
        {
            "id": i + 1,
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "pull_request" if i % 2 == 0 else "push",
            "status": "success" if i % 3 != 0 else "failed",
            "processing_time_ms": 150 + (i * 10),
            "actions_triggered": ["pr_analysis"] if i % 2 == 0 else ["code_analysis"],
            "error_message": "Rate limit exceeded" if i % 3 == 0 else None
        }
        for i in range(skip, skip + min(limit, 10))  # Limit sample data
    ]
    
    return {
        "webhook_id": webhook_id,
        "total_logs": 100,  # Sample total
        "logs": sample_logs
    }

@router.put("/{webhook_id}/toggle")
async def toggle_webhook_status(
    webhook_id: int,
    current_user: User = Depends(require_permission(Permission.CONFIGURE_SETTINGS)),
    db: Session = Depends(get_db)
):
    """Enable or disable a webhook"""
    
    config = db.query(RepositoryConfig).filter(RepositoryConfig.id == webhook_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    config.is_active = not config.is_active
    config.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(config)
    
    return {
        "webhook_id": webhook_id,
        "is_active": config.is_active,
        "message": f"Webhook {'enabled' if config.is_active else 'disabled'} successfully"
    }

@router.get("/events/types")
async def get_supported_webhook_events():
    """Get list of supported webhook events"""
    
    return {
        "github": {
            "supported_events": [
                "push",
                "pull_request",
                "pull_request_review",
                "issues",
                "issue_comment",
                "commit_comment",
                "create",
                "delete",
                "release"
            ],
            "recommended_events": ["push", "pull_request"],
            "content_type": "application/json"
        },
        "gitlab": {
            "supported_events": [
                "push",
                "merge_request",
                "merge_request_comment",
                "issue",
                "issue_comment",
                "tag_push",
                "pipeline",
                "wiki_page"
            ],
            "recommended_events": ["push", "merge_request"],
            "content_type": "application/json"
        },
        "bitbucket": {
            "supported_events": [
                "push",
                "pullrequest:created",
                "pullrequest:updated",
                "pullrequest:approved",
                "pullrequest:unapproved",
                "pullrequest:fulfilled",
                "pullrequest:rejected"
            ],
            "recommended_events": ["push", "pullrequest:created"],
            "content_type": "application/json"
        }
    }

def get_webhook_setup_instructions(config: RepositoryConfig) -> Dict[str, Any]:
    """Generate setup instructions for a webhook"""
    
    webhook_url = f"https://your-domain.com/api/v1/repositories/webhooks/{config.repository_type}/{config.id}"
    
    if config.repository_type == "github":
        return {
            "platform": "GitHub",
            "steps": [
                "Go to your repository settings",
                "Click on 'Webhooks' in the left sidebar",
                "Click 'Add webhook'",
                f"Set Payload URL to: {webhook_url}",
                "Set Content type to: application/json",
                f"Set Secret to your webhook secret",
                "Select 'Let me select individual events'",
                "Check: Push events, Pull requests",
                "Ensure 'Active' is checked",
                "Click 'Add webhook'"
            ],
            "webhook_url": webhook_url,
            "content_type": "application/json",
            "events": ["push", "pull_request"],
            "secret_required": True
        }
    
    elif config.repository_type == "gitlab":
        return {
            "platform": "GitLab",
            "steps": [
                "Go to your project settings",
                "Click on 'Webhooks' in the left sidebar",
                f"Set URL to: {webhook_url}",
                f"Set Secret Token to your webhook secret",
                "Check: Push events, Merge request events",
                "Ensure 'Enable SSL verification' is checked",
                "Click 'Add webhook'"
            ],
            "webhook_url": webhook_url,
            "content_type": "application/json",
            "events": ["push", "merge_request"],
            "secret_required": True
        }
    
    else:
        return {
            "platform": config.repository_type.title(),
            "webhook_url": webhook_url,
            "message": "Please refer to your platform's documentation for webhook setup instructions"
        }

def create_test_payload(repository_type: str) -> Dict[str, Any]:
    """Create a test payload for webhook testing"""
    
    if repository_type == "github":
        return {
            "action": "opened",
            "number": 1,
            "pull_request": {
                "id": 123456789,
                "number": 1,
                "title": "Test PR for webhook",
                "body": "This is a test pull request",
                "head": {"ref": "feature/test-branch"},
                "base": {"ref": "main"},
                "user": {"login": "test-user"}
            },
            "repository": {
                "name": "test-repo",
                "full_name": "user/test-repo"
            }
        }
    
    elif repository_type == "gitlab":
        return {
            "object_kind": "merge_request",
            "object_attributes": {
                "id": 123,
                "iid": 1,
                "title": "Test MR for webhook",
                "description": "This is a test merge request",
                "source_branch": "feature/test-branch",
                "target_branch": "main",
                "action": "open"
            },
            "user": {"username": "test-user"},
            "project": {"name": "test-repo"}
        }
    
    else:
        return {"test": True, "repository_type": repository_type}

@router.get("/repositories/{config_id}/fetch-prs")
async def fetch_repository_prs(
    config_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Fetch existing PR analysis records from database - display only, no creation"""
    
    logger.info(f"Fetching existing PR analyses for repository config {config_id}")
    
    try:
        # Get repository config
        repo_config = db.query(RepositoryConfig).filter(
            RepositoryConfig.id == config_id
        ).first()
        
        if not repo_config:
            logger.error(f"Repository config {config_id} not found")
            raise HTTPException(status_code=404, detail="Repository configuration not found")
        
        logger.info(f"Found repo config: {repo_config.repository_name} ({repo_config.provider})")
        
        # Query existing PR analysis records
        pr_analyses = db.query(PRAnalysis).filter(
            PRAnalysis.repository_config_id == config_id
        ).order_by(PRAnalysis.created_at.desc()).all()
        
        # Separate PRs from commits
        prs_data = []
        commits_data = []
        
        for analysis in pr_analyses:
            analysis_data = {
                "id": analysis.id,
                "number": analysis.pr_number if analysis.pr_number > 0 else None,
                "title": analysis.pr_title,
                "author": analysis.author,
                "status": analysis.status,
                "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
                "updated_at": analysis.updated_at.isoformat() if analysis.updated_at else None,
                "source_branch": analysis.source_branch,
                "target_branch": analysis.target_branch,
                "event_type": analysis.event_type,
                "url": analysis.pr_url
            }
            
            if analysis.pr_number == 0:  # It's a commit
                # Extract SHA from URL or title for display
                sha = ""
                if analysis.pr_url and "/commit/" in analysis.pr_url:
                    sha = analysis.pr_url.split("/commit/")[-1][:8]
                elif "Commit" in analysis.pr_title:
                    # Try to extract from title
                    parts = analysis.pr_title.split()
                    for part in parts:
                        if len(part) >= 7 and all(c in '0123456789abcdef' for c in part.lower()):
                            sha = part[:8]
                            break
                
                analysis_data["sha"] = sha
                analysis_data["message"] = analysis.pr_title
                commits_data.append(analysis_data)
            else:  # It's a PR
                prs_data.append(analysis_data)
        
        logger.info(f"Found {len(prs_data)} PR records and {len(commits_data)} commit records")
        
        return {
            "success": True,
            "repository_config": {
                "id": repo_config.id,
                "name": repo_config.repository_name,
                "provider": repo_config.provider,
                "url": repo_config.repository_url
            },
            "prs": prs_data,
            "commits": commits_data,
            "total_analyses": len(pr_analyses),
            "message": f"Retrieved {len(prs_data)} PR(s) and {len(commits_data)} commit(s) from database. Use webhooks to capture new events automatically."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching PR analyses: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch PR analyses: {str(e)}")