from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.database.models import User, TestCase, TestCaseVersion, TestSuite
from app.auth.deps import get_current_active_user, require_permission
from app.core.permissions import Permission
import difflib

router = APIRouter()

@router.get("/test-case/{case_id}/versions")
async def get_test_case_versions(
    case_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all versions of a test case"""
    
    test_case = db.query(TestCase).filter(TestCase.id == case_id).first()
    if not test_case:
        raise HTTPException(status_code=404, detail="Test case not found")
    
    versions = db.query(TestCaseVersion).filter(
        TestCaseVersion.test_case_id == case_id
    ).order_by(TestCaseVersion.version_number.desc()).all()
    
    return versions

@router.get("/test-case/{case_id}/compare")
async def compare_test_case_versions(
    case_id: int,
    version1: int = Query(..., description="First version to compare"),
    version2: int = Query(..., description="Second version to compare"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Compare two versions of a test case"""
    
    test_case = db.query(TestCase).filter(TestCase.id == case_id).first()
    if not test_case:
        raise HTTPException(status_code=404, detail="Test case not found")
    
    # Get the two versions
    version_1 = db.query(TestCaseVersion).filter(
        TestCaseVersion.test_case_id == case_id,
        TestCaseVersion.version_number == version1
    ).first()
    
    version_2 = db.query(TestCaseVersion).filter(
        TestCaseVersion.test_case_id == case_id,
        TestCaseVersion.version_number == version2
    ).first()
    
    if not version_1 or not version_2:
        raise HTTPException(status_code=404, detail="One or both versions not found")
    
    # Generate diff for each field
    fields_to_compare = ['title', 'description', 'preconditions', 'steps', 'expected_results']
    
    diffs = {}
    for field in fields_to_compare:
        value1 = getattr(version_1, field, '') or ''
        value2 = getattr(version_2, field, '') or ''
        
        # Generate unified diff
        diff = list(difflib.unified_diff(
            value1.splitlines(keepends=True),
            value2.splitlines(keepends=True),
            fromfile=f"Version {version1}",
            tofile=f"Version {version2}",
            lineterm=''
        ))
        
        diffs[field] = {
            'has_changes': value1 != value2,
            'diff': ''.join(diff) if diff else None,
            'old_value': value1,
            'new_value': value2
        }
    
    return {
        'test_case_id': case_id,
        'version_1': {
            'version_number': version1,
            'created_at': version_1.created_at,
            'created_by_id': version_1.created_by_id
        },
        'version_2': {
            'version_number': version2,
            'created_at': version_2.created_at,
            'created_by_id': version_2.created_by_id
        },
        'changes': diffs
    }

@router.get("/test-case/{case_id}/delta/{version_number}")
async def get_version_delta(
    case_id: int,
    version_number: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get delta between a specific version and the previous version"""
    
    test_case = db.query(TestCase).filter(TestCase.id == case_id).first()
    if not test_case:
        raise HTTPException(status_code=404, detail="Test case not found")
    
    # Get current version
    current_version = db.query(TestCaseVersion).filter(
        TestCaseVersion.test_case_id == case_id,
        TestCaseVersion.version_number == version_number
    ).first()
    
    if not current_version:
        raise HTTPException(status_code=404, detail="Version not found")
    
    # Get previous version
    previous_version = db.query(TestCaseVersion).filter(
        TestCaseVersion.test_case_id == case_id,
        TestCaseVersion.version_number == version_number - 1
    ).first()
    
    if not previous_version:
        return {
            'test_case_id': case_id,
            'version_number': version_number,
            'is_initial_version': True,
            'changes': {},
            'change_summary': 'Initial version'
        }
    
    # Generate changes
    fields_to_compare = ['title', 'description', 'preconditions', 'steps', 'expected_results', 'priority', 'type']
    changes = {}
    change_count = 0
    
    for field in fields_to_compare:
        old_value = getattr(previous_version, field, '')
        new_value = getattr(current_version, field, '')
        
        if old_value != new_value:
            changes[field] = {
                'old_value': old_value,
                'new_value': new_value,
                'changed': True
            }
            change_count += 1
        else:
            changes[field] = {
                'value': old_value,
                'changed': False
            }
    
    return {
        'test_case_id': case_id,
        'version_number': version_number,
        'previous_version_number': version_number - 1,
        'is_initial_version': False,
        'changes': changes,
        'change_count': change_count,
        'change_summary': f"{change_count} field{'s' if change_count != 1 else ''} changed",
        'version_info': {
            'created_at': current_version.created_at,
            'created_by_id': current_version.created_by_id,
            'change_reason': current_version.change_reason
        }
    }

@router.get("/project/{project_id}/recent-changes")
async def get_recent_test_case_changes(
    project_id: int,
    days: int = Query(7, description="Number of days to look back"),
    limit: int = Query(50, description="Maximum number of changes to return"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get recent test case changes in a project"""
    
    from datetime import datetime, timedelta
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    # Get recent versions for test cases in this project
    recent_versions = db.query(TestCaseVersion).join(TestCase).join(TestSuite).filter(
        TestSuite.project_id == project_id,
        TestCaseVersion.created_at >= cutoff_date
    ).order_by(TestCaseVersion.created_at.desc()).limit(limit).all()
    
    changes = []
    for version in recent_versions:
        test_case = version.test_case
        changes.append({
            'version_id': version.id,
            'test_case_id': test_case.id,
            'test_case_title': test_case.title,
            'version_number': version.version_number,
            'change_reason': version.change_reason,
            'created_at': version.created_at,
            'created_by_id': version.created_by_id,
            'is_initial_version': version.version_number == 1
        })
    
    return {
        'project_id': project_id,
        'days': days,
        'total_changes': len(changes),
        'changes': changes
    }