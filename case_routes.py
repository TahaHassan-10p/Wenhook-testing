from typing import List
import logging
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session, joinedload, selectinload
from sqlalchemy.sql import func
from app.database.session import get_db
from app.database.models import User, TestCase, TestSuite, TestCaseVersion, TestCaseActivity, ActivityType, TestCasePriority, TestCaseType
from app.schemas.testworthy import TestCaseCreate, TestCaseUpdate, TestCaseResponse
from app.auth.deps import get_current_active_user, require_permission
from app.core.permissions import Permission
from app.services.analytics import analytics

router = APIRouter()
logger = logging.getLogger(__name__)

def check_case_access(case: TestCase, current_user: User, required_permission: str):
    """Check if user has permission to access test case"""
    from app.core.permissions import has_permission
    if not has_permission(current_user.role, required_permission):
        # Allow case creators to manage their own cases
        if required_permission != Permission.READ and case.created_by_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions for this test case"
            )

@router.get("/")
async def get_test_cases(
    project_id: int = None,
    suite_id: int = None,
    milestone_id: int = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission(Permission.READ))
):
    """Get test cases with optional filters, returns paginated results with total count"""
    try:
        query = db.query(TestCase)

        if suite_id:
            query = query.filter(TestCase.suite_id == suite_id)
        if milestone_id:
            query = query.filter(TestCase.milestone_id == milestone_id)
        if project_id:
            query = query.filter(TestCase.project_id == project_id)

        # OPTIMIZED: Combine count and stats into single query
        from sqlalchemy.sql import case as sql_case
        
        stats_query = query.with_entities(
            func.count(TestCase.id).label('total'),
            func.sum(sql_case((TestCase.priority == TestCasePriority.HIGH, 1), else_=0)).label('high_priority'),
            func.sum(sql_case((TestCase.priority == TestCasePriority.CRITICAL, 1), else_=0)).label('critical_priority'),
            func.sum(sql_case((TestCase.type == TestCaseType.FUNCTIONAL, 1), else_=0)).label('functional')
        ).first()
        
        total_count = int(stats_query.total or 0)
        
        stats = {
            'total': total_count,
            'high_priority': int(stats_query.high_priority or 0),
            'critical_priority': int(stats_query.critical_priority or 0),
            'functional': int(stats_query.functional or 0),
            'automated': 0  # Not tracked in current model, would need is_automated field
        }

        # Use selectinload to prevent N+1 query problem and apply pagination
        test_cases = query.options(selectinload(TestCase.assigned_to)).offset(skip).limit(limit).all()
        
        # Add assigned_to_name for each test case
        items = []
        for case in test_cases:
            case_dict = {k: v for k, v in case.__dict__.items() if not k.startswith('_')}
            case_dict['assigned_to_name'] = case.assigned_to.name if case.assigned_to else None
            # Ensure JSON fields are lists, not empty dicts
            case_dict['reference_ids'] = case.reference_ids if case.reference_ids else []
            case_dict['attachments'] = case.attachments if case.attachments else []
            items.append(case_dict)
        
        return {
            'items': items,
            'total': total_count,
            'skip': skip,
            'limit': limit,
            'stats': stats
        }
    except Exception as e:
        logger.error(f"Error loading test cases: {str(e)}")
        # Return empty result set instead of error to prevent UI from breaking
        return {
            'items': [],
            'total': 0,
            'skip': skip,
            'limit': limit,
            'stats': {
                'total': 0,
                'high_priority': 0,
                'critical_priority': 0,
                'functional': 0,
                'automated': 0
            }
        }


@router.get("/stats")
async def get_test_cases_stats(
    project_id: int = None,
    suite_id: int = None,
    milestone_id: int = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission(Permission.READ))
):
    """Get test cases statistics only (no items) - lightweight endpoint for overview tab"""
    try:
        query = db.query(TestCase)

        if suite_id:
            query = query.filter(TestCase.suite_id == suite_id)
        if milestone_id:
            query = query.filter(TestCase.milestone_id == milestone_id)
        if project_id:
            query = query.filter(TestCase.project_id == project_id)

        # Get stats only - no pagination needed
        from sqlalchemy.sql import case as sql_case
        
        stats_query = query.with_entities(
            func.count(TestCase.id).label('total'),
            func.sum(sql_case((TestCase.priority == TestCasePriority.HIGH, 1), else_=0)).label('high_priority'),
            func.sum(sql_case((TestCase.priority == TestCasePriority.CRITICAL, 1), else_=0)).label('critical_priority'),
            func.sum(sql_case((TestCase.type == TestCaseType.FUNCTIONAL, 1), else_=0)).label('functional')
        ).first()
        
        return {
            'total': int(stats_query.total or 0),
            'high_priority': int(stats_query.high_priority or 0),
            'critical_priority': int(stats_query.critical_priority or 0),
            'functional': int(stats_query.functional or 0),
            'automated': 0  # Not tracked in current model
        }
    except Exception as e:
        logger.error(f"Error loading test case stats: {str(e)}")
        return {
            'total': 0,
            'high_priority': 0,
            'critical_priority': 0,
            'functional': 0,
            'automated': 0
        }


@router.get("/by-suite/{suite_id}", response_model=List[TestCaseResponse])
async def read_test_cases_by_suite(
    suite_id: int,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get test cases by suite ID"""
    # Check if suite exists
    suite = db.query(TestSuite).filter(TestSuite.id == suite_id).first()
    if not suite:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test suite not found"
        )
    
    test_cases = db.query(TestCase).filter(
        TestCase.suite_id == suite_id
    ).offset(skip).limit(limit).all()
    
    return test_cases

@router.post("/", response_model=TestCaseResponse)
async def create_test_case(
    case_data: TestCaseCreate,
    current_user: User = Depends(require_permission(Permission.CREATE)),
    db: Session = Depends(get_db)
):
    """Create new test case"""
    # Check if suite exists
    suite = db.query(TestSuite).filter(TestSuite.id == case_data.suite_id).first()
    #Get project id from suite if not sent
    if not suite:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test suite not found"
        )
    if not case_data.project_id:
        case_data.project_id = suite.project_id if suite else None
    
    # Get next case number for this project
    max_case_number = db.query(func.max(TestCase.case_number)).filter(
        TestCase.project_id == case_data.project_id
    ).scalar()
    next_case_number = (max_case_number or 0) + 1
    
    db_case = TestCase(
        case_number=next_case_number,
        title=case_data.title,
        description=case_data.description,
        preconditions=case_data.preconditions,
        steps=case_data.steps,
        expected_results=case_data.expected_results,
        priority=case_data.priority,
        type=case_data.type,
        reference_ids=case_data.reference_ids or [],
        attachments=case_data.attachments or [],
        milestone_id=case_data.milestone_id,
        assigned_to_id=case_data.assigned_to_id,
        project_id=case_data.project_id,
        suite_id=case_data.suite_id,
        created_by_id=current_user.id
    )
    
    db.add(db_case)
    db.commit()
    db.refresh(db_case)
    
    # Create initial version
    version = TestCaseVersion(
        case_id=db_case.id,
        version_number=1,
        title=db_case.title,
        description=db_case.description,
        preconditions=db_case.preconditions,
        steps=db_case.steps,
        expected_results=db_case.expected_results,
        priority=db_case.priority,
        type=db_case.type,
        created_by_id=current_user.id
    )
    db.add(version)
    
    # Log activity
    activity = TestCaseActivity(
        test_case_id=db_case.id,
        activity_type=ActivityType.CREATED,
        description=f"Test case created",
        created_by_id=current_user.id
    )
    db.add(activity)
    db.commit()
    
    # Track test case creation
    analytics.track_event(
        "test_case_created",
        user_id=current_user.id,
        properties={
            "test_case_id": db_case.id,
            "case_number": db_case.case_number,
            "project_id": db_case.project_id,
            "suite_id": db_case.suite_id,
            "priority": db_case.priority.value if db_case.priority else None,
            "type": db_case.type.value if db_case.type else None,
            "has_steps": bool(db_case.steps),
            "steps_count": len(db_case.steps) if db_case.steps else 0
        }
    )
    
    return db_case

@router.get("/{case_id}", response_model=TestCaseResponse)
async def read_test_case(
    case_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get test case by ID"""
    case = db.query(TestCase).options(joinedload(TestCase.assigned_to)).filter(TestCase.id == case_id).first()
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test case not found"
        )
    
    check_case_access(case, current_user, Permission.READ)
    
    # Use TestCaseResponse model to ensure proper serialization
    return TestCaseResponse(
        id=case.id,
        case_number=case.case_number,
        title=case.title,
        description=case.description,
        preconditions=case.preconditions,
        steps=case.steps,
        expected_results=case.expected_results,
        priority=case.priority,
        type=case.type,
        reference_ids=case.reference_ids if case.reference_ids else [],
        attachments=case.attachments if case.attachments else [],
        suite_id=case.suite_id,
        milestone_id=case.milestone_id,
        assigned_to_id=case.assigned_to_id,
        assigned_to_name=case.assigned_to.name if case.assigned_to else None,
        created_at=case.created_at,
        updated_at=case.updated_at,
        created_by_id=case.created_by_id
    )

@router.put("/{case_id}", response_model=TestCaseResponse)
async def update_test_case(
    case_id: int,
    case_update: TestCaseUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update test case"""
    case = db.query(TestCase).filter(TestCase.id == case_id).first()
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test case not found"
        )
    
    check_case_access(case, current_user, Permission.UPDATE)
    
    # Check if any content changed for versioning
    content_changed = False
    old_values = {}
    
    # Update fields and track changes
    if case_update.title is not None and case_update.title != case.title:
        old_values['title'] = case.title
        case.title = case_update.title
        content_changed = True
    
    if case_update.description is not None and case_update.description != case.description:
        old_values['description'] = case.description
        case.description = case_update.description
        content_changed = True
    
    if case_update.preconditions is not None and case_update.preconditions != case.preconditions:
        old_values['preconditions'] = case.preconditions
        case.preconditions = case_update.preconditions
        content_changed = True
    
    if case_update.steps is not None and case_update.steps != case.steps:
        old_values['steps'] = case.steps
        case.steps = case_update.steps
        content_changed = True
    
    if case_update.expected_results is not None and case_update.expected_results != case.expected_results:
        old_values['expected_results'] = case.expected_results
        case.expected_results = case_update.expected_results
        content_changed = True
    
    # Type changes should create a new version (changes test methodology)
    if case_update.type is not None and case_update.type != case.type:
        old_values['type'] = case.type
        case.type = case_update.type
        content_changed = True
    
    # Metadata fields (no version created)
    if case_update.priority is not None:
        case.priority = case_update.priority
    
    if case_update.suite_id is not None:
        case.suite_id = case_update.suite_id
    
    if case_update.reference_ids is not None:
        case.reference_ids = case_update.reference_ids
    
    if case_update.attachments is not None:
        case.attachments = case_update.attachments
    
    if case_update.milestone_id is not None:
        case.milestone_id = case_update.milestone_id
    
    if case_update.assigned_to_id is not None:
        case.assigned_to_id = case_update.assigned_to_id
    
    # Create activity log for the update
    activity = TestCaseActivity(
        test_case_id=case_id,
        activity_type=ActivityType.UPDATED,
        description=f"Test case updated{' - Version created' if content_changed else ''}",
        activity_data={
            "action": "manual_update",
            "content_changed": content_changed,
            "updated_fields": list(old_values.keys()) if old_values else [],
            "old_values": old_values if old_values else {}
        },
        created_by_id=current_user.id
    )
    db.add(activity)
    
    db.commit()
    
    # Create new version if content changed
    if content_changed:
        latest_version = db.query(TestCaseVersion).filter(
            TestCaseVersion.case_id == case_id
        ).order_by(TestCaseVersion.version_number.desc()).first()
        
        new_version_number = (latest_version.version_number + 1) if latest_version else 1
        
        version = TestCaseVersion(
            case_id=case.id,
            version_number=new_version_number,
            title=case.title,
            description=case.description,
            preconditions=case.preconditions,
            steps=case.steps,
            expected_results=case.expected_results,
            priority=case.priority,
            type=case.type,
            created_by_id=current_user.id
        )
        db.add(version)
        
        # Update activity to include version number
        activity.activity_data["version_created"] = new_version_number
        
        db.commit()
    
    db.refresh(case)
    
    # Track test case update
    analytics.track_event(
        "test_case_updated",
        user_id=current_user.id,
        properties={
            "test_case_id": case.id,
            "case_number": case.case_number,
            "project_id": case.project_id,
            "content_changed": content_changed,
            "version_created": content_changed,
            "fields_updated": list(old_values.keys()) if old_values else [],
            "activity_logged": True
        }
    )
    
    return case

@router.delete("/{case_id}")
async def delete_test_case(
    case_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete test case"""
    case = db.query(TestCase).filter(TestCase.id == case_id).first()
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test case not found"
        )
    
    check_case_access(case, current_user, Permission.DELETE)
    
    try:
        # Delete dependent records first to avoid foreign key constraint violations
        # Must be done in the correct order due to foreign key relationships
        
        # Import dependent models
        from app.database.models import TestExecution, TestPlanItem
        
        # Delete test plan items that reference this test case
        db.query(TestPlanItem).filter(TestPlanItem.test_case_id == case_id).delete(synchronize_session=False)
        
        # Delete test executions that reference this test case  
        db.query(TestExecution).filter(TestExecution.case_id == case_id).delete(synchronize_session=False)
        
        # Delete test case versions that reference this test case
        db.query(TestCaseVersion).filter(TestCaseVersion.case_id == case_id).delete(synchronize_session=False)
        
        # Now delete the test case itself
        db.delete(case)
        
        # Commit all changes
        db.commit()
        
        # Track test case deletion
        analytics.track_event(
            "test_case_deleted",
            user_id=current_user.id,
            properties={
                "test_case_id": case_id,
                "project_id": case.project_id,
                "suite_id": case.suite_id
            }
        )
        
        return {"message": "Test case deleted successfully"}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete test case: {str(e)}"
        )

@router.get("/{case_id}/versions")
async def get_test_case_versions(
    case_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all versions of a test case"""
    case = db.query(TestCase).filter(TestCase.id == case_id).first()
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test case not found"
        )
    
    check_case_access(case, current_user, Permission.READ)
    
    versions = db.query(TestCaseVersion).filter(
        TestCaseVersion.case_id == case_id
    ).order_by(TestCaseVersion.version_number.desc()).all()
    
    return versions

@router.get("/{case_id}/versions/{version_number}")
async def get_test_case_version(
    case_id: int,
    version_number: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get specific version of a test case"""
    case = db.query(TestCase).filter(TestCase.id == case_id).first()
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test case not found"
        )
    
    check_case_access(case, current_user, Permission.READ)
    
    version = db.query(TestCaseVersion).filter(
        TestCaseVersion.case_id == case_id,
        TestCaseVersion.version_number == version_number
    ).first()
    
    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test case version not found"
        )
    
    return version

@router.post("/{case_id}/restore-version/{version_number}")
async def restore_test_case_version(
    case_id: int,
    version_number: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Restore a test case to a previous version"""
    import logging
    logger = logging.getLogger(__name__)
    
    # Get the test case
    case = db.query(TestCase).filter(TestCase.id == case_id).first()
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test case not found"
        )
    
    check_case_access(case, current_user, Permission.WRITE)
    
    # Get the version to restore
    version = db.query(TestCaseVersion).filter(
        TestCaseVersion.case_id == case_id,
        TestCaseVersion.version_number == version_number
    ).first()
    
    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test case version not found"
        )
    
    logger.info(f"🔄 RESTORING TEST CASE #{case.case_number} TO VERSION {version_number}")
    logger.info(f"   Current title: {case.title}")
    logger.info(f"   Version title: {version.title}")
    
    # Create a new version with current state before restoring
    max_version = db.query(TestCaseVersion).filter(
        TestCaseVersion.case_id == case_id
    ).order_by(TestCaseVersion.version_number.desc()).first()
    
    next_version = (max_version.version_number + 1) if max_version else 1
    
    backup_version = TestCaseVersion(
        case_id=case_id,
        version_number=next_version,
        title=case.title,
        description=case.description,
        preconditions=case.preconditions,
        steps=case.steps,
        expected_results=case.expected_results,
        priority=case.priority,
        type=case.type,
        created_by_id=current_user.id
    )
    db.add(backup_version)
    logger.info(f"   ✅ Created backup version {next_version}")
    
    # Restore the test case to the selected version
    case.title = version.title
    case.description = version.description
    case.preconditions = version.preconditions
    case.steps = version.steps
    case.expected_results = version.expected_results
    case.priority = version.priority
    case.type = version.type
    
    # Log activity
    activity = TestCaseActivity(
        test_case_id=case_id,
        activity_type=ActivityType.UPDATED,
        description=f"Restored to version {version_number}",
        activity_data={
            "restored_version": version_number,
            "backup_version": next_version
        },
        created_by_id=current_user.id
    )
    db.add(activity)
    
    db.commit()
    db.refresh(case)
    
    logger.info(f"✅ Successfully restored test case to version {version_number}")
    
    return {
        "success": True,
        "message": f"Test case restored to version {version_number}",
        "restored_version": version_number,
        "backup_version": next_version,
        "test_case_id": case.id
    }
