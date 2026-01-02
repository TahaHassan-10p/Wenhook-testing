"""
Summary API Routes - Optimized endpoints for frontend pages
Returns minimal data needed for UI display with single-query performance
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, case as sql_case, distinct
from typing import List, Optional
from app.database.session import get_db
from app.database.models import (
    User, Project, TestSuite, TestCase, TestRun, TestExecution,
    Milestone, TestPlan, Section
)
from app.auth.deps import get_current_active_user
from pydantic import BaseModel

router = APIRouter()

# Response models for summary data
class ProjectSummary(BaseModel):
    id: int
    name: str
    test_case_count: int
    test_suite_count: int
    test_run_count: int
    milestone_count: int
    
    class Config:
        from_attributes = True

class TestSuiteSummary(BaseModel):
    id: int
    name: str
    description: Optional[str]
    test_case_count: int
    section_count: int
    
    class Config:
        from_attributes = True

class TestRunSummary(BaseModel):
    id: int
    name: str
    status: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    
    class Config:
        from_attributes = True

class MilestoneSummary(BaseModel):
    id: int
    name: str
    status: str
    test_case_count: int
    test_run_count: int
    
    class Config:
        from_attributes = True

class ProjectDetailsSummary(BaseModel):
    """Complete summary for project details page"""
    project: ProjectSummary
    test_suites: List[TestSuiteSummary]
    test_runs: List[TestRunSummary]
    milestones: List[MilestoneSummary]
    test_plans_count: int
    
    class Config:
        from_attributes = True


@router.get("/project/{project_id}/details", response_model=ProjectDetailsSummary)
async def get_project_details_summary(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get complete project details summary in ONE optimized query batch.
    Returns all data needed for the project details page.
    
    Replaces multiple slow endpoints:
    - /api/v1/test-suites/by-project/{id}
    - /api/v1/test-runs/by-project/{id}
    - /api/v1/milestones/?project_id={id}
    - /api/v1/test-plans/?project_id={id}
    """
    
    # Verify project exists and user has access
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Build project summary with counts
    test_case_count = db.query(func.count(TestCase.id))\
        .join(TestSuite, TestCase.suite_id == TestSuite.id)\
        .filter(TestSuite.project_id == project_id)\
        .scalar() or 0
    
    test_suite_count = db.query(func.count(TestSuite.id))\
        .filter(TestSuite.project_id == project_id)\
        .scalar() or 0
    
    test_run_count = db.query(func.count(TestRun.id))\
        .filter(TestRun.project_id == project_id)\
        .scalar() or 0
    
    milestone_count = db.query(func.count(Milestone.id))\
        .filter(Milestone.project_id == project_id)\
        .scalar() or 0
    
    test_plans_count = db.query(func.count(TestPlan.id))\
        .filter(TestPlan.project_id == project_id)\
        .scalar() or 0
    
    project_summary = ProjectSummary(
        id=project.id,
        name=project.name,
        test_case_count=test_case_count,
        test_suite_count=test_suite_count,
        test_run_count=test_run_count,
        milestone_count=milestone_count
    )
    
    # Get test suites with counts (optimized single query)
    suite_data = db.query(
        TestSuite.id,
        TestSuite.name,
        TestSuite.description,
        func.count(distinct(TestCase.id)).label('test_case_count'),
        func.count(distinct(Section.id)).label('section_count')
    ).outerjoin(TestCase, TestSuite.id == TestCase.suite_id)\
     .outerjoin(Section, TestSuite.id == Section.suite_id)\
     .filter(TestSuite.project_id == project_id)\
     .group_by(TestSuite.id, TestSuite.name, TestSuite.description)\
     .all()
    
    test_suites = [
        TestSuiteSummary(
            id=row.id,
            name=row.name,
            description=row.description,
            test_case_count=row.test_case_count,
            section_count=row.section_count
        )
        for row in suite_data
    ]
    
    # Get test runs with execution stats (optimized single query)
    run_data = db.query(
        TestRun.id,
        TestRun.name,
        TestRun.status,
        func.count(TestExecution.id).label('total_cases'),
        func.sum(sql_case((TestExecution.status == 'passed', 1), else_=0)).label('passed_cases'),
        func.sum(sql_case((TestExecution.status == 'failed', 1), else_=0)).label('failed_cases')
    ).outerjoin(TestExecution, TestRun.id == TestExecution.run_id)\
     .filter(TestRun.project_id == project_id)\
     .group_by(TestRun.id, TestRun.name, TestRun.status)\
     .order_by(TestRun.id.desc())\
     .limit(50)\
     .all()
    
    test_runs = [
        TestRunSummary(
            id=row.id,
            name=row.name,
            status=row.status,
            total_cases=row.total_cases or 0,
            passed_cases=row.passed_cases or 0,
            failed_cases=row.failed_cases or 0
        )
        for row in run_data
    ]
    
    # Get milestones with counts (optimized single query)
    milestone_data = db.query(
        Milestone.id,
        Milestone.name,
        Milestone.status,
        func.count(distinct(TestCase.id)).label('test_case_count'),
        func.count(distinct(TestRun.id)).label('test_run_count')
    ).outerjoin(TestCase, Milestone.id == TestCase.milestone_id)\
     .outerjoin(TestRun, Milestone.id == TestRun.milestone_id)\
     .filter(Milestone.project_id == project_id)\
     .group_by(Milestone.id, Milestone.name, Milestone.status)\
     .all()
    
    milestones = [
        MilestoneSummary(
            id=row.id,
            name=row.name,
            status=row.status,
            test_case_count=row.test_case_count,
            test_run_count=row.test_run_count
        )
        for row in milestone_data
    ]
    
    return ProjectDetailsSummary(
        project=project_summary,
        test_suites=test_suites,
        test_runs=test_runs,
        milestones=milestones,
        test_plans_count=test_plans_count
    )


@router.get("/projects/list", response_model=List[ProjectSummary])
async def get_projects_list_summary(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get lightweight project list with counts.
    Much faster than /api/v1/projects/with-stats
    """
    from app.core.permissions import UserRole
    from sqlalchemy import or_
    from app.database.models import ProjectMember
    
    # Get projects user has access to
    if current_user.role == UserRole.ADMIN:
        base_query = db.query(Project.id, Project.name)
    else:
        base_query = db.query(Project.id, Project.name)\
            .outerjoin(ProjectMember)\
            .filter(
                or_(
                    Project.created_by_id == current_user.id,
                    ProjectMember.user_id == current_user.id
                )
            ).distinct()
    
    projects = base_query.offset(skip).limit(limit).all()
    project_ids = [p.id for p in projects]
    
    if not project_ids:
        return []
    
    # Get counts for all projects in batch queries
    test_case_counts = dict(
        db.query(TestSuite.project_id, func.count(TestCase.id))
        .join(TestCase, TestSuite.id == TestCase.suite_id)
        .filter(TestSuite.project_id.in_(project_ids))
        .group_by(TestSuite.project_id)
        .all()
    )
    
    test_suite_counts = dict(
        db.query(TestSuite.project_id, func.count(TestSuite.id))
        .filter(TestSuite.project_id.in_(project_ids))
        .group_by(TestSuite.project_id)
        .all()
    )
    
    test_run_counts = dict(
        db.query(TestRun.project_id, func.count(TestRun.id))
        .filter(TestRun.project_id.in_(project_ids))
        .group_by(TestRun.project_id)
        .all()
    )
    
    milestone_counts = dict(
        db.query(Milestone.project_id, func.count(Milestone.id))
        .filter(Milestone.project_id.in_(project_ids))
        .group_by(Milestone.project_id)
        .all()
    )
    
    return [
        ProjectSummary(
            id=p.id,
            name=p.name,
            test_case_count=test_case_counts.get(p.id, 0),
            test_suite_count=test_suite_counts.get(p.id, 0),
            test_run_count=test_run_counts.get(p.id, 0),
            milestone_count=milestone_counts.get(p.id, 0)
        )
        for p in projects
    ]
