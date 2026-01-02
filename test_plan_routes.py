from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, selectinload
from app.database.session import get_db
from app.database.models import User, Project, TestPlan, TestPlanItem, TestCase, TestSuite
from app.schemas.test_management import (
    TestPlanCreate, TestPlanUpdate, TestPlan as TestPlanSchema,
    TestPlanItemCreate, TestPlanItem as TestPlanItemSchema
)
from app.auth.deps import get_current_active_user, require_permission
from app.core.permissions import Permission

router = APIRouter()

# Test Plans
@router.post("/", response_model=TestPlanSchema)
async def create_test_plan(
    plan_data: TestPlanCreate,
    current_user: User = Depends(require_permission(Permission.CREATE)),
    db: Session = Depends(get_db)
):
    """Create a new test plan"""
    
    # Validate project exists
    project = db.query(Project).filter(Project.id == plan_data.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    test_plan = TestPlan(
        name=plan_data.name,
        description=plan_data.description,
        project_id=plan_data.project_id,
        milestone_id=plan_data.milestone_id,
        is_active=plan_data.is_active if plan_data.is_active is not None else True,
        created_by_id=current_user.id
    )
    
    db.add(test_plan)
    db.commit()
    db.refresh(test_plan)
    
    return test_plan

@router.get("/", response_model=List[TestPlanSchema])
async def get_test_plans(
    project_id: int = Query(None, description="Filter by project ID"),
    is_active: bool = Query(None, description="Filter by active status"),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get test plans with optional filtering"""
    
    query = db.query(TestPlan)
    
    if project_id:
        query = query.filter(TestPlan.project_id == project_id)
    
    if is_active is not None:
        query = query.filter(TestPlan.is_active == is_active)
    
    # Use selectinload to efficiently load related data
    plans = query.options(
        selectinload(TestPlan.created_by),
        selectinload(TestPlan.project)
    ).order_by(TestPlan.created_at.desc()).offset(skip).limit(limit).all()
    return plans

@router.get("/{plan_id}", response_model=TestPlanSchema)
async def get_test_plan(
    plan_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get a specific test plan"""
    
    plan = db.query(TestPlan).filter(TestPlan.id == plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Test plan not found")
    
    return plan

@router.put("/{plan_id}", response_model=TestPlanSchema)
async def update_test_plan(
    plan_id: int,
    plan_data: TestPlanUpdate,
    current_user: User = Depends(require_permission(Permission.UPDATE)),
    db: Session = Depends(get_db)
):
    """Update an existing test plan"""
    
    plan = db.query(TestPlan).filter(TestPlan.id == plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Test plan not found")
    
    # Update fields
    for key, value in plan_data.dict(exclude_unset=True).items():
        setattr(plan, key, value)
    
    plan.updated_at = db.func.now()
    
    db.commit()
    db.refresh(plan)
    
    return plan

@router.delete("/{plan_id}")
async def delete_test_plan(
    plan_id: int,
    current_user: User = Depends(require_permission(Permission.DELETE)),
    db: Session = Depends(get_db)
):
    """Delete a test plan"""
    
    plan = db.query(TestPlan).filter(TestPlan.id == plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Test plan not found")
    
    # Delete all plan items first
    db.query(TestPlanItem).filter(TestPlanItem.test_plan_id == plan_id).delete()
    
    # Delete the plan
    db.delete(plan)
    db.commit()
    
    return {"message": "Test plan deleted successfully"}

# Test Plan Items
@router.post("/{plan_id}/items", response_model=TestPlanItemSchema)
async def add_test_case_to_plan(
    plan_id: int,
    item_data: TestPlanItemCreate,
    current_user: User = Depends(require_permission(Permission.CREATE)),
    db: Session = Depends(get_db)
):
    """Add a test case to a test plan"""
    
    # Validate plan exists
    plan = db.query(TestPlan).filter(TestPlan.id == plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Test plan not found")
    
    # Validate test case exists
    test_case = db.query(TestCase).filter(TestCase.id == item_data.case_id).first()
    if not test_case:
        raise HTTPException(status_code=404, detail="Test case not found")
    
    # Check if test case is already in the plan
    existing_item = db.query(TestPlanItem).filter(
        TestPlanItem.plan_id == plan_id,
        TestPlanItem.case_id == item_data.case_id
    ).first()
    
    if existing_item:
        raise HTTPException(status_code=400, detail="Test case is already in this plan")
    
    # Get the next order value
    max_order = db.query(db.func.max(TestPlanItem.order)).filter(
        TestPlanItem.plan_id == plan_id
    ).scalar() or 0
    
    plan_item = TestPlanItem(
        plan_id=plan_id,
        case_id=item_data.case_id,
        order=max_order + 1,
        is_required=item_data.is_required,
        notes=item_data.notes,
        created_by_id=current_user.id
    )
    
    db.add(plan_item)
    db.commit()
    db.refresh(plan_item)
    
    return plan_item

@router.get("/{plan_id}/items", response_model=List[TestPlanItemSchema])
async def get_plan_items(
    plan_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all test cases in a test plan"""
    
    # Validate plan exists
    plan = db.query(TestPlan).filter(TestPlan.id == plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Test plan not found")
    
    items = db.query(TestPlanItem).filter(
        TestPlanItem.plan_id == plan_id
    ).order_by(TestPlanItem.order).all()
    
    return items

@router.put("/{plan_id}/items/{item_id}", response_model=TestPlanItemSchema)
async def update_plan_item(
    plan_id: int,
    item_id: int,
    item_data: Dict[str, Any],
    current_user: User = Depends(require_permission(Permission.UPDATE)),
    db: Session = Depends(get_db)
):
    """Update a test plan item"""
    
    item = db.query(TestPlanItem).filter(
        TestPlanItem.id == item_id,
        TestPlanItem.plan_id == plan_id
    ).first()
    
    if not item:
        raise HTTPException(status_code=404, detail="Plan item not found")
    
    # Update allowed fields
    allowed_fields = ['order', 'is_required', 'notes']
    for key, value in item_data.items():
        if key in allowed_fields:
            setattr(item, key, value)
    
    item.updated_at = db.func.now()
    
    db.commit()
    db.refresh(item)
    
    return item

@router.delete("/{plan_id}/items/{item_id}")
async def remove_test_case_from_plan(
    plan_id: int,
    item_id: int,
    current_user: User = Depends(require_permission(Permission.DELETE)),
    db: Session = Depends(get_db)
):
    """Remove a test case from a test plan"""
    
    item = db.query(TestPlanItem).filter(
        TestPlanItem.id == item_id,
        TestPlanItem.plan_id == plan_id
    ).first()
    
    if not item:
        raise HTTPException(status_code=404, detail="Plan item not found")
    
    db.delete(item)
    db.commit()
    
    return {"message": "Test case removed from plan successfully"}

@router.post("/{plan_id}/bulk-add")
async def bulk_add_test_cases_to_plan(
    plan_id: int,
    case_ids: List[int],
    current_user: User = Depends(require_permission(Permission.CREATE)),
    db: Session = Depends(get_db)
):
    """Add multiple test cases to a test plan"""
    
    # Validate plan exists
    plan = db.query(TestPlan).filter(TestPlan.id == plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Test plan not found")
    
    # Get existing items to avoid duplicates
    existing_case_ids = set(
        item.case_id for item in db.query(TestPlanItem).filter(
            TestPlanItem.plan_id == plan_id
        ).all()
    )
    
    # Filter out cases that are already in the plan
    new_case_ids = [case_id for case_id in case_ids if case_id not in existing_case_ids]
    
    if not new_case_ids:
        return {"message": "All test cases are already in the plan", "added_count": 0}
    
    # Validate all test cases exist
    existing_cases = db.query(TestCase).filter(TestCase.id.in_(new_case_ids)).all()
    found_case_ids = {case.id for case in existing_cases}
    
    missing_case_ids = set(new_case_ids) - found_case_ids
    if missing_case_ids:
        raise HTTPException(
            status_code=404, 
            detail=f"Test cases not found: {list(missing_case_ids)}"
        )
    
    # Get the next order value
    max_order = db.query(db.func.max(TestPlanItem.order)).filter(
        TestPlanItem.plan_id == plan_id
    ).scalar() or 0
    
    # Create plan items
    plan_items = []
    for i, case_id in enumerate(new_case_ids):
        plan_item = TestPlanItem(
            plan_id=plan_id,
            case_id=case_id,
            order=max_order + i + 1,
            is_required=True,
            created_by_id=current_user.id
        )
        plan_items.append(plan_item)
    
    db.add_all(plan_items)
    db.commit()
    
    return {
        "message": f"Added {len(new_case_ids)} test cases to plan",
        "added_count": len(new_case_ids),
        "skipped_count": len(case_ids) - len(new_case_ids)
    }

@router.post("/{plan_id}/reorder")
async def reorder_plan_items(
    plan_id: int,
    item_orders: List[Dict[str, int]],  # [{"item_id": 1, "order": 1}, ...]
    current_user: User = Depends(require_permission(Permission.UPDATE)),
    db: Session = Depends(get_db)
):
    """Reorder test plan items"""
    
    # Validate plan exists
    plan = db.query(TestPlan).filter(TestPlan.id == plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Test plan not found")
    
    # Update orders
    for order_data in item_orders:
        item_id = order_data.get("item_id")
        new_order = order_data.get("order")
        
        if item_id and new_order is not None:
            item = db.query(TestPlanItem).filter(
                TestPlanItem.id == item_id,
                TestPlanItem.plan_id == plan_id
            ).first()
            
            if item:
                item.order = new_order
                item.updated_at = db.func.now()
    
    db.commit()
    
    return {"message": "Plan items reordered successfully"}