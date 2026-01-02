
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional
from pydantic import BaseModel
from app.database import models
from app.database.session import get_db
from typing import Optional, List
from app.database.models import TestCase, SectionTestCase

router = APIRouter()
class SectionCreate(BaseModel):
    name: str
    suite_id: int
    parent_id: Optional[int] = None
    display_order: Optional[int] = 0
    is_copy: Optional[bool] = False
    copyof_id: Optional[int] = None
    depth: Optional[int] = 0
    description: Optional[str] = None


class SectionUpdate(BaseModel):
    name: Optional[str] = None
    suite_id: Optional[int] = None
    parent_id: Optional[int] = None
    display_order: Optional[int] = None
    is_copy: Optional[bool] = None
    copyof_id: Optional[int] = None
    depth: Optional[int] = None
    description: Optional[str] = None


class SectionResponse(BaseModel):
    id: int
    name: str
    suite_id: int
    parent_id: Optional[int] = None
    display_order: int = 0
    is_copy: bool = False
    copyof_id: Optional[int] = None
    depth: int = 0
    description: Optional[str] = None

    class Config:
        from_attributes = True 

class TestCaseResponse(BaseModel):
    id: int
    case_number: int
    title: str
    description: Optional[str] = None
    suite_id: int
    project_id: int
    class Config:
        from_attributes = True

@router.post("/", response_model=dict, status_code=status.HTTP_201_CREATED)
def create_section(section: SectionCreate, db: Session = Depends(get_db)):
    new_section = models.Section(
        name=section.name,
        suite_id=section.suite_id,
        parent_id=section.parent_id if section.parent_id not in (0, None) else None,
        display_order=section.display_order if section.display_order is not None else 0,
        is_copy=section.is_copy or False,
        copyof_id=section.copyof_id if section.copyof_id not in (0, None) else None,
        depth=section.depth if section.depth is not None else 0,
        description=section.description,
    )

    db.add(new_section)
    db.commit()
    db.refresh(new_section)

    return {"id": new_section.id, "name": new_section.name}



@router.patch("/{section_id}", response_model=SectionResponse)
def update_section(
    section_id: int,
    section_update: SectionUpdate,
    db: Session = Depends(get_db),
):
    section = db.query(models.Section).filter(models.Section.id == section_id).first()
    if not section:
        raise HTTPException(status_code=404, detail="Section not found")

    update_data = section_update.dict(exclude_unset=True)

    for field in ["parent_id", "copyof_id"]:
        if field in update_data and update_data[field] == 0:
            update_data[field] = None

    for key, value in update_data.items():
        setattr(section, key, value)

    db.commit()
    db.refresh(section)
    return section

@router.delete("/{section_id}", status_code=status.HTTP_200_OK)
def delete_section(section_id: int, db: Session = Depends(get_db)):
    section = db.query(models.Section).filter(models.Section.id == section_id).first()
    if not section:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Section not found")

    try:
        db.query(models.Section).filter(models.Section.parent_id == section_id).update(
            {"parent_id": None}, synchronize_session=False
        )

        db.query(models.Section).filter(models.Section.copyof_id == section_id).update(
            {"copyof_id": None}, synchronize_session=False
        )

        db.delete(section)
        db.commit()

        return {"message": f"Section {section_id} deleted successfully."}

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete section: {str(e)}",
        )



@router.get("/by-suite/{suite_id}", response_model=List[SectionResponse])
def get_sections_by_suite(suite_id: int, db: Session = Depends(get_db)):
    all_sections = db.query(models.Section).filter(models.Section.suite_id == suite_id).order_by(models.Section.display_order, models.Section.id).all()

    # Build a dictionary for quick lookup
    section_dicts = {s.id: SectionResponse.from_orm(s).dict() for s in all_sections}

    # Build the tree structure
    tree = []
    for s_id, s_dict in section_dicts.items():
        parent_id = s_dict.get('parent_id')
        if parent_id:
            parent = section_dicts.get(parent_id)
            if parent:
                if 'subsections' not in parent:
                    parent['subsections'] = []
                parent['subsections'].append(s_dict)
        else:
            tree.append(s_dict)

    # Recursively sort subsections by display_order and id
    def sort_subsections(node):
        if 'subsections' in node:
            node['subsections'].sort(key=lambda x: (x['display_order'], x['id']))
            for sub in node['subsections']:
                sort_subsections(sub)

    # Sort top-level and their children
    tree.sort(key=lambda x: (x['display_order'], x['id']))
    for t in tree:
        sort_subsections(t)

    return tree


@router.get("/{section_id}/subsections", response_model=List[SectionResponse])
def get_subsections(section_id: int, db: Session = Depends(get_db)):
    subsections = db.query(models.Section).filter(models.Section.parent_id == section_id).order_by(models.Section.display_order, models.Section.id).all()
    return [SectionResponse.from_orm(s) for s in subsections]



@router.get("/{section_id}/test-cases", response_model=List[TestCaseResponse])
def get_test_cases_for_section(section_id: int, db: Session = Depends(get_db)):
    section = db.query(models.Section).filter(models.Section.id == section_id).first()
    if not section:
        raise HTTPException(status_code=404, detail="Section not found")
    return [TestCaseResponse.from_orm(tc) for tc in section.test_cases]


class AddTestCaseToSectionRequest(BaseModel):
    test_case_id: int

class RemoveTestCaseFromSectionRequest(BaseModel):
    test_case_id: int


@router.post("/{section_id}/add-test-case", status_code=status.HTTP_200_OK)
def add_test_case_to_section(section_id: int, req: AddTestCaseToSectionRequest, db: Session = Depends(get_db)):
    section = db.query(models.Section).filter(models.Section.id == section_id).first()
    if not section:
        raise HTTPException(status_code=404, detail="Section not found")

    test_case = db.query(models.TestCase).filter(models.TestCase.id == req.test_case_id).first()
    if not test_case:
        raise HTTPException(status_code=404, detail="Test case not found")

    # Prevent duplicate association
    if test_case in section.test_cases:
        return {"message": "Test case already associated with section."}

    section.test_cases.append(test_case)
    db.commit()

    return {"message": "Test case associated with section successfully."}


@router.post("/{section_id}/remove-test-case", status_code=status.HTTP_200_OK)
def remove_test_case_from_section(section_id: int, req: RemoveTestCaseFromSectionRequest, db: Session = Depends(get_db)):
    section = db.query(models.Section).filter(models.Section.id == section_id).first()
    if not section:
        raise HTTPException(status_code=404, detail="Section not found")

    test_case = db.query(models.TestCase).filter(models.TestCase.id == req.test_case_id).first()
    if not test_case:
        raise HTTPException(status_code=404, detail="Test case not found")

    if test_case in section.test_cases:
        section.test_cases.remove(test_case)
        db.commit()
        return {"message": "Test case removed from section successfully."}

    return {"message": "Test case was not associated with section."}


# Move API Models
class MoveTestCaseRequest(BaseModel):
    destination_section_id: int
    position: Optional[str] = "end"  # "start", "end", or number

class MoveSectionRequest(BaseModel):
    destination_parent_id: Optional[int] = None  # None means root level
    destination_suite_id: Optional[int] = None  # For moving to different suite
    position: Optional[str] = "end"  # "start", "end", or number

# Get all sections in tree format for move destination picker
@router.get("/tree/{suite_id}", response_model=List[dict])
def get_sections_tree_for_move(suite_id: int, db: Session = Depends(get_db)):
    """Get all sections in tree format for move destination picker"""
    all_sections = db.query(models.Section).filter(
        models.Section.suite_id == suite_id
    ).order_by(models.Section.display_order, models.Section.id).all()

    # Build the tree structure with additional info for move operations
    section_dicts = {}
    for section in all_sections:
        section_dicts[section.id] = {
            "id": section.id,
            "name": section.name,
            "parent_id": section.parent_id,
            "display_order": section.display_order,
            "depth": section.depth,
            "suite_id": section.suite_id,
            "subsections": []
        }

    # Build tree
    tree = []
    for s_id, s_dict in section_dicts.items():
        parent_id = s_dict.get('parent_id')
        if parent_id and parent_id in section_dicts:
            section_dicts[parent_id]['subsections'].append(s_dict)
        else:
            tree.append(s_dict)

    # Sort recursively
    def sort_tree(nodes):
        nodes.sort(key=lambda x: (x['display_order'], x['id']))
        for node in nodes:
            if node['subsections']:
                sort_tree(node['subsections'])

    sort_tree(tree)
    return tree

# Move test case to different section
@router.post("/test-cases/{test_case_id}/move", status_code=status.HTTP_200_OK)
def move_test_case_to_section(
    test_case_id: int, 
    move_request: MoveTestCaseRequest, 
    db: Session = Depends(get_db)
):
    """Move a test case to a different section"""
    # Verify test case exists
    test_case = db.query(models.TestCase).filter(models.TestCase.id == test_case_id).first()
    if not test_case:
        raise HTTPException(status_code=404, detail="Test case not found")
    
    # Verify destination section exists
    dest_section = db.query(models.Section).filter(
        models.Section.id == move_request.destination_section_id
    ).first()
    if not dest_section:
        raise HTTPException(status_code=404, detail="Destination section not found")
    
    # Find current section associations
    current_associations = db.query(models.SectionTestCase).filter(
        models.SectionTestCase.test_case_id == test_case_id
    ).all()
    
    # Remove from current sections
    for assoc in current_associations:
        db.delete(assoc)
    
    # Create new association (no order_index needed - we'll use test_case_id for natural ordering)
    new_association = models.SectionTestCase(
        section_id=move_request.destination_section_id,
        test_case_id=test_case_id
    )
    db.add(new_association)
    
    db.commit()
    
    return {"message": "Test case moved successfully"}

# Move section to different parent or position
@router.post("/{section_id}/move", status_code=status.HTTP_200_OK)
def move_section(
    section_id: int, 
    move_request: MoveSectionRequest, 
    db: Session = Depends(get_db)
):
    """Move a section to different parent or position"""
    # Verify section exists
    section = db.query(models.Section).filter(models.Section.id == section_id).first()
    if not section:
        raise HTTPException(status_code=404, detail="Section not found")
    
    # Verify destination parent exists (if specified)
    if move_request.destination_parent_id:
        dest_parent = db.query(models.Section).filter(
            models.Section.id == move_request.destination_parent_id
        ).first()
        if not dest_parent:
            raise HTTPException(status_code=404, detail="Destination parent section not found")
        
        # Prevent moving to own subsection (would create cycle)
        current = dest_parent
        while current.parent_id:
            if current.parent_id == section_id:
                raise HTTPException(status_code=400, detail="Cannot move section to its own subsection")
            current = db.query(models.Section).filter(models.Section.id == current.parent_id).first()
            if not current:
                break
    
    # Determine target suite
    target_suite_id = move_request.destination_suite_id or section.suite_id
    
    # Get next display_order in destination (simple approach - use max + 1)
    max_order = db.query(models.Section.display_order).filter(
        models.Section.suite_id == target_suite_id,
        models.Section.parent_id == move_request.destination_parent_id
    ).order_by(models.Section.display_order.desc()).first()
    
    next_order = 0 if not max_order or max_order[0] is None else max_order[0] + 1
    
    # Update section
    section.parent_id = move_request.destination_parent_id
    section.suite_id = target_suite_id
    section.display_order = next_order
    
    # Update depth based on new parent
    if move_request.destination_parent_id:
        parent_depth = db.query(models.Section.depth).filter(
            models.Section.id == move_request.destination_parent_id
        ).scalar()
        section.depth = (parent_depth or 0) + 1
    else:
        section.depth = 0
    
    db.commit()
    
    return {"message": "Section moved successfully"}
