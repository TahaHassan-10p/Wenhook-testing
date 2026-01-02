from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.database.session import get_db
from app.database.models import User, RefreshToken, Project, ProjectMember, UserRole, Milestone, TestPlan, TestSuite, TestCase, TestRun, TestCaseVersion
from app.schemas.user import User as UserSchema, UserCreate, UserUpdate, UserChangePassword
from app.auth.deps import get_current_active_user, require_admin, require_permission
from app.auth.security import verify_password, get_password_hash
from app.core.permissions import Permission

router = APIRouter()

class ThemePreferenceUpdate(BaseModel):
    theme_preference: str

@router.get("/me", response_model=UserSchema)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user profile"""
    return current_user

@router.put("/me", response_model=UserSchema)
async def update_users_me(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update current user profile"""
    # Users can only update their own profile (limited fields)
    if user_update.email is not None:
        current_user.email = user_update.email
    if user_update.name is not None:
        current_user.name = user_update.name
    if user_update.avatar_url is not None:
        current_user.avatar_url = user_update.avatar_url
    
    db.commit()
    db.refresh(current_user)
    return current_user

@router.put("/me/theme")
async def update_user_theme(
    theme_update: ThemePreferenceUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update current user theme preference"""
    if theme_update.theme_preference not in ['light', 'dark']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Theme must be 'light' or 'dark'"
        )
    
    current_user.theme_preference = theme_update.theme_preference
    db.commit()
    db.refresh(current_user)
    
    return {"message": "Theme preference updated successfully", "theme_preference": current_user.theme_preference}

@router.post("/me/change-password")
async def change_password(
    password_data: UserChangePassword,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Change current user password"""
    # Verify current password
    if not verify_password(password_data.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )
    
    # Update password
    current_user.password_hash = get_password_hash(password_data.new_password)
    db.commit()
    
    return {"message": "Password changed successfully"}

@router.get("/search", response_model=List[UserSchema])
async def search_users(
    q: str = "",
    limit: int = 10,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Search users by email or name - for autocomplete"""
    if not q or len(q) < 2:
        return []
    
    # Search by email or name (case-insensitive)
    search_pattern = f"%{q.lower()}%"
    users = db.query(User).filter(
        (User.email.ilike(search_pattern)) | (User.name.ilike(search_pattern)),
        User.is_active == True
    ).limit(limit).all()
    
    return users

@router.get("/", response_model=List[UserSchema])
async def read_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS)),
    db: Session = Depends(get_db)
):
    """Get all users (admin only)"""
    users = db.query(User).offset(skip).limit(limit).all()
    return users

@router.post("/", response_model=UserSchema)
async def create_user(
    user_data: UserCreate,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Create new user (admin only)"""
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    try:
        hashed_password = get_password_hash(user_data.password)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password processing failed"
        )
    db_user = User(
        email=user_data.email,
        name=user_data.name,
        password_hash=hashed_password,
        role=user_data.role,
        avatar_url=user_data.avatar_url,
        is_active=user_data.is_active
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

@router.get("/{user_id}", response_model=UserSchema)
async def read_user(
    user_id: int,
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS)),
    db: Session = Depends(get_db)
):
    """Get user by ID (admin only)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user

@router.put("/{user_id}", response_model=UserSchema)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Update user (admin only) - can update email, name, role, is_active, and password"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check email uniqueness if email is being updated
    if user_update.email is not None and user_update.email != user.email:
        existing_user = db.query(User).filter(
            User.email == user_update.email,
            User.id != user_id
        ).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered by another user"
            )
        user.email = user_update.email
    
    # Update other fields
    if user_update.name is not None:
        user.name = user_update.name
    if user_update.role is not None:
        user.role = user_update.role
    if user_update.avatar_url is not None:
        user.avatar_url = user_update.avatar_url
    if user_update.is_active is not None:
        user.is_active = user_update.is_active
    
    # Update password if provided
    if user_update.password is not None:
        if len(user_update.password) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 6 characters long"
            )
        user.password_hash = get_password_hash(user_update.password)
    
    try:
        db.commit()
        db.refresh(user)
        return user
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )

@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Delete user (admin only)
    
    COMPREHENSIVE USER DELETION WITH AUTOMATIC REASSIGNMENT:
    This endpoint ensures no projects or entities are orphaned when a user is deleted.
    
    Reassignment Strategy:
    1. Projects created by user → created_by_id reassigned to first available admin
    2. ALL admins are granted owner role on these projects via ProjectMember
    3. Projects where user is sole owner → ALL admins become owners
    4. All entities (milestones, test plans, suites, cases, runs, versions) → reassigned to first admin
    
    Recursive Protection:
    - If reassigned admin is later deleted → same logic applies again
    - Ensures continuous ownership chain through all remaining admins
    - Prevents orphaned projects in any deletion scenario
    
    Safety Checks:
    - Prevents self-deletion
    - Requires at least one other active admin if user created projects or is sole owner
    - Uses processed_memberships set to avoid duplicate ownership entries
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Prevent self-deletion
    if user.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    # Find all active admins (excluding the user being deleted)
    active_admins = db.query(User).filter(
        User.role == UserRole.ADMIN,
        User.id != user_id,
        User.is_active == True
    ).all()
    
    # Get projects created by this user
    projects_created = db.query(Project).filter(Project.created_by_id == user_id).all()
    
    # Find projects where this user is the ONLY owner (to prevent orphaned projects)
    # Get all projects where user is an owner
    user_owner_memberships = db.query(ProjectMember).filter(
        ProjectMember.user_id == user_id,
        ProjectMember.role == "owner"
    ).all()
    
    projects_where_only_owner = []
    for membership in user_owner_memberships:
        # Count how many owners this project has
        owner_count = db.query(ProjectMember).filter(
            ProjectMember.project_id == membership.project_id,
            ProjectMember.role == "owner"
        ).count()
        
        # If this user is the only owner, we need to add admins
        if owner_count == 1:
            project = db.query(Project).filter(Project.id == membership.project_id).first()
            if project:
                projects_where_only_owner.append(project)
    
    # If user created projects or is sole owner of projects, we need admins
    if (projects_created or projects_where_only_owner) and not active_admins:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete user: No other active administrator found to reassign project ownership"
        )
    
    projects_reassigned = 0
    admins_granted_ownership = []
    # Track which (project_id, user_id) pairs we've already processed to avoid duplicates
    processed_memberships = set()
    
    # Handle projects created by the deleted user
    for project in projects_created:
        # Assign the first admin as the creator
        project.created_by_id = active_admins[0].id
        projects_reassigned += 1
        
        # Grant owner role to ALL admins for this project
        for admin in active_admins:
            membership_key = (project.id, admin.id)
            
            # Skip if we've already processed this membership in this deletion
            if membership_key in processed_memberships:
                continue
            
            # Check if admin is already a member
            existing_membership = db.query(ProjectMember).filter(
                ProjectMember.project_id == project.id,
                ProjectMember.user_id == admin.id
            ).first()
            
            if existing_membership:
                # Update existing membership to owner (only if not already owner)
                if existing_membership.role != "owner":
                    existing_membership.role = "owner"
            else:
                # Create new owner membership for admin
                new_membership = ProjectMember(
                    project_id=project.id,
                    user_id=admin.id,
                    role="owner",
                    added_by_id=current_user.id
                )
                db.add(new_membership)
            
            processed_memberships.add(membership_key)
            if admin.email not in admins_granted_ownership:
                admins_granted_ownership.append(admin.email)
    
    # Handle projects where user is the sole owner (but didn't create)
    for project in projects_where_only_owner:
        # Skip if already processed (project was also created by user)
        if project.created_by_id == user_id:
            continue
            
        projects_reassigned += 1
        
        # Grant owner role to ALL admins for this project
        for admin in active_admins:
            membership_key = (project.id, admin.id)
            
            # Skip if we've already processed this membership in this deletion
            if membership_key in processed_memberships:
                continue
            
            # Check if admin is already a member
            existing_membership = db.query(ProjectMember).filter(
                ProjectMember.project_id == project.id,
                ProjectMember.user_id == admin.id
            ).first()
            
            if existing_membership:
                # Update existing membership to owner (only if not already owner)
                if existing_membership.role != "owner":
                    existing_membership.role = "owner"
            else:
                # Create new owner membership for admin
                new_membership = ProjectMember(
                    project_id=project.id,
                    user_id=admin.id,
                    role="owner",
                    added_by_id=current_user.id
                )
                db.add(new_membership)
            
            processed_memberships.add(membership_key)
            if admin.email not in admins_granted_ownership:
                admins_granted_ownership.append(admin.email)
    
    # Reassign all entities created by the deleted user to the first admin
    if active_admins:
        replacement_admin_id = active_admins[0].id
        
        # Reassign milestones
        db.query(Milestone).filter(Milestone.created_by_id == user_id).update(
            {"created_by_id": replacement_admin_id}, synchronize_session=False
        )
        
        # Reassign test plans
        db.query(TestPlan).filter(TestPlan.created_by_id == user_id).update(
            {"created_by_id": replacement_admin_id}, synchronize_session=False
        )
        
        # Reassign test suites
        db.query(TestSuite).filter(TestSuite.created_by_id == user_id).update(
            {"created_by_id": replacement_admin_id}, synchronize_session=False
        )
        
        # Reassign test cases
        db.query(TestCase).filter(TestCase.created_by_id == user_id).update(
            {"created_by_id": replacement_admin_id}, synchronize_session=False
        )
        
        # Reassign test runs
        db.query(TestRun).filter(TestRun.created_by_id == user_id).update(
            {"created_by_id": replacement_admin_id}, synchronize_session=False
        )
        
        # Reassign test case versions
        db.query(TestCaseVersion).filter(TestCaseVersion.created_by_id == user_id).update(
            {"created_by_id": replacement_admin_id}, synchronize_session=False
        )
    
    # Flush changes to ensure all reassignments are committed before deleting user
    db.flush()
    
    # Delete all project memberships for the user being deleted
    db.query(ProjectMember).filter(ProjectMember.user_id == user_id).delete()
    
    # Delete user's refresh tokens to avoid foreign key constraint violation
    db.query(RefreshToken).filter(RefreshToken.user_id == user_id).delete()
    
    db.delete(user)
    db.commit()
    
    return {
        "message": "User deleted successfully",
        "projects_reassigned": projects_reassigned,
        "admins_granted_ownership": admins_granted_ownership
    }

# ==================== Token Quota Management Endpoints ====================
# Added: 2026-01-01 - Token quota system for AI usage tracking and limiting

