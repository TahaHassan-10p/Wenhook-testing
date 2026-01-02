# attachments_routes.py
from typing import List
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.database.models import TestCaseActivity, ActivityType, Attachment
from app.auth.deps import get_current_active_user
from app.database.session import get_db
from app.database.models import User, TestCase
from app.services.storage_service import storage_service

router = APIRouter()


@router.get("/{case_id}/attachments")
async def get_attachments(case_id: int, db: Session = Depends(get_db)):
    """Get all attachments for a test case from database"""
    test_case = db.query(TestCase).filter(TestCase.id == case_id).first()
    if not test_case:
        raise HTTPException(status_code=404, detail="Test case not found")

    # Get attachments from database
    attachments = db.query(Attachment).filter(
        Attachment.entity_type == "test_case",
        Attachment.entity_id == case_id,
        Attachment.is_deleted == False
    ).all()
    
    return [
        {
            "id": att.id,
            "name": att.original_filename,
            "filename": att.original_filename,
            "size": att.file_size,
            "content_type": att.content_type,
            "uploaded_by": att.uploaded_by.name,
            "uploaded_at": att.created_at,
            "project_id": test_case.project_id
        }
        for att in attachments
    ]


@router.post("/{case_id}/attachments")
async def upload_attachments(
    case_id: int,
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Upload attachments to GCS and save metadata to database"""
    test_case = db.query(TestCase).filter(TestCase.id == case_id).first()
    if not test_case:
        raise HTTPException(status_code=404, detail="Test case not found")

    uploaded_attachments = []
    
    for file in files:
        # Upload to GCS
        gcs_metadata = await storage_service.upload_file(
            file=file,
            project_id=test_case.project_id,
            entity_type="test_case",
            entity_id=case_id
        )
        
        # Create database record
        attachment = Attachment(
            entity_type="test_case",
            entity_id=case_id,
            project_id=test_case.project_id,
            original_filename=gcs_metadata['original_filename'],
            storage_filename=gcs_metadata['storage_filename'],
            storage_path=gcs_metadata['storage_path'],
            gcs_bucket=gcs_metadata['gcs_bucket'],
            gcs_object_name=gcs_metadata['gcs_object_name'],
            file_size=gcs_metadata['file_size'],
            content_type=gcs_metadata['content_type'],
            uploaded_by_id=current_user.id
        )
        db.add(attachment)
        db.flush()  # Get the ID
        
        # Log activity
        activity = TestCaseActivity(
            test_case_id=case_id,
            activity_type=ActivityType.UPDATED,
            description=f"Attachment uploaded: {file.filename}",
            activity_data={
                "attachment_id": attachment.id,
                "filename": file.filename,
                "file_size": gcs_metadata['file_size']
            },
            created_by_id=current_user.id
        )
        db.add(activity)
        
        uploaded_attachments.append({
            "id": attachment.id,
            "name": attachment.original_filename,
            "filename": attachment.original_filename,
            "size": attachment.file_size,
            "content_type": attachment.content_type,
            "uploaded_by": current_user.name,
            "uploaded_at": attachment.created_at,
            "project_id": test_case.project_id
        })
    
    db.commit()
    
    return {
        "message": "Files uploaded successfully",
        "attachments": uploaded_attachments
    }


@router.get("/{case_id}/attachments/{attachment_id}")
async def download_attachment(
    case_id: int,
    attachment_id: int,
    db: Session = Depends(get_db)
):
    """Download a specific attachment file from GCS by ID"""
    
    # Get attachment record from database
    attachment = db.query(Attachment).filter(
        Attachment.id == attachment_id,
        Attachment.entity_type == "test_case",
        Attachment.entity_id == case_id,
        Attachment.is_deleted == False
    ).first()
    
    if not attachment:
        raise HTTPException(status_code=404, detail="Attachment not found")
    
    # Download from GCS
    file_content = storage_service.download_file(attachment.gcs_object_name)
    
    # Return as streaming response
    return Response(
        content=file_content,
        media_type=attachment.content_type or 'application/octet-stream',
        headers={
            "Content-Disposition": f'attachment; filename="{attachment.original_filename}"'
        }
    )


@router.delete("/{case_id}/attachments/{attachment_id}")
async def delete_attachment(
    case_id: int,
    attachment_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Soft delete an attachment by marking it as deleted"""
    
    # Get attachment record from database
    attachment = db.query(Attachment).filter(
        Attachment.id == attachment_id,
        Attachment.entity_type == "test_case",
        Attachment.entity_id == case_id,
        Attachment.is_deleted == False
    ).first()
    
    if not attachment:
        raise HTTPException(status_code=404, detail="Attachment not found")
    
    # Verify test case exists and user has access
    test_case = db.query(TestCase).filter(TestCase.id == case_id).first()
    if not test_case:
        raise HTTPException(status_code=404, detail="Test case not found")
    
    # Soft delete the attachment
    attachment.is_deleted = True
    attachment.updated_at = db.execute(text("SELECT NOW()")).scalar()
    
    # Log activity
    activity = TestCaseActivity(
        test_case_id=case_id,
        activity_type=ActivityType.UPDATED,
        description=f"Attachment deleted: {attachment.original_filename}",
        activity_data={
            "attachment_id": attachment.id,
            "filename": attachment.original_filename,
            "action": "deleted"
        },
        created_by_id=current_user.id
    )
    db.add(activity)
    
    db.commit()
    
    return {
        "message": "Attachment deleted successfully",
        "attachment_id": attachment_id,
        "filename": attachment.original_filename
    }
