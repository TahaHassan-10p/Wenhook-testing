"""
Token Quota Management API Routes

This module provides REST API endpoints for managing and monitoring AI token usage quotas.

Features:
- User-level quota viewing (any authenticated user can view their own)
- Admin quota management (set limits, add purchased tokens)
- Detailed usage statistics and operation history
- Token operation audit trail

Token System Design:
1. Monthly Quota: Resets on 1st of each month (default 2M tokens)
2. Purchased Tokens: Never expire, used after monthly quota exhausted
3. Usage Tracking: Every AI operation recorded with full context
4. Cost Estimation: Calculated based on model pricing

Created: 2026-01-01
Last Modified: 2026-01-01
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import select, func, desc

from app.database.session import get_db
from app.database.models import User, UserTokenUsage, Project
from app.auth.deps import get_current_active_user, require_admin
from app.services.user_quota_manager import get_user_usage_stats
from app.schemas.token_quota import (
    TokenQuotaUpdate,
    PurchasedTokensAdd,
    DetailedUserTokenStats,
    TokenOperationHistory,
    TokenOperationRecord,
    QuotaCheckResponse
)

router = APIRouter(prefix="/token-quota", tags=["Token Quota"])


@router.get("/me", response_model=DetailedUserTokenStats)
async def get_my_token_usage(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get current user's token usage statistics and quota information.
    
    **Returns:**
    - Monthly quota and current usage
    - Purchased token balance (permanent tokens)
    - Detailed breakdown by operation type
    - Monthly statistics (input/output tokens, cost)
    - Quota reset date
    
    **Use Cases:**
    - Display quota status in user dashboard
    - Show warning when approaching limit
    - Provide usage insights to users
    
    **Example Response:**
    ```json
    {
      "monthly_quota": 2000000,
      "tokens_used_this_month": 450000,
      "remaining_monthly": 1550000,
      "purchased_tokens_balance": 500000,
      "total_available": 2050000,
      "quota_reset_date": "2026-02-01T00:00:00Z",
      "total_tokens_lifetime": 5680000,
      "monthly_stats": {
        "total_tokens": 450000,
        "input_tokens": 320000,
        "output_tokens": 130000,
        "estimated_cost": 1.35
      },
      "operation_breakdown": [
        {"operation_type": "project_import", "tokens": 250000, "count": 3},
        {"operation_type": "chunk_processing", "tokens": 150000, "count": 12}
      ]
    }
    ```
    """
    stats = await get_user_usage_stats(current_user.id, db)
    
    if 'error' in stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=stats['error']
        )
    
    return stats


@router.get("/users/{user_id}", response_model=DetailedUserTokenStats)
async def get_user_token_usage(
    user_id: int,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Get token usage statistics for a specific user (admin only).
    
    **Admin Use Cases:**
    - Monitor platform-wide token consumption
    - Identify users who may need quota adjustments
    - Investigate quota-related support tickets
    - Generate usage reports
    
    **Parameters:**
    - `user_id`: Target user's ID
    
    **Authorization:** Admin only
    """
    stats = await get_user_usage_stats(user_id, db)
    
    if 'error' in stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=stats['error']
        )
    
    return stats


@router.put("/users/{user_id}/quota")
async def update_user_token_quota(
    user_id: int,
    quota_update: TokenQuotaUpdate,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Update a user's monthly token quota (admin only).
    
    **Use Cases:**
    - Increase quota for power users or enterprise customers
    - Reduce quota for trial/free-tier users
    - Set custom limits based on subscription tier
    - Temporarily boost quota for special projects
    
    **Important Notes:**
    - This only affects monthly quota (resets on 1st of month)
    - Does NOT affect purchased token balance
    - Current month's usage is NOT reset
    - Set to very high value (e.g., 999999999) for effectively unlimited
    
    **Request Body:**
    ```json
    {
      "monthly_token_quota": 5000000
    }
    ```
    
    **Response:**
    ```json
    {
      "message": "Token quota updated successfully",
      "user_id": 42,
      "user_email": "user@example.com",
      "old_quota": 2000000,
      "new_quota": 5000000,
      "current_usage": 450000,
      "remaining": 4550000
    }
    ```
    
    **Authorization:** Admin only
    """
    # Get the user
    result = db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found"
        )
    
    # Update quota
    old_quota = user.monthly_token_quota
    user.monthly_token_quota = quota_update.monthly_token_quota
    
    db.commit()
    db.refresh(user)
    
    return {
        "message": "Token quota updated successfully",
        "user_id": user_id,
        "user_email": user.email,
        "old_quota": old_quota,
        "new_quota": user.monthly_token_quota,
        "current_usage": user.tokens_used_this_month,
        "remaining": max(0, user.monthly_token_quota - user.tokens_used_this_month)
    }


@router.post("/users/{user_id}/purchased-tokens")
async def add_purchased_tokens(
    user_id: int,
    token_purchase: PurchasedTokensAdd,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Add purchased tokens to a user's account (admin only).
    
    **Purchased Token Characteristics:**
    - Never expire (permanent balance)
    - Separate from monthly quota
    - Used after monthly quota is exhausted
    - Survives monthly resets
    
    **Use Cases:**
    - Fulfill token purchase orders (future Stripe integration)
    - Grant promotional token packages
    - Resolve support tickets with token compensation
    - Testing quota system behavior
    
    **Request Body:**
    ```json
    {
      "amount": 1000000,
      "note": "Stripe payment confirmed - Order #12345"
    }
    ```
    
    **Response:**
    ```json
    {
      "message": "Purchased tokens added successfully",
      "user_id": 42,
      "user_email": "user@example.com",
      "tokens_added": 1000000,
      "old_balance": 500000,
      "new_balance": 1500000,
      "note": "Stripe payment confirmed - Order #12345"
    }
    ```
    
    **Future Integration:**
    This endpoint will be called automatically by Stripe webhooks
    when users purchase token packages.
    
    **Authorization:** Admin only
    """
    # Get the user
    result = db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found"
        )
    
    # Add purchased tokens
    old_balance = user.purchased_tokens_balance or 0
    user.purchased_tokens_balance = old_balance + token_purchase.amount
    
    db.commit()
    db.refresh(user)
    
    # TODO: Create TokenPurchase record for audit trail (future enhancement)
    # This will include: user_id, amount, payment_method, transaction_id, note, timestamp
    
    return {
        "message": "Purchased tokens added successfully",
        "user_id": user_id,
        "user_email": user.email,
        "tokens_added": token_purchase.amount,
        "old_balance": old_balance,
        "new_balance": user.purchased_tokens_balance,
        "note": token_purchase.note
    }


@router.get("/users/{user_id}/operations", response_model=TokenOperationHistory)
async def get_user_token_operations(
    user_id: int,
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Get detailed history of token operations for a user (admin only).
    
    **Returns paginated list of all AI operations with:**
    - Operation type (project_import, chunk_processing, test_healing, etc.)
    - Associated project (if applicable)
    - AI provider and model used
    - Input/output/total token counts
    - Estimated cost in USD
    - Timestamp
    
    **Query Parameters:**
    - `page`: Page number (default: 1)
    - `page_size`: Items per page (default: 50, max: 200)
    
    **Use Cases:**
    - Debug quota-related issues
    - Identify expensive operations
    - Analyze usage patterns
    - Generate billing reports
    - Audit AI API usage
    
    **Example Response:**
    ```json
    {
      "total": 156,
      "page": 1,
      "page_size": 50,
      "items": [
        {
          "id": 12345,
          "project_id": 42,
          "project_name": "Mobile App Test Suite",
          "operation_type": "project_import",
          "ai_provider": "anthropic",
          "ai_model": "claude-3-opus-20240229",
          "input_tokens": 45000,
          "output_tokens": 12000,
          "total_tokens": 57000,
          "estimated_cost": 0.95,
          "created_at": "2026-01-15T14:30:00Z"
        }
      ]
    }
    ```
    
    **Authorization:** Admin only
    """
    # Verify user exists
    result = db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found"
        )
    
    # Get total count
    count_result = db.execute(
        select(func.count(UserTokenUsage.id))
        .where(UserTokenUsage.user_id == user_id)
    )
    total = count_result.scalar()
    
    # Get paginated operations
    offset = (page - 1) * page_size
    operations_result = db.execute(
        select(UserTokenUsage, Project.name.label('project_name'))
        .outerjoin(Project, UserTokenUsage.project_id == Project.id)
        .where(UserTokenUsage.user_id == user_id)
        .order_by(desc(UserTokenUsage.created_at))
        .limit(page_size)
        .offset(offset)
    )
    
    items = []
    for row in operations_result:
        usage = row[0]
        project_name = row[1]
        items.append(TokenOperationRecord(
            id=usage.id,
            project_id=usage.project_id,
            project_name=project_name,
            operation_type=usage.operation_type,
            ai_provider=usage.ai_provider,
            ai_model=usage.ai_model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost=usage.estimated_cost,
            created_at=usage.created_at
        ))
    
    return TokenOperationHistory(
        total=total,
        items=items,
        page=page,
        page_size=page_size
    )


@router.get("/platform-stats")
async def get_platform_token_stats(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Get platform-wide token usage statistics (admin only).
    
    **Provides aggregate metrics for:**
    - Total tokens used today/this month/all time
    - Total estimated costs
    - Average tokens per user
    - Most expensive operations
    - Top users by consumption
    
    **Use Cases:**
    - Platform health monitoring
    - Cost forecasting
    - Capacity planning
    - Executive reporting
    
    **Authorization:** Admin only
    """
    from datetime import datetime, timezone, timedelta
    
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    # Today's usage
    today_result = db.execute(
        select(
            func.sum(UserTokenUsage.total_tokens).label('tokens'),
            func.sum(UserTokenUsage.estimated_cost).label('cost'),
            func.count(UserTokenUsage.id).label('operations')
        )
        .where(UserTokenUsage.created_at >= today_start)
    )
    today_stats = today_result.one()
    
    # This month's usage
    month_result = db.execute(
        select(
            func.sum(UserTokenUsage.total_tokens).label('tokens'),
            func.sum(UserTokenUsage.estimated_cost).label('cost'),
            func.count(UserTokenUsage.id).label('operations')
        )
        .where(UserTokenUsage.created_at >= month_start)
    )
    month_stats = month_result.one()
    
    # All-time usage
    total_result = db.execute(
        select(
            func.sum(UserTokenUsage.total_tokens).label('tokens'),
            func.sum(UserTokenUsage.estimated_cost).label('cost'),
            func.count(UserTokenUsage.id).label('operations')
        )
    )
    total_stats = total_result.one()
    
    # User count
    user_count_result = db.execute(select(func.count(User.id)))
    user_count = user_count_result.scalar()
    
    # Average per user
    avg_per_user = (month_stats.tokens or 0) / user_count if user_count > 0 else 0
    
    return {
        "today": {
            "total_tokens": today_stats.tokens or 0,
            "estimated_cost": float(today_stats.cost or 0.0),
            "operations": today_stats.operations or 0
        },
        "this_month": {
            "total_tokens": month_stats.tokens or 0,
            "estimated_cost": float(month_stats.cost or 0.0),
            "operations": month_stats.operations or 0,
            "avg_tokens_per_user": int(avg_per_user)
        },
        "all_time": {
            "total_tokens": total_stats.tokens or 0,
            "estimated_cost": float(total_stats.cost or 0.0),
            "operations": total_stats.operations or 0
        },
        "user_count": user_count
    }
