"""Feedback endpoints.

POST /feedback              — submit thumbs-up/down on a message
GET  /feedback/analytics    — aggregate feedback stats (admin/manager only)
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user, require_role
from app.auth.roles import Role
from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.database.models.chat import ChatMessage
from app.database.models.feedback import Feedback
from app.database.models.user import User
from app.database.session import get_db_session

router = APIRouter(prefix="/feedback", tags=["feedback"])
log = get_logger("api.feedback")


class FeedbackRequest(BaseModel):
    message_id: int
    is_positive: bool
    comment: str | None = None


class FeedbackResponse(BaseModel):
    id: int
    message_id: int
    is_positive: bool
    comment: str | None


class FeedbackAnalytics(BaseModel):
    total_feedback: int
    positive: int
    negative: int
    positive_rate: float


@router.post("", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(
    payload: FeedbackRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> FeedbackResponse:
    """Submit feedback (thumbs up/down) on an assistant message."""
    # Verify the message exists
    message = await session.get(ChatMessage, payload.message_id)
    if message is None:
        raise NotFoundError(f"Message {payload.message_id} not found.")

    feedback = Feedback(
        message_id=payload.message_id,
        user_id=current_user.id,
        is_positive=payload.is_positive,
        comment=payload.comment,
    )
    session.add(feedback)
    await session.flush()
    await session.refresh(feedback)
    await session.commit()

    log.bind(
        user_id=current_user.id,
        message_id=payload.message_id,
        is_positive=payload.is_positive,
    ).info("Feedback submitted")

    return FeedbackResponse(
        id=feedback.id,
        message_id=feedback.message_id,
        is_positive=feedback.is_positive,
        comment=feedback.comment,
    )


@router.get("/analytics", response_model=FeedbackAnalytics)
async def get_feedback_analytics(
    _: User = Depends(require_role(Role.ADMIN, Role.MANAGER)),
    session: AsyncSession = Depends(get_db_session),
) -> FeedbackAnalytics:
    """Aggregate feedback statistics. Requires Admin or Manager role."""
    result = await session.execute(
        select(
            func.count(Feedback.id).label("total"),
            func.sum(
                func.cast(Feedback.is_positive, sqlalchemy_integer())
            ).label("positive"),
        )
    )
    row = result.one()
    total = row.total or 0
    positive = int(row.positive or 0)
    negative = total - positive

    return FeedbackAnalytics(
        total_feedback=total,
        positive=positive,
        negative=negative,
        positive_rate=round(positive / total, 4) if total > 0 else 0.0,
    )


def sqlalchemy_integer():
    from sqlalchemy import Integer
    return Integer
