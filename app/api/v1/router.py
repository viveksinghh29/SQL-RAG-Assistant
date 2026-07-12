"""API v1 router — aggregates all endpoint sub-routers.

All routes are prefixed with `/api/v1` (set in `app/main.py` via
`settings.api_v1_prefix`). Adding a new endpoint group means adding
one `include_router` line here.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth,
    chat,
    feedback,
    health,
    history,
    schema,
    users,
)

api_router = APIRouter()

api_router.include_router(auth.router)
api_router.include_router(chat.router)
api_router.include_router(history.router)
api_router.include_router(schema.router)
api_router.include_router(feedback.router)
api_router.include_router(users.router)
api_router.include_router(health.router)
