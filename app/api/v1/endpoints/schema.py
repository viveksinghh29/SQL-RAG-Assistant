"""Schema endpoint.

GET /schema   — return target DB tables/columns visible to the user's role
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.auth.dependencies import get_current_user
from app.auth.roles import Role, get_restricted_tables
from app.core.logging import get_logger
from app.database.models.user import User
from app.sql_validator.schema_registry import get_known_tables

router = APIRouter(prefix="/schema", tags=["schema"])
log = get_logger("api.schema")


class TableSchema(BaseModel):
    name: str
    columns: list[str]


class SchemaResponse(BaseModel):
    tables: list[TableSchema]
    total_tables: int
    role: str


@router.get("", response_model=SchemaResponse)
async def get_schema(
    current_user: User = Depends(get_current_user),
) -> SchemaResponse:
    """Return the target DB schema filtered by the user's RBAC role.

    Tables restricted for the current role are completely omitted — this
    is the schema browser in the Streamlit UI and must only show what
    the user is allowed to query.
    """
    role = Role(current_user.role.value)
    restricted = get_restricted_tables(role)
    known = get_known_tables()

    visible_tables = [
        TableSchema(name=table, columns=sorted(columns))
        for table, columns in sorted(known.items())
        if table not in restricted
    ]

    log.bind(
        user_id=current_user.id,
        role=role.value,
        visible_tables=len(visible_tables),
    ).debug("Schema requested")

    return SchemaResponse(
        tables=visible_tables,
        total_tables=len(visible_tables),
        role=role.value,
    )
