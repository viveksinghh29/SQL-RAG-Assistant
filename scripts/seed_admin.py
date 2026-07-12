"""Admin user seeder — creates the initial admin account.

Usage:
    python -m scripts.seed_admin

Reads credentials from environment / .env:
    ADMIN_EMAIL     (default: admin@example.com)
    ADMIN_PASSWORD  (required — must be set explicitly)
    ADMIN_NAME      (default: System Administrator)

Run once after `alembic upgrade head` to bootstrap the system.
Idempotent: if the email already exists, prints a message and exits.
"""

import asyncio
import os
import sys
from pathlib import Path

# Make the project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.auth.roles import Role
from app.auth.security import hash_password
from app.config.settings import get_settings
from app.database.models.user import User
from app.database.repositories.user_repository import UserRepository
from app.schemas.user import UserCreate


async def seed_admin() -> None:
    settings = get_settings()

    admin_email = os.getenv("ADMIN_EMAIL", "admin@example.com")
    admin_password = os.getenv("ADMIN_PASSWORD", "")
    admin_name = os.getenv("ADMIN_NAME", "System Administrator")

    if not admin_password:
        print("ERROR: ADMIN_PASSWORD environment variable is not set.")
        print("Set it in your .env file or as an environment variable and retry.")
        sys.exit(1)

    if len(admin_password) < 8:
        print("ERROR: ADMIN_PASSWORD must be at least 8 characters.")
        sys.exit(1)

    engine = create_async_engine(settings.app_database_url, echo=False)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    async with Session() as session:
        repo = UserRepository(session)
        existing = await repo.get_by_email(admin_email)

        if existing is not None:
            print(f"Admin user '{admin_email}' already exists (id={existing.id}).")
            print("No changes made.")
            await engine.dispose()
            return

        user = await repo.create(
            UserCreate(
                email=admin_email,
                password=admin_password,
                full_name=admin_name,
                role=Role.ADMIN,
            )
        )
        await session.commit()
        print(f"Admin user created successfully:")
        print(f"  Email: {user.email}")
        print(f"  Name:  {user.full_name}")
        print(f"  Role:  {user.role.value}")
        print(f"  ID:    {user.id}")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(seed_admin())
