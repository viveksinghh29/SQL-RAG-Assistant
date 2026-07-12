"""Connection pool configuration.

Centralises pool sizing so it's tuneable from one place.
Both the app DB (session.py) and target DB (query_executor/connection.py)
reference these constants.

Sizing rationale (for a single-host deployment):
  - APP_DB: stores users, conversations, feedback — high concurrency,
    short-lived transactions. Pool of 10 + 20 overflow handles 30
    simultaneous requests without queueing.
  - TARGET_DB: runs potentially slow analytical queries — smaller pool
    (5+5) prevents monopolising the DB server on a slow query storm.
    The timeout prevents pool exhaustion from hanging queries.

For a scaled deployment (multiple API pods):
  - Reduce pool_size proportionally (total = pods × pool_size ≤ max_connections)
  - Consider PgBouncer / ProxySQL to multiplex at the proxy layer
"""

APP_DB_POOL = {
    "pool_size": 10,
    "max_overflow": 20,
    "pool_pre_ping": True,       # detect stale connections before using
    "pool_recycle": 1800,        # recycle connections after 30 min (avoids MySQL 8h timeout)
    "pool_timeout": 30,          # wait max 30s for a free connection
}

TARGET_DB_POOL = {
    "pool_size": 5,
    "max_overflow": 5,
    "pool_pre_ping": True,
    "pool_recycle": 1800,
    "pool_timeout": 15,
    "echo": False,               # never echo raw SQL for the target DB
}
