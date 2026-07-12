"""Centralized SQL safety rules defining forbidden keywords and injection patterns for defense-in-depth query validation."""
import re

# ---------------------------------------------------------------------------
# Forbidden SQL operation keywords
# Any statement whose first non-whitespace keyword matches one of these is
# hard-blocked before execution.  Case-insensitive at the point of use.
# ---------------------------------------------------------------------------

FORBIDDEN_OPERATIONS: frozenset[str] = frozenset({
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "REPLACE",     # MySQL REPLACE INTO is effectively INSERT+DELETE
    "RENAME",
    "GRANT",
    "REVOKE",
    "LOCK",
    "UNLOCK",
    "CALL",
    "EXEC",
    "EXECUTE",
    "LOAD",        # LOAD DATA INFILE
    "IMPORT",
    "FLUSH",
    "KILL",
    "SHUTDOWN",
})

# ---------------------------------------------------------------------------
# SQL injection pattern regexes
# Detected in the raw (pre-parse) SQL string.  Order matters: more specific
# patterns are listed first so the first match gives the clearest message.
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[tuple[str, str]] = [
    # Stacked statements via semicolons — e.g.  SELECT 1; DROP TABLE users
    (r";\s*\w", "stacked_statement"),
    # Classic comment-based injection:  ' OR 1=1 --  /  ' OR 1=1 #
    (r"--\s", "comment_injection"),
    (r"#\s*$", "comment_injection"),
    (r"/\*.*?\*/", "block_comment"),
    # UNION-based extraction
    (r"\bUNION\s+(ALL\s+)?SELECT\b", "union_select"),
    # Boolean-based blind injection keywords
    (r"\bOR\s+1\s*=\s*1\b", "boolean_injection"),
    (r"\bAND\s+1\s*=\s*1\b", "boolean_injection"),
    # Time-based blind injection
    (r"\bSLEEP\s*\(", "sleep_injection"),
    (r"\bBENCHMARK\s*\(", "benchmark_injection"),
    # Out-of-band / file read-write
    (r"\bINTO\s+OUTFILE\b", "file_write"),
    (r"\bLOAD_FILE\s*\(", "file_read"),
    # Information schema / system table probing
    (r"\binformation_schema\b", "schema_probing"),
    (r"\bperformance_schema\b", "schema_probing"),
    (r"\bmysql\.\w+\b", "system_table_access"),
    # XSS in SQL output (exotic, but seen in injection payloads)
    (r"<script\b", "xss_attempt"),
    # Hex/char encoding used to bypass keyword filters
    (r"\b0x[0-9a-fA-F]{4,}\b", "hex_encoding"),
    (r"\bCHAR\s*\(", "char_encoding"),
]

INJECTION_REGEXES: list[tuple[re.Pattern, str]] = [
    (re.compile(pattern, re.IGNORECASE | re.DOTALL), label)
    for pattern, label in _INJECTION_PATTERNS
]
