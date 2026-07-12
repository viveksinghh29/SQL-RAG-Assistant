"""Schema ingestion utilities that combine database DDL with business metadata to create accurate, context-rich RAG documents."""
import re
from dataclasses import dataclass, field

from app.rag.base import DocumentType


@dataclass
class ColumnInfo:
    name: str
    type: str
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references: str | None = None  # "table.column"


@dataclass
class TableInfo:
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)


_CREATE_TABLE_RE = re.compile(
    r"CREATE TABLE(?:\s+IF NOT EXISTS)?\s+(\w+)\s*\((.*?)\);", re.IGNORECASE | re.DOTALL
)
_COLUMN_LINE_RE = re.compile(r"^\s*(\w+)\s+([A-Z]+(?:\(\d+(?:,\s*\d+)?\))?)", re.IGNORECASE)
_FK_RE = re.compile(
    r"FOREIGN KEY\s*\(\s*(\w+)\s*\)\s*REFERENCES\s+(\w+)\s*\(\s*(\w+)\s*\)", re.IGNORECASE
)
_PK_RE = re.compile(r"^\s*(\w+)\s+\S+.*PRIMARY KEY", re.IGNORECASE)


def parse_ddl(sql_text: str) -> list[TableInfo]:
    """Parse simple CREATE TABLE statements (the subset used in
    data/seed/schema.sql) into structured TableInfo objects.

    This is intentionally a lightweight regex parser, not a full SQL
    grammar — sufficient for our own schema.sql, not a general-purpose
    DDL parser. If a more complex schema needs ingesting, swap this for
    sqlparse-based parsing or live DB introspection via SQLAlchemy's
    `inspect()`, both already future-proofed by the TableInfo contract.
    """
    tables: list[TableInfo] = []

    for match in _CREATE_TABLE_RE.finditer(sql_text):
        table_name = match.group(1)
        body = match.group(2)

        fk_map: dict[str, str] = {}
        for fk_match in _FK_RE.finditer(body):
            col, ref_table, ref_col = fk_match.groups()
            fk_map[col] = f"{ref_table}.{ref_col}"

        columns: list[ColumnInfo] = []
        for line in body.split(","):
            line = line.strip()
            if not line or line.upper().startswith(("FOREIGN KEY", "PRIMARY KEY", "INDEX", "UNIQUE")):
                continue
            col_match = _COLUMN_LINE_RE.match(line)
            if not col_match:
                continue
            col_name, col_type = col_match.groups()
            columns.append(
                ColumnInfo(
                    name=col_name,
                    type=col_type.upper(),
                    is_primary_key=bool(_PK_RE.match(line)),
                    is_foreign_key=col_name in fk_map,
                    references=fk_map.get(col_name),
                )
            )

        tables.append(TableInfo(name=table_name, columns=columns))

    return tables


# Hand-authored business context. In a real deployment this would come
# from a data dictionary / catalog tool — here it's the bridge between
# raw DDL and the business documentation requirements.
TABLE_DESCRIPTIONS: dict[str, str] = {
    "customers": "Stores customer accounts. One row per customer.",
    "products": "Product catalog. 'price' is the customer-facing sale price; "
    "'cost' is the internal acquisition cost — never expose cost-based margin "
    "calculations without checking the user's role.",
    "orders": "One row per customer order. 'total_amount' is the pre-computed order "
    "total; do not recompute it from order_items unless explicitly asked to verify it.",
    "order_items": "Line items within an order. An order can have multiple line items "
    "(multiple products). Join to 'orders' via order_id and 'products' via product_id.",
    "suppliers": "Vendors that supply products. Joined to products via supplier_id.",
    "employees": "Internal staff records. RESTRICTED: never queryable by the 'employee' "
    "or unauthenticated roles; managers can see names/roles/departments but not payroll.",
    "payroll": "Compensation data. RESTRICTED to the 'admin' role only — never join or "
    "expose this table for any other role under any circumstances.",
}

COLUMN_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "customers": {
        "segment": "Enum: 'consumer', 'small_business', or 'enterprise'.",
        "region": "Free-text sales region, e.g. 'North', 'South', 'East', 'West', 'Central'.",
    },
    "orders": {
        "status": "Enum: 'completed', 'pending', 'cancelled', or 'refunded'. "
        "Revenue analyses should typically filter to status = 'completed' unless "
        "the user asks about cancellations/refunds specifically.",
    },
    "products": {
        "category": "Free-text category, e.g. 'Electronics', 'Home & Kitchen', 'Apparel', "
        "'Sporting Goods', 'Books', 'Office Supplies'.",
    },
}

KPI_DEFINITIONS: dict[str, str] = {
    "revenue": "Sum of orders.total_amount where status = 'completed'.",
    "average_order_value": "Revenue divided by count of completed orders (AOV).",
    "gross_margin": "(products.price - products.cost) / products.price, per product. "
    "Restricted: only meaningful/shown to admin and manager roles.",
    "customer_lifetime_value": "Sum of total_amount across all of a customer's completed "
    "orders to date.",
    "order_fulfillment_rate": "Count of completed orders divided by count of all orders "
    "(completed + pending + cancelled + refunded).",
}


def build_schema_documents(ddl_text: str) -> list[tuple[str, DocumentType, str, dict]]:
    """Produce (content, document_type, source, metadata) tuples ready for
    `VectorStore.add_documents`, covering schema, table descriptions,
    column descriptions, relationships, and KPI definitions.
    """
    tables = parse_ddl(ddl_text)
    documents: list[tuple[str, DocumentType, str, dict]] = []

    for table in tables:
        col_lines = ", ".join(f"{c.name} ({c.type})" for c in table.columns)
        schema_text = f"Table '{table.name}' has columns: {col_lines}."
        documents.append((schema_text, DocumentType.SCHEMA, "schema.sql", {"table": table.name}))

        if table.name in TABLE_DESCRIPTIONS:
            documents.append(
                (
                    f"Table '{table.name}': {TABLE_DESCRIPTIONS[table.name]}",
                    DocumentType.TABLE_DESCRIPTION,
                    "data_dictionary",
                    {"table": table.name},
                )
            )

        for col_name, description in COLUMN_DESCRIPTIONS.get(table.name, {}).items():
            documents.append(
                (
                    f"Column '{table.name}.{col_name}': {description}",
                    DocumentType.COLUMN_DESCRIPTION,
                    "data_dictionary",
                    {"table": table.name, "column": col_name},
                )
            )

        for col in table.columns:
            if col.is_foreign_key and col.references:
                documents.append(
                    (
                        f"Relationship: {table.name}.{col.name} references {col.references}.",
                        DocumentType.RELATIONSHIP,
                        "schema.sql",
                        {"table": table.name, "column": col.name, "references": col.references},
                    )
                )

    for kpi_name, definition in KPI_DEFINITIONS.items():
        documents.append(
            (
                f"KPI '{kpi_name}': {definition}",
                DocumentType.KPI_DEFINITION,
                "kpi_definitions",
                {"kpi": kpi_name},
            )
        )

    return documents
