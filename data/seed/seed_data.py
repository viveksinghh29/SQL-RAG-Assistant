"""Generate and insert synthetic data into the target retail database.

Usage:
    python -m data.seed.seed_data

Reads TARGET_DB_* from settings/.env. Uses a fixed Faker seed so the
dataset is reproducible across runs (useful for stable RAG SQL examples
and deterministic tests). Idempotent in the sense that it truncates
existing data before reseeding — do not run against a database you care
about.
"""

import random
from datetime import date, timedelta
from decimal import Decimal

from faker import Faker
from sqlalchemy import create_engine, text

from app.config.settings import get_settings

SEED = 42
N_SUPPLIERS = 15
N_CUSTOMERS = 200
N_PRODUCTS = 80
N_ORDERS = 600
N_EMPLOYEES = 25

CATEGORIES = ["Electronics", "Home & Kitchen", "Apparel", "Sporting Goods", "Books", "Office Supplies"]
SEGMENTS = ["consumer", "small_business", "enterprise"]
ORDER_STATUSES = ["completed", "completed", "completed", "pending", "cancelled", "refunded"]
DEPARTMENTS = ["Sales", "Engineering", "Marketing", "Operations", "Support", "Finance"]
REGIONS = ["North", "South", "East", "West", "Central"]


def _sync_target_db_url() -> str:
    """The async driver (aiomysql) can't be used by a sync script — swap
    to the sync pymysql driver for this one-off seeding script."""
    settings = get_settings()
    return settings.target_database_url.replace("mysql+aiomysql", "mysql+pymysql")


def generate_and_seed() -> None:
    fake = Faker()
    Faker.seed(SEED)
    random.seed(SEED)

    engine = create_engine(_sync_target_db_url(), echo=False)

    with engine.begin() as conn:
        print("Truncating existing data...")
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
        for table in ["payroll", "employees", "order_items", "orders", "products", "customers", "suppliers"]:
            conn.execute(text(f"TRUNCATE TABLE {table}"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))

        print(f"Seeding {N_SUPPLIERS} suppliers...")
        for _ in range(N_SUPPLIERS):
            conn.execute(
                text(
                    "INSERT INTO suppliers (name, country, contact_email) "
                    "VALUES (:name, :country, :email)"
                ),
                {"name": fake.company(), "country": fake.country(), "email": fake.company_email()},
            )

        print(f"Seeding {N_CUSTOMERS} customers...")
        for _ in range(N_CUSTOMERS):
            signup = fake.date_between(start_date="-3y", end_date="-1d")
            conn.execute(
                text(
                    "INSERT INTO customers (name, email, region, segment, signup_date) "
                    "VALUES (:name, :email, :region, :segment, :signup_date)"
                ),
                {
                    "name": fake.name(),
                    "email": fake.unique.email(),
                    "region": random.choice(REGIONS),
                    "segment": random.choice(SEGMENTS),
                    "signup_date": signup,
                },
            )

        supplier_ids = list(range(1, N_SUPPLIERS + 1))
        print(f"Seeding {N_PRODUCTS} products...")
        for _ in range(N_PRODUCTS):
            cost = Decimal(random.uniform(5, 300)).quantize(Decimal("0.01"))
            price = (cost * Decimal(random.uniform(1.3, 2.5))).quantize(Decimal("0.01"))
            conn.execute(
                text(
                    "INSERT INTO products (name, category, price, cost, supplier_id) "
                    "VALUES (:name, :category, :price, :cost, :supplier_id)"
                ),
                {
                    "name": fake.catch_phrase(),
                    "category": random.choice(CATEGORIES),
                    "price": price,
                    "cost": cost,
                    "supplier_id": random.choice(supplier_ids),
                },
            )

        customer_ids = list(range(1, N_CUSTOMERS + 1))
        product_rows = list(conn.execute(text("SELECT id, price FROM products")))

        print(f"Seeding {N_ORDERS} orders with line items...")
        for _ in range(N_ORDERS):
            order_date = fake.date_between(start_date="-1y", end_date="today")
            status = random.choice(ORDER_STATUSES)
            items = random.sample(product_rows, k=random.randint(1, 4))
            line_items = [(p.id, random.randint(1, 5), p.price) for p in items]
            total = sum(qty * price for _, qty, price in line_items)

            result = conn.execute(
                text(
                    "INSERT INTO orders (customer_id, order_date, status, total_amount) "
                    "VALUES (:customer_id, :order_date, :status, :total)"
                ),
                {
                    "customer_id": random.choice(customer_ids),
                    "order_date": order_date,
                    "status": status,
                    "total": total,
                },
            )
            order_id = result.lastrowid
            for product_id, qty, unit_price in line_items:
                conn.execute(
                    text(
                        "INSERT INTO order_items (order_id, product_id, quantity, unit_price) "
                        "VALUES (:order_id, :product_id, :qty, :unit_price)"
                    ),
                    {"order_id": order_id, "product_id": product_id, "qty": qty, "unit_price": unit_price},
                )

        print(f"Seeding {N_EMPLOYEES} employees with payroll...")
        for _ in range(N_EMPLOYEES):
            hire_date = fake.date_between(start_date="-5y", end_date="-30d")
            dept = random.choice(DEPARTMENTS)
            result = conn.execute(
                text(
                    "INSERT INTO employees (name, role, department, hire_date) "
                    "VALUES (:name, :role, :dept, :hire_date)"
                ),
                {
                    "name": fake.name(),
                    "role": fake.job(),
                    "dept": dept,
                    "hire_date": hire_date,
                },
            )
            employee_id = result.lastrowid
            salary = Decimal(random.uniform(45000, 160000)).quantize(Decimal("0.01"))
            bonus = (salary * Decimal(random.uniform(0, 0.15))).quantize(Decimal("0.01"))
            conn.execute(
                text("INSERT INTO payroll (employee_id, salary, bonus) VALUES (:eid, :salary, :bonus)"),
                {"eid": employee_id, "salary": salary, "bonus": bonus},
            )

    print("Seeding complete.")


if __name__ == "__main__":
    generate_and_seed()
