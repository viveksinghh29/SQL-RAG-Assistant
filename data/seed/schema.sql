-- Sample retail/e-commerce database — this is the TARGET database that
-- end users query in natural language. It is intentionally separate from
-- the app DB. In production this schema is owned by the company; here we
-- create it for demo/dev purposes.
--
-- Run against TARGET_DB_NAME (see .env). The application connects to this
-- DB with a READ-ONLY credential — only SELECTs are ever executed against
-- it (enforced again in Phase 7's SQL validator as defense in depth).

CREATE TABLE IF NOT EXISTS suppliers (
    id              INT PRIMARY KEY AUTO_INCREMENT,
    name            VARCHAR(255) NOT NULL,
    country         VARCHAR(100) NOT NULL,
    contact_email   VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS customers (
    id              INT PRIMARY KEY AUTO_INCREMENT,
    name            VARCHAR(255) NOT NULL,
    email           VARCHAR(255) NOT NULL UNIQUE,
    region          VARCHAR(100) NOT NULL,
    segment         VARCHAR(50)  NOT NULL,  -- 'consumer' | 'small_business' | 'enterprise'
    signup_date     DATE NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    id              INT PRIMARY KEY AUTO_INCREMENT,
    name            VARCHAR(255) NOT NULL,
    category        VARCHAR(100) NOT NULL,
    price           DECIMAL(10, 2) NOT NULL,
    cost            DECIMAL(10, 2) NOT NULL,
    supplier_id     INT NOT NULL,
    FOREIGN KEY (supplier_id) REFERENCES suppliers(id)
);

CREATE TABLE IF NOT EXISTS orders (
    id              INT PRIMARY KEY AUTO_INCREMENT,
    customer_id     INT NOT NULL,
    order_date      DATE NOT NULL,
    status          VARCHAR(50) NOT NULL,  -- 'completed' | 'pending' | 'cancelled' | 'refunded'
    total_amount    DECIMAL(12, 2) NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);

CREATE TABLE IF NOT EXISTS order_items (
    id              INT PRIMARY KEY AUTO_INCREMENT,
    order_id        INT NOT NULL,
    product_id      INT NOT NULL,
    quantity        INT NOT NULL,
    unit_price      DECIMAL(10, 2) NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);

-- Sensitive: RESTRICTED_TABLES in app/auth/roles.py blocks 'employees' and
-- 'payroll' for the Employee role, and 'payroll' for Manager.
CREATE TABLE IF NOT EXISTS employees (
    id              INT PRIMARY KEY AUTO_INCREMENT,
    name            VARCHAR(255) NOT NULL,
    role            VARCHAR(100) NOT NULL,
    department      VARCHAR(100) NOT NULL,
    hire_date       DATE NOT NULL
);

CREATE TABLE IF NOT EXISTS payroll (
    employee_id     INT PRIMARY KEY,
    salary          DECIMAL(12, 2) NOT NULL,
    bonus           DECIMAL(12, 2) NOT NULL DEFAULT 0,
    FOREIGN KEY (employee_id) REFERENCES employees(id)
);

CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_order_date ON orders(order_date);
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);
CREATE INDEX idx_products_supplier_id ON products(supplier_id);
