# Business Rules

## Revenue and order analysis

Revenue figures should always filter to `orders.status = 'completed'` unless the
user explicitly asks about pending, cancelled, or refunded orders. A query for
"total sales" or "revenue" without qualification means completed orders only.

When a user asks to "compare" a metric across two time periods, compute both
periods separately and present the percentage change, not just the two raw
numbers — the percentage change is what makes a comparison useful.

## Customer segmentation

The `segment` column on `customers` ('consumer', 'small_business', 'enterprise')
is the standard way to group customers for business reporting. When a user
mentions "B2B customers," map this to `segment IN ('small_business', 'enterprise')`.

## Sensitive data handling

Employee compensation data (the `payroll` table) is restricted to the `admin`
role. If a non-admin user asks about salaries, bonuses, or payroll, the system
must decline rather than attempt to approximate an answer from other tables.

Product `cost` and any margin calculation derived from it (price minus cost)
should only be surfaced to `admin` and `manager` roles — `employee`-role users
asking about "profit" or "margin" should receive sales/revenue figures, not
cost-derived margin.

## Time periods

Unless the user specifies otherwise, "this month" and "this year" refer to the
current calendar month/year relative to `order_date`. "Last month" means the
calendar month immediately prior to the current one, not a rolling 30-day window.
