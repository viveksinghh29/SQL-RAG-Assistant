# Frequently Asked Questions

**Q: Why doesn't my "total sales" number match what I expected?**
A: Sales/revenue figures only include orders with status 'completed' by default.
Pending, cancelled, and refunded orders are excluded unless you ask for them
specifically.

**Q: What counts as a "top customer"?**
A: Unless otherwise specified, customers are ranked by total revenue (sum of
`total_amount` on their completed orders), not by order count.

**Q: Can I see employee salary information?**
A: Only if you are logged in with the Admin role. Manager and Employee roles
cannot access payroll data through this assistant.

**Q: What regions does the company operate in?**
A: Customer records are tagged with one of five regions: North, South, East,
West, Central.

**Q: How is "average order value" calculated?**
A: Total completed-order revenue divided by the count of completed orders.
See the `average_order_value` KPI definition for the exact formula.

**Q: Why did a query about "profit" not return cost or margin figures for me?**
A: Cost and margin data is restricted to Admin and Manager roles. Employee-role
users receive revenue/sales figures instead.
