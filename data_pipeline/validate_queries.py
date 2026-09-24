import sqlite3
import pandas as pd


DB_NAME = "books.db"

conn = sqlite3.connect(DB_NAME)


# -----------------------------------
# 1. Read SQL query result using pandas
# -----------------------------------

query_1 = """
SELECT title, price_gbp
FROM books
WHERE price_gbp > 50;
"""

result_1 = pd.read_sql(query_1, conn)

print("\nQuery 1 result using pd.read_sql:")
print(result_1.head())


# -----------------------------------
# 2. Read JOIN result using pd.read_sql
# -----------------------------------

join_query = """
SELECT
    b.title,
    b.price_gbp,
    c.category_name
FROM books b
JOIN categories c
    ON b.category_id = c.category_id;
"""

sql_join_result = pd.read_sql(join_query, conn)

print("\nSQL JOIN result:")
print(sql_join_result.head())


# -----------------------------------
# 3. Reproduce same JOIN using pandas
# -----------------------------------

books_df = pd.read_sql(
    """
    SELECT
        title,
        price_gbp,
        category_id
    FROM books;
    """,
    conn
)

categories_df = pd.read_sql(
    """
    SELECT
        category_id,
        category_name
    FROM categories;
    """,
    conn
)

pandas_join_result = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)

pandas_join_result = pandas_join_result[
    ["title", "price_gbp", "category_name"]
]


# -----------------------------------
# 4. Check equivalence
# -----------------------------------

sql_sorted = sql_join_result.sort_values(
    ["title", "price_gbp", "category_name"]
).reset_index(drop=True)

pandas_sorted = pandas_join_result.sort_values(
    ["title", "price_gbp", "category_name"]
).reset_index(drop=True)


are_equal = sql_sorted.equals(pandas_sorted)


print("\nPandas JOIN result:")
print(pandas_join_result.head())

print("\nSQL JOIN and Pandas JOIN are equivalent:", are_equal)


conn.close()