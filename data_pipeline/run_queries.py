import sqlite3
import pandas as pd


DB_NAME = "books.db"


conn = sqlite3.connect(DB_NAME)


queries = {
    "Query 1 - WHERE": """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp > 50;
    """,

    "Query 2 - ORDER BY": """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC;
    """,

    "Query 3 - LIMIT": """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
    """,

    "Query 4 - DISTINCT": """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name;
    """,

    "Query 5 - BETWEEN": """
        SELECT title, price_gbp, price_inr
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40;
    """,

    "Query 6 - JOIN": """
        SELECT
            b.title,
            b.price_gbp,
            c.category_name
        FROM books b
        JOIN categories c
            ON b.category_id = c.category_id;
    """
}


with open("data_pipeline/query_outputs.txt", "w", encoding="utf-8") as output_file:

    for name, query in queries.items():

        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)

        result = pd.read_sql(query, conn)

        print(result.to_string(index=False))

        output_file.write("\n" + "=" * 60 + "\n")
        output_file.write(name + "\n")
        output_file.write("=" * 60 + "\n")
        output_file.write(query.strip() + "\n\n")
        output_file.write(result.to_string(index=False))
        output_file.write("\n\n")


conn.close()

print("\nAll query strings and outputs saved to query_outputs.txt")