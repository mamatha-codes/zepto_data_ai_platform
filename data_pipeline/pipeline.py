import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
from urllib.parse import urljoin


BASE_URL = "https://books.toscrape.com/"
DB_NAME = "books.db"


def scrape_books():
    all_books = []

    # -----------------------------
    # 1. SCRAPE 4 PAGES = 80 BOOKS
    # -----------------------------
    for page in range(1, 5):

        if page == 1:
            url = BASE_URL
        else:
            url = f"{BASE_URL}catalogue/page-{page}.html"

        response = requests.get(url)

        print(f"Page {page} status:", response.status_code)

        soup = BeautifulSoup(response.text, "html.parser")

        books = soup.select("article.product_pod")

        print(f"Books found on page {page}:", len(books))

        for book in books:

            title = book.h3.a["title"]

            price = book.select_one(
                ".price_color"
            ).text.strip()

            rating = book.select_one(
                "p.star-rating"
            )["class"][1]

            availability = book.select_one(
                ".availability"
            ).text.strip()

            # Book detail page
            book_link = book.h3.a["href"]
            book_url = urljoin(url, book_link)

            book_response = requests.get(book_url)

            book_soup = BeautifulSoup(
                book_response.text,
                "html.parser"
            )

            # Category
            breadcrumb_items = book_soup.select(
                "ul.breadcrumb li"
            )

            if len(breadcrumb_items) >= 3:
                category = breadcrumb_items[-2].get_text(
                    strip=True
                )
            else:
                category = "Unknown"

            all_books.append({
                "title": title,
                "price_gbp": price,
                "star_rating": rating,
                "availability": availability,
                "category": category
            })

    # -----------------------------
    # 2. CREATE DATAFRAME
    # -----------------------------
    df = pd.DataFrame(all_books)

    # -----------------------------
    # 3. CLEAN PRICE
    # -----------------------------
    df["price_gbp"] = (
        df["price_gbp"]
        .str.replace("Â£", "", regex=False)
        .str.replace("£", "", regex=False)
        .astype(float)
    )

    # -----------------------------
    # 4. CLEAN RATING
    # -----------------------------
    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }

    df["star_rating"] = df["star_rating"].map(
        rating_map
    )

    # -----------------------------
    # 5. AVAILABILITY → BOOLEAN
    # -----------------------------
    df["in_stock"] = df["availability"].str.contains(
        "In stock",
        case=False,
        na=False
    )

    # -----------------------------
    # 6. GBP → INR
    # -----------------------------
    df["price_inr"] = df["price_gbp"] * 105.50

    print("\nTotal books scraped:", len(df))

    # -----------------------------
    # 7. CREATE SQLITE DATABASE
    # -----------------------------
    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    # Categories table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        )
    """)

    # Books table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL,
            star_rating INTEGER,
            availability TEXT,
            in_stock INTEGER,
            price_inr REAL,
            category_id INTEGER,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
    """)

    # -----------------------------
    # 8. INSERT CATEGORIES
    # -----------------------------
    categories = df["category"].dropna().unique()

    for category in categories:
        cursor.execute(
            """
            INSERT OR IGNORE INTO categories (category_name)
            VALUES (?)
            """,
            (category,)
        )

    # -----------------------------
    # 9. INSERT BOOKS
    # -----------------------------
    for _, row in df.iterrows():

        cursor.execute(
            """
            SELECT category_id
            FROM categories
            WHERE category_name = ?
            """,
            (row["category"],)
        )

        category_id = cursor.fetchone()[0]

        cursor.execute(
            """
            INSERT INTO books (
                title,
                price_gbp,
                star_rating,
                availability,
                in_stock,
                price_inr,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["title"],
                row["price_gbp"],
                row["star_rating"],
                row["availability"],
                int(row["in_stock"]),
                row["price_inr"],
                category_id
            )
        )

    conn.commit()

    # -----------------------------
    # 10. VERIFY DATABASE
    # -----------------------------
    book_count = cursor.execute(
        "SELECT COUNT(*) FROM books"
    ).fetchone()[0]

    category_count = cursor.execute(
        "SELECT COUNT(*) FROM categories"
    ).fetchone()[0]

    print("Books stored in database:", book_count)
    print("Categories stored:", category_count)

    conn.close()

    return df


if __name__ == "__main__":
    scrape_books()