# Data Pipeline

## Overview

This module scrapes book data from Books to Scrape, cleans and transforms the data, stores it in a normalized SQLite database, and validates SQL results using pandas.

## Data Source

Website: Books to Scrape

Scraping scope:
- First 4 paginated product pages
- 20 books per page
- Total: 80 books

## Scraped Fields

- title
- price_gbp
- star_rating
- availability
- category

## Data Cleaning

The following transformations were applied:

- `price_gbp` was converted from currency text to float.
- `star_rating` was converted from text (`One` to `Five`) to integers (`1` to `5`).
- `availability` was converted into the boolean `in_stock`.
- Original availability text was retained.
- `price_inr` was calculated using the project-required fixed conversion rate:

```text
price_inr = price_gbp * 105.50
No external currency API was used.

Database Design

SQLite database: books.db

categories table
category_id — Primary Key
category_name — Unique category name
books table
book_id — Primary Key
title
price_gbp
star_rating
availability
in_stock
price_inr
category_id — Foreign Key referencing categories.category_id

The schema is normalized by storing categories separately and connecting them to books using a foreign key.

SQL Analysis

The project includes SQL queries demonstrating:

SELECT with WHERE
ORDER BY
LIMIT
DISTINCT
BETWEEN
JOIN

SQL queries and their outputs are stored in:

query_outputs.txt

Pandas Validation

SQL results were read using pandas.read_sql().

The SQL JOIN between books and categories was reproduced using pandas.merge().

The SQL JOIN and pandas JOIN results were compared and found to be equivalent.

Files
pipeline.py — scraping, cleaning, transformation, and database loading
books.db — SQLite database
queries.sql — SQL queries
query_outputs.txt — executed query strings and outputs
run_queries.py — executes SQL queries
validate_queries.py — validates SQL and pandas JOIN results
requirements.txt — Python dependencies
How to Run

Install dependencies:

py -m pip install -r data_pipeline/requirements.txt

Run the complete pipeline:

py data_pipeline/pipeline.py

Run SQL queries:

py data_pipeline/run_queries.py

Validate SQL and pandas JOIN results:

py data_pipeline/validate_queries.py