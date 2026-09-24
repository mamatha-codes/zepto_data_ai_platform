-- Query 1: SELECT + WHERE
SELECT title, price_gbp
FROM books
WHERE price_gbp > 50;


-- Query 2: ORDER BY
SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC;


-- Query 3: LIMIT
SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC
LIMIT 10;


-- Query 4: DISTINCT
SELECT DISTINCT category_name
FROM categories
ORDER BY category_name;


-- Query 5: BETWEEN
SELECT title, price_gbp, price_inr
FROM books
WHERE price_gbp BETWEEN 20 AND 40;


-- Query 6: JOIN
SELECT
    b.title,
    b.price_gbp,
    c.category_name
FROM books b
JOIN categories c
    ON b.category_id = c.category_id;