# Zepto Data & AI Platform

Capstone Project — Certificate Program in Artificial Intelligence and Machine Learning

This repository contains three connected modules:

- **Data Pipeline** — Web scraping, data cleaning, currency conversion, SQLite database, and SQL analysis.
- **Analytics Pipeline** — Titanic EDA, preprocessing, classification, imbalance handling, hyperparameter tuning, regression, and model persistence.
- **Support Assistant** — Policy document retrieval using embeddings, ChromaDB, LangGraph, and FastAPI.

## Project Structure

```text
zepto-data-ai-platform/
│
├── data_pipeline/
├── analytics/
├── support_assistant/
└── README.md
Module 1 — Data Pipeline

The data pipeline scrapes book data from Books to Scrape using Requests and BeautifulSoup.

The pipeline:

Collects book information
Cleans price and rating values
Converts GBP prices to INR using the fixed rate 1 GBP = 105.50 INR
Creates a normalized SQLite database
Executes SQL queries for analysis
Compares SQL JOIN results with Pandas operations
Module 2 — Analytics

The analytics pipeline uses the Titanic dataset to perform:

Data profiling and cleaning
Exploratory Data Analysis
Missing-value handling
Outlier analysis
Correlation analysis
Classification using Logistic Regression, Decision Tree, and Random Forest
Class imbalance handling
Hyperparameter tuning
Fare regression
Model persistence using Joblib
Module 3 — Support Assistant

The support assistant uses project policy documents to provide retrieval-based answers.

Technologies used:

Python
Sentence Transformers
ChromaDB
LangGraph
FastAPI
Pydantic

The assistant retrieves relevant policy context before generating an answer.

The default configuration uses mock responses and does not require an external API key.

Notes

The policy documents in support_assistant/ are synthetic documents created for this capstone demonstration.