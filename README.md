
# BudgetFly AI

An AI-powered flight price analysis and travel
recommendation system built using Python.

## Overview

BudgetFly AI combines historical flight pricing data,
live flight search results, and LLM-generated insights
to help users identify affordable flight options.

## Key Features

- Analyzes historical flight prices
- Retrieves live flight offers using SerpApi
- Identifies the cheapest available flights
- Generates AI-assisted travel recommendations
- Validates flight prices and recommendation accuracy

## Technologies Used

- Python
- Pandas
- SerpApi (Google Flights)
- OpenAI API
- Data Analytics
- Large Language Models (LLMs)

## Project Files

- budgetfly_ai_public.py - Main application
- accuracy_analyzer.py - Output accuracy validation
- flights.csv - Historical flight dataset
- requirements.txt - Python dependencies

## Installation

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   pip install -r requirements.txt

4. Create a .env file and configure your API keys.
5. Run the application:

   python budgetfly_ai_public.py

## Security

API keys and environment variables are excluded
from the public repository.
