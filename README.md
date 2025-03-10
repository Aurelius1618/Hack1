# Tap Bonds AI Layer

An AI-powered bond analysis system using LangGraph and Lamini finetuned models.

## Overview

Tap Bonds AI Layer is a sophisticated system for analyzing bond data, providing insights, and answering queries about bonds. It leverages Lamini's finetuned models to provide accurate and domain-specific responses.

## Features

- **Bond Directory**: Look up specific bonds by ISIN
- **Yield Comparison**: Compare yields across different bonds
- **Cash Flow Analysis**: Analyze payment schedules and cash flows
- **Bond Screening**: Screen bonds based on various criteria
- **Yield Calculation**: Calculate yield to maturity, bond prices, and duration

## Architecture

The system uses a LangGraph workflow to route queries to specialized agents:

1. **Orchestrator**: Classifies queries and routes them to the appropriate agent
2. **Directory Agent**: Handles queries about specific bonds
3. **Finder Agent**: Handles queries about comparing yields
4. **Cashflow Agent**: Handles queries about cash flow schedules
5. **Screener Agent**: Handles queries about screening bonds
6. **Yield Calculator Agent**: Handles queries about calculating yields

## Lamini Integration

The system uses Lamini's finetuned models for:

- Query classification
- Bond information summarization
- Financial reasoning
- Response generation

### Setting Up Lamini

1. Sign up for a Lamini account at [lamini.ai](https://lamini.ai)
2. Get your API key from the Lamini dashboard
3. Set the `LAMINI_API_KEY` and `FINETUNED_MODEL_ID` environment variables

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your API keys
4. Run the application:
   ```
   uvicorn app.api.main:app --reload
   ```

## API Endpoints

- `GET /`: Root endpoint
- `POST /api/query`: Process a natural language query
- `GET /api/bond/{isin}`: Get details for a specific bond
- `GET /api/cashflow/{isin}`: Get cash flow schedule for a bond
- `POST /api/compare`: Compare multiple bonds
- `POST /api/portfolio/analyze`: Analyze a portfolio of bonds
- `GET /api/screen`: Screen bonds based on criteria
- `GET /api/health`: Health check endpoint
- `GET /api/metrics`: Get system metrics
- `GET /api/report`: Generate a system report

## Testing

Run the tests with:

```
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
