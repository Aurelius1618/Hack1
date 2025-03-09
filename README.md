# Tap Bonds AI Layer

A comprehensive AI system for bond market analysis and information retrieval.

## Architecture Overview

The Tap Bonds AI Layer is built on a hierarchical agent architecture using LangGraph for orchestration and Mistral-7B for reasoning. The system includes:

1. **Directory Agent**: Handles ISIN lookups and bond details
2. **Finder Agent**: Compares yields and finds bonds with specific returns
3. **Cashflow Agent**: Processes cash flow schedules and payment details
4. **Screener Agent**: Screens bonds and analyzes financial health

## Key Features

- **Hybrid Search**: Combines dense and sparse vectors for improved retrieval
- **Financial Reasoning**: Uses Contrastive Chain-of-Thought methodology
- **ISIN Validation**: Validates ISINs and handles issuer mismatches
- **Yield Normalization**: Normalizes yields across different platforms
- **Day Count Conventions**: Supports multiple day count conventions (30/360, ACT/ACT, etc.)
- **Z-Score Calculation**: Calculates Altman Z-Score for financial health assessment
- **Weighted Average Life**: Calculates WAL for bonds

## Technical Implementation

- **Model**: Mistral-7B with 4-bit quantization and LoRA adapters
- **Vector Database**: PineconeDB with hybrid search (BM25+ and dense embeddings)
- **State Management**: LangGraph for agent orchestration
- **Data Processing**: Pandas and Dask for efficient data processing

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tapbonds.git
cd tapbonds

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Usage

```python
from app.core.workflow import workflow

# Process a query
result = workflow.invoke({
    "query": "What is the yield for ISIN INE123456789?"
})

print(result["agent_results"])
```

## Agent Capabilities

### Directory Agent
- ISIN lookup with document validation
- Maturity date filtering
- Security detail resolution
- ISIN-issuer mismatch handling

### Finder Agent
- Yield comparison across platforms
- Platform reliability scoring
- Normalized yield calculations

### Cashflow Agent
- Day count convention calculations
- Accrued interest calculations
- Cashflow schedule analysis

### Screener Agent
- Financial health assessment
- Z-Score calculations
- Credit risk analysis

## Performance Optimization

- LRU caching for frequent lookups
- 4-bit quantization for reduced memory usage
- Parallel processing for cashflow calculations

## License

MIT
