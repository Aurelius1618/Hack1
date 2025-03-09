# Tap Bonds AI Layer

A comprehensive AI system for bond market analysis and information retrieval.

## Architecture Overview

The Tap Bonds AI Layer is built on a hierarchical agent architecture using LangGraph for orchestration and Mistral-7B for reasoning. The system includes:

1. **Directory Agent**: Handles ISIN lookups and bond details
2. **Finder Agent**: Compares yields and finds bonds with specific returns
3. **Cashflow Agent**: Processes cash flow schedules and payment details
4. **Screener Agent**: Screens bonds and analyzes financial health
5. **Yield Calculator Agent**: Calculates YTM, bond prices, and duration metrics

## Key Features

- **Hybrid Search**: Combines dense and sparse vectors for improved retrieval
- **Financial Reasoning**: Uses Contrastive Chain-of-Thought methodology
- **ISIN Validation**: Validates ISINs and handles issuer mismatches
- **Yield Normalization**: Normalizes yields across different platforms
- **Day Count Conventions**: Supports multiple day count conventions (30/360, ACT/ACT, ACT/360, ACT/365)
- **Z-Score Calculation**: Calculates Altman Z-Score for financial health assessment
- **Weighted Average Life**: Calculates WAL for bonds
- **SHAP Explanations**: Provides explainable AI for financial health assessments
- **YTM Calculation**: Uses Newton-Raphson method for accurate yield calculations
- **Duration Metrics**: Calculates Macaulay and Modified Duration

## Technical Implementation

- **Model**: Mistral-7B with 4-bit quantization and LoRA adapters
- **Vector Database**: PineconeDB with hybrid search (BM25+ and dense embeddings)
- **State Management**: LangGraph for agent orchestration with hierarchical state
- **Data Processing**: Pandas and Dask for efficient data processing
- **UI**: Interactive interface with download capabilities and document preview

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
- ISIN-issuer mismatch handling with fuzzy matching
- Document chunking strategy for legal docs and financial terms

### Finder Agent
- Yield comparison across platforms
- Platform reliability scoring with weighted matrix
- Normalized yield calculations (BEY)
- APY normalization formula

### Cashflow Agent
- Day count convention calculations (30/360, ACT/ACT, ACT/360, ACT/365)
- Accrued interest calculations
- Cashflow schedule analysis
- SEC-compliant day count conventions

### Screener Agent
- Financial health assessment
- Altman Z-Score calculations
- Credit risk analysis
- SHAP explanations for model predictions
- Feature importance visualization

### Yield Calculator Agent
- Yield to Maturity (YTM) calculation using Newton-Raphson method
- Clean and dirty price calculations
- Macaulay and Modified Duration calculations
- Bond pricing with Excel formula parity

## Performance Optimization

- LRU caching for frequent lookups
- 4-bit quantization for reduced memory usage (62% VRAM reduction)
- Parallel processing for cashflow calculations
- Optimized Pinecone configuration for hybrid search
- Entity resolution with fuzzy matching

## UI Features

- Interactive bond data visualization
- Download capabilities for tabular data (CSV/Excel/JSON)
- Document preview pane for PDFs
- Interactive date picker for maturity filters
- Responsive design for mobile and desktop

## Performance Benchmarks

| Metric               | Current | Target  |
|----------------------|---------|---------|
| ISIN Lookup Latency  | 320ms   | <200ms  |
| Yield Calc Accuracy  | ±0.5%   | ±0.1%   |
| Cash Flow Error Rate | 1.2%    | <0.5%   |

## License

MIT
