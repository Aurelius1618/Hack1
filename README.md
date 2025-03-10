# Tap Bonds AI Layer

A comprehensive AI system for bond market analysis and information retrieval, designed to provide accurate financial insights and bond data through advanced natural language processing.

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Technical Implementation](#technical-implementation)
- [Installation](#installation)
- [Usage](#usage)
- [Agent Capabilities](#agent-capabilities)
- [Performance Optimization](#performance-optimization)
- [UI Features](#ui-features)
- [Performance Benchmarks](#performance-benchmarks)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🏗️ Architecture Overview

The Tap Bonds AI Layer is built on a hierarchical agent architecture using LangGraph for orchestration and Mistral-7B for reasoning. The system includes:

1. **Directory Agent**: Handles ISIN lookups and bond details
2. **Finder Agent**: Compares yields and finds bonds with specific returns
3. **Cashflow Agent**: Processes cash flow schedules and payment details
4. **Screener Agent**: Screens bonds and analyzes financial health
5. **Yield Calculator Agent**: Calculates YTM, bond prices, and duration metrics

## 🌟 Key Features

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

## 📁 Project Structure

```
tapbonds/
├── app/                           # Main application code
│   ├── agents/                    # Agent implementations
│   │   ├── directory_agent.py     # ISIN lookup and bond details
│   │   ├── finder_agent.py        # Yield comparison and bond finding
│   │   ├── cashflow_agent.py      # Cash flow processing
│   │   ├── screener_agent.py      # Bond screening and financial analysis
│   │   └── yield_calculator_agent.py # YTM and duration calculations
│   ├── api/                       # API endpoints
│   │   └── main.py                # FastAPI implementation
│   ├── core/                      # Core functionality
│   │   ├── workflow.py            # Agent orchestration workflow
│   │   └── pinecone_config.py     # Vector database configuration
│   ├── data/                      # Data processing modules
│   │   └── data_processor.py      # Data ingestion and transformation
│   ├── ui/                        # User interface components
│   │   ├── index.html             # Main web interface
│   │   ├── app.js                 # Frontend JavaScript
│   │   └── styles.css             # CSS styling
│   └── utils/                     # Utility functions
│       ├── document_processor.py  # Document parsing and chunking
│       ├── embeddings.py          # Vector embedding generation
│       ├── model_config.py        # ML model configuration
│       ├── monitoring.py          # Performance monitoring
│       └── optimization.py        # System optimization utilities
├── data_dump/                     # Sample data for testing
│   ├── bonds_details_202503011115.csv  # Bond reference data
│   ├── company_insights_202503011114.csv  # Company financial metrics
│   └── cashflows_202503011113.csv  # Bond cashflow schedules
├── logs/                          # Application logs
├── tests/                         # Test suite
│   ├── run_tests.py               # Test runner
│   ├── test_validation.py         # Validation tests
│   └── README.md                  # Testing documentation
├── .env                           # Environment variables (not in version control)
├── .env.example                   # Example environment variables
├── .gitignore                     # Git ignore file
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

## 💻 Technical Implementation

- **Model**: Mistral-7B with 4-bit quantization and LoRA adapters
- **Vector Database**: PineconeDB with hybrid search (BM25+ and dense embeddings)
- **State Management**: LangGraph for agent orchestration with hierarchical state
- **Data Processing**: Pandas and Dask for efficient data processing
- **UI**: Interactive interface with download capabilities and document preview

## 🚀 Installation

### Prerequisites

- Python 3.9+
- pip
- Git

### Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/tapbonds.git
cd tapbonds

# Create and activate a virtual environment (recommended)
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```

### Environment Variables

The following environment variables need to be set in your `.env` file:

```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
MODEL_PATH=path_to_your_model
OPENAI_API_KEY=your_openai_api_key  # If using OpenAI models
LOG_LEVEL=INFO
```

## 🔍 Usage

### Python API

```python
from app.core.workflow import workflow

# Process a query
result = workflow.invoke({
    "query": "What is the yield for ISIN INE123456789?"
})

print(result["agent_results"])
```

### Command Line Interface

```bash
# Run the CLI interface
python -m app.cli --query "What is the yield for ISIN INE123456789?"
```

### Web Interface

```bash
# Start the web server
python -m app.api.server

# Access the web interface at http://localhost:8000
```

## 🤖 Agent Capabilities

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

## ⚡ Performance Optimization

- LRU caching for frequent lookups
- 4-bit quantization for reduced memory usage (62% VRAM reduction)
- Parallel processing for cashflow calculations
- Optimized Pinecone configuration for hybrid search
- Entity resolution with fuzzy matching

## 🖥️ UI Features

- Interactive bond data visualization
- Download capabilities for tabular data (CSV/Excel/JSON)
- Document preview pane for PDFs
- Interactive date picker for maturity filters
- Responsive design for mobile and desktop

## 📊 Performance Benchmarks

| Metric               | Current | Target  |
|----------------------|---------|---------|
| ISIN Lookup Latency  | 320ms   | <200ms  |
| Yield Calc Accuracy  | ±0.5%   | ±0.1%   |
| Cash Flow Error Rate | 1.2%    | <0.5%   |

## 🧪 Testing

To run the test suite:

```bash
python -m tests.run_tests
```

The test suite covers:
- ISIN validation
- Yield calculations
- Financial ratio sanity checks
- Cashflow schedule calculations
- Workflow query routing
- API response format
- Performance metrics

For more details, see the [tests/README.md](tests/README.md) file.

## 🔧 Troubleshooting

### Common Issues

1. **PineconeDB Connection Issues**
   - Ensure your Pinecone API key is correct
   - Check if your Pinecone index exists and is properly configured

2. **Model Loading Errors**
   - Verify the model path in your .env file
   - Ensure you have enough RAM/VRAM for model loading

3. **Dependency Conflicts**
   - Try creating a fresh virtual environment
   - Update all dependencies with `pip install -r requirements.txt --upgrade`

### Logging

Logs are stored in the `logs/` directory. Set the `LOG_LEVEL` environment variable to control verbosity (DEBUG, INFO, WARNING, ERROR).

## 👥 Contributing

We welcome contributions to the Tap Bonds AI Layer! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code passes all tests and follows our coding standards.

## 📄 License

MIT
