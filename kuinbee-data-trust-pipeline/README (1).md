# KDTS Automation System

**Kuinbee Data Trust Score (KDTS)** - Production-grade system for quantifying dataset credibility, safety, and usability.

## Overview

KDTS is a comprehensive scoring system that evaluates datasets across five critical dimensions:
- **Quality (30%)**: Schema integrity, completeness, accuracy, uniqueness, distribution health
- **Legal & Compliance (25%)**: Ownership, resale permissions, PII risk, jurisdiction fit
- **Provenance (20%)**: Collection methodology, source credibility, transformation lineage, bias disclosure
- **Usability (15%)**: Joinability, documentation, delivery readiness, integration ease
- **Freshness (10%)**: Update reliability, latency, historical depth

Final scores range from 0-100 and map to trust bands:
- **85-100**: Production-Grade (safe for ML, analytics, operations)
- **70-84**: Business-Ready (good for decision support)
- **55-69**: Experimental (research/exploration only)
- **<55**: Restricted (not recommended)

## Key Features

✅ **Production-Ready**: Handles datasets from 1K to 100M+ rows  
✅ **No Database Required**: File-based processing for easy deployment  
✅ **Multi-Format Support**: CSV, Parquet, Excel, JSON  
✅ **Hard Legal Gate**: Auto-rejects datasets with L < 60  
✅ **Automated PII Detection**: Identifies direct and indirect identifiers  
✅ **Comprehensive Reporting**: JSON, Markdown, and PDF trust cards  
✅ **Highly Configurable**: YAML-based threshold customization  
✅ **Performance Optimized**: Chunked processing for large files  

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd kdts-automation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Basic Usage

```bash
# Score a single dataset
python -m src.cli score data/input/dataset.csv --output results/

# Batch process multiple datasets
python -m src.cli batch data/input/ --output results/ --workers 4

# Use different modes
python -m src.cli score dataset.csv --mode deep  # Thorough analysis
python -m src.cli score dataset.csv --mode quick  # Fast basic checks
```

### Python API

```python
from kdts_automation import KDTSPipeline
import pandas as pd

# Load dataset
df = pd.read_csv('dataset.csv')

# Prepare metadata
metadata = {
    'dataset_name': 'Urban Property Transactions',
    'expected_schema': {...},
    'critical_fields': ['property_id', 'price', 'location'],
    'accuracy_rules': {...},
    'license_info': {...}
}

# Calculate KDTS
pipeline = KDTSPipeline(mode='standard')
result = pipeline.process_dataframe(df, metadata)

# Access results
print(f"KDTS Score: {result.kdts_score}")
print(f"Band: {result.band}")
print(f"Quality: {result.quality_score}")
print(f"Legal: {result.legal_score}")

# Generate reports
result.save_json('trust_card.json')
result.save_markdown('trust_card.md')
result.save_pdf('trust_card.pdf')
```

## Project Structure

```
kdts-automation/
├── src/
│   ├── config/           # Configuration management
│   ├── core/             # Base classes and utilities
│   ├── ingestion/        # Data loading and profiling
│   ├── quality/          # Quality score calculators
│   ├── legal/            # Legal & compliance checks
│   ├── provenance/       # Provenance scoring
│   ├── usability/        # Usability assessment
│   ├── freshness/        # Freshness evaluation
│   ├── calculator/       # KDTS aggregation
│   ├── output/           # Report generation
│   ├── pipeline.py       # Main orchestrator
│   └── cli.py            # Command-line interface
├── tests/                # Test suite
├── config/               # Configuration files
│   └── thresholds.yaml   # Scoring thresholds
├── data/
│   ├── input/           # Input datasets
│   ├── output/          # Generated reports
│   └── temp/            # Temporary processing files
├── logs/                # Application logs
├── docs/                # Documentation
├── requirements.txt     # Python dependencies
├── setup.py            # Package setup
└── README.md           # This file
```

## Configuration

### Environment Variables (.env)

```bash
# Environment
ENVIRONMENT=production
DEBUG=false

# Paths
DATA_INPUT_DIR=data/input
DATA_OUTPUT_DIR=data/output
DATA_TEMP_DIR=data/temp
LOG_DIR=logs

# Processing
CHUNK_SIZE=50000
MAX_WORKERS=4
USE_POLARS=true

# Performance
MEMORY_LIMIT_GB=4
TIMEOUT_SECONDS=3600

# Logging
LOG_LEVEL=INFO
LOG_ROTATION=1 day
LOG_RETENTION=30 days
```

### Threshold Configuration (config/thresholds.yaml)

```yaml
quality:
  weights:
    schema: 0.25
    completeness: 0.25
    accuracy: 0.20
    uniqueness: 0.15
    distribution: 0.15

legal:
  hard_gate: 60
  weights:
    ownership: 0.40
    resale: 0.30
    pii_risk: 0.20
    jurisdiction: 0.10

# See config/thresholds.yaml for complete configuration
```

## Metadata Specification

Provide metadata to improve scoring accuracy:

```json
{
  "dataset_name": "Urban Property Transactions",
  "source_type": "licensed",
  "expected_schema": {
    "columns": {
      "property_id": {"type": "string", "nullable": false},
      "price": {"type": "float64", "nullable": false},
      "area_sqft": {"type": "int64", "nullable": false}
    },
    "non_nullable": ["property_id", "price"]
  },
  "critical_fields": ["property_id", "price", "location"],
  "primary_keys": ["property_id"],
  "accuracy_rules": {
    "price": {"min": 0, "max": 100000000},
    "area_sqft": {"min": 100, "max": 100000}
  },
  "license_info": {
    "has_written_agreement": true,
    "sub_licensing_allowed": true,
    "commercial_use_allowed": true
  },
  "compliance": {
    "dpdp_compliant": true,
    "it_act_compliant": true
  },
  "provenance": {
    "collection_method": "broker_aggregation",
    "reconciliation_performed": true,
    "etl_documented": true
  },
  "update_frequency": "quarterly"
}
```

## Output: Trust Card

The system generates a comprehensive trust card:

```json
{
  "dataset_name": "Urban Property Transactions",
  "kdts_score": 94.50,
  "band": "Production-Grade",
  "legal_passed": true,
  
  "component_scores": {
    "quality": 96.36,
    "legal": 98.10,
    "provenance": 90.85,
    "usability": 91.60,
    "freshness": 91.50
  },
  
  "quality_breakdown": {
    "schema": 97.5,
    "completeness": 96.0,
    "accuracy": 97.0,
    "uniqueness": 98.5,
    "distribution": 92.0
  },
  
  "legal_breakdown": {
    "ownership": 100.0,
    "resale": 95.0,
    "pii_risk": 98.0,
    "jurisdiction": 100.0
  },
  
  "known_limitations": [
    "Tier-2 cities underrepresented",
    "Minor distribution skew detected"
  ],
  
  "intended_use_cases": [
    "ML model training",
    "Market analysis",
    "Price prediction"
  ],
  
  "warnings": [],
  "blocking_issues": []
}
```

## Development

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific module
pytest tests/test_quality_scorer.py

# Performance tests
pytest -m slow
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/
pylint src/

# Type checking
mypy src/
```

### Performance Profiling

```bash
# Memory profiling
python -m memory_profiler src/pipeline.py

# CPU profiling
py-spy record -o profile.svg -- python -m src.cli score large_dataset.csv
```

## Performance Benchmarks

Tested on 16GB RAM, 8-core CPU:

| Dataset Size | Rows | Processing Time | Memory Usage |
|-------------|------|-----------------|--------------|
| Small | 10K | < 5 sec | < 100 MB |
| Medium | 1M | < 60 sec | < 500 MB |
| Large | 10M | < 10 min | < 2 GB |
| XLarge | 100M | < 2 hours | < 4 GB |

## Troubleshooting

### Common Issues

**1. Memory Error on Large Files**
```bash
# Use chunked processing
python -m src.cli score large.csv --mode quick

# Or increase chunk size
export CHUNK_SIZE=100000
```

**2. Slow Processing**
```bash
# Enable Polars for faster processing
export USE_POLARS=true

# Increase workers for batch processing
python -m src.cli batch input/ --workers 8
```

**3. PII Detection False Positives**
```yaml
# Adjust patterns in config/thresholds.yaml
legal:
  pii_patterns:
    phone: '\+?91[-\s]?\d{10}'  # More strict pattern
```

## Validation

Test against the provided example dataset:

```bash
# Should produce KDTS = 94.50
python -m src.cli validate

# Expected output:
# ✅ Quality Score: 96.36
# ✅ Legal Score: 98.10
# ✅ Provenance Score: 90.85
# ✅ Usability Score: 91.60
# ✅ Freshness Score: 91.50
# ✅ Final KDTS: 94.50 (Production-Grade)
```

## Roadmap

### Phase 1 (Completed) ✅
- [x] Core scoring modules
- [x] CLI interface
- [x] Basic reporting
- [x] Unit tests

### Phase 2 (Planned)
- [ ] Web API (FastAPI)
- [ ] Real-time dashboard
- [ ] Database integration
- [ ] Historical trend analysis

### Phase 3 (Future)
- [ ] ML-based anomaly detection
- [ ] Automated remediation suggestions
- [ ] Integration with data catalogs
- [ ] Multi-language support

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

[Your License Here]

## Contact

- **CTO**: [Your contact]
- **Issues**: [GitHub Issues URL]
- **Slack**: #kdts-automation

## Acknowledgments

- Based on Kuinbee's Dataset Credibility Checks (DCC) framework
- Inspired by data marketplace best practices
- Built for production-grade data quality assessment

---

**Built with ❤️ for the Kuinbee Data Marketplace**
