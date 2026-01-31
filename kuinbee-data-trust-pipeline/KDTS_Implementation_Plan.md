# KDTS Automation: 12-Day Implementation Plan

**Project**: Kuinbee Data Trust Score (KDTS) Automation System  
**Tech Stack**: Python 3.9+, No Database Required (File-based processing)  
**Target**: Production-grade application capable of processing large datasets  
**Team**: Interns under CTO guidance

---

## System Architecture Overview

```
Input Dataset → Ingestion Pipeline → 5 Score Calculators → Final KDTS → Trust Card Output
```

### Core Components:
1. **Data Ingestion Engine** - Schema validation, profiling
2. **Quality Score Module (Q)** - 5 sub-scores
3. **Legal & Compliance Module (L)** - Hard gate checks
4. **Provenance Score Module (P)** - Source credibility
5. **Usability Score Module (U)** - Buyer readiness
6. **Freshness Score Module (F)** - Temporal validity
7. **KDTS Calculator** - Final score aggregation
8. **Trust Card Generator** - Buyer-facing output

---

## Tech Stack Details

### Core Libraries:
- **pandas** (>=2.0.0) - Data manipulation
- **numpy** (>=1.24.0) - Numerical operations
- **great_expectations** (>=0.18.0) - Data validation framework
- **pydantic** (>=2.0.0) - Schema validation & settings
- **scipy** (>=1.11.0) - Statistical tests (PSI, Z-scores)
- **jsonschema** (>=4.19.0) - JSON validation
- **python-dotenv** (>=1.0.0) - Configuration
- **pyyaml** (>=6.0) - Config files
- **pytest** (>=7.4.0) - Testing

### Optional but Recommended:
- **polars** (>=0.19.0) - Fast dataframe operations for large files
- **pyarrow** (>=14.0.0) - Parquet file handling
- **openpyxl** (>=3.1.0) - Excel file reading
- **tabulate** (>=0.9.0) - Report formatting

---

## 12-Day Implementation Plan

---

## **DAY 1: Project Setup & Core Architecture**

### Tasks:
1. **Repository Structure**
   ```
   kdts-automation/
   ├── src/
   │   ├── __init__.py
   │   ├── config/
   │   │   ├── __init__.py
   │   │   ├── settings.py
   │   │   └── thresholds.yaml
   │   ├── core/
   │   │   ├── __init__.py
   │   │   ├── base_scorer.py
   │   │   └── exceptions.py
   │   ├── ingestion/
   │   ├── quality/
   │   ├── legal/
   │   ├── provenance/
   │   ├── usability/
   │   ├── freshness/
   │   ├── calculator/
   │   └── output/
   ├── tests/
   ├── data/
   │   ├── input/
   │   ├── output/
   │   └── temp/
   ├── config/
   ├── logs/
   ├── requirements.txt
   ├── setup.py
   └── README.md
   ```

2. **Create Base Classes**
   - `BaseScorer` abstract class with common methods
   - `KDTSException` custom exception hierarchy
   - `DatasetMetadata` dataclass
   - `ScoreResult` dataclass

3. **Configuration Management**
   - `settings.py` with Pydantic BaseSettings
   - `thresholds.yaml` for scoring parameters
   - Environment variable support (.env)

4. **Logging Setup**
   - Structured logging with rotation
   - Different log levels for development/production
   - Performance metrics logging

### Deliverables:
- [ ] Complete project structure
- [ ] Base classes implemented
- [ ] Configuration system working
- [ ] Logging framework active
- [ ] README with setup instructions

### Code Example (base_scorer.py):
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any
import logging

@dataclass
class ScoreResult:
    score: float
    sub_scores: Dict[str, float]
    flags: list[str]
    metadata: Dict[str, Any]

class BaseScorer(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def calculate(self, df, metadata) -> ScoreResult:
        """Calculate score - must be implemented by subclasses"""
        pass
    
    def validate_input(self, df) -> bool:
        """Common validation logic"""
        pass
```

---

## **DAY 2: Data Ingestion Pipeline**

### Tasks:
1. **File Reader Factory**
   - Support CSV, Parquet, Excel, JSON
   - Automatic format detection
   - Chunked reading for large files (>1GB)
   - Memory-efficient processing

2. **Schema Detection & Validation**
   - Automatic type inference
   - Schema drift detection
   - Null constraint validation
   - Foreign key relationship checks

3. **Initial Profiling**
   - Row/column counts
   - Memory footprint estimation
   - Basic statistics (mean, median, std)
   - Missing value summary

4. **Metadata Extraction**
   - File size, format
   - Column types
   - Unique identifiers
   - Temporal columns detection

### Deliverables:
- [ ] Multi-format file reader
- [ ] Schema validator
- [ ] Profiler module
- [ ] Metadata extractor
- [ ] Unit tests (>80% coverage)

### Key Functions:
```python
def ingest_dataset(file_path: str, chunk_size: int = 50000):
    """Ingest dataset with chunking support"""
    
def detect_schema(df):
    """Auto-detect and validate schema"""
    
def profile_dataset(df) -> Dict:
    """Generate basic statistics"""
```

---

## **DAY 3: Quality Score Module - Part 1 (S, C, A)**

### Tasks:
1. **S - Schema Integrity (25%)**
   - Type consistency checks
   - Null constraint violations
   - Schema version comparison
   - Column count verification

2. **C - Completeness (25%)**
   - Weighted null calculation
   - Critical field identification
   - Row-level completeness scoring
   - Missing pattern detection

3. **A - Accuracy/Sanity (20%)**
   - Range validation (min/max)
   - Impossible value detection
   - Cross-field logic checks
   - Domain-specific validations

### Deliverables:
- [ ] Schema integrity checker
- [ ] Completeness calculator
- [ ] Accuracy validator
- [ ] Configuration file for rules
- [ ] Test suite

### Implementation Example:
```python
class SchemaIntegrityScorer:
    def calculate(self, df, expected_schema):
        violations = 0
        total_checks = 0
        
        # Type mismatches
        for col, expected_type in expected_schema.items():
            if df[col].dtype != expected_type:
                violations += 1
            total_checks += 1
        
        # Null constraint violations
        for col in expected_schema.get('non_nullable', []):
            if df[col].isnull().any():
                violations += 1
            total_checks += 1
        
        score = 100 * (1 - violations / total_checks)
        return score
```

---

## **DAY 4: Quality Score Module - Part 2 (U, D)**

### Tasks:
1. **U - Uniqueness (15%)**
   - Exact duplicate detection
   - Primary key uniqueness
   - Near-duplicate detection (fuzzy matching)
   - Composite key validation

2. **D - Distribution Health (15%)**
   - PSI (Population Stability Index) calculation
   - Z-score outlier detection
   - Distribution skewness analysis
   - Temporal consistency checks

3. **Quality Score Integration**
   - Combine all 5 sub-scores
   - Weight application
   - Final Q score calculation
   - Performance optimization

### Deliverables:
- [ ] Uniqueness detector
- [ ] Distribution analyzer
- [ ] PSI calculator
- [ ] Complete Q module
- [ ] Benchmark tests (process 1M rows < 30 sec)

### Key Algorithms:
```python
def calculate_psi(expected, actual, buckets=10):
    """Calculate Population Stability Index"""
    expected_hist, _ = np.histogram(expected, bins=buckets)
    actual_hist, _ = np.histogram(actual, bins=buckets)
    
    expected_pct = expected_hist / len(expected)
    actual_pct = actual_hist / len(actual)
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi

def detect_duplicates(df, subset=None):
    """Detect exact and fuzzy duplicates"""
    exact = df.duplicated(subset=subset).sum()
    # Fuzzy logic here
    return exact, fuzzy_count
```

---

## **DAY 5: Legal & Compliance Module (L)**

### Tasks:
1. **O - Ownership (40%)**
   - License file validation
   - Ownership declaration checks
   - Resale rights verification
   - Contract metadata parsing

2. **R - Resale Permission (30%)**
   - Sub-licensing flag checks
   - TOS compliance validation
   - Usage restriction parsing
   - Exclusivity claim verification

3. **P - PII Risk (20%)**
   - Direct identifier detection (phone, email, Aadhaar)
   - Indirect identifier analysis
   - Re-identification risk scoring
   - Anonymization quality check

4. **J - Jurisdiction Fit (10%)**
   - DPDP Act compliance checks
   - IT Act alignment
   - Cross-border transfer rules
   - Data localization requirements

5. **Hard Gate Implementation**
   - L < 60 → auto-reject
   - Legal flag system
   - Audit trail generation

### Deliverables:
- [ ] PII detection engine
- [ ] License parser
- [ ] Compliance checker
- [ ] Hard gate logic
- [ ] Legal audit logs

### PII Detection:
```python
import re

class PIIDetector:
    PATTERNS = {
        'phone': r'\+?91[-\s]?\d{10}',
        'email': r'[\w\.-]+@[\w\.-]+\.\w+',
        'aadhaar': r'\d{4}\s?\d{4}\s?\d{4}',
        'pan': r'[A-Z]{5}\d{4}[A-Z]',
    }
    
    def detect(self, df):
        pii_columns = []
        for col in df.select_dtypes(include=['object']):
            for pii_type, pattern in self.PATTERNS.items():
                if df[col].astype(str).str.contains(pattern).any():
                    pii_columns.append((col, pii_type))
        return pii_columns
```

---

## **DAY 6: Provenance Score Module (P)**

### Tasks:
1. **M - Methodology Clarity (30%)**
   - Collection method scoring
   - Documentation completeness
   - Sampling strategy validation
   - Reconciliation proof checking

2. **S - Source Type (25%)**
   - Primary vs secondary classification
   - Licensed vs scraped detection
   - Source credibility scoring
   - Multi-source reconciliation

3. **T - Transformation Lineage (25%)**
   - ETL documentation parsing
   - Cleaning step validation
   - Aggregation transparency
   - Version control tracking

4. **B - Bias Disclosure (20%)**
   - Known gap identification
   - Coverage limitation scoring
   - Sampling bias detection
   - Demographic representation analysis

### Deliverables:
- [ ] Methodology scorer
- [ ] Source classifier
- [ ] Lineage tracker
- [ ] Bias analyzer
- [ ] Provenance report generator

### Example Implementation:
```python
class ProvenanceScorer:
    SOURCE_SCORES = {
        'primary': 100,
        'licensed': 90,
        'aggregated': 75,
        'scraped': 40,
        'unknown': 0
    }
    
    def score_source_type(self, metadata):
        source_type = metadata.get('source_type', 'unknown')
        base_score = self.SOURCE_SCORES.get(source_type, 0)
        
        # Adjust for multi-source reconciliation
        if metadata.get('reconciliation_performed'):
            base_score += 5
        
        return min(100, base_score)
```

---

## **DAY 7: Usability Score Module (U) & Freshness Score Module (F)**

### Tasks - Usability:
1. **J - Joinability (30%)**
   - Common key detection (PIN, city codes)
   - Census data alignment
   - Public dataset compatibility
   - Foreign key validation

2. **N - Naming & Documentation (25%)**
   - Data dictionary completeness
   - Column name clarity
   - Metadata richness
   - Example data provided

3. **D - Delivery Readiness (25%)**
   - File format scoring (Parquet > CSV)
   - Query optimization
   - Index availability
   - Partitioning strategy

4. **I - Integration Ease (20%)**
   - API availability
   - Multiple format support
   - SDK/library provided
   - Sample code quality

### Tasks - Freshness:
1. **R - Refresh Reliability (50%)**
   - Update schedule adherence
   - Historical SLA compliance
   - Missed update tracking

2. **L - Latency (30%)**
   - Time since last update
   - Expected vs actual lag
   - Real-time scoring

3. **H - Historical Depth (20%)**
   - Time coverage span
   - Backfill completeness
   - Temporal granularity

### Deliverables:
- [ ] Joinability checker
- [ ] Documentation scorer
- [ ] Format analyzer
- [ ] Freshness calculator
- [ ] Combined U & F modules

---

## **DAY 8: KDTS Calculator & Aggregation Engine**

### Tasks:
1. **Score Aggregator**
   - Weighted combination logic
   - Hard gate enforcement (L ≥ 60)
   - Score normalization
   - Edge case handling

2. **KDTS Bands Classification**
   - Production-Grade (85-100)
   - Business-Ready (70-84)
   - Experimental (55-69)
   - Restricted (<55)

3. **Override Rules Implementation**
   - PII leak → auto-delist
   - Legal ambiguity → delist
   - False provenance → blacklist
   - Quality drift → downgrade

4. **Performance Optimization**
   - Parallel processing for large datasets
   - Caching for repeated calculations
   - Memory management
   - Progress tracking

### Deliverables:
- [ ] KDTS calculator
- [ ] Band classifier
- [ ] Override engine
- [ ] Performance benchmarks
- [ ] Integration tests

### Core Calculator:
```python
class KDTSCalculator:
    WEIGHTS = {
        'quality': 0.30,
        'legal': 0.25,
        'provenance': 0.20,
        'usability': 0.15,
        'freshness': 0.10
    }
    
    def calculate(self, scores: Dict[str, float]) -> Dict:
        # Hard gate check
        if scores['legal'] < 60:
            return {
                'kdts': 0,
                'band': 'REJECTED',
                'reason': 'Legal score below threshold'
            }
        
        # Calculate weighted score
        kdts = sum(
            scores[component] * self.WEIGHTS[component]
            for component in self.WEIGHTS
        )
        
        # Apply overrides
        kdts = self.apply_overrides(kdts, scores)
        
        return {
            'kdts': kdts,
            'band': self.get_band(kdts),
            'components': scores
        }
    
    def get_band(self, score):
        if score >= 85: return 'Production-Grade'
        if score >= 70: return 'Business-Ready'
        if score >= 55: return 'Experimental'
        return 'Restricted'
```

---

## **DAY 9: Trust Card Generator & Reporting**

### Tasks:
1. **Trust Card Structure**
   - JSON output format
   - Human-readable summary
   - Component breakdown
   - Flag/warning system

2. **Report Components**
   - Executive summary
   - Detailed score breakdown
   - Known limitations section
   - Intended use cases
   - Update frequency info
   - Compliance badges

3. **Visualization Generation**
   - Score spider chart
   - Component bar chart
   - Trend analysis (if historical data)
   - Distribution plots

4. **Export Formats**
   - JSON (machine-readable)
   - Markdown (documentation)
   - PDF (buyer presentation)
   - HTML (web display)

### Deliverables:
- [ ] TrustCard dataclass
- [ ] JSON serializer
- [ ] Markdown generator
- [ ] PDF exporter
- [ ] Sample outputs

### Trust Card Example:
```python
@dataclass
class TrustCard:
    dataset_name: str
    kdts_score: float
    band: str
    
    # Component scores
    quality_score: float
    legal_score: float
    provenance_score: float
    usability_score: float
    freshness_score: float
    
    # Sub-scores breakdown
    quality_breakdown: Dict[str, float]
    legal_breakdown: Dict[str, float]
    # ... etc
    
    # Metadata
    known_limitations: List[str]
    intended_use_cases: List[str]
    update_frequency: str
    legal_badges: List[str]
    
    # Flags
    warnings: List[str]
    blocking_issues: List[str]
    
    def to_json(self) -> str:
        """Export as JSON"""
        
    def to_markdown(self) -> str:
        """Generate markdown report"""
        
    def to_pdf(self, output_path: str):
        """Generate PDF report"""
```

---

## **DAY 10: End-to-End Pipeline Integration**

### Tasks:
1. **Main Pipeline Orchestrator**
   - Sequential execution of all modules
   - Error handling and recovery
   - Checkpoint/resume capability
   - Progress reporting

2. **Parallel Processing**
   - Multi-file batch processing
   - Thread pool for sub-calculations
   - Memory-efficient chunking
   - Resource management

3. **Configuration Profiles**
   - Quick mode (basic checks)
   - Standard mode (full scoring)
   - Deep mode (advanced validations)
   - Custom profiles

4. **CLI Interface**
   - Command-line argument parsing
   - Interactive mode
   - Batch mode
   - Configuration file support

### Deliverables:
- [ ] Pipeline orchestrator
- [ ] CLI interface
- [ ] Batch processor
- [ ] Configuration system
- [ ] End-to-end test

### CLI Example:
```python
# cli.py
import click
from kdts_automation import KDTSPipeline

@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', default='output.json')
@click.option('--mode', type=click.Choice(['quick', 'standard', 'deep']))
@click.option('--config', type=click.Path())
def score_dataset(input_file, output, mode, config):
    """Calculate KDTS for a dataset"""
    pipeline = KDTSPipeline(mode=mode, config_path=config)
    result = pipeline.process(input_file)
    result.save(output)
    click.echo(f"KDTS Score: {result.kdts_score} ({result.band})")
```

---

## **DAY 11: Testing, Validation & Edge Cases**

### Tasks:
1. **Unit Tests**
   - All scorer modules (>90% coverage)
   - Edge case handling
   - Invalid input rejection
   - Boundary conditions

2. **Integration Tests**
   - Full pipeline execution
   - Multi-format file handling
   - Large dataset processing (>10M rows)
   - Memory leak detection

3. **Edge Case Testing**
   - Empty datasets
   - Single-row datasets
   - All-null columns
   - Massive files (>50GB)
   - Corrupted data
   - Mixed encodings

4. **Performance Benchmarks**
   - Processing speed tests
   - Memory usage profiling
   - Scaling tests (1K to 100M rows)
   - Optimization identification

5. **Validation Against Examples**
   - Process the artificial dataset example
   - Verify scores match expected (94.5)
   - Cross-check all sub-scores

### Deliverables:
- [ ] Complete test suite (pytest)
- [ ] Performance benchmarks
- [ ] Edge case documentation
- [ ] Bug fixes
- [ ] Optimization report

### Test Structure:
```python
# tests/test_quality_scorer.py
import pytest
from kdts_automation import QualityScorer

def test_schema_integrity_perfect():
    """Test with perfect schema compliance"""
    df = create_test_df()
    scorer = QualityScorer()
    result = scorer.calculate(df)
    assert result.sub_scores['schema'] == 100.0

def test_schema_integrity_violations():
    """Test with known violations"""
    df = create_test_df_with_violations()
    scorer = QualityScorer()
    result = scorer.calculate(df)
    assert 0 <= result.sub_scores['schema'] < 100

@pytest.mark.slow
def test_large_dataset():
    """Test with 10M row dataset"""
    df = create_large_df(rows=10_000_000)
    scorer = QualityScorer()
    result = scorer.calculate(df)
    assert result.score > 0
```

---

## **DAY 12: Documentation, Deployment & Handoff**

### Tasks:
1. **Documentation**
   - Architecture documentation
   - API reference (auto-generated)
   - User guide with examples
   - Configuration guide
   - Troubleshooting guide
   - Contributing guide

2. **Deployment Preparation**
   - Docker container creation
   - Requirements freeze
   - Environment setup guide
   - CI/CD pipeline configuration
   - Monitoring setup (logs, metrics)

3. **Examples & Tutorials**
   - Jupyter notebooks with examples
   - Sample datasets (small, medium, large)
   - Use case walkthroughs
   - Integration examples

4. **Production Readiness**
   - Security review
   - Performance optimization
   - Error handling review
   - Logging enhancement
   - Health check endpoint

5. **Knowledge Transfer**
   - Code walkthrough session
   - Q&A documentation
   - Maintenance guide
   - Extension guide
   - Roadmap for future features

### Deliverables:
- [ ] Complete documentation site
- [ ] Docker image
- [ ] Example notebooks
- [ ] Deployment guide
- [ ] Maintenance playbook
- [ ] Knowledge transfer session

### Documentation Structure:
```markdown
# KDTS Automation Documentation

## Quick Start
1. Installation
2. Basic Usage
3. Configuration

## Architecture
- System Overview
- Module Breakdown
- Data Flow

## API Reference
- Core Classes
- Scorers
- Utilities

## User Guide
- File Formats
- Configuration Options
- Output Formats
- Troubleshooting

## Developer Guide
- Setup Development Environment
- Running Tests
- Adding New Scorers
- Contributing Guidelines

## Deployment
- Docker Deployment
- Environment Variables
- Monitoring Setup
- Performance Tuning
```

---

## Success Criteria

### Functionality:
- ✅ Processes all supported file formats (CSV, Parquet, Excel, JSON)
- ✅ Calculates all 5 component scores accurately
- ✅ Enforces legal hard gate (L ≥ 60)
- ✅ Generates complete Trust Cards
- ✅ Handles edge cases gracefully

### Performance:
- ✅ Processes 1M rows in < 60 seconds
- ✅ Processes 10M rows in < 10 minutes
- ✅ Memory usage < 4GB for 100M row dataset
- ✅ Supports datasets up to 100GB

### Quality:
- ✅ Test coverage > 85%
- ✅ Zero critical bugs
- ✅ Matches example calculation (KDTS = 94.5)
- ✅ All edge cases documented and handled

### Production Readiness:
- ✅ Comprehensive error handling
- ✅ Structured logging
- ✅ Configuration management
- ✅ Docker deployment ready
- ✅ Complete documentation

---

## Risk Management

### Potential Risks:

1. **Performance Issues with Large Files**
   - Mitigation: Implement chunked processing, use Polars for speed
   - Fallback: Streaming processing mode

2. **Memory Overflow**
   - Mitigation: Memory profiling, garbage collection optimization
   - Fallback: Disk-based processing for huge datasets

3. **Complex PII Detection**
   - Mitigation: Use proven regex patterns, incremental improvements
   - Fallback: Conservative flagging (false positives > false negatives)

4. **Scoring Accuracy**
   - Mitigation: Extensive testing against known datasets
   - Fallback: Manual review mode for ambiguous cases

5. **Integration Complexity**
   - Mitigation: Clear interfaces, comprehensive tests
   - Fallback: Module-by-module delivery

---

## Post-Implementation Roadmap

### Phase 2 (Weeks 3-4):
- [ ] Web API wrapper (FastAPI)
- [ ] Async processing support
- [ ] Database integration for results storage
- [ ] Historical trend analysis

### Phase 3 (Weeks 5-8):
- [ ] Machine learning for anomaly detection
- [ ] Automated bias detection improvements
- [ ] Custom rule engine
- [ ] Real-time monitoring dashboard

### Phase 4 (Months 3-6):
- [ ] Multi-language support
- [ ] Advanced visualization
- [ ] Automated remediation suggestions
- [ ] Integration with data catalogs

---

## Daily Standups & Reviews

### Daily Format (15 min):
1. What was completed yesterday?
2. What's planned for today?
3. Any blockers?
4. Code review requests?

### End-of-Day:
- Commit all code to version control
- Update progress tracker
- Document any decisions made

### Weekly Review (Fridays):
- Demo completed features
- Review test coverage
- Performance benchmark review
- Next week planning

---

## Tools & Resources

### Development:
- **IDE**: VS Code / PyCharm
- **Version Control**: Git
- **Testing**: pytest, pytest-cov
- **Linting**: black, flake8, mypy
- **Profiling**: cProfile, memory_profiler

### Documentation:
- **API Docs**: Sphinx / mkdocs
- **Diagrams**: draw.io / mermaid
- **Notebooks**: Jupyter

### Monitoring:
- **Logging**: structlog
- **Metrics**: prometheus-client (optional)
- **Profiling**: py-spy

---

## Contact & Support

- **CTO**: [Your contact info]
- **Code Repository**: [Git URL]
- **Documentation**: [Docs URL]
- **Slack Channel**: #kdts-automation

---

## Appendix: Example Dataset Processing

Using the artificial dataset from the PDF:

```python
# Expected output for Urban India Residential Property Transactions
{
    "dataset_name": "Urban India Residential Property Transactions",
    "kdts_score": 94.50,
    "band": "Production-Grade",
    "quality_score": 96.36,
    "legal_score": 98.10,
    "provenance_score": 90.85,
    "usability_score": 91.60,
    "freshness_score": 91.50,
    "quality_breakdown": {
        "schema": 97.5,
        "completeness": 96.0,
        "accuracy": 97.0,
        "uniqueness": 98.5,
        "distribution": 92.0
    },
    # ... full breakdown
}
```

This serves as the validation test case for Day 11.
