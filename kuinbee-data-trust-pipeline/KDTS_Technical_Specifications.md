# KDTS Technical Specifications

## Module-by-Module Implementation Details

---

## 1. Data Ingestion Module

### File: `src/ingestion/file_reader.py`

```python
from typing import Union, Iterator
import pandas as pd
import polars as pl
from pathlib import Path

class FileReaderFactory:
    """Factory for creating appropriate file readers"""
    
    SUPPORTED_FORMATS = {
        '.csv': 'CSVReader',
        '.parquet': 'ParquetReader',
        '.xlsx': 'ExcelReader',
        '.json': 'JSONReader',
        '.tsv': 'TSVReader'
    }
    
    @staticmethod
    def create_reader(file_path: str, use_polars: bool = True):
        """Create appropriate reader based on file extension"""
        ext = Path(file_path).suffix.lower()
        reader_class = FileReaderFactory.SUPPORTED_FORMATS.get(ext)
        if not reader_class:
            raise ValueError(f"Unsupported format: {ext}")
        # Return reader instance
        
class ChunkedDataReader:
    """Memory-efficient chunked reading for large files"""
    
    def __init__(self, file_path: str, chunk_size: int = 50000):
        self.file_path = file_path
        self.chunk_size = chunk_size
    
    def read_chunks(self) -> Iterator[pd.DataFrame]:
        """Yield dataframe chunks"""
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size):
            yield chunk
    
    def read_with_sampling(self, sample_rate: float = 0.1) -> pd.DataFrame:
        """Read large file with sampling"""
        return pd.read_csv(
            self.file_path,
            skiprows=lambda i: i > 0 and random.random() > sample_rate
        )
```

### File: `src/ingestion/schema_detector.py`

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd

@dataclass
class ColumnSchema:
    name: str
    dtype: str
    nullable: bool
    unique_count: int
    null_percentage: float
    sample_values: List[str]

class SchemaDetector:
    """Automatic schema detection and validation"""
    
    NUMERIC_TYPES = ['int64', 'float64', 'int32', 'float32']
    STRING_TYPES = ['object', 'string']
    DATETIME_TYPES = ['datetime64[ns]', 'datetime64']
    
    def detect_schema(self, df: pd.DataFrame) -> Dict[str, ColumnSchema]:
        """Detect schema from dataframe"""
        schema = {}
        for col in df.columns:
            schema[col] = ColumnSchema(
                name=col,
                dtype=str(df[col].dtype),
                nullable=df[col].isnull().any(),
                unique_count=df[col].nunique(),
                null_percentage=df[col].isnull().mean() * 100,
                sample_values=df[col].dropna().head(5).astype(str).tolist()
            )
        return schema
    
    def validate_schema(self, df: pd.DataFrame, expected_schema: Dict) -> Dict:
        """Validate against expected schema"""
        violations = []
        
        # Column count check
        if len(df.columns) != len(expected_schema):
            violations.append({
                'type': 'column_count',
                'expected': len(expected_schema),
                'actual': len(df.columns)
            })
        
        # Type checks
        for col, expected_type in expected_schema.items():
            if col not in df.columns:
                violations.append({'type': 'missing_column', 'column': col})
                continue
            
            actual_type = str(df[col].dtype)
            if not self._types_compatible(actual_type, expected_type):
                violations.append({
                    'type': 'type_mismatch',
                    'column': col,
                    'expected': expected_type,
                    'actual': actual_type
                })
        
        return {
            'valid': len(violations) == 0,
            'violations': violations,
            'violation_rate': len(violations) / len(expected_schema)
        }
    
    def _types_compatible(self, actual: str, expected: str) -> bool:
        """Check if types are compatible"""
        # Implementation for type compatibility logic
        pass
```

### File: `src/ingestion/profiler.py`

```python
from typing import Dict, Any
import pandas as pd
import numpy as np

class DatasetProfiler:
    """Generate comprehensive dataset profile"""
    
    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate full profile"""
        return {
            'basic_stats': self._basic_statistics(df),
            'column_stats': self._column_statistics(df),
            'missing_analysis': self._missing_analysis(df),
            'correlation': self._correlation_analysis(df),
            'memory_usage': self._memory_analysis(df)
        }
    
    def _basic_statistics(self, df: pd.DataFrame) -> Dict:
        return {
            'row_count': len(df),
            'column_count': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'duplicate_rows': df.duplicated().sum(),
            'total_nulls': df.isnull().sum().sum()
        }
    
    def _column_statistics(self, df: pd.DataFrame) -> Dict:
        stats = {}
        for col in df.columns:
            col_data = df[col]
            stats[col] = {
                'dtype': str(col_data.dtype),
                'unique_count': col_data.nunique(),
                'null_count': col_data.isnull().sum(),
                'null_percentage': col_data.isnull().mean() * 100
            }
            
            # Numeric columns
            if pd.api.types.is_numeric_dtype(col_data):
                stats[col].update({
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'q25': col_data.quantile(0.25),
                    'q75': col_data.quantile(0.75)
                })
            
            # String columns
            elif pd.api.types.is_string_dtype(col_data):
                stats[col].update({
                    'avg_length': col_data.str.len().mean(),
                    'max_length': col_data.str.len().max(),
                    'most_common': col_data.mode().iloc[0] if len(col_data.mode()) > 0 else None
                })
        
        return stats
    
    def _missing_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyze missing patterns"""
        missing = df.isnull()
        return {
            'columns_with_missing': missing.any().sum(),
            'rows_with_missing': missing.any(axis=1).sum(),
            'total_missing': missing.sum().sum(),
            'missing_percentage': (missing.sum().sum() / df.size) * 100,
            'columns_all_missing': missing.all().sum(),
            'missing_by_column': missing.sum().to_dict()
        }
```

---

## 2. Quality Score Module

### File: `src/quality/quality_scorer.py`

```python
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np
from scipy import stats
from src.core.base_scorer import BaseScorer, ScoreResult

@dataclass
class QualityConfig:
    """Configuration for quality scoring"""
    weights: Dict[str, float] = None
    schema_violation_threshold: float = 0.05
    completeness_critical_fields: List[str] = None
    accuracy_rules: Dict = None
    uniqueness_threshold: float = 0.02
    psi_threshold: float = 0.1
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                'schema': 0.25,
                'completeness': 0.25,
                'accuracy': 0.20,
                'uniqueness': 0.15,
                'distribution': 0.15
            }

class QualityScorer(BaseScorer):
    """Main quality scorer - orchestrates all sub-scorers"""
    
    def __init__(self, config: QualityConfig):
        super().__init__(config)
        self.config = config
        self.schema_scorer = SchemaIntegrityScorer(config)
        self.completeness_scorer = CompletenessScorer(config)
        self.accuracy_scorer = AccuracyScorer(config)
        self.uniqueness_scorer = UniquenessScorer(config)
        self.distribution_scorer = DistributionScorer(config)
    
    def calculate(self, df: pd.DataFrame, metadata: Dict = None) -> ScoreResult:
        """Calculate overall quality score"""
        
        # Calculate sub-scores
        schema_score = self.schema_scorer.score(df, metadata)
        completeness_score = self.completeness_scorer.score(df, metadata)
        accuracy_score = self.accuracy_scorer.score(df, metadata)
        uniqueness_score = self.uniqueness_scorer.score(df, metadata)
        distribution_score = self.distribution_scorer.score(df, metadata)
        
        sub_scores = {
            'schema': schema_score,
            'completeness': completeness_score,
            'accuracy': accuracy_score,
            'uniqueness': uniqueness_score,
            'distribution': distribution_score
        }
        
        # Calculate weighted total
        total_score = sum(
            sub_scores[key] * self.config.weights[key]
            for key in self.config.weights
        )
        
        # Collect flags
        flags = []
        if schema_score < 90:
            flags.append("Schema violations detected")
        if completeness_score < 90:
            flags.append("Critical fields have missing data")
        
        return ScoreResult(
            score=total_score,
            sub_scores=sub_scores,
            flags=flags,
            metadata={'calculation_method': 'weighted_average'}
        )
```

### File: `src/quality/schema_integrity.py`

```python
class SchemaIntegrityScorer:
    """Schema integrity validation and scoring"""
    
    def score(self, df: pd.DataFrame, metadata: Dict) -> float:
        """Calculate schema integrity score (0-100)"""
        
        violations = 0
        total_checks = 0
        
        expected_schema = metadata.get('expected_schema', {})
        
        # Check 1: Column count
        total_checks += 1
        if len(df.columns) != len(expected_schema.get('columns', [])):
            violations += 1
        
        # Check 2: Data type consistency
        for col in df.columns:
            total_checks += 1
            expected_type = expected_schema.get('columns', {}).get(col, {}).get('type')
            actual_type = str(df[col].dtype)
            
            if expected_type and not self._type_matches(actual_type, expected_type):
                violations += 1
        
        # Check 3: Null constraints
        for col in expected_schema.get('non_nullable', []):
            total_checks += 1
            if col in df.columns and df[col].isnull().any():
                violations += 1
        
        # Check 4: Schema drift (if previous schema provided)
        if 'previous_schema' in metadata:
            drift_violations = self._detect_schema_drift(df, metadata['previous_schema'])
            violations += drift_violations
            total_checks += len(metadata['previous_schema'])
        
        # Calculate score
        if total_checks == 0:
            return 100.0
        
        violation_rate = violations / total_checks
        score = 100 * (1 - violation_rate)
        
        return max(0.0, min(100.0, score))
    
    def _type_matches(self, actual: str, expected: str) -> bool:
        """Check if types are compatible"""
        type_groups = {
            'integer': ['int64', 'int32', 'int16', 'int8'],
            'float': ['float64', 'float32'],
            'string': ['object', 'string'],
            'datetime': ['datetime64[ns]', 'datetime64']
        }
        
        for group, types in type_groups.items():
            if expected in types and actual in types:
                return True
        
        return actual == expected
    
    def _detect_schema_drift(self, df: pd.DataFrame, previous_schema: Dict) -> int:
        """Detect schema drift compared to previous version"""
        drift_count = 0
        
        # New columns
        new_cols = set(df.columns) - set(previous_schema.keys())
        drift_count += len(new_cols)
        
        # Removed columns
        removed_cols = set(previous_schema.keys()) - set(df.columns)
        drift_count += len(removed_cols)
        
        # Type changes
        for col in set(df.columns) & set(previous_schema.keys()):
            if str(df[col].dtype) != previous_schema[col]:
                drift_count += 1
        
        return drift_count
```

### File: `src/quality/completeness.py`

```python
class CompletenessScorer:
    """Calculate completeness score based on missing data"""
    
    def score(self, df: pd.DataFrame, metadata: Dict) -> float:
        """Calculate completeness score (0-100)"""
        
        # Get critical fields from metadata
        critical_fields = metadata.get('critical_fields', [])
        
        if not critical_fields:
            # If no critical fields specified, use all fields
            critical_fields = df.columns.tolist()
        
        # Calculate weighted null percentage
        total_weight = 0
        weighted_nulls = 0
        
        for col in critical_fields:
            if col not in df.columns:
                continue
            
            # Weight: 1.0 for critical, 0.5 for others
            weight = 1.0 if col in critical_fields else 0.5
            null_rate = df[col].isnull().mean()
            
            weighted_nulls += null_rate * weight
            total_weight += weight
        
        if total_weight == 0:
            return 100.0
        
        avg_null_rate = weighted_nulls / total_weight
        score = 100 * (1 - avg_null_rate)
        
        return max(0.0, min(100.0, score))
    
    def get_row_completeness(self, df: pd.DataFrame) -> pd.Series:
        """Calculate completeness for each row"""
        return (1 - df.isnull().mean(axis=1)) * 100
```

### File: `src/quality/accuracy.py`

```python
class AccuracyScorer:
    """Validate accuracy and sanity of values"""
    
    def score(self, df: pd.DataFrame, metadata: Dict) -> float:
        """Calculate accuracy score (0-100)"""
        
        accuracy_rules = metadata.get('accuracy_rules', {})
        
        invalid_count = 0
        total_checks = 0
        
        for col, rules in accuracy_rules.items():
            if col not in df.columns:
                continue
            
            col_data = df[col]
            
            # Range checks
            if 'min' in rules:
                invalid = (col_data < rules['min']).sum()
                invalid_count += invalid
                total_checks += len(col_data)
            
            if 'max' in rules:
                invalid = (col_data > rules['max']).sum()
                invalid_count += invalid
                total_checks += len(col_data)
            
            # Impossible values (domain-specific)
            if 'impossible_values' in rules:
                for impossible_val in rules['impossible_values']:
                    invalid = (col_data == impossible_val).sum()
                    invalid_count += invalid
                    total_checks += len(col_data)
            
            # Custom validators
            if 'validator' in rules:
                validator_func = rules['validator']
                invalid = (~col_data.apply(validator_func)).sum()
                invalid_count += invalid
                total_checks += len(col_data)
        
        # Cross-field logic checks
        if 'cross_field_rules' in metadata:
            cross_invalid = self._validate_cross_field_logic(
                df, metadata['cross_field_rules']
            )
            invalid_count += cross_invalid
            total_checks += len(df) * len(metadata['cross_field_rules'])
        
        if total_checks == 0:
            return 100.0
        
        invalid_rate = invalid_count / total_checks
        score = 100 * (1 - invalid_rate)
        
        return max(0.0, min(100.0, score))
    
    def _validate_cross_field_logic(self, df: pd.DataFrame, rules: List) -> int:
        """Validate cross-field relationships"""
        invalid_count = 0
        
        for rule in rules:
            field1, operator, field2 = rule['field1'], rule['op'], rule['field2']
            
            if field1 not in df.columns or field2 not in df.columns:
                continue
            
            if operator == '>':
                invalid_count += (df[field1] <= df[field2]).sum()
            elif operator == '<':
                invalid_count += (df[field1] >= df[field2]).sum()
            elif operator == '==':
                invalid_count += (df[field1] != df[field2]).sum()
            # Add more operators as needed
        
        return invalid_count
```

### File: `src/quality/uniqueness.py`

```python
class UniquenessScorer:
    """Detect and score duplicate records"""
    
    def score(self, df: pd.DataFrame, metadata: Dict) -> float:
        """Calculate uniqueness score (0-100)"""
        
        # Exact duplicates
        exact_duplicates = df.duplicated().sum()
        
        # Key-based duplicates
        primary_keys = metadata.get('primary_keys', [])
        key_duplicates = 0
        
        if primary_keys:
            key_duplicates = df.duplicated(subset=primary_keys).sum()
        
        # Fuzzy duplicates (expensive, only for smaller datasets)
        fuzzy_duplicates = 0
        if len(df) < 10000 and metadata.get('check_fuzzy_duplicates'):
            fuzzy_duplicates = self._detect_fuzzy_duplicates(df, metadata)
        
        total_duplicates = max(exact_duplicates, key_duplicates) + fuzzy_duplicates
        duplication_rate = total_duplicates / len(df)
        
        score = 100 * (1 - duplication_rate)
        
        return max(0.0, min(100.0, score))
    
    def _detect_fuzzy_duplicates(self, df: pd.DataFrame, metadata: Dict) -> int:
        """Detect near-duplicates using fuzzy matching"""
        # Implementation for fuzzy matching
        # This would use techniques like:
        # - Levenshtein distance for strings
        # - Approximate matching for numbers
        # - Record linkage libraries
        pass
```

### File: `src/quality/distribution.py`

```python
from scipy import stats
import numpy as np

class DistributionScorer:
    """Analyze distribution health"""
    
    def score(self, df: pd.DataFrame, metadata: Dict) -> float:
        """Calculate distribution health score (0-100)"""
        
        issues = []
        total_columns = 0
        
        for col in df.select_dtypes(include=[np.number]).columns:
            total_columns += 1
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            # Check 1: Z-score outliers
            z_scores = np.abs(stats.zscore(col_data))
            outlier_rate = (z_scores > 3).sum() / len(col_data)
            if outlier_rate > 0.05:  # More than 5% outliers
                issues.append(f"{col}: High outlier rate")
            
            # Check 2: Skewness
            skewness = stats.skew(col_data)
            if abs(skewness) > 2:  # Highly skewed
                issues.append(f"{col}: High skewness")
            
            # Check 3: Zero-variance
            if col_data.std() == 0:
                issues.append(f"{col}: Zero variance")
            
            # Check 4: PSI (if reference distribution provided)
            if 'reference_distribution' in metadata:
                psi = self._calculate_psi(
                    col_data,
                    metadata['reference_distribution'].get(col)
                )
                if psi > 0.1:  # Significant distribution shift
                    issues.append(f"{col}: Distribution shift (PSI={psi:.3f})")
        
        if total_columns == 0:
            return 100.0
        
        issue_rate = len(issues) / total_columns
        score = 100 * (1 - issue_rate)
        
        return max(0.0, min(100.0, score))
    
    def _calculate_psi(self, actual, expected, buckets=10):
        """Calculate Population Stability Index"""
        if expected is None or len(expected) == 0:
            return 0.0
        
        # Create bins
        bins = np.linspace(min(expected.min(), actual.min()),
                          max(expected.max(), actual.max()),
                          buckets + 1)
        
        # Calculate distributions
        expected_dist, _ = np.histogram(expected, bins=bins)
        actual_dist, _ = np.histogram(actual, bins=bins)
        
        # Convert to percentages
        expected_pct = expected_dist / len(expected)
        actual_pct = actual_dist / len(actual)
        
        # Avoid division by zero
        expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
        actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
        
        # Calculate PSI
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        
        return psi
```

---

## 3. Legal & Compliance Module

### File: `src/legal/legal_scorer.py`

```python
from src.core.base_scorer import BaseScorer, ScoreResult

class LegalScorer(BaseScorer):
    """Legal & compliance scoring with hard gate"""
    
    HARD_GATE_THRESHOLD = 60
    
    WEIGHTS = {
        'ownership': 0.40,
        'resale': 0.30,
        'pii_risk': 0.20,
        'jurisdiction': 0.10
    }
    
    def __init__(self, config):
        super().__init__(config)
        self.pii_detector = PIIDetector()
        self.license_validator = LicenseValidator()
    
    def calculate(self, df: pd.DataFrame, metadata: Dict) -> ScoreResult:
        """Calculate legal score with hard gate"""
        
        # Calculate sub-scores
        ownership_score = self._score_ownership(metadata)
        resale_score = self._score_resale_permission(metadata)
        pii_score = self._score_pii_risk(df, metadata)
        jurisdiction_score = self._score_jurisdiction_fit(metadata)
        
        sub_scores = {
            'ownership': ownership_score,
            'resale': resale_score,
            'pii_risk': pii_score,
            'jurisdiction': jurisdiction_score
        }
        
        # Calculate weighted score
        total_score = sum(
            sub_scores[key] * self.WEIGHTS[key]
            for key in self.WEIGHTS
        )
        
        # Hard gate check
        flags = []
        if total_score < self.HARD_GATE_THRESHOLD:
            flags.append("HARD GATE FAILURE: Legal score below threshold")
            flags.append("Dataset is NOT SELLABLE")
        
        # Auto-fail conditions
        if pii_score == 0:
            flags.append("CRITICAL: PII leak detected")
            total_score = 0
        
        return ScoreResult(
            score=total_score,
            sub_scores=sub_scores,
            flags=flags,
            metadata={'hard_gate_passed': total_score >= self.HARD_GATE_THRESHOLD}
        )
    
    def _score_ownership(self, metadata: Dict) -> float:
        """Score ownership clarity"""
        ownership_doc = metadata.get('ownership_document')
        if not ownership_doc:
            return 0.0
        
        score = 0.0
        
        # Has written agreement
        if ownership_doc.get('has_written_agreement'):
            score += 50
        
        # Clear resale rights
        if ownership_doc.get('resale_rights_explicit'):
            score += 30
        
        # Ownership verified
        if ownership_doc.get('ownership_verified'):
            score += 20
        
        return min(100.0, score)
    
    def _score_resale_permission(self, metadata: Dict) -> float:
        """Score resale permissions"""
        license_info = metadata.get('license_info', {})
        
        score = 0.0
        
        # Sub-licensing allowed
        if license_info.get('sub_licensing_allowed'):
            score += 60
        
        # No TOS violations
        if not license_info.get('tos_violations'):
            score += 30
        
        # Commercial use allowed
        if license_info.get('commercial_use_allowed'):
            score += 10
        
        return min(100.0, score)
    
    def _score_pii_risk(self, df: pd.DataFrame, metadata: Dict) -> float:
        """Score PII risk"""
        pii_results = self.pii_detector.detect(df)
        
        if pii_results['direct_identifiers']:
            return 0.0  # Auto-fail
        
        indirect_count = len(pii_results['indirect_identifiers'])
        
        if indirect_count == 0:
            return 100.0  # Fully anonymized
        elif indirect_count <= 2:
            return 85.0  # Low risk
        elif indirect_count <= 4:
            return 70.0  # Medium risk
        else:
            return 40.0  # High risk
    
    def _score_jurisdiction_fit(self, metadata: Dict) -> float:
        """Score jurisdiction compliance"""
        compliance = metadata.get('compliance', {})
        
        score = 0.0
        
        # DPDP Act compliant
        if compliance.get('dpdp_compliant'):
            score += 50
        
        # IT Act compliant
        if compliance.get('it_act_compliant'):
            score += 30
        
        # Data localization
        if compliance.get('data_localized'):
            score += 20
        
        return min(100.0, score)
```

### File: `src/legal/pii_detector.py`

```python
import re
from typing import Dict, List

class PIIDetector:
    """Detect personally identifiable information"""
    
    # Indian-specific patterns
    PATTERNS = {
        'phone': r'\+?91[-\s]?\d{10}|\d{10}',
        'email': r'[\w\.-]+@[\w\.-]+\.\w+',
        'aadhaar': r'\d{4}\s?\d{4}\s?\d{4}',
        'pan': r'[A-Z]{5}\d{4}[A-Z]',
        'passport': r'[A-Z]\d{7}',
        'voter_id': r'[A-Z]{3}\d{7}',
        'driving_license': r'[A-Z]{2}\d{13}',
        'bank_account': r'\d{9,18}',
        'ifsc': r'[A-Z]{4}0[A-Z0-9]{6}',
    }
    
    INDIRECT_IDENTIFIERS = [
        'name', 'address', 'dob', 'age', 'gender',
        'pincode', 'location', 'lat', 'lon', 'coordinates'
    ]
    
    def detect(self, df: pd.DataFrame) -> Dict:
        """Detect PII in dataframe"""
        
        direct_identifiers = []
        indirect_identifiers = []
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check column names for indirect identifiers
            if any(term in col_lower for term in self.INDIRECT_IDENTIFIERS):
                indirect_identifiers.append({
                    'column': col,
                    'type': 'column_name_match'
                })
            
            # Check string columns for patterns
            if df[col].dtype == 'object':
                sample = df[col].dropna().astype(str).head(1000)
                
                for pii_type, pattern in self.PATTERNS.items():
                    if sample.str.match(pattern).any():
                        direct_identifiers.append({
                            'column': col,
                            'type': pii_type,
                            'pattern': pattern
                        })
        
        # Calculate re-identification risk
        risk_score = self._calculate_reidentification_risk(
            len(direct_identifiers),
            len(indirect_identifiers),
            len(df)
        )
        
        return {
            'direct_identifiers': direct_identifiers,
            'indirect_identifiers': indirect_identifiers,
            'reidentification_risk': risk_score
        }
    
    def _calculate_reidentification_risk(self, direct: int, indirect: int, rows: int) -> float:
        """Calculate risk of re-identification"""
        if direct > 0:
            return 1.0  # Maximum risk
        
        if indirect == 0:
            return 0.0  # No risk
        
        # Simple heuristic: more indirect identifiers + smaller dataset = higher risk
        risk = min(1.0, (indirect * 0.1) + (1 / np.log10(max(rows, 10))))
        
        return risk
```

---

## 4. Configuration Files

### File: `config/thresholds.yaml`

```yaml
# KDTS Configuration

quality:
  weights:
    schema: 0.25
    completeness: 0.25
    accuracy: 0.20
    uniqueness: 0.15
    distribution: 0.15
  
  thresholds:
    schema_violation: 0.05
    completeness_minimum: 0.90
    accuracy_minimum: 0.95
    uniqueness_minimum: 0.98
    psi_warning: 0.1
    psi_critical: 0.25

legal:
  weights:
    ownership: 0.40
    resale: 0.30
    pii_risk: 0.20
    jurisdiction: 0.10
  
  hard_gate: 60
  
  pii_patterns:
    phone: '\+?91[-\s]?\d{10}'
    email: '[\w\.-]+@[\w\.-]+\.\w+'
    aadhaar: '\d{4}\s?\d{4}\s?\d{4}'

provenance:
  weights:
    methodology: 0.30
    source_type: 0.25
    transformation: 0.25
    bias: 0.20
  
  source_scores:
    primary: 100
    licensed: 90
    aggregated: 75
    scraped: 40
    unknown: 0

usability:
  weights:
    joinability: 0.30
    naming: 0.25
    delivery: 0.25
    integration: 0.20

freshness:
  weights:
    refresh_reliability: 0.50
    latency: 0.30
    historical_depth: 0.20
  
  latency_thresholds:
    real_time: 1  # days
    daily: 7
    weekly: 30
    monthly: 90

kdts:
  weights:
    quality: 0.30
    legal: 0.25
    provenance: 0.20
    usability: 0.15
    freshness: 0.10
  
  bands:
    production_grade: 85
    business_ready: 70
    experimental: 55
    restricted: 0
```

### File: `src/config/settings.py`

```python
from pydantic_settings import BaseSettings
from typing import Optional
import yaml

class Settings(BaseSettings):
    """Application settings"""
    
    # Environment
    environment: str = "development"
    debug: bool = False
    
    # Paths
    data_input_dir: str = "data/input"
    data_output_dir: str = "data/output"
    data_temp_dir: str = "data/temp"
    log_dir: str = "logs"
    config_file: str = "config/thresholds.yaml"
    
    # Processing
    chunk_size: int = 50000
    max_workers: int = 4
    use_polars: bool = True
    
    # Performance
    memory_limit_gb: int = 4
    timeout_seconds: int = 3600
    
    # Logging
    log_level: str = "INFO"
    log_rotation: str = "1 day"
    log_retention: str = "30 days"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def load_thresholds(self):
        """Load threshold configuration"""
        with open(self.config_file, 'r') as f:
            return yaml.safe_load(f)

# Global settings instance
settings = Settings()
```

---

## 5. CLI Implementation

### File: `src/cli.py`

```python
import click
from pathlib import Path
import json
from src.pipeline import KDTSPipeline
from src.config.settings import settings

@click.group()
def cli():
    """KDTS Automation CLI"""
    pass

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', default='output', help='Output directory')
@click.option('--mode', type=click.Choice(['quick', 'standard', 'deep']), default='standard')
@click.option('--config', type=click.Path(), help='Custom config file')
@click.option('--metadata', type=click.Path(), help='Metadata JSON file')
@click.option('--format', type=click.Choice(['json', 'markdown', 'pdf']), default='json')
def score(input_file, output, mode, config, metadata, format):
    """Calculate KDTS for a dataset"""
    
    click.echo(f"Processing: {input_file}")
    click.echo(f"Mode: {mode}")
    
    # Load metadata if provided
    metadata_dict = {}
    if metadata:
        with open(metadata, 'r') as f:
            metadata_dict = json.load(f)
    
    # Initialize pipeline
    pipeline = KDTSPipeline(mode=mode, config_path=config)
    
    # Process
    with click.progressbar(length=100, label='Calculating KDTS') as bar:
        result = pipeline.process(input_file, metadata_dict, progress_callback=bar.update)
    
    # Save output
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        result.save_json(output_path / 'trust_card.json')
    elif format == 'markdown':
        result.save_markdown(output_path / 'trust_card.md')
    elif format == 'pdf':
        result.save_pdf(output_path / 'trust_card.pdf')
    
    # Display summary
    click.echo("\n" + "="*50)
    click.echo(f"KDTS Score: {result.kdts_score:.2f}")
    click.echo(f"Band: {result.band}")
    click.echo(f"Legal Gate: {'PASSED' if result.legal_passed else 'FAILED'}")
    click.echo("="*50)
    
    if result.flags:
        click.echo("\nWarnings:")
        for flag in result.flags:
            click.echo(f"  ⚠️  {flag}")

@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output', '-o', default='output', help='Output directory')
@click.option('--workers', type=int, default=4, help='Number of parallel workers')
def batch(input_dir, output, workers):
    """Process multiple datasets in batch"""
    
    input_path = Path(input_dir)
    files = list(input_path.glob('*.csv')) + list(input_path.glob('*.parquet'))
    
    click.echo(f"Found {len(files)} datasets to process")
    
    pipeline = KDTSPipeline()
    results = pipeline.process_batch(files, num_workers=workers)
    
    # Save summary
    summary = {
        'total': len(results),
        'production_grade': sum(1 for r in results if r.band == 'Production-Grade'),
        'business_ready': sum(1 for r in results if r.band == 'Business-Ready'),
        'experimental': sum(1 for r in results if r.band == 'Experimental'),
        'restricted': sum(1 for r in results if r.band == 'Restricted'),
    }
    
    click.echo(json.dumps(summary, indent=2))

@cli.command()
def validate():
    """Run validation tests on example datasets"""
    click.echo("Running validation tests...")
    # Implementation for validation

if __name__ == '__main__':
    cli()
```

---

This technical specification provides detailed implementation guidance for each module. The interns should follow this structure and implement each component according to the daily plan.
