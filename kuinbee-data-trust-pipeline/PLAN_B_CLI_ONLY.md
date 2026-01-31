# KDTS Automation - Plan B: Pure Python CLI

**Duration:** 10 days  
**Framework:** None (Pure Python + Click CLI)  
**Approach:** Command-line tool for local batch processing  
**No Database, No Web Server, No Frontend**

---

## 🎯 System Architecture

```
Terminal Command
      ↓
┌──────────────────────────────────────────┐
│  kdts-cli (Python Script)                │
│                                           │
│  Commands:                                │
│    kdts upload data.csv                   │
│    kdts quality file_abc123              │
│    kdts legal file_abc123 --scores ...   │
│    kdts calculate file_abc123            │
│    kdts result file_abc123               │
└──────────────────────────────────────────┘
      ↓
data/results/file_abc123.json
```

---

## 📦 Tech Stack

```bash
# Minimal dependencies
pip install pandas==2.1.3
pip install numpy==1.26.2
pip install scipy==1.11.4
pip install click==8.1.7       # CLI framework
pip install rich==13.7.0       # Pretty terminal output
pip install pyarrow==14.0.1    # Parquet support
pip install openpyxl==3.1.2    # Excel support
pip install pytest==7.4.3      # Testing
```

**No FastAPI, No Uvicorn, No Pydantic needed!**

---

## 📂 Project Structure

```
kdts-cli/
├── src/
│   ├── __init__.py
│   ├── cli.py                    # Main CLI commands
│   ├── models.py                 # Data classes (simple)
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── file_reader.py       # Read CSV/Parquet/Excel
│   │   └── profiler.py          # Schema detection
│   │
│   ├── quality/
│   │   ├── __init__.py
│   │   ├── schema_scorer.py
│   │   ├── completeness_scorer.py
│   │   ├── accuracy_scorer.py
│   │   ├── uniqueness_scorer.py
│   │   ├── distribution_scorer.py
│   │   └── quality_scorer.py    # Combines all 5
│   │
│   ├── usability/
│   │   └── usability_scorer.py
│   │
│   ├── freshness/
│   │   └── freshness_scorer.py
│   │
│   ├── calculator/
│   │   └── kdts_calculator.py   # Final KDTS
│   │
│   └── utils/
│       ├── file_manager.py      # Save/load JSON
│       └── logger.py            # Simple logging
│
├── data/
│   ├── uploads/                 # User files
│   └── results/                 # JSON outputs
│
├── tests/
│   └── test_*.py
│
├── requirements.txt
└── README.md
```

---

## 📅 10-Day Plan

---

## **DAY 1: Setup + CLI Framework**

### 🎯 Learning Goals:
- Learn Click framework for CLIs
- Understand command-line arguments
- Practice file operations
- Learn Rich for pretty output

### 📝 Problem Statement:
> "Create a CLI tool that can accept file uploads via command line and assign unique IDs."

### Tasks:

#### 1.1 Environment Setup (30 min)
```bash
mkdir kdts-cli && cd kdts-cli
python -m venv venv
source venv/bin/activate

pip install click rich pandas
```

#### 1.2 Basic CLI Structure (2 hours)
```python
# src/cli.py
import click
from rich.console import Console
from rich.table import Table

console = Console()

@click.group()
def kdts():
    """KDTS CLI - Dataset Trust Score Calculator"""
    pass

@kdts.command()
def version():
    """Show version"""
    console.print("[bold green]KDTS CLI v1.0.0[/bold green]")

if __name__ == "__main__":
    kdts()
```

**Run:**
```bash
python src/cli.py version
# Output: KDTS CLI v1.0.0
```

#### 1.3 Upload Command (3 hours)
```python
import uuid
import shutil
from pathlib import Path

@kdts.command()
@click.argument('file_path', type=click.Path(exists=True))
def upload(file_path):
    """
    Upload a dataset and get a file_id

    Usage: python src/cli.py upload data/sample.csv

    TODO:
    1. Generate file_id using uuid.uuid4()
    2. Validate file format (.csv, .parquet, .xlsx)
    3. Copy file to data/uploads/{file_id}{extension}
    4. Save metadata to data/uploads/{file_id}_meta.json
    5. Print file_id to terminal
    """
    # TODO: Implement
    file_id = str(uuid.uuid4())[:8]  # Short ID

    # Validate extension
    ext = Path(file_path).suffix
    if ext not in ['.csv', '.parquet', '.xlsx']:
        console.print(f"[red]Error: Unsupported format {ext}[/red]")
        return

    # Copy file
    dest = Path(f"data/uploads/{file_id}{ext}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(file_path, dest)

    # Show success
    console.print(f"[green]✓ File uploaded successfully![/green]")
    console.print(f"File ID: [bold]{file_id}[/bold]")
    console.print(f"Saved to: {dest}")
```

### ✅ Success Criteria:
- [ ] CLI runs: `python src/cli.py --help`
- [ ] Upload command works
- [ ] Files copied to data/uploads/
- [ ] File ID generated and displayed

### 🧪 Test:
```bash
python src/cli.py upload test.csv
# Expected output:
# ✓ File uploaded successfully!
# File ID: a1b2c3d4
# Saved to: data/uploads/a1b2c3d4.csv
```

---

## **DAY 2: File Reader + Schema Detection**

### 🎯 Learning Goals:
- Learn pandas multi-format reading
- Understand schema profiling
- Practice JSON serialization
- Learn dataclasses

### Tasks:

#### 2.1 File Reader (2 hours)
```python
# src/ingestion/file_reader.py
import pandas as pd
from pathlib import Path

class FileReader:
    def read(self, file_id: str) -> pd.DataFrame:
        """
        TODO:
        1. Find file in data/uploads/ by file_id
        2. Detect format from extension
        3. Use appropriate pandas reader
        4. Return DataFrame
        """
        # Find file
        upload_dir = Path("data/uploads")
        files = list(upload_dir.glob(f"{file_id}.*"))

        if not files:
            raise FileNotFoundError(f"No file found for ID: {file_id}")

        file_path = files[0]
        ext = file_path.suffix

        # Read based on format
        if ext == '.csv':
            return pd.read_csv(file_path)
        elif ext == '.parquet':
            return pd.read_parquet(file_path)
        elif ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported format: {ext}")
```

#### 2.2 Schema Detector (2 hours)
```python
# src/ingestion/profiler.py
from dataclasses import dataclass, asdict
import json

@dataclass
class ColumnSchema:
    name: str
    dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    sample_values: list

class SchemaDetector:
    def detect(self, df: pd.DataFrame) -> dict:
        """TODO: Extract schema for each column"""
        schema = {}

        for col in df.columns:
            schema[col] = ColumnSchema(
                name=col,
                dtype=str(df[col].dtype),
                null_count=int(df[col].isnull().sum()),
                null_percentage=float(df[col].isnull().mean() * 100),
                unique_count=int(df[col].nunique()),
                sample_values=df[col].dropna().head(3).tolist()
            )

        return {k: asdict(v) for k, v in schema.items()}
```

#### 2.3 Inspect Command (2 hours)
```python
# src/cli.py (add command)
from src.ingestion.file_reader import FileReader
from src.ingestion.profiler import SchemaDetector
import json

@kdts.command()
@click.argument('file_id')
def inspect(file_id):
    """
    Inspect file schema

    Usage: python src/cli.py inspect a1b2c3d4
    """
    console.print(f"[yellow]Loading file {file_id}...[/yellow]")

    # Read file
    reader = FileReader()
    df = reader.read(file_id)

    # Detect schema
    detector = SchemaDetector()
    schema = detector.detect(df)

    # Save to JSON
    output_path = Path(f"data/results/{file_id}_schema.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=2)

    # Display summary
    console.print(f"\n[green]✓ Schema detected![/green]")
    console.print(f"Rows: {len(df):,}")
    console.print(f"Columns: {len(df.columns)}")
    console.print(f"Saved to: {output_path}")

    # Show table
    table = Table(title="Column Summary")
    table.add_column("Column")
    table.add_column("Type")
    table.add_column("Nulls %")
    table.add_column("Unique")

    for col, info in list(schema.items())[:5]:
        table.add_row(
            col,
            info['dtype'],
            f"{info['null_percentage']:.1f}%",
            str(info['unique_count'])
        )

    console.print(table)
```

### ✅ Success Criteria:
- [ ] FileReader loads all formats
- [ ] SchemaDetector extracts info
- [ ] inspect command displays summary
- [ ] Schema saved as JSON

---

## **DAY 3-4: Quality Score (5 Components)**

### Tasks:

#### 3.1 Create All 5 Scorers (Same as FastAPI plan)
```python
# Copy logic from FastAPI plan:
# - src/quality/schema_scorer.py
# - src/quality/completeness_scorer.py
# - src/quality/accuracy_scorer.py
# - src/quality/uniqueness_scorer.py
# - src/quality/distribution_scorer.py
```

#### 3.2 Quality Command (1 hour)
```python
@kdts.command()
@click.argument('file_id')
def quality(file_id):
    """
    Calculate Quality score (automated)

    Usage: python src/cli.py quality a1b2c3d4
    """
    console.print(f"[yellow]Calculating quality for {file_id}...[/yellow]")

    # Load file
    reader = FileReader()
    df = reader.read(file_id)

    # Calculate
    from src.quality.quality_scorer import QualityScorer
    scorer = QualityScorer()
    result = scorer.calculate(df, config={})

    # Save
    output_path = Path(f"data/results/{file_id}_quality.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    # Display
    console.print(f"\n[green]✓ Quality Score: {result['quality_score']:.2f}[/green]")

    # Show breakdown
    table = Table(title="Quality Breakdown")
    table.add_column("Component")
    table.add_column("Score")
    table.add_column("Weight")

    for comp, score in result['sub_scores'].items():
        weight = QualityScorer.WEIGHTS[comp]
        table.add_row(comp.title(), f"{score:.2f}", f"{weight*100}%")

    console.print(table)
```

### ✅ Success Criteria:
- [ ] All 5 sub-scorers working
- [ ] quality command calculates correctly
- [ ] Result saved as JSON
- [ ] Pretty terminal output

---

## **DAY 5: Manual Inputs (Legal & Provenance)**

### Tasks:

#### 5.1 Legal Input Command (2 hours)
```python
@kdts.command()
@click.argument('file_id')
@click.option('--ownership', type=float, required=True, help='Ownership score (0-100)')
@click.option('--resale', type=float, required=True, help='Resale permission (0-100)')
@click.option('--pii', type=float, required=True, help='PII risk score (0-100)')
@click.option('--jurisdiction', type=float, required=True, help='Jurisdiction score (0-100)')
@click.option('--reviewer', required=True, help='Reviewer name')
@click.option('--notes', default='', help='Review notes')
def legal(file_id, ownership, resale, pii, jurisdiction, reviewer, notes):
    """
    Submit manual Legal score

    Usage:
    python src/cli.py legal a1b2c3d4 \
        --ownership 100 --resale 95 --pii 98 --jurisdiction 100 \
        --reviewer "John Doe"
    """
    # Validate ranges
    scores = [ownership, resale, pii, jurisdiction]
    if any(s < 0 or s > 100 for s in scores):
        console.print("[red]Error: All scores must be 0-100[/red]")
        return

    # Calculate weighted score
    legal_score = (
        0.40 * ownership +
        0.30 * resale +
        0.20 * pii +
        0.10 * jurisdiction
    )

    # Check hard gate
    hard_gate_passed = legal_score >= 60

    # Build result
    result = {
        'file_id': file_id,
        'legal_score': legal_score,
        'hard_gate_passed': hard_gate_passed,
        'sub_scores': {
            'ownership': ownership,
            'resale_permission': resale,
            'pii_risk': pii,
            'jurisdiction': jurisdiction
        },
        'reviewer': reviewer,
        'notes': notes,
        'reviewed_at': datetime.now().isoformat()
    }

    # Save
    output_path = Path(f"data/results/{file_id}_legal.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    # Display
    if hard_gate_passed:
        console.print(f"\n[green]✓ Legal Score: {legal_score:.2f} - PASSED[/green]")
    else:
        console.print(f"\n[red]✗ Legal Score: {legal_score:.2f} - REJECTED (< 60)[/red]")
```

#### 5.2 Provenance Input Command (Similar)
```python
@kdts.command()
@click.argument('file_id')
@click.option('--methodology', type=float, required=True)
@click.option('--source-type', type=float, required=True)
@click.option('--lineage', type=float, required=True)
@click.option('--bias', type=float, required=True)
@click.option('--reviewer', required=True)
def provenance(file_id, methodology, source_type, lineage, bias, reviewer):
    """Submit manual Provenance score"""
    # TODO: Similar to legal
    pass
```

### ✅ Success Criteria:
- [ ] legal command accepts inputs
- [ ] provenance command accepts inputs
- [ ] Hard gate check works
- [ ] Scores saved as JSON

---

## **DAY 6: Usability & Freshness**

### Tasks:
(Same logic as FastAPI plan, just add CLI commands)

```python
@kdts.command()
@click.argument('file_id')
def usability(file_id):
    """Calculate Usability score (automated)"""
    # TODO: Load file, run UsabilityScorer, save result
    pass

@kdts.command()
@click.argument('file_id')
@click.option('--last-update', type=str, help='Last update date (YYYY-MM-DD)')
@click.option('--frequency', type=click.Choice(['daily', 'weekly', 'monthly', 'quarterly']))
def freshness(file_id, last_update, frequency):
    """Calculate Freshness score"""
    # TODO: Needs metadata input, then calculate
    pass
```

---

## **DAY 7: KDTS Calculator**

### Tasks:

#### 7.1 Calculate Command (3 hours)
```python
@kdts.command()
@click.argument('file_id')
def calculate(file_id):
    """
    Calculate final KDTS score

    Requires all 5 component scores to exist:
    - Quality (auto)
    - Legal (manual)
    - Provenance (manual)
    - Usability (auto)
    - Freshness (manual/auto)

    Usage: python src/cli.py calculate a1b2c3d4
    """
    console.print(f"[yellow]Calculating KDTS for {file_id}...[/yellow]")

    # Load all scores
    results_dir = Path("data/results")
    scores = {}

    for component in ['quality', 'legal', 'provenance', 'usability', 'freshness']:
        score_file = results_dir / f"{file_id}_{component}.json"

        if not score_file.exists():
            console.print(f"[red]Error: Missing {component} score[/red]")
            console.print(f"Run: python src/cli.py {component} {file_id}")
            return

        with open(score_file) as f:
            data = json.load(f)
            scores[component] = data[f'{component}_score']

    # Calculate KDTS
    from src.calculator.kdts_calculator import KDTSCalculator
    calc = KDTSCalculator()
    result = calc.calculate(scores)

    # Save
    output_path = results_dir / f"{file_id}_kdts.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    # Display
    console.print(f"\n[bold green]✓ KDTS: {result['kdts_score']:.2f}[/bold green]")
    console.print(f"Band: [cyan]{result['band']}[/cyan]")

    # Show breakdown
    table = Table(title="KDTS Breakdown")
    table.add_column("Component")
    table.add_column("Score")
    table.add_column("Weight")
    table.add_column("Contribution")

    weights = KDTSCalculator.WEIGHTS
    for comp, score in scores.items():
        weight = weights[comp]
        contrib = score * weight
        table.add_row(
            comp.title(),
            f"{score:.2f}",
            f"{weight*100}%",
            f"{contrib:.2f}"
        )

    console.print(table)
```

#### 7.2 Result Command (1 hour)
```python
@kdts.command()
@click.argument('file_id')
@click.option('--format', type=click.Choice(['json', 'pretty']), default='pretty')
def result(file_id, format):
    """
    View KDTS result

    Usage:
    python src/cli.py result a1b2c3d4
    python src/cli.py result a1b2c3d4 --format json
    """
    result_file = Path(f"data/results/{file_id}_kdts.json")

    if not result_file.exists():
        console.print(f"[red]No result found for {file_id}[/red]")
        console.print("Run: python src/cli.py calculate {file_id}")
        return

    with open(result_file) as f:
        data = json.load(f)

    if format == 'json':
        print(json.dumps(data, indent=2))
    else:
        # Pretty print (same as calculate command output)
        pass
```

---

## **DAY 8-9: Testing**

### Tasks:

```python
# tests/test_cli.py
from click.testing import CliRunner
from src.cli import kdts

def test_upload_command():
    runner = CliRunner()
    result = runner.invoke(kdts, ['upload', 'test.csv'])
    assert result.exit_code == 0
    assert 'File ID' in result.output

def test_quality_command():
    runner = CliRunner()
    # Upload first
    result = runner.invoke(kdts, ['upload', 'test.csv'])
    file_id = extract_file_id(result.output)

    # Calculate quality
    result = runner.invoke(kdts, ['quality', file_id])
    assert result.exit_code == 0
    assert 'Quality Score' in result.output

# TODO: Write tests for all commands
```

---

## **DAY 10: Documentation & Batch Processing**

### Tasks:

#### 10.1 Batch Command (2 hours)
```python
@kdts.command()
@click.argument('directory')
def batch(directory):
    """
    Process all files in a directory

    Usage: python src/cli.py batch data/batch/
    """
    files = Path(directory).glob('*.*')

    for file_path in files:
        console.print(f"\n[yellow]Processing {file_path.name}...[/yellow]")

        # Upload
        result = runner.invoke(upload, [str(file_path)])
        file_id = extract_file_id(result.output)

        # Auto-calculate what we can
        runner.invoke(quality, [file_id])
        runner.invoke(usability, [file_id])

        console.print(f"[green]✓ Completed {file_id}[/green]")
```

#### 10.2 Complete README (2 hours)

### ✅ Final Deliverables:
- [ ] Working CLI tool
- [ ] All commands functional
- [ ] Tests passing
- [ ] Documentation complete

---

## 🚀 Usage Examples

### Complete Workflow:
```bash
# 1. Upload dataset
python src/cli.py upload data/sample.csv
# Output: File ID: a1b2c3d4

# 2. Auto-calculate scores
python src/cli.py quality a1b2c3d4
python src/cli.py usability a1b2c3d4

# 3. Manual inputs
python src/cli.py legal a1b2c3d4 \
    --ownership 100 --resale 95 --pii 98 --jurisdiction 100 \
    --reviewer "John Doe"

python src/cli.py provenance a1b2c3d4 \
    --methodology 92 --source-type 88 --lineage 95 --bias 89 \
    --reviewer "Jane Smith"

python src/cli.py freshness a1b2c3d4 \
    --last-update 2024-10-01 --frequency quarterly

# 4. Calculate final KDTS
python src/cli.py calculate a1b2c3d4

# 5. View result
python src/cli.py result a1b2c3d4
```

### Batch Processing:
```bash
python src/cli.py batch data/incoming/
```

---

## 🎯 Advantages of CLI Approach

1. **Simpler** - No web server, just scripts
2. **Faster development** - Less boilerplate
3. **Easy debugging** - Direct output to terminal
4. **Scriptable** - Can wrap in bash/cron
5. **Lower learning curve** - Just Python

Perfect for internal tools and batch processing!
