# KDTS Intern Learning Guide

**Purpose:** Prepare you with the knowledge needed to successfully complete the KDTS automation project.

---

## 📚 Prerequisites Check

### Required Skills (Must Have):
- ✅ Python basics (variables, functions, loops, if/else)
- ✅ Reading/writing files
- ✅ Command line basics (cd, ls, running python scripts)
- ✅ Git basics (clone, commit, push)

### Nice to Have (Will Learn on the Job):
- ⭐ pandas library
- ⭐ Working with JSON
- ⭐ Writing tests with pytest
- ⭐ API concepts (for Plan A)

### Can Be Complete Beginner In:
- 🆕 Data quality concepts
- 🆕 Statistical calculations
- 🆕 FastAPI/Click frameworks
- 🆕 Docker

---

## 🎓 Learning Path: Week Before Starting

### Day -7 to -5: Python Refresher

#### 1. Virtual Environments (30 min)
```bash
# Why: Isolate project dependencies
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate    # Windows

pip install pandas
```

**Practice:**
- Create a venv
- Install pandas
- Run: `python -c "import pandas; print(pandas.__version__)"`

#### 2. Working with Files (1 hour)
```python
# Reading CSV
import pandas as pd

df = pd.read_csv('data.csv')
print(df.head())          # First 5 rows
print(df.shape)           # (rows, columns)
print(df.columns)         # Column names
print(df['column'].mean()) # Average

# Writing JSON
import json
data = {'score': 95.5, 'status': 'pass'}
with open('result.json', 'w') as f:
    json.dump(data, f, indent=2)

# Reading JSON
with open('result.json', 'r') as f:
    loaded = json.load(f)
    print(loaded['score'])
```

**Practice:**
- Download a sample CSV from Kaggle
- Load it with pandas
- Print first 10 rows
- Save summary as JSON

#### 3. Functions and Classes (1 hour)
```python
# Function
def calculate_percentage(part, total):
    """Calculate percentage with validation"""
    if total == 0:
        return 0
    return (part / total) * 100

# Class
class Scorer:
    def __init__(self, name):
        self.name = name
        self.score = 0

    def calculate(self, data):
        self.score = sum(data) / len(data)
        return self.score

# Usage
scorer = Scorer("Quality")
result = scorer.calculate([90, 95, 88])
print(f"{scorer.name}: {result}")
```

**Practice:**
- Write a function that calculates null percentage in a pandas column
- Create a class that loads a file and calculates basic stats

---

### Day -4 to -2: Domain Knowledge

#### 1. Data Quality Concepts (2 hours)

**Key Concepts You'll Use:**

1. **Completeness** - How much data is missing?
```python
df['column'].isnull().sum()  # Count nulls
df['column'].isnull().mean() * 100  # Null percentage
```

2. **Accuracy** - Are values in valid ranges?
```python
# Example: Prices should be > 0
invalid = df[df['price'] < 0]
accuracy = 1 - (len(invalid) / len(df))
```

3. **Uniqueness** - How many duplicates?
```python
duplicates = df.duplicated().sum()
uniqueness = 1 - (duplicates / len(df))
```

4. **Consistency** - Do data types match?
```python
df['age'].dtype  # Should be int or float, not string
```

**Read:** 
- [Data Quality Dimensions](https://www.talend.com/resources/what-is-data-quality/)
- [pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)

#### 2. Statistical Basics (1 hour)

**You'll need these calculations:**

```python
import numpy as np
from scipy import stats

# Mean, Median, Standard Deviation
data = [10, 20, 30, 40, 50]
mean = np.mean(data)
median = np.median(data)
std = np.std(data)

# Z-score (detect outliers)
# Z = (value - mean) / std
# If |Z| > 3, it's an outlier
z_scores = stats.zscore(data)
outliers = abs(z_scores) > 3

# Skewness (data distribution)
# -1 to 1: symmetric
# > 1: right-skewed
# < -1: left-skewed
skewness = stats.skew(data)
```

**Practice:**
- Load a dataset
- Calculate mean, median, std for numeric columns
- Find outliers using Z-score method

#### 3. Weighted Averages (30 min)

**Core concept of KDTS:**

```python
# KDTS = 0.30*Q + 0.25*L + 0.20*P + 0.15*U + 0.10*F

scores = {
    'quality': 96.36,
    'legal': 98.10,
    'provenance': 90.85,
    'usability': 91.60,
    'freshness': 91.50
}

weights = {
    'quality': 0.30,
    'legal': 0.25,
    'provenance': 0.20,
    'usability': 0.15,
    'freshness': 0.10
}

kdts = sum(scores[k] * weights[k] for k in scores)
print(f"KDTS: {kdts:.2f}")  # Should be 94.50
```

**Practice:**
- Calculate weighted average for different scenarios
- Verify your math matches the PDF examples

---

### Day -1: Tool Familiarity

#### For Plan A (FastAPI):

**Watch:** [FastAPI in 30 Minutes](https://www.youtube.com/watch?v=0sOvCWFmrtA)

**Try:**
```bash
pip install fastapi uvicorn

# Create app.py
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

# Run
uvicorn app:app --reload

# Visit: http://localhost:8000/docs
```

#### For Plan B (Click):

**Read:** [Click Documentation](https://click.palletsprojects.com/en/8.1.x/quickstart/)

**Try:**
```bash
pip install click rich

# Create cli.py
import click

@click.group()
def cli():
    pass

@cli.command()
@click.argument('name')
def hello(name):
    click.echo(f"Hello {name}!")

if __name__ == '__main__':
    cli()

# Run
python cli.py hello World
```

---

## 🧠 Key Concepts to Understand

### 1. File Formats

| Format | Pros | Cons | When to Use |
|--------|------|------|-------------|
| CSV | Universal, human-readable | Large size, no schema | Default choice |
| Parquet | Compressed, fast, typed | Binary (can't open in notepad) | Big data |
| Excel | Familiar to business users | Slow, proprietary | Reports |

### 2. JSON Structure
```json
{
  "file_id": "abc123",
  "kdts_score": 94.50,
  "sub_scores": {
    "quality": 96.36,
    "legal": 98.10
  },
  "metadata": {
    "calculated_at": "2024-01-29T10:00:00"
  }
}
```

### 3. Error Handling
```python
try:
    df = pd.read_csv('file.csv')
except FileNotFoundError:
    print("Error: File not found")
    return None
except pd.errors.EmptyDataError:
    print("Error: File is empty")
    return None
```

### 4. Testing Basics
```python
# tests/test_example.py
def calculate_percentage(part, total):
    return (part / total) * 100

def test_calculate_percentage():
    result = calculate_percentage(50, 200)
    assert result == 25.0

# Run: pytest tests/
```

---

## 📖 Recommended Resources

### Python & Pandas:
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Real Python Tutorials](https://realpython.com/)
- [Kaggle Learn: Pandas](https://www.kaggle.com/learn/pandas)

### Data Quality:
- [Great Expectations Docs](https://docs.greatexpectations.io/)
- [Data Quality Wikipedia](https://en.wikipedia.org/wiki/Data_quality)

### Testing:
- [Pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://realpython.com/pytest-python-testing/)

### FastAPI (Plan A):
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [FastAPI YouTube Crash Course](https://www.youtube.com/watch?v=7t2alSnE2-I)

### Click (Plan B):
- [Click Documentation](https://click.palletsprojects.com/)
- [Building CLI Apps with Click](https://www.youtube.com/watch?v=kNke39OZ2k0)

---

## 🛠️ Setup Checklist (Day 0)

### Development Environment:
- [ ] Python 3.9+ installed (`python --version`)
- [ ] pip updated (`pip install --upgrade pip`)
- [ ] Git installed (`git --version`)
- [ ] Code editor ready (VS Code recommended)
- [ ] Terminal comfortable (bash/zsh/cmd)

### Project Setup:
- [ ] Clone repository
- [ ] Create virtual environment
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Run hello world script
- [ ] Run first test (`pytest`)

### Understanding:
- [ ] Read both PDF files completely
- [ ] Understand KDTS formula
- [ ] Know what each score means
- [ ] Understand hard gate concept (Legal >= 60)
- [ ] Reviewed example: KDTS = 94.50

### Communication:
- [ ] Know who to ask for help
- [ ] Set up daily standup time
- [ ] Understand task submission process
- [ ] Clarify expected working hours

---

## 💡 Learning Tips

### 1. Don't Memorize - Understand
- Don't memorize pandas syntax
- Understand WHAT you're calculating and WHY
- Google syntax when needed

### 2. Test as You Go
- Write a function → Write a test immediately
- Don't wait until Day 8 to test

### 3. Ask Questions Early
- Stuck for 30 min? Ask!
- Don't waste hours on small issues

### 4. Use Print Debugging
```python
def calculate_score(data):
    print(f"Input data: {data}")  # Debug
    result = sum(data) / len(data)
    print(f"Result: {result}")    # Debug
    return result
```

### 5. Read Error Messages
```
FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'
                    ^^^^^^^^^ What went wrong
```

### 6. Use AI Assistants (But Wisely)
- ✅ "How do I calculate null percentage in pandas?"
- ✅ "What does this error mean: KeyError: 'column'?"
- ❌ "Write the entire QualityScorer class for me"

### 7. Comment Your Code
```python
def calculate_kdts(scores, weights):
    """
    Calculate final KDTS score using weighted sum

    Args:
        scores: dict of component scores (0-100)
        weights: dict of component weights (sum to 1.0)

    Returns:
        float: KDTS score (0-100)
    """
    # TODO: Add validation
    return sum(scores[k] * weights[k] for k in scores)
```

---

## 🎯 Day 1 Mental Checklist

Before starting Day 1, you should be able to:

1. **Run Python:**
```bash
python --version  # 3.9+
```

2. **Install packages:**
```bash
pip install pandas
python -c "import pandas"
```

3. **Read a CSV:**
```python
import pandas as pd
df = pd.read_csv('test.csv')
print(df.shape)
```

4. **Write JSON:**
```python
import json
with open('test.json', 'w') as f:
    json.dump({'key': 'value'}, f)
```

5. **Run a script:**
```bash
python my_script.py
```

If you can do all 5, **you're ready!** 🚀

---

## 🆘 When You're Stuck

### Problem-Solving Flowchart:
```
Issue occurs
    ↓
Read error message carefully
    ↓
Google: "python [error message]"
    ↓
Still stuck after 15 min?
    ↓
Check documentation
    ↓
Still stuck after 30 min?
    ↓
Ask teammate/senior
    ↓
Still stuck after 1 hour?
    ↓
Escalate to CTO
```

### Common Issues:

**"ModuleNotFoundError: No module named 'pandas'"**
→ Activate virtual environment: `source venv/bin/activate`

**"FileNotFoundError"**
→ Check your current directory: `pwd` (Mac/Linux) or `cd` (Windows)
→ Use absolute paths or `Path(__file__).parent / 'data'`

**"KeyError: 'column_name'"**
→ Column doesn't exist. Check: `df.columns.tolist()`

**"IndexError: list index out of range"**
→ List is empty or shorter than expected. Check: `len(my_list)`

**"TypeError: unsupported operand type(s)"**
→ Mixing types (str + int). Convert: `int("123")` or `str(456)`

---

## 📝 Pre-Project Quiz

Test yourself before Day 1:

1. What does this code output?
```python
data = [1, 2, None, 4, 5]
result = [x for x in data if x is not None]
print(len(result))
```

2. Calculate manually: KDTS with Q=90, L=80, P=85, U=88, F=92

3. What's wrong with this code?
```python
def calculate_score(values):
    return sum(values) / len(values)

scores = []
result = calculate_score(scores)  # Error?
```

4. What does this pandas code do?
```python
df['age'].isnull().sum()
```

5. How do you read a JSON file in Python?

**Answers:**
1. `4` (4 non-None values)
2. KDTS = 0.30(90) + 0.25(80) + 0.20(85) + 0.15(88) + 0.10(92) = 86.4
3. Division by zero when list is empty
4. Counts null values in 'age' column
5. `json.load(open('file.json'))`

---

## 🎓 Graduation Criteria

By Day 12, you will have learned:

✅ **Technical Skills:**
- pandas data manipulation
- Statistical calculations
- API design (Plan A) or CLI design (Plan B)
- Testing with pytest
- Error handling
- Docker basics
- Git workflow

✅ **Soft Skills:**
- Breaking down complex problems
- Writing clean, documented code
- Debugging systematically
- Asking effective questions
- Time management

✅ **Domain Knowledge:**
- Data quality dimensions
- Scoring methodologies
- Weighted aggregation
- Business rule enforcement

**You're building a real production system - be proud!** 🏆

---

## 📞 Support

- **Stuck on concepts?** → Read this guide again
- **Stuck on code?** → Check INTERN_EXECUTION_GUIDE.md
- **Stuck for >30 min?** → Ask your team
- **Need clarification?** → Ask CTO

**Remember:** Everyone was a beginner once. Asking questions is a sign of engagement, not weakness!

Good luck! 🚀
