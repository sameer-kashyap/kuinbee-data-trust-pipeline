# KDTS Intern Execution Guide

**Purpose:** Daily workflow, debugging tips, and submission criteria for the 12-day project.

---

## 🌅 Daily Workflow Template

### Every Morning (30 min):

```markdown
## Daily Standup Notes - Day X

**Yesterday:** 
- Completed: [task list]
- Challenges: [what was hard]

**Today:**
- Goal: [what you'll complete]
- Tasks: [specific items]

**Blockers:**
- [Any issues needing help]

**Questions for Team:**
- [Prepare questions in advance]
```

### During the Day (6-7 hours):

1. **Read task description** (15 min)
   - Understand the problem statement
   - Review learning goals
   - Check success criteria

2. **Plan before coding** (15 min)
   - Sketch out the logic on paper
   - List what functions you need
   - Identify inputs and outputs

3. **Code in small chunks** (2-3 hours)
   - Write 1 function at a time
   - Test immediately
   - Don't move on until it works

4. **Mid-day check** (30 min)
   - Are you on track?
   - Need to ask for help?
   - Document what you've done

5. **Continue coding** (2-3 hours)
   - Complete remaining tasks
   - Write docstrings
   - Add comments

6. **End-of-day cleanup** (30 min)
   - Run all tests
   - Format code with `black`
   - Commit to git
   - Update daily notes

### Every Evening:

```bash
# Code quality check
black src/
mypy src/ --ignore-missing-imports
pytest tests/ -v

# Git commit
git add .
git commit -m "Day X: Completed [task description]"
git push origin main
```

---

## 📋 Day-by-Day Execution Checklist

### **Day 1 Checklist:**
- [ ] Environment setup completed
- [ ] FastAPI/Click hello world works
- [ ] Upload endpoint/command functional
- [ ] File saved to data/uploads/
- [ ] file_id generated and returned
- [ ] Basic error handling (invalid format)
- [ ] Test with CSV, Parquet, Excel files
- [ ] Git commit: "Day 1: File upload system"

**Time allocation:**
- Setup: 1 hour
- Upload logic: 3 hours
- Testing: 1 hour
- Documentation: 30 min

**Common issues:**
- Virtual environment not activated → `source venv/bin/activate`
- Import errors → Check if packages installed
- File not found → Use absolute paths or `Path()`

---

### **Day 2 Checklist:**
- [ ] FileReader class works for all formats
- [ ] SchemaDetector extracts column info
- [ ] DatasetProfiler calculates stats
- [ ] Metadata saved as JSON
- [ ] API endpoint/CLI command works
- [ ] Test with 3 different files
- [ ] Git commit: "Day 2: Schema detection"

**Validation test:**
```python
# Should output schema correctly
df = pd.DataFrame({
    'id': [1, 2, None, 4],
    'value': [100, 200, 300, 400]
})

detector = SchemaDetector()
schema = detector.detect(df)

assert schema['id']['null_count'] == 1
assert schema['id']['null_percentage'] == 25.0
```

**Common issues:**
- Different formats have different pandas functions
- Nulls vs NaN vs None - pandas uses NaN
- JSON serialization errors → Use `int()`, `float()` to convert numpy types

---

### **Day 3-4 Checklist:**

**Day 3:**
- [ ] SchemaIntegrityScorer complete
- [ ] CompletenessScorer complete
- [ ] Both tested with sample data
- [ ] Results match expected calculations

**Day 4:**
- [ ] AccuracyScorer complete
- [ ] UniquenessScorer complete
- [ ] DistributionScorer complete
- [ ] QualityScorer combines all 5
- [ ] API endpoint/CLI command works
- [ ] **Critical:** Test against PDF example
- [ ] Git commit: "Day 3-4: Quality scoring module"

**Validation test (MOST IMPORTANT):**
```python
def test_quality_calculation():
    """This must pass before moving forward"""
    # Use artificial dataset from PDF
    # Expected: Q = 96.36

    scorer = QualityScorer()
    result = scorer.calculate(df, config)

    # Allow small floating point differences
    assert abs(result['quality_score'] - 96.36) < 0.1

    # Check sub-scores
    assert abs(result['sub_scores']['schema'] - 97.5) < 0.1
    assert abs(result['sub_scores']['completeness'] - 96.0) < 0.1
```

**Common issues:**
- Weighted average calculation wrong → Double-check weights sum to 1.0
- Percentage vs decimal → Use 0.25, not 25
- Division by zero → Always check denominators
- Outlier detection → Use `scipy.stats.zscore()`

---

### **Day 5 Checklist:**
- [ ] Legal score input endpoint/command works
- [ ] Sub-scores validated (0-100 range)
- [ ] Weighted calculation correct (L = 0.40*O + 0.30*R + 0.20*P + 0.10*J)
- [ ] Hard gate check enforced (>= 60)
- [ ] Provenance score input works
- [ ] Both save to JSON files
- [ ] ScoreStorage utility works
- [ ] Git commit: "Day 5: Manual input APIs"

**Validation test:**
```python
def test_legal_hard_gate():
    # Should pass
    input1 = LegalScoreInput(
        file_id="test",
        ownership_score=100,
        resale_permission_score=95,
        pii_risk_score=98,
        jurisdiction_score=100,
        reviewer_name="Test"
    )
    assert input1.calculate_legal_score() >= 60

    # Should fail
    input2 = LegalScoreInput(
        file_id="test",
        ownership_score=40,
        resale_permission_score=50,
        pii_risk_score=60,
        jurisdiction_score=50,
        reviewer_name="Test"
    )
    assert input2.calculate_legal_score() < 60
```

**Common issues:**
- Forgetting to save reviewer metadata
- Hard gate logic in wrong place
- Not handling missing optional fields (notes)

---

### **Day 6 Checklist:**
- [ ] UsabilityScorer with 4 components
- [ ] Joinability detection works
- [ ] Naming quality check works
- [ ] Format scoring works
- [ ] FreshnessScorer with 3 components
- [ ] Date parsing works correctly
- [ ] Both endpoints/commands functional
- [ ] Git commit: "Day 6: Usability and Freshness"

**Validation test:**
```python
def test_usability_calculation():
    df = pd.DataFrame({
        'pincode': [110001, 110002],
        'property_value': [1000000, 2000000],
        'date': ['2024-01-01', '2024-01-02']
    })

    scorer = UsabilityScorer()
    result = scorer.score_joinability(df)

    # Should find 2 join keys: pincode, date
    assert result >= 80  # Should score high
```

**Common issues:**
- Date parsing errors → Use `pd.to_datetime()` with `errors='coerce'`
- String matching for join keys → Use `.lower()` for case-insensitive
- File format detection → Use `Path(file).suffix`

---

### **Day 7 Checklist:**
- [ ] KDTSCalculator validates all scores present
- [ ] Weighted calculation correct
- [ ] Hard gate check works
- [ ] Band classification correct
- [ ] Breakdown calculation accurate
- [ ] Result API/command returns JSON
- [ ] **CRITICAL:** Exact match with PDF example
- [ ] Git commit: "Day 7: KDTS calculator"

**Validation test (MUST PASS):**
```python
def test_pdf_example_exact():
    """Test against PDF example - must match exactly"""
    scores = {
        'quality': 96.36,
        'legal': 98.10,
        'provenance': 90.85,
        'usability': 91.60,
        'freshness': 91.50
    }

    calc = KDTSCalculator()
    result = calc.calculate(scores)

    # Check KDTS
    expected_kdts = (
        96.36 * 0.30 +
        98.10 * 0.25 +
        90.85 * 0.20 +
        91.60 * 0.15 +
        91.50 * 0.10
    )

    assert abs(result['kdts_score'] - expected_kdts) < 0.01
    assert result['band'] == 'Production-Grade'
    assert result['hard_gate_passed'] == True

    # Check breakdown
    assert abs(result['breakdown']['quality_contribution'] - 28.91) < 0.1
    assert abs(result['breakdown']['legal_contribution'] - 24.53) < 0.1
```

**Common issues:**
- Floating point precision → Use `round(score, 2)`
- Band thresholds off-by-one → Use `>=` not `>`
- Not checking all scores present before calculation

---

### **Day 8-9 Checklist:**

**Day 8:**
- [ ] Unit tests for all scorers
- [ ] Test fixtures created
- [ ] Edge cases tested (empty data, all nulls, etc.)
- [ ] Integration tests for workflow
- [ ] Test coverage > 70%

**Day 9:**
- [ ] Error handling added everywhere
- [ ] Custom exceptions defined
- [ ] Validation logic complete
- [ ] PDF example test passes
- [ ] All tests green
- [ ] Git commit: "Day 8-9: Testing and error handling"

**Test checklist:**
```python
# Must have tests for:
- [ ] Empty dataset
- [ ] All null columns
- [ ] Invalid file format
- [ ] Missing scores for KDTS calculation
- [ ] Legal hard gate failure
- [ ] Invalid score ranges (< 0 or > 100)
- [ ] Division by zero cases
- [ ] File not found errors
```

**Common issues:**
- Fixtures not cleaning up → Use `@pytest.fixture` with `yield`
- Import errors in tests → Check `__init__.py` files
- Tests passing locally but failing in CI → Path issues

---

### **Day 10 Checklist:**
- [ ] TrustCard model defined with all fields
- [ ] JSON schema validated
- [ ] API documentation complete (Plan A)
- [ ] README with examples complete
- [ ] Example requests tested
- [ ] All endpoints documented
- [ ] Git commit: "Day 10: Documentation and output"

**Documentation checklist:**
- [ ] Installation instructions
- [ ] Usage examples for each command/endpoint
- [ ] Expected input/output formats
- [ ] Error code explanations
- [ ] Configuration options
- [ ] Troubleshooting section

---

### **Day 11 Checklist:**
- [ ] Code formatted with `black`
- [ ] Type hints added with `mypy`
- [ ] Imports organized with `isort`
- [ ] Logging added to key functions
- [ ] Performance test with large file (1M+ rows)
- [ ] Memory profiling done
- [ ] Git commit: "Day 11: Code quality and performance"

**Code quality commands:**
```bash
# Format
black src/ tests/

# Type check
mypy src/ --ignore-missing-imports

# Sort imports
isort src/ tests/

# Lint
flake8 src/ --max-line-length=100

# Security check
bandit -r src/
```

---

### **Day 12 Checklist:**
- [ ] Dockerfile builds successfully
- [ ] Docker image runs correctly
- [ ] All tests pass in container
- [ ] README finalized
- [ ] Code walkthrough prepared
- [ ] Handoff document created
- [ ] Git commit: "Day 12: Deployment ready"
- [ ] **Final submission**

**Deployment test:**
```bash
# Build
docker build -t kdts-backend .

# Run (Plan A)
docker run -p 8000:8000 kdts-backend
curl http://localhost:8000/

# Run (Plan B)
docker run kdts-backend python src/cli.py --help
```

---

## 🐛 Debugging Guide

### When Code Doesn't Run:

#### 1. Read the Error Message
```python
Traceback (most recent call last):
  File "scorer.py", line 45, in calculate
    result = sum(values) / len(values)
ZeroDivisionError: division by zero
    ^^^^^^^^^^^^^^^ The problem
```

**Fix:** Check if `len(values) == 0` before dividing

#### 2. Use Print Debugging
```python
def calculate_score(data):
    print(f"DEBUG: Input data type: {type(data)}")
    print(f"DEBUG: Data length: {len(data)}")
    print(f"DEBUG: First 5 values: {data[:5]}")

    result = sum(data) / len(data)
    print(f"DEBUG: Result: {result}")

    return result
```

#### 3. Use Python Debugger
```python
import pdb

def calculate_score(data):
    pdb.set_trace()  # Execution stops here
    result = sum(data) / len(data)
    return result

# Then in terminal:
# (Pdb) print(data)  # Inspect variables
# (Pdb) next         # Step to next line
# (Pdb) continue     # Resume execution
```

#### 4. Check pandas DataFrame
```python
df = pd.read_csv('data.csv')

# Debugging commands:
print(df.shape)              # (rows, cols)
print(df.columns.tolist())   # Column names
print(df.head())             # First 5 rows
print(df.dtypes)             # Data types
print(df.isnull().sum())     # Null counts
print(df.describe())         # Statistics

# Check specific column:
print(df['column'].unique())  # Unique values
print(df['column'].value_counts())  # Frequency
```

---

### Common Error Patterns:

#### Pattern 1: File Not Found
```python
# ❌ Error
df = pd.read_csv('data.csv')
# FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'

# ✅ Fix
from pathlib import Path

file_path = Path(__file__).parent / 'data' / 'data.csv'
if not file_path.exists():
    raise FileNotFoundError(f"File not found: {file_path}")
df = pd.read_csv(file_path)
```

#### Pattern 2: KeyError in Dict
```python
# ❌ Error
score = scores['quality']
# KeyError: 'quality'

# ✅ Fix
score = scores.get('quality')
if score is None:
    raise ValueError("Quality score not found")

# Or with default:
score = scores.get('quality', 0.0)
```

#### Pattern 3: Type Mismatch
```python
# ❌ Error
total = "100" + 50
# TypeError: can only concatenate str (not "int") to str

# ✅ Fix
total = int("100") + 50
# Or: total = float("100") + 50
```

#### Pattern 4: Division by Zero
```python
# ❌ Error
average = sum(values) / len(values)
# ZeroDivisionError: division by zero

# ✅ Fix
if len(values) == 0:
    average = 0.0
else:
    average = sum(values) / len(values)

# Or:
average = sum(values) / len(values) if values else 0.0
```

#### Pattern 5: Null Handling
```python
# ❌ Error
df['price'].mean()  # Can produce NaN if all nulls

# ✅ Fix
if df['price'].isnull().all():
    mean_price = 0.0
else:
    mean_price = df['price'].mean()

# Or: 
mean_price = df['price'].mean() or 0.0
```

---

## ✅ Submission Criteria

### Daily Submission:
Every day at end-of-day, submit:
1. **Code** (via git push)
2. **Daily notes** (standup markdown file)
3. **Test results** (screenshot of `pytest` passing)

### Weekly Review (Day 4, 8, 12):
Prepare:
1. **Demo** - Show working features
2. **Code walkthrough** - Explain key decisions
3. **Challenges document** - What was hard, how you solved
4. **Next week plan** - What you'll work on

### Final Submission (Day 12):
Deliver:
1. **Complete codebase**
   - All features working
   - Tests passing
   - Documented

2. **Documentation**
   - README.md
   - API documentation (if Plan A)
   - Usage examples

3. **Docker image**
   - Builds successfully
   - Runs correctly

4. **Test results**
   - Coverage report
   - PDF example validation passing

5. **Handoff document**
   - Architecture overview
   - Design decisions
   - Known limitations
   - Future improvements

---

## 🎯 Quality Standards

### Code Quality:
- [ ] Follows PEP 8 style guide
- [ ] Functions have docstrings
- [ ] Complex logic has comments
- [ ] No hardcoded values (use config)
- [ ] Error handling present
- [ ] Type hints used

### Testing:
- [ ] Every function has a test
- [ ] Edge cases covered
- [ ] Integration tests present
- [ ] Example validation passes
- [ ] No skipped tests

### Documentation:
- [ ] README complete
- [ ] Functions documented
- [ ] Examples provided
- [ ] Known issues listed

---

## 🚨 Red Flags to Avoid

### ❌ Don't:
- Copy-paste code without understanding
- Skip tests "to save time"
- Hardcode file paths
- Commit broken code
- Wait until last day to test
- Ignore error messages
- Work more than 8 hours/day
- Skip daily standups

### ✅ Do:
- Ask questions early
- Test frequently
- Commit often (small commits)
- Document as you go
- Take breaks
- Review your code before committing
- Celebrate small wins
- Keep CTO updated

---

## 📞 Getting Help

### When to Ask for Help:

**Immediately:** 
- Can't set up environment
- Don't understand the task
- Found a bug in requirements

**After 30 minutes:**
- Error message you don't understand
- Logic seems wrong
- Test failing unexpectedly

**After 2 hours:**
- Major blocker
- Design decision needed
- Unclear requirements

### How to Ask for Help:

**Bad:**
```
"My code doesn't work. Please help."
```

**Good:**
```
"I'm working on the CompletenessScorer (Day 3, task 3.2).

Problem: When calculating weighted null rate, I'm getting 0.08 but expected 0.04.

What I tried:
1. Verified critical fields list
2. Checked null percentage calculations  
3. Printed intermediate values

Code snippet:
[paste relevant 10-15 lines]

Error/Output:
[paste actual vs expected]

Question: Am I applying the weights correctly in the average calculation?
"
```

---

## 🎓 Success Metrics

### You're on track if:
- ✅ Each day's tasks complete by EOD
- ✅ Tests passing for completed tasks
- ✅ Can explain your code to others
- ✅ Git commits every day
- ✅ No major blockers > 2 hours

### You're ahead if:
- 🚀 Finishing tasks early
- 🚀 Adding extra tests
- 🚀 Helping other interns
- 🚀 Suggesting improvements

### You need help if:
- ⚠️ 2+ days behind schedule
- ⚠️ Tests failing for > 1 day
- ⚠️ Can't explain what your code does
- ⚠️ No git commits in 2 days
- ⚠️ Stuck on same issue > 4 hours

**Don't wait - escalate immediately!**

---

## 🏆 Final Advice

### From Past Successful Interns:

1. **"Start simple, then improve"**
   - Get basic version working first
   - Optimize later

2. **"Test early, test often"**
   - Don't wait until Day 8
   - Write tests as you code

3. **"Read the docs, don't guess"**
   - pandas docs are excellent
   - Don't assume behavior

4. **"Git commit like you're narrating"**
   - "Added schema scorer"
   - "Fixed null handling bug"
   - "Refactored quality module"

5. **"Take breaks when stuck"**
   - Walk away for 10 minutes
   - Solution often comes when relaxed

6. **"You're building something real"**
   - This will be used in production
   - Take pride in your work

---

## 📅 Daily Time Budget

**Total: 7 hours/day**

- Coding: 5 hours
- Testing: 1 hour
- Documentation: 30 min
- Standup/communication: 30 min

**Don't:**
- Work >8 hours (diminishing returns)
- Skip breaks
- Rush through tests

**Do:**
- Take 10 min break every hour
- Eat lunch away from computer
- Stop when you're stuck (ask for help)

---

## 🎯 Remember:

**This is a learning experience.**

- Mistakes are expected
- Questions are encouraged
- Progress > perfection
- Understanding > speed

**By Day 12, you will have:**
- Built a production system
- Learned modern Python
- Practiced testing
- Shipped something real

**That's HUGE! 🚀**

You got this! 💪
