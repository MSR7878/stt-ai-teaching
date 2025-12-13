---
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Fira+Code&display=swap');
  @import 'custom.css';
---

<!-- _class: lead -->

# Week 2: Data Validation

**CS 203: Software Tools and Techniques for AI**

Prof. Nipun Batra
IIT Gandhinagar

---

# The Data Quality Crisis

**Reality Check**:
- You collected 1,000 movies from OMDb.
- You try to train a model.
- **Crash!** `ValueError: could not convert string to float: 'N/A'`

**Common Issues**:
1.  **Missing Data**: `NULL`, `NaN`, `""`, `"N/A"`, `"-"`.
2.  **Type Mismatches**: Rating `"8.5"` (string) instead of `8.5` (float).
3.  **Outliers**: A movie from year `20250` (typo).
4.  **Drift**: Last month's data format $\neq$ today's data format.

---

# Data Quality Dimensions

**Six dimensions of data quality**:

1. **Accuracy**: Values correctly represent the real-world entity
2. **Completeness**: All required data is present
3. **Consistency**: No contradictions (e.g., age=5, birth_year=1950)
4. **Timeliness**: Data is up-to-date and relevant
5. **Validity**: Data conforms to defined formats and constraints
6. **Uniqueness**: No unwanted duplicates

**ML Impact**: Poor quality in any dimension → degraded model performance.

---

# Types of Data Quality Issues

**Structural issues**:
- Missing values (nulls, empty strings)
- Wrong data types (strings instead of numbers)
- Invalid formats (dates, emails, phone numbers)

**Semantic issues**:
- Outliers and anomalies
- Inconsistent units (km vs miles)
- Invalid categories
- Logical contradictions

**Temporal issues**:
- Data drift (distribution changes)
- Schema drift (structure changes)
- Concept drift (target relationship changes)

---

# The Cost of Poor Data Quality

**Training time**:
- Model fails to converge
- Excessive debugging time
- Wasted compute resources

**Production time**:
- Incorrect predictions
- Model failures and crashes
- Data pipeline failures
- User trust erosion

**Rule**: **1 hour of validation saves 10 hours of debugging.**

---

# The Validation Pipeline

![width:900px](../figures/data_validation_pipeline.png)
*[diagram-generators/data_validation_pipeline.py](../diagram-generators/data_validation_pipeline.py)*

**Tools**:
- **Inspect**: `jq` (JSON), `csvkit` (CSV).
- **Schema**: `Pydantic` (Row-level), `Pandera` (Batch-level).
- **Clean**: `pandas`.

---

# Part 1: Inspection (CLI Tools)

Look at your data *before* loading it into Python.

---

# Data Profiling: First Look

**Before writing any code, ask**:

1. How many records?
2. How many features?
3. What are the data types?
4. How many nulls per column?
5. What's the value distribution?

**Tools**: `head`, `wc`, `jq`, `csvstat`

**Why CLI first?**
- Fast for large files (GB-scale)
- No Python dependencies
- Quick sanity checks
- Scriptable and pipeable

---

# jq: JSON Power Tool

**Find Broken Records**:
Movies where `BoxOffice` is missing ("N/A").

```bash
cat movies.json | jq '.[] | select(.BoxOffice == "N/A") | .Title'
```

**Check Distribution**:
Count movies by Year.

```bash
cat movies.json | jq '.[].Year' | sort | uniq -c
```

**Quick Sanity Check**:
Are there any ratings > 10?

```bash
cat movies.json | jq '.[] | select(.imdbRating | tonumber > 10)'
```

**Extract specific fields**:
```bash
cat movies.json | jq '.[] | {title: .Title, year: .Year, rating: .imdbRating}'
```

---

# jq Advanced Patterns

**Compute statistics**:
```bash
# Average rating
cat movies.json | jq '[.[].imdbRating | tonumber] | add/length'

# Find max budget
cat movies.json | jq '[.[].Budget | tonumber] | max'
```

**Filter and transform**:
```bash
# High-rated recent movies
cat movies.json | jq '
  .[] |
  select(.Year | tonumber > 2020) |
  select(.imdbRating | tonumber > 8.0) |
  {title: .Title, rating: .imdbRating}
'
```

**Why jq matters**: Debug API responses before writing Python code.

---

# csvkit: SQL for CSVs

**Statistics (`csvstat`)**:
Gives mean, median, null count, unique values.

```bash
csvstat movies.csv
```

**Output**:
```
  1. "Title"
        Type of data:          Text
        Contains null values:  False
        Unique values:         1000

  2. "Year"
        Type of data:          Number
        Contains null values:  False
        Min:                   1920
        Max:                   2024
        Mean:                  1995.3
```

---

# csvkit: More Commands

**Query (`csvsql`)**:
Run SQL directly on CSV!

```bash
csvsql --query "SELECT Title FROM movies WHERE Rating > 9" movies.csv
```

**Cut columns (`csvcut`)**:
```bash
csvcut -c Title,Year,Rating movies.csv
```

**Look at specific rows (`csvlook`)**:
```bash
head -20 movies.csv | csvlook
```

**Join CSVs (`csvjoin`)**:
```bash
csvjoin -c movie_id movies.csv reviews.csv
```

---

# Data Profiling with pandas-profiling

**Automatic report generation**:

```python
from ydata_profiling import ProfileReport
import pandas as pd

df = pd.read_csv("movies.csv")
profile = ProfileReport(df, title="Movie Dataset Report")
profile.to_file("report.html")
```

**Generates**:
- Overview (rows, columns, missing %, duplicates)
- Variable distributions (histograms)
- Correlations (heatmap)
- Missing value patterns
- Warnings (high cardinality, high correlation, skewness)

**Use case**: Initial data exploration before validation.

---

# Part 2: Schema Validation (Pydantic)

**The Contract**: Define what valid data *looks* like.

```python
from pydantic import BaseModel, Field, validator
from typing import Optional

class Movie(BaseModel):
    title: str
    year: int = Field(gt=1888, lt=2030) # Constraints
    rating: float = Field(ge=0, le=10)
    genre: str

    # Custom Validator
    @validator('genre')
    def genre_must_be_valid(cls, v):
        allowed = {'Action', 'Comedy', 'Drama'}
        if v not in allowed:
            raise ValueError(f"Unknown genre: {v}")
        return v
```

---

# Why Pydantic?

1.  **Type Coercion**: Converts `"2010"` (str) -> `2010` (int) automatically.
2.  **Early Failure**: Errors catch invalid data immediately.
3.  **Documentation**: The code *is* the documentation of your data format.

```python
try:
    # This will fail (Year out of bounds)
    m = Movie(title="Future", year=3000, rating=5.0, genre="Action")
except ValueError as e:
    print(e)
    # Output: 1 validation error for Movie
    # year: ensure this value is less than 2030
```

---

# Pydantic Field Constraints

**Numeric constraints**:
- `gt`, `ge`: Greater than, greater or equal
- `lt`, `le`: Less than, less or equal
- `multiple_of`: Must be multiple of N

**String constraints**:
- `min_length`, `max_length`: Length bounds
- `regex`: Pattern matching
- `strip_whitespace`: Remove leading/trailing spaces

**Example**:
```python
class User(BaseModel):
    username: str = Field(min_length=3, max_length=20, regex="^[a-zA-Z0-9_]+$")
    age: int = Field(ge=0, le=120)
    email: str = Field(regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
```

---

# Pydantic Validators

**Field validators** (validate single field):
```python
@validator('email')
def email_must_be_lowercase(cls, v):
    return v.lower()
```

**Root validators** (validate entire object):
```python
@root_validator
def check_consistency(cls, values):
    start = values.get('start_date')
    end = values.get('end_date')
    if start and end and end < start:
        raise ValueError('end_date must be after start_date')
    return values
```

**Pre vs Post validation**:
- `@validator(..., pre=True)`: Runs before type coercion
- `@validator(..., pre=False)`: Runs after type coercion (default)

---

# Pydantic: Handling Missing Values

**Optional fields**:
```python
from typing import Optional

class Movie(BaseModel):
    title: str  # Required
    budget: Optional[float] = None  # Optional, defaults to None
    release_date: Optional[str]  # Optional, no default (must be provided, can be None)
```

**Default values**:
```python
class Movie(BaseModel):
    title: str
    rating: float = 5.0  # Default if not provided
    is_released: bool = True
```

**Field with default factory**:
```python
from datetime import datetime

class Movie(BaseModel):
    title: str
    created_at: datetime = Field(default_factory=datetime.now)
```

---

# Pydantic: Validation Modes

**Strict mode** (no coercion):
```python
class StrictMovie(BaseModel):
    year: int  # "2010" will fail, must be int

    class Config:
        strict = True
```

**Fail fast vs collect all errors**:
```python
# Default: fail on first error
m = Movie(title="", year="invalid")  # Fails on year

# Collect all errors
class Config:
    validate_all = True
# Now shows errors for both title and year
```

---

# Pydantic: Custom Types

**Create reusable validated types**:

```python
from pydantic import constr, conint, condecimal

# Constrained types
Username = constr(min_length=3, max_length=20, regex="^[a-zA-Z0-9_]+$")
PositiveInt = conint(gt=0)
Rating = condecimal(ge=0, le=10, decimal_places=1)

class User(BaseModel):
    username: Username
    age: PositiveInt
    rating: Rating
```

**Benefits**: Reuse validation logic across models.

---

# Validation Strategy: Fail-Fast vs Fail-Safe

**Fail-Fast** (Pydantic default):
- Invalid data raises exception immediately
- Stops pipeline at first error
- **Use when**: Training pipeline, data must be perfect

**Fail-Safe**:
- Log errors, skip invalid records
- Continue processing
- **Use when**: Production inference, can't reject all data

```python
# Fail-safe pattern
valid_movies = []
for raw_data in data:
    try:
        movie = Movie(**raw_data)
        valid_movies.append(movie)
    except ValidationError as e:
        logger.warning(f"Invalid movie: {e}")
        continue
```

---

# Part 3: Cleaning with Pandas

**Handling Missing Data**:

1.  **Drop**: If label is missing, drop row. `df.dropna(subset=['rating'])`
2.  **Impute**: If feature is missing, fill with mean/median. `df.fillna(df.mean())`
3.  **Flag**: Create a boolean column `is_missing`.

**Type Conversion**:
```python
# Force numeric, turn errors ('N/A') into NaN
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
```

---

# Missing Data Strategies

**Types of missingness**:

1. **MCAR** (Missing Completely At Random): Random, unrelated to data
   - **Strategy**: Impute with mean/median or drop

2. **MAR** (Missing At Random): Related to other observed variables
   - **Strategy**: Model-based imputation (KNN, regression)

3. **MNAR** (Missing Not At Random): Related to unobserved data
   - **Strategy**: Domain-specific imputation or drop

**Impact on ML**:
- Some algorithms (XGBoost, CatBoost) handle missing values natively
- Others (linear regression, SVM) require imputation

---

# Imputation Techniques

**Simple imputation**:
```python
# Mean for numeric features
df['age'].fillna(df['age'].mean(), inplace=True)

# Median (robust to outliers)
df['income'].fillna(df['income'].median(), inplace=True)

# Mode for categorical
df['country'].fillna(df['country'].mode()[0], inplace=True)

# Forward/backward fill (time series)
df['temperature'].fillna(method='ffill', inplace=True)
```

**Advanced imputation**:
```python
from sklearn.impute import KNNImputer, SimpleImputer

# KNN-based (uses similar rows)
imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df)

# Iterative (model each feature)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer = IterativeImputer()
df_imputed = imputer.fit_transform(df)
```

---

# Outlier Detection

**Statistical methods**:

**Z-score**:
```python
from scipy import stats
import numpy as np

z_scores = np.abs(stats.zscore(df['price']))
outliers = df[z_scores > 3]  # More than 3 std dev
```

**IQR (Interquartile Range)**:
```python
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

outliers = df[
    (df['price'] < Q1 - 1.5 * IQR) |
    (df['price'] > Q3 + 1.5 * IQR)
]
```

---

# Outlier Handling Strategies

**1. Remove outliers**:
```python
# Keep only values within 3 std dev
df = df[np.abs(stats.zscore(df['price'])) < 3]
```

**2. Cap outliers** (winsorization):
```python
# Cap at 5th and 95th percentile
lower = df['price'].quantile(0.05)
upper = df['price'].quantile(0.95)
df['price'] = df['price'].clip(lower, upper)
```

**3. Transform** (log, sqrt):
```python
# Log transform reduces impact of outliers
df['price_log'] = np.log1p(df['price'])
```

**4. Keep but flag**:
```python
df['is_outlier'] = np.abs(stats.zscore(df['price'])) > 3
```

---

# Data Type Validation and Conversion

**Check data types**:
```python
df.dtypes  # Show current types
df.info()  # Detailed type information
```

**Convert types**:
```python
# Numeric conversion
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Datetime conversion
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

# Category (memory efficient)
df['genre'] = df['genre'].astype('category')

# Boolean
df['is_released'] = df['is_released'].astype(bool)
```

**Handle mixed types**:
```python
# Find rows with mixed types
mixed = df[df['price'].apply(lambda x: isinstance(x, str))]

# Clean before conversion
df['price'] = df['price'].str.replace('$', '').str.replace(',', '')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
```

---

# Duplicate Detection and Handling

**Find duplicates**:
```python
# Exact duplicates
duplicates = df[df.duplicated()]

# Duplicates based on specific columns
duplicates = df[df.duplicated(subset=['title', 'year'])]

# Keep only first occurrence
df_clean = df.drop_duplicates()

# Keep last occurrence
df_clean = df.drop_duplicates(keep='last')
```

**Fuzzy duplicates** (similar but not identical):
```python
from fuzzywuzzy import fuzz

# Find similar titles
for i, row1 in df.iterrows():
    for j, row2 in df.iterrows():
        if i < j:
            ratio = fuzz.ratio(row1['title'], row2['title'])
            if ratio > 90:  # 90% similar
                print(f"Potential duplicate: {row1['title']} ~ {row2['title']}")
```

---

# Data Consistency Checks

**Logical constraints**:
```python
# Age should be positive
assert (df['age'] >= 0).all(), "Found negative ages"

# Dates should be in order
assert (df['end_date'] >= df['start_date']).all(), "End before start"

# Percentages should sum to 100
assert df[['cat1', 'cat2', 'cat3']].sum(axis=1).between(99, 101).all()
```

**Cross-field validation**:
```python
# Birth year should match age
df['calculated_age'] = 2024 - df['birth_year']
inconsistent = df[abs(df['age'] - df['calculated_age']) > 1]

# Price should be positive
assert (df['price'] > 0).all(), "Found non-positive prices"
```

---

# Part 4: Batch Validation (Pandera)

**Pydantic** checks one object at a time.
**Pandera** checks the entire DataFrame (statistical checks).

```python
import pandera as pa

schema = pa.DataFrameSchema({
    "rating": pa.Column(float, checks=[
        pa.Check.ge(0),
        pa.Check.le(10),
        # Mean rating should be reasonable
        pa.Check.mean_in_range(5, 9)
    ]),
    "year": pa.Column(int, checks=pa.Check.gt(1900)),
})

schema.validate(df)
```

---

# Pandera: Statistical Checks

**Built-in checks**:
```python
schema = pa.DataFrameSchema({
    "price": pa.Column(float, checks=[
        pa.Check.greater_than(0),
        pa.Check.less_than(1000000),
        pa.Check.isin([10.0, 20.0, 30.0]),  # Only these values
        pa.Check.str_startswith("$"),  # For string columns
    ]),
    "category": pa.Column(str, checks=[
        pa.Check.isin(["A", "B", "C"]),
        pa.Check.str_length(1, 10),
    ])
})
```

**Custom checks**:
```python
# Check that 95% of values are within 2 std dev
def check_normal_distribution(series):
    z_scores = np.abs(stats.zscore(series))
    return (z_scores < 2).sum() / len(series) >= 0.95

schema = pa.DataFrameSchema({
    "rating": pa.Column(float, checks=pa.Check(check_normal_distribution))
})
```

---

# Pandera: DataFrame-Level Checks

**Multi-column checks**:
```python
@pa.check("price", "discount", name="discount_validation")
def check_discount(price, discount):
    return discount < price

schema = pa.DataFrameSchema(
    columns={
        "price": pa.Column(float),
        "discount": pa.Column(float),
    },
    checks=check_discount
)
```

**Row-wise checks**:
```python
def check_row_sum(df):
    return (df[['col1', 'col2', 'col3']].sum(axis=1) == 100).all()

schema = pa.DataFrameSchema(
    columns={"col1": pa.Column(float), ...},
    checks=pa.Check(check_row_sum, element_wise=False)
)
```

---

# Part 5: Data Drift

**The Silent Killer**.
Data valid today might be invalid tomorrow *statistically*.

**Three types of drift**:

1. **Schema Drift**: Structure changes
   - Field name changes (`imdbRating` → `rating`)
   - New fields added/removed
   - Type changes (int → float)

2. **Data Drift**: Input distribution changes
   - Feature statistics change (mean, variance)
   - New categories appear
   - Example: Users suddenly review only bad movies

3. **Concept Drift**: X→Y relationship changes
   - Target distribution shifts
   - Example: "Horror" movies become popular in Summer

---

# Detecting Schema Drift

**Compare schemas**:
```python
import pandas as pd

# Load reference schema
ref_df = pd.read_csv('training_data.csv')
new_df = pd.read_csv('production_data.csv')

# Check column names
missing_cols = set(ref_df.columns) - set(new_df.columns)
new_cols = set(new_df.columns) - set(ref_df.columns)

if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

# Check data types
for col in ref_df.columns:
    if col in new_df.columns:
        if ref_df[col].dtype != new_df[col].dtype:
            print(f"Type mismatch in {col}: {ref_df[col].dtype} vs {new_df[col].dtype}")
```

---

# Detecting Data Drift

**Statistical tests**:

**Kolmogorov-Smirnov test** (continuous variables):
```python
from scipy.stats import ks_2samp

# Compare distributions
statistic, p_value = ks_2samp(
    ref_df['rating'],
    new_df['rating']
)

if p_value < 0.05:
    print("Distribution has changed significantly")
```

**Chi-square test** (categorical variables):
```python
from scipy.stats import chi2_contingency

# Compare category distributions
ref_counts = ref_df['genre'].value_counts()
new_counts = new_df['genre'].value_counts()

statistic, p_value, _, _ = chi2_contingency([ref_counts, new_counts])

if p_value < 0.05:
    print("Category distribution has changed")
```

---

# Monitoring Data Drift

**Track key metrics over time**:
```python
import json
from datetime import datetime

def log_data_statistics(df, dataset_name):
    stats = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_name,
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "numeric_stats": {
            col: {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
            for col in df.select_dtypes(include=['number']).columns
        },
        "missing_pct": (df.isnull().sum() / len(df) * 100).to_dict()
    }

    with open('data_stats.jsonl', 'a') as f:
        f.write(json.dumps(stats) + '\n')
```

---

# Data Drift Response Strategies

**When drift detected**:

1. **Retrain model**: Use recent data
2. **Adjust preprocessing**: Update scalers, encoders
3. **Alert humans**: Investigate root cause
4. **Reject data**: If too different, don't use
5. **Adaptive models**: Use online learning

**Tools**:
- **Evidently AI**: Drift detection and reports
- **Alibi Detect**: Statistical drift tests
- **Great Expectations**: Data validation framework

*Detailed monitoring covered in Week 14.*

---

# Summary: The Checklist

Before training a model, ask:

1.  [ ] **Schema**: Do all fields exist with correct types? (Pydantic)
2.  [ ] **Nulls**: How are missing values handled?
3.  [ ] **Ranges**: Are numbers within physical bounds? (Age > 0)
4.  [ ] **Duplicates**: Are primary keys unique?
5.  [ ] **Stats**: Does the distribution look normal? (Pandera)

**Lab**: You will build a script that takes a raw JSON dump and produces a clean, validated CSV ready for training.

```