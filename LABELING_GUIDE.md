# Entity Resolution Labeling Guide

## Understanding the `_l` and `_r` Suffixes

### Quick Reference

| Suffix | Meaning | Description | Examples |
|--------|---------|-------------|----------|
| **`_l`** | **REFERENCE** | Your dim_org database (known good data) | `unique_id_l`, `name_l`, `country_code_l` |
| **`_r`** | **CANDIDATE** | Incoming record needing ID assignment | `unique_id_r`, `name_r`, `country_code_r` |

---

## Detailed Explanation

### `_l` = REFERENCE RECORD (Left side in Splink)
**Source:** `allsci_prod_gold.dim_organization` table

**Purpose:** Your organization master database with known good identifiers

**Characteristics:**
- Has `ror_id` (Research Organization Registry ID)
- Has `grid_id` (Global Research Identifier Database ID)
- Curated, high-quality data
- Known aliases and locations
- This is your "source of truth"

**Example:**
```
unique_id_l: dim_ASC-OR-0000000041087-1.0-1724880247
name_l: Alvogen (South Korea)
country_code_l: KR
city_l: Seoul
```

---

### `_r` = CANDIDATE RECORD (Right side in Splink)
**Source:** `allsci_prod_gold.potential_mismatched_organizations` table

**Purpose:** Incoming records from external sources that need to be linked to your database

**Characteristics:**
- Comes from various sources (see `source_table` column)
- May have incomplete or inconsistent data
- Requires ID assignment to connect to dim_org
- May represent same org, subsidiary, branch, or different org

**Example:**
```
unique_id_r: mis_002cf792e88bbb56fb08e8f0c8af0f59
name_r: Alvogen Inc.
source_table: open_fda_silver.ndc_drugs
```

---

## Why This Matters

### The Matching Question
> **"Should the CANDIDATE record (`_r`) be assigned the same ID as the REFERENCE record (`_l`)?"**

### Common Scenarios

#### ✅ YES - Assign Same ID (Match)
- **Exact match:** "Pfizer" ↔ "Pfizer"
- **Legal suffix only:** "Apple Inc." ↔ "Apple"
- **Location qualifier:** "Bayer (Italy)" ↔ "Bayer"
- **Abbreviation:** "MIT" ↔ "Massachusetts Institute of Technology"
- **Department:** "Stanford Medicine" ↔ "Stanford University"

#### ❌ NO - Different IDs (No Match)
- **Subsidiary:** "Pfizer Canada" ↔ "Pfizer Inc." (different legal entities)
- **Branch:** "Memorial Sloan Kettering Westchester" ↔ "Memorial Sloan Kettering Cancer Center"
- **Different orgs:** "Apple Records" ↔ "Apple Inc."
- **Person vs org:** "Dr. John Smith" ↔ "Smith Medical Group"

---

## Hierarchy Handling

### The Hierarchy Rollup Challenge

Your pipeline needs to decide at what level in the organizational hierarchy to assign IDs:

```
Parent Company (HQ)
├── Regional Subsidiary (different legal entity)
│   ├── Branch Office
│   └── Department
└── Division
```

### Decision Rules

**Match to Parent (Roll Up):**
- Departments within an organization
- Schools within a university
- Research centers within an institution

**Don't Match (Keep Separate):**
- Regional subsidiaries (different legal entities)
- Branch locations (different physical locations)
- Affiliated but independent organizations

---

## Common Error Patterns

### Error Type 1: Truncated Responses
**Before:**
```
parse_error: Let's resolve this systematically: 1. **Core Identity Check**: - "Comprehensive Cancer Centers o
```

**Cause:** `max_tokens=200` was too low

**Fix:** ✅ Increased to `max_tokens=500`

---

### Error Type 2: Subsidiary Confusion
**Example:**
- REFERENCE: "Novartis AG" (Switzerland)
- CANDIDATE: "Novartis Canada"

**Common Mistake:** Matching these as SAME

**Correct:** NO MATCH - Canadian subsidiary is separate legal entity

**Why:** Different country codes, "Canada" indicates regional subsidiary

---

### Error Type 3: Branch Location Confusion
**Example:**
- REFERENCE: "Kaiser Permanente"
- CANDIDATE: "Kaiser Permanente - Denver"

**Common Mistake:** Matching as SAME

**Correct:** Depends on use case:
- If tracking locations separately → NO MATCH
- If just need parent org → YES MATCH

**Current pipeline:** Prefers NO MATCH for branches to maintain granularity

---

## Column Reference

### Essential Columns

```python
# Identifiers
unique_id_l          # dim_org ID (your database)
unique_id_r          # Candidate ID (incoming)

# Names
name_l               # dim_org name (e.g., "Pfizer (United States)")
name_r               # Candidate name (e.g., "Pfizer")
name_normalized_l    # Cleaned version
name_normalized_r    # Cleaned version

# Geographic
country_code_l       # ISO country code (reference)
country_code_r       # ISO country code (candidate)
city_l, city_r       # Cities

# Matching
match_probability    # Splink confidence (0-1)
llm_match           # LLM judgment (True/False/None)
llm_confidence      # LLM confidence (0-1)
llm_reason          # Explanation

# Metadata
source_table        # Where candidate came from
cluster_id          # Graph cluster assignment
```

---

## Updated LLM Prompt Labels

The LLM judge now uses clearer labels:

**Old (confusing):**
```
Record A (from source): "..."
Record B (reference database): "..."
```

**New (explicit):**
```
CANDIDATE RECORD (incoming data from {source} requiring ID assignment):
Name: "..."
Location: ...

REFERENCE RECORD (your organization database - known good data):
Name: "..."
Known Aliases: ...
Location: ...
```

This makes it crystal clear:
- **CANDIDATE** = what you're trying to match (`_r`)
- **REFERENCE** = what you're matching against (`_l`)

---

## Code Updates

### llm_judge.py Changes

1. **Increased max_tokens:** 200 → 500 to prevent truncation
2. **Clearer prompt labels:** CANDIDATE vs REFERENCE
3. **Better error handling:** Shows position of JSON parse errors
4. **Temperature=0:** For deterministic, consistent results
5. **Improved JSON extraction:** Handles markdown and embedded JSON

### Configuration

All settings in `config.py`:

```python
# LLM Judge
config.llm_judge.ENABLE_LLM_VALIDATION = True
config.llm_judge.PRIMARY_MODEL = "claude-sonnet-4-20250514"

# Active Learning (reduces API calls)
config.active_learning.ENABLE_ACTIVE_LEARNING = True
config.active_learning.VALIDATION_BUDGET = 500

# Multi-Agent (further reduces costs)
config.multi_agent.ENABLE_MULTI_AGENT = True
```

---

## Testing Your Changes

### 1. Check for Truncation Errors

```python
# In predictions.csv or after running inference
errors = predictions[predictions['llm_reason'].str.contains('parse_error', na=False)]
print(f"Parse errors: {len(errors)}")

# Should be 0 or very few with new max_tokens=500
```

### 2. Review Disagreements

```python
# Where Splink says YES but LLM says NO (or vice versa)
disagreements = predictions[
    (predictions['match_probability'] >= 0.95) &
    (predictions['llm_match'] == False)
]

# Look for patterns:
# - Subsidiaries?
# - Branches?
# - Different orgs with similar names?
```

### 3. Validate Hierarchy Decisions

```python
# Check branch/location matches
branches = predictions[
    predictions['name_r'].str.contains(' - ', na=False) |
    predictions['name_r'].str.contains('Branch', na=False, case=False)
]

# Review these carefully - are they matched correctly?
```

---

## Summary

**Key Changes:**
1. ✅ `max_tokens`: 200 → 500 (prevents truncation)
2. ✅ Labels: "Record A/B" → "CANDIDATE/REFERENCE"
3. ✅ Error handling: Better JSON parsing
4. ✅ Prompt: Explicit hierarchy guidance
5. ✅ Temperature: 0 (deterministic)

**Expected Results:**
- Fewer parse errors (from ~10% to <1%)
- Clearer LLM reasoning
- Better subsidiary/branch handling
- More consistent judgments

---

## Questions?

**Q: Should "Pfizer Canada" match "Pfizer Inc."?**
A: NO - Different legal entities (subsidiary vs parent)

**Q: Should "Stanford Medicine" match "Stanford University"?**
A: YES - Department within university

**Q: Should "Memorial Sloan Kettering Westchester" match "Memorial Sloan Kettering Cancer Center"?**
A: NO - Branch location vs main center

**Q: Should "Alvogen Inc." match "Alvogen (South Korea)"?**
A: MAYBE - Need to check if South Korea qualifier indicates subsidiary or just location. Default: YES if just location qualifier.

**Q: How do I change hierarchy rules?**
A: Edit the prompt in `llm_judge.py` → `build_prompt()` → "Hierarchy & Subsidiary Check" section

---

For more details, see:
- `IMPROVEMENTS.md` - Full feature documentation
- `config.py` - All configuration options
- `llm_judge.py` - Validation logic
