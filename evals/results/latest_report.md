# Eval report — 2026-10-06 10:39:55

Mode: **OFFLINE**

| Suite | Headline | Threshold | Status | Time / run |
|---|---|---|---|---|
| routing | accuracy = 100.0% | 90 | PASS | 0.0s · offline 2026-10-06 10:39:55 |
| retrieval | hit@1 = 95.0% | 85 | PASS | 0.0s · offline 2026-10-06 10:39:55 |
| json_parsing | pass_rate = 100.0% | 100 | PASS | 0.2s · offline 2026-10-06 10:39:55 |

## routing

- **accuracy**: 100.0
- **acc_assignment**: 100.0
- **acc_code_generate**: 100.0
- **acc_code_review**: 100.0
- **acc_fact_check**: 100.0
- **acc_grade**: 100.0
- **acc_interview**: 100.0
- **acc_research**: 100.0
- **acc_tutor**: 100.0

## retrieval

- **hit@1**: 95.0
- **hit@3**: 100.0
- **mrr**: 0.975
- **context_precision**: 88.3
- **offtopic_rejection**: 83.3

Failures (3):

- `k18` 'what is corrective RAG (CRAG)' → {"expected": ["Corrective RAG"], "got": ["Retrieval-augmented generation", "Corrective RAG"]}
- `k33` 'how to impute missing values' → {"expected": ["Missing values and outliers"], "got": ["Exploratory data analysis", "Missing values and outliers", "Python for data science"]}
- `k46` 'what is the best value for money laptop' → {"expected": [], "got": ["Hypothesis testing and p-values"]}

## json_parsing

- **pass_rate**: 100.0
