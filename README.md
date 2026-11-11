# ops-report-automation

Turns raw IT-support tickets into a daily operations report: volume by category, mean time to resolve (MTTR) and SLA breach rate by priority, repeating issues, and the accuracy of a baseline ticket classifier.

> **All data is synthetic.** Tickets are generated with a fixed random seed. No real customer, employer or government data is used or stored.

## Why this exists

Support teams spend time building the same daily report by hand. This project shows the full loop in a small, testable form: load tickets into SQLite, query them with plain SQL, write a Markdown report, and check a classifier against known labels. It was built from first-hand experience of 7x24 incident handling and from automating reporting with SQL and scripts.

## Quick start

```bash
# Python 3.10+, the CLI uses the standard library only
PYTHONPATH=src python3 -m ops_report --tickets 300 --seed 42
cat report.md
```

Run the tests (API tests need `pip install -r requirements.txt`; without it they are skipped):

```bash
python3 -m unittest discover -s tests -v
```

## HTTP API (FastAPI)

```bash
pip install -r requirements.txt
PYTHONPATH=src uvicorn ops_report.api:app --reload
# interactive docs: http://127.0.0.1:8000/docs
```

| Endpoint | What it does |
|---|---|
| `GET /health` | Liveness check |
| `GET /report` | The Markdown operations report |
| `GET /tickets?category=&priority=&status=&limit=` | Filter tickets (parameterised SQL, validated inputs) |
| `POST /classify` `{"summary": "..."}` | Baseline classifier, returns the category |

The API serves its own in-memory set of synthetic tickets (seeded, reproducible).

Run in Docker (CLI report, no extra dependencies):

```bash
docker build -t ops-report .
docker run --rm ops-report
```

## What the report contains

- Ticket totals (resolved / open)
- Tickets by category
- Per priority (P1-P4): ticket count, MTTR, SLA breach rate
- Top repeating issues
- Baseline classifier accuracy against the recorded category

## Design notes

- `data.py` generates reproducible synthetic tickets (same seed, same data).
- `db.py` owns the SQLite schema and loading.
- `classify.py` is a keyword-scoring baseline. It is deliberately simple so a better classifier can be compared against it.
- `report.py` computes everything with SQL (`julianday` arithmetic for resolution time).
- CI (`.github/workflows/ci.yml`) runs the tests and uploads a sample report as an artifact.

## Limitations

- Synthetic data: resolution times follow a log-normal distribution I chose, so the numbers show the mechanism, not real operational performance.
- The baseline classifier reaches about 94% on its own synthetic labels. That figure says little about real tickets, which are messier.
- SLA breach rate counts resolved tickets only; open tickets are not yet treated as breached.

## Roadmap

- [x] HTTP API (FastAPI) with tests
- [ ] LLM-based classifier compared with the baseline on the same labels
- [ ] Treat overdue open tickets as SLA breaches
- [ ] HTML report and a scheduled daily run
- [ ] Deploy to a cloud function and publish the report
