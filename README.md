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
| `GET /` | A small web page with a question box that calls `/ask` |
| `GET /health` | Liveness check |
| `GET /report` | The Markdown operations report |
| `GET /tickets?category=&priority=&status=&limit=` | Filter tickets (parameterised SQL, validated inputs) |
| `POST /classify` `{"summary": "..."}` | Baseline classifier, returns the category |
| `POST /ask` `{"question": "...", "k": 3}` | RAG over a small synthetic troubleshooting knowledge base; returns an answer with sources |

The API serves its own in-memory set of synthetic tickets (seeded, reproducible).

Run in Docker (serves the HTTP API on port 8080):

```bash
docker build -t ops-report .
docker run --rm -p 8080:8080 ops-report
curl localhost:8080/health

# CLI report instead of the API:
docker run --rm ops-report python -m ops_report --db /tmp/tickets.db --out /tmp/report.md
```

The image runs as a non-root user and reads `$PORT` (default 8080), so it runs unchanged on Google Cloud Run.

## What the report contains

- Ticket totals (resolved / open)
- Tickets by category
- Per priority (P1-P4): ticket count, MTTR, SLA breach rate
- Top repeating issues
- Baseline classifier accuracy against the recorded category

## RAG (retrieval-augmented generation)

`POST /ask` answers questions from a small synthetic knowledge base (`src/ops_report/knowledge/*.md`, 6 articles, 12 chunks).

1. **Chunk**: one chunk per `##` section, remembering its source file and heading.
2. **Embed**: TF-IDF vectors (standard library only).
3. **Retrieve**: cosine similarity between the question and every chunk, top-k above a minimum score.
4. **Answer**: the best chunk plus all sources. Below the score threshold it says it does not know instead of guessing.
5. `build_prompt()` shows how the same hits would be passed to an LLM with a "use only this context, cite sources" instruction. No LLM is called yet.

**Known limitation (on purpose, and tested by hand):** TF-IDF matches words, not meaning. "I can't sign in to my PC" retrieves a virtual-desktop *login* article instead of the locked-account article, because it shares a word and not the idea. The `Embedder` interface exists so a neural embedding model (for example a local `bge-m3` via Ollama, or a hosted API) can replace it without touching retrieval or the API. Next step: swap it in and compare accuracy on a small labelled question set, the same way the classifier is compared with its baseline.

### Measuring retrieval

`eval/questions.json` holds 20 labelled questions: 5 literal, 9 paraphrased, 6 Cantonese or mixed Chinese/English, plus 3 out of scope. Run:

```bash
PYTHONPATH=src python -m ops_report.rag_eval                    # TF-IDF baseline
PYTHONPATH=src python -m ops_report.rag_eval --embedder ollama  # neural embeddings
```

| Metric | TF-IDF | bge-m3 (Ollama) |
|---|---:|---:|
| top-1, answerable (17) | 55% | **95%** |
| literal wording | 100% | 100% |
| paraphrased wording | 44% | **89%** |
| Cantonese / mixed | 33% | **100%** |
| top-3, answerable | 65% | **100%** |
| refuses out-of-scope (3) | 100% | 100% |

TF-IDF cannot read Chinese at all (its tokenizer keeps only `[a-z0-9]`), and it scores unrelated paraphrases at zero. `bge-m3` is multilingual, so a Cantonese question finds the English article.

Caveats, stated plainly:
- 20 questions is small, and I wrote them while looking at the knowledge base, so treat the numbers as direction, not a benchmark.
- The refuse threshold for `bge-m3` (0.5) sits in the gap between the weakest correct match (0.54) and the strongest out-of-scope match (0.36) **on these same questions**. It is tuned on the evaluation set and needs fresh questions to confirm.
- One question still fails at top-1: "I can't sign in to my PC" returns the VPN article (the right one is in the top 3).
- The Cloud Run demo still uses TF-IDF, because the container has no Ollama. To use neural embeddings locally: `ollama pull bge-m3`, then `RAG_EMBEDDER=ollama PYTHONPATH=src uvicorn ops_report.api:app`.

## Deploy to Google Cloud Run

Live demo (synthetic data, scales to zero so the first request can be slow): https://ops-report-1015015901936.asia-east2.run.app/docs

Data is synthetic, so a public demo is safe. Cost is capped by scaling to zero and limiting instances.

```bash
gcloud run deploy ops-report \
  --source . \
  --region asia-east2 \
  --allow-unauthenticated \
  --max-instances 1
```

Set a budget alert on the billing account first.

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
- [x] RAG baseline (TF-IDF retrieval, sources, refuses unknown topics)
- [x] Neural embeddings (bge-m3 via Ollama) compared with TF-IDF on a labelled question set
- [ ] Neural embeddings in the deployed service (hosted embedding API or a container with a model)
- [ ] LLM answer generation on top of retrieval
- [ ] LLM-based classifier compared with the baseline on the same labels
- [ ] Treat overdue open tickets as SLA breaches
- [ ] HTML report and a scheduled daily run
- [x] Dockerfile serves the API (tested locally)
- [x] Deploy to Cloud Run (asia-east2) and publish the URL
- [ ] Deploy automatically from CI
