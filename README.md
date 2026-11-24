# Piano take key-set classifier

Decides whether two recorded attempts of a piano exercise use the **same set of
keys**, reports witnesses for every case the evidence allows, and lets an editor
confirm or correct individual key presses against a **versioned** report.

Built with FastAPI + SciPy (HiGHS MILP). Start with Docker Compose.

## Run

```bash
docker compose up --build
# service: http://127.0.0.1:8000  (interactive docs at /docs)
python demo/run_demo.py                 # runs all demo beats against :8000
```

Local dev without Docker:

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000
python -m pytest -q                     # 29 tests
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/analyses` | Create an analysis; returns report version 1 |
| `GET` | `/analyses/{id}?version=N` | Fetch a report version (default: current) |
| `GET` | `/analyses/{id}/versions` | List report versions |
| `GET` | `/analyses/{id}/confirmations` | Current editor confirmations |
| `POST` | `/analyses/{id}/confirmations` | Apply/confirm or correct edits against `expected_version` |

`POST /analyses` body: `key_names` (≤ 24), `band_names`, `templates` (keys ×
bands), `band_energy_1/2`, `band_error_1/2`, `gain_lower`, `gain_upper`,
optional `time_limit`. Gains are closed intervals and must be strictly positive
(a played key scales the whole template by a positive gain; a zero lower bound
would make "played at zero gain" indistinguishable from "not played").

Report body highlights:

- `kind`: `same` (certainly identical sets), `different` (certainly different),
  `both` (either case is possible), `impossible` (evidence conflicts).
- `possibilities[]`: one entry per feasible hypothesis, each holding both takes
  with their `pressed_keys` and per-key `gains`.
- `evidence`: solver feasibility per hypothesis; for `impossible`, both models'
  diagnostics (`reason: conflicting_evidence`).
- `confirmations`: the editor state this version was computed from.

## Model

Each key `k` has a non-negative template `T[k,:]`. For a take with gain vector
`g >= 0` (zero for unplayed keys, `lo[k] <= g[k] <= hi[k]` for played keys) the
bands are `g @ T` and must satisfy `|g @ T - b| <= e` for every band. The two
takes may use different gains.

Two MILPs are solved; neither enumerates key subsets:

- **same** — one shared binary vector `x`, per-take gains `g1,g2`, linked by
  `lo[k]·x[k] <= g[k] <= hi[k]·x[k]`.
- **different** — two binary vectors `x1,x2`, with `d[k] = |x1[k]-x2[k]|`
  linearised and `sum(d) >= 1`.

Outcome from feasibility:

| same | different | kind |
| --- | --- | --- |
| feasible | infeasible | `same` |
| infeasible | feasible | `different` |
| feasible | feasible | `both` |
| infeasible | infeasible | `impossible` |

Witnesses are extracted and re-verified numerically (closed-interval gains,
band residuals, unplayed gains exactly zero). An undecided model (time limit)
raises rather than guessing.

## Editing semantics

- The editor submits confirmations with the report `expected_version`.
- The service recomputes from the **full** confirmation set: previous
  confirmations merged with this request's edits (a repeated key with the same
  value is idempotent; a changed value is a correction).
- The new report version and the merged confirmations take effect **together**,
  atomically, only after recomputation succeeds.
- A stale `expected_version` is rejected (409 `stale_version`); the original
  confirmations and reports remain queryable.
- If the confirmed edits make the problem infeasible, the request is rejected
  (409 `infeasible_after_confirmations`) with evidence, and nothing is applied.
- Confirmations that force take 0 and take 1 to different states for a key are
  not an input error: they make the same-set hypothesis infeasible and let the
  different-set hypothesis answer (or produce `impossible` if nothing fits).

## Demo beats (`demo/`)

1. `01_same.json` → `same`, both takes use C4+G4.
2. `02_different.json` → `different` (C4+G4 vs E4+G4).
3. `03_impossible.json` → `impossible`, conflicting evidence.
4. `04_ambiguous.json` → `both`; apply `04_confirmation.json` to edit, then a
   replayed request with the same `expected_version` returns `stale_version`
   while version 1 stays retrievable.

## Implementation notes / limits

- State is in-memory (single process); restarting the service clears analyses.
  The store is lock-protected, and the solver runs outside the lock with a
  version re-check before commit.
- Solver time limit defaults to 10 s per model (`time_limit` field, ≤ 120 s).
- Templates must be non-negative and finite; band errors non-negative.
