# legacy/

The original (v0) implementation of this project: the modular pipeline (`src/`), FastAPI app (`api/`), Streamlit dashboard (`dashboard/`), scripts and launchers, as they were before the refinement. The same state is available at git tag `v0-legacy`.

## Why it is kept

For transparency and as an audit trail. The v0 results were retracted because of target leakage and related defects; see [ADR-0002](../docs/adr/0002-legacy-results-retracted.md). Keeping the code lets anyone verify those findings (for example `legacy/src/features.py::get_feature_columns`).

## Status

- Frozen: no fixes or features. New work happens in `src/volforecast/`.
- Not supported and not to be used for conclusions or as a template.
- Excluded from lint (`ruff`) and type checks (`mypy`) via `pyproject.toml`, and not part of the test suite.
