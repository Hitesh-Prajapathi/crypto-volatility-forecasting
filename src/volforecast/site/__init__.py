"""Static public dashboard (GitHub Pages): builds ``_site/`` from the forecast store.

    uv run python -m volforecast.site --out _site --database-url sqlite:///store.db

See :func:`volforecast.site.build.build_site`.
"""

from __future__ import annotations

from volforecast.site.build import build_site, parse_backtest_tables

__all__ = ["build_site", "parse_backtest_tables"]
