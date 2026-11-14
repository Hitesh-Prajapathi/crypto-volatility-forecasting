"""Repository over the forecast store: idempotent upserts and the read helpers the API uses.

All write methods are safe to repeat (upserts on natural keys; dialect-aware ``ON CONFLICT`` for
sqlite and PostgreSQL). Values are sanitized before hitting the driver: numpy scalars become
Python scalars, ``NaN``/``inf`` become NULL, pandas ``Timestamp`` becomes ``date``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from volforecast.contracts import COL_COMPLETE, COL_RET, COL_RV
from volforecast.forecast_store.models import (
    SERVED_ROLES,
    Base,
    Forecast,
    MonitoringMetric,
    PipelineRun,
    PromotionDecision,
    Realized,
    utcnow,
)
from volforecast.logging import get_logger

log = get_logger(__name__)

REALIZED_COLUMNS: tuple[str, ...] = ("rv", "bpv", "rs_pos", "rs_neg", "rq", "ret")
# SQLite caps bound parameters per statement (999 on old builds): stay under it.
_MAX_PARAMS = 900


def clean_value(x: Any) -> Any:
    """Convert ``x`` to something every DB driver / the JSON encoder accepts."""
    if x is None or x is pd.NaT:
        return None
    if isinstance(x, pd.Timestamp):
        return x.date()
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, float | np.floating):
        f = float(x)
        return f if math.isfinite(f) else None
    if isinstance(x, np.ndarray):
        return [clean_value(v) for v in x.tolist()]
    if isinstance(x, Mapping):
        return {str(k): clean_value(v) for k, v in x.items()}
    if isinstance(x, list | tuple | set):
        return [clean_value(v) for v in x]
    return x


def clean_json(x: Any) -> Any:
    """Like :func:`clean_value`, but dates/datetimes become ISO strings (JSON columns)."""
    v = clean_value(x)
    if isinstance(v, dict):
        return {k: clean_json(val) for k, val in v.items()}
    if isinstance(v, list):
        return [clean_json(val) for val in v]
    if isinstance(v, datetime | date):
        return v.isoformat()
    return v


def to_date(x: Any) -> date:
    """Coerce str / Timestamp / datetime / date to a ``date``."""
    if isinstance(x, pd.Timestamp):
        return x.date()
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return pd.Timestamp(x).date()


def make_engine(url: str | Path) -> Engine:
    """Engine for ``url``; creates the parent directory of a file-based sqlite database."""
    text = str(url)
    parsed = make_url(text)
    if parsed.get_backend_name() == "sqlite" and parsed.database not in (None, "", ":memory:"):
        Path(str(parsed.database)).expanduser().parent.mkdir(parents=True, exist_ok=True)
    return create_engine(text, future=True, pool_pre_ping=True)


def init_db(url_or_engine: str | Engine) -> Engine:
    """Create all tables (idempotent) and return the engine.

    ``create_all`` never alters existing tables; see :mod:`volforecast.forecast_store.models`.
    """
    engine = make_engine(url_or_engine) if isinstance(url_or_engine, str) else url_or_engine
    Base.metadata.create_all(engine)
    return engine


def _dedupe(rows: list[dict[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for r in rows:
        seen[tuple(r[k] for k in keys)] = r  # last one wins
    return list(seen.values())


class ForecastRepository:
    """Data access for the forecast store. ``url`` is a SQLAlchemy URL or an Engine."""

    def __init__(self, url: str | Path | Engine, *, create: bool = True) -> None:
        self.engine = make_engine(url) if not isinstance(url, Engine) else url
        if create:
            Base.metadata.create_all(self.engine)

    # ------------------------------------------------------------------ plumbing
    @contextmanager
    def session(self) -> Iterator[Session]:
        """Transactional session (commit on success, rollback on error)."""
        with Session(self.engine, expire_on_commit=False) as s, s.begin():
            yield s

    def _upsert(
        self,
        session: Session,
        model: type[Base],
        rows: list[dict[str, Any]],
        key_cols: Sequence[str],
        update_cols: Sequence[str],
    ) -> int:
        if not rows:
            return 0
        rows = _dedupe(rows, key_cols)
        table: Any = model.__table__
        dialect = session.get_bind().dialect.name
        if dialect == "postgresql":
            ins: Any = pg_insert
        elif dialect == "sqlite":
            ins = sqlite_insert
        else:
            raise NotImplementedError(f"upsert not implemented for dialect {dialect!r}")
        chunk = max(1, _MAX_PARAMS // max(len(r) for r in rows))
        for i in range(0, len(rows), chunk):
            stmt = ins(table).values(rows[i : i + chunk])
            set_ = {c: stmt.excluded[c] for c in update_cols}
            if set_:
                stmt = stmt.on_conflict_do_update(index_elements=list(key_cols), set_=set_)
            else:
                stmt = stmt.on_conflict_do_nothing(index_elements=list(key_cols))
            session.execute(stmt)
        return len(rows)

    # ------------------------------------------------------------------ forecasts
    _FORECAST_KEY = ("asset", "horizon", "model_version", "origin_date")
    _FORECAST_UPDATE = (
        "model_name",
        "role",
        "target_date",
        "forecast_var",
        "var95",
        "var99",
        "es975",
        "dist",
        "data_hash",
        "created_at",
    )

    def upsert_forecasts(self, records: Iterable[Mapping[str, Any]]) -> int:
        """Insert or update forecast rows (key: asset, horizon, model_version, origin_date)."""
        rows: list[dict[str, Any]] = []
        now = utcnow()
        for rec in records:
            r = {k: clean_value(v) for k, v in rec.items()}
            r["origin_date"] = to_date(r["origin_date"])
            r["target_date"] = to_date(r["target_date"])
            r["horizon"] = int(r["horizon"])
            r["model_version"] = str(r["model_version"])
            r.setdefault("role", "champion")
            for c in ("var95", "var99", "es975", "dist", "data_hash"):
                r.setdefault(c, None)
            r["created_at"] = now
            rows.append(r)
        with self.session() as s:
            return self._upsert(s, Forecast, rows, self._FORECAST_KEY, self._FORECAST_UPDATE)

    def forecasts_df(
        self,
        asset: str,
        horizon: int,
        roles: Sequence[str] | None = None,
        start: date | str | None = None,
        end: date | str | None = None,
        model_version: str | None = None,
    ) -> pd.DataFrame:
        """Forecast rows as a DataFrame sorted by origin date (empty frame if none)."""
        stmt = select(Forecast).where(Forecast.asset == asset, Forecast.horizon == horizon)
        if roles is not None:
            stmt = stmt.where(Forecast.role.in_(list(roles)))
        if model_version is not None:
            stmt = stmt.where(Forecast.model_version == model_version)
        if start is not None:
            stmt = stmt.where(Forecast.origin_date >= to_date(start))
        if end is not None:
            stmt = stmt.where(Forecast.origin_date <= to_date(end))
        stmt = stmt.order_by(Forecast.origin_date, Forecast.created_at)
        with self.session() as s:
            rows = [_row_dict(r) for r in s.scalars(stmt)]
        cols = [c.name for c in Forecast.__table__.columns]  # type: ignore[attr-defined]
        df = pd.DataFrame(rows, columns=cols)
        for c in ("origin_date", "target_date"):
            df[c] = pd.to_datetime(df[c])
        return df

    def latest_forecast(self, asset: str, horizon: int) -> dict[str, Any] | None:
        """Most recent *served* (champion / fallback) forecast, or None."""
        stmt = (
            select(Forecast)
            .where(
                Forecast.asset == asset,
                Forecast.horizon == horizon,
                Forecast.role.in_(SERVED_ROLES),
            )
            .order_by(Forecast.origin_date.desc(), Forecast.created_at.desc(), Forecast.id.desc())
            .limit(1)
        )
        with self.session() as s:
            row = s.scalars(stmt).first()
            return None if row is None else _row_dict(row)

    def forecast_history(
        self,
        asset: str,
        horizon: int,
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pd.DataFrame:
        """Served forecasts joined with realized outcomes, one row per origin date.

        Adds ``realized_var`` (sum of rv over the h target days; NaN until all are known and
        complete) and ``realized_ret`` (log return of day origin+1, the VaR scoring target).
        """
        fc = self.forecasts_df(asset, horizon, roles=SERVED_ROLES, start=start, end=end)
        if fc.empty:
            fc["realized_var"] = pd.Series(dtype=float)
            fc["realized_ret"] = pd.Series(dtype=float)
            return fc
        # latest write wins when two served versions exist on one origin (e.g. promotion day)
        fc = fc.sort_values(["origin_date", "created_at"]).drop_duplicates(
            "origin_date", keep="last"
        )
        lo = fc["origin_date"].min()
        hi = fc["origin_date"].max() + pd.Timedelta(days=horizon)
        real = self.get_realized(asset, start=lo, end=hi)
        fc = fc.reset_index(drop=True)
        fc["realized_var"] = realized_horizon_sum(real, fc["origin_date"], horizon)
        ret = real[COL_RET] if not real.empty else pd.Series(dtype=float)
        nxt = pd.DatetimeIndex(fc["origin_date"]) + pd.Timedelta(days=1)
        fc["realized_ret"] = ret.reindex(nxt).to_numpy()
        return fc

    def models_summary(self) -> list[dict[str, Any]]:
        """One row per (asset, horizon): the served model, latest origin, shadow challenger."""
        out: list[dict[str, Any]] = []
        with self.session() as s:
            pairs = s.execute(
                select(Forecast.asset, Forecast.horizon)
                .distinct()
                .order_by(Forecast.asset, Forecast.horizon)
            ).all()
            for asset, horizon in pairs:
                served = s.scalars(
                    select(Forecast)
                    .where(
                        Forecast.asset == asset,
                        Forecast.horizon == horizon,
                        Forecast.role.in_(SERVED_ROLES),
                    )
                    .order_by(Forecast.origin_date.desc(), Forecast.created_at.desc())
                    .limit(1)
                ).first()
                shadow = s.scalars(
                    select(Forecast)
                    .where(
                        Forecast.asset == asset,
                        Forecast.horizon == horizon,
                        Forecast.role == "challenger",
                    )
                    .order_by(Forecast.origin_date.desc(), Forecast.created_at.desc())
                    .limit(1)
                ).first()
                n = s.scalar(
                    select(func.count())
                    .select_from(Forecast)
                    .where(
                        Forecast.asset == asset,
                        Forecast.horizon == horizon,
                        Forecast.role.in_(SERVED_ROLES),
                    )
                )
                if served is None:
                    continue
                out.append(
                    {
                        "asset": asset,
                        "horizon": int(horizon),
                        "model_name": served.model_name,
                        "model_version": served.model_version,
                        "role": served.role,
                        "latest_origin_date": served.origin_date,
                        "n_forecasts": int(n or 0),
                        "challenger_version": None if shadow is None else shadow.model_version,
                        "challenger_model": None if shadow is None else shadow.model_name,
                    }
                )
        return out

    # ------------------------------------------------------------------ realized
    def upsert_realized(self, asset: str, panel: pd.DataFrame) -> int:
        """Upsert the realized columns of a daily panel (index = date)."""
        rows: list[dict[str, Any]] = []
        for ts, rec in panel.iterrows():
            row: dict[str, Any] = {"asset": asset, "date": to_date(ts)}
            for c in REALIZED_COLUMNS:
                row[c] = clean_value(rec.get(c))
            row["complete"] = bool(rec.get(COL_COMPLETE, True))
            rows.append(row)
        with self.session() as s:
            return self._upsert(
                s, Realized, rows, ("asset", "date"), (*REALIZED_COLUMNS, "complete")
            )

    def get_realized(
        self,
        asset: str,
        start: date | str | pd.Timestamp | None = None,
        end: date | str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Realized measures as a daily DataFrame indexed by ``date`` (Timestamp, sorted)."""
        stmt = select(Realized).where(Realized.asset == asset)
        if start is not None:
            stmt = stmt.where(Realized.date >= to_date(start))
        if end is not None:
            stmt = stmt.where(Realized.date <= to_date(end))
        stmt = stmt.order_by(Realized.date)
        with self.session() as s:
            rows = [_row_dict(r) for r in s.scalars(stmt)]
        df = pd.DataFrame(rows, columns=["asset", "date", *REALIZED_COLUMNS, "complete"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop(columns="asset").set_index("date")
        df.index.name = "date"
        for c in REALIZED_COLUMNS:
            df[c] = df[c].astype(float)
        df["complete"] = df["complete"].astype(bool)
        return df

    def last_realized_date(self, asset: str) -> date | None:
        with self.session() as s:
            return s.scalar(select(func.max(Realized.date)).where(Realized.asset == asset))

    # ------------------------------------------------------------------ pipeline runs
    def start_run(
        self, flow: str, run_date: date | str | None = None, detail: Mapping[str, Any] | None = None
    ) -> int:
        with self.session() as s:
            run = PipelineRun(
                flow=flow,
                run_date=None if run_date is None else to_date(run_date),
                started_at=utcnow(),
                status="running",
                detail=clean_json(dict(detail or {})),
            )
            s.add(run)
            s.flush()
            return int(run.id)

    def finish_run(self, run_id: int, status: str, detail: Mapping[str, Any] | None = None) -> None:
        with self.session() as s:
            run = s.get(PipelineRun, run_id)
            if run is None:
                raise KeyError(f"unknown pipeline run {run_id}")
            run.status = status
            run.finished_at = utcnow()
            if detail is not None:
                merged = {**(run.detail or {}), **clean_json(dict(detail))}
                run.detail = merged

    def last_success(self) -> dict[str, datetime]:
        """Latest ``finished_at`` (UTC) of a successful run per flow."""
        with self.session() as s:
            rows = s.execute(
                select(PipelineRun.flow, func.max(PipelineRun.finished_at))
                .where(PipelineRun.status == "success")
                .group_by(PipelineRun.flow)
            ).all()
        return {str(f): _as_utc(t) for f, t in rows if t is not None}

    def pipeline_runs(self, flow: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        stmt = select(PipelineRun).order_by(PipelineRun.id.desc()).limit(limit)
        if flow is not None:
            stmt = stmt.where(PipelineRun.flow == flow)
        with self.session() as s:
            return [_row_dict(r) for r in s.scalars(stmt)]

    # ------------------------------------------------------------------ promotion decisions
    def add_promotion_decision(self, session: Session, **fields: Any) -> PromotionDecision:
        """Add a decision inside the caller's session (so the alias move can share its fate)."""
        fields["as_of_date"] = to_date(fields["as_of_date"])
        fields["reasons"] = clean_json(list(fields["reasons"]))
        fields["metrics"] = clean_json(dict(fields["metrics"]))
        row = PromotionDecision(**fields)
        session.add(row)
        session.flush()
        return row

    def promotion_decisions(
        self, asset: str | None = None, horizon: int | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        stmt = select(PromotionDecision).order_by(PromotionDecision.id.desc()).limit(limit)
        if asset is not None:
            stmt = stmt.where(PromotionDecision.asset == asset)
        if horizon is not None:
            stmt = stmt.where(PromotionDecision.horizon == horizon)
        with self.session() as s:
            return [_row_dict(r) for r in s.scalars(stmt)]

    def has_decision(self, asset: str, horizon: int, as_of_date: date | str) -> bool:
        with self.session() as s:
            n = s.scalar(
                select(func.count())
                .select_from(PromotionDecision)
                .where(
                    PromotionDecision.asset == asset,
                    PromotionDecision.horizon == horizon,
                    PromotionDecision.as_of_date == to_date(as_of_date),
                )
            )
        return bool(n)

    # ------------------------------------------------------------------ monitoring metrics
    def upsert_metrics(self, rows: Iterable[Mapping[str, Any]]) -> int:
        """Upsert ``MonitoringMetric`` rows (non-finite values are dropped)."""
        out: list[dict[str, Any]] = []
        now = utcnow()
        for rec in rows:
            value = clean_value(rec["value"])
            if value is None:
                continue
            out.append(
                {
                    "asset": str(rec["asset"]),
                    "horizon": int(rec.get("horizon", 0)),
                    "model": str(rec.get("model", "")),
                    "metric": str(rec["metric"]),
                    "window": int(rec.get("window", 0)),
                    "as_of_date": to_date(rec["as_of_date"]),
                    "value": float(value),
                    "created_at": now,
                }
            )
        key = ("asset", "horizon", "model", "metric", "window", "as_of_date")
        with self.session() as s:
            return self._upsert(s, MonitoringMetric, out, key, ("value", "created_at"))

    def get_metrics(
        self,
        metric: str,
        asset: str | None = None,
        horizon: int | None = None,
        model: str | None = None,
        window: int | None = None,
        start: date | str | None = None,
    ) -> pd.DataFrame:
        stmt = select(MonitoringMetric).where(MonitoringMetric.metric == metric)
        if asset is not None:
            stmt = stmt.where(MonitoringMetric.asset == asset)
        if horizon is not None:
            stmt = stmt.where(MonitoringMetric.horizon == horizon)
        if model is not None:
            stmt = stmt.where(MonitoringMetric.model == model)
        if window is not None:
            stmt = stmt.where(MonitoringMetric.window == window)
        if start is not None:
            stmt = stmt.where(MonitoringMetric.as_of_date >= to_date(start))
        stmt = stmt.order_by(MonitoringMetric.as_of_date)
        with self.session() as s:
            rows = [_row_dict(r) for r in s.scalars(stmt)]
        cols = [c.name for c in MonitoringMetric.__table__.columns]  # type: ignore[attr-defined]
        df = pd.DataFrame(rows, columns=cols)
        df["as_of_date"] = pd.to_datetime(df["as_of_date"])
        return df


def _row_dict(obj: Any) -> dict[str, Any]:
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}


def _as_utc(t: datetime) -> datetime:
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def realized_horizon_sum(real: pd.DataFrame, origins: Iterable[Any], h: int) -> np.ndarray:
    """sum_{k=1..h} rv_{origin+k} per origin; NaN if any day is missing / incomplete."""
    idx = pd.DatetimeIndex(list(origins))
    if real.empty:
        return np.full(len(idx), np.nan)
    rv = real[COL_RV].astype(float)
    ok = real[COL_COMPLETE].astype(bool)
    total = np.zeros(len(idx))
    for k in range(1, h + 1):
        day = idx + pd.Timedelta(days=k)
        vals = rv.reindex(day).to_numpy()
        good = ok.reindex(day, fill_value=False).to_numpy(dtype=bool)
        total = total + np.where(good, vals, np.nan)
    return total
