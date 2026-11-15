"""Container entrypoint: ``python -m volforecast.serving``.

* ``WEB_CONCURRENCY`` (default 1 here, 2 in the image) is the number of uvicorn workers.
* With more than one worker, Prometheus multiprocess mode is enabled: the directory
  (``PROMETHEUS_MULTIPROC_DIR``, default ``/tmp/volforecast-prom``) is wiped and set *before*
  uvicorn imports the app (and therefore ``prometheus_client``).
* SIGTERM/SIGINT: uvicorn stops accepting connections, drains in-flight requests (at most
  ``--timeout-graceful-shutdown`` seconds; keep it below the pod's terminationGracePeriodSeconds),
  then runs the lifespan shutdown (stops the reload task, disposes the DB engine).

This module must not import the app or ``prometheus_client`` at import time.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def prepare_multiprocess_dir(
    workers: int, default_dir: str = "/tmp/volforecast-prom"
) -> str | None:
    """Create/wipe the multiprocess dir when more than one worker runs; returns the path."""
    if workers <= 1 and not os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        return None
    path = Path(os.environ.get("PROMETHEUS_MULTIPROC_DIR") or default_dir)
    shutil.rmtree(path, ignore_errors=True)  # stale files of a previous run would be aggregated
    path.mkdir(parents=True, exist_ok=True)
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = str(path)
    return str(path)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m volforecast.serving")
    p.add_argument("--host", default=os.environ.get("VOLF_API_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("VOLF_API_PORT", "8000")))
    p.add_argument("--workers", type=int, default=int(os.environ.get("WEB_CONCURRENCY", "1")))
    p.add_argument(
        "--graceful-timeout",
        type=int,
        default=int(os.environ.get("VOLF_API_GRACEFUL_TIMEOUT_S", "25")),
    )
    args = p.parse_args(argv)
    workers = max(1, args.workers)
    prepare_multiprocess_dir(workers)

    import uvicorn

    uvicorn.run(
        "volforecast.serving.app:app",
        host=args.host,
        port=args.port,
        workers=workers,
        timeout_graceful_shutdown=args.graceful_timeout,
        access_log=False,  # the app emits one structured log line per request
        proxy_headers=False,  # client IP handling is explicit (VOLF_API_TRUSTED_PROXY_HOPS)
        server_header=False,
    )


if __name__ == "__main__":
    main()
