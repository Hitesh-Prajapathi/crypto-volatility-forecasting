# 0004. Partitioned Parquet with manifest content hashes instead of DVC

- Status: Accepted
- Date: 2026-11-13

## Context

Data must be versioned and verifiable so that any reported result can be tied to exact inputs. The raw data is public and re-downloadable (data.binance.vision publishes checksums); derived data is deterministic.

## Decision

Store data as Parquet partitioned by symbol and date, with a manifest file recording the SHA-256 content hash of every partition (and the source checksum for raw archives). Runs record the manifest hash. DuckDB reads the partitions. Data is not committed to git, and DVC is not used.

## Consequences

- No extra tool or remote to operate; the manifest is small, diffable text.
- Integrity checks are simple (recompute and compare hashes).
- No built-in pipeline caching or remote data sharing as DVC provides; reproducibility relies on re-ingestion plus hash verification.

## Alternatives considered

- DVC: powerful, but adds a tool, remote storage setup and a learning surface for data that is public and re-fetchable.
- Git LFS: poor fit for a growing partitioned dataset.
- Committing data to git: bloats the repository.
