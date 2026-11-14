# 0003. Missing-bar and incomplete-day policy

- Status: Accepted
- Date: 2026-11-13

## Context

Daily realized variance is computed from 288 five-minute bars per UTC day. Real exchange data has gaps (maintenance, outages). Silently interpolating, or computing RV on a partial day, biases the estimate downward, and the target must not be quietly corrupted.

## Decision

- Forward-fill up to 3 consecutive missing 5-minute bars, treating them as zero returns.
- A day with fewer than 280 of 288 observed bars, or with any gap longer than 3 consecutive bars, is an incomplete day.
- Incomplete days are excluded from evaluation targets (they never appear as a realized value to score against) but remain allowed as input history for features, with the missingness recorded.
- The number of observed bars and the incomplete flag are stored alongside every daily row.

## Consequences

- Targets are comparable across days; evaluation never scores against a biased realized variance.
- Slightly fewer evaluation days. The count of excluded days is reported.
- Thresholds are configuration, but changing them requires a new ADR because it changes the target definition.

## Alternatives considered

- Interpolate all gaps: invents data and understates variance around the events that matter most.
- Drop every day with any missing bar: discards too much data on ordinary one-bar gaps.
- Compute RV on available bars and rescale: noisy and hard to defend.
