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

## Addendum: off-grid timestamps (grid snapping)

Binance archives occasionally contain bars whose `open_time` is off the interval grid. In the BTCUSDT and ETHUSDT 2018-02 monthly files, 241 consecutive bars (2018-02-09 09:58:14.789 → 2018-02-10 05:58:14.789 UTC) are offset +194.8 s after the Feb 2018 outage. Ingestion normalizes each file by flooring any bar offset by less than one interval to its grid slot. Only timestamps move; OHLCV values are untouched. If snapping would collide with another bar in the same slot, the on-grid bar is kept (or the first, if both are off-grid) and the other is dropped. Each file logs its snapped and dropped counts, and the manifest records `snapped_bars` and `dropped_bars`. Snapping runs before schema validation, so strict grid alignment is still enforced on everything stored. Impact: a ≤ 5-minute timing shift for 241 of ~970k bars; negligible for daily realized variance.
