# Test fixtures

## `btcusdt_5m_sample.parquet`

Real BTCUSDT 5-minute spot klines, 2024-02-01 00:00 UTC through 2024-03-31 23:55 UTC
(60 days x 288 bars = 17,280 rows, no missing bars). Schema is `contracts.KLINE_COLUMNS`
(`open_time` datetime64[ns, UTC], float64 prices/volumes, int64 `n_trades`), zstd Parquet,
about 0.85 MB.

Provenance: Binance public data archive (https://data.binance.vision), monthly spot klines,
parsed (millisecond timestamps in 2024 files) and validated with `volforecast.data`. Rebuild with
`BinanceVisionSource(cache).fetch_klines("BTCUSDT", date(2024, 2, 1), date(2024, 3, 31))`.

SHA-256 of the source zips (verified against the published `.zip.CHECKSUM` files):

| File | sha256 |
| --- | --- |
| `BTCUSDT-5m-2024-02.zip` | `4bbed105653be96229f6678b3ce3df7366a6ced5aea32285da2bf62ac5f890ed` |
| `BTCUSDT-5m-2024-03.zip` | `fd3670210b38640ee9562b7761879fae8cfed3beec7d7313b868cf618ca4c72a` |
