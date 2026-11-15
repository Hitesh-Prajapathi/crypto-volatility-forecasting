// k6 load test for the volforecast API (real /v1 endpoints).
//
//   k6 run -e BASE_URL=http://localhost:8000 --summary-export=reports/k6-summary.json loadtest/k6/forecast.js
//
// The API rate limits per client IP (VOLF_API_RATE_LIMIT_RPS, default 20 rps, per worker). A load
// generator is one IP, so run the target with the limit off or high, e.g. VOLF_API_RATE_LIMIT_RPS=0
// (compose: API_RATE_LIMIT_RPS=0). A 429 counts as a FAILURE here, never as success.
//
// Request mix (weights): 70% GET /v1/forecast, 10% /v1/forecast/history?limit=30, 10% /v1/models,
// 5% /v1/backtest/summary, 5% /readyz. A third of the /v1 requests repeat with If-None-Match
// (200 and 304 are both fine).
//
// Env:
//   BASE_URL   target (default http://localhost:8000)
//   ASSETS     comma list (default BTCUSDT,ETHUSDT,SOLUSDT)
//   HORIZONS   comma list (default 1,5)
//   SUMMARY    1 (default) includes /v1/backtest/summary; 0 skips it (404 without a results file)
//   SMOKE_MIX  1: with SMOKE=1, send the real request mix instead of SMOKE_PATH
//   SMOKE=1    20 s at SMOKE_RPS (default 5) hitting SMOKE_PATH (default /healthz); for validating the script
import http from 'k6/http';
import { check } from 'k6';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const ASSETS = (__ENV.ASSETS || 'BTCUSDT,ETHUSDT,SOLUSDT').split(',');
const HORIZONS = (__ENV.HORIZONS || '1,5').split(',');
const SMOKE = __ENV.SMOKE === '1';
const PATH = SMOKE ? (__ENV.SMOKE_PATH || '/healthz') : null;
const MIX = __ENV.SMOKE_MIX === '1'; // smoke run with the real request mix
const SUMMARY = (__ENV.SUMMARY || '1') === '1';
const etags = {}; // per VU: url -> last ETag

const scenarios = SMOKE
  ? {
      smoke: {
        executor: 'constant-arrival-rate',
        rate: Number(__ENV.SMOKE_RPS || 5),
        timeUnit: '1s',
        duration: __ENV.SMOKE_DURATION || '20s',
        preAllocatedVUs: 10,
        maxVUs: 50,
      },
    }
  : {
      steady_50rps: {
        executor: 'constant-arrival-rate',
        rate: 50,
        timeUnit: '1s',
        duration: '5m',
        preAllocatedVUs: 50,
        maxVUs: 400,
      },
      ramp_to_300rps: {
        executor: 'ramping-arrival-rate',
        startTime: '5m', // starts when the steady phase ends
        startRate: 50,
        timeUnit: '1s',
        preAllocatedVUs: 100,
        maxVUs: 1000,
        stages: [
          { target: 150, duration: '2m' },
          { target: 300, duration: '3m' },
          { target: 300, duration: '2m' },
          { target: 0, duration: '30s' },
        ],
      },
    };

export const options = {
  scenarios,
  thresholds: {
    http_req_duration: ['p(95)<100'],
    http_req_failed: ['rate<0.005'],
    checks: ['rate>0.995'],
  },
  summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
};

function pick() {
  const asset = ASSETS[__ITER % ASSETS.length];
  const horizon = HORIZONS[Math.floor(__ITER / ASSETS.length) % HORIZONS.length];
  const r = (__ITER * 7919) % 100; // deterministic spread over 0..99
  if (r < 70) return { name: '/v1/forecast', path: `/v1/forecast?asset=${asset}&horizon=${horizon}` };
  if (r < 80) {
    return {
      name: '/v1/forecast/history',
      path: `/v1/forecast/history?asset=${asset}&horizon=${horizon}&limit=30`,
    };
  }
  if (r < 90) return { name: '/v1/models', path: '/v1/models' };
  if (r < 95 && SUMMARY) return { name: '/v1/backtest/summary', path: '/v1/backtest/summary' };
  return { name: '/readyz', path: '/readyz' };
}

export default function () {
  const t = SMOKE && !MIX ? { name: PATH, path: PATH } : pick();
  const url = `${BASE_URL}${t.path}`;
  const cacheable = t.path.startsWith('/v1/');
  const params = { tags: { name: t.name }, headers: {} };
  if (cacheable && etags[url] && __ITER % 3 === 0) params.headers['If-None-Match'] = etags[url];
  const res = http.get(url, params);
  if (cacheable && res.headers['Etag']) etags[url] = res.headers['Etag'];
  check(res, { 'status is 200 or 304': (r) => r.status === 200 || r.status === 304 });
}
