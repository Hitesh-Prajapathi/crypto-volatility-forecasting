// k6 load test for GET /v1/forecast.
//
//   k6 run -e BASE_URL=http://localhost:8000 --summary-export=reports/k6-summary.json loadtest/k6/forecast.js
//
// Env:
//   BASE_URL   target (default http://localhost:8000)
//   ASSETS     comma list (default BTCUSDT,ETHUSDT,SOLUSDT)
//   HORIZONS   comma list (default 1,5)
//   SMOKE=1    20 s at SMOKE_RPS (default 5) hitting SMOKE_PATH (default /healthz); for validating the script
import http from 'k6/http';
import { check } from 'k6';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const ASSETS = (__ENV.ASSETS || 'BTCUSDT,ETHUSDT,SOLUSDT').split(',');
const HORIZONS = (__ENV.HORIZONS || '1,5').split(',');
const SMOKE = __ENV.SMOKE === '1';
const PATH = SMOKE ? (__ENV.SMOKE_PATH || '/healthz') : null;

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

export default function () {
  const url = SMOKE
    ? `${BASE_URL}${PATH}`
    : `${BASE_URL}/v1/forecast?asset=${ASSETS[__ITER % ASSETS.length]}&horizon=${HORIZONS[__ITER % HORIZONS.length]}`;
  const res = http.get(url, { tags: { name: SMOKE ? PATH : '/v1/forecast' } });
  check(res, { 'status is 200': (r) => r.status === 200 });
}
