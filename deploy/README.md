# API call
---
To request this API in python you can use the `requests` library as follows:

```python
import requests

BASE = "https://fgindexestimator.onrender.com"

params = {
  "range": "1Y",
  "include": "fgi,sp500,nasdaq",
  "with_components": "true",
  "use_calibrated_model": "true",
}

r = requests.get(f"{BASE}/v1/chart", params=params, timeout=60)
r.raise_for_status()
data = r.json()

print(data["meta"])
print(len(data["datasets"]["fgi"]))
```

---

## Manual cache workflow (recommended)

To avoid recomputing the whole history on every API call, the API can read a **manual cache** stored in the repo under `data/cache_api/`.

1. Update the cache locally (and commit/push it when you want to refresh the API baseline):

```bash
python -m get_fg
```

Choose:
- **Update the manual API cache**: `y` (default)
- **Use calibrated model**: `y` (default)

This creates/updates (same naming convention as the API cache):
- `data/cache_api/fgi_MAX_YYYY-MM-DD_components0_calib1.parquet`
- `data/cache_api/fgi_MAX_YYYY-MM-DD_components1_calib1.parquet` (optional)

Optionally, it can also update the **market** MAX caches:
- `data/cache_api/market_sp500_MAX_YYYY-MM-DD.parquet`
- `data/cache_api/market_nasdaq_MAX_YYYY-MM-DD.parquet`
- `data/cache_api/market_cac40_MAX_YYYY-MM-DD.parquet`
- `data/cache_api/market_msciworld_MAX_YYYY-MM-DD.parquet`

2. Runtime behavior:
- On each request, the API loads the **latest** `fgi_MAX_*` cache file that matches the request flags.
- It recomputes only from **(last cached date − 7 days)** up to the requested `end_date`.
- It concatenates, removes duplicates, and returns the full series for the requested `range`.

For market series, the API prefers the latest `market_<id>_MAX_*` cache and slices it to the requested `range` (only fetching the last few days if the requested `end_date` is newer than the cache).