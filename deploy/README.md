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