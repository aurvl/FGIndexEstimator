const API_BASE = ""; // use same-origin by default

export async function fetchChart({ range, include, with_components, use_calibrated_model }) {
  const params = new URLSearchParams();
  params.set("range", range);
  params.set("include", include.join(","));
  params.set("with_components", String(!!with_components));
  params.set("use_calibrated_model", String(!!use_calibrated_model));

  const url = `${API_BASE}/v1/chart?${params.toString()}`;
  console.log("fetchChart ->", url, { range, include, with_components, use_calibrated_model });
  const res = await fetch(url, { credentials: "same-origin" });
  const txt = await res.text();
  if (!res.ok) {
    let body = txt;
    try { body = JSON.parse(txt); } catch (e) {}
    console.error("API fetch error", res.status, body);
    throw new Error(`API error ${res.status}: ${typeof body === 'string' ? body : JSON.stringify(body)}`);
  }
  try {
    return JSON.parse(txt);
  } catch (e) {
    throw new Error(`Invalid JSON from API: ${e.message}`);
  }
}
