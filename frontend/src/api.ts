// Backend client. If the backend is unreachable, GET helpers fall back to static JSON copies in
// public/fallback/ (populated by `make frontend-fallback`).
import type {
  AssistRequest, AssistResponse, CheckoutRequest, CheckoutResponse, DemoSampleSummary,
  Evaluation, HealthResponse, PredictResponse, ShopIdentifyResponse,
} from "./types";

async function getJson<T>(path: string, fallbackFile?: string): Promise<T> {
  try {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return (await r.json()) as T;
  } catch (e) {
    if (!fallbackFile) throw e;
    const r = await fetch(`/fallback/${fallbackFile}`);
    if (!r.ok) throw e;
    return (await r.json()) as T;
  }
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return (await r.json()) as T;
}

export const api = {
  health: () => getJson<HealthResponse>("/api/health"),
  evaluation: () => getJson<Evaluation>("/api/evaluation", "evaluation.json"),
  demoSamples: () => getJson<DemoSampleSummary[]>("/api/demo/samples"),
  // TODO(D): offline fallback for demo lookups (derive from fallback/demo_cache.json).
  demo: (baseImageId: string, corruption: string, severity: number) =>
    getJson<PredictResponse>(
      `/api/demo/${encodeURIComponent(baseImageId)}?corruption=${encodeURIComponent(corruption)}&severity=${severity}`,
    ),
  shopSamples: () => getJson<unknown[]>("/api/shop/samples"),
  shopIdentify: (photo_id: string) => postJson<ShopIdentifyResponse>("/api/shop/identify", { photo_id }),
  shopAssist: (body: AssistRequest) => postJson<AssistResponse>("/api/shop/assist", body),
  shopCheckout: (body: CheckoutRequest) => postJson<CheckoutResponse>("/api/shop/checkout", body),
};
