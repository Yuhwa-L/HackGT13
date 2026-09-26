// Mirrors common/schemas.py. Change the Python contract first, then this file.

export type Dataset = "cifar10_test" | "cifar10c" | "cifar10_1" | "imagenet_products" | "imagenet_products_c";
export type Family = "clean" | "noise" | "blur" | "weather" | "digital";
export type Split = "train" | "val" | "cal" | "test";
export type Decision = "trust" | "caution" | "reject";
export type Method =
  | "raw_confidence"
  | "temp_scaled_clean"
  | "temp_scaled_corrupted"
  | "tta_only"
  | "logreg"
  | "trust_layer";

// 5.5
export interface Thresholds {
  is_mock: boolean;
  fold: string;
  tau_reject: number | null;
  tau_trust: number | null;
  reject_target_error: number;
  trust_target_error: number;
  cal_error_at_reject: number | null;
  cal_error_at_trust: number | null;
}

// 5.6
export interface MethodMetrics {
  aurc: number;
  auroc_pooled: number;
  auroc_within_group: number;
  ece: number;
  brier: number;
  nll: number;
  error_at_coverage: Record<string, number>;
  ci95: Record<string, [number, number]>;
}
export interface SeverityRow {
  severity: number;
  accuracy: number;
  mean_raw_conf: number;
  mean_temp_clean: number;
  mean_temp_corrupted: number;
  mean_p_correct: number;
}
export interface ReliabilityBin { bin_lo: number; bin_hi: number; conf: number; acc: number; count: number }
export interface RiskCoveragePoint { coverage: number; risk: number }
export interface BrokenPromise { target_error: number; rule_source: string; actual_error: number }
export interface FoldResult {
  methods: Partial<Record<Method, MethodMetrics>>;
  by_severity: SeverityRow[];
  reliability: Partial<Record<Method, ReliabilityBin[]>>;
  risk_coverage: Partial<Record<Method, RiskCoveragePoint[]>>;
  broken_promise: BrokenPromise;
}
export interface Evaluation {
  is_mock: boolean;
  headline_fold: string;
  folds: Record<string, FoldResult>;
  natural_shift: Record<string, { methods: Partial<Record<Method, MethodMetrics>> }>;
}

// 5.7
export interface TopK { class: string; prob: number }
export interface Reason { signal: string; text: string; shap: number }
export interface PredictResponse {
  sample_id: string;
  base_image_id: string;
  corruption: string;
  severity: number;
  image_url: string;
  true_class: string;
  target_model: { prediction: string; raw_confidence: number; entropy: number; margin: number; top_k: TopK[] };
  baselines: { temp_scaled_clean: number; temp_scaled_corrupted: number };
  trust_layer: { p_correct: number; decision: Decision; reasons: Reason[] };
}
export interface DemoCache { is_mock: boolean; samples: PredictResponse[] }

// API
export interface HealthResponse {
  status: "ok";
  artifacts: Record<string, boolean>;
  any_mock: boolean;
  shop_enabled: boolean;
}
export interface DemoSampleSummary {
  base_image_id: string;
  true_class: string;
  corruptions: string[];
  severities: number[];
}

// Shop (Section 10)
export interface Candidate { class: string; prob: number }
export interface ShopIdentifyResponse extends PredictResponse { photo_id: string; candidates: Candidate[] }
export interface ProductSummary { product_id: string; name: string; price: number; class: string }
export interface AssistRequest { photo_id: string; user_message: string; confirmed_class?: string }
export interface AssistResponse {
  decision: Decision;
  allowed_actions: string[];
  assistant_message: string;
  candidates: Candidate[];
  products: ProductSummary[];
  cart_allowed: boolean;
  llm_used: boolean;
}
export interface CheckoutRequest { photo_id: string; product_ids: string[]; confirmed_class?: string }
export interface CheckoutResponse { order_id: string; items: ProductSummary[]; total: number; is_mock: true }
