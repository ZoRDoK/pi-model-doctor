/**
 * Official-source adapters for model-doctor.
 *
 * Design rules (from the approved plan):
 *  - Only hardcoded HTTPS hosts; redirects forbidden (redirect:"error").
 *  - No auth headers: these metadata endpoints are public.
 *  - AbortSignal + timeout + response-size limit + content-type/schema checks.
 *  - Missing field != zero/default; ambiguity blocks the affected patch.
 *  - OpenRouter aliases are informational, never rewritten to canonical id.
 *  - models.dev has no OpenCode routing/pricing; it is canonical metadata only,
 *    matched to pi opencode models by unique slug suffix.
 *  - opencode-go and opencode (zen) are both LIVE public catalogs
 *    (https://opencode.ai/zen/go/v1/models and /zen/v1/models): each is the
 *    AVAILABILITY authority for its own provider, enriched with models.dev
 *    metadata. The two catalogs must never be confused (distinct providers).
 *
 * All paths/functions are injectable via the FetchFn option so tests use a fully
 * mocked transport and never touch the network.
 */
import { createHash } from "node:crypto";

import type {
  Confidence,
  Evidence,
  FetchFn,
  FetchResponse,
  NormalizedCost,
  NormalizedModel,
  OfficialSourceId,
  ManagedProvider,
  SourceResult,
} from "./types.ts";

/** Official live catalog endpoints (all public, no auth). */
export const SOURCE_ENDPOINTS: Readonly<Record<OfficialSourceId, string>> = {
  openrouter: "https://openrouter.ai/api/v1/models",
  // OpenCode Zen AVAILABILITY authority: the live official catalog. Only IDs
  // offered here are routable opencode models (models.dev is metadata-only).
  "opencode-zen": "https://opencode.ai/zen/v1/models",
  // OpenCode Go AVAILABILITY authority: the live public Go catalog (sibling of
  // Zen /zen/v1/models). Only IDs offered here are routable opencode-go models.
  "opencode-go": "https://opencode.ai/zen/go/v1/models",
  // Charm Hyper public catalog: metadata-only source for per-model reasoning
  // effort levels (reasoning.effort_levels). Never a managed provider.
  hyper: "https://hyper.charm.land/v1/models",
};

/** models.dev canonical metadata; used ONLY to enrich the Zen/Go intersection.
 *  The nested org view (`/api.json`, {org:{models:{slug:{…}}}}) is a superset of
 *  the flat `/models.json` list and carries first-party `opencode` /
 *  `opencode-go` sections whose `limit.output` is the authoritative max output
 *  for Zen/Go-only models (e.g. space-bunny-free) that the flat list omits. */
export const MODELS_DEV_URL = "https://models.dev/api.json";

/** models.dev org sections preferred when the same model slug appears under
 *  several orgs: first-party OpenCode metadata wins over third-party mirrors. */
const MODELS_DEV_PREFERRED_ORGS: readonly string[] = ["opencode", "opencode-go"];

/** Hosts the adapters are allowed to contact. */
export const ALLOWED_HOSTS: ReadonlySet<string> = new Set([
  "openrouter.ai",
  "models.dev",
  "opencode.ai",
  "hyper.charm.land",
]);

export const REQUEST_TIMEOUT_MS = 45_000;
export const MAX_RESPONSE_BYTES = 8 * 1024 * 1024;

/** Total attempts per safeFetch call for transient transport failures. */
export const FETCH_ATTEMPTS = 3;
/** Linear backoff between attempts: 500ms, then 1000ms. */
const RETRY_BASE_DELAY_MS = 500;

/** ISO timestamp for snapshots. */
export function nowIso(now: () => number = Date.now): string {
  return new Date(now()).toISOString();
}

/** sha256 hex of a string or byte buffer. */
export function sha256Hex(data: string | Uint8Array | ArrayBuffer): string {
  const buf =
    typeof data === "string"
      ? Buffer.from(data, "utf8")
      : data instanceof ArrayBuffer
        ? new Uint8Array(data)
        : data;
  return createHash("sha256").update(buf).digest("hex");
}

function assertAllowedUrl(rawUrl: string): { url: URL; host: string } {
  let url: URL;
  try {
    url = new URL(rawUrl);
  } catch {
    throw new Error(`invalid url: ${rawUrl}`);
  }
  if (url.protocol !== "https:") {
    throw new Error(`refusing non-https url: ${rawUrl}`);
  }
  if (!ALLOWED_HOSTS.has(url.hostname)) {
    throw new Error(`host not allowlisted: ${url.hostname}`);
  }
  // No credentials in URL.
  if (url.username || url.password) {
    throw new Error(`credentials in url are forbidden: ${url.hostname}`);
  }
  return { url, host: url.hostname };
}

/**
 * Enforce the full transport safety contract and return the raw response body
 * bytes plus a sha256 digest. Rejects on redirect, non-2xx, wrong content-type,
 * or oversized body.
 */
export async function safeFetch(
  rawUrl: string,
  fetchFn: FetchFn,
  signal?: AbortSignal,
): Promise<{ bytes: Uint8Array; text: string; digest: string; url: string }> {
  const { host } = assertAllowedUrl(rawUrl); // throws on bad scheme/host before any I/O

  // Transient transport failures (stalled body, 429/5xx) are retried on a
  // fresh connection; deterministic refusals (bad url/host/content-type,
  // oversize, caller abort) fail fast on the first attempt.
  let lastError = new Error(`network error for ${host}: request failed`);
  for (let attempt = 1; attempt <= FETCH_ATTEMPTS; attempt += 1) {
    if (signal?.aborted) throw new Error(`network error for ${host}: aborted`);
    try {
      return await fetchOnce(rawUrl, host, fetchFn, signal);
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      lastError = error;
      if (!isRetryableTransportError(error, signal) || attempt === FETCH_ATTEMPTS) {
        throw error;
      }
      await sleep(RETRY_BASE_DELAY_MS * attempt, signal);
    }
  }
  throw lastError;
}

/** Only transport-level failures retry; deterministic refusals do not. */
function isRetryableTransportError(err: Error, signal: AbortSignal | undefined): boolean {
  if (signal?.aborted) return false;
  const m = err.message;
  if (m.startsWith("network error for ")) return true; // rejected fetch / aborted body read
  return / returned status (429|5\d\d)$/.test(m); // throttling / upstream outage
}

function sleep(ms: number, signal: AbortSignal | undefined): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new Error("aborted"));
      },
      { once: true },
    );
  });
}

/** Single transport attempt: timeout, redirect refusal, status, content-type. */
async function fetchOnce(
  rawUrl: string,
  host: string,
  fetchFn: FetchFn,
  signal?: AbortSignal,
): Promise<{ bytes: Uint8Array; text: string; digest: string; url: string }> {
  // Combine the caller's signal with a per-attempt timeout. Never attach auth headers.
  const timeout = AbortSignal.timeout(REQUEST_TIMEOUT_MS);
  const combined =
    signal !== undefined ? AbortSignal.any([signal, timeout]) : timeout;

  let response: FetchResponse;
  try {
    response = await fetchFn(rawUrl, {
      method: "GET",
      redirect: "error", // forbid redirects off the allowlisted host
      signal: combined,
    });
  } catch (err) {
    throw new Error(`network error for ${host}: ${messageOf(err)}`);
  }

  if (!response.ok || response.status < 200 || response.status >= 300) {
    throw new Error(`${host} returned status ${response.status}`);
  }

  const contentType = response.headers.get("content-type") ?? "";
  if (!/application\/json/i.test(contentType)) {
    throw new Error(`${host} returned non-json content-type: ${contentType}`);
  }

  const bytes = await readLimited(response, MAX_RESPONSE_BYTES);
  const text = new TextDecoder().decode(bytes);
  return { bytes, text, digest: sha256Hex(bytes), url: response.url || rawUrl };
}

/** Read a response body in chunks, aborting if it exceeds maxBytes. */
async function readLimited(
  response: FetchResponse,
  maxBytes: number,
): Promise<Uint8Array> {
  // Prefer the web stream reader when available (true size cap mid-stream).
  const body = (response as unknown as { body?: unknown }).body;
  if (body && typeof (body as { getReader?: unknown }).getReader === "function") {
    const reader = (body as { getReader: () => Reader }).getReader();
    const chunks: Uint8Array[] = [];
    let total = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        total += value.byteLength;
        if (total > maxBytes) {
          try {
            await reader.cancel();
          } catch {
            /* ignore */
          }
          throw new Error(`response exceeds ${maxBytes} bytes`);
        }
        chunks.push(value);
      }
    }
    return concatBytes(chunks, total);
  }
  // Fallback: arrayBuffer with a post-hoc size check.
  const ab = await response.arrayBuffer();
  const view = new Uint8Array(ab);
  if (view.byteLength > maxBytes) {
    throw new Error(`response exceeds ${maxBytes} bytes`);
  }
  return view;
}

interface Reader {
  read(): Promise<{ done: boolean; value?: Uint8Array }>;
  cancel(): Promise<void>;
}

function concatBytes(chunks: Uint8Array[], total: number): Uint8Array {
  const out = new Uint8Array(total);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.byteLength;
  }
  return out;
}

function messageOf(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

function ev<T>(
  value: T,
  source: string,
  confidence: Confidence,
  plannable: boolean,
  note?: string,
): Evidence<T> {
  return note !== undefined
    ? { value, source, confidence, plannable, note }
    : { value, source, confidence, plannable };
}

// ─── OpenRouter normalization ────────────────────────────────────────────────

interface OpenRouterModel {
  id?: unknown;
  name?: unknown;
  created?: unknown;
  canonical_slug?: unknown;
  context_length?: unknown;
  top_provider?: { max_completion_tokens?: unknown; context_length?: unknown } | null;
  pricing?: Record<string, unknown> | null;
  reasoning?: { mandatory?: unknown; supported_efforts?: unknown } | boolean | null;
  supported_parameters?: unknown[] | null;
  architecture?: { input_modalities?: unknown[] | null; modality?: unknown } | null;
}

const REASONING_PARAMS = new Set([
  "reasoning",
  "include_reasoning",
  "reasoning_effort",
]);

/** Effort values pi can represent: the six thinking levels plus "none",
 *  which maps to pi's `off` level. Anything else never enters a map. */
const KNOWN_EFFORTS: ReadonlySet<string> = new Set([
  "none",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
]);

/** pi thinking levels in canonical order (off is handled separately). */
const PI_EFFORT_LEVELS: readonly string[] = ["minimal", "low", "medium", "high", "xhigh", "max"];

export interface ThinkingMapResult {
  readonly map: Readonly<Record<string, string | null>>;
  readonly unknown: readonly string[];
}

/** Build a pi thinkingLevelMap from provider effort values: every pi level is
 *  present (unsupported levels map to null = hidden) and "none" becomes
 *  off:"none". Unknown effort values are collected, never silently dropped. */
export function buildThinkingLevelMap(efforts: readonly string[]): ThinkingMapResult {
  const known = new Set(efforts.filter((e) => KNOWN_EFFORTS.has(e)));
  const unknown = [...new Set(efforts.filter((e) => !KNOWN_EFFORTS.has(e)))];
  const map: Record<string, string | null> = {};
  map.off = known.has("none") ? "none" : null;
  for (const level of PI_EFFORT_LEVELS) {
    map[level] = known.has(level) ? level : null;
  }
  return { map, unknown };
}

const KNOWN_MODALITIES = new Set(["text", "image"]);

export function normalizeOpenRouterList(
  raw: unknown,
  provider: ManagedProvider = "openrouter",
): NormalizedModel[] {
  if (!isObject(raw) || !Array.isArray((raw as { data?: unknown }).data)) {
    throw new Error("openrouter payload missing data[]");
  }
  const data = (raw as { data: unknown[] }).data;
  const out: NormalizedModel[] = [];
  const seenIds = new Set<string>();
  for (const item of data) {
    if (!isObject(item)) continue;
    const model = item as OpenRouterModel;
    const id = asString(model.id);
    if (!id) continue;
    // Duplicate model ids are ambiguous: a second record could carry hostile
    // metadata that first-match resolution would silently pick. Fail closed.
    if (seenIds.has(id)) {
      throw new Error(`openrouter payload has duplicate model id '${id}' (ambiguous; refusing)`);
    }
    seenIds.add(id);
    const normalized = normalizeOpenRouterModel(model, id, provider);
    if (normalized) out.push(normalized);
  }
  return out;
}

function normalizeOpenRouterModel(
  model: OpenRouterModel,
  id: string,
  provider: ManagedProvider,
): NormalizedModel | null {
  const source = "openrouter:list";
  const name = asString(model.name);
  const reasoning = readOpenRouterReasoning(model, source);
  const thinkingLevelMap = readOpenRouterThinkingLevels(model, source);
  const input = readOpenRouterInput(model, source);
  const contextWindow = asNonNegLimit(model.context_length);
  const maxTokens = model.top_provider
    ? asNonNegLimit(model.top_provider.max_completion_tokens)
    : undefined;
  const cost = readOpenRouterCost(model, source);
  const canonical = asString(model.canonical_slug);
  const aliases: string[] = [];
  const notes: string[] = [];
  if (canonical && canonical !== id) {
    aliases.push(canonical);
  }

  const out: NormalizedModel = { provider, id };
  // Informational creation timestamp (never patched): used by the retire-cutoff.
  const created = asStrictNumber(model.created);
  if (created !== undefined && created > 0) out.createdMs = created * 1000;
  if (name !== undefined) out.name = ev(name, source, "authoritative", true);
  if (reasoning !== undefined) out.reasoning = reasoning;
  if (thinkingLevelMap !== undefined) out.thinkingLevelMap = thinkingLevelMap;
  if (input !== undefined) out.input = input;
  if (contextWindow !== undefined) {
    out.contextWindow = ev(contextWindow, source, "authoritative", true);
  }
  if (maxTokens !== undefined) {
    out.maxTokens = ev(maxTokens, source, "reported", true);
  }
  if (cost !== undefined) out.cost = cost;
  if (aliases.length > 0) out.aliases = aliases;
  if (notes.length > 0) out.notes = notes;
  return out;
}

function readOpenRouterReasoning(
  model: OpenRouterModel,
  source: string,
): Evidence<boolean> | undefined {
  const params = Array.isArray(model.supported_parameters)
    ? model.supported_parameters
    : [];
  const hasParam = params.some((p) => typeof p === "string" && REASONING_PARAMS.has(p));
  const hasField =
    model.reasoning !== null && model.reasoning !== undefined;
  if (hasParam || hasField) {
    return ev(true, source, "reported", true);
  }
  return undefined; // unknown, NOT false
}

/** reasoning.supported_efforts -> thinkingLevelMap evidence. A missing or
 *  EMPTY array is unknown (an empty set is often missing data, e.g. a thinking
 *  model listed with no effort control), never "no levels". Unknown effort
 *  values make the map non-plannable (never auto-applied). */
function readOpenRouterThinkingLevels(
  model: OpenRouterModel,
  source: string,
): Evidence<Readonly<Record<string, string | null>>> | undefined {
  const reasoning = model.reasoning;
  if (!isObject(reasoning)) return undefined;
  const raw = (reasoning as { supported_efforts?: unknown }).supported_efforts;
  if (!Array.isArray(raw) || raw.length === 0) return undefined;
  const efforts: string[] = [];
  for (const e of raw) {
    if (typeof e === "string" && e.trim().length > 0) efforts.push(e.trim().toLowerCase());
  }
  if (efforts.length === 0) return undefined;
  const { map, unknown } = buildThinkingLevelMap(efforts);
  if (unknown.length > 0) {
    return ev(
      map,
      source,
      "reported",
      false,
      `unsupported effort values [${unknown.join(",")}]; thinkingLevelMap not auto-applied`,
    );
  }
  return ev(map, source, "reported", true);
}

function readOpenRouterInput(
  model: OpenRouterModel,
  source: string,
): Evidence<ReadonlyArray<"text" | "image">> | undefined {
  const arr = model.architecture?.input_modalities;
  if (!Array.isArray(arr) || arr.length === 0) return undefined;
  const mapped: ("text" | "image")[] = [];
  for (const m of arr) {
    if (typeof m === "string" && KNOWN_MODALITIES.has(m)) {
      mapped.push(m as "text" | "image");
    }
  }
  if (mapped.length === 0) return undefined;
  return ev(mapped, source, "authoritative", true);
}

function readOpenRouterCost(
  model: OpenRouterModel,
  source: string,
): Evidence<NormalizedCost> | undefined {
  const pricing = model.pricing;
  if (!isObject(pricing)) return undefined;
  const cost: NormalizedCost = {};
  let hasField = false;
  const take = (key: string, target: "input" | "output" | "cacheRead" | "cacheWrite") => {
    const raw = (pricing as Record<string, unknown>)[key];
    const num = usdPerTokenToPerMillion(raw);
    if (num !== undefined) {
      (cost as Record<"input" | "output" | "cacheRead" | "cacheWrite", number | undefined>)[target] = num;
      hasField = true;
    }
  };
  take("prompt", "input");
  take("completion", "output");
  take("input_cache_read", "cacheRead");
  take("input_cache_write", "cacheWrite");
  if (!hasField) return undefined;

  // Conditional / per-request pricing cannot be represented exactly in pi tiers.
  const overrides = (pricing as Record<string, unknown>).overrides;
  const plannable = !(overrides !== null && overrides !== undefined);
  const note = plannable
    ? undefined
    : "openrouter reports conditional/per-request pricing; cost not auto-applied";
  return ev(cost, source, "reported", plannable, note);
}

/** OpenRouter prices are USD per token as strings; pi uses USD per million. */
function usdPerTokenToPerMillion(raw: unknown): number | undefined {
  const n = asStrictNumber(raw);
  if (n === undefined || n < 0 || !Number.isFinite(n)) return undefined;
  return round6(n * 1_000_000);
}

function round6(n: number): number {
  return Math.round(n * 1e6) / 1e6;
}

/**
 * Build the OpenRouter per-model endpoints URL, preserving the full
 * `org/model` id by percent-encoding each path segment separately (the live
 * API returns 404 for the slug-only form `/models/<slug>/endpoints`).
 */
export function openRouterDetailUrl(modelId: string): string {
  const segments = modelId.split("/").map((s) => encodeURIComponent(s)).join("/");
  return `https://openrouter.ai/api/v1/models/${segments}/endpoints`;
}

/** Outcome of folding an OpenRouter endpoints (detail) payload. */
export type DetailStatus = "ok" | "malformed" | "ambiguous";

export interface DetailRefinement {
  readonly model: NormalizedModel;
  readonly status: DetailStatus;
  readonly note?: string;
}

/**
 * Demote a model's list-derived context/max limits to non-plannable (reported)
 * evidence with an explanatory note, so they can never become an apply
 * operation without endpoint confirmation. Used when the detail payload is
 * malformed, missing, or unavailable.
 */
export function demoteListLimits(
  base: NormalizedModel,
  source: string,
  note: string,
): NormalizedModel {
  const out: NormalizedModel = { ...base, notes: base.notes ? [...base.notes] : [] };
  if (base.contextWindow) {
    out.contextWindow = { ...base.contextWindow, source, confidence: "reported", plannable: false, note };
  }
  if (base.maxTokens) {
    out.maxTokens = { ...base.maxTokens, source, confidence: "reported", plannable: false, note };
  }
  return out;
}

/**
 * Fold a fetched OpenRouter endpoints payload into the normalized model.
 *  - malformed (non-JSON / no endpoints array / empty / any non-object entry):
 *    list-derived limits are demoted to non-plannable so they are NEVER
 *    auto-applied without endpoint confirmation; status "malformed".
 *  - disagreeing endpoints: the disagreeing field(s) become non-plannable
 *    (ambiguity); status "ambiguous".
 *  - agreeing endpoints: the field is promoted to authoritative, plannable;
 *    status "ok".
 * Limits BEGIN demoted and are promoted per-field only after valid, agreeing
 * endpoint evidence, so a detail that carries no usable limit can never leave a
 * list-derived limit plannable. Pure: takes the already-fetched text so the
 * caller can capture and fingerprint the raw payload.
 */
export function applyOpenRouterEndpoints(base: NormalizedModel, detailText: string): DetailRefinement {
  let parsed: unknown;
  try {
    parsed = JSON.parse(detailText);
  } catch {
    return {
      model: demoteListLimits(base, "openrouter:endpoints", "detail payload is not valid JSON; list-derived limits not auto-applied"),
      status: "malformed",
      note: "detail payload is not valid JSON",
    };
  }
  const data = isObject(parsed) ? (parsed as { data?: unknown }).data : undefined;
  const endpoints = isObject(data) ? (data as { endpoints?: unknown }).endpoints : undefined;
  if (!Array.isArray(endpoints) || endpoints.length === 0) {
    return {
      model: demoteListLimits(base, "openrouter:endpoints", "detail payload has no endpoints array; list-derived limits not auto-applied"),
      status: "malformed",
      note: "detail payload has no endpoints array",
    };
  }
  // Any malformed (non-object) entry in a non-empty endpoints array fails the
  // WHOLE detail response: demote both limits and report malformed. Never
  // silently skip an invalid entry and never return ok / plannable list limits.
  for (const ep of endpoints) {
    if (!isObject(ep)) {
      return {
        model: demoteListLimits(base, "openrouter:endpoints", "detail endpoints contain a malformed (non-object) entry; list-derived limits not auto-applied"),
        status: "malformed",
        note: "detail endpoints contain a malformed (non-object) entry",
      };
    }
  }

  // Begin with both limits demoted (non-plannable); promote each field only
  // after valid, agreeing endpoint evidence.
  const out: NormalizedModel = demoteListLimits(
    base,
    "openrouter:endpoints",
    "list-derived limits held non-plannable until endpoint evidence confirms them",
  );
  const ctxList: number[] = [];
  const maxList: number[] = [];
  for (const ep of endpoints) {
    const ctx = asNonNegLimit((ep as { context_length?: unknown }).context_length);
    const max = asNonNegLimit((ep as { max_completion_tokens?: unknown }).max_completion_tokens);
    if (ctx !== undefined) ctxList.push(ctx);
    if (max !== undefined) maxList.push(max);
  }
  let ambiguous = false;
  if (ctxList.length > 0) {
    const uniq = uniqueNumbers(ctxList);
    if (uniq.length === 1) {
      out.contextWindow = ev(uniq[0]!, "openrouter:endpoints", "authoritative", true);
    } else {
      out.contextWindow = ev(uniq[0]!, "openrouter:endpoints", "reported", false, `endpoints disagree on context length: ${uniq.join(", ")}`);
      ambiguous = true;
    }
  }
  if (maxList.length > 0) {
    const uniq = uniqueNumbers(maxList);
    if (uniq.length === 1) {
      out.maxTokens = ev(uniq[0]!, "openrouter:endpoints", "authoritative", true);
    } else {
      out.maxTokens = ev(uniq[0]!, "openrouter:endpoints", "reported", false, `endpoints disagree on max output: ${uniq.join(", ")}`);
      ambiguous = true;
    }
  }
  return {
    model: out,
    status: ambiguous ? "ambiguous" : "ok",
    note: ambiguous ? "endpoints disagree on one or more limits" : undefined,
  };
}

/**
 * Fetch the OpenRouter endpoints view for a single model and fold per-endpoint
 * pricing/limits back into the normalized model. (Convenience wrapper; the plan
 * layer fetches directly so it can fingerprint the raw detail payload.)
 */
export async function refineOpenRouterModel(
  base: NormalizedModel,
  detailUrl: string,
  fetchFn: FetchFn,
  signal?: AbortSignal,
): Promise<NormalizedModel> {
  let text: string;
  try {
    const fetched = await safeFetch(detailUrl, fetchFn, signal);
    text = fetched.text;
  } catch {
    // Detail endpoint unavailable: demote list-derived limits so they are never
    // auto-applied without endpoint confirmation.
    return demoteListLimits(base, "openrouter:endpoints", "detail endpoint unavailable; list-derived limits not auto-applied");
  }
  return applyOpenRouterEndpoints(base, text).model;
}

// ─── models.dev (OpenCode Zen) normalization ────────────────────────────────

interface ModelsDevModel {
  id?: unknown;
  name?: unknown;
  reasoning?: unknown;
  modalities?: { input?: unknown[] | null; output?: unknown[] | null } | null;
  limit?: { context?: unknown; output?: unknown } | null;
  attachment?: unknown;
}

/** Normalize the models.dev org view into NormalizedModel[] keyed by unique
 *  slug. The same slug routinely appears under several orgs (first-party plus
 *  mirrors); preferred orgs are scanned first so first-party metadata wins,
 *  and the first sighting of a slug is kept deterministically. */
export function normalizeModelsDev(raw: unknown): NormalizedModel[] {
  if (!isObject(raw)) throw new Error("models.dev payload is not an object");
  // A top-level `data` array means we got the wrong endpoint (e.g. an
  // OpenRouter-shaped payload) -> fail closed.
  if (Array.isArray((raw as { data?: unknown }).data)) {
    throw new Error("models.dev payload has unexpected 'data' array (wrong endpoint?)");
  }
  const orgs = Object.entries(raw as Record<string, unknown>);
  // An empty payload (e.g. `{}`) is not a valid models.dev catalog response
  // (the live source always carries hundreds of orgs); treat it as a wrong /
  // empty endpoint and fail closed rather than reporting status:ok with 0 models.
  if (orgs.length === 0) {
    throw new Error("models.dev payload has no org entries (wrong/empty endpoint?)");
  }
  const ordered = [
    ...MODELS_DEV_PREFERRED_ORGS.filter((org) => org in (raw as Record<string, unknown>)),
    ...orgs.map(([org]) => org).filter((org) => !MODELS_DEV_PREFERRED_ORGS.includes(org)).sort(),
  ];
  const out: NormalizedModel[] = [];
  const seenSlugs = new Set<string>();
  for (const org of ordered) {
    const orgEntry = (raw as Record<string, unknown>)[org]!;
    // Non-object / null org entries are malformed; fail closed (do not skip).
    if (!isObject(orgEntry)) {
      throw new Error(`models.dev org '${org}' is not an object (malformed; refusing)`);
    }
    const models = (orgEntry as { models?: unknown }).models;
    // Orgs without a models map are legitimately thin (provider stubs); skip.
    if (models === undefined) continue;
    if (!isObject(models)) {
      throw new Error(`models.dev org '${org}' has a malformed models map (refusing)`);
    }
    for (const [modelKey, entry] of Object.entries(models)) {
      // Non-object / null entries are malformed; fail closed (do not silently skip).
      if (!isObject(entry)) {
        throw new Error(`models.dev entry '${org}/${modelKey}' is not an object (malformed; refusing)`);
      }
      const model = entry as ModelsDevModel;
      const id = asString(model.id) ?? modelKey;
      const slug = slugOf(id);
      // The same slug under several orgs is EXPECTED (mirrors): the preferred
      // orgs were scanned first, so the first sighting is the best one.
      if (seenSlugs.has(slug)) continue;
      seenSlugs.add(slug);
      const normalized = normalizeModelsDevEntry(model, slug, `models.dev:${org}`);
      normalized.notes = [`models.dev canonical id: ${org}/${id}`];
      out.push(normalized);
    }
  }
  if (out.length === 0) {
    throw new Error("models.dev payload has no model entries (wrong/empty endpoint?)");
  }
  return out;
}

/** Normalize one models.dev model entry under its zen/go slug. */
function normalizeModelsDevEntry(
  model: ModelsDevModel,
  slug: string,
  source: string,
): NormalizedModel {
  const normalized: NormalizedModel = { provider: "opencode", id: slug };
  const name = asString(model.name);
  if (name !== undefined) normalized.name = ev(name, source, "authoritative", true);
  if (model.reasoning === true) {
    normalized.reasoning = ev(true, source, "reported", true);
  } // false is unknown, not authoritative false
  const input = readModelsDevInput(model, source);
  if (input !== undefined) normalized.input = input;
  const context = asNonNegLimit(model.limit?.context);
  if (context !== undefined) normalized.contextWindow = ev(context, source, "authoritative", true);
  const output = asNonNegLimit(model.limit?.output);
  if (output !== undefined) normalized.maxTokens = ev(output, source, "authoritative", true);
  return normalized;
}

function readModelsDevInput(
  model: ModelsDevModel,
  source: string,
): Evidence<ReadonlyArray<"text" | "image">> | undefined {
  const arr = model.modalities?.input;
  if (!Array.isArray(arr) || arr.length === 0) return undefined;
  const mapped: ("text" | "image")[] = [];
  for (const m of arr) {
    if (typeof m === "string" && KNOWN_MODALITIES.has(m)) mapped.push(m as "text" | "image");
  }
  if (mapped.length === 0) return undefined;
  return ev(mapped, source, "authoritative", true);
}

export function slugOf(id: string): string {
  const idx = id.lastIndexOf("/");
  return idx >= 0 ? id.slice(idx + 1) : id;
}

// ─── Charm Hyper (metadata-only catalog for reasoning effort levels) ─────────

interface HyperModel {
  id?: unknown;
  display_name?: unknown;
  reasoning?: { effort_levels?: unknown; default_effort_level?: unknown } | null;
}

/** Normalize the Charm Hyper catalog ({data:[{id, display_name, reasoning:{
 *  effort_levels:[{value,display}], ...}}]}). Metadata-only: entries carry the
 *  bare model id, a display name and (when the catalog reports them) a
 *  thinkingLevelMap — never cost/limits, never a managed provider block. */
export function normalizeHyperList(raw: unknown): NormalizedModel[] {
  if (!isObject(raw) || !Array.isArray((raw as { data?: unknown }).data)) {
    throw new Error("hyper payload missing data[]");
  }
  const data = (raw as { data: unknown[] }).data;
  const out: NormalizedModel[] = [];
  const seenIds = new Set<string>();
  for (const item of data) {
    if (!isObject(item)) {
      throw new Error("hyper data[] contains a malformed (non-object) entry");
    }
    const model = item as HyperModel;
    const id = asString(model.id);
    if (!id) continue;
    if (seenIds.has(id)) {
      throw new Error(`hyper payload has duplicate model id '${id}' (ambiguous; refusing)`);
    }
    seenIds.add(id);
    const source = "hyper:list";
    const entry: NormalizedModel = { provider: "hyper", id };
    const name = asString(model.display_name);
    if (name !== undefined) entry.name = ev(name, source, "authoritative", true);
    const thinkingLevelMap = readHyperEffortLevels(model, source);
    if (thinkingLevelMap !== undefined) entry.thinkingLevelMap = thinkingLevelMap;
    out.push(entry);
  }
  if (out.length === 0) {
    throw new Error("hyper payload has no model entries (wrong/empty endpoint?)");
  }
  return out;
}

/** reasoning.effort_levels -> thinkingLevelMap evidence. Entries are
 *  {value,display} objects (plain strings are also accepted). A missing or
 *  empty list is unknown, never "no levels" (Hyper sometimes lists thinking
 *  models with an empty array — that data is known-broken). Malformed entries
 *  and unknown effort values make the map non-plannable. */
function readHyperEffortLevels(
  model: HyperModel,
  source: string,
): Evidence<Readonly<Record<string, string | null>>> | undefined {
  const reasoning = model.reasoning;
  if (!isObject(reasoning)) return undefined;
  const raw = (reasoning as { effort_levels?: unknown }).effort_levels;
  if (!Array.isArray(raw) || raw.length === 0) return undefined;
  const efforts: string[] = [];
  let hasMalformed = false;
  for (const e of raw) {
    const value = typeof e === "string" ? e : isObject(e) ? asString((e as { value?: unknown }).value) : undefined;
    if (value === undefined || value.trim().length === 0) {
      hasMalformed = true;
      continue;
    }
    efforts.push(value.trim().toLowerCase());
  }
  if (efforts.length === 0) return undefined;
  const { map, unknown } = buildThinkingLevelMap(efforts);
  const problems: string[] = [];
  if (hasMalformed) problems.push("malformed effort entries ignored");
  if (unknown.length > 0) problems.push(`unsupported effort values [${unknown.join(",")}]`);
  if (problems.length > 0) {
    return ev(
      map,
      source,
      "reported",
      false,
      `${problems.join("; ")}; thinkingLevelMap not auto-applied`,
    );
  }
  return ev(map, source, "reported", true);
}

// ─── OpenCode Zen: live availability authority + models.dev enrichment ───────

/**
 * Parse the live OpenCode Zen catalog ({data:[{id,...}]}) into the
 * authoritative set of routable opencode slug IDs. Rejects malformed/
 * duplicate IDs (fail closed). This is the AVAILABILITY authority: only IDs
 * present here are real Zen offerings (the global models.dev catalog is
 * metadata-only and must not invent unroutable opencode models).
 */
export function normalizeZenLive(raw: unknown): string[] {
  return parseLiveCatalogIds(raw, "opencode zen");
}

/**
 * Strict parser for an OpenAI-style live catalog ({data:[{id,...}]}). Returns the
 * authoritative set of routable IDs. Rejects malformed (non-object) entries and
 * duplicate IDs (fail closed). Shared by the Zen and Go live catalogs.
 */
export function parseLiveCatalogIds(raw: unknown, label: string): string[] {
  if (!isObject(raw) || !Array.isArray((raw as { data?: unknown }).data)) {
    throw new Error(`${label} payload missing data[]`);
  }
  const data = (raw as { data: unknown[] }).data;
  const ids: string[] = [];
  const seen = new Set<string>();
  for (const item of data) {
    if (!isObject(item)) {
      throw new Error(`${label} data[] contains a malformed (non-object) entry`);
    }
    const id = asString((item as { id?: unknown }).id);
    if (!id) throw new Error(`${label} entry has no string id`);
    if (seen.has(id)) throw new Error(`${label} has duplicate id '${id}'`);
    seen.add(id);
    ids.push(id);
  }
  return ids;
}

/**
 * Fetch OpenCode Zen: the LIVE official catalog is the availability authority
 * (only its IDs are routable); models.dev is fetched only to ENRICH those IDs
 * with canonical metadata (matched by slug). models.dev-only slugs are never
 * emitted, so unroutable opencode models cannot be invented. Never throws:
 * returns unavailable if the Zen authority itself cannot be fetched.
 */
export async function fetchOpenCodeZenSource(
  fetchFn: FetchFn,
  signal?: AbortSignal,
): Promise<SourceResult> {
  const fetchedAt = nowIso();
  const zenUrl = SOURCE_ENDPOINTS["opencode-zen"];
  let zenText: string;
  let zenDigest: string;
  try {
    const r = await safeFetch(zenUrl, fetchFn, signal);
    zenText = r.text;
    zenDigest = r.digest;
  } catch (err) {
    return { id: "opencode-zen", status: "unavailable", digest: sha256Hex(`unavailable:opencode-zen:${messageOf(err)}`), models: [], error: messageOf(err), fetchedAt };
  }
  let zenIds: string[];
  try {
    zenIds = normalizeZenLive(JSON.parse(zenText));
  } catch (err) {
    return { id: "opencode-zen", status: "unavailable", digest: zenDigest, models: [], error: messageOf(err), fetchedAt };
  }
  // models.dev enrichment is optional; if it fails, Zen IDs are still authoritative.
  let devModels: NormalizedModel[] = [];
  let devText = "";
  try {
    const dr = await safeFetch(MODELS_DEV_URL, fetchFn, signal);
    devText = dr.text;
    devModels = normalizeModelsDev(JSON.parse(devText));
  } catch {
    /* enrichment unavailable: keep Zen IDs with minimal metadata */
  }
  const devMap = new Map<string, NormalizedModel>(devModels.map((m) => [m.id, m]));
  const source = "opencode-zen";
  const models: NormalizedModel[] = zenIds.map((id) => {
    const base = devMap.get(id);
    if (base) return { ...base, provider: "opencode", id };
    // Zen-routable but no models.dev metadata: carry the id only.
    const m: NormalizedModel = { provider: "opencode", id };
    m.notes = [`${source}: live zen offering (no models.dev metadata)`];
    return m;
  });
  // Digest covers both the Zen authority and the enrichment payload.
  const digest = sha256Hex(zenText + "\n" + devText);
  return { id: "opencode-zen", status: "ok", digest, models, fetchedAt };
}

/**
 * Fetch OpenCode Go: the LIVE public Go catalog
 * (https://opencode.ai/zen/go/v1/models) is the availability authority for
 * `opencode-go` — only its IDs are routable. models.dev is fetched only to
 * ENRICH those IDs with canonical metadata (matched by slug). Same shape /
 * contract as the Zen source; the two live catalogs must never be confused.
 * Never throws: returns unavailable if the Go authority itself cannot be fetched.
 */
export async function fetchOpenCodeGoSource(
  fetchFn: FetchFn,
  signal?: AbortSignal,
): Promise<SourceResult> {
  const fetchedAt = nowIso();
  const goUrl = SOURCE_ENDPOINTS["opencode-go"];
  let goText: string;
  let goDigest: string;
  try {
    const r = await safeFetch(goUrl, fetchFn, signal);
    goText = r.text;
    goDigest = r.digest;
  } catch (err) {
    return { id: "opencode-go", status: "unavailable", digest: sha256Hex(`unavailable:opencode-go:${messageOf(err)}`), models: [], error: messageOf(err), fetchedAt };
  }
  let goIds: string[];
  try {
    goIds = parseLiveCatalogIds(JSON.parse(goText), "opencode go");
  } catch (err) {
    return { id: "opencode-go", status: "unavailable", digest: goDigest, models: [], error: messageOf(err), fetchedAt };
  }
  // models.dev enrichment is optional (same metadata source as Zen).
  let devModels: NormalizedModel[] = [];
  let devText = "";
  try {
    const dr = await safeFetch(MODELS_DEV_URL, fetchFn, signal);
    devText = dr.text;
    devModels = normalizeModelsDev(JSON.parse(devText));
  } catch {
    /* enrichment unavailable: keep Go IDs with minimal metadata */
  }
  const devMap = new Map<string, NormalizedModel>(devModels.map((m) => [m.id, m]));
  const source = "opencode-go";
  const models: NormalizedModel[] = goIds.map((id) => {
    const base = devMap.get(id);
    if (base) return { ...base, provider: "opencode-go", id };
    const m: NormalizedModel = { provider: "opencode-go", id };
    m.notes = [`${source}: live go offering (no models.dev metadata)`];
    return m;
  });
  const digest = sha256Hex(goText + "\n" + devText);
  return { id: "opencode-go", status: "ok", digest, models, fetchedAt };
}

// ─── Source dispatcher ───────────────────────────────────────────────────────

/** Fetch and normalize one official source. Never throws: returns unavailable. */
export async function fetchSource(
  id: OfficialSourceId,
  fetchFn: FetchFn,
  signal?: AbortSignal,
): Promise<SourceResult> {
  const fetchedAt = nowIso();
  if (id === "opencode-zen") {
    return fetchOpenCodeZenSource(fetchFn, signal);
  }
  if (id === "opencode-go") {
    return fetchOpenCodeGoSource(fetchFn, signal);
  }
  const url = SOURCE_ENDPOINTS[id];
  try {
    const { text, digest } = await safeFetch(url, fetchFn, signal);
    const parsed = JSON.parse(text);
    let models: NormalizedModel[];
    // hyper is metadata-only (effort levels); openrouter is the only other id
    // reaching this branch (zen + go are handled above).
    if (id === "hyper") {
      models = normalizeHyperList(parsed);
    } else {
      models = normalizeOpenRouterList(parsed, "openrouter");
    }
    return { id, status: "ok", digest, models, fetchedAt };
  } catch (err) {
    return {
      id,
      status: "unavailable",
      digest: sha256Hex(`unavailable:${id}:${messageOf(err)}`),
      models: [],
      error: messageOf(err),
      fetchedAt,
    };
  }
}

// ─── OpenCode routability probes (new zen/go models only) ─────────────────────

/** Live chat endpoints used ONLY to probe whether a NEWLY listed zen/go model
 *  is actually served by the router (the /models catalogs are the availability
 *  authority, but freshly listed models sometimes 404/500 until deployed). */
export const OPENCODE_PROBE_ENDPOINTS: Readonly<Record<"opencode" | "opencode-go", string>> = {
  opencode: "https://opencode.ai/zen/v1/chat/completions",
  "opencode-go": "https://opencode.ai/zen/go/v1/chat/completions",
};

export interface ProbeOutcome {
  readonly status: "routable" | "unroutable" | "inconclusive";
  readonly detail: string;
}

/** One 1-token POST against the live chat endpoint. Never throws.
 *  Deterministic evidence of NOT being served (404 no-route, 5xx, explicit
 *  "not supported") marks the model unroutable; throttling (429) proves the
 *  route exists; anything else (bad key, network error) is inconclusive and
 *  never blocks an add. */
export async function probeOpenCodeModel(
  provider: "opencode" | "opencode-go",
  modelId: string,
  apiKey: string,
  fetchFn: FetchFn,
  signal?: AbortSignal,
): Promise<ProbeOutcome> {
  const url = OPENCODE_PROBE_ENDPOINTS[provider];
  const timeout = AbortSignal.timeout(REQUEST_TIMEOUT_MS);
  const combined =
    signal !== undefined ? AbortSignal.any([signal, timeout]) : timeout;
  let status: number;
  let body = "";
  try {
    const response = await fetchFn(url, {
      method: "POST",
      redirect: "error",
      signal: combined,
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ model: modelId, messages: [{ role: "user", content: "ping" }], max_tokens: 1, stream: false }),
    });
    status = response.status;
    // Cap the read: probe answers are tiny; error pages are bounded HTML.
    body = (await response.text()).slice(0, 4096);
  } catch (err) {
    return { status: "inconclusive", detail: messageOf(err) };
  }
  if (status >= 200 && status < 300) return { status: "routable", detail: `HTTP ${status}` };
  if (status === 429) return { status: "routable", detail: "HTTP 429 (throttled; route exists)" };
  if (status === 404) return { status: "unroutable", detail: "HTTP 404 (no route)" };
  if (status >= 500) return { status: "unroutable", detail: `HTTP ${status} (upstream error)` };
  const lower = body.toLowerCase();
  if (
    (status === 400 || status === 401 || status === 403) &&
    (lower.includes("not supported") || lower.includes("not found"))
  ) {
    return { status: "unroutable", detail: `HTTP ${status}: model not supported` };
  }
  return { status: "inconclusive", detail: `HTTP ${status}` };
}

// ─── helpers ─────────────────────────────────────────────────────────────────

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function asString(v: unknown): string | undefined {
  return typeof v === "string" && v.length > 0 ? v : undefined;
}

/**
 * Strict whole-string numeric parse. Unlike Number.parseFloat, this REJECTS
 * partial numeric strings with trailing garbage (e.g. "12junk" -> undefined,
 * not 12). Accepts finite numbers and pure numeric strings (incl. sign,
 * decimal, exponent). Returns undefined for NaN/Infinity/empty/non-numeric.
 */
function asStrictNumber(v: unknown): number | undefined {
  if (typeof v === "number") return Number.isFinite(v) ? v : undefined;
  if (typeof v === "string") {
    const s = v.trim();
    if (s.length === 0) return undefined;
    // Number() (not parseFloat) yields NaN for trailing garbage; reject those.
    const n = Number(s);
    return Number.isFinite(n) ? n : undefined;
  }
  return undefined;
}

/**
 * Strict non-negative finite number for model limits (context/output tokens).
 * Rejects partial numerics, negatives, NaN, and Infinity. A rejected value is
 * treated as unknown (never coerced to 0) so it cannot become an operation.
 */
function asNonNegLimit(v: unknown): number | undefined {
  const n = asStrictNumber(v);
  if (n === undefined || n < 0 || !Number.isFinite(n)) return undefined;
  return n;
}

function uniqueNumbers(arr: number[]): number[] {
  return [...new Set(arr)].sort((a, b) => a - b);
}
