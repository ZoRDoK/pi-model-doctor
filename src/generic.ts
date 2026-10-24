/**
 * Sync for every provider without dedicated doctor rules (OpenAI-compatible,
 * Anthropic, Gemini), using the transport and credentials already configured.
 *
 * The live /models listing decides which ids exist. Ids pi already serves are
 * never duplicated; a genuinely new id is added only when the listing publishes
 * both limits. Existing entries are never rewritten, nothing is removed, and the
 * api key is used for the request only — never logged or copied anywhere.
 */
import type {
  FetchFn,
  FetchResponse,
  GenericProviderProtocol,
  ManagedProvider,
  ModelEntry,
  ModelsConfig,
  NormalizedModel,
  PatchOperation,
  ProviderEntry,
} from "./types.ts";
import { isManagedProvider, protocolForApi } from "./types.ts";
import { asPositiveNumber, asString, evidence, isObject } from "./util.ts";
import { makeSkipped, missingLimitReason, upsertNewModelOps, type ModelResult, type ProviderTotal } from "./doctor.ts";

const GENERIC_TIMEOUT_MS = 15_000;
const GENERIC_MAX_BYTES = 2_000_000;

interface GenericCandidate {
  readonly provider: string;
  readonly protocol: GenericProviderProtocol;
  readonly baseUrl: string;
  readonly api: string;
  readonly apiKey: string | undefined;
}

/** Read plain API keys from auth.json ({provider:{key}}). OAuth entries are
 *  ignored (no static key). Values stay in memory and never reach output. */
export function readAuthKeys(text: string | undefined): Record<string, string> {
  if (!text) return {};
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return {};
  }
  if (!isObject(parsed)) return {};

  const keys: Record<string, string> = {};

  for (const [provider, value] of Object.entries(parsed)) {
    const key = isObject(value) ? asString(value.key) : undefined;
    if (key !== undefined) keys[provider] = key;
  }

  return keys;
}

function resolveApiKey(provider: string, entry: ProviderEntry, authKeys: Record<string, string>): string | undefined {
  const declared = entry.apiKey;
  if (typeof declared === "string" && declared.length > 0) {
    const env = /^\$([A-Za-z_][A-Za-z0-9_]*)$/.exec(declared);
    if (env) {
      const fromEnv = process.env[env[1] ?? ""];
      if (fromEnv && fromEnv.length > 0) return fromEnv;
      // Unset env var: fall through to auth.json, never hard-fail.
    } else {
      return declared;
    }
  }
  return authKeys[provider];
}

function firstModelApi(entry: ProviderEntry): { api?: string; baseUrl?: string } {
  const models = Array.isArray(entry.models) ? entry.models : [];
  for (const m of models as ModelEntry[]) {
    if (typeof m.api === "string" || typeof m.baseUrl === "string") {
      return { api: typeof m.api === "string" ? m.api : undefined, baseUrl: typeof m.baseUrl === "string" ? m.baseUrl : undefined };
    }
  }
  return {};
}

/** Build the /models listing URL for a protocol from the configured base. */
export function listingUrl(protocol: GenericProviderProtocol, baseUrl: string): string {
  const base = baseUrl.replace(/\/+$/, "");
  const versioned = /\/v\d+[a-z]*$/.test(base);
  if (protocol === "anthropic") {
    return versioned ? base + "/models" : base + "/v1/models";
  }
  if (protocol === "google") {
    return base + "/v1beta/models";
  }
  return versioned ? base + "/models" : base + "/v1/models";
}

/** Discover generic sync candidates from the user's models.json. */
export function genericCandidates(
  userModels: ModelsConfig,
  authKeys: Record<string, string>,
  activeProvider: (provider: string) => boolean,
): GenericCandidate[] {
  const out: GenericCandidate[] = [];
  for (const [provider, entryRaw] of Object.entries(userModels.providers ?? {})) {
    if (isManagedProvider(provider)) continue;
    const entry = entryRaw ?? {};
    const inherited = firstModelApi(entry);
    const api = typeof entry.api === "string" ? entry.api : inherited.api;
    const baseUrl = typeof entry.baseUrl === "string" ? entry.baseUrl : inherited.baseUrl;
    const protocol = protocolForApi(api);
    if (!protocol) continue; // no known wire protocol -> doctor leaves it alone
    if (!baseUrl) continue; // no transport base -> nothing to list
    if (!activeProvider(provider)) continue; // not configured in pi -> skip
    out.push({
      provider,
      protocol,
      baseUrl,
      api: api as string,
      apiKey: resolveApiKey(provider, entry, authKeys),
    });
  }
  return out;
}

interface ListedModel {
  id: string;
  name?: string;
  contextWindow?: number;
  maxTokens?: number;
  input?: ("text" | "image")[];
}

/** Positive finite number or undefined (listings often carry strings/null). */
/** Limits an OpenAI-compatible listing MAY publish. Only unambiguous field
 *  names are read: a bare `max_tokens` describes nothing specific enough to
 *  trust as an output ceiling and is ignored. */
function listedLimits(row: Record<string, unknown>): { contextWindow?: number; maxTokens?: number } {
  return {
    contextWindow: asPositiveNumber(row.context_length) ?? asPositiveNumber(row.context_window) ?? asPositiveNumber(row.max_context_length),
    maxTokens: asPositiveNumber(row.max_completion_tokens) ?? asPositiveNumber(row.max_output_tokens),
  };
}

const ID_RE = /^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$/;

function parseListing(protocol: GenericProviderProtocol, text: string): ListedModel[] {
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch {
    throw new Error("non-json response");
  }
  const root = (json ?? {}) as Record<string, unknown>;
  const rows: unknown[] = Array.isArray(root.data)
    ? root.data
    : Array.isArray(root.models)
      ? root.models
      : [];
  const out: ListedModel[] = [];
  for (const rowRaw of rows) {
    if (rowRaw === null || typeof rowRaw !== "object") continue;
    const row = rowRaw as Record<string, unknown>;
    let id: string | undefined;
    let name: string | undefined;
    let contextWindow: number | undefined;
    let maxTokens: number | undefined;
    let input: ("text" | "image")[] | undefined;
    if (protocol === "google") {
      // Only chat-capable models (generateContent) are useful to pi.
      const methods = row.supportedGenerationMethods;
      if (Array.isArray(methods) && !methods.includes("generateContent")) continue;
      if (typeof row.name === "string") id = row.name.replace(/^models\//, "");
      if (typeof row.displayName === "string") name = row.displayName;
      if (typeof row.inputTokenLimit === "number" && row.inputTokenLimit > 0) contextWindow = row.inputTokenLimit;
      if (typeof row.outputTokenLimit === "number" && row.outputTokenLimit > 0) maxTokens = row.outputTokenLimit;
      const modalities = row.inputModalities;
      if (Array.isArray(modalities)) {
        const kinds = modalities.filter((m): m is string => typeof m === "string");
        input = [];
        if (kinds.includes("text")) input.push("text");
        if (kinds.includes("image")) input.push("image");
        if (input.length === 0) input = undefined;
      }
    } else {
      if (typeof row.id === "string") id = row.id;
      if (typeof row.display_name === "string") name = row.display_name;
      const limits = listedLimits(row);
      contextWindow = limits.contextWindow;
      maxTokens = limits.maxTokens;
    }
    if (!id || !ID_RE.test(id)) continue;
    out.push(name || contextWindow || maxTokens || input ? { id, name, contextWindow, maxTokens, input } : { id });
  }
  return out;
}

async function fetchListing(
  candidate: GenericCandidate,
  fetchFn: FetchFn,
  signal: AbortSignal | undefined,
): Promise<ListedModel[]> {
  const url = listingUrl(candidate.protocol, candidate.baseUrl);
  const headers: Record<string, string> = {};
  let requestUrl = url;
  if (candidate.protocol === "anthropic") {
    headers["anthropic-version"] = "2023-06-01";
    if (candidate.apiKey) headers["x-api-key"] = candidate.apiKey;
  } else if (candidate.protocol === "google") {
    if (!candidate.apiKey) throw new Error("missing api key");
    requestUrl = url + "?key=" + encodeURIComponent(candidate.apiKey) + "&pageSize=1000";
  } else if (candidate.apiKey) {
    headers.Authorization = "Bearer " + candidate.apiKey;
  }
  const parsed = new URL(requestUrl);
  if (parsed.protocol !== "https:" && parsed.protocol !== "http:") {
    throw new Error("unsupported url scheme");
  }
  if (parsed.username || parsed.password) throw new Error("credentials in url");
  let response: FetchResponse;
  try {
    response = await fetchFn(requestUrl, {
      method: "GET",
      headers,
      redirect: "error",
      signal: signal !== undefined ? AbortSignal.any([signal, AbortSignal.timeout(GENERIC_TIMEOUT_MS)]) : AbortSignal.timeout(GENERIC_TIMEOUT_MS),
    });
  } catch {
    throw new Error("request failed");
  }
  if (!response.ok) throw new Error("http " + response.status);
  const contentType = response.headers.get("content-type") ?? "";
  if (!/json/i.test(contentType)) throw new Error("non-json content-type");
  const text = await response.text();
  if (text.length > GENERIC_MAX_BYTES) throw new Error("oversized response");
  return parseListing(candidate.protocol, text);
}

function toNormalized(candidate: GenericCandidate, m: ListedModel): NormalizedModel {
  const source = candidate.provider + ":models";
  const nm: NormalizedModel = { provider: candidate.provider, id: m.id };

  nm.name = evidence(m.name ?? m.id, source);
  nm.api = evidence(candidate.api, source);
  nm.baseUrl = evidence(candidate.baseUrl, source);
  if (m.contextWindow !== undefined) nm.contextWindow = evidence(m.contextWindow, source);
  if (m.maxTokens !== undefined) nm.maxTokens = evidence(m.maxTokens, source);
  if (m.input !== undefined) nm.input = evidence(m.input, source);
  return nm;
}

export interface GenericSyncResult {
  readonly operations: readonly PatchOperation[];
  readonly results: readonly ModelResult[];
  readonly totals: readonly ProviderTotal[];
  readonly warnings: readonly string[];
}

export interface GenericSyncOptions {
  readonly userModels: ModelsConfig;
  readonly authKeys: Record<string, string>;
  readonly activeProvider: (provider: string) => boolean;
  /** True when Pi's own catalog already serves this id for the provider. */
  readonly effectiveHasId: (provider: string, id: string) => boolean;
  readonly fetchFn: FetchFn;
  readonly signal?: AbortSignal;
}

/** Sync every generic candidate: fetch listings (in parallel), add new ids. */
export async function syncGenericProviders(opts: GenericSyncOptions): Promise<GenericSyncResult> {
  const candidates = genericCandidates(opts.userModels, opts.authKeys, opts.activeProvider);
  const operations: PatchOperation[] = [];
  const results: ModelResult[] = [];
  const totals: ProviderTotal[] = [];
  const warnings: string[] = [];

  const listings = await Promise.all(
    candidates.map(async (candidate) => {
      try {
        return { candidate, models: await fetchListing(candidate, opts.fetchFn, opts.signal) };
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { candidate, models: undefined, error: message };
      }
    }),
  );

  for (const { candidate, models, error } of listings) {
    const entry = (opts.userModels.providers[candidate.provider] ?? {}) as ProviderEntry;
    const existing = Array.isArray(entry.models) ? (entry.models as ModelEntry[]) : [];
    let added = 0;
    let skipped = 0;
    if (models === undefined) {
      warnings.push("generic_source_unavailable: " + candidate.provider + " (" + error + ")");
      totals.push({ provider: candidate.provider, sourceStatus: "unavailable", added: 0, updated: 0, skipped: 0, unchanged: existing.length });
      continue;
    }
    for (const m of models) {
      if (existing.some((e) => e.id === m.id)) {
        continue; // already configured — generic sync never rewrites entries
      }
      if (opts.effectiveHasId(candidate.provider, m.id)) {
        continue; // Pi's own catalog already provides it — never duplicate it
      }
      const nm = toNormalized(candidate, m);
      const gap = missingLimitReason(nm);
      if (gap !== undefined) {
        skipped += 1;
        results.push(makeSkipped(candidate.provider, m.id, gap, nm));
        continue;
      }
      const ops = upsertNewModelOps(candidate.provider, nm);
      if (ops.length > 0) {
        added += 1;
        operations.push(...ops);
        results.push({
          provider: candidate.provider,
          modelId: m.id,
          action: "added",
          characteristics: {
            name: nm.name?.value,
            api: nm.api?.value,
            baseUrl: nm.baseUrl?.value,
            contextWindow: nm.contextWindow?.value,
            maxTokens: nm.maxTokens?.value,
            input: nm.input?.value ? [...nm.input.value] : undefined,
            source: candidate.provider + ":models",
          },
        });
      } else {
        skipped += 1;
      }
    }
    totals.push({ provider: candidate.provider, sourceStatus: "ok", added, updated: 0, skipped, unchanged: existing.length });
  }
  return { operations, results, totals, warnings };
}
