/**
 * Shared types and constants for the model-doctor extension.
 *
 * No runtime imports. This module is pure declarations so it can be imported
 * from both the extension entry and the test suite without side effects.
 */

/**
 * Provider ids with dedicated official sources and hard-coded reconcile rules.
 * Every OTHER configured provider is synced generically by wire protocol
 * (see GenericProviderProtocol below): the live /models listing decides which
 * model ids exist; only genuinely new ids are added, nothing is removed.
 */
export type ManagedProvider = "openrouter" | "opencode" | "opencode-go";

export const MANAGED_PROVIDERS: readonly ManagedProvider[] = [
  "openrouter",
  "opencode",
  "opencode-go",
];

export function isManagedProvider(value: string): value is ManagedProvider {
  return (MANAGED_PROVIDERS as readonly string[]).includes(value);
}

/**
 * Wire protocols recognized for generic (non-managed) providers. Resolved from
 * the provider/model `api` field in models.json.
 *
 * - `openai`    -> OpenAI-compatible GET {base}/models, Authorization: Bearer.
 * - `anthropic` -> Anthropic GET {base}/v1/models, x-api-key + anthropic-version.
 * - `google`    -> Gemini GET {base}/v1beta/models?key= (limits applied when present).
 */
export type GenericProviderProtocol = "openai" | "anthropic" | "google";

/** api values (provider- or model-level) mapped onto generic protocols. */
export const GENERIC_PROTOCOL_APIS: Readonly<Record<string, GenericProviderProtocol>> = {
  "openai-completions": "openai",
  "openai-responses": "openai",
  "anthropic-messages": "anthropic",
  "google-generative-ai": "google",
};

export function protocolForApi(api: string | undefined): GenericProviderProtocol | undefined {
  if (!api) return undefined;
  return GENERIC_PROTOCOL_APIS[api];
}

/**
 * The official, allowlisted metadata sources.
 *
 * - `openrouter`   -> OpenRouter public models + per-model endpoints API.
 * - `opencode-zen` -> live OpenCode Zen catalog (availability) + models.dev metadata.
 * - `opencode-go`  -> live OpenCode Go catalog (availability) + models.dev metadata.
 * - `hyper`        -> Charm Hyper public catalog; metadata-only (reasoning
 *                     effort levels). Never a managed provider: no provider
 *                     block is created and no model is added from it.
 */
export type OfficialSourceId = "openrouter" | "opencode-zen" | "opencode-go" | "hyper";

export const OFFICIAL_SOURCES: readonly OfficialSourceId[] = [
  "openrouter",
  "opencode-zen",
  "opencode-go",
  "hyper",
];

/**
 * Model-entry fields that an apply operation is allowed to change. Everything
 * else (provider-level transport, unknown model props) is left untouched.
 */
export type MutationPath =
  | "name"
  | "api"
  | "baseUrl"
  | "reasoning"
  | "thinkingLevelMap"
  | "input"
  | "contextWindow"
  | "maxTokens"
  | "cost"
  | "compat";

export const MUTATION_PATHS: readonly MutationPath[] = [
  "name",
  "api",
  "baseUrl",
  "reasoning",
  "thinkingLevelMap",
  "input",
  "contextWindow",
  "maxTokens",
  "cost",
  "compat",
];

export function isMutationPath(value: string): value is MutationPath {
  return (MUTATION_PATHS as readonly string[]).includes(value);
}

/** Composite (object/map) mutation paths whose value is merged subfield-by-subfield. */
export const COMPOSITE_PATHS: ReadonlySet<MutationPath> = new Set([
  "cost",
  "compat",
  "thinkingLevelMap",
]);

export function isCompositePath(value: string): boolean {
  return COMPOSITE_PATHS.has(value as MutationPath);
}

/**
 * Allowlisted subfields for each composite path. When merging a composite
 * change, ONLY these subfields may be set/changed; every other existing member
 * (e.g. an unknown `vendorSecret`) is preserved untouched and never disclosed.
 */
export const COST_SUBFIELDS = ["input", "output", "cacheRead", "cacheWrite", "tiers"] as const;
export const THINKING_SUBFIELDS = [
  "off", "minimal", "low", "medium", "high", "xhigh", "max",
] as const;
export const COMPAT_SUBFIELDS = [
  "supportsStore", "supportsDeveloperRole", "supportsReasoningEffort",
  "supportsUsageInStreaming", "supportsFinishReason", "maxTokensField",
  "requiresToolResultName", "requiresAssistantAfterToolResult",
  "requiresThinkingAsText", "requiresReasoningContentOnAssistantMessages",
  "thinkingFormat", "chatTemplateKwargs", "chatTemplateArgs",
  "openRouterRouting", "vercelGatewayRouting", "zaiToolStream",
  "supportsThinkingTokenBudget", "supportsOpenAIGrammarTools",
  "supportsStrictMode", "cacheControlFormat", "sendSessionAffinityHeaders",
  "deferredToolsMode", "sessionAffinityFormat", "supportsLongCacheRetention",
  "supportsToolSearch", "supportsExplicitPromptCacheMode",
  "supportsEagerToolInputStreaming", "supportsCacheControlOnTools",
  "supportsTemperature", "forceAdaptiveThinking", "allowEmptySignature",
  "supportsStrictTools", "supportsToolReferences",
] as const;

export const COMPOSITE_SUBFIELDS: Readonly<Record<"cost" | "compat" | "thinkingLevelMap", readonly string[]>> = {
  cost: COST_SUBFIELDS,
  compat: COMPAT_SUBFIELDS,
  thinkingLevelMap: THINKING_SUBFIELDS,
};

/** Subfields allowed for a composite path (empty for scalar paths). */
export function allowlistedSubfields(path: MutationPath): readonly string[] {
  return COMPOSITE_SUBFIELDS[path as "cost" | "compat" | "thinkingLevelMap"] ?? [];
}

/**
 * Provider-level fields that model-doctor must NEVER touch, read into evidence,
 * or echo. They are transport/auth and are out of scope for catalog sync.
 */
export const PROVIDER_TRANSPORT_FIELDS: readonly string[] = [
  "apiKey",
  "baseUrl",
  "headers",
  "authHeader",
  "name",
  "oauth",
  "streamSimple",
  "refreshModels",
  "compat",
] as const;

/** Confidence of a piece of evidence. */
export type Confidence = "authoritative" | "reported" | "unknown";

/**
 * A single piece of normalized metadata plus its provenance. `plannable` is
 * false when the value is ambiguous or unconfirmed and therefore must NOT be
 * turned into an apply operation.
 */
export interface Evidence<T> {
  readonly value: T;
  readonly source: string;
  readonly confidence: Confidence;
  readonly plannable: boolean;
  readonly note?: string;
}

/** Normalized cost in USD per million tokens. Only present fields are set. */
export interface NormalizedCost {
  readonly input?: number;
  readonly output?: number;
  readonly cacheRead?: number;
  readonly cacheWrite?: number;
  readonly tiers?: ReadonlyArray<{
    readonly inputTokensAbove: number;
    readonly input?: number;
    readonly output?: number;
    readonly cacheRead?: number;
    readonly cacheWrite?: number;
  }>;
}

/** Provider scope of a normalized model: a managed provider block, or the
 *  metadata-only `hyper` catalog (never reconciled into models.json itself). */
/** Managed special providers, "hyper", or any generic provider id. */
export type NormalizedProvider = ManagedProvider | "hyper" | (string & {});

/**
 * Normalized model metadata for one model id, scoped to a managed provider.
 * Each field is optional Evidence: absence means "unknown" (never a default).
 */
export interface NormalizedModel {
  readonly provider: NormalizedProvider;
  readonly id: string;
  name?: Evidence<string>;
  api?: Evidence<string>;
  /** Transport base URL (generic providers only; inherited, never invented). */
  baseUrl?: Evidence<string>;
  reasoning?: Evidence<boolean>;
  thinkingLevelMap?: Evidence<Readonly<Record<string, string | null>>>;
  input?: Evidence<ReadonlyArray<"text" | "image">>;
  contextWindow?: Evidence<number>;
  maxTokens?: Evidence<number>;
  cost?: Evidence<NormalizedCost>;
  compat?: Evidence<Readonly<Record<string, unknown>>>;
  /** Informational aliases/links; never patched. */
  aliases?: string[];
  /** Informational notes; never patched. */
  notes?: string[];
  /** Informational creation timestamp (ms since epoch); never patched.
   *  Used by the OpenRouter retire-cutoff (pre-2026-04 models are never
   *  re-added). */
  createdMs?: number;
}

/** Result of fetching one official source. */
export interface SourceResult {
  readonly id: OfficialSourceId;
  readonly status: "ok" | "unavailable";
  /** sha256 of the canonical fetched payload (raw bytes for network sources). */
  readonly digest: string;
  readonly models: readonly NormalizedModel[];
  readonly error?: string;
  readonly fetchedAt: string;
}

export type Severity = "info" | "warning" | "error";

/**
 * A diagnostic finding. `provider` is a provider id or "*" for cross-provider /
 * reference findings. `ref` is an allowlisted JSON path or location label and
 * never contains secret values.
 */
export interface Finding {
  readonly provider: string;
  readonly modelId?: string;
  readonly severity: Severity;
  readonly code: string;
  readonly message: string;
  readonly ref?: string;
}

/** A single surgical, allowlisted change to one model field in models.json. */
export interface PatchOperation {
  /** Managed special provider id or a generic provider id from models.json. */
  readonly provider: string;
  readonly modelId: string;
  readonly path: MutationPath;
  readonly before: unknown;
  readonly after: unknown;
  readonly source: string;
  readonly reason: string;
}

/** Effective pi catalog snapshot for the managed providers (read-only). */
export interface EffectiveSnapshot {
  readonly providers: Readonly<Record<ManagedProvider, readonly NormalizedModel[]>>;
}

/** Full diagnosis produced by `check` / `plan`. */
export interface Diagnosis {
  readonly sources: readonly SourceResult[];
  readonly findings: readonly Finding[];
  readonly effective: EffectiveSnapshot;
  /** `provider/modelId` keys that the live catalog lists but a routing probe
   *  could not serve (catalog-vs-router drift); reconcileAll never adds them. */
  readonly unroutable?: ReadonlySet<string>;
}

/**
 * A plan held only in memory by the current extension instance. `apply`
 * requires a non-expired plan whose fingerprints still match the on-disk files
 * AND every input used to derive the operations (config bytes, list sources,
 * the targeted OpenRouter detail payload, and the effective catalog snapshot).
 */
export interface PendingPlan {
  readonly id: string;
  readonly createdAt: string;
  readonly expiresAt: string;
  readonly target: string;
  /** sha256 of raw models.json bytes captured at plan time (same read used for ops). */
  readonly modelsHash: string;
  /** sha256 of raw settings.json bytes captured at plan time. */
  readonly settingsHash: string;
  /** source id -> sha256 of canonical payload captured at plan time. */
  readonly sourceDigests: Readonly<Record<string, string>>;
  /** sha256 of the canonical serialized effective catalog captured at plan time. */
  readonly effectiveDigest: string;
  /** For a targeted openrouter plan: the detail endpoint URL + sha256 of its raw bytes. */
  readonly detail?: { readonly url: string; readonly digest: string };
  readonly operations: readonly PatchOperation[];
  readonly findings: readonly Finding[];
}

/** Loose shape of a model entry inside models.json (user contract). */
export interface ModelEntry {
  id: string;
  name?: string;
  api?: string;
  baseUrl?: string;
  reasoning?: boolean;
  thinkingLevelMap?: Record<string, string | null>;
  input?: ("text" | "image")[];
  contextWindow?: number;
  maxTokens?: number;
  cost?: Record<string, unknown>;
  compat?: Record<string, unknown>;
  headers?: Record<string, string>;
  [key: string]: unknown;
}

/** Loose shape of a provider entry inside models.json (user contract). */
export interface ProviderEntry {
  name?: string;
  baseUrl?: string;
  api?: string;
  apiKey?: string;
  headers?: Record<string, string>;
  authHeader?: boolean;
  models?: ModelEntry[];
  modelOverrides?: Record<string, Partial<ModelEntry>>;
  [key: string]: unknown;
}

/** Loose shape of the whole models.json (user contract). */
export interface ModelsConfig {
  providers: Record<string, ProviderEntry>;
}

/** Result of an apply attempt. */
export interface ApplyResult {
  readonly ok: boolean;
  readonly applied: number;
  readonly backups: readonly string[];
  readonly error?: string;
  readonly findings: readonly Finding[];
}

/** Options bag passed through every subsystem so paths are injectable in tests. */
export interface DoctorPaths {
  readonly modelsPath: string;
  readonly settingsPath: string;
  /** Directory used for timestamped backups (same dir as models.json by default). */
  readonly backupDir: string;
  /** Optional path to pi's auth.json: enables routability probes of NEW
   *  opencode/opencode-go models (read-only; keys never appear in output). */
  readonly authPath?: string;
}

/** Network fetch options, injectable so tests can mock the transport. */
export interface FetchFn {
  (input: string, init?: FetchInit): Promise<FetchResponse>;
}

export interface FetchInit {
  readonly method?: string;
  readonly headers?: Record<string, string>;
  readonly body?: string;
  readonly signal?: AbortSignal;
  readonly redirect?: "follow" | "error" | "manual";
}

export interface FetchResponse {
  readonly ok: boolean;
  readonly status: number;
  readonly url: string;
  readonly headers: {
    get(name: string): string | null;
  };
  text(): Promise<string>;
  arrayBuffer(): Promise<ArrayBuffer>;
}
