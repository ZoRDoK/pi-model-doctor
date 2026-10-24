/**
 * Shared types for the model-doctor extension. Declarations only: no imports,
 * no side effects.
 */

/** Providers with dedicated official sources and reconcile rules. */
export type ManagedProvider = "openrouter" | "opencode" | "opencode-go";

export const MANAGED_PROVIDERS: readonly ManagedProvider[] = ["openrouter", "opencode", "opencode-go"];

export function isManagedProvider(value: string): value is ManagedProvider {
  return (MANAGED_PROVIDERS as readonly string[]).includes(value);
}

/** Wire protocol for every other provider, mapped from its models.json `api`. */
export type GenericProviderProtocol = "openai" | "anthropic" | "google";

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
 * Official metadata sources.
 * - openrouter: public models list + per-model endpoints view.
 * - opencode-zen / opencode-go: live catalogs (availability) + models.dev metadata.
 * - hyper: Charm Hyper reasoning-effort levels; metadata only, never added.
 */
export type OfficialSourceId = "openrouter" | "opencode-zen" | "opencode-go" | "hyper";

/** Model fields a patch operation may change; everything else is preserved. */
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

/** Composite paths are merged subfield-by-subfield; other members survive. */
export function isCompositePath(value: string): boolean {
  return value === "cost" || value === "compat" || value === "thinkingLevelMap";
}

/** Allowlisted subfields per composite path; anything else is preserved and
 *  never disclosed. */
export const COMPOSITE_SUBFIELDS: Readonly<Record<"cost" | "compat" | "thinkingLevelMap", readonly string[]>> = {
  cost: ["input", "output", "cacheRead", "cacheWrite", "tiers"],
  thinkingLevelMap: ["off", "minimal", "low", "medium", "high", "xhigh", "max"],
  compat: [
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
  ],
};

export function allowlistedSubfields(path: MutationPath): readonly string[] {
  return COMPOSITE_SUBFIELDS[path as "cost" | "compat" | "thinkingLevelMap"] ?? [];
}

/** One piece of normalized metadata plus its provenance. Missing evidence means
 *  unknown, never a default. */
export interface Evidence<T> {
  readonly value: T;
  readonly source: string;
  /** False when the value is ambiguous or unconfirmed: report it, never write it. */
  readonly plannable: boolean;
  readonly note?: string;
}

/** Cost in USD per million tokens. Only published rates are set. */
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

/** Normalized metadata for one model id, scoped to a provider. */
export interface NormalizedModel {
  readonly provider: string;
  readonly id: string;
  name?: Evidence<string>;
  api?: Evidence<string>;
  baseUrl?: Evidence<string>;
  reasoning?: Evidence<boolean>;
  thinkingLevelMap?: Evidence<Readonly<Record<string, string | null>>>;
  input?: Evidence<ReadonlyArray<"text" | "image">>;
  contextWindow?: Evidence<number>;
  maxTokens?: Evidence<number>;
  cost?: Evidence<NormalizedCost>;
  compat?: Evidence<Readonly<Record<string, unknown>>>;
  /** Informational; never patched. */
  aliases?: string[];
  notes?: string[];
  /** Creation timestamp; used only by the OpenRouter retire cutoff. */
  createdMs?: number;
}

export interface SourceResult {
  readonly id: OfficialSourceId;
  readonly status: "ok" | "unavailable";
  /** sha256 of the fetched payload (or of the composed list for mixed sources). */
  readonly digest: string;
  readonly models: readonly NormalizedModel[];
  readonly error?: string;
  readonly fetchedAt: string;
}

export type Severity = "info" | "warning" | "error";

/** A diagnostic finding. `ref` never contains secret values. */
export interface Finding {
  readonly provider: string;
  readonly modelId?: string;
  readonly severity: Severity;
  readonly code: string;
  readonly message: string;
  readonly ref?: string;
}

/** One surgical, allowlisted change to a model field in models.json. */
export interface PatchOperation {
  readonly provider: string;
  readonly modelId: string;
  readonly path: MutationPath;
  readonly before: unknown;
  readonly after: unknown;
  readonly source: string;
  readonly reason: string;
}

/** What Pi currently serves, per provider (built-in catalog + models.json). */
export interface EffectiveSnapshot {
  readonly providers: Readonly<Record<string, readonly NormalizedModel[]>>;
}

export interface Diagnosis {
  readonly sources: readonly SourceResult[];
  readonly findings: readonly Finding[];
  readonly effective: EffectiveSnapshot;
  /** `provider/modelId` keys the live catalog lists but the router cannot serve. */
  readonly unroutable?: ReadonlySet<string>;
}

/** Operations held in memory until apply; every input is fingerprinted so a
 *  concurrent change refuses the write instead of clobbering it. */
export interface PendingPlan {
  readonly id: string;
  readonly createdAt: string;
  readonly expiresAt: string;
  readonly target: string;
  readonly modelsHash: string;
  readonly settingsHash: string;
  readonly sourceDigests: Readonly<Record<string, string>>;
  readonly effectiveDigest: string;
  /** Targeted openrouter plan: detail endpoint url + sha256 of its bytes. */
  readonly detail?: { readonly url: string; readonly digest: string };
  readonly operations: readonly PatchOperation[];
  readonly findings: readonly Finding[];
}

/** A model entry in models.json. Unknown members are preserved untouched. */
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

/** A provider entry in models.json. Transport fields are never rewritten. */
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

export interface ModelsConfig {
  providers: Record<string, ProviderEntry>;
}

export interface ApplyResult {
  readonly ok: boolean;
  readonly applied: number;
  readonly backups: readonly string[];
  readonly error?: string;
  readonly findings: readonly Finding[];
}

/** Paths are injected so every subsystem runs against temp dirs in tests. */
export interface DoctorPaths {
  readonly modelsPath: string;
  readonly settingsPath: string;
  readonly backupDir: string;
  /** Optional auth.json path: enables routability probes of new zen/go models. */
  readonly authPath?: string;
}

/** Network transport, injectable for tests. */
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
