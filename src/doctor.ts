/**
 * Core doctor logic: three-way diagnosis (official source vs effective pi
 * catalog vs user models.json), read-only reference validation, surgical
 * allowlisted patch building, in-memory plan construction, and the atomic
 * verify-and-rollback apply flow.
 *
 * Every IO path and the network transport are injected so the whole module is
 * testable against temp directories and a mocked fetch.
 */
import {
  applyOpenRouterEndpoints,
  demoteListLimits,
  fetchSource,
  openRouterDetailUrl,
  probeOpenCodeModel,
  safeFetch,
  sha256Hex,
  nowIso,
  enrichThinkingLevels,
} from "./sources.ts";
import { readAuthKeys, syncGenericProviders } from "./generic.ts";
import {
  deepClone,
  parseConfig,
  readRaw,
  safeWriteWithBackup,
  serializeConfig,
  sha256Buffer,
} from "./storage.ts";
import {
  isMutationPath,
  isManagedProvider,
  isCompositePath,
  allowlistedSubfields,
} from "./types.ts";
import { asString, isObject, messageOf } from "./util.ts";
import type {
  ApplyResult,
  Diagnosis,
  DoctorPaths,
  EffectiveSnapshot,
  Evidence,
  FetchFn,
  Finding,
  ManagedProvider,
  ModelEntry,
  ModelsConfig,
  MutationPath,
  NormalizedModel,
  OfficialSourceId,
  PatchOperation,
  PendingPlan,
  ProviderEntry,
  SourceResult,
} from "./types.ts";

export const DEFAULT_PLAN_TTL_MS = 15 * 60 * 1000;

/** Parsed command target. */
export interface ParsedTarget {
  readonly kind: "all" | "provider" | "model";
  readonly provider?: ManagedProvider;
  readonly modelId?: string;
}

export interface RefCatalog {
  /** Effective model ids for a provider (any provider, not just managed). */
  modelIds(provider: string): readonly string[];
  /** Whether the provider has complete auth for the given model. */
  hasAuth(provider: string, modelId: string): boolean;
}

export interface DiagnoseOptions {
  readonly paths: DoctorPaths;
  readonly fetchFn: FetchFn;
  readonly effective: EffectiveSnapshot;
  readonly refCatalog: RefCatalog;
  readonly signal?: AbortSignal;
}

export interface PlanOptions extends DiagnoseOptions {
  readonly target: ParsedTarget;
  readonly ttlMs?: number;
  readonly now?: () => number;
}

export interface PlanResult {
  readonly diagnosis: Diagnosis;
  readonly plan: PendingPlan;
}

export interface ApplyOptions {
  readonly plan: PendingPlan;
  readonly paths: DoctorPaths;
  readonly fetchFn: FetchFn;
  readonly effective: EffectiveSnapshot;
  readonly refCatalog: RefCatalog;
  readonly now?: () => number;
  readonly signal?: AbortSignal;
  /** Skip re-fetching + comparing source digests at apply. Used by the atomic
   *  one-shot command (no plan->apply review gap); models/settings CAS +
   *  effective revalidation still run. */
  readonly skipSourceRevalidation?: boolean;
}

// ─── Diagnosis ───────────────────────────────────────────────────────────────

/** Fetch all official sources and read the user files; never throws per-source. */
export async function diagnose(opts: DiagnoseOptions): Promise<Diagnosis> {
  const ids: OfficialSourceId[] = ["openrouter", "opencode-zen", "opencode-go", "hyper"];
  const sources = await Promise.all(
    ids.map((id) => fetchSource(id, opts.fetchFn, opts.signal)),
  );

  const findings: Finding[] = [];
  for (const s of sources) {
    if (s.status === "unavailable") {
      findings.push({
        provider: "*",
        severity: "warning",
        code: "source_unavailable",
        message: `official source ${s.id} unavailable: ${s.error ?? "unknown"}`,
      });
    }
  }

  // Read user models.json + settings.json (read-only) for reference findings
  // and secret-safety audit (literal api keys).
  let settingsObj: unknown = undefined;
  let userModels: ModelsConfig | undefined = undefined;
  try {
    const settingsRaw = await readRaw(opts.paths.settingsPath);
    settingsObj = parseConfig(settingsRaw.text);
  } catch (err) {
    findings.push({
      provider: "*",
      severity: "warning",
      code: "settings_unreadable",
      message: `could not read settings.json: ${messageOf(err)}`,
    });
  }
  try {
    const modelsRaw = await readRaw(opts.paths.modelsPath);
    userModels = asModelsConfig(parseConfig(modelsRaw.text));
  } catch (err) {
    findings.push({
      provider: "*",
      severity: "warning",
      code: "models_unreadable",
      message: `could not read models.json: ${messageOf(err)}`,
    });
  }
  if (settingsObj !== undefined) {
    findings.push(...validateReferences(settingsObj, opts.refCatalog));
  }
  if (userModels !== undefined) {
    findings.push(...auditUserModels(userModels, opts.effective));
    findings.push(...diagnoseManagedDrift(userModels, opts.effective, sources));
  }

  return { sources, findings, effective: opts.effective };
}

/**
 * Three-way drift findings: source/effective vs user models.json. Emits
 * drift / ambiguity / catalog_only / alias findings for the managed providers.
 * Read-only: never mutates anything.
 */
export function diagnoseManagedDrift(
  userModels: ModelsConfig,
  effective: EffectiveSnapshot,
  sources: readonly SourceResult[],
): Finding[] {
  const findings: Finding[] = [];
  const driftSource = (provider: ManagedProvider, entry: ModelEntry, nm: NormalizedModel): void => {
    if (nm.aliases && nm.aliases.length > 0) {
      findings.push({
        provider, modelId: entry.id, severity: "info", code: "alias",
        message: `${provider}/${entry.id} canonical alias(es): ${nm.aliases.join(", ")} (not rewritten)`,
        ref: `models.providers.${provider}.models[${entry.id}]`,
      });
    }
    for (const { path } of MODEL_FIELD_KEYS) {
      const ev = evidenceOf(nm, path);
      if (!ev) continue;
      const current = (entry as Record<string, unknown>)[path];
      if (!ev.plannable) {
        findings.push({
          provider, modelId: entry.id, severity: "warning", code: "ambiguity",
          message: `${path} source data is ambiguous/conditional${ev.note ? ": " + ev.note : ""}; not auto-applied`,
          ref: `models.providers.${provider}.models[${entry.id}].${path}`,
        });
      } else if (!canonicalEqual(current, ev.value)) {
        findings.push({
          provider, modelId: entry.id, severity: "info", code: "drift",
          message: `${path} differs from source (${ev.source}): ${brief(current)} -> ${brief(ev.value)}`,
          ref: `models.providers.${provider}.models[${entry.id}].${path}`,
        });
      }
    }
  };

  // Source-backed providers (openrouter, opencode).
  for (const provider of ["openrouter", "opencode"] as ManagedProvider[]) {
    const source = sourceFor({ sources } as Diagnosis, provider);
    if (!source || source.status !== "ok") continue;
    const providerEntry = userModels.providers[provider];
    const models = providerEntry && Array.isArray(providerEntry.models) ? providerEntry.models : [];
    const userModelIds = new Set(models.map((m) => m.id));
    for (const entry of models) {
      const nm = source.models.find((m) => m.id === entry.id);
      if (nm) driftSource(provider, entry, nm);
    }
    for (const nm of source.models) {
      // Retired OpenRouter models are not reported as catalog_only noise.
      if (provider === "openrouter" && openRouterRetired(nm)) continue;
      if (!userModelIds.has(nm.id)) {
        findings.push({
          provider, modelId: nm.id, severity: "info", code: "catalog_only",
          message: `${provider}/${nm.id} is in the official catalog but not in models.json (not added by a no-target/provider plan)`,
          ref: `models.providers.${provider}.models`,
        });
      }
    }
  }

  // Three-way contradiction for the source-backed providers: also compare each
  // user entry to the EFFECTIVE pi catalog (official vs effective-pi vs user).
  // This surfaces cases where the user matches the official source but pi's
  // compiled effective catalog disagrees (e.g. pi is stale or diverges), which
  // the source-vs-user pass above cannot detect on its own.
  for (const provider of ["openrouter", "opencode"] as ManagedProvider[]) {
    const effModels = effective.providers[provider] ?? [];
    const providerEntry = userModels.providers[provider];
    const models = providerEntry && Array.isArray(providerEntry.models) ? providerEntry.models : [];
    for (const entry of models) {
      const nm = effModels.find((m) => m.id === entry.id);
      if (!nm) continue;
      for (const { path } of MODEL_FIELD_KEYS) {
        const ev = evidenceOf(nm, path);
        if (!ev || !ev.plannable) continue;
        const current = (entry as Record<string, unknown>)[path];
        if (!canonicalEqual(current, ev.value)) {
          findings.push({
            provider, modelId: entry.id, severity: "info", code: "effective_drift",
            message: `${path} differs from effective pi catalog (${ev.source}): ${brief(current)} -> ${brief(ev.value)}`,
            ref: `models.providers.${provider}.models[${entry.id}].${path}`,
          });
        }
      }
    }
  }
  return findings;
}

function brief(v: unknown): string {
  if (v === undefined) return "(absent)";
  const s = JSON.stringify(v);
  return s !== undefined && s.length > 60 ? s.slice(0, 57) + "..." : (s ?? "?");
}

/**
 * Secret-safety + drift audit of the user's models.json. Never echoes secret
 * values; a literal apiKey is reported only as "provider has literal apiKey".
 */
export function auditUserModels(
  userModels: ModelsConfig,
  effective: EffectiveSnapshot,
): Finding[] {
  const findings: Finding[] = [];
  const providers = userModels.providers ?? {};

  for (const provider of Object.keys(providers)) {
    const entry = providers[provider];
    if (!entry) continue;
    if (hasLiteralApiKey(entry)) {
      findings.push({
        provider,
        severity: "info",
        code: "literal_apikey",
        message: `provider has literal apiKey (value not shown)`,
        ref: `models.providers.${provider}.apiKey`,
      });
    }
  }

  return findings;
}

function hasLiteralApiKey(entry: ProviderEntry): boolean {
  const key = entry.apiKey;
  if (typeof key !== "string" || key.length === 0) return false;
  // Env interpolation ($VAR / ${VAR}) and shell commands (!cmd) are not literals.
  if (key.startsWith("$") || key.startsWith("!")) return false;
  return true;
}

// ─── Reference validation (read-only) ────────────────────────────────────────

const THINKING_TOKENS = new Set([
  "off",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
]);

interface SplitRef {
  provider: string;
  modelId: string;
  thinking?: string;
  hasGlob: boolean;
  raw: string;
}

function parseRef(raw: string): SplitRef | null {
  if (typeof raw !== "string" || raw.length === 0) return null;
  const trimmed = raw.trim();
  const slash = trimmed.indexOf("/");
  if (slash < 0) return null; // require provider/model form
  const provider = trimmed.slice(0, slash);
  let rest = trimmed.slice(slash + 1);
  // Only the final :thinking-token is split; a ":" inside the real id is kept.
  const colon = rest.lastIndexOf(":");
  let thinking: string | undefined;
  if (colon >= 0 && THINKING_TOKENS.has(rest.slice(colon + 1))) {
    thinking = rest.slice(colon + 1);
    rest = rest.slice(0, colon);
  }
  const hasGlob = /[*?]/.test(rest);
  return { provider, modelId: rest, thinking, hasGlob, raw: trimmed };
}

/** Validate model references in settings against the effective catalog. */
export function validateReferences(
  settingsObj: unknown,
  catalog: RefCatalog,
): Finding[] {
  const findings: Finding[] = [];
  const settings = isObject(settingsObj) ? settingsObj : {};

  const check = (raw: unknown, where: string) => {
    const ref = parseRef(typeof raw === "string" ? raw : "");
    if (!ref) return;
    const ids = catalog.modelIds(ref.provider);
    // Resolve to concrete model ids (globs expanded), so auth is never checked
    // against the literal glob pattern.
    const matchedIds = ref.hasGlob
      ? ids.filter((id) => globMatch(ref.modelId, id))
      : ids.includes(ref.modelId)
        ? [ref.modelId]
        : [];
    if (matchedIds.length === 0) {
      findings.push({
        provider: ref.provider,
        severity: "warning",
        code: "missing_reference",
        message: `referenced model not in effective catalog: ${ref.raw}`,
        ref: where,
      });
      return;
    }
    // Report missing_auth per concrete matched model that lacks auth.
    for (const id of matchedIds) {
      if (!catalog.hasAuth(ref.provider, id)) {
        findings.push({
          provider: ref.provider,
          modelId: id,
          severity: "info",
          code: "missing_auth",
          message: `model known but provider auth not configured: ${ref.provider}/${id}`,
          ref: where,
        });
      }
    }
  };

  const enabled = (settings as { enabledModels?: unknown }).enabledModels;
  if (Array.isArray(enabled)) {
    enabled.forEach((m, i) => check(m, `settings.enabledModels[${i}]`));
  }
  const dp = asString((settings as { defaultProvider?: unknown }).defaultProvider);
  const dm = asString((settings as { defaultModel?: unknown }).defaultModel);
  if (dp && dm) check(`${dp}/${dm}`, "settings.defaultProvider/defaultModel");
  const wvm = asString((settings as { workingVibeModel?: unknown }).workingVibeModel);
  if (wvm) check(wvm, "settings.workingVibeModel");

  const overrides = (settings as { subagents?: { agentOverrides?: unknown } }).subagents
    ?.agentOverrides;
  if (isObject(overrides)) {
    for (const [agent, ov] of Object.entries(overrides)) {
      if (!isObject(ov)) continue;
      const model = asString((ov as { model?: unknown }).model);
      if (model) check(model, `settings.subagents.agentOverrides.${agent}.model`);
      const fb = (ov as { fallbackModels?: unknown }).fallbackModels;
      if (Array.isArray(fb)) {
        fb.forEach((m, i) =>
          check(m, `settings.subagents.agentOverrides.${agent}.fallbackModels[${i}]`),
        );
      }
    }
  }
  return findings;
}

/** Minimal glob matcher supporting `*` and `?` (single-segment, anchored). */
export function globMatch(pattern: string, value: string): boolean {
  let rx = "^";
  for (const ch of pattern) {
    if (ch === "*") rx += ".*";
    else if (ch === "?") rx += ".";
    else rx += ch.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }
  rx += "$";
  return new RegExp(rx).test(value);
}

// ─── Plan building ───────────────────────────────────────────────────────────

export async function buildPlanFromSources(opts: PlanOptions): Promise<PlanResult> {
  const now = opts.now ?? Date.now;
  let diagnosis = await diagnose(opts);

  // Targeted openrouter model: fetch the endpoints detail, fold it in, and
  // capture its digest so apply can detect detail drift.
  let detail: { url: string; digest: string } | undefined;
  if (
    opts.target.kind === "model" &&
    opts.target.provider === "openrouter" &&
    opts.target.modelId
  ) {
    const r = await captureOpenRouterDetail(diagnosis, opts);
    diagnosis = r.diagnosis;
    detail = r.detail;
  }

  // Read models.json ONCE and use the same bytes for operation-building and the
  // fingerprint (avoids the plan-time read/hash race). Strict schema: refuse to
  // plan against a malformed models.json (fail closed).
  let modelsRaw;
  try {
    modelsRaw = await readRaw(opts.paths.modelsPath);
  } catch (err) {
    throw new Error(`cannot read models.json for planning: ${messageOf(err)}`);
  }
  const userModels = parseModelsConfigStrict(modelsRaw.text);
  const operations = buildOperations(diagnosis, userModels, opts.target);

  const settingsRaw = await readRaw(opts.paths.settingsPath);
  const sourceDigests: Record<string, string> = {};
  for (const s of diagnosis.sources) sourceDigests[s.id] = s.digest;

  const ttlMs = opts.ttlMs ?? DEFAULT_PLAN_TTL_MS;
  const createdAt = nowIso(now);
  const expiresAt = new Date(now() + ttlMs).toISOString();
  const plan: PendingPlan = {
    id: cryptoRandomId(),
    createdAt,
    expiresAt,
    target: describeTarget(opts.target),
    modelsHash: modelsRaw.hash,
    settingsHash: settingsRaw.hash,
    sourceDigests,
    effectiveDigest: canonicalEffectiveDigest(opts.effective),
    detail,
    operations,
    findings: diagnosis.findings,
  };
  return { diagnosis, plan };
}

/** Fetch the OpenRouter endpoints payload for the targeted model; fold evidence
 *  in, capture the raw digest for apply-time revalidation, demote list-derived
 *  limits when the detail is malformed/ambiguous/unavailable (so they are never
 *  auto-applied without endpoint confirmation), and emit safe findings. */
async function captureOpenRouterDetail(
  diagnosis: Diagnosis,
  opts: PlanOptions,
): Promise<{ diagnosis: Diagnosis; detail?: { url: string; digest: string } }> {
  const modelId = opts.target.modelId!;
  const orSource = diagnosis.sources.find((s) => s.id === "openrouter");
  if (!orSource || orSource.status !== "ok") return { diagnosis };
  const base = orSource.models.find((m) => m.id === modelId);
  if (!base) return { diagnosis };
  const url = openRouterDetailUrl(modelId);
  const detailFindings: Finding[] = [];
  let digest: string | undefined;
  let refined = base;
  try {
    const fetched = await safeFetch(url, opts.fetchFn, opts.signal);
    digest = fetched.digest;
    const r = applyOpenRouterEndpoints(base, fetched.text);
    refined = r.model;
    if (r.status === "malformed") {
      detailFindings.push({
        provider: "openrouter", modelId, severity: "warning", code: "detail_malformed",
        message: `OpenRouter detail payload for ${modelId} was malformed (${r.note}); list-derived limits not auto-applied`,
        ref: `openrouter:endpoints:${modelId}`,
      });
    } else if (r.status === "ambiguous") {
      detailFindings.push({
        provider: "openrouter", modelId, severity: "warning", code: "detail_ambiguity",
        message: `OpenRouter endpoints disagree on limits for ${modelId}; affected limits not auto-applied`,
        ref: `openrouter:endpoints:${modelId}`,
      });
    }
  } catch {
    // Detail endpoint unavailable: demote list limits and emit a finding. No
    // detail digest is captured (nothing to revalidate at apply).
    refined = demoteListLimits(base, "openrouter:endpoints", "detail endpoint unavailable; list-derived limits not auto-applied");
    detailFindings.push({
      provider: "openrouter", modelId, severity: "warning", code: "detail_unavailable",
      message: `OpenRouter detail endpoint for ${modelId} unavailable; list-derived limits not auto-applied`,
      ref: `openrouter:endpoints:${modelId}`,
    });
  }
  const models = orSource.models.map((m) => (m.id === modelId ? refined : m));
  const sources = diagnosis.sources.map((s) =>
    s.id === "openrouter" ? { ...s, models } : s,
  );
  const refinedDiagnosis = {
    ...diagnosis,
    sources,
    findings: [...diagnosis.findings, ...detailFindings],
  };
  return {
    diagnosis: refinedDiagnosis,
    detail: digest !== undefined ? { url, digest } : undefined,
  };
}

/** Run `fn` over `items` with bounded concurrency (preserves input order of effects). */
async function mapBounded<T>(
  items: readonly T[],
  concurrency: number,
  fn: (item: T) => Promise<void>,
): Promise<void> {
  if (items.length === 0) return;
  let i = 0;
  const n = Math.min(concurrency, items.length);
  const workers = new Array(n).fill(0).map(async () => {
    while (i < items.length) {
      const idx = i++;
      await fn(items[idx]!);
    }
  });
  await Promise.all(workers);
}

/**
 * One-command OpenRouter detail refinement: for every candidate model
 * (existing source-backed entry OR source model missing from Pi's registry),
 * fetch + validate its /endpoints detail. List-derived limits are demoted
 * (never auto-applied) unless valid, agreeing endpoint evidence confirms them.
 * Per-model failures (unavailable/malformed/disagreeing) demote safely and emit
 * warnings; the other candidates continue. Returns a refined diagnosis.
 */
async function refineOpenRouterCandidates(
  diagnosis: Diagnosis,
  userModels: ModelsConfig,
  fetchFn: FetchFn,
  signal?: AbortSignal,
): Promise<{ readonly diagnosis: Diagnosis }> {
  const orSource = diagnosis.sources.find((s) => s.id === "openrouter");
  if (!orSource || orSource.status !== "ok") return { diagnosis };
  const existingEntry = userModels.providers.openrouter;
  const existing = existingEntry && Array.isArray(existingEntry.models) ? existingEntry.models : [];
  const effectiveIds = new Set((diagnosis.effective.providers.openrouter ?? []).map((m) => m.id));
  const candidateIds = new Set<string>();
  for (const e of existing) {
    if (orSource.models.some((m) => m.id === e.id)) candidateIds.add(e.id);
  }
  for (const m of orSource.models) {
    if (!effectiveIds.has(m.id) && !existing.some((e) => e.id === m.id)) candidateIds.add(m.id);
  }
  if (candidateIds.size === 0) return { diagnosis };

  const detailFindings: Finding[] = [];
  const refinedMap = new Map<string, NormalizedModel>();
  await mapBounded([...candidateIds], 8, async (id) => {
    const base = orSource.models.find((m) => m.id === id);
    if (!base) return;
    const url = openRouterDetailUrl(id);
    try {
      const fetched = await safeFetch(url, fetchFn, signal);
      const r = applyOpenRouterEndpoints(base, fetched.text);
      refinedMap.set(id, r.model);
      if (r.status === "malformed") {
        detailFindings.push({ provider: "openrouter", modelId: id, severity: "warning", code: "detail_malformed", message: `OpenRouter detail for ${id} malformed (${r.note}); list-derived limits not auto-applied`, ref: `openrouter:endpoints:${id}` });
      } else if (r.status === "ambiguous") {
        detailFindings.push({ provider: "openrouter", modelId: id, severity: "warning", code: "detail_ambiguity", message: `OpenRouter endpoints disagree for ${id}; affected limits not auto-applied`, ref: `openrouter:endpoints:${id}` });
      }
    } catch {
      refinedMap.set(id, demoteListLimits(base, "openrouter:endpoints", "detail endpoint unavailable; list-derived limits not auto-applied"));
      detailFindings.push({ provider: "openrouter", modelId: id, severity: "warning", code: "detail_unavailable", message: `OpenRouter detail endpoint for ${id} unavailable; list-derived limits not auto-applied`, ref: `openrouter:endpoints:${id}` });
    }
  });
  if (refinedMap.size === 0 && detailFindings.length === 0) return { diagnosis };
  const models = orSource.models.map((m) => (refinedMap.has(m.id) ? refinedMap.get(m.id)! : m));
  const sources = diagnosis.sources.map((s) => (s.id === "openrouter" ? { ...s, models } : s));
  return { diagnosis: { ...diagnosis, sources, findings: [...diagnosis.findings, ...detailFindings] } };
}

/**
 * Probe NEW zen/go catalog models for actual routability before they become ADD
 * candidates. The live catalogs are the availability authority, but a freshly
 * listed model is sometimes not served by the router yet (catalog lists it,
 * chat endpoint 404/500s — e.g. union-alpha on 2026-09-16). Unroutable models
 * are recorded in diagnosis.unroutable and skipped by reconcileAll with a
 * visible reason, so a broken listing never lands in models.json. Probes are
 * 1-token POSTs authorized from auth.json (read-only; the key never appears in
 * any output). No auth.json / no key / inconclusive probes change nothing.
 */
async function probeLiveCatalogCandidates(
  diagnosis: Diagnosis,
  userModels: ModelsConfig,
  opts: {
    readonly paths: DoctorPaths;
    readonly fetchFn: FetchFn;
    readonly signal?: AbortSignal;
  },
): Promise<{ readonly diagnosis: Diagnosis }> {
  const authPath = opts.paths.authPath;
  if (authPath === undefined) return { diagnosis };
  let auth: unknown;
  try {
    auth = parseConfig((await readRaw(authPath)).text);
  } catch {
    return { diagnosis }; // unreadable auth -> no probes, current behavior
  }
  const keyFor = (provider: string): string | undefined => {
    if (typeof auth !== "object" || auth === null) return undefined;
    const entry = (auth as Record<string, unknown>)[provider];
    if (typeof entry !== "object" || entry === null) return undefined;
    const key = (entry as Record<string, unknown>).key;
    return typeof key === "string" && key.length > 0 ? key : undefined;
  };

  const unroutable = new Set<string>(diagnosis.unroutable ?? []);
  const probeFindings: Finding[] = [];
  for (const provider of ["opencode", "opencode-go"] as const) {
    const source = sourceFor(diagnosis, provider);
    if (!source || source.status !== "ok") continue;
    const key = keyFor(provider);
    if (key === undefined) continue;
    const providerEntry = userModels.providers[provider];
    if (providerEntry === undefined) continue; // no block -> adds impossible anyway
    const existing = Array.isArray(providerEntry.models) ? providerEntry.models : [];
    const effectiveIds = new Set((diagnosis.effective.providers[provider] ?? []).map((m) => m.id));
    const candidates = source.models.filter(
      (m) => !effectiveIds.has(m.id) && !existing.some((e) => e.id === m.id),
    );
    for (const m of candidates) {
      const outcome = await probeOpenCodeModel(provider, m.id, key, opts.fetchFn, opts.signal);
      if (outcome.status === "unroutable") {
        unroutable.add(`${provider}/${m.id}`);
        probeFindings.push({
          provider,
          modelId: m.id,
          severity: "warning",
          code: "probe_unroutable",
          message: `${provider}/${m.id} is listed by the live catalog but not routable (${outcome.detail}); not added`,
          ref: `${provider}:probe:${m.id}`,
        });
      }
    }
  }
  if (unroutable.size === (diagnosis.unroutable?.size ?? 0) && probeFindings.length === 0) {
    return { diagnosis };
  }
  return { diagnosis: { ...diagnosis, unroutable, findings: [...diagnosis.findings, ...probeFindings] } };
}

// ─── Operation building ──────────────────────────────────────────────────────

/** Allowlisted model fields and where their evidence lives. */
const MODEL_FIELD_KEYS: ReadonlyArray<{ readonly path: MutationPath; readonly key: keyof NormalizedModel }> = [
  { path: "name", key: "name" },
  { path: "baseUrl", key: "baseUrl" },
  { path: "api", key: "api" },
  { path: "reasoning", key: "reasoning" },
  { path: "thinkingLevelMap", key: "thinkingLevelMap" },
  { path: "input", key: "input" },
  { path: "contextWindow", key: "contextWindow" },
  { path: "maxTokens", key: "maxTokens" },
  { path: "cost", key: "cost" },
  { path: "compat", key: "compat" },
];

/** pi fills a missing maxTokens with this tiny default; it is an absence, not
 *  a ceiling (a `high` reasoning budget alone is 16384). */
export const PI_DEFAULT_MAX_OUTPUT_TOKENS = 16384;

/** Why a catalog model must NOT be added: the source did not publish both
 *  limits as confirmed numbers. Guessing them (a flat fallback, or the context
 *  window reused as max output) writes a wrong entry that silently degrades the
 *  model, so an unknown-limits model is reported as skipped instead. */
export function missingLimitReason(nm: NormalizedModel): string | undefined {
  const known = (ev: Evidence<number> | undefined): boolean =>
    ev !== undefined && ev.plannable && typeof ev.value === "number" && Number.isFinite(ev.value) && ev.value > 0;
  if (!known(nm.contextWindow)) return "context window not published by source; not added";
  if (!known(nm.maxTokens)) return "max output tokens not published by source; not added";
  return undefined;
}

/** Build ops for a brand-new entry. Returns [] when the source carries no
 *  confirmed limits (see missingLimitReason): a new entry is never written with
 *  guessed context/max-output numbers. */
export function upsertNewModelOps(
  provider: string,
  nmInput: NormalizedModel,
  sourceLabel?: string,
): PatchOperation[] {
  if (missingLimitReason(nmInput) !== undefined) return [];
  const ops: PatchOperation[] = [];
  const src = sourceLabel ?? nmInput.name?.source ?? "official-source";
  for (const { path } of MODEL_FIELD_KEYS) {
    const ev = evidenceOf(nmInput, path);
    if (!ev || !ev.plannable) continue;
    const opSource = ev.source;
    ops.push({
      provider,
      modelId: nmInput.id,
      path,
      before: undefined,
      after: ev.value,
      source: opSource,
      reason: `${opSource}: add model ${nmInput.id}`,
    });
  }
  // A new entry must carry its id even if the source did not name it.
  if (!ops.some((o) => o.path === "name")) {
    ops.unshift({
      provider,
      modelId: nmInput.id,
      path: "name",
      before: undefined,
      after: nmInput.id,
      source: src,
      reason: `${src}: add model ${nmInput.id}`,
    });
  }
  return ops;
}

function evidenceOf(nm: NormalizedModel, path: MutationPath): Evidence<unknown> | undefined {
  const key = MODEL_FIELD_KEYS.find((f) => f.path === path)?.key;
  if (!key) return undefined;

  const value = nm[key] as Evidence<unknown> | undefined;

  if (!isObject(value) || !("plannable" in value) || !("value" in value)) return undefined;
  return value;
}

/**
 * Whether a field value differs between `before` (existing) and `after` (source).
 * For composite paths, only ALLOWLISTED subfields present in `after` are compared
 * (unknown existing members are preserved, never reported as a diff).
 */
function fieldDiffers(path: MutationPath, before: unknown, after: unknown): boolean {
  if (!isCompositePath(path)) return !canonicalEqual(before, after);
  const allowed = allowlistedSubfields(path);
  const b = before !== null && typeof before === "object" && !Array.isArray(before)
    ? (before as Record<string, unknown>) : {};
  const a = after !== null && typeof after === "object" && !Array.isArray(after)
    ? (after as Record<string, unknown>) : {};
  for (const k of allowed) {
    if (Object.prototype.hasOwnProperty.call(a, k) && !canonicalEqual(b[k], a[k])) return true;
  }
  return false;
}

/** Pure: build surgical operations for a target from a diagnosis + user config. */
export function buildOperations(
  diagnosis: Diagnosis,
  userModels: ModelsConfig,
  target: ParsedTarget,
): PatchOperation[] {
  const ops: PatchOperation[] = [];
  const providers = target.kind === "all"
    ? (["openrouter", "opencode", "opencode-go"] as ManagedProvider[])
    : ([target.provider] as ManagedProvider[]);

  for (const provider of providers) {
    // openrouter / opencode: reconcile user entries from the source. A
    // provider/model target is EXACT — only the requested model is reconciled
    // never unrelated existing source-backed models.
    const source = sourceFor(diagnosis, provider);
    if (!source || source.status !== "ok") continue;
    const providerEntry = userModels.providers[provider];
    const models = providerEntry?.models;
    if (!Array.isArray(models)) continue;

    for (const entry of models) {
      if (target.kind === "model" && target.modelId && entry.id !== target.modelId) continue;
      const nmRaw = source.models.find((m) => m.id === entry.id);
      if (!nmRaw) continue;
      ops.push(...opsForModel(provider, entry, enrichThinkingLevels(provider, nmRaw, diagnosis)));
    }

    // Targeted upsert: ensure the specific model exists.
    if (target.kind === "model" && target.modelId) {
      const exists = models.some((m) => m.id === target.modelId);
      if (!exists) {
        const nmRaw = source.models.find((m) => m.id === target.modelId);
        if (nmRaw) {
          ops.push(...upsertNewModelOps(provider, enrichThinkingLevels(provider, nmRaw, diagnosis)));
        }
      }
    }
  }
  return ops;
}

const SOURCE_BY_PROVIDER: Readonly<Record<ManagedProvider, OfficialSourceId>> = {
  openrouter: "openrouter",
  opencode: "opencode-zen",
  "opencode-go": "opencode-go",
};

function sourceFor(diagnosis: Diagnosis, provider: ManagedProvider): SourceResult | undefined {
  const id = SOURCE_BY_PROVIDER[provider];
  return diagnosis.sources.find((s) => s.id === id);
}


/** Build ops for each plannable field of a model that differs from the entry. */
function opsForModel(
  provider: ManagedProvider,
  entry: ModelEntry,
  nmInput: NormalizedModel,
  sourceLabel?: string,
): PatchOperation[] {
  const ops: PatchOperation[] = [];
  const src = sourceLabel ?? nmInput.name?.source ?? "official-source";
  for (const { path } of MODEL_FIELD_KEYS) {
    const ev = evidenceOf(nmInput, path);
    if (!ev || !ev.plannable) continue; // absent or ambiguous -> no mutation
    const before = (entry as Record<string, unknown>)[path];
    const after = ev.value;
    if (fieldDiffers(path, before, after)) {
      // Per-field provenance (e.g. openrouter:endpoints), not the model-level one.
      const opSource = ev.source;
      ops.push({
        provider,
        modelId: entry.id,
        path,
        before,
        after,
        source: opSource,
        reason: ev.note ? `${opSource}: ${ev.note}` : opSource,
      });
    }
  }
  return ops;
}

// ─── Full one-shot reconcile (single /model-doctor command) ──────────────────

/** Characteristics of a model, materialized for the result report. */
export interface ModelCharacteristics {
  name?: string;
  api?: string;
  baseUrl?: string;
  contextWindow?: number;
  maxTokens?: number;
  input?: readonly string[];
  reasoning?: boolean;
  thinkingLevelMap?: Record<string, string | null>;
  cost?: unknown;
  compat?: unknown;
  source?: string;
}

export type ModelAction = "added" | "updated" | "skipped";

/** One model's outcome in the reconcile report. */
export interface ModelResult {
  /** Managed special provider id or a generic provider id from models.json. */
  readonly provider: string;
  readonly modelId: string;
  readonly action: ModelAction;
  /** Present for added models (full written characteristics). */
  readonly characteristics?: ModelCharacteristics;
  /** Present for updated models (per-field changes). */
  readonly changes?: ReadonlyArray<{ readonly path: MutationPath; readonly before: unknown; readonly after: unknown; readonly source: string }>;
  /** Present for skipped models. */
  readonly reason?: string;
}

/** Per-provider rollup. Unchanged models appear ONLY here, never as results. */
export interface ProviderTotal {
  readonly provider: string;
  readonly sourceStatus: "ok" | "unavailable";
  readonly added: number;
  readonly updated: number;
  readonly skipped: number;
  readonly unchanged: number;
}

export interface ReconcileReport {
  readonly results: readonly ModelResult[];
  readonly totals: readonly ProviderTotal[];
  /** Provider-level source failures + detail issues (never hidden). */
  readonly warnings: readonly string[];
}

/** True if `nm` has any non-plannable evidence that differs from `entry`
 *  (drift we cannot safely apply -> the model is "skipped", not "unchanged"). */
function hasAmbiguousDrift(entry: ModelEntry, nm: NormalizedModel): boolean {
  for (const { path } of MODEL_FIELD_KEYS) {
    const ev = evidenceOf(nm, path);
    if (!ev) continue;
    const current = (entry as Record<string, unknown>)[path];
    if (!ev.plannable && !canonicalEqual(current, ev.value)) return true;
  }
  return false;
}

function characteristicsFromOps(ops: readonly PatchOperation[]): ModelCharacteristics {
  const c: ModelCharacteristics = {};
  let nameSource: string | undefined;
  for (const op of ops) {
    switch (op.path) {
      case "name": c.name = op.after as string; nameSource = op.source; break;
      case "api": c.api = op.after as string; break;
      case "contextWindow": c.contextWindow = op.after as number; break;
      case "maxTokens": c.maxTokens = op.after as number; break;
      case "input": c.input = op.after as readonly string[]; break;
      case "reasoning": c.reasoning = op.after as boolean; break;
      case "thinkingLevelMap": c.thinkingLevelMap = op.after as Record<string, string | null>; break;
      case "cost": c.cost = op.after; break;
      case "compat": c.compat = op.after; break;
    }
  }
  c.source = nameSource ?? ops[0]?.source;
  return c;
}

function makeAdded(provider: ManagedProvider, modelId: string, ops: readonly PatchOperation[]): ModelResult {
  return { provider, modelId, action: "added", characteristics: characteristicsFromOps(ops) };
}

function makeUpdated(provider: ManagedProvider, modelId: string, ops: readonly PatchOperation[]): ModelResult {
  return {
    provider, modelId, action: "updated",
    changes: ops.map((op) => ({ path: op.path, before: op.before, after: op.after, source: op.source })),
  };
}

/** Build characteristics from a model's Evidence (available metadata + provenance). */
function characteristicsFromModel(nm: NormalizedModel): ModelCharacteristics {
  const c: ModelCharacteristics = {};
  let firstSource: string | undefined;
  for (const { path } of MODEL_FIELD_KEYS) {
    const ev = evidenceOf(nm, path);
    if (!ev) continue;
    if (!firstSource) firstSource = ev.source;
    const v = ev.value;
    switch (path) {
      case "name": c.name = v as string; break;
      case "api": c.api = v as string; break;
      case "contextWindow": c.contextWindow = v as number; break;
      case "maxTokens": c.maxTokens = v as number; break;
      case "input": c.input = v as readonly string[]; break;
      case "reasoning": c.reasoning = v as boolean; break;
      case "thinkingLevelMap": c.thinkingLevelMap = v as Record<string, string | null>; break;
      case "cost": c.cost = v; break;
      case "compat": c.compat = v; break;
    }
  }
  c.source = firstSource ?? nm.name?.source;
  return c;
}

export function makeSkipped(
  provider: string,
  modelId: string,
  reason: string,
  nm?: NormalizedModel,
): ModelResult {
  // A skipped model still reports its AVAILABLE characteristics + provenance.
  const characteristics = nm ? characteristicsFromModel(nm) : undefined;
  return characteristics ? { provider, modelId, action: "skipped", reason, characteristics } : { provider, modelId, action: "skipped", reason };
}

/** OpenRouter models created before this date are never re-added; entries the
 *  user already keeps are still updated in place. */
export const OPENROUTER_RETIRE_CUTOFF_MS = Date.parse("2026-04-01T00:00:00Z");

/** Models without a parseable created timestamp are never treated as retired. */
export function openRouterRetired(nm: NormalizedModel): boolean {
  return (
    typeof nm.createdMs === "number" &&
    Number.isFinite(nm.createdMs) &&
    nm.createdMs < OPENROUTER_RETIRE_CUTOFF_MS
  );
}

/**
 * Reconcile the three managed providers in one pass: update drifted fields on
 * existing entries, add genuinely new catalog models (never duplicating pi's
 * built-ins, never with unpublished limits), and report every skip with its
 * reason. Pure: returns operations plus the report, writes nothing.
 */
export function reconcileAll(
  diagnosis: Diagnosis,
  userModels: ModelsConfig,
): { readonly operations: readonly PatchOperation[]; readonly report: ReconcileReport } {
  const operations: PatchOperation[] = [];
  const results: ModelResult[] = [];
  const totals: ProviderTotal[] = [];
  const warnings: string[] = [];

  for (const provider of ["openrouter", "opencode", "opencode-go"] as ManagedProvider[]) {
    const source = sourceFor(diagnosis, provider);
    const srcOk = !!source && source.status === "ok";
    const providerEntry = userModels.providers[provider];
    const existing = providerEntry && Array.isArray(providerEntry.models) ? providerEntry.models : [];
    const effectiveIds = new Set((diagnosis.effective.providers[provider] ?? []).map((m) => m.id));
    let added = 0, updated = 0, skipped = 0, unchanged = 0;

    // UPDATE existing managed entries from the source. Thinking levels are
    // resolved cross-provider (openrouter efforts > hyper; see sources.ts)
    // because the per-provider sources do not publish effort maps.
    for (const entry of existing) {
      const nmRaw = srcOk ? source!.models.find((m) => m.id === entry.id) : undefined;
      if (!nmRaw) { unchanged++; continue; }
      const nm = enrichThinkingLevels(provider, nmRaw, diagnosis);
      const ops = opsForModel(provider, entry, nm);
      if (ops.length > 0) {
        updated++; operations.push(...ops); results.push(makeUpdated(provider, entry.id, ops));
      } else if (hasAmbiguousDrift(entry, nm)) {
        skipped++; results.push(makeSkipped(provider, entry.id, "ambiguous/unsafe source fields; not auto-applied", nm));
      } else {
        unchanged++;
      }
    }

    // ADD official models missing from Pi's effective registry (narrow safe scope).
    // The provider must ALREADY have a block in models.json: creating a new
    // provider block is outside the allowlist proof (verifyDiffAllowlisted
    // rejects it), so a block-less provider gets a warning, never add ops.
    const addCandidates = srcOk
      ? source!.models.filter((nm) => {
          // OpenRouter retire-cutoff: never re-add pre-2026-04 models.
          if (provider === "openrouter" && openRouterRetired(nm)) return false;
          return !effectiveIds.has(nm.id) && !existing.some((e) => e.id === nm.id);
        })
      : [];
    if (providerEntry === undefined && addCandidates.length > 0) {
      warnings.push(
        `provider_block_absent: providers.${provider} has no models.json block; ${addCandidates.length} new source model(s) not added (create the provider block manually to manage them)`,
      );
    } else {
      for (const nmRaw of addCandidates) {
        if (diagnosis.unroutable?.has(`${provider}/${nmRaw.id}`) === true) {
          skipped++;
          results.push(makeSkipped(provider, nmRaw.id, "listed by live catalog but routing probe failed; not added", nmRaw));
          continue;
        }
        const nm = enrichThinkingLevels(provider, nmRaw, diagnosis);
        const gap = missingLimitReason(nm);
        if (gap !== undefined) {
          skipped++; results.push(makeSkipped(provider, nm.id, gap, nm));
          continue;
        }
        const ops = upsertNewModelOps(provider, nm);
        if (ops.length > 0) {
          added++; operations.push(...ops); results.push(makeAdded(provider, nm.id, ops));
        } else {
          skipped++; results.push(makeSkipped(provider, nm.id, "no safe plannable fields; not added", nm));
        }
      }
    }

    totals.push({ provider, sourceStatus: srcOk ? "ok" : "unavailable", added, updated, skipped, unchanged });
  }

  // opencode-go is now a real source-backed provider (reconciled in the loop
  // above); no separate hardcoded total is needed.

  // Surface provider-level source failures + detail issues as warnings.
  warnings.push(
    ...diagnosis.findings
      .filter((f) => f.code === "source_unavailable" || f.code === "detail_malformed" || f.code === "detail_ambiguity" || f.code === "detail_unavailable")
      .map((f) => `${f.code}: ${f.message}`),
  );

  return { operations, report: { results, totals, warnings } };
}

export interface ReconcileOptions extends DiagnoseOptions {
  readonly ttlMs?: number;
  readonly now?: () => number;
}

export interface ReconcileOutcome {
  readonly result: ApplyResult;
  readonly report: ReconcileReport;
}

/**
 * Orchestrate the full one-shot reconcile: diagnose -> read models.json once
 * (strict) -> reconcileAll -> assemble plan (hashes/digests/TTL) -> apply
 * atomically (backup/CAS/rollback). Returns the apply result and the report
 * (computed from the PRE-apply config). Throws on unreadable/malformed config.
 */
export async function runReconcile(opts: ReconcileOptions): Promise<ReconcileOutcome> {
  const now = opts.now ?? Date.now;
  let diagnosis = await diagnose(opts);

  let modelsRaw;
  try {
    modelsRaw = await readRaw(opts.paths.modelsPath);
  } catch (err) {
    throw new Error(`cannot read models.json: ${messageOf(err)}`);
  }
  const userModels = parseModelsConfigStrict(modelsRaw.text);

  // OpenRouter one-command flow: fetch/validate per-model endpoint details for
  // every candidate BEFORE limits become plannable. Malformed/unavailable/
  // disagreeing details demote (list-derived limits never auto-applied) and emit
  // warnings; other candidates continue. (Returns a refined diagnosis.)
  const detailRefine = await refineOpenRouterCandidates(diagnosis, userModels, opts.fetchFn, opts.signal);
  diagnosis = detailRefine.diagnosis;

  // Zen/Go one-command flow: probe NEW live-catalog models for routability
  // before they can be added (a fresh listing is not always deployed yet).
  const probeRefine = await probeLiveCatalogCandidates(diagnosis, userModels, opts);
  diagnosis = probeRefine.diagnosis;

  const managed = reconcileAll(diagnosis, userModels);
  let operations = [...managed.operations];
  let report: ReconcileReport = managed.report;

  // Generic providers (everything without dedicated rules): list live models
  // over the user's own transport config and add genuinely new ids.
  {
    let authText: string | undefined;
    try {
      if (opts.paths.authPath) authText = (await readRaw(opts.paths.authPath)).text;
    } catch {
      authText = undefined; // no auth.json -> only models.json api keys apply
    }
    const generic = await syncGenericProviders({
      userModels,
      authKeys: readAuthKeys(authText),
      activeProvider: (provider) => (opts.refCatalog.modelIds(provider) ?? []).length > 0,
      effectiveHasId: (provider, id) => (opts.effective.providers[provider] ?? []).some((m) => m.id === id),
      fetchFn: opts.fetchFn,
      signal: opts.signal,
    });
    operations = [...operations, ...generic.operations];
    report = {
      results: [...report.results, ...generic.results],
      totals: [...report.totals, ...generic.totals],
      warnings: [...report.warnings, ...generic.warnings],
    };
  }

  const settingsRaw = await readRaw(opts.paths.settingsPath);
  const sourceDigests: Record<string, string> = {};
  for (const s of diagnosis.sources) sourceDigests[s.id] = s.digest;
  const ttlMs = opts.ttlMs ?? DEFAULT_PLAN_TTL_MS;
  const plan: PendingPlan = {
    id: cryptoRandomId(),
    createdAt: nowIso(now),
    expiresAt: new Date(now() + ttlMs).toISOString(),
    target: "(all)",
    modelsHash: modelsRaw.hash,
    settingsHash: settingsRaw.hash,
    sourceDigests,
    effectiveDigest: canonicalEffectiveDigest(opts.effective),
    operations,
    findings: diagnosis.findings,
  };

  const result = await applyPlan({
    plan,
    paths: opts.paths,
    fetchFn: opts.fetchFn,
    effective: opts.effective,
    refCatalog: opts.refCatalog,
    now: opts.now,
    signal: opts.signal,
    skipSourceRevalidation: true,
  });
  return { result, report };
}

// ─── Apply ───────────────────────────────────────────────────────────────────

export async function applyPlan(opts: ApplyOptions): Promise<ApplyResult> {
  const now = opts.now ?? Date.now;
  const plan = opts.plan;
  const findings: Finding[] = [];

  // 1. TTL.
  const expires = Date.parse(plan.expiresAt);
  if (!Number.isFinite(expires) || now() > expires) {
    return fail(0, findings, `plan expired at ${plan.expiresAt}; run plan again`);
  }

  // 1b. Empty plan: nothing to apply. Do not create a backup or rewrite the
  // file (avoids a cosmetic trailing-newline reformat on a no-op apply).
  if (plan.operations.length === 0) {
    return { ok: true, applied: 0, backups: [], findings };
  }

  // 2-3. Re-read raw files and compare fingerprints.
  let modelsRaw: { bytes: Buffer; text: string; hash: string };
  let settingsRaw: { bytes: Buffer; text: string; hash: string };
  try {
    modelsRaw = await readRaw(opts.paths.modelsPath);
    settingsRaw = await readRaw(opts.paths.settingsPath);
  } catch (err) {
    return fail(0, findings, `cannot re-read config: ${messageOf(err)}`);
  }
  if (modelsRaw.hash !== plan.modelsHash) {
    return fail(0, findings, "models.json changed since plan; run plan again");
  }
  if (settingsRaw.hash !== plan.settingsHash) {
    return fail(0, findings, "settings.json changed since plan; run plan again");
  }

  // 4. Re-fetch source digests; any drift -> refuse. Skipped for the atomic
  // one-shot command (no plan->apply review gap; live sources may vary byte-for-
  // byte between two fetches). models/settings CAS + effective revalidation still run.
  if (!opts.skipSourceRevalidation) {
    const ids = Object.keys(plan.sourceDigests) as OfficialSourceId[];
    const refetched = await Promise.all(
      ids.map((id) => fetchSource(id, opts.fetchFn, opts.signal)),
    );
    for (const s of refetched) {
      const expected = plan.sourceDigests[s.id];
      if (s.digest !== expected) {
        return fail(0, findings, `official source ${s.id} changed since plan; run plan again`);
      }
    }
  }

  // 4b. Revalidate the effective catalog snapshot (Go-2 mirror source).
  if (canonicalEffectiveDigest(opts.effective) !== plan.effectiveDigest) {
    return fail(0, findings, "effective pi catalog changed since plan; run plan again");
  }

  // 4c. Revalidate the targeted OpenRouter detail payload, if one was used.
  if (plan.detail) {
    try {
      const refetchedDetail = await safeFetch(plan.detail.url, opts.fetchFn, opts.signal);
      if (refetchedDetail.digest !== plan.detail.digest) {
        return fail(0, findings, "OpenRouter detail payload changed since plan; run plan again");
      }
    } catch (err) {
      return fail(0, findings, `OpenRouter detail source unavailable at apply: ${messageOf(err)}; run plan again`);
    }
  }

  // 5. Apply operations to a deep clone of the CURRENT models.json. Strict schema:
  // a malformed models.json refuses the apply (fail closed).
  const original = parseModelsConfigStrict(modelsRaw.text);
  const config = asModelsConfig(deepClone(original));
  applyOperationsToClone(config, plan.operations);

  // 6. Deep-diff proof: the only allowed changes are to allowlisted model fields.
  const proof = verifyDiffAllowlisted(original, config);
  if (!proof.ok) {
    findings.push({
      provider: "*",
      severity: "error",
      code: "allowlist_violation",
      message: `apply would touch non-allowlisted paths: ${proof.violations.join("; ")}`,
    });
    return fail(0, findings, "allowlist verification failed; no changes written");
  }

  // 6b. PRE-WRITE schema gate: validate the FINAL models.json against Pi's model
  // schema (cost four-keys, numeric limits, boolean reasoning, no null metadata).
  // Refuse to write anything invalid — never regress Pi startup.
  const schemaErrors = validateModelsConfig(config);
  if (schemaErrors.length > 0) {
    findings.push({
      provider: "*",
      severity: "error",
      code: "schema_violation",
      message: `apply would write schema-invalid models.json: ${schemaErrors.slice(0, 5).join("; ")}${schemaErrors.length > 5 ? ` (+${schemaErrors.length - 5} more)` : ""}`,
    });
    return fail(0, findings, `schema validation failed; no changes written (${schemaErrors[0]})`);
  }

  // 7-10. Backup (0600), atomic write WITH compare-and-swap on the plan's
  // modelsHash, then verify. CAS aborts (without clobbering) if models.json
  // changed between the start-of-apply check and the rename. The post-write
  // readback re-runs the schema gate and rolls back on any violation.
  const nextBytes = Buffer.from(serializeConfig(config), "utf8");
  let backups: readonly string[];
  try {
    const res = await safeWriteWithBackup(
      opts.paths.modelsPath,
      nextBytes,
      opts.paths.backupDir,
      now,
      plan.modelsHash,
      undefined,
      validateModelsConfigText,
    );
    backups = res.backups;
  } catch (err) {
    return fail(0, findings, messageOf(err));
  }

  return { ok: true, applied: plan.operations.length, backups, findings };
}

function fail(applied: number, findings: Finding[], error: string): ApplyResult {
  return { ok: false, applied, backups: [], error, findings };
}

/** Mutate a clone in place: set allowlisted fields, upsert new model entries.
 *  Composite paths (cost/compat/thinkingLevelMap) are MERGED: only allowlisted
 *  subfields from `op.after` are set, every other existing member is preserved. */
export function applyOperationsToClone(
  config: ModelsConfig,
  operations: readonly PatchOperation[],
): void {
  for (const op of operations) {
    const providerEntry = ensureProvider(config, op.provider);
    const models = ensureModelsArray(providerEntry);
    let entry = models.find((m) => m.id === op.modelId);
    if (!entry) {
      entry = { id: op.modelId } as ModelEntry;
      models.push(entry);
    }
    if (op.after === undefined) continue;
    if (op.after === null) {
      delete (entry as Record<string, unknown>)[op.path];
    } else if (isCompositePath(op.path)) {
      mergeComposite(entry as Record<string, unknown>, op.path, op.after);
    } else {
      (entry as Record<string, unknown>)[op.path] = deepClone(op.after);
    }
  }
}

/** Merge only allowlisted subfields of a composite value, preserving unknown members.
 *  For `cost`, the result is MATERIALIZED with all four Pi-required rate keys
 *  (input/output/cacheRead/cacheWrite); any rate missing from both the existing
 *  value and the source defaults to 0. Pi's ModelCost schema requires all four. */
function mergeComposite(
  entry: Record<string, unknown>,
  path: MutationPath,
  after: unknown,
): void {
  const allowed = allowlistedSubfields(path);
  const cur = entry[path];
  const base: Record<string, unknown> =
    cur !== null && typeof cur === "object" && !Array.isArray(cur)
      ? { ...(cur as Record<string, unknown>) }
      : {};
  if (after !== null && typeof after === "object" && !Array.isArray(after)) {
    for (const k of allowed) {
      if (Object.prototype.hasOwnProperty.call(after, k)) {
        base[k] = deepClone((after as Record<string, unknown>)[k]);
      }
    }
  }
  if (path === "cost") materializeCostRates(base);
  entry[path] = base;
}

/** Pi ModelCost requires input/output/cacheRead/cacheWrite (all numbers). Ensure
 *  all four are present, defaulting any missing rate to 0 (matches the user-file
 *  convention). `tiers` and any unknown members are preserved untouched. */
const COST_RATE_KEYS = ["input", "output", "cacheRead", "cacheWrite"] as const;
function materializeCostRates(cost: Record<string, unknown>): void {
  for (const k of COST_RATE_KEYS) {
    if (typeof cost[k] !== "number" || !Number.isFinite(cost[k] as number)) cost[k] = 0;
  }
}

function ensureProvider(config: ModelsConfig, provider: string): ProviderEntry {
  let entry = config.providers[provider];
  if (!entry) {
    entry = {};
    config.providers[provider] = entry;
  }
  return entry;
}

function ensureModelsArray(providerEntry: ProviderEntry): ModelEntry[] {
  if (!Array.isArray(providerEntry.models)) {
    providerEntry.models = [];
  }
  return providerEntry.models!;
}

export interface DiffProof {
  readonly ok: boolean;
  readonly violations: readonly string[];
}

/**
 * Prove the diff only touches allowlisted model fields. The doctor edits ONLY
 * model entries, so every provider-level field (anything except `models`) must
 * be byte-for-byte unchanged, new provider blocks are rejected, model entries
 * may only change/add allowlisted fields, and no model may be deleted.
 */
export function verifyDiffAllowlisted(
  original: ModelsConfig,
  next: ModelsConfig,
): DiffProof {
  const violations: string[] = [];
  const allProviders = new Set([
    ...Object.keys(original.providers),
    ...Object.keys(next.providers),
  ]);

  for (const provider of allProviders) {
    const o = original.providers[provider];
    const n = next.providers[provider];
    if (!o && n) {
      violations.push(`providers.${provider} would be created (new provider block)`);
      continue;
    }
    if (o && n) {
      // Every provider-level field except `models` must be unchanged.
      const oRest = withoutModels(o);
      const nRest = withoutModels(n);
      if (!canonicalEqual(oRest, nRest)) {
        const keys = new Set([...Object.keys(oRest), ...Object.keys(nRest)]);
        for (const k of keys) {
          if (!canonicalEqual((oRest as Record<string, unknown>)[k], (nRest as Record<string, unknown>)[k])) {
            violations.push(`providers.${provider}.${k} changed (provider-level field must be unchanged)`);
          }
        }
      }
    }
    const oModels = indexById((o?.models) ?? []);
    const nModels = (n?.models) ?? [];
    // Managed providers may edit model fields in place. Generic providers
    // (any other id) may ONLY receive appended new model entries; existing
    // entries must stay byte-identical.
    const managed = isManagedProvider(provider);
    if (!managed && o && n) {
      for (const [id, oe] of oModels) {
        const ne = nModels.find((m) => m.id === id);
        if (!canonicalEqual(oe, ne)) {
          violations.push(`providers.${provider}.models[${id}] changed (generic provider entries are never modified)`);
        }
      }
    }
    // Reject duplicate model ids within a single models[] array.
    const seen = new Set<string>();
    for (const entry of nModels) {
      if (typeof entry.id === "string") {
        if (seen.has(entry.id)) {
          violations.push(`providers.${provider}.models[${entry.id}] duplicate id`);
        }
        seen.add(entry.id);
      }
    }
    // Model-array order: pre-existing ids must keep their relative order
    // (model-doctor only edits fields in place and appends new entries).
    const oOrder = (o?.models ?? [])
      .map((m) => m.id)
      .filter((id): id is string => typeof id === "string");
    const nOrder = nModels
      .map((m) => m.id)
      .filter((id): id is string => typeof id === "string");
    const nIdSet = new Set(nOrder);
    const preservedOrder = oOrder.filter((id) => nIdSet.has(id));
    {
      let pi = 0;
      for (const id of preservedOrder) {
        while (pi < nOrder.length && nOrder[pi] !== id) pi++;
        if (pi >= nOrder.length) {
          violations.push(`providers.${provider}.models order changed (${id} moved)`);
          break;
        }
        pi++;
      }
    }
    // New-entry field allowlist applies to managed AND generic providers.
    {
      for (const entry of nModels) {
        const oe = oModels.get(entry.id);
        if (!oe) {
          // New model entry: every field except id must be allowlisted.
          for (const key of Object.keys(entry)) {
            if (key !== "id" && !isMutationPath(key)) {
              violations.push(`providers.${provider}.models[${entry.id}].${key} added (non-allowlisted)`);
            }
          }
          continue;
        }
        for (const key of Object.keys(entry)) {
          if (key === "id") continue;
          if (!canonicalEqual((oe as Record<string, unknown>)[key], (entry as Record<string, unknown>)[key])) {
            if (!isMutationPath(key)) {
              violations.push(`providers.${provider}.models[${entry.id}].${key} changed (non-allowlisted)`);
            }
          }
        }
        // Detect deletion of non-allowlisted fields: present before, absent after.
        for (const key of Object.keys(oe)) {
          if (key === "id") continue;
          if (
            (entry as Record<string, unknown>)[key] === undefined &&
            (oe as Record<string, unknown>)[key] !== undefined &&
            !isMutationPath(key)
          ) {
            violations.push(`providers.${provider}.models[${entry.id}].${key} removed (non-allowlisted)`);
          }
        }
        // Deep composite proof: for cost/compat/thinkingLevelMap, only ALLOWLISTED
        // subfields may change; every unknown member (e.g. vendorSecret) must be
        // preserved byte-for-byte (never replaced wholesale).
        for (const cpath of ["cost", "compat", "thinkingLevelMap"] as const) {
          const ov = (oe as Record<string, unknown>)[cpath];
          const nv = (entry as Record<string, unknown>)[cpath];
          const oObj = ov !== null && typeof ov === "object" && !Array.isArray(ov) ? (ov as Record<string, unknown>) : {};
          const nObj = nv !== null && typeof nv === "object" && !Array.isArray(nv) ? (nv as Record<string, unknown>) : {};
          const allowed = new Set(allowlistedSubfields(cpath));
          for (const sub of new Set([...Object.keys(oObj), ...Object.keys(nObj)])) {
            if (allowed.has(sub)) continue; // allowlisted subfield may change
            if (!canonicalEqual(oObj[sub], nObj[sub])) {
              violations.push(`providers.${provider}.models[${entry.id}].${cpath}.${sub} changed (non-allowlisted composite subfield)`);
            }
          }
        }
      }
    }
    // Model deletions are never allowed.
    for (const id of oModels.keys()) {
      if (!nModels.some((m) => m.id === id)) {
        violations.push(`providers.${provider}.models[${id}] would be deleted`);
      }
    }
  }
  return { ok: violations.length === 0, violations };
}

function withoutModels(entry: ProviderEntry): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(entry)) {
    if (k !== "models") out[k] = v;
  }
  return out;
}

// ─── helpers ─────────────────────────────────────────────────────────────────

function asModelsConfig(v: unknown): ModelsConfig {
  if (isObject(v) && isObject((v as { providers?: unknown }).providers)) {
    return v as unknown as ModelsConfig;
  }
  return { providers: {} };
}

/** Strict models.json parse: throw on malformed input instead of treating it as
 *  empty. `models[]` entries must be objects with string ids. */
export function parseModelsConfigStrict(text: string): ModelsConfig {
  const v = parseConfig(text);
  if (!isObject(v) || !isObject((v as { providers?: unknown }).providers)) {
    throw new Error("models.json is not a { providers: {...} } object");
  }
  const providers = (v as { providers: Record<string, unknown> }).providers;
  for (const [name, pe] of Object.entries(providers)) {
    if (!isObject(pe)) {
      throw new Error(`models.json providers.${name} is not an object`);
    }
    if ("models" in pe) {
      const models = (pe as { models?: unknown }).models;
      if (!Array.isArray(models)) {
        throw new Error(`models.json providers.${name}.models is not an array`);
      }
      const seenIds = new Set<string>();
      for (let i = 0; i < models.length; i++) {
        const m = models[i];
        if (!isObject(m) || typeof (m as { id?: unknown }).id !== "string") {
          throw new Error(`models.json providers.${name}.models[${i}] is not an object with a string id`);
        }
        const id = ((m as { id: unknown }).id as string).trim();
        if (id.length === 0) {
          throw new Error(`models.json providers.${name}.models[${i}] has an empty/whitespace id`);
        }
        const rawId = (m as { id: string }).id;
        if (seenIds.has(rawId)) {
          throw new Error(`models.json providers.${name}.models[${i}] has a duplicate id '${rawId}'`);
        }
        seenIds.add(rawId);
      }
    }
    if ("modelOverrides" in pe) {
      const mo = (pe as { modelOverrides?: unknown }).modelOverrides;
      if (!isObject(mo)) {
        throw new Error(`models.json providers.${name}.modelOverrides is not an object`);
      }
    }
  }
  return v as unknown as ModelsConfig;
}

/**
 * Pre-write schema gate: validates the final models.json against the pi model
 * schema (pi exports no runtime validator). Returns human-readable violations;
 * any violation refuses the write. Optional fields are checked only when
 * present: cost rates are finite numbers (tiers too), limits are finite and
 * non-negative, input is "text"/"image", and no metadata field is null.
 */
export function validateModelsConfig(config: ModelsConfig): string[] {
  const errors: string[] = [];
  const isFiniteNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
  const providers = (config && (config as { providers?: Record<string, unknown> }).providers) || {};
  for (const [provName, pe] of Object.entries(providers)) {
    if (!pe || typeof pe !== "object") continue;
    const models = (pe as { models?: unknown }).models;
    if (!Array.isArray(models)) continue;
    for (let i = 0; i < models.length; i++) {
      const m = models[i] as Record<string, unknown>;
      if (!m || typeof m !== "object") continue;
      const where = `providers.${provName}.models[${m.id ?? i}]`;
      const cost = m.cost;
      if (cost !== undefined) {
        if (cost === null || typeof cost !== "object" || Array.isArray(cost)) {
          errors.push(`${where}.cost is not an object`);
        } else {
          const c = cost as Record<string, unknown>;
          for (const k of COST_RATE_KEYS) {
            if (!isFiniteNum(c[k])) errors.push(`${where}.cost.${k} missing or not a finite number`);
          }
          const tiers = c.tiers;
          if (tiers !== undefined) {
            if (!Array.isArray(tiers)) {
              errors.push(`${where}.cost.tiers is not an array`);
            } else {
              tiers.forEach((t, ti) => {
                if (!t || typeof t !== "object" || Array.isArray(t)) {
                  errors.push(`${where}.cost.tiers[${ti}] is not an object`);
                } else {
                  for (const k of COST_RATE_KEYS) {
                    if (!isFiniteNum((t as Record<string, unknown>)[k]))
                      errors.push(`${where}.cost.tiers[${ti}].${k} missing or not a finite number`);
                  }
                }
              });
            }
          }
        }
      }
      if (m.contextWindow !== undefined && (!isFiniteNum(m.contextWindow) || (m.contextWindow as number) < 0))
        errors.push(`${where}.contextWindow is not a finite non-negative number`);
      if (m.maxTokens !== undefined && (!isFiniteNum(m.maxTokens) || (m.maxTokens as number) < 0))
        errors.push(`${where}.maxTokens is not a finite non-negative number`);
      if (m.reasoning !== undefined && (m.reasoning === null || typeof m.reasoning !== "boolean"))
        errors.push(`${where}.reasoning is not a boolean`);
      if (m.input !== undefined) {
        if (!Array.isArray(m.input) || !m.input.every((x) => x === "text" || x === "image"))
          errors.push(`${where}.input is not an array of text/image`);
      }
      if (m.name !== undefined && (m.name === null || typeof m.name !== "string"))
        errors.push(`${where}.name is not a string`);
      if (m.api !== undefined && (m.api === null || typeof m.api !== "string"))
        errors.push(`${where}.api is not a string`);
      if (m.thinkingLevelMap !== undefined && (m.thinkingLevelMap === null || typeof m.thinkingLevelMap !== "object" || Array.isArray(m.thinkingLevelMap)))
        errors.push(`${where}.thinkingLevelMap is not an object`);
    }
  }
  return errors;
}

/** Parse + validate models.json text; returns the first violation or undefined. */
export function validateModelsConfigText(text: string): string | undefined {
  let parsed: unknown;
  try {
    parsed = parseConfig(text);
  } catch (err) {
    return `models.json is not valid JSON: ${messageOf(err)}`;
  }
  const config = asModelsConfig(parsed);
  const errors = validateModelsConfig(config);
  return errors[0];
}

/** Canonical sha256 of the effective catalog snapshot (order-insensitive). */
export function canonicalEffectiveDigest(effective: EffectiveSnapshot): string {
  return sha256Hex(canonicalJson(effective));
}

function indexById(models: readonly ModelEntry[]): Map<string, ModelEntry> {
  const m = new Map<string, ModelEntry>();
  for (const e of models) if (typeof e.id === "string") m.set(e.id, e);
  return m;
}

/** Order-insensitive deep equality for plain JSON values. */
function canonicalEqual(a: unknown, b: unknown): boolean {
  return canonicalJson(a) === canonicalJson(b);
}

function canonicalJson(v: unknown): string {
  if (v === undefined) return "undefined";
  return JSON.stringify(sortKeys(v));
}

function sortKeys(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(sortKeys);
  if (isObject(v)) {
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(v).sort()) out[k] = sortKeys((v as Record<string, unknown>)[k]);
    return out;
  }
  return v;
}

function describeTarget(t: ParsedTarget): string {
  if (t.kind === "model" && t.provider && t.modelId) return `${t.provider}/${t.modelId}`;
  if (t.kind === "provider" && t.provider) return t.provider;
  return "";
}

function cryptoRandomId(): string {
  // Stable, dependency-free id (crypto is a node: builtin).
  return sha256Hex(`${nowIso()}:${Math.random()}`).slice(0, 16);
}

/** Re-exported for tests. */
export { sha256Buffer };
