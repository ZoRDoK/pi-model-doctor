import { describe, test } from "node:test";
import {
  deepEqual,
  notStrictEqual,
  ok as assertOk,
  rejects,
  strictEqual,
  throws,
} from "node:assert/strict";
import { readFile, stat, writeFile, mkdir } from "node:fs/promises";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const SRC_DIR = dirname(dirname(fileURLToPath(import.meta.url)));

import {
  applyOperationsToClone,
  applyPlan,
  auditUserModels,
  buildOperations,
  buildPlanFromSources,
  canonicalEffectiveDigest,
  diagnose,
  diagnoseManagedDrift,
  globMatch,
  parseModelsConfigStrict,
  reconcileAll,
  runReconcile,
  validateReferences,
  verifyDiffAllowlisted,
  validateModelsConfig,
  validateModelsConfigText,
  type ParsedTarget,
  type RefCatalog,
} from "../doctor.ts";
import { fetchSource, nowIso, normalizeHyperList, normalizeModelsDev, normalizeOpenRouterList } from "../sources.ts";
import { readRaw, serializeConfig } from "../storage.ts";
import type {
  Diagnosis,
  DoctorPaths,
  EffectiveSnapshot,
  FetchFn,
  FetchInit,
  Finding,
  ManagedProvider,
  ModelsConfig,
  MutationPath,
  NormalizedCost,
  NormalizedModel,
  SourceResult,
} from "../types.ts";
import factory, { __internals, sanitize } from "../index.ts";
import {
  effectiveGoFixture,
  findSecrets,
  goFixture,
  makeTempDir,
  mockFetch,
  modelsDevFixture,
  openRouterListFixture,
  openRouterEndpointsFixture,
  settingsFixture,
  userModelsFixture,
  writeJson,
} from "./helpers.ts";


// ─── fixture builders ────────────────────────────────────────────────────────

function ev<T>(value: T, source = "pi:effective", plannable = true) {
  return { value, source, confidence: "authoritative" as const, plannable };
}

/** Build an EffectiveSnapshot from raw model objects (one list per provider). */
function effectiveFromRaw(
  raw: Record<string, Array<Record<string, unknown>>>,
): EffectiveSnapshot {
  const providers = {} as Record<ManagedProvider, NormalizedModel[]>;
  for (const [provider, list] of Object.entries(raw)) {
    providers[provider as ManagedProvider] = list.map((m) => {
      const out: NormalizedModel = { provider: provider as ManagedProvider, id: m.id as string };
      if (typeof m.name === "string") out.name = ev(m.name);
      if (typeof m.api === "string") out.api = ev(m.api);
      if (typeof m.reasoning === "boolean") out.reasoning = ev(m.reasoning);
      if (m.thinkingLevelMap) out.thinkingLevelMap = ev(m.thinkingLevelMap as Record<string, string | null>);
      if (Array.isArray(m.input)) out.input = ev(m.input as ("text" | "image")[]);
      if (typeof m.cost === "object" && m.cost) out.cost = ev(m.cost as NormalizedCost);
      if (typeof m.contextWindow === "number") out.contextWindow = ev(m.contextWindow);
      if (typeof m.maxTokens === "number") out.maxTokens = ev(m.maxTokens);
      if (m.compat) out.compat = ev(m.compat as Record<string, unknown>);
      return out;
    });
  }
  return { providers };
}

function makeDiagnosis(sources: SourceResult[], effective: EffectiveSnapshot): Diagnosis {
  return { sources, findings: [], effective };
}

function openRouterSource(): SourceResult {
  return {
    id: "openrouter",
    status: "ok",
    digest: "d-or",
    models: normalizeOpenRouterList(JSON.parse(openRouterListFixture())),
    fetchedAt: nowIso(),
  };
}

function zenSource(): SourceResult {
  return {
    id: "opencode-zen",
    status: "ok",
    digest: "d-zen",
    models: normalizeModelsDev(JSON.parse(modelsDevFixture())),
    fetchedAt: nowIso(),
  };
}

const refCatalog: RefCatalog = {
  modelIds(p: string): readonly string[] {
    const map: Record<string, string[]> = {
      "openai-codex": ["gpt-5.6-sol", "gpt-5.6-luna"],
      "opencode-go": ["glm-5.2"],
      zai: ["glm-5.2"],
      opencode: ["glm-5.2"],
      anthropic: ["claude-sonnet-5"],
    };
    return map[p] ?? [];
  },
  hasAuth(_p: string, _id: string): boolean {
    return true;
  },
};

// ─── buildOperations ─────────────────────────────────────────────────────────

describe("buildOperations no-target", () => {
  const diagnosis = makeDiagnosis(
    [openRouterSource(), zenSource()],
    effectiveFromRaw(effectiveGoFixture() as Record<string, Array<Record<string, unknown>>>),
  );

  test("does not add the whole catalog", () => {
    const user = userModelsFixture() as ModelsConfig;
    const ops = buildOperations(diagnosis, user, { kind: "all" });
    const ids = new Set(ops.map((o) => `${o.provider}/${o.modelId}`));
    // Only existing user model ids may appear (openai/gpt-test, glm-5.2 opencode).
    assertOk(ids.has("openrouter/openai/gpt-test"));
    assertOk(ids.has("opencode/glm-5.2"));
    // Catalog-only models are NOT added.
    strictEqual(ids.has("openrouter/deepseek/ds-alias"), false);
    strictEqual(ids.has("openrouter/free/zero-cost"), false);
  });

  test("updates existing openrouter entry fields, no delete ops", () => {
    const user = userModelsFixture() as ModelsConfig;
    const ops = buildOperations(diagnosis, user, { kind: "all" });
    const gptOps = ops.filter((o) => o.modelId === "openai/gpt-test");
    assertOk(gptOps.length > 0, "expected at least one patch for gpt-test");
    const paths = new Set(gptOps.map((o) => o.path));
    assertOk(paths.has("contextWindow"));
    assertOk(paths.has("maxTokens"));
    assertOk(paths.has("reasoning"));
    // No delete operation is ever produced.
    assertOk(ops.every((o) => o.after !== undefined), "no delete ops (after undefined)");
  });

  test("preserves unrelated provider, modelOverrides, and unknown properties", () => {
    const user = userModelsFixture() as ModelsConfig;
    const clone = structuredClone(user);
    applyOperationsToClone(clone, buildOperations(diagnosis, user, { kind: "all" }));
    // Unrelated provider untouched.
    assertOk(clone.providers.unrelated);
    strictEqual(clone.providers.unrelated!.models![0]!.id, "keepme");
    // openrouter modelOverrides preserved.
    assertOk(clone.providers.openrouter!.modelOverrides?.["openai/gpt-test"]);
    // Provider-level apiKey preserved.
    strictEqual(clone.providers.openrouter!.apiKey, "$OPENROUTER_API_KEY");
  });
});

describe("buildOperations targeted upsert", () => {
  const diagnosis = makeDiagnosis(
    [openRouterSource()],
    effectiveFromRaw(effectiveGoFixture() as Record<string, Array<Record<string, unknown>>>),
  );

  test("adds a brand-new openrouter model via provider/model target", () => {
    const user = userModelsFixture() as ModelsConfig;
    const target: ParsedTarget = { kind: "model", provider: "openrouter", modelId: "deepseek/ds-alias" };
    const ops = buildOperations(diagnosis, user, target);
    const upsert = ops.filter((o) => o.modelId === "deepseek/ds-alias");
    assertOk(upsert.length > 0, "targeted upsert should describe the new model");
    assertOk(upsert.some((o) => o.path === "name"));
    // Applying creates the entry.
    const clone = structuredClone(user);
    applyOperationsToClone(clone, ops);
    const created = clone.providers.openrouter!.models!.find((m) => m.id === "deepseek/ds-alias");
    assertOk(created, "new entry created");
  });
});

// ─── validateReferences ──────────────────────────────────────────────────────

describe("validateReferences", () => {
  test("resolves literal id before splitting thinking suffix", () => {
    const findings = validateReferences(settingsFixture(), refCatalog);
    const codes = findings.map((f) => f.code);
    // gpt-5.6-sol (defaultModel) resolves; not flagged missing.
    const missingDefault = findings.find(
      (f) => f.code === "missing_reference" && f.ref === "settings.defaultProvider/defaultModel",
    );
    strictEqual(missingDefault, undefined, "default model resolves despite separate provider/model");
    // workingVibeModel opencode/glm-5.2:max resolves (thinking stripped).
    const wvm = findings.find((f) => f.ref === "settings.workingVibeModel");
    strictEqual(wvm, undefined, "working vibe model resolves after thinking-suffix strip");
  });

  test("flags missing reference but not known ones", () => {
    const findings = validateReferences(settingsFixture(), refCatalog);
    const ghost = findings.find((f) => f.code === "missing_reference" && f.modelId === undefined && f.message.includes("ghost-model"));
    assertOk(ghost, "missing/ghost-model flagged");
    // zai/glm-5.2 is known -> not missing.
    const zaiMissing = findings.find(
      (f) => f.code === "missing_reference" && f.provider === "zai",
    );
    strictEqual(zaiMissing, undefined);
  });

  test("glob patterns match via local matcher", () => {
    const findings = validateReferences(settingsFixture(), refCatalog);
    // enabledModels has openai-codex/gpt-* which matches gpt-5.6-sol -> not missing.
    const globMissing = findings.find(
      (f) => f.code === "missing_reference" && f.message.includes("gpt-*"),
    );
    strictEqual(globMissing, undefined, "glob should match");
  });

  test("subagent model + fallbackModels validated", () => {
    const findings = validateReferences(settingsFixture(), refCatalog);
    const plannerModel = findings.find((f) => f.ref?.includes("planner.model"));
    strictEqual(plannerModel, undefined, "planner.model resolves");
    // claude-sonnet-5 (no thinking) is known.
    const fbMissing = findings.find(
      (f) => f.ref?.includes("fallbackModels") && f.code === "missing_reference",
    );
    strictEqual(fbMissing, undefined);
  });
});

describe("globMatch", () => {
  test("supports * and ?", () => {
    strictEqual(globMatch("gpt-*", "gpt-5.6-sol"), true);
    strictEqual(globMatch("gpt-?.6-*", "gpt-5.6-sol"), true);
    strictEqual(globMatch("gpt-*", "claude-x"), false);
  });
});

// ─── verifyDiffAllowlisted ───────────────────────────────────────────────────

describe("verifyDiffAllowlisted", () => {
  const base: ModelsConfig = {
    providers: {
      openrouter: { apiKey: "$X", models: [{ id: "m1", name: "old", reasoning: false, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 1, maxTokens: 1, input: ["text"], headers: { X: "y" } }] },
    },
  };

  test("allows allowlisted field change", () => {
    const next = structuredClone(base);
    next.providers.openrouter!.models![0]!.name = "new";
    strictEqual(verifyDiffAllowlisted(base, next).ok, true);
  });

  test("rejects provider-level apiKey change", () => {
    const next = structuredClone(base);
    next.providers.openrouter!.apiKey = "sk-changed";
    const proof = verifyDiffAllowlisted(base, next);
    strictEqual(proof.ok, false);
    assertOk(proof.violations.some((v) => v.includes("apiKey")));
  });

  test("rejects non-allowlisted model field change (headers)", () => {
    const next = structuredClone(base);
    next.providers.openrouter!.models![0]!.headers = { X: "z" };
    const proof = verifyDiffAllowlisted(base, next);
    strictEqual(proof.ok, false);
    assertOk(proof.violations.some((v) => v.includes("headers")));
  });

  test("rejects model deletion", () => {
    const next = structuredClone(base);
    next.providers.openrouter!.models = [];
    const proof = verifyDiffAllowlisted(base, next);
    strictEqual(proof.ok, false);
    assertOk(proof.violations.some((v) => v.includes("deleted")));
  });

  test("rejects new provider block creation", () => {
    const next = structuredClone(base);
    next.providers.sneaky = { apiKey: "x", models: [] };
    const proof = verifyDiffAllowlisted(base, next);
    strictEqual(proof.ok, false);
    assertOk(proof.violations.some((v) => v.includes("created")));
  });

  test("allows new model entry with only allowlisted fields", () => {
    const next = structuredClone(base);
    next.providers.openrouter!.models!.push({ id: "m2", name: "New", reasoning: true, cost: { input: 1, output: 2, cacheRead: 0, cacheWrite: 0 }, contextWindow: 100, maxTokens: 10, input: ["text"] });
    strictEqual(verifyDiffAllowlisted(base, next).ok, true);
  });
});

// ─── auditUserModels ─────────────────────────────────────────────────────────

describe("auditUserModels", () => {
  test("reports literal apiKey without leaking the value", () => {
    const user = userModelsFixture({ withSecret: true }) as ModelsConfig;
    const eff = effectiveFromRaw(effectiveGoFixture() as Record<string, Array<Record<string, unknown>>>);
    const findings = auditUserModels(user, eff);
    const lit = findings.find((f) => f.code === "literal_apikey" && f.provider === "openrouter");
    assertOk(lit, "literal apiKey flagged");
    strictEqual(findSecrets(JSON.stringify(findings)).length, 0, "no secret value in findings");
  });

  test("env-style apiKey ($VAR) is not treated as literal", () => {
    const user = userModelsFixture() as ModelsConfig;
    const eff = effectiveFromRaw(effectiveGoFixture() as Record<string, Array<Record<string, unknown>>>);
    const findings = auditUserModels(user, eff);
    strictEqual(findings.find((f) => f.code === "literal_apikey" && f.provider === "opencode"), undefined);
  });
});

// ─── applyPlan end-to-end (temp files + mocked fetch) ────────────────────────

async function setupPlanEnv() {
  const dir = await makeTempDir();
  const paths: DoctorPaths = {
    modelsPath: join(dir, "models.json"),
    settingsPath: join(dir, "settings.json"),
    backupDir: dir,
  };
  await writeJson(paths.modelsPath, userModelsFixture());
  await writeJson(paths.settingsPath, settingsFixture());
  const effective = effectiveFromRaw(effectiveGoFixture() as Record<string, Array<Record<string, unknown>>>);
  const { fetch } = mockFetch({
    "https://openrouter.ai/api/v1/models": { body: openRouterListFixture() },
    "https://models.dev/api.json": { body: modelsDevFixture() },
    "https://opencode.ai/zen/go/v1/models": { status: 404 },
    "https://hyper.charm.land/v1/models": { status: 404 },
  });
  return { dir, paths, effective, fetch };
}

describe("buildPlanFromSources + applyPlan", () => {
  test("apply writes models.json only; settings unchanged; backups 0600; secrets absent from plan", async () => {
    const { dir, paths, effective, fetch } = await setupPlanEnv();
    const settingsBefore = await readRaw(paths.settingsPath);
    const { plan } = await buildPlanFromSources({ paths, fetchFn: fetch, effective, refCatalog, target: { kind: "all" } });

    // No secrets leak into the plan (operations/findings only carry allowlisted metadata).
    strictEqual(findSecrets(JSON.stringify(plan)).length, 0, "plan must not contain secret values");

    const result = await applyPlan({ plan, paths, fetchFn: fetch, effective, refCatalog });
    strictEqual(result.ok, true, result.error ?? "apply should succeed");

    // settings.json untouched.
    const settingsAfter = await readRaw(paths.settingsPath);
    strictEqual(settingsAfter.hash, settingsBefore.hash);

    // backups are mode 0600.
    for (const b of result.backups) {
      strictEqual((await stat(b)).mode & 0o777, 0o600);
    }

    // models.json changed and transport preserved.
    const modelsAfter = JSON.parse((await readFile(paths.modelsPath, "utf8")));
    strictEqual(modelsAfter.providers.openrouter.apiKey, "$OPENROUTER_API_KEY");
    // unrelated provider still present.
    assertOk(modelsAfter.providers.unrelated);
    void dir;
  });

  test("stale (expired) plan is refused", async () => {
    const { paths, effective, fetch } = await setupPlanEnv();
    const { plan } = await buildPlanFromSources({ paths, fetchFn: fetch, effective, refCatalog, target: { kind: "all" } });
    const expired = { ...plan, expiresAt: new Date(Date.now() - 1000).toISOString() };
    const result = await applyPlan({ plan: expired, paths, fetchFn: fetch, effective, refCatalog });
    strictEqual(result.ok, false);
    assertOk(result.error?.includes("expired"));
  });

  test("config drift (models.json changed since plan) is refused", async () => {
    const { paths, effective, fetch } = await setupPlanEnv();
    const { plan } = await buildPlanFromSources({ paths, fetchFn: fetch, effective, refCatalog, target: { kind: "all" } });
    // Mutate models.json after planning.
    const cur = JSON.parse((await readFile(paths.modelsPath, "utf8")));
    cur.providers.openrouter.models[0].name = "DRIFT";
    await writeFile(paths.modelsPath, JSON.stringify(cur), "utf8");
    const result = await applyPlan({ plan, paths, fetchFn: fetch, effective, refCatalog });
    strictEqual(result.ok, false);
    assertOk(result.error?.includes("models.json changed"));
  });

  test("source drift (official payload changed since plan) is refused", async () => {
    const { paths, effective, fetch } = await setupPlanEnv();
    const { plan } = await buildPlanFromSources({ paths, fetchFn: fetch, effective, refCatalog, target: { kind: "all" } });
    // Different fetch returns a different openrouter payload -> digest mismatch.
    const drifted = mockFetch({
      "https://openrouter.ai/api/v1/models": { body: openRouterListFixture().replace("GPT Test", "GPT Test DRIFTED") },
      "https://models.dev/api.json": { body: modelsDevFixture() },
      "https://opencode.ai/zen/go/v1/models": { status: 404 },
      "https://hyper.charm.land/v1/models": { status: 404 },
    });
    const result = await applyPlan({ plan, paths, fetchFn: drifted.fetch, effective, refCatalog });
    strictEqual(result.ok, false);
    assertOk(result.error?.includes("changed since plan"));
  });

  test("empty plan (0 operations) writes nothing and creates no backup", async () => {
    const { paths, effective, fetch } = await setupPlanEnv();
    const before = await readRaw(paths.modelsPath);
    const plan: import("../types.ts").PendingPlan = {
      id: "empty",
      createdAt: nowIso(),
      expiresAt: new Date(Date.now() + 60_000).toISOString(),
      target: "",
      modelsHash: before.hash,
      settingsHash: (await readRaw(paths.settingsPath)).hash,
      sourceDigests: {
        openrouter: (await fetchSource("openrouter", fetch)).digest,
        "opencode-zen": (await fetchSource("opencode-zen", fetch)).digest,
        "opencode-go": (await fetchSource("opencode-go", fetch)).digest,
      },
      effectiveDigest: canonicalEffectiveDigest(effective),
      operations: [],
      findings: [],
    };
    const result = await applyPlan({ plan, paths, fetchFn: fetch, effective, refCatalog });
    strictEqual(result.ok, true);
    strictEqual(result.applied, 0);
    strictEqual(result.backups.length, 0, "no backup created for empty plan");
    const after = await readRaw(paths.modelsPath);
    strictEqual(after.hash, before.hash, "models.json byte-identical after empty apply");
  });

  test("check (diagnose) writes nothing", async () => {
    const { paths, effective, fetch } = await setupPlanEnv();
    const modelsBefore = await readRaw(paths.modelsPath);
    const settingsBefore = await readRaw(paths.settingsPath);
    await diagnose({ paths, fetchFn: fetch, effective, refCatalog });
    const modelsAfter = await readRaw(paths.modelsPath);
    const settingsAfter = await readRaw(paths.settingsPath);
    strictEqual(modelsAfter.hash, modelsBefore.hash, "check must not change models.json");
    strictEqual(settingsAfter.hash, settingsBefore.hash, "check must not change settings.json");
  });
});

// Real config/auth is NEVER read by tests (isolation). Real-config byte-identity
// is proven by the acceptance smoke (temp config only), not by the test suite.

// ─── Regression: reviewer blockers ──────────────────────────────────────────

describe("regression: no nonexistent Pi API in production source", () => {
  test("production source never calls ModelRegistry.getModels()", async () => {
    const { readFile: rf } = await import("node:fs/promises");
    const srcs = ["index.ts", "doctor.ts", "sources.ts", "storage.ts", "types.ts"];
    for (const f of srcs) {
      const txt = (await rf(join(SRC_DIR, f))).toString();
      strictEqual(txt.includes(".getModels("), false, `${f} must not call getModels()`);
      strictEqual(txt.includes("getModels("), false, `${f} must not reference getModels(`);
    }
  });
});

describe("regression: terminal-injection sanitization", () => {
  test("control chars in result fields are stripped from the report", () => {
    const esc = "\u001B[31mred\u001B[0m";
    const report = __internals.buildResultReport(
      {
        results: [{
          provider: "openrouter", modelId: `${esc}evil/id`, action: "added",
          characteristics: { name: `${esc}name`, api: "x", contextWindow: 1, maxTokens: 1, input: ["text"], reasoning: true, source: `${esc}src` },
        }],
        totals: [{ provider: "openrouter", sourceStatus: "ok", added: 1, updated: 0, skipped: 0, unchanged: 0 }],
        warnings: [],
      },
      { ok: true, applied: 1, backups: [`${esc}bak`], findings: [] },
    );
    strictEqual(report.includes("\u001B"), false, "no raw ESC in report");
    strictEqual(report.includes("\u0000"), false, "no raw NUL");
    strictEqual(sanitize(esc).includes("\u001B"), false);
  });
});

describe("regression: stale detail / effective races", () => {
  test("detail payload drift since plan is refused at apply", async () => {
    const { paths, effective, fetch } = await setupPlanEnv();
    // Targeted openrouter plan: detail endpoints are fetched and digested.
    const detailUrl = "https://openrouter.ai/api/v1/models/openai/gpt-test/endpoints";
    const fetchWithDetail = mockFetch({
      "https://openrouter.ai/api/v1/models": { body: openRouterListFixture() },
      "https://models.dev/api.json": { body: modelsDevFixture() },
      "https://opencode.ai/zen/go/v1/models": { status: 404 },
      "https://hyper.charm.land/v1/models": { status: 404 },
      [detailUrl]: { body: openRouterEndpointsFixture({ context: [128000], max: [16384] }) },
    });
    const { plan } = await buildPlanFromSources({
      paths, fetchFn: fetchWithDetail.fetch, effective, refCatalog,
      target: { kind: "model", provider: "openrouter", modelId: "openai/gpt-test" },
    });
    assertOk(plan.detail, "plan captured the detail digest");
    // At apply, the detail payload changed -> refused.
    const driftedDetail = mockFetch({
      "https://openrouter.ai/api/v1/models": { body: openRouterListFixture() },
      "https://models.dev/api.json": { body: modelsDevFixture() },
      "https://opencode.ai/zen/go/v1/models": { status: 404 },
      "https://hyper.charm.land/v1/models": { status: 404 },
      [detailUrl]: { body: openRouterEndpointsFixture({ context: [999999], max: [1] }) },
    });
    const result = await applyPlan({ plan, paths, fetchFn: driftedDetail.fetch, effective, refCatalog });
    strictEqual(result.ok, false);
    assertOk(result.error?.includes("detail"), result.error);
  });

  test("effective catalog drift since plan is refused at apply", async () => {
    const { paths, effective, fetch } = await setupPlanEnv();
    const { plan } = await buildPlanFromSources({ paths, fetchFn: fetch, effective, refCatalog, target: { kind: "all" } });
    // Different effective snapshot at apply time.
    const driftedEffective = effectiveFromRaw({
      "opencode-go": [{ id: "glm-5.2", name: "CHANGED", api: "anthropic-messages", reasoning: true, input: ["text"], cost: {}, contextWindow: 1, maxTokens: 1 }],
      openrouter: [], opencode: [],
    });
    const result = await applyPlan({ plan, paths, fetchFn: fetch, effective: driftedEffective, refCatalog });
    strictEqual(result.ok, false);
    assertOk(result.error?.includes("effective pi catalog changed"), result.error);
  });

  test("concurrent models.json mutation during apply is refused without clobber (CAS)", async () => {
    const { paths, effective, fetch } = await setupPlanEnv();
    const { plan } = await buildPlanFromSources({ paths, fetchFn: fetch, effective, refCatalog, target: { kind: "all" } });
    // A fetchFn that mutates models.json on disk during apply's source re-fetch,
    // so the bytes differ at the CAS check right before rename.
    let mutated = false;
    const mutatingFetch: typeof fetch = (async (url: string, init?: unknown) => {
      if (!mutated && String(url).includes("openrouter.ai/api/v1/models")) {
        mutated = true;
        await writeFile(paths.modelsPath, JSON.stringify({ providers: { concurrent: { models: [] } } }), "utf8");
      }
      return fetch(url, init as never);
    }) as typeof fetch;
    const result = await applyPlan({ plan, paths, fetchFn: mutatingFetch, effective, refCatalog });
    strictEqual(result.ok, false, "apply must refuse due to CAS");
    // The file was mutated by the concurrent writer and must NOT be clobbered by us.
    const after = JSON.parse((await readFile(paths.modelsPath, "utf8")));
    assertOk(after.providers.concurrent, "concurrent writer's change preserved (not clobbered)");
  });
});

describe("regression: invalid schema refusal", () => {
  test("parseModelsConfigStrict rejects providers not an object", () => {
    throws(() => parseModelsConfigStrict('{"providers":[]}'), /is not a \{ providers/);
  });
  test("parseModelsConfigStrict rejects models not an array", () => {
    throws(() => parseModelsConfigStrict('{"providers":{"openrouter":{"models":"not-an-array"}}}'), /models is not an array/);
  });
  test("buildPlanFromSources refuses to plan a malformed models.json", async () => {
    const dir = await makeTempDir();
    const paths: DoctorPaths = { modelsPath: join(dir, "models.json"), settingsPath: join(dir, "settings.json"), backupDir: dir };
    await writeFile(paths.modelsPath, '{"providers":{"openrouter":{"models":"not-an-array"}}}', "utf8");
    await writeJson(paths.settingsPath, {});
    const effective = effectiveFromRaw(effectiveGoFixture() as Record<string, Array<Record<string, unknown>>>);
    const { fetch } = mockFetch({
      "https://openrouter.ai/api/v1/models": { body: openRouterListFixture() },
      "https://models.dev/api.json": { body: modelsDevFixture() },
      "https://opencode.ai/zen/go/v1/models": { status: 404 },
      "https://hyper.charm.land/v1/models": { status: 404 },
    });
    await rejects(
      () => buildPlanFromSources({ paths, fetchFn: fetch, effective, refCatalog, target: { kind: "all" } }),
      /models is not an array|not a \{ providers/,
    );
  });
});

describe("regression: deep-diff allowlist completeness", () => {
  test("changing an arbitrary provider-level transport field is rejected", () => {
    const base: ModelsConfig = {
      providers: { openrouter: { apiKey: "$X", models: [{ id: "m", name: "n", reasoning: false, cost: {}, contextWindow: 1, maxTokens: 1, input: ["text"] }] } },
    };
    const next = structuredClone(base);
    (next.providers.openrouter as Record<string, unknown>).authToken = "should-not-be-allowed";
    const proof = verifyDiffAllowlisted(base, next);
    strictEqual(proof.ok, false);
    assertOk(proof.violations.some((v) => v.includes("authToken")));
  });
  test("duplicate model id within models[] is rejected", () => {
    const base: ModelsConfig = { providers: { openrouter: { apiKey: "$X", models: [{ id: "m", name: "n", reasoning: false, cost: {}, contextWindow: 1, maxTokens: 1, input: ["text"] }] } } };
    const next = structuredClone(base);
    next.providers.openrouter!.models!.push({ id: "m", name: "n2", reasoning: false, cost: {}, contextWindow: 1, maxTokens: 1, input: ["text"] });
    const proof = verifyDiffAllowlisted(base, next);
    strictEqual(proof.ok, false);
    assertOk(proof.violations.some((v) => v.includes("duplicate")));
  });
});

describe("regression: glob auth resolved against concrete ids", () => {
  test("glob reference with auth on matched model -> no missing_auth", () => {
    const cat: RefCatalog = {
      modelIds: (p) => (p === "openai-codex" ? ["gpt-5.6-sol", "gpt-5.6-luna"] : []),
      hasAuth: (_p, id) => id === "gpt-5.6-sol" || id === "gpt-5.6-luna",
    };
    const findings = validateReferences({ enabledModels: ["openai-codex/gpt-*"] }, cat);
    strictEqual(findings.find((f) => f.code === "missing_auth"), undefined);
    strictEqual(findings.find((f) => f.code === "missing_reference"), undefined);
  });
  test("glob reference where matched models lack auth -> missing_auth per model", () => {
    const cat: RefCatalog = {
      modelIds: (p) => (p === "openai-codex" ? ["gpt-5.6-sol"] : []),
      hasAuth: () => false,
    };
    const findings = validateReferences({ enabledModels: ["openai-codex/gpt-*"] }, cat);
    assertOk(findings.find((f) => f.code === "missing_auth" && f.modelId === "gpt-5.6-sol"));
  });
});

describe("regression: three-way drift findings", () => {
  test("diagnoseManagedDrift emits drift, catalog_only, and alias findings", () => {
    // Start from the shared fixture, then add the aliased `deepseek/ds-alias`
    // (whose source canonical_slug differs) to the user's openrouter models so
    // the alias finding is exercised in addition to drift + catalog_only.
    const user = structuredClone(userModelsFixture()) as ModelsConfig;
    user.providers.openrouter!.models!.push({
      id: "deepseek/ds-alias",
      name: "DS Alias OLD",
      reasoning: false,
      input: ["text"],
      contextWindow: 1000,
      maxTokens: 100,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    });
    const eff = effectiveFromRaw(effectiveGoFixture() as Record<string, Array<Record<string, unknown>>>);
    const sources: SourceResult[] = [
      { id: "openrouter", status: "ok", digest: "d", fetchedAt: nowIso(), models: normalizeOpenRouterList(JSON.parse(openRouterListFixture())) },
      { id: "opencode-zen", status: "ok", digest: "d2", fetchedAt: nowIso(), models: normalizeModelsDev(JSON.parse(modelsDevFixture())) },
      { id: "opencode-go", status: "unavailable", digest: "d3", fetchedAt: nowIso(), models: [] },
    ];
    const findings = diagnoseManagedDrift(user, eff, sources);
    const codes = new Set(findings.map((f) => f.code));
    assertOk(codes.has("drift"), `expected drift finding; got ${[...codes].join(",")}`);
    assertOk(codes.has("catalog_only"), "expected catalog_only finding");
    assertOk(codes.has("alias"), "expected alias finding");
    // Drift must be reported on the diverging field, not silently auto-applied
    // through any non-allowlisted channel.
    const driftAlias = findings.find((f) => f.code === "alias");
    assertOk(driftAlias?.ref?.includes("deepseek/ds-alias"), "alias finding scoped to the aliased model");
    // No secrets leak into drift findings.
    strictEqual(findSecrets(JSON.stringify(findings)).length, 0);
  });
});

// ─── Regression: confirmed blockers from second rereview ────────────────────

describe("regression: parseModelsConfigStrict rejects duplicate / empty ids", () => {
  test("duplicate model id within a provider is rejected", () => {
    throws(
      () => parseModelsConfigStrict(JSON.stringify({
        providers: { openrouter: { apiKey: "$X", models: [
          { id: "dup", name: "a", reasoning: false, cost: {}, contextWindow: 1, maxTokens: 1, input: ["text"] },
          { id: "dup", name: "b", reasoning: false, cost: {}, contextWindow: 1, maxTokens: 1, input: ["text"] },
        ] } },
      })),
      /duplicate id/i,
    );
  });
  test("empty / whitespace id is rejected", () => {
    throws(
      () => parseModelsConfigStrict(JSON.stringify({
        providers: { openrouter: { apiKey: "$X", models: [{ id: "   ", name: "x" }] } },
      })),
      /empty|whitespace|string id/i,
    );
  });
});

describe("regression: deep allowlist proof — managed scope + order", () => {
  test("editing a model in a NON-managed provider is rejected", () => {
    const base: ModelsConfig = {
      providers: {
        unrelated: { apiKey: "$X", models: [{ id: "keepme", name: "Keep", reasoning: false, cost: {}, contextWindow: 1, maxTokens: 1, input: ["text"] }] },
      },
    };
    const next = structuredClone(base);
    next.providers.unrelated!.models![0]!.name = "Tampered"; // allowlisted field, wrong provider
    const proof = verifyDiffAllowlisted(base, next);
    strictEqual(proof.ok, false);
    assertOk(proof.violations.some((v) => v.includes("unrelated") && v.includes("generic provider entries are never modified")));
  });

  test("reordering managed models[] is rejected even if contents are identical", () => {
    const base: ModelsConfig = {
      providers: {
        openrouter: { apiKey: "$X", models: [
          { id: "a", name: "A", reasoning: false, cost: {}, contextWindow: 1, maxTokens: 1, input: ["text"] },
          { id: "b", name: "B", reasoning: false, cost: {}, contextWindow: 1, maxTokens: 1, input: ["text"] },
        ] },
      },
    };
    const next = structuredClone(base);
    // Swap order; contents byte-identical otherwise.
    next.providers.openrouter!.models = [base.providers.openrouter!.models![1]!, base.providers.openrouter!.models![0]!];
    const proof = verifyDiffAllowlisted(base, next);
    strictEqual(proof.ok, false);
    assertOk(proof.violations.some((v) => v.includes("order changed")));
  });

  test("managed provider: appending a new model at the end keeps order valid", () => {
    const base: ModelsConfig = {
      providers: {
        openrouter: { apiKey: "$X", models: [
          { id: "a", name: "A", reasoning: false, cost: {}, contextWindow: 1, maxTokens: 1, input: ["text"] },
        ] },
      },
    };
    const next = structuredClone(base);
    next.providers.openrouter!.models!.push({ id: "b", name: "B", reasoning: false, cost: {}, contextWindow: 1, maxTokens: 1, input: ["text"] });
    strictEqual(verifyDiffAllowlisted(base, next).ok, true, "append keeps relative order");
  });
});

describe("regression: composite cost merge preserves unknown members + never discloses them (reviewer 820bdad3)", () => {
  test("cost op MERGES allowlisted subfields, preserves cacheWrite + vendorSecret; report never prints them", () => {
    const secret = "sk-DO-NOT-REPORT-vendorSecret";
    const beforeCost = { input: 1, output: 2, cacheWrite: 0, vendorSecret: secret };
    const base: ModelsConfig = {
      providers: { openrouter: { apiKey: "$X", models: [
        { id: "m", name: "M", reasoning: false, input: ["text"], contextWindow: 1, maxTokens: 1, cost: beforeCost },
      ] } },
    };
    const op = {
      provider: "openrouter" as ManagedProvider, modelId: "m", path: "cost" as MutationPath,
      before: beforeCost, after: { input: 5, output: 15, cacheRead: 1.25 },
      source: "openrouter:list", reason: "openrouter:list",
    };
    const next = structuredClone(base);
    applyOperationsToClone(next, [op]);
    const merged = next.providers.openrouter!.models![0]!.cost as Record<string, unknown>;
    // allowlisted subfields updated
    strictEqual(merged.input, 5);
    strictEqual(merged.output, 15);
    strictEqual(merged.cacheRead, 1.25);
    // unknown / not-in-source members PRESERVED (not wholesale-replaced)
    strictEqual(merged.cacheWrite, 0, "cacheWrite preserved");
    strictEqual(merged.vendorSecret, secret, "unknown member preserved");
    // deep proof passes (only allowlisted subfields changed)
    strictEqual(verifyDiffAllowlisted(base, next).ok, true, "deep composite proof ok");
    // wholesale replace WOULD fail the proof:
    const wholesale = structuredClone(base);
    wholesale.providers.openrouter!.models![0]!.cost = { input: 5, output: 15, cacheRead: 1.25 };
    strictEqual(verifyDiffAllowlisted(base, wholesale).ok, false, "wholesale replace rejected");
    // report never discloses the complete prior composite (no vendorSecret)
    const report = __internals.buildResultReport(
      { results: [{ provider: "openrouter", modelId: "m", action: "updated", changes: [op] }],
        totals: [{ provider: "openrouter", sourceStatus: "ok", added: 0, updated: 1, skipped: 0, unchanged: 0 }], warnings: [] },
      { ok: true, applied: 1, backups: [], findings: [] },
    );
    strictEqual(report.includes(secret), false, "prior composite secret never disclosed");
    strictEqual(report.includes("vendorSecret"), false, "unknown member name never disclosed");
    assertOk(report.includes("cost.input"), "subfield-level diff shown");
  });
});

describe("regression: three-way effective-pi drift", () => {
  test("user matches official source but differs from effective pi -> effective_drift", () => {
    // Official source says contextWindow 128000; user matches it (128000);
    // but pi's effective catalog has 200000. The source-vs-user pass sees no
    // drift; the effective-vs-user pass must surface the three-way contradiction.
    const user: ModelsConfig = {
      providers: {
        openrouter: { apiKey: "$X", models: [
          { id: "openai/gpt-test", name: "GPT Test", reasoning: false, input: ["text"], contextWindow: 128000, maxTokens: 16384, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } },
        ] },
      },
    };
    const eff = effectiveFromRaw({
      openrouter: [{ id: "openai/gpt-test", name: "GPT Test", api: "openai", reasoning: false, input: ["text"], contextWindow: 200000, maxTokens: 16384, cost: {} }],
      opencode: [], "opencode-go": [],
    });
    const sources: SourceResult[] = [
      { id: "openrouter", status: "ok", digest: "d", fetchedAt: nowIso(), models: normalizeOpenRouterList(JSON.parse(openRouterListFixture())) },
      { id: "opencode-zen", status: "ok", digest: "d2", fetchedAt: nowIso(), models: normalizeModelsDev(JSON.parse(modelsDevFixture())) },
      { id: "opencode-go", status: "unavailable", digest: "d3", fetchedAt: nowIso(), models: [] },
    ];
    const findings = diagnoseManagedDrift(user, eff, sources);
    const ed = findings.find((f) => f.code === "effective_drift" && f.modelId === "openai/gpt-test" && f.ref?.includes("contextWindow"));
    assertOk(ed, `expected effective_drift on contextWindow; got ${[...new Set(findings.map((f) => f.code))].join(",")}`);
    // And NO source-vs-user drift on contextWindow (user matches official).
    const srcDrift = findings.find((f) => f.code === "drift" && f.ref?.includes("contextWindow"));
    strictEqual(srcDrift, undefined, "user matches official source -> no source drift");
  });
});

describe("regression: terminal line-spoofing sanitization", () => {
  test("sanitize strips LF and TAB (and all controls) from untrusted fields", () => {
    const spoof = "id\n\tSPOOF";
    const out = sanitize(spoof);
    strictEqual(out.includes("\n"), false, "no raw LF");
    strictEqual(out.includes("\t"), false, "no raw TAB");
    strictEqual(out.includes("SPOOF"), true, "visible text retained");
  });

  test("a model id with an embedded LF cannot forge extra report lines", () => {
    // A malicious catalog could place a LF in a model id / reason to inject a
    // forged report line. Per-field sanitize (modelId/reason) strips it before
    // interpolation, and reportSanitized scrubs line content.
    const report = __internals.buildResultReport(
      {
        results: [{
          provider: "openrouter", modelId: "evil/m\n  WARNING spoofed-line", action: "skipped",
          reason: "unsafe\n  INFO forged",
        }],
        totals: [{ provider: "openrouter", sourceStatus: "ok", added: 0, updated: 0, skipped: 1, unchanged: 0 }],
        warnings: [],
      },
      { ok: true, applied: 0, backups: [], findings: [] },
    );
    strictEqual(report.includes("spoofed-line"), true, "visible content kept");
    const lines = report.split("\n");
    // The spoofed text must NOT become a standalone forged report line.
    strictEqual(lines.some((l) => /^\s*WARNING spoofed-line/.test(l)), false, "no forged standalone WARNING line");
    strictEqual(lines.some((l) => /^\s*INFO forged/.test(l)), false, "no forged standalone INFO line");
  });
});

describe("regression: malformed / disagreeing OpenRouter detail", () => {
  // openRouterDetailUrl encodes each id segment but keeps the org/model slash.
  const detailUrl = "https://openrouter.ai/api/v1/models/openai/gpt-test/endpoints";

  async function planWithDetail(detailBody: string | number) {
    const { paths, effective, fetch } = await setupPlanEnv();
    const routes: Record<string, { body?: string; status?: number }> = {
      "https://openrouter.ai/api/v1/models": { body: openRouterListFixture() },
      "https://models.dev/api.json": { body: modelsDevFixture() },
      "https://opencode.ai/zen/go/v1/models": { status: 404 },
      "https://hyper.charm.land/v1/models": { status: 404 },
    };
    if (typeof detailBody === "string") routes[detailUrl] = { body: detailBody };
    else routes[detailUrl] = { status: detailBody };
    const mocked = mockFetch(routes);
    const { plan } = await buildPlanFromSources({
      paths, fetchFn: mocked.fetch, effective, refCatalog,
      target: { kind: "model", provider: "openrouter", modelId: "openai/gpt-test" },
    });
    return { plan, paths };
  }

  test("malformed detail: no list-derived limit ops + detail_malformed finding", async () => {
    const { plan } = await planWithDetail("<html>not json</html>");
    const gptOps = plan.operations.filter((o) => o.modelId === "openai/gpt-test");
    strictEqual(gptOps.some((o) => o.path === "contextWindow"), false, "list-derived contextWindow must NOT be applied");
    // ensureMaxTokens policy: an explicit maxTokens is ALWAYS planned; with the
    // endpoint confirmation gone it must be the derived fallback, never the
    // list-derived 16384.
    const maxOps = gptOps.filter((o) => o.path === "maxTokens");
    assertOk(maxOps.every((o) => o.source === "derived:max-tokens-fallback"), "maxTokens only from the derived fallback");
    assertOk(maxOps.every((o) => o.after === 65536), "derived fallback value");
    assertOk(plan.findings.some((f) => f.code === "detail_malformed"), "detail_malformed finding emitted");
  });

  test("disagreeing detail: no context op + detail_ambiguity finding", async () => {
    const { plan } = await planWithDetail(openRouterEndpointsFixture({ context: [100000, 120000], max: [16384, 16384] }));
    const gptOps = plan.operations.filter((o) => o.modelId === "openai/gpt-test");
    strictEqual(gptOps.some((o) => o.path === "contextWindow"), false, "ambiguous context must NOT be applied");
    assertOk(plan.findings.some((f) => f.code === "detail_ambiguity"), "detail_ambiguity finding emitted");
  });

  test("unavailable detail (non-2xx): no list-derived limit ops + detail_unavailable finding", async () => {
    const { plan } = await planWithDetail(500);
    const gptOps = plan.operations.filter((o) => o.modelId === "openai/gpt-test");
    strictEqual(gptOps.some((o) => o.path === "contextWindow"), false);
    assertOk(plan.findings.some((f) => f.code === "detail_unavailable"), "detail_unavailable finding emitted");
  });
});

describe("regression: targeted plan is exclusive to the requested model (reviewer 1c9740c2)", () => {
  const diagnosis = makeDiagnosis(
    [openRouterSource()],
    effectiveFromRaw(effectiveGoFixture() as Record<string, Array<Record<string, unknown>>>),
  );

  test("plan provider/model reconciles ONLY the target, never unrelated existing models", () => {
    const user = userModelsFixture() as ModelsConfig;
    const target: ParsedTarget = { kind: "model", provider: "openrouter", modelId: "deepseek/ds-alias" };
    const ops = buildOperations(diagnosis, user, target);
    const modelIds = new Set(ops.map((o) => o.modelId));
    assertOk(modelIds.has("deepseek/ds-alias"), "target model present");
    // openai/gpt-test exists in the user config but is NOT the target -> excluded.
    strictEqual(modelIds.has("openai/gpt-test"), false, "unrelated existing model must not be reconciled");
  });

  test("targeting an EXISTING model reconciles only that one model", () => {
    const user = userModelsFixture() as ModelsConfig;
    const target: ParsedTarget = { kind: "model", provider: "openrouter", modelId: "openai/gpt-test" };
    const ops = buildOperations(diagnosis, user, target);
    const modelIds = new Set(ops.map((o) => o.modelId));
    strictEqual(modelIds.size, 1, "only the targeted model appears");
    assertOk(modelIds.has("openai/gpt-test"));
  });

  test("no-target (provider/all) still reconciles all existing models", () => {
    const user = userModelsFixture() as ModelsConfig;
    const ops = buildOperations(diagnosis, user, { kind: "provider", provider: "openrouter" });
    const modelIds = new Set(ops.map((o) => o.modelId));
    assertOk(modelIds.has("openai/gpt-test"), "no-target reconciles existing openrouter models");
  });
});

describe("regression: per-field operation provenance", () => {
  test("detail-refined limit op carries openrouter:endpoints source, not openrouter:list", async () => {
    const detailUrl = "https://openrouter.ai/api/v1/models/openai/gpt-test/endpoints";
    const { paths, effective, fetch } = await setupPlanEnv();
    const mocked = mockFetch({
      "https://openrouter.ai/api/v1/models": { body: openRouterListFixture() },
      "https://models.dev/api.json": { body: modelsDevFixture() },
      "https://opencode.ai/zen/go/v1/models": { status: 404 },
      "https://hyper.charm.land/v1/models": { status: 404 },
      [detailUrl]: { body: openRouterEndpointsFixture({ context: [128000], max: [16384] }) },
    });
    const { plan } = await buildPlanFromSources({
      paths, fetchFn: mocked.fetch, effective, refCatalog,
      target: { kind: "model", provider: "openrouter", modelId: "openai/gpt-test" },
    });
    const ctxOp = plan.operations.find((o) => o.modelId === "openai/gpt-test" && o.path === "contextWindow");
    assertOk(ctxOp, "context op exists from agreeing endpoint evidence");
    strictEqual(ctxOp!.source, "openrouter:endpoints", "provenance reflects the field's evidence source");
  });
});

describe("regression: opencode-go is a real reconcile provider (update + add)", () => {
  test("a mocked 200 Go source updates drifted entries and adds registry-absent models", () => {
    // Go source: glm-5.2 (enriched, drifts from user) + kimi-k3 (has a plannable field, absent from registry).
    const goSource: SourceResult = {
      id: "opencode-go", status: "ok", digest: "d-go", fetchedAt: nowIso(),
      models: [
        { provider: "opencode-go", id: "glm-5.2", name: ev("GLM-5.2 GO"), contextWindow: ev(1000000) },
        { provider: "opencode-go", id: "kimi-k3", name: ev("Kimi K3"), contextWindow: ev(256000) },
      ],
    };
    // Pi effective opencode-go has glm-5.2 (so not added, only updated) but NOT kimi-k3 (-> added).
    const eff = effectiveFromRaw({
      "opencode-go": [{ id: "glm-5.2", name: "GLM-5.2", api: "anthropic-messages", reasoning: true, input: ["text"], contextWindow: 1000000, maxTokens: 131072, cost: {} }],
      openrouter: [], opencode: [],
    });
    const user: ModelsConfig = {
      providers: {
        "opencode-go": { apiKey: "$GO_KEY", baseUrl: "https://go.example", models: [
          { id: "glm-5.2", name: "OLD", reasoning: false, input: ["text"], contextWindow: 50000, maxTokens: 4096, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } },
        ] },
        },
    };
    const { operations, report } = reconcileAll({ sources: [goSource], findings: [], effective: eff }, user);
    const goTotal = report.totals.find((t) => t.provider === "opencode-go");
    assertOk(goTotal, "opencode-go total present");
    strictEqual(goTotal!.sourceStatus, "ok", "opencode-go source ok (real provider)");
    // glm-5.2 drifts (name/context) -> updated; kimi-k3 absent from registry -> added.
    assertOk(goTotal!.updated >= 1, "drifted opencode-go entry updated");
    assertOk(goTotal!.added >= 1, "registry-absent opencode-go model added");
    // transport preserved in the cloned result
    const clone = structuredClone(user);
    applyOperationsToClone(clone, operations);
    strictEqual(clone.providers["opencode-go"]!.apiKey, "$GO_KEY", "apiKey preserved");
    strictEqual(clone.providers["opencode-go"]!.baseUrl, "https://go.example", "baseUrl preserved");
    assertOk(clone.providers["opencode-go"]!.models!.some((m) => m.id === "kimi-k3"), "kimi-k3 added");
    // deep proof passes
    strictEqual(verifyDiffAllowlisted(user, clone).ok, true, "allowlist proof ok");
  });

  test("opencode-go source failure is isolated (total unavailable, others proceed)", () => {
    const goSource: SourceResult = { id: "opencode-go", status: "unavailable", digest: "d", models: [], error: "404/500", fetchedAt: nowIso() };
    const eff = effectiveFromRaw({ openrouter: [], opencode: [], "opencode-go": [] });
    const user: ModelsConfig = { providers: {} };
    const { report } = reconcileAll({ sources: [goSource], findings: [{ provider: "opencode-go", severity: "warning", code: "source_unavailable", message: "opencode-go unavailable" }], effective: eff }, user);
    const goTotal = report.totals.find((t) => t.provider === "opencode-go");
    strictEqual(goTotal!.sourceStatus, "unavailable", "failure visible in total");
    assertOk(report.warnings.some((w) => w.includes("opencode-go")), "failure surfaced as warning");
  });
});

// ─── Regression: block-less providers must never receive add ops ───────────

describe("regression: block-less providers never get add ops (allowlist refusal bug)", () => {
  test("opencode-go without a models.json block: registry-absent source model is NOT added; warning emitted", () => {
    // Production shape (2026-08-20): models.json has NO `opencode-go` block, and
    // the live Go catalog carries a model newer than pi's builtin registry
    // (muse-spark-1.2-contributor). The old planner emitted add ops that created
    // a provider block, verifyDiffAllowlisted refused the ENTIRE apply, and the
    // report hid the reason ("allowlist verification failed").
    const goSource: SourceResult = {
      id: "opencode-go", status: "ok", digest: "d-go", fetchedAt: nowIso(),
      models: [
        { provider: "opencode-go", id: "glm-5.2", name: ev("GLM-5.2 GO"), contextWindow: ev(1000000) },
        { provider: "opencode-go", id: "muse-spark-1.2-contributor", name: ev("Muse Spark 1.2 Contributor"), contextWindow: ev(256000) },
      ],
    };
    const eff = effectiveFromRaw({
      "opencode-go": [{ id: "glm-5.2", name: "GLM-5.2", api: "anthropic-messages", reasoning: true, input: ["text"], contextWindow: 1000000, maxTokens: 131072, cost: {} }],
      openrouter: [], opencode: [],
    });
    const user: ModelsConfig = { providers: {} }; // NO opencode-go block
    const { operations, report } = reconcileAll({ sources: [goSource], findings: [], effective: eff }, user);
    strictEqual(operations.filter((o) => o.provider === "opencode-go").length, 0, "no opencode-go ops (block absent)");
    const goTotal = report.totals.find((t) => t.provider === "opencode-go");
    strictEqual(goTotal!.added, 0, "nothing added for block-less provider");
    assertOk(
      report.warnings.some((w) => w.includes("provider_block_absent") && w.includes("providers.opencode-go has")),
      "provider_block_absent warning emitted",
    );
    // The proof that used to fail: applying the planned ops must stay allowlisted.
    const clone = structuredClone(user);
    applyOperationsToClone(clone, operations);
    strictEqual(verifyDiffAllowlisted(user, clone).ok, true, "no provider block created");
  });

  test("a provider WITH a block still receives add ops (openrouter without models[])", () => {
    // Guard against over-correction: an existing block without a models array
    // (the real openrouter entry: modelOverrides only) still gets safe adds —
    // ensureModelsArray creates models[] inside the EXISTING block, which the
    // allowlist proof permits (provider-level fields untouched).
    const eff = effectiveFromRaw({ openrouter: [], opencode: [], "opencode-go": [] });
    const orSource = openRouterSource();
    const user: ModelsConfig = {
      providers: { openrouter: { apiKey: "$OPENROUTER_API_KEY", modelOverrides: { "openai/gpt-test": { contextWindow: 60000 } } } },
    };
    const { operations, report } = reconcileAll({ sources: [orSource], findings: [], effective: eff }, user);
    assertOk(operations.some((o) => o.provider === "openrouter" && o.modelId === "openai/gpt-test"), "add op for existing block");
    const clone = structuredClone(user);
    applyOperationsToClone(clone, operations);
    strictEqual(verifyDiffAllowlisted(user, clone).ok, true, "block-present add stays allowlisted");
    strictEqual(report.warnings.some((w) => w.includes("provider_block_absent")), false, "no absent-block warning");
  });
});

// ─── Regression: cost four-keys rule (production regression: 59 broken models)

describe("regression: cost is always emitted with ALL FOUR rate keys", () => {
  test("a source cost missing cacheRead/cacheWrite is materialized to 0 on write (would have failed without the fix)", () => {
    // Exact failing shape: endpoint-derived prices WITHOUT cache info -> source
    // cost = {input, output} only. Previously written as-is (partial) -> Pi broke.
    const base: ModelsConfig = {
      providers: { openrouter: { apiKey: "$X", models: [
        { id: "m", name: "M", reasoning: false, input: ["text"], contextWindow: 1, maxTokens: 1,
          cost: { input: 1, output: 2, cacheRead: 0, cacheWrite: 0 } },
      ] } },
    };
    const clone = structuredClone(base);
    // op.after carries ONLY input/output (no cache info) — the broken shape.
    applyOperationsToClone(clone, [{
      provider: "openrouter", modelId: "m", path: "cost",
      before: clone.providers.openrouter!.models![0]!.cost,
      after: { input: 5, output: 15 }, source: "openrouter:list", reason: "openrouter:list",
    }]);
    const written = clone.providers.openrouter!.models![0]!.cost as Record<string, unknown>;
    // ALL FOUR keys must be present (cacheRead/cacheWrite defaulted to 0).
    strictEqual(written.input, 5);
    strictEqual(written.output, 15);
    strictEqual(written.cacheRead, 0, "cacheRead materialized to 0");
    strictEqual(written.cacheWrite, 0, "cacheWrite materialized to 0");
    // the written config passes the schema gate.
    strictEqual(validateModelsConfig(clone).length, 0, "materialized cost is schema-valid");
  });

  test("a brand-new model upsert with a cache-less source cost still gets all four keys", () => {
    const base: ModelsConfig = { providers: { openrouter: { apiKey: "$X", models: [] } } };
    const clone = structuredClone(base);
    applyOperationsToClone(clone, [{
      provider: "openrouter", modelId: "new", path: "cost",
      before: undefined, after: { input: 0.2, output: 0.8 }, source: "openrouter:list", reason: "openrouter:list",
    }]);
    const written = clone.providers.openrouter!.models![0]!.cost as Record<string, unknown>;
    strictEqual(written.cacheRead, 0);
    strictEqual(written.cacheWrite, 0);
    strictEqual(validateModelsConfig(clone).length, 0);
  });
});

describe("regression: schema gate rejects the exact failing shapes (59 broken models)", () => {
  test("cost missing cacheRead is rejected", () => {
    const cfg = { providers: { openrouter: { models: [
      { id: "m", cost: { input: 1, output: 2, cacheWrite: 0 } },
    ] } } } as unknown as ModelsConfig;
    const errs = validateModelsConfig(cfg);
    assertOk(errs.some((e) => e.includes("cost.cacheRead")), "cacheRead missing flagged");
  });

  test("cost missing cacheWrite is rejected", () => {
    const cfg = { providers: { openrouter: { models: [
      { id: "m", cost: { input: 1, output: 2, cacheRead: 0 } },
    ] } } } as unknown as ModelsConfig;
    const errs = validateModelsConfig(cfg);
    assertOk(errs.some((e) => e.includes("cost.cacheWrite")), "cacheWrite missing flagged");
  });

  test("endpoint-derived cost with only input/output is rejected", () => {
    const cfg = { providers: { openrouter: { models: [
      { id: "m", cost: { input: 5, output: 15 } },
    ] } } } as unknown as ModelsConfig;
    const errs = validateModelsConfig(cfg);
    assertOk(errs.length >= 2, "both missing rates flagged");
  });

  test("non-integer/non-numeric limits, non-boolean reasoning, null metadata rejected", () => {
    const cfg = { providers: { openrouter: { models: [
      { id: "a", contextWindow: "big" },
      { id: "b", reasoning: "yes" },
      { id: "c", cost: null },
      { id: "d", maxTokens: -1 },
    ] } } } as unknown as ModelsConfig;
    const errs = validateModelsConfig(cfg);
    assertOk(errs.some((e) => e.includes("contextWindow")));
    assertOk(errs.some((e) => e.includes("reasoning")));
    assertOk(errs.some((e) => e.includes("cost")));
    assertOk(errs.some((e) => e.includes("maxTokens")));
  });

  test("a valid four-key cost passes the gate", () => {
    const cfg = { providers: { openrouter: { models: [
      { id: "m", cost: { input: 1, output: 2, cacheRead: 0.25, cacheWrite: 0 } },
    ] } } } as unknown as ModelsConfig;
    strictEqual(validateModelsConfig(cfg).length, 0);
  });
});

describe("regression: applyPlan pre-write schema gate refuses invalid content", () => {
  test("a plan whose final config has a partial cost (pre-existing) is refused, nothing written", async () => {
    const dir = await makeTempDir();
    const paths: DoctorPaths = { modelsPath: join(dir, "models.json"), settingsPath: join(dir, "settings.json"), backupDir: dir };
    // Existing config already has a PARTIAL cost (missing cacheWrite) on an
    // untouched model; the doctor's own op is a valid name change on another model.
    await writeJson(paths.modelsPath, {
      providers: { openrouter: { apiKey: "$X", models: [
        { id: "bad", name: "Bad", cost: { input: 1, output: 2, cacheRead: 0 } },
        { id: "good", name: "Good OLD", reasoning: false, input: ["text"], contextWindow: 1, maxTokens: 1, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } },
      ] } },
    });
    await writeJson(paths.settingsPath, {});
    const before = (await readRaw(paths.modelsPath)).hash;
    const plan: import("../types.ts").PendingPlan = {
      id: "p", createdAt: nowIso(), expiresAt: new Date(Date.now() + 60_000).toISOString(), target: "",
      modelsHash: before, settingsHash: (await readRaw(paths.settingsPath)).hash, sourceDigests: {}, effectiveDigest: canonicalEffectiveDigest(effectiveFromRaw({ openrouter: [], opencode: [], "opencode-go": [] })),
      operations: [{ provider: "openrouter", modelId: "good", path: "name", before: "Good OLD", after: "Good NEW", source: "src", reason: "src" }],
      findings: [],
    };
    const result = await applyPlan({ plan, paths, fetchFn: (async () => ({ ok: true, status: 200, url: "", headers: { get: () => null }, text: async () => "", arrayBuffer: async () => new ArrayBuffer(0) })) as never, effective: effectiveFromRaw({ openrouter: [], opencode: [], "opencode-go": [] }), refCatalog: { modelIds: () => [], hasAuth: () => true } });
    strictEqual(result.ok, false, "apply refused due to schema violation");
    assertOk(result.error?.includes("schema"), "schema error surfaced");
    // nothing written
    strictEqual((await readRaw(paths.modelsPath)).hash, before, "models.json unchanged");
  });
});

// ─── Regression: routability probe gates ADD of new zen/go models ───────────
// Real-world case: 2026-09-16, union-alpha appeared in BOTH live catalogs but
// the router served a marketing-site 404 (later 500) — model-doctor added it
// to models.json and pi could not use it. Now every NEW zen/go candidate is
// probed with a 1-token POST; unroutable models are skipped with a reason.

describe("regression: live-catalog routability probe gates ADD (union-alpha case)", () => {
  function zenCatalogWithUnionFixture(): string {
    return JSON.stringify({
      object: "list",
      data: [
        { id: "glm-5.2", object: "model", created: 1786480581, owned_by: "opencode" },
        { id: "union-alpha", object: "model", created: 1789580434, owned_by: "opencode" },
        { id: "zen-healthy-new", object: "model", created: 1789580434, owned_by: "opencode" },
      ],
    });
  }

  function makeResponse(status: number, body: string, contentType = "application/json") {
    return {
      ok: status >= 200 && status < 300,
      status,
      url: "fixture",
      headers: { get: (n: string) => (n.toLowerCase() === "content-type" ? contentType : null) },
      text: async () => body,
      arrayBuffer: async () => {
        const buf = Buffer.from(body, "utf8");
        return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
      },
    };
  }

  async function setupProbeEnv() {
    const dir = await makeTempDir();
    const paths: DoctorPaths = {
      modelsPath: join(dir, "models.json"),
      settingsPath: join(dir, "settings.json"),
      backupDir: dir,
      authPath: join(dir, "auth.json"),
    };
    await writeJson(paths.modelsPath, userModelsFixture());
    await writeJson(paths.settingsPath, settingsFixture());
    await writeJson(paths.authPath!, {
      opencode: { type: "api", key: "sk-zen-PROBE-FIXTURE" },
      "opencode-go": { type: "api", key: "sk-go-PROBE-FIXTURE" },
    });
    const effective = effectiveFromRaw(effectiveGoFixture() as Record<string, Array<Record<string, unknown>>>);
    const calls: string[] = [];
    const fetchFn: FetchFn = async (input: string, init?: FetchInit) => {
      calls.push(`${init?.method ?? "GET"} ${input}`);
      if (init?.method === "POST") {
        const model = (JSON.parse(String(init.body ?? "{}")) as { model?: string }).model ?? "?";
        // Router state: union-alpha unroutable everywhere; go-only-model 500s; the rest serve.
        if (model === "union-alpha") return makeResponse(404, "<!DOCTYPE html><html>SPA</html>", "text/html");
        if (model === "go-only-model") return makeResponse(500, "Internal server error", "text/plain;charset=UTF-8");
        return makeResponse(200, JSON.stringify({ choices: [{ message: { content: "ok" } }] }));
      }
      switch (input) {
        case "https://openrouter.ai/api/v1/models":
          return makeResponse(200, openRouterListFixture());
        case "https://models.dev/api.json":
          return makeResponse(200, modelsDevFixture());
        case "https://opencode.ai/zen/v1/models":
          return makeResponse(200, zenCatalogWithUnionFixture());
        case "https://opencode.ai/zen/go/v1/models":
          return makeResponse(200, goFixture());
        default:
          if (input.endsWith("/endpoints")) return makeResponse(200, openRouterEndpointsFixture());
          throw new Error(`unexpected fetch: ${input}`);
      }
    };
    return { dir, paths, effective, fetchFn, calls };
  }

  test("unroutable new models are skipped with a reason; healthy ones are added; keys never leak", async () => {
    const { paths, effective, fetchFn, calls } = await setupProbeEnv();
    const { result, report } = await runReconcile({ paths, fetchFn, effective, refCatalog });
    strictEqual(result.ok, true, result.error ?? "reconcile should succeed");

    // Both new zen models were probed against the live chat endpoint.
    assertOk(calls.includes("POST https://opencode.ai/zen/v1/chat/completions"), "zen probe issued");

    // opencode: zen-healthy-new added, union-alpha skipped (probe failed).
    const zenTotal = report.totals.find((t) => t.provider === "opencode");
    assertOk(zenTotal, "opencode total present");
    strictEqual(zenTotal!.added, 1, "only the routable new zen model is added");
    const skippedUnion = report.results.find((r) => r.provider === "opencode" && r.modelId === "union-alpha");
    assertOk(skippedUnion, "union-alpha present in results");
    strictEqual(skippedUnion!.action, "skipped");
    assertOk((skippedUnion as { reason?: string }).reason?.includes("routing probe failed"), "skip reason names the probe");

    // models.json: healthy added, unroutable absent — on BOTH providers.
    const models = JSON.parse(await readFile(paths.modelsPath, "utf8"));
    assertOk(models.providers.opencode.models.some((m: { id: string }) => m.id === "zen-healthy-new"), "zen-healthy-new added");
    assertOk(!models.providers.opencode.models.some((m: { id: string }) => m.id === "union-alpha"), "union-alpha NOT added to opencode");
    assertOk(models.providers["opencode-go"].models.some((m: { id: string }) => m.id === "kimi-k3"), "kimi-k3 (probe 200) added to opencode-go");
    assertOk(!models.providers["opencode-go"].models.some((m: { id: string }) => m.id === "go-only-model"), "go-only-model (probe 500) NOT added");

    // Probe authorization keys never appear anywhere in the output.
    const blob = JSON.stringify({ report, result }) + JSON.stringify(models);
    strictEqual(blob.includes("sk-zen-PROBE-FIXTURE"), false, "zen probe key must not leak");
    strictEqual(blob.includes("sk-go-PROBE-FIXTURE"), false, "go probe key must not leak");
  });

  test("without authPath the probe is disabled and the old behavior is preserved", async () => {
    const { dir, paths, effective, fetchFn } = await setupProbeEnv();
    const noAuthPaths: DoctorPaths = { modelsPath: paths.modelsPath, settingsPath: paths.settingsPath, backupDir: paths.backupDir };
    const { result, report } = await runReconcile({ paths: noAuthPaths, fetchFn, effective, refCatalog });
    strictEqual(result.ok, true);
    const models = JSON.parse(await readFile(noAuthPaths.modelsPath, "utf8"));
    assertOk(models.providers.opencode.models.some((m: { id: string }) => m.id === "union-alpha"), "no probe -> catalog authority adds it (unchanged behavior)");
    const zenTotal = report.totals.find((t) => t.provider === "opencode");
    strictEqual(zenTotal!.added, 2, "both new zen models added without probes");
    void dir;
  });
});

// ─── Thinking levels: cross-provider resolution in reconcile ────────────────

describe("reconcile thinking levels (openrouter efforts vs hyper)", () => {
  const KIMI_EFFORTS = ["low", "high", "max"];

  function orEffortsSource(efforts: readonly string[]): SourceResult {
    const payload = JSON.stringify({
      data: [
        {
          id: "moonshotai/kimi-k3",
          name: "Kimi K3",
          reasoning: { mandatory: false, supported_efforts: [...efforts] },
          supported_parameters: ["reasoning"],
        },
      ],
    });
    return {
      id: "openrouter",
      status: "ok",
      digest: "d-or-efforts",
      models: normalizeOpenRouterList(JSON.parse(payload)),
      fetchedAt: nowIso(),
    };
  }

  function hyperSource(efforts: readonly string[] | null): SourceResult {
    const entry =
      efforts === null
        ? { id: "kimi-k3", display_name: "Kimi K3" }
        : {
            id: "kimi-k3",
            display_name: "Kimi K3",
            reasoning: {
              effort_levels: efforts.map((v) => ({ value: v, display: v })),
              default_effort_level: efforts[0],
            },
          };
    const models = normalizeHyperList(JSON.parse(JSON.stringify({ data: [entry] })));
    return { id: "hyper", status: "ok", digest: "d-hyper", models, fetchedAt: nowIso() };
  }

  function zenSourceWithKimi(): SourceResult {
    return {
      id: "opencode-zen",
      status: "ok",
      digest: "d-zen",
      models: [{ provider: "opencode", id: "kimi-k3" }],
      fetchedAt: nowIso(),
    };
  }

  function emptyEffective(): EffectiveSnapshot {
    return effectiveFromRaw({ openrouter: [], opencode: [], "opencode-go": [] });
  }

  function userWithKimi(entry: Record<string, unknown>): ModelsConfig {
    return { providers: { opencode: { apiKey: "$KEY", models: [entry] as never } } } as ModelsConfig;
  }

  test("agreement writes thinkingLevelMap + reasoning=true ops for a zen model", () => {
    const diagnosis = makeDiagnosis(
      [orEffortsSource(KIMI_EFFORTS), zenSourceWithKimi(), hyperSource(KIMI_EFFORTS)],
      emptyEffective(),
    );
    const userModels = userWithKimi({ id: "kimi-k3", maxTokens: 65536 });
    const { operations, report } = reconcileAll(diagnosis, userModels);

    const thinkingOp = operations.find((o) => o.provider === "opencode" && o.path === "thinkingLevelMap");
    assertOk(thinkingOp, "thinkingLevelMap op planned");
    deepEqual(thinkingOp!.after, {
      off: null, minimal: null, low: "low", medium: null, high: "high", xhigh: null, max: "max",
    });
    strictEqual(thinkingOp!.source, "openrouter+hyper:efforts");

    const reasoningOp = operations.find((o) => o.provider === "opencode" && o.path === "reasoning");
    assertOk(reasoningOp, "resolved levels imply reasoning=true");
    strictEqual(reasoningOp!.after, true);

    const kimi = report.results.find((r) => r.provider === "opencode" && r.modelId === "kimi-k3");
    assertOk(kimi);
    strictEqual(kimi!.action, "updated");
  });

  test("disagreement never writes a map: the model is reported skipped (ambiguous)", () => {
    const diagnosis = makeDiagnosis(
      [orEffortsSource(KIMI_EFFORTS), zenSourceWithKimi(), hyperSource(["high", "xhigh"])],
      emptyEffective(),
    );
    // Entry matches every other plannable field (zen source carries only the id,
    // maxTokens pinned to the derived fallback), so ONLY the disputed thinking
    // map can differ.
    const userModels = userWithKimi({ id: "kimi-k3", maxTokens: 65536 });
    const { operations, report } = reconcileAll(diagnosis, userModels);

    strictEqual(
      operations.some((o) => o.provider === "opencode" && o.path === "thinkingLevelMap"),
      false,
      "disputed map must not become an operation",
    );
    const kimi = report.results.find((r) => r.provider === "opencode" && r.modelId === "kimi-k3");
    assertOk(kimi);
    strictEqual(kimi!.action, "skipped");
    assertOk((kimi as { reason?: string }).reason?.includes("ambiguous"), (kimi as { reason?: string }).reason);
  });

  test("hyper-only gaps are filled for zen models without openrouter data", () => {
    const orNoEfforts: SourceResult = {
      id: "openrouter",
      status: "ok",
      digest: "d-or",
      models: normalizeOpenRouterList(JSON.parse(JSON.stringify({ data: [{ id: "moonshotai/kimi-k3", name: "Kimi K3" }] }))),
      fetchedAt: nowIso(),
    };
    const diagnosis = makeDiagnosis([orNoEfforts, zenSourceWithKimi(), hyperSource(KIMI_EFFORTS)], emptyEffective());
    const userModels = userWithKimi({ id: "kimi-k3", maxTokens: 65536 });
    const { operations } = reconcileAll(diagnosis, userModels);
    const thinkingOp = operations.find((o) => o.provider === "opencode" && o.path === "thinkingLevelMap");
    assertOk(thinkingOp, "hyper fills the gap");
    strictEqual(thinkingOp!.source, "hyper:efforts");
  });
});
