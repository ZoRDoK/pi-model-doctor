import { describe, test, beforeEach } from "node:test";
import { strictEqual, notStrictEqual, ok as assertOk } from "node:assert/strict";
import { writeFile, mkdir, rm, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import factory, { __internals } from "../index.ts";
import {
  mockFetch,
  modelsDevFixture,
  openRouterEndpointsFixture,
  openRouterListFixture,
  effectiveGoFixture,
  userModelsFixture,
  settingsFixture,
  findSecrets,
  zenFixture,
} from "./helpers.ts";

// ─── Minimal mock of the pi ExtensionAPI surface the entry touches ───────────

interface RegisteredCommand {
  name: string;
  description?: string;
  handler: (args: string, ctx: unknown) => Promise<void>;
}

function createMockApi() {
  const commands: RegisteredCommand[] = [];
  const sentMessages: Array<{ msg: Record<string, unknown>; opts: Record<string, unknown> | undefined }> = [];
  const api = {
    commands,
    sentMessages,
    registerCommand(name: string, options: Omit<RegisteredCommand, "name">): void {
      commands.push({ name, ...options });
    },
    sendMessage(msg: Record<string, unknown>, opts?: Record<string, unknown>): void {
      sentMessages.push({ msg, opts });
    },
    on: () => {},
  };
  return { api, commands };
}

/** Full Markdown report text captured through the sendMessage mock. */
function reportOf(api: { sentMessages: Array<{ msg: Record<string, unknown> }> }): string {
  return api.sentMessages.map((s) => String(s.msg.content ?? "")).join("\n\n");
}

// ─── temp HOME, mocked fetch, mock ctx (flat registry like 0.84.1) ───────────

async function withTempHome<T>(fn: (home: string) => Promise<T>): Promise<T> {
  const home = join(tmpdir(), `md-cmd-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  await mkdir(join(home, ".pi", "agent"), { recursive: true });
  await writeFile(join(home, ".pi", "agent", "models.json"), JSON.stringify(userModelsFixture(), undefined, 2) + "\n");
  await writeFile(join(home, ".pi", "agent", "settings.json"), JSON.stringify(settingsFixture(), undefined, 2) + "\n");
  const oldHome = process.env.HOME;
  const oldAgent = process.env.PI_CODING_AGENT_DIR;
  process.env.HOME = home;
  process.env.PI_CODING_AGENT_DIR = join(home, ".pi", "agent");
  try {
    return await fn(home);
  } finally {
    process.env.HOME = oldHome;
    if (oldAgent === undefined) delete process.env.PI_CODING_AGENT_DIR;
    else process.env.PI_CODING_AGENT_DIR = oldAgent;
    await rm(home, { recursive: true, force: true });
  }
}

interface CtxLogs {
  __logs: string[];
}

function mockCtx(): unknown {
  const logs: string[] = [];
  const effective = effectiveGoFixture();
  // Flat catalog matching pi 0.84.1 ModelRegistry.getAll(): every model has `provider`.
  const all: Array<Record<string, unknown>> = [];
  for (const [provider, list] of Object.entries(effective)) {
    for (const m of list) all.push({ ...(m as Record<string, unknown>), provider });
  }
  all.push({ id: "gpt-5.6-sol", name: "GPT 5.6 Sol", provider: "openai-codex", api: "x", reasoning: true, input: ["text"], cost: {}, contextWindow: 1, maxTokens: 1 });
  all.push({ id: "glm-5.2", name: "GLM", provider: "zai", api: "x", reasoning: true, input: ["text"], cost: {}, contextWindow: 1, maxTokens: 1 });
  return {
    mode: "tui",
    hasUI: true,
    cwd: "/tmp",
    ui: {
      notify: (msg: string, _type?: string) => { logs.push(msg); },
    },
    modelRegistry: {
      getAll() { return all; },
      find(provider: string, id: string) { return all.find((m) => m.provider === provider && m.id === id); },
      hasConfiguredAuth(_m: unknown) { return true; },
    },
    __logs: logs,
  };
}

function fetchRoutes() {
  return mockFetch({
    "https://openrouter.ai/api/v1/models": { body: openRouterListFixture() },
    "https://opencode.ai/zen/v1/models": { body: zenFixture() },
    "https://models.dev/api.json": { body: modelsDevFixture() },
    "https://opencode.ai/zen/go/v1/models": { status: 404 },
    "https://hyper.charm.land/v1/models": { status: 404 },
    // The one-command flow endpoint-confirms limits for every managed
    // openrouter model; agreeing evidence keeps an already-matching entry
    // genuinely UNCHANGED.
    "https://openrouter.ai/api/v1/models/openai/gpt-test/endpoints": {
      body: openRouterEndpointsFixture({ context: [128000], max: [16384] }),
    },
  });
}

async function readFileSha(path: string): Promise<string> {
  const { createHash } = await import("node:crypto");
  const bytes = await readFile(path);
  return createHash("sha256").update(bytes).digest("hex");
}

// ─── tests ───────────────────────────────────────────────────────────────────

describe("command registration", () => {
  test("registers exactly ONE command named model-doctor with a no-arg description", () => {
    const { api, commands } = createMockApi();
    factory(api as unknown as never);
    strictEqual(commands.length, 1, "exactly one command");
    strictEqual(commands[0]!.name, "model-doctor");
    assertOk(commands[0]!.description?.length, "command has a description");
    // No subcommands/arguments advertised.
    assertOk(/no subcommand|no argument/i.test(commands[0]!.description!), "description states no subcommands/args");
  });

  test("help text describes a single one-shot command, no check/plan/apply", () => {
    const h = __internals.helpText();
    assertOk(/\/model-doctor\b/.test(h));
    assertOk(/one-shot|once|single/i.test(h), "describes a one-shot command");
    strictEqual(/\/model-doctor check\b/.test(h), false, "no check subcommand");
    strictEqual(/\/model-doctor plan\b/.test(h), false, "no plan subcommand");
    strictEqual(/\/model-doctor apply\b/.test(h), false, "no apply subcommand");
  });
});

describe("single /model-doctor command", () => {
  let originalFetch: typeof globalThis.fetch;
  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });
  test("teardown restores fetch", () => {
    globalThis.fetch = originalFetch;
  });

  test("reconciles all providers: writes models.json once, preserves creds/baseUrl/headers/unrelated, no secrets", async () => {
    await withTempHome(async () => {
      const { api, commands } = createMockApi();
      factory(api as unknown as never);
      const { fetch } = fetchRoutes();
      globalThis.fetch = fetch as unknown as typeof globalThis.fetch;
      const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");

      const before = await readFileSha(modelsPath);
      const ctx = mockCtx();
      await commands[0]!.handler("", ctx); // no args
      const after = await readFileSha(modelsPath);
      notStrictEqual(before, after, "models.json changed after /model-doctor");

      const out = reportOf(api);
      assertOk(out.length > 0, "report sent as a chat message (markdown)");
      strictEqual(api.sentMessages[0]!.opts?.triggerTurn, false, "report must not trigger a turn");
      strictEqual(findSecrets(out).length, 0, "no secrets in output");
      assertOk(out.includes("| provider | added | updated | skipped | unchanged | source |"), "output has provider totals table");
      // No warnings block, no provenance noise (user-facing report is compact).
      strictEqual(out.includes("warnings ("), false, "warnings block omitted");
      strictEqual(out.includes("skipped:"), false, "skipped reasons omitted; names only");
      strictEqual(out.includes("credentials, base URLs"), false, "verbose footer omitted");

      const m = JSON.parse((await readFile(modelsPath, "utf8")));
      // Credentials / base URL / headers preserved.
      strictEqual(m.providers.openrouter.apiKey, "$OPENROUTER_API_KEY");
      strictEqual(m.providers.unrelated.models[0].id, "keepme");
      strictEqual(m.providers.openrouter.apiKey, "$OPENROUTER_API_KEY");
      // Unrelated provider untouched.
      assertOk(m.providers.unrelated);
      strictEqual(m.providers.unrelated.models[0].id, "keepme");
      // Newly added models from the official catalog (absent from the registry).
      assertOk(m.providers.openrouter.models.some((x: { id: string }) => x.id === "deepseek/ds-alias"), "added missing openrouter model");
      assertOk(m.providers.openrouter.models.some((x: { id: string }) => x.id === "free/zero-cost"), "added missing openrouter model");
    });
    globalThis.fetch = originalFetch;
  });

  test("does not dump unchanged models: a matching managed model is counted only in totals", async () => {
    await withTempHome(async () => {
      // Rewrite models.json so openrouter/openai/gpt-test ALREADY matches the
      // official source exactly -> it must be UNCHANGED (totals only), never
      // listed as added/updated/skipped.
      const cfg = userModelsFixture() as { providers: Record<string, { models?: Array<Record<string, unknown>> }> };
      cfg.providers.openrouter!.models = [{
        id: "openai/gpt-test",
        name: "GPT Test",
        reasoning: true,
        input: ["text", "image"],
        contextWindow: 128000,
        maxTokens: 16384,
        cost: { input: 5, output: 15, cacheRead: 1.25, cacheWrite: 0 },
      }];
      await writeFile(join(process.env.PI_CODING_AGENT_DIR!, "models.json"), JSON.stringify(cfg, undefined, 2) + "\n", "utf8");

      const { api, commands } = createMockApi();
      factory(api as unknown as never);
      const { fetch } = fetchRoutes();
      globalThis.fetch = fetch as unknown as typeof globalThis.fetch;
      const ctx = mockCtx();
      await commands[0]!.handler("", ctx);
      const out = reportOf(api);
      // The matching model is NOT individually listed anywhere (tables/list only name changed models).
      strictEqual(out.includes("openai/gpt-test"), false, "unchanged model not listed anywhere");
      // ...but the openrouter provider total row reports >=1 unchanged (5th column).
      const orRow = out.split("\n").find((l) => l.startsWith("| openrouter |"));
      assertOk(orRow, "openrouter totals row present");
      const cols = orRow!.split("|").map((c) => c.trim());
      assertOk(Number(cols[5] ?? 0) >= 1, `openrouter unchanged count >= 1 (row: ${orRow})`);
    });
    globalThis.fetch = originalFetch;
  });

  test("per-provider source failure is isolated: opencode-go 404 does not corrupt others", async () => {
    await withTempHome(async () => {
      const { api, commands } = createMockApi();
      factory(api as unknown as never);
      const { fetch } = fetchRoutes(); // opencode-go returns 404
      globalThis.fetch = fetch as unknown as typeof globalThis.fetch;
      const ctx = mockCtx();
      await commands[0]!.handler("", ctx);
      const out = reportOf(api);
      // openrouter/opencode still processed (added/updated); opencode-go-2 mirrored.
      assertOk(/\| openrouter \|[^\n]*\| ok \|/.test(out), "openrouter processed");
      assertOk(/\| opencode \|[^\n]*\| ok \|/.test(out), "opencode processed");
      const m = JSON.parse((await readFile(join(process.env.PI_CODING_AGENT_DIR!, "models.json"), "utf8")));
      // No invented opencode-go models (source unavailable -> check-only).
      strictEqual(Array.isArray(m.providers["opencode-go"]?.models) ? m.providers["opencode-go"].models.length : 0, 0, "no invented opencode-go models");
    });
    globalThis.fetch = originalFetch;
  });

  test("a second invocation with no further drift writes nothing new (idempotent)", async () => {
    await withTempHome(async () => {
      const { api, commands } = createMockApi();
      factory(api as unknown as never);
      globalThis.fetch = fetchRoutes().fetch as unknown as typeof globalThis.fetch;
      const ctx1 = mockCtx();
      await commands[0]!.handler("", ctx1);
      const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
      const after1 = await readFileSha(modelsPath);
      // Second run: catalog already reconciled -> 0 changes.
      globalThis.fetch = fetchRoutes().fetch as unknown as typeof globalThis.fetch;
      const ctx2 = mockCtx();
      await commands[0]!.handler("", ctx2);
      const after2 = await readFileSha(modelsPath);
      strictEqual(after1, after2, "second invocation is a no-op write (idempotent)");
      const out2 = reportOf(api);
      assertOk(/applied 0 change/.test(out2), "second run applies 0 changes");
    });
    globalThis.fetch = originalFetch;
  });
});

describe("argument rejection + output bounding", () => {
  test("any non-empty argument is rejected WITHOUT mutation (reviewer 820bdad3)", async () => {
    const orig = globalThis.fetch;
    await withTempHome(async () => {
      const { api, commands } = createMockApi();
      factory(api as unknown as never);
      globalThis.fetch = fetchRoutes().fetch as unknown as typeof globalThis.fetch;
      const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
      const before = await readFileSha(modelsPath);
      const ctx = mockCtx();
      await commands[0]!.handler("check", ctx); // rejected: non-empty arg
      const after = await readFileSha(modelsPath);
      strictEqual(before, after, "no mutation when an argument is supplied");
      const out = (ctx as unknown as CtxLogs).__logs.join("\n");
      assertOk(/takes no arguments/i.test(out), "rejection message printed");
    });
    globalThis.fetch = orig;
  });

  test("a refused apply surfaces its findings in the report (allowlist refusal was opaque)", () => {
    // Production bug (2026-08-20): applyPlan refused with "allowlist verification
    // failed" and buildResultReport dropped result.findings, so the actual
    // violation paths were invisible. The refusal must list them.
    const refused = {
      ok: false as const,
      applied: 0,
      backups: [] as readonly string[],
      error: "allowlist verification failed; no changes written",
      findings: [
        {
          provider: "*",
          severity: "error" as const,
          code: "allowlist_violation",
          message:
            "apply would touch non-allowlisted paths: providers.opencode-go would be created (new provider block)",
        },
      ],
    };
    const text = __internals.buildResultReport({ results: [], totals: [], warnings: [] }, refused);
    assertOk(text.includes("update refused"), "refusal headline present");
    assertOk(text.includes("reasons (1)"), "findings count present");
    assertOk(text.includes("allowlist_violation"), "finding code present");
    assertOk(text.includes("providers.opencode-go would be created"), "violation path present");
  });

  test("output is bounded: a multi-megabyte report is capped and full report written to a 0600 file", async () => {
    // 4,000,000 untrusted chars must NOT produce a 4,000,000-char inline report.
    const tmp = join(tmpdir(), `md-bound-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    await mkdir(tmp, { recursive: true });
    const oldAgent = process.env.PI_CODING_AGENT_DIR;
    process.env.PI_CODING_AGENT_DIR = tmp;
    try {
      const huge = "x".repeat(4_000_000);
      const { shown } = await __internals.boundReport(huge);
      assertOk(shown.length < huge.length, "inline output capped");
      assertOk(shown.length <= __internals.MAX_REPORT_CHARS + 400, "within bound + marker");
      assertOk(/truncated/i.test(shown), "truncation marker present");
      // a full report file was written (mode 0600) with the complete content
      const { readdir, stat, readFile } = await import("node:fs/promises");
      const files = (await readdir(tmp)).filter((f) => f.startsWith("model-doctor-report-"));
      assertOk(files.length === 1, "exactly one full-report file written");
      strictEqual((await stat(join(tmp, files[0]!))).mode & 0o777, 0o600, "report file is 0600");
      strictEqual((await readFile(join(tmp, files[0]!), "utf8")).length, huge.length, "full content in file");
    } finally {
      if (oldAgent === undefined) delete process.env.PI_CODING_AGENT_DIR;
      else process.env.PI_CODING_AGENT_DIR = oldAgent;
      await rm(tmp, { recursive: true, force: true });
    }
  });
});
