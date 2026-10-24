import { describe, test, beforeEach } from "node:test";
import { strictEqual, deepStrictEqual, ok as assertOk } from "node:assert/strict";
import { writeFile, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  listingUrl,
  readAuthKeys,
  genericCandidates,
  syncGenericProviders,
} from "../generic.ts";
import { runReconcile } from "../doctor.ts";
import type { ModelsConfig, FetchFn, FetchInit, FetchResponse } from "../types.ts";

// ─── listingUrl ───────────────────────────────────────────────────────────────

describe("listingUrl", () => {
  test("openai: versioned base gets /models, bare base gets /v1/models", () => {
    strictEqual(listingUrl("openai", "https://api.deepseek.com"), "https://api.deepseek.com/v1/models");
    strictEqual(listingUrl("openai", "https://api.openai.com/v1/"), "https://api.openai.com/v1/models");
    strictEqual(listingUrl("openai", "http://localhost:11434/v1"), "http://localhost:11434/v1/models");
    strictEqual(listingUrl("openai", "https://api.z.ai/api/coding/paas/v4"), "https://api.z.ai/api/coding/paas/v4/models");
  });
  test("anthropic and google paths", () => {
    strictEqual(listingUrl("anthropic", "https://api.anthropic.com"), "https://api.anthropic.com/v1/models");
    strictEqual(listingUrl("anthropic", "https://x.example/v1"), "https://x.example/v1/models");
    strictEqual(listingUrl("google", "https://generativelanguage.googleapis.com"), "https://generativelanguage.googleapis.com/v1beta/models");
  });
});

// ─── readAuthKeys ─────────────────────────────────────────────────────────────

describe("readAuthKeys", () => {
  test("extracts plain keys, ignores oauth entries and malformed json", () => {
    const keys = readAuthKeys(JSON.stringify({
      deepseek: { key: "sk-ds-TEST-KEY", type: "api" },
      "openai-codex": { access: "oa", refresh: "r", type: "oauth" },
      empty: { type: "api" },
    }));
    deepStrictEqual(keys, { deepseek: "sk-ds-TEST-KEY" });
    deepStrictEqual(readAuthKeys("not json"), {});
    deepStrictEqual(readAuthKeys(undefined), {});
  });
});

// ─── genericCandidates ────────────────────────────────────────────────────────

function candidateModels(): ModelsConfig {
  return {
    providers: {
      openrouter: { apiKey: "$OR_KEY", models: [{ id: "x" }] }, // managed -> skipped
      legacy: { models: [{ id: "y" }] }, // no api -> skipped
      deepseek: {
        baseUrl: "https://api.deepseek.com",
        api: "openai-completions",
        apiKey: "$DEEPSEEK_KEY",
        models: [{ id: "deepseek-chat" }],
      },
      zai: {
        // provider-level api + model-level baseUrl (inherited transport)
        api: "openai-completions",
        models: [{ id: "glm-5.3", api: "openai-completions", baseUrl: "https://api.z.ai/api/coding/paas/v4" }],
      },
      inactive: { baseUrl: "https://a.example", api: "openai-completions", models: [{ id: "z" }] },
    },
  };
}

describe("genericCandidates", () => {
  const active = (p: string) => p !== "inactive";
  test("resolves protocol + transport, skips managed/inactive/unprotocoled", () => {
    const c = genericCandidates(candidateModels(), {}, active);
    deepStrictEqual(c.map((x) => x.provider).sort(), ["deepseek", "zai"]);
    const zai = c.find((x) => x.provider === "zai")!;
    assertOk(zai.baseUrl.includes("api.z.ai"), "model-level baseUrl inherited");
    strictEqual(zai.apiKey, undefined, "no key declared -> undefined (auth.json resolution happens at fetch)");
  });
});

// ─── syncGenericProviders ─────────────────────────────────────────────────────

function jsonResponse(body: unknown): FetchResponse {
  return {
    ok: true,
    status: 200,
    url: "https://x.example/models",
    headers: { get: (n: string) => (n.toLowerCase() === "content-type" ? "application/json" : null) },
    text: async () => JSON.stringify(body),
  } as unknown as FetchResponse;
}

const fetchedUrls: string[] = [];
const fetchRecorder = (route: (url: string) => FetchResponse | null): FetchFn => {
  return (async (input: string, init?: FetchInit) => {
    fetchedUrls.push(input);
    const hit = route(input);
    if (hit) return hit;
    void init;
    throw new Error("unexpected fetch " + input);
  }) as FetchFn;
};

describe("syncGenericProviders", () => {
  beforeEach(() => {
    fetchedUrls.length = 0;
  });

  test("adds genuinely new ids, leaves existing entries alone, never logs keys", async () => {
    const fetchFn = fetchRecorder((url) => {
      if (url === "https://api.deepseek.com/v1/models") {
        return jsonResponse({ data: [{ id: "deepseek-chat" }, { id: "deepseek-v4-new" }] });
      }
      return null;
    });
    const res = await syncGenericProviders({
      userModels: candidateModels(),
      authKeys: { deepseek: "sk-ds-TEST-KEY" },
      activeProvider: (p) => p !== "inactive",
      fetchFn,
    });
    strictEqual(res.totals.find((t) => t.provider === "deepseek")?.sourceStatus, "ok");
    strictEqual(res.totals.find((t) => t.provider === "deepseek")?.added, 1);
    strictEqual(res.results.filter((r) => r.action === "added").length, 1);
    assertOk(res.results.some((r) => r.modelId === "deepseek-v4-new"));
    // added model carries inherited transport
    const added = res.results.find((r) => r.modelId === "deepseek-v4-new")!;
    strictEqual(added.characteristics?.api, "openai-completions");
    strictEqual(added.characteristics?.baseUrl, "https://api.deepseek.com");
    // Authorization header carried the key, but no output contains it
    assertOk(fetchedUrls.includes("https://api.deepseek.com/v1/models"));
    strictEqual(JSON.stringify(res).includes("sk-ds-TEST-KEY"), false, "no key in sync result");
  });

  test("google listing maps limits onto added models", async () => {
    const user: ModelsConfig = {
      providers: {
        gemini: {
          baseUrl: "https://generativelanguage.googleapis.com",
          api: "google-generative-ai",
          apiKey: "$GEMINI_TEST_KEY",
          models: [],
        },
      },
    };
    process.env.GEMINI_TEST_KEY = "gem-test-key";
    try {
      const fetchFn = fetchRecorder((url) => {
        if (url.startsWith("https://generativelanguage.googleapis.com/v1beta/models")) {
          return jsonResponse({
            models: [
              { name: "models/gemini-3.1-pro", displayName: "Gemini 3.1 Pro", inputTokenLimit: 1048576, outputTokenLimit: 65536, inputModalities: ["text", "image"], supportedGenerationMethods: ["generateContent"] },
              { name: "models/embedding-001", supportedGenerationMethods: ["embedContent"] },
            ],
          });
        }
        return null;
      });
      const res = await syncGenericProviders({
        userModels: user,
        authKeys: {},
        activeProvider: () => true,
        fetchFn,
      });
      const added = res.results.find((r) => r.modelId === "gemini-3.1-pro")!;
      assertOk(added, "gemini model added");
      strictEqual(added.characteristics?.contextWindow, 1048576);
      strictEqual(added.characteristics?.maxTokens, 65536);
      deepStrictEqual(added.characteristics?.input, ["text", "image"]);
      strictEqual(res.results.some((r) => r.modelId === "embedding-001"), false, "non-generateContent models filtered");
      strictEqual(JSON.stringify(res).includes("gem-test-key"), false, "key never in output");
    } finally {
      delete process.env.GEMINI_TEST_KEY;
    }
  });

  test("unreachable provider -> unavailable total, no ops, warning recorded", async () => {
    const fetchFn = (async () => {
      throw new Error("boom");
    }) as unknown as FetchFn;
    const res = await syncGenericProviders({
      userModels: candidateModels(),
      authKeys: {},
      activeProvider: () => true,
      fetchFn,
    });
    strictEqual(res.operations.length, 0);
    assertOk(res.totals.every((t) => t.sourceStatus === "unavailable"));
    assertOk(res.warnings.length > 0);
    assertOk(res.warnings.some((w) => w.startsWith("generic_source_unavailable: ")), "warning recorded");
  });
});

// ─── end-to-end via runReconcile (temp home) ─────────────────────────────────

describe("runReconcile integrates generic providers", () => {
  test("a configured deepseek-like provider gets new models written to models.json", async () => {
    const home = join(tmpdir(), "md-generic-" + Date.now() + "-" + Math.random().toString(36).slice(2));
    const agentDir = join(home, ".pi", "agent");
    await mkdir(agentDir, { recursive: true });
    const modelsJson = {
      providers: {
        deepseek: {
          baseUrl: "https://api.deepseek.com",
          api: "openai-completions",
          apiKey: "sk-ds-FIXTURE-PLAIN",
          models: [{ id: "deepseek-chat", name: "Chat", api: "openai-completions", contextWindow: 64000, maxTokens: 8192 }],
        },
      },
    };
    await writeFile(join(agentDir, "models.json"), JSON.stringify(modelsJson, undefined, 2), "utf8");
    await writeFile(join(agentDir, "settings.json"), "{}", "utf8");
    await writeFile(join(agentDir, "auth.json"), JSON.stringify({ zai: { key: "sk-zai-FIXTURE", type: "api" } }), "utf8");
    const oldAgent = process.env.PI_CODING_AGENT_DIR;
    process.env.PI_CODING_AGENT_DIR = agentDir;
    try {
      const fetchFn = fetchRecorder((url) => {
        if (url === "https://api.deepseek.com/v1/models") {
          return jsonResponse({ data: [{ id: "deepseek-chat" }, { id: "deepseek-reasoner-v4" }] });
        }
        return null;
      });
      const outcome = await runReconcile({
        paths: {
          modelsPath: join(agentDir, "models.json"),
          settingsPath: join(agentDir, "settings.json"),
          backupDir: agentDir,
          authPath: join(agentDir, "auth.json"),
        },
        fetchFn,
        effective: { providers: { openrouter: [], opencode: [], "opencode-go": [] } },
        refCatalog: {
          modelIds: (p) => (p === "deepseek" ? ["deepseek-chat"] : []),
          hasAuth: () => true,
        },
      });
      assertOk(outcome.result.ok, "apply ok");
      assertOk(outcome.result.applied >= 1, "wrote the new model");
      const total = outcome.report.totals.find((t) => t.provider === "deepseek");
      assertOk(total, "deepseek total present");
      strictEqual(total?.added, 1);
      strictEqual(total?.sourceStatus, "ok");
      const written = JSON.parse(await (await import("node:fs/promises")).readFile(join(agentDir, "models.json"), "utf8"));
      const added = written.providers.deepseek.models.find((m: { id: string }) => m.id === "deepseek-reasoner-v4");
      assertOk(added, "new model present in models.json");
      strictEqual(added.api, "openai-completions", "transport inherited");
      strictEqual(added.baseUrl, "https://api.deepseek.com");
      strictEqual(JSON.stringify(written).includes("sk-zai-FIXTURE"), false, "auth.json key never copied");
    } finally {
      if (oldAgent === undefined) delete process.env.PI_CODING_AGENT_DIR;
      else process.env.PI_CODING_AGENT_DIR = oldAgent;
      await rm(home, { recursive: true, force: true });
    }
  });
});
