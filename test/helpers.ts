/**
 * Test helpers: temp dirs, mocked fetch, fixtures, secret-safety scanning.
 *
 * Every test uses mkdtemp for writes and a fully mocked fetch. No real network
 * and no writes outside the temp dir.
 */
import { mkdtemp, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { FetchFn, FetchInit, FetchResponse } from "../types.ts";

export async function makeTempDir(prefix = "md-test-"): Promise<string> {
  return mkdtemp(join(tmpdir(), prefix));
}

export async function writeJson(path: string, value: unknown): Promise<void> {
  await mkdir(join(path, ".."), { recursive: true });
  await writeFile(path, JSON.stringify(value, undefined, 2) + "\n", "utf8");
}

export interface MockRoute {
  readonly status?: number;
  readonly body?: string;
  readonly contentType?: string;
  readonly redirectTo?: string;
  readonly networkError?: string;
}

export interface MockFetch {
  fetch: FetchFn;
  calls: string[];
}

/**
 * Build a mocked fetch keyed by full URL. Honors redirect:"error" (throws),
 * networkError, status, contentType, and body. Defaults to JSON 200.
 */
export function mockFetch(routes: Record<string, MockRoute>): MockFetch {
  const calls: string[] = [];
  const fetch: FetchFn = async (input: string, init?: FetchInit): Promise<FetchResponse> => {
    calls.push(input);
    const route = routes[input];
    if (!route) {
      throw new Error(`unexpected fetch: ${input}`);
    }
    if (route.networkError) {
      throw new Error(route.networkError);
    }
    if (route.redirectTo && init?.redirect === "error") {
      throw new Error(`redirect not allowed -> ${route.redirectTo}`);
    }
    const status = route.status ?? 200;
    const ct = route.contentType ?? "application/json";
    const body = route.body ?? "";
    return {
      ok: status >= 200 && status < 300,
      status,
      url: input,
      headers: {
        get(name: string): string | null {
          return name.toLowerCase() === "content-type" ? ct : null;
        },
      },
      async text(): Promise<string> {
        return body;
      },
      async arrayBuffer(): Promise<ArrayBuffer> {
        const buf = Buffer.from(body, "utf8");
        return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
      },
    };
  };
  return { fetch, calls };
}

// ─── Fixtures ────────────────────────────────────────────────────────────────

/** A small OpenRouter /models payload with two models. */
export function openRouterListFixture(extra?: Record<string, unknown>): string {
  const data = [
    {
      id: "openai/gpt-test",
      name: "GPT Test",
      canonical_slug: "openai/gpt-test",
      context_length: 128000,
      top_provider: { max_completion_tokens: 16384 },
      pricing: { prompt: "0.000005", completion: "0.000015", input_cache_read: "0.00000125" },
      reasoning: { mandatory: false },
      supported_parameters: ["reasoning", "tools"],
      architecture: { input_modalities: ["text", "image"], modality: "text->text" },
    },
    {
      id: "deepseek/ds-alias",
      name: "DS Alias",
      canonical_slug: "deepseek/ds-canonical",
      context_length: 64000,
      top_provider: { max_completion_tokens: 8192 },
      pricing: { prompt: "0.0000002", completion: "0.0000008", overrides: { ">1m": { prompt: "0.0000001" } } },
      supported_parameters: ["temperature"],
      architecture: { input_modalities: ["text"], modality: "text->text" },
    },
    {
      id: "free/zero-cost",
      name: "Free Zero",
      context_length: 32000,
      pricing: { prompt: "0", completion: "0" },
      architecture: { input_modalities: ["text"], modality: "text->text" },
    },
  ];
  return JSON.stringify({ data, ...extra });
}

/** OpenRouter per-model endpoints payload (varying limits -> ambiguity). */
export function openRouterEndpointsFixture(opts?: { context?: number[]; max?: number[] }): string {
  const ctx = opts?.context ?? [100000, 120000];
  const max = opts?.max ?? [16000, 16000];
  const endpoints = ctx.map((c, i) => ({ context_length: c, max_completion_tokens: max[i] }));
  return JSON.stringify({ data: { id: "openai/gpt-test", name: "GPT Test", endpoints } });
}

/** OpenCode Go live catalog ({data:[{id,...}]}). Sibling of Zen /zen/v1/models;
 *  distinct provider — must never be confused with the Zen catalog. */
export function goFixture(): string {
  return JSON.stringify({
    object: "list",
    data: [
      { id: "glm-5.2", object: "model", created: 1786535914, owned_by: "opencode" },
      { id: "kimi-k3", object: "model", created: 1786535914, owned_by: "opencode" },
      { id: "go-only-model", object: "model", created: 1786535914, owned_by: "opencode" },
    ],
  });
}

/** OpenCode Zen live catalog ({data:[{id,...}]}). Only these IDs are routable
 *  opencode offerings; models.dev-only slugs must NEVER be emitted as opencode. */
export function zenFixture(): string {
  return JSON.stringify({
    object: "list",
    data: [
      { id: "glm-5.2", object: "model", created: 1786480581, owned_by: "opencode" },
      { id: "gpt-test", object: "model", created: 1786480581, owned_by: "opencode" },
      { id: "zen-only-model", object: "model", created: 1786480581, owned_by: "opencode" },
    ],
  });
}

/** models.dev org-view fixture ({org:{models:{slug:{…}}}}, the /api.json shape).
 *  Preferred orgs (opencode/opencode-go) absent here on purpose. */
export function modelsDevFixture(): string {
  return JSON.stringify({
    openai: {
      models: {
        "openai/gpt-test": {
          id: "openai/gpt-test",
          name: "GPT Test Canonical",
          reasoning: true,
          modalities: { input: ["text", "image"], output: ["text"] },
          limit: { context: 128000, output: 16384 },
        },
      },
    },
    zhipuai: {
      models: {
        "zhipuai/glm-5.2": {
          id: "zhipuai/glm-5.2",
          name: "GLM-5.2",
          reasoning: true,
          modalities: { input: ["text"], output: ["text"] },
          limit: { context: 1000000, output: 131072 },
        },
      },
    },
    deepseek: {
      models: {
        "deepseek/ds-alias": {
          id: "deepseek/ds-alias",
          name: "DS Alias",
          reasoning: false,
          modalities: { input: ["text"], output: ["text"] },
          limit: { context: 64000, output: 8192 },
        },
      },
    },
  });
}

/** Charm Hyper catalog fixture ({data:[{id, display_name, reasoning:{effort_levels}}]}). */
export function hyperListFixture(): string {
  return JSON.stringify({
    object: "list",
    data: [
      {
        id: "glm-5.2",
        display_name: "GLM-5.2",
        reasoning: {
          effort_levels: [{ value: "high", display: "High" }, { value: "xhigh", display: "X-High" }],
          default_effort_level: "high",
        },
      },
      {
        id: "kimi-k3",
        display_name: "Kimi K3",
        reasoning: {
          effort_levels: [{ value: "low", display: "Low" }, { value: "high", display: "High" }, { value: "max", display: "Max" }],
          default_effort_level: "high",
        },
      },
      { id: "empty-levels-model", display_name: "Empty Levels", reasoning: { effort_levels: [] } },
      { id: "plain-model", display_name: "Plain" },
    ],
  });
}

/** A user models.json with managed providers, an unrelated provider, and overrides. */
export function userModelsFixture(opts?: { withSecret?: boolean }): unknown {
  return {
    providers: {
      openrouter: {
        apiKey: opts?.withSecret ? "sk-secret-FIXTURE-DO-NOT-LEAK" : "$OPENROUTER_API_KEY",
        models: [
          {
            id: "openai/gpt-test",
            name: "GPT Test OLD",
            reasoning: false,
            input: ["text"],
            contextWindow: 64000,
            maxTokens: 4096,
            cost: { input: 1, output: 3, cacheRead: 0.25, cacheWrite: 0 },
          },
        ],
        modelOverrides: { "openai/gpt-test": { contextWindow: 60000 } },
      },
      opencode: {
        apiKey: "$OPENCODE_KEY",
        models: [
          {
            id: "glm-5.2",
            name: "GLM-5.2 OLD",
            reasoning: false,
            input: ["text"],
            contextWindow: 200000,
            maxTokens: 8192,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          },
        ],
      },
      "opencode-go": {
        modelOverrides: { "glm-5.2": { thinkingLevelMap: { high: "high" } } },
      },
      unrelated: {
        apiKey: "sk-unrelated-FIXTURE-SECRET",
        models: [{ id: "keepme", name: "Keep Me", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 1000, maxTokens: 100 }],
      },
    },
  };
}

/** Effective opencode-go snapshot fixture. */
export function effectiveGoFixture(): Record<string, unknown[]> {
  return {
    "opencode-go": [
      {
        id: "glm-5.2",
        name: "GLM-5.2",
        api: "anthropic-messages",
        reasoning: true,
        input: ["text", "image"],
        contextWindow: 1000000,
        maxTokens: 131072,
        cost: { input: 0.3, output: 1.2, cacheRead: 0.06, cacheWrite: 0 },
        thinkingLevelMap: { high: "high", xhigh: "xhigh" },
        compat: { supportsLongCacheRetention: true },
      },
    ],
    openrouter: [],
    opencode: [],
      };
}

/** A user settings.json with model references to validate. */
export function settingsFixture(): unknown {
  return {
    defaultProvider: "openai-codex",
    defaultModel: "gpt-5.6-sol",
    workingVibeModel: "opencode/glm-5.2:max",
    enabledModels: ["opencode-go/glm-5.2", "zai/glm-5.2", "missing/ghost-model", "openai-codex/gpt-*"],
    subagents: {
      agentOverrides: {
        planner: {
          model: "openai-codex/gpt-5.6-sol:max",
          fallbackModels: ["zai/glm-5.2:xhigh", "anthropic/claude-sonnet-5"],
        },
      },
    },
  };
}

/** Scan an arbitrary text blob for fixture secret markers; returns hits. */
export const SECRET_MARKERS = [
  "sk-secret-FIXTURE",
  "sk-go2-FIXTURE-SECRET",
  "sk-unrelated-FIXTURE-SECRET",
  "FIXTURE-DO-NOT-LEAK",
  "sk-go2-FIXTURE",
];

export function findSecrets(text: string): string[] {
  return SECRET_MARKERS.filter((m) => text.includes(m));
}
