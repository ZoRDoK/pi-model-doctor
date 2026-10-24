import { afterEach, describe, test } from "node:test";
import { strictEqual, notStrictEqual, ok as assertOk, rejects, throws, deepStrictEqual } from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  applyOpenRouterEndpoints,
  fetchSource,
  fetchOpenCodeGoSource,
  normalizeHyperList,
  normalizeOpenRouterList,
  normalizeModelsDev,
  normalizeZenLive,
  parseLiveCatalogIds,
  fetchOpenCodeZenSource,
  openRouterDetailUrl,
  probeOpenCodeModel,
  safeFetch,
  sha256Hex,
  MAX_RESPONSE_BYTES,
  SOURCE_ENDPOINTS,
} from "../sources.ts";
import type { NormalizedModel } from "../types.ts";
import {
  mockFetch,
  openRouterListFixture,
  openRouterEndpointsFixture,
  modelsDevFixture,
  hyperListFixture,
  zenFixture,
  goFixture,
} from "./helpers.ts";

const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe("openrouter normalization", () => {
  test("missing field is not turned into a zero/default", () => {
    const list = normalizeOpenRouterList(JSON.parse(openRouterListFixture()));
    const free = list.find((m) => m.id === "free/zero-cost")!;
    // Present "0" pricing is a real free value (kept), not omitted.
    assertOk(free.cost, "cost present for free model");
    strictEqual(free.cost!.value.input, 0);
    strictEqual(free.cost!.value.output, 0);
    // No cacheRead present in source -> absent, not 0.
    strictEqual(free.cost!.value.cacheRead, undefined);
    // maxTokens absent (no top_provider) -> undefined, not 0.
    strictEqual(free.maxTokens, undefined);
  });

  test("USD/token converted to USD/million only for present fields", () => {
    const list = normalizeOpenRouterList(JSON.parse(openRouterListFixture()));
    const gpt = list.find((m) => m.id === "openai/gpt-test")!;
    // prompt 0.000005 USD/token -> 5 USD/million
    strictEqual(gpt.cost!.value.input, 5);
    strictEqual(gpt.cost!.value.output, 15);
    strictEqual(gpt.cost!.value.cacheRead, 1.25);
    strictEqual(gpt.cost!.value.cacheWrite, undefined);
  });

  test("conditional pricing is reported but not plannable", () => {
    const list = normalizeOpenRouterList(JSON.parse(openRouterListFixture()));
    const alias = list.find((m) => m.id === "deepseek/ds-alias")!;
    assertOk(alias.cost);
    strictEqual(alias.cost!.plannable, false, "conditional pricing must not be auto-applied");
    assertOk(alias.cost!.note?.includes("conditional"));
  });

  test("alias (canonical_slug != id) is informational, never rewritten", () => {
    const list = normalizeOpenRouterList(JSON.parse(openRouterListFixture()));
    const alias = list.find((m) => m.id === "deepseek/ds-alias")!;
    assertOk(alias.aliases?.includes("deepseek/ds-canonical"));
    // The id stays the user-facing id, not the canonical slug.
    strictEqual(alias.id, "deepseek/ds-alias");
  });

  test("reasoning true only with explicit signal; absence is unknown, not false", () => {
    const list = normalizeOpenRouterList(JSON.parse(openRouterListFixture()));
    const gpt = list.find((m) => m.id === "openai/gpt-test")!;
    const free = list.find((m) => m.id === "free/zero-cost")!;
    assertOk(gpt.reasoning, "explicit reasoning -> set");
    strictEqual(gpt.reasoning!.value, true);
    strictEqual(free.reasoning, undefined, "no reasoning signal -> unknown (not false)");
  });
});

describe("openrouter reasoning.supported_efforts -> thinkingLevelMap", () => {
  const listWithEfforts = (efforts: unknown): NormalizedModel[] =>
    normalizeOpenRouterList(
      JSON.parse(
        JSON.stringify({
          data: [
            {
              id: "openai/gpt-test",
              name: "GPT Test",
              reasoning: { mandatory: false, supported_efforts: efforts },
              supported_parameters: ["reasoning"],
            },
          ],
        }),
      ),
    );

  test("known efforts map to identity levels; absent levels hidden; none -> off", () => {
    const m = listWithEfforts(["none", "low", "high", "max"]).find((x) => x.id === "openai/gpt-test")!;
    assertOk(m.thinkingLevelMap, "efforts present -> map evidence");
    deepStrictEqual(m.thinkingLevelMap!.value, {
      off: "none",
      minimal: null,
      low: "low",
      medium: null,
      high: "high",
      xhigh: null,
      max: "max",
    });
    strictEqual(m.thinkingLevelMap!.plannable, true);
    strictEqual(m.thinkingLevelMap!.source, "openrouter:list");
  });

  test("unknown effort values make the map non-plannable (never silently dropped)", () => {
    const m = listWithEfforts(["low", "turbo"]).find((x) => x.id === "openai/gpt-test")!;
    assertOk(m.thinkingLevelMap);
    strictEqual(m.thinkingLevelMap!.plannable, false);
    assertOk(m.thinkingLevelMap!.note?.includes("turbo"), m.thinkingLevelMap!.note);
  });

  test("missing or empty efforts are unknown, never an empty map", () => {
    strictEqual(listWithEfforts([])[0]!.thinkingLevelMap, undefined);
    strictEqual(listWithEfforts(undefined)[0]!.thinkingLevelMap, undefined);
    strictEqual(
      normalizeOpenRouterList(JSON.parse(openRouterListFixture())).find((m) => m.id === "free/zero-cost")!.thinkingLevelMap,
      undefined,
    );
  });
});

describe("models.dev normalization", () => {
  test("normalized by unique slug, canonical id kept as note", () => {
    const list = normalizeModelsDev(JSON.parse(modelsDevFixture()));
    const glm = list.find((m) => m.id === "glm-5.2")!;
    strictEqual(glm.name!.value, "GLM-5.2");
    strictEqual(glm.contextWindow!.value, 1000000);
    strictEqual(glm.maxTokens!.value, 131072);
    strictEqual(glm.input!.value.includes("text"), true);
    assertOk(glm.notes?.some((n) => n.includes("zhipuai/glm-5.2")));
  });

  test("reasoning false is unknown, not authoritative false", () => {
    const list = normalizeModelsDev(JSON.parse(modelsDevFixture()));
    const ds = list.find((m) => m.id === "ds-alias")!;
    strictEqual(ds.reasoning, undefined, "models.dev reasoning:false -> unknown");
  });
});

describe("safeFetch transport safety", () => {
  test("rejects redirects when redirect:error", async () => {
    const { fetch } = mockFetch({
      "https://openrouter.ai/api/v1/models": { redirectTo: "https://evil.example/x", status: 302 },
    });
    await rejects(
      () => safeFetch("https://openrouter.ai/api/v1/models", fetch),
      /redirect not allowed|status/i,
    );
  });

  test("rejects non-https and non-allowlisted hosts", async () => {
    const { fetch } = mockFetch({});
    await rejects(() => safeFetch("http://openrouter.ai/x", fetch), /non-https/);
    await rejects(() => safeFetch("https://evil.example/x", fetch), /not allowlisted/);
  });

  test("rejects oversized bodies", async () => {
    const huge = "x".repeat(MAX_RESPONSE_BYTES + 10);
    const { fetch } = mockFetch({
      "https://models.dev/api.json": { body: huge },
    });
    await rejects(
      () => safeFetch("https://models.dev/api.json", fetch),
      /exceeds/,
    );
  });

  test("rejects malformed (non-json) and wrong content-type", async () => {
    const { fetch } = mockFetch({
      "https://models.dev/api.json": { body: "<html>not json</html>", contentType: "text/html" },
    });
    await rejects(
      () => safeFetch("https://models.dev/api.json", fetch),
      /non-json content-type/,
    );
  });

  test("rejects non-2xx status", async () => {
    const { fetch } = mockFetch({
      "https://opencode.ai/zen/go/v1/models": { status: 404, body: "not found" },
    });
    await rejects(
      () => safeFetch("https://opencode.ai/zen/go/v1/models", fetch),
      /status 404/,
    );
  });
});

describe("hyper catalog (metadata-only effort levels)", () => {
  test("effort_levels objects build a plannable map; empty/missing levels stay unknown", () => {
    const models = normalizeHyperList(JSON.parse(hyperListFixture()));
    const glm = models.find((m) => m.id === "glm-5.2")!;
    strictEqual(glm.provider, "hyper");
    assertOk(glm.thinkingLevelMap);
    deepStrictEqual(glm.thinkingLevelMap!.value, {
      off: null,
      minimal: null,
      low: null,
      medium: null,
      high: "high",
      xhigh: "xhigh",
      max: null,
    });
    strictEqual(glm.thinkingLevelMap!.plannable, true);
    strictEqual(glm.thinkingLevelMap!.source, "hyper:list");
    // Empty effort list is broken data (thinking models listed with []),
    // never an all-null map; a model without reasoning is simply unknown.
    strictEqual(models.find((m) => m.id === "empty-levels-model")!.thinkingLevelMap, undefined);
    strictEqual(models.find((m) => m.id === "plain-model")!.thinkingLevelMap, undefined);
    // display_name maps to the (informational) name.
    strictEqual(glm.name!.value, "GLM-5.2");
  });

  test("malformed effort entries demote the map to non-plannable", () => {
    const payload = JSON.stringify({
      data: [
        {
          id: "broken-levels",
          display_name: "Broken",
          reasoning: { effort_levels: [{ display: "High" }, "low"], default_effort_level: "low" },
        },
      ],
    });
    const m = normalizeHyperList(JSON.parse(payload))[0]!;
    assertOk(m.thinkingLevelMap);
    strictEqual(m.thinkingLevelMap!.plannable, false);
    assertOk(m.thinkingLevelMap!.note?.includes("malformed"), m.thinkingLevelMap!.note);
    // The usable string entry still shaped the (non-applied) map.
    strictEqual(m.thinkingLevelMap!.value.low, "low");
  });

  test("duplicate ids, malformed entries, and wrong shapes are rejected (fail closed)", () => {
    throws(() => normalizeHyperList({ data: [{ id: "a" }, { id: "a" }] }), /duplicate model id 'a'/);
    throws(() => normalizeHyperList({ data: [null] }), /malformed \(non-object\) entry/);
    throws(() => normalizeHyperList({}), /missing data\[/);
    throws(() => normalizeHyperList({ data: [] }), /no model entries/);
    throws(() => normalizeHyperList({ data: [{ display_name: "No Id" }] }), /no model entries/);
  });

  test("fetchSource('hyper') normalizes the public catalog and isolates failures", async () => {
    const okFetch = mockFetch({ "https://hyper.charm.land/v1/models": { body: hyperListFixture() } });
    const hyper = await fetchSource("hyper", okFetch.fetch);
    strictEqual(hyper.status, "ok");
    strictEqual(hyper.models.length, 4);
    assertOk(hyper.models.every((m) => m.provider === "hyper"), "hyper models are metadata-only");

    const downFetch = mockFetch({ "https://hyper.charm.land/v1/models": { status: 500 } });
    const down = await fetchSource("hyper", downFetch.fetch);
    strictEqual(down.status, "unavailable");
    strictEqual(down.models.length, 0);
  });

  test("endpoint constants include the hyper catalog", () => {
    strictEqual(SOURCE_ENDPOINTS.hyper, "https://hyper.charm.land/v1/models");
  });
});

describe("fetchSource dispatcher", () => {
  test("openrouter ok, opencode-zen ok (live zen), opencode-go ok (live go); zen/go never confused", async () => {
    const { fetch } = mockFetch({
      "https://openrouter.ai/api/v1/models": { body: openRouterListFixture() },
      "https://opencode.ai/zen/v1/models": { body: zenFixture() },
      "https://opencode.ai/zen/go/v1/models": { body: goFixture() },
      "https://models.dev/api.json": { body: modelsDevFixture() },
    });
    const [or, zen, go] = await Promise.all([
      fetchSource("openrouter", fetch),
      fetchSource("opencode-zen", fetch),
      fetchSource("opencode-go", fetch),
    ]);
    strictEqual(or.status, "ok");
    strictEqual(zen.status, "ok");
    strictEqual(go.status, "ok", "opencode-go live catalog is now reachable (200)");
    // Zen emits ONLY zen IDs; Go emits ONLY go IDs; the two live catalogs are distinct.
    const zenIds = new Set(zen.models.map((m) => m.id));
    const goIds = new Set(go.models.map((m) => m.id));
    assertOk(zen.models.every((m) => m.provider === "opencode"), "zen models are opencode");
    assertOk(go.models.every((m) => m.provider === "opencode-go"), "go models are opencode-go");
    strictEqual(goIds.has("go-only-model"), true, "go-only model emitted by go source");
    strictEqual(zenIds.has("go-only-model"), false, "go-only model NOT in zen source");
    strictEqual(zenIds.has("zen-only-model"), true, "zen-only model emitted by zen source");
    strictEqual(goIds.has("zen-only-model"), false, "zen-only model NOT in go source");
    // digests are stable sha256 hex
    strictEqual(or.digest.length, 64);
    notStrictEqual(or.digest, zen.digest);
    notStrictEqual(zen.digest, go.digest, "zen and go digests differ");
  });

  test("digest is deterministic for identical payloads", async () => {
    const body = openRouterListFixture();
    const a = mockFetch({ "https://openrouter.ai/api/v1/models": { body } });
    const b = mockFetch({ "https://openrouter.ai/api/v1/models": { body } });
    const ra = await fetchSource("openrouter", a.fetch);
    const rb = await fetchSource("openrouter", b.fetch);
    strictEqual(ra.digest, rb.digest);
  });

  test("endpoints source is a constant allowlisted https url", () => {
    strictEqual(SOURCE_ENDPOINTS.openrouter, "https://openrouter.ai/api/v1/models");
    strictEqual(SOURCE_ENDPOINTS["opencode-zen"], "https://opencode.ai/zen/v1/models");
    strictEqual(SOURCE_ENDPOINTS["opencode-go"], "https://opencode.ai/zen/go/v1/models");
  });

  test("endpoint disagreement yields non-plannable evidence", async () => {
    const { fetch } = mockFetch({
      "https://openrouter.ai/api/v1/models/openai%2Fgpt-test/endpoints": {
        body: openRouterEndpointsFixture({ context: [100000, 120000] }),
      },
    });
    // Reuse refineOpenRouterModel via fetchSource + refine path indirectly:
    // construct a base model and refine it.
    const { refineOpenRouterModel } = await import("../sources.ts");
    const base = normalizeOpenRouterList(JSON.parse(openRouterListFixture())).find((m) => m.id === "openai/gpt-test")!;
    const refined = await refineOpenRouterModel(
      base,
      "https://openrouter.ai/api/v1/models/openai%2Fgpt-test/endpoints",
      fetch,
    );
    strictEqual(refined.contextWindow!.plannable, false, "disagreeing endpoints -> not plannable");
    assertOk(refined.contextWindow!.note?.includes("disagree"));
  });
});

describe("sha256Hex", () => {
  test("stable over string and buffer forms", () => {
    strictEqual(sha256Hex("abc"), sha256Hex(Buffer.from("abc", "utf8")));
  });
});

describe("models.dev strictness", () => {
  test("same slug under several orgs: deterministic first sighting wins (mirrors expected)", () => {
    const mirrored = {
      alpha: { models: { foo: { id: "alpha/foo", name: "ALPHA MIRROR", reasoning: true, modalities: { input: ["text"] }, limit: { context: 1, output: 1 } } } },
      zeta: { models: { foo: { id: "zeta/foo", name: "ZETA MIRROR", reasoning: true, modalities: { input: ["text"] }, limit: { context: 2, output: 2 } } } },
    };
    const list = normalizeModelsDev(mirrored);
    strictEqual(list.length, 1, "one entry per unique slug");
    strictEqual(list[0]!.name!.value, "ALPHA MIRROR", "alphabetically-first org is scanned first");
    assertOk(list[0]!.notes?.some((n) => n.includes("alpha/foo")), "canonical id kept as note");
  });

  test("preferred org (opencode) wins over alphabetically-earlier org", () => {
    const mirrored = {
      alpha: { models: { glm: { id: "alpha/glm", name: "THIRD-PARTY", reasoning: true, modalities: { input: ["text"] }, limit: { context: 1, output: 1 } } } },
      opencode: { models: { glm: { id: "opencode/glm", name: "FIRST-PARTY", reasoning: true, modalities: { input: ["text"] }, limit: { context: 3, output: 3 } } } },
    };
    const list = normalizeModelsDev(mirrored);
    strictEqual(list[0]!.name!.value, "FIRST-PARTY", "preferred org scanned before sorted orgs");
  });

  test("wrong-shape payload (top-level data array) is rejected", () => {
    throws(() => normalizeModelsDev({ data: [] }), /unexpected 'data' array/i);
  });
});

describe("openRouterDetailUrl preserves full id segments", () => {
  test("encodes each segment, keeps the slash", () => {
    strictEqual(
      openRouterDetailUrl("openai/gpt-4o"),
      "https://openrouter.ai/api/v1/models/openai/gpt-4o/endpoints",
    );
    strictEqual(
      openRouterDetailUrl("a b/c d"),
      "https://openrouter.ai/api/v1/models/a%20b/c%20d/endpoints",
    );
  });
});

describe("applyOpenRouterEndpoints", () => {
  test("disagreeing endpoint limits mark fields non-plannable and report ambiguity", () => {
    const base = normalizeOpenRouterList(JSON.parse(openRouterListFixture())).find((m) => m.id === "openai/gpt-test")!;
    const detail = openRouterEndpointsFixture({ context: [100000, 120000], max: [16000, 16000] });
    const r = applyOpenRouterEndpoints(base, detail);
    strictEqual(r.status, "ambiguous");
    strictEqual(r.model.contextWindow!.plannable, false);
    assertOk(r.model.contextWindow!.note?.includes("disagree"));
  });

  test("agreeing endpoints are authoritative + plannable (status ok)", () => {
    const base = normalizeOpenRouterList(JSON.parse(openRouterListFixture())).find((m) => m.id === "openai/gpt-test")!;
    const detail = openRouterEndpointsFixture({ context: [128000], max: [16384] });
    const r = applyOpenRouterEndpoints(base, detail);
    strictEqual(r.status, "ok");
    strictEqual(r.model.contextWindow!.value, 128000);
    strictEqual(r.model.contextWindow!.plannable, true);
  });

  test("malformed detail demotes list limits (never auto-applied) and never throws", () => {
    const base = normalizeOpenRouterList(JSON.parse(openRouterListFixture())).find((m) => m.id === "openai/gpt-test")!;
    const originalCtxValue = base.contextWindow!.value;
    // Non-JSON body.
    let r = applyOpenRouterEndpoints(base, "<html>oops</html>");
    strictEqual(r.status, "malformed");
    strictEqual(r.model.contextWindow!.plannable, false, "list limit demoted on malformed detail");
    assertOk(r.model.contextWindow!.note?.includes("not auto-applied"));
    // JSON but no endpoints array.
    r = applyOpenRouterEndpoints(base, JSON.stringify({ data: { id: "x" } }));
    strictEqual(r.status, "malformed");
    strictEqual(r.model.contextWindow!.plannable, false);
    // Empty endpoints array.
    r = applyOpenRouterEndpoints(base, JSON.stringify({ data: { endpoints: [] } }));
    strictEqual(r.status, "malformed");
    strictEqual(r.model.contextWindow!.plannable, false);
    // List value is retained (reported) but never coerced to a default / never throws.
    strictEqual(r.model.contextWindow!.value, originalCtxValue);
  });

  test("malformed entry in a NON-EMPTY endpoints array fails the whole detail (reviewer 226d8fda repro)", () => {
    const base = normalizeOpenRouterList(JSON.parse(openRouterListFixture())).find((m) => m.id === "openai/gpt-test")!;
    // {"data":{"endpoints":[null]}} previously returned status ok and kept
    // list-derived plannable limits. It must now fail closed.
    const r = applyOpenRouterEndpoints(base, JSON.stringify({ data: { endpoints: [null] } }));
    strictEqual(r.status, "malformed", "non-object endpoint entry -> malformed");
    strictEqual(r.model.contextWindow!.plannable, false, "list context demoted");
    strictEqual(r.model.maxTokens!.plannable, false, "list max demoted");
    assertOk(r.note?.includes("malformed"));
    // A non-object entry among otherwise-valid endpoints also fails the whole detail.
    const r2 = applyOpenRouterEndpoints(
      base,
      JSON.stringify({ data: { endpoints: [{ context_length: 128000, max_completion_tokens: 16384 }, "junk"] } }),
    );
    strictEqual(r2.status, "malformed");
    strictEqual(r2.model.contextWindow!.plannable, false);
  });

  test("detail with valid objects but no limit fields holds limits demoted (never ok+plannable)", () => {
    const base = normalizeOpenRouterList(JSON.parse(openRouterListFixture())).find((m) => m.id === "openai/gpt-test")!;
    // Endpoints are objects but carry no usable limits -> list limits stay
    // non-plannable (begin-d demoted, never promoted without agreeing evidence).
    const r = applyOpenRouterEndpoints(base, JSON.stringify({ data: { endpoints: [{ name: "ep1" }] } }));
    strictEqual(r.model.contextWindow!.plannable, false, "no endpoint evidence -> context stays demoted");
    strictEqual(r.model.maxTokens!.plannable, false, "no endpoint evidence -> max stays demoted");
  });
});

// ─── Regression: source schemas must fail closed (reviewer 65718af4) ───────

describe("regression: openrouter duplicate / negative / partial-numeric refusal", () => {
  test("duplicate model ids are rejected (fail closed)", () => {
    const dup = {
      data: [
        { id: "evil/m", name: "Good", context_length: 100 },
        { id: "evil/m", name: "Hostile", context_length: -1 },
      ],
    };
    throws(() => normalizeOpenRouterList(dup), /duplicate model id 'evil\/m'/);
  });

  test("negative context_length is rejected (treated unknown, not applied)", () => {
    const list = normalizeOpenRouterList({ data: [{ id: "a/b", name: "A", context_length: -5 }] });
    strictEqual(list[0]!.contextWindow, undefined, "negative limit must not become evidence");
  });

  test("partial numeric string ('12junk') is rejected, not parsed as 12", () => {
    const list = normalizeOpenRouterList({ data: [{ id: "a/b", name: "A", context_length: "12junk" }] });
    strictEqual(list[0]!.contextWindow, undefined, "partial numeric must not be accepted");
  });

  test("negative / partial-numeric prices are rejected", () => {
    const list = normalizeOpenRouterList({
      data: [{ id: "a/b", name: "A", pricing: { prompt: "-0.001", completion: "0.5junk" } }],
    });
    // negative prompt rejected, partial completion rejected -> no cost evidence
    strictEqual(list[0]!.cost, undefined);
  });

  test("non-finite / NaN limits are rejected", () => {
    const list = normalizeOpenRouterList({ data: [{ id: "a/b", name: "A", context_length: "NaN", top_provider: { max_completion_tokens: Infinity } }] });
    strictEqual(list[0]!.contextWindow, undefined);
    strictEqual(list[0]!.maxTokens, undefined);
  });
});

describe("regression: models.dev strictness (empty / null / numeric)", () => {
  test("empty payload {} is rejected (fail closed)", () => {
    throws(() => normalizeModelsDev({}), /no org entries/);
  });

  test("null / non-object entry is rejected (not silently skipped)", () => {
    throws(
      () => normalizeModelsDev({ org: { models: { foo: null } } }),
      /entry 'org\/foo' is not an object/,
    );
  });

  test("negative / partial-numeric limits are rejected", () => {
    const list = normalizeModelsDev({ org: { models: { foo: { id: "org/foo", name: "Foo", limit: { context: -5, output: "12junk" } } } } });
    strictEqual(list[0]!.contextWindow, undefined);
    strictEqual(list[0]!.maxTokens, undefined);
  });
});

// ─── Regression: OpenCode Zen is the live availability authority (reviewer 3020e430)

describe("regression: opencode zen live authority (no unroutable models.dev slugs)", () => {
  test("normalizeZenLive parses the live {data:[{id}]} catalog and rejects malformed/dup", () => {
    strictEqual(normalizeZenLive(JSON.parse(zenFixture())).sort().join(","), "glm-5.2,gpt-test,zen-only-model");
    throws(() => normalizeZenLive({ data: [null] }), /malformed/);
    throws(() => normalizeZenLive({ data: [{ id: "x" }, { id: "x" }] }), /duplicate id 'x'/);
    throws(() => normalizeZenLive({}), /missing data/);
  });

  test("opencode-zen emits ONLY live Zen IDs (models.dev-only slugs are never emitted)", async () => {
    // models.dev has slugs: gpt-test, glm-5.2, ds-alias. Zen offers only
    // glm-5.2, gpt-test, zen-only-model. The unroutable models.dev-only slug
    // `ds-alias` must NOT appear; the Zen-only `zen-only-model` IS emitted.
    const { fetch } = mockFetch({
      "https://opencode.ai/zen/v1/models": { body: zenFixture() },
      "https://models.dev/api.json": { body: modelsDevFixture() },
    });
    const zen = await fetchOpenCodeZenSource(fetch);
    strictEqual(zen.status, "ok");
    const ids = new Set(zen.models.map((m) => m.id));
    strictEqual(ids.has("glm-5.2"), true, "zen+models.dev intersection emitted");
    strictEqual(ids.has("gpt-test"), true);
    strictEqual(ids.has("zen-only-model"), true, "zen-only model emitted (routable)");
    strictEqual(ids.has("ds-alias"), false, "models.dev-only slug is NOT emitted (unroutable)");
    strictEqual(zen.models.length, 3, "exactly the 3 live Zen IDs");
  });

  test("models.dev enrichment failure leaves Zen IDs authoritative (minimal metadata)", async () => {
    const { fetch } = mockFetch({
      "https://opencode.ai/zen/v1/models": { body: zenFixture() },
      "https://models.dev/api.json": { status: 500 },
    });
    const zen = await fetchOpenCodeZenSource(fetch);
    strictEqual(zen.status, "ok");
    strictEqual(zen.models.length, 3);
  });

  test("zen authority unavailable -> source unavailable (never invents models)", async () => {
    const { fetch } = mockFetch({
      "https://opencode.ai/zen/v1/models": { status: 500 },
      "https://models.dev/api.json": { body: modelsDevFixture() },
    });
    const zen = await fetchOpenCodeZenSource(fetch);
    strictEqual(zen.status, "unavailable");
    strictEqual(zen.models.length, 0);
  });
});

// ─── Regression: opencode-go is a real sync provider (live /zen/go/v1/models)

describe("regression: opencode-go live catalog (real provider, not check-only)", () => {
  test("the stale 404 URL must NEVER appear; the live Go URL is the source", () => {
    strictEqual(SOURCE_ENDPOINTS["opencode-go"], "https://opencode.ai/zen/go/v1/models");
    // The old wrong URL from stale research must be gone entirely from source.
    const src = readFileSync("sources.ts", "utf8");
    strictEqual(src.includes("https://opencode.ai/go/v1/models"), false, "stale /go/v1 URL removed");
    strictEqual(src.includes("check-only"), false, "check-only wording removed");
  });

  test("parseLiveCatalogIds is shared + strict (rejects malformed/dup) for go", () => {
    strictEqual(parseLiveCatalogIds(JSON.parse(goFixture()), "opencode go").sort().join(","), "glm-5.2,go-only-model,kimi-k3");
    throws(() => parseLiveCatalogIds({ data: [null] }, "opencode go"), /malformed/);
    throws(() => parseLiveCatalogIds({ data: [{ id: "x" }, { id: "x" }] }, "opencode go"), /duplicate id 'x'/);
    throws(() => parseLiveCatalogIds({}, "opencode go"), /missing data/);
  });

  test("a mocked 200 Go payload drives a full opencode-go source (availability + enrichment)", async () => {
    const { fetch } = mockFetch({
      "https://opencode.ai/zen/go/v1/models": { body: goFixture() },
      "https://models.dev/api.json": { body: modelsDevFixture() },
    });
    const go = await fetchOpenCodeGoSource(fetch);
    strictEqual(go.status, "ok");
    const ids = new Set(go.models.map((m) => m.id));
    // goFixture IDs: glm-5.2, kimi-k3, go-only-model. glm-5.2 is enriched by
    // models.dev; the others carry minimal (id-only) metadata. None invented.
    strictEqual(go.models.length, 3);
    assertOk(ids.has("glm-5.2"));
    assertOk(ids.has("kimi-k3"));
    assertOk(ids.has("go-only-model"));
    const glm = go.models.find((m) => m.id === "glm-5.2")!;
    strictEqual(glm.provider, "opencode-go");
    assertOk(glm.name, "glm-5.2 enriched with models.dev name");
  });

  test("Go authority unavailable -> source unavailable (failure isolation, never invents)", async () => {
    const { fetch } = mockFetch({
      "https://opencode.ai/zen/go/v1/models": { status: 500 },
      "https://models.dev/api.json": { body: modelsDevFixture() },
    });
    const go = await fetchOpenCodeGoSource(fetch);
    strictEqual(go.status, "unavailable");
    strictEqual(go.models.length, 0);
  });

  test("Zen /zen/v1 and Go /zen/go/v1 catalogs are never confused (distinct providers)", async () => {
    const { fetch } = mockFetch({
      "https://opencode.ai/zen/v1/models": { body: zenFixture() },
      "https://opencode.ai/zen/go/v1/models": { body: goFixture() },
      "https://models.dev/api.json": { body: modelsDevFixture() },
    });
    const [zen, go] = await Promise.all([
      fetchOpenCodeZenSource(fetch),
      fetchOpenCodeGoSource(fetch),
    ]);
    strictEqual(zen.status, "ok");
    strictEqual(go.status, "ok");
    // zen-only is a Zen offering, NOT a Go offering; go-only is Go, NOT Zen.
    strictEqual(new Set(zen.models.map((m) => m.id)).has("go-only-model"), false);
    strictEqual(new Set(go.models.map((m) => m.id)).has("zen-only-model"), false);
    // a shared id (glm-5.2) appears in BOTH but under the correct provider each time.
    strictEqual(zen.models.find((m) => m.id === "glm-5.2")!.provider, "opencode");
    strictEqual(go.models.find((m) => m.id === "glm-5.2")!.provider, "opencode-go");
  });
});

// ─── Routability probe of new zen/go models ─────────────────────────────────

describe("probeOpenCodeModel routability classification", () => {
  const responds =
    (status: number, body: string, contentType = "application/json") =>
    async (): Promise<unknown> => ({
      ok: status >= 200 && status < 300,
      status,
      url: "probe",
      headers: { get: (n: string) => (n.toLowerCase() === "content-type" ? contentType : null) },
      text: async () => body,
      arrayBuffer: async () => new ArrayBuffer(0),
    });

  test("2xx serves and 429 throttling prove the route exists", async () => {
    strictEqual((await probeOpenCodeModel("opencode", "m", "k", responds(200, "{}") as never)).status, "routable");
    strictEqual(
      (await probeOpenCodeModel("opencode-go", "m", "k", responds(429, "slow down", "text/plain") as never)).status,
      "routable",
      "429 means routed but throttled",
    );
  });

  test("404 (SPA no-route), 5xx and explicit 'not supported' are unroutable", async () => {
    strictEqual(
      (await probeOpenCodeModel("opencode", "union-alpha", "k", responds(404, "<!DOCTYPE html><html>SPA</html>", "text/html") as never)).status,
      "unroutable",
      "marketing-site 404 = router has no such model",
    );
    strictEqual((await probeOpenCodeModel("opencode", "m", "k", responds(500, "Internal server error", "text/plain") as never)).status, "unroutable");
    strictEqual(
      (await probeOpenCodeModel("opencode", "m", "k", responds(401, '{"error":{"message":"Model m is not supported"}}') as never)).status,
      "unroutable",
    );
  });

  test("plain auth failures and network errors are inconclusive (never block adds)", async () => {
    strictEqual((await probeOpenCodeModel("opencode", "m", "k", responds(401, "unauthorized", "text/plain") as never)).status, "inconclusive");
    const dead = (async () => {
      throw new Error("fetch failed");
    }) as never;
    strictEqual((await probeOpenCodeModel("opencode", "m", "k", dead)).status, "inconclusive");
  });
});
