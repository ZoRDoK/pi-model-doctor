/**
 * Cross-provider thinking-level resolution tests (openrouter efforts vs hyper).
 *
 * All inputs are in-memory Diagnosis objects; no network, no filesystem.
 */
import { describe, test } from "node:test";
import { strictEqual, ok as assertOk, deepStrictEqual } from "node:assert/strict";

import { enrichThinkingLevels, resolveThinkingLevels } from "../thinking.ts";
import { normalizeHyperList, normalizeOpenRouterList } from "../sources.ts";
import type {
  Diagnosis,
  Evidence,
  ManagedProvider,
  NormalizedModel,
  SourceResult,
} from "../types.ts";

const LOW_HIGH_MAX = ["low", "high", "max"] as const;

function orEffortsModel(id: string, efforts: readonly string[]): NormalizedModel[] {
  return normalizeOpenRouterList(
    JSON.parse(
      JSON.stringify({
        data: [
          {
            id,
            name: id,
            reasoning: { mandatory: false, supported_efforts: [...efforts] },
            supported_parameters: ["reasoning"],
          },
        ],
      }),
    ),
  );
}

function hyperModel(id: string, efforts: readonly string[]): NormalizedModel[] {
  return normalizeHyperList(
    JSON.parse(
      JSON.stringify({
        data: [
          {
            id,
            display_name: id,
            reasoning: { effort_levels: efforts.map((v) => ({ value: v, display: v })), default_effort_level: null },
          },
        ],
      }),
    ),
  );
}

function diagnosisWith(
  openrouter: readonly NormalizedModel[],
  hyper: readonly NormalizedModel[],
): Diagnosis {
  const orSource: SourceResult = {
    id: "openrouter",
    status: "ok",
    digest: "d-or",
    models: openrouter,
    fetchedAt: "2026-01-01T00:00:00Z",
  };
  const hyperSource: SourceResult = {
    id: "hyper",
    status: "ok",
    digest: "d-hyper",
    models: hyper,
    fetchedAt: "2026-01-01T00:00:00Z",
  };
  return { sources: [orSource, hyperSource], findings: [], effective: { providers: {
    openrouter: [], opencode: [], "opencode-go": [],
  } } };
}

describe("resolveThinkingLevels: openrouter (popular) vs hyper priority", () => {
  test("both sources agree -> plannable map sourced from both", () => {
    const d = diagnosisWith(
      orEffortsModel("moonshotai/kimi-k3", LOW_HIGH_MAX),
      hyperModel("kimi-k3", LOW_HIGH_MAX),
    );
    const resolved = resolveThinkingLevels("opencode", "kimi-k3", d)!;
    assertOk(resolved, "agreement resolves");
    strictEqual(resolved.plannable, true);
    strictEqual(resolved.source, "openrouter+hyper:efforts");
    deepStrictEqual(resolved.value, {
      off: null, minimal: null, low: "low", medium: null, high: "high", xhigh: null, max: "max",
    });
  });

  test("openrouter only -> plannable, sourced from openrouter (slug match for zen ids)", () => {
    const d = diagnosisWith(orEffortsModel("moonshotai/kimi-k3", LOW_HIGH_MAX), []);
    const resolved = resolveThinkingLevels("opencode", "kimi-k3", d)!;
    strictEqual(resolved.plannable, true);
    strictEqual(resolved.source, "openrouter:efforts");
  });

  test("hyper only fills the gap when openrouter has nothing", () => {
    const d = diagnosisWith([], hyperModel("kimi-k3", LOW_HIGH_MAX));
    const resolved = resolveThinkingLevels("opencode", "kimi-k3", d)!;
    strictEqual(resolved.plannable, true);
    strictEqual(resolved.source, "hyper:efforts");
  });

  test("neither source knows the model -> undefined (never an empty map)", () => {
    const d = diagnosisWith([], []);
    strictEqual(resolveThinkingLevels("opencode", "ghost", d), undefined);
  });

  test("disagreement -> non-plannable with both level sets in the note", () => {
    const d = diagnosisWith(
      orEffortsModel("moonshotai/kimi-k3", LOW_HIGH_MAX),
      hyperModel("kimi-k3", ["high", "xhigh"]),
    );
    const resolved = resolveThinkingLevels("opencode", "kimi-k3", d)!;
    strictEqual(resolved.plannable, false, "never auto-apply a disputed map");
    assertOk(resolved.note?.includes("openrouter [low,high,max]"), resolved.note);
    assertOk(resolved.note?.includes("hyper [high,xhigh]"), resolved.note);
  });

  test("openrouter slug collision with differing sets -> non-plannable, no hyper fallback", () => {
    const d = diagnosisWith(
      [...orEffortsModel("moonshotai/kimi-k3", LOW_HIGH_MAX), ...orEffortsModel("mirror/kimi-k3", ["high"])],
      hyperModel("kimi-k3", LOW_HIGH_MAX),
    );
    const resolved = resolveThinkingLevels("opencode", "kimi-k3", d)!;
    strictEqual(resolved.plannable, false);
    assertOk(resolved.note?.includes("differing effort levels"), resolved.note);
  });

  test("openrouter-managed models match by exact id; hyper fills by slug", () => {
    // OR knows the model but reports no efforts; hyper lists the bare slug.
    const d = diagnosisWith(
      normalizeOpenRouterList(JSON.parse(JSON.stringify({ data: [{ id: "deepseek/deepseek-v4-pro", name: "DS" }] }))),
      hyperModel("deepseek-v4-pro", ["high", "xhigh"]),
    );
    const resolved = resolveThinkingLevels("openrouter", "deepseek/deepseek-v4-pro", d)!;
    strictEqual(resolved.plannable, true);
    strictEqual(resolved.source, "hyper:efforts");
    strictEqual(resolved.value.high, "high");
  });

  test("unavailable sources never resolve anything", () => {
    const d: Diagnosis = {
      sources: [
        { id: "openrouter", status: "unavailable", digest: "x", models: [], fetchedAt: "t" },
        { id: "hyper", status: "unavailable", digest: "y", models: [], fetchedAt: "t" },
      ],
      findings: [],
      effective: { providers: { openrouter: [], opencode: [], "opencode-go": [] } },
    };
    strictEqual(resolveThinkingLevels("opencode", "kimi-k3", d), undefined);
  });
});

describe("enrichThinkingLevels", () => {
  const zenModel = (id: string): NormalizedModel => ({ provider: "opencode", id });
  const agreed = diagnosisWith(
    orEffortsModel("moonshotai/kimi-k3", LOW_HIGH_MAX),
    hyperModel("kimi-k3", LOW_HIGH_MAX),
  );

  test("attaches the resolved map and the implied reasoning=true", () => {
    const enriched = enrichThinkingLevels("opencode", zenModel("kimi-k3"), agreed);
    assertOk(enriched.thinkingLevelMap);
    strictEqual(enriched.thinkingLevelMap!.plannable, true);
    assertOk(enriched.reasoning, "levels imply the model reasons");
    strictEqual(enriched.reasoning!.value, true);
    strictEqual(enriched.reasoning!.plannable, true);
  });

  test("existing evidence is never replaced (direct OR map / pi built-in map)", () => {
    const builtinMap: Evidence<Readonly<Record<string, string | null>>> = {
      value: { medium: "high" },
      source: "pi:effective",
      confidence: "authoritative",
      plannable: true,
    };
    const model: NormalizedModel = { provider: "opencode-go", id: "kimi-k3", thinkingLevelMap: builtinMap };
    const enriched = enrichThinkingLevels("opencode-go", model, agreed);
    strictEqual(enriched.thinkingLevelMap, builtinMap, "pi:effective map preserved as-is");
  });

  test("plannable reasoning=true is never overwritten", () => {
    const model: NormalizedModel = {
      provider: "opencode",
      id: "kimi-k3",
      reasoning: { value: true, source: "models.dev:opencode", confidence: "reported", plannable: true },
    };
    const enriched = enrichThinkingLevels("opencode", model, agreed);
    strictEqual(enriched.reasoning!.source, "models.dev:opencode", "original reasoning provenance kept");
  });

  test("non-plannable resolution (disagreement) attaches evidence without reasoning", () => {
    const disagree = diagnosisWith(
      orEffortsModel("moonshotai/kimi-k3", LOW_HIGH_MAX),
      hyperModel("kimi-k3", ["high", "xhigh"]),
    );
    const enriched = enrichThinkingLevels("opencode", zenModel("kimi-k3"), disagree);
    assertOk(enriched.thinkingLevelMap);
    strictEqual(enriched.thinkingLevelMap!.plannable, false);
    strictEqual(enriched.reasoning, undefined, "no reasoning op from a disputed map");
  });

  test("no resolution -> model unchanged", () => {
    const model = zenModel("ghost");
    strictEqual(enrichThinkingLevels("opencode", model, agreed), model);
  });
});
