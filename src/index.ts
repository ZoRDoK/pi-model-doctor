/**
 * pi extension entry: one command, no arguments.
 *
 * `/model-doctor` reconciles every supported provider in a single pass and
 * writes models.json once (atomic + backup + CAS + rollback), then reports what
 * changed. Only node:* and type-only pi imports are used.
 */
import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";

import { PI_DEFAULT_MAX_OUTPUT_TOKENS, runReconcile, type RefCatalog } from "./doctor.ts";
import { agentDir, defaultPaths } from "./paths.ts";
import { boundReport, buildResultReport, MAX_REPORT_CHARS, reportSanitized, sanitize } from "./report.ts";
import { MANAGED_PROVIDERS } from "./types.ts";
import type { EffectiveSnapshot, FetchFn, NormalizedModel } from "./types.ts";
import { evidence, messageOf } from "./util.ts";

// ─── Pi registry adapters ────────────────────────────────────────────────────

interface EffectiveLike {
  readonly id: string;
  readonly provider: string;
  readonly name: string;
  readonly api: string;
  readonly reasoning: boolean;
  readonly thinkingLevelMap?: Record<string, string | null>;
  readonly input: readonly ("text" | "image")[];
  readonly cost: Record<string, unknown>;
  readonly contextWindow: number;
  readonly maxTokens: number;
  readonly compat?: Record<string, unknown>;
}

/** Structural slice of pi's ModelRegistry (getAll / find / hasConfiguredAuth). */
interface RegistryLike {
  getAll(): readonly EffectiveLike[];
  find(provider: string, modelId: string): EffectiveLike | undefined;
  hasConfiguredAuth(model: EffectiveLike): boolean;
}

function normalizeEffective(model: EffectiveLike): NormalizedModel {
  const source = "pi:effective";
  const out: NormalizedModel = {
    provider: model.provider,
    id: model.id,
    name: evidence(model.name, source),
    api: evidence(model.api, source),
    reasoning: evidence(model.reasoning, source),
    input: evidence([...model.input], source),
    cost: evidence(model.cost, source),
    contextWindow: evidence(model.contextWindow, source),
  };

  if (model.thinkingLevelMap) out.thinkingLevelMap = evidence(model.thinkingLevelMap, source);
  if (model.compat) out.compat = evidence(model.compat, source);

  // pi fills a missing maxTokens with 16384; that is an absence, not a ceiling.
  if (model.maxTokens !== PI_DEFAULT_MAX_OUTPUT_TOKENS) {
    out.maxTokens = evidence(model.maxTokens, source);
  }

  return out;
}

/** Every provider pi serves, built-in catalog included. Generic sync uses this
 *  so a model pi already provides is never duplicated into models.json. */
function buildEffective(registry: RegistryLike): EffectiveSnapshot {
  let models: readonly EffectiveLike[];

  try {
    models = registry.getAll();
  } catch {
    models = [];
  }

  const providers: Record<string, NormalizedModel[]> = {};

  for (const provider of MANAGED_PROVIDERS) providers[provider] = [];

  for (const model of models) {
    if (!model || typeof model.provider !== "string" || typeof model.id !== "string") continue;
    (providers[model.provider] ??= []).push(normalizeEffective(model));
  }

  return { providers };
}

function buildRefCatalog(registry: RegistryLike): RefCatalog {
  return {
    modelIds(provider: string): readonly string[] {
      try {
        return registry.getAll().filter((m) => m.provider === provider && typeof m.id === "string").map((m) => m.id);
      } catch {
        return [];
      }
    },
    hasAuth(provider: string, modelId: string): boolean {
      try {
        const model = registry.find(provider, modelId);
        return model ? registry.hasConfiguredAuth(model) : false;
      } catch {
        return false;
      }
    },
  };
}

// ─── Command ─────────────────────────────────────────────────────────────────

function defaultFetch(): FetchFn {
  return globalThis.fetch as unknown as FetchFn;
}

export default function (pi: ExtensionAPI): void {
  pi.registerCommand("model-doctor", {
    description:
      "Reconcile all model catalogs (official vs pi vs models.json) in one safe pass and write models.json once. No subcommands or arguments.",
    handler: async (args: string, ctx: ExtensionCommandContext) => {
      if ((args ?? "").trim().length > 0) {
        notify(ctx, reportSanitized(`model-doctor: this command takes no arguments or subcommands (got ${JSON.stringify(args)}). Usage: /model-doctor\n${helpText()}`), "warning");
        return;
      }

      const registry = ctx.modelRegistry as unknown as RegistryLike;
      const paths = defaultPaths();
      const fetchFn = defaultFetch();

      try {
        const { result, report } = await runReconcile({
          paths,
          fetchFn,
          effective: buildEffective(registry),
          refCatalog: buildRefCatalog(registry),
        });
        const { shown } = await boundReport(reportSanitized(buildResultReport(report, result)));
        sendReport(ctx, pi, shown, result.ok ? "info" : "warning");
      } catch (err) {
        sendReport(ctx, pi, reportSanitized(`model-doctor: error — ${sanitize(messageOf(err))}`), "error");
      }
    },
  });
}

/** Reports go to the transcript as a Markdown message so tables render. */
function sendReport(ctx: ExtensionCommandContext, pi: ExtensionAPI, text: string, type: "info" | "warning" | "error"): void {
  if (ctx.hasUI && typeof pi.sendMessage === "function") {
    try {
      pi.sendMessage({ customType: "model-doctor-report", content: text, display: true }, { triggerTurn: false });
      return;
    } catch {
      // fall through to notify
    }
  }

  notify(ctx, text, type);
}

function notify(ctx: ExtensionCommandContext, text: string, type: "info" | "warning" | "error"): void {
  if (ctx.hasUI) ctx.ui.notify(text, type);
  console.log(text);
}

function helpText(): string {
  return [
    "model-doctor — one-shot model catalog reconciliation",
    "",
    "Usage:",
    "  /model-doctor     reconcile all providers and write models.json once",
    "",
    "Fetches the official catalogs (OpenRouter, models.dev, OpenCode, Charm",
    "Hyper), reads pi's effective catalog and your models.json/settings.json, then:",
    "  • updates drifted fields on existing managed entries;",
    "  • adds genuinely new models — only when the source publishes both the",
    "    context window and the max output, never with guessed numbers;",
    "  • skips unsafe, ambiguous or unknown-limits models and reports them.",
    "",
    "models.json is written atomically (timestamped backup + CAS + rollback);",
    "settings.json, auth.json and models-store.json are read-only, and provider",
    "credentials, base URLs and headers are never changed or echoed.",
  ].join("\n");
}

/** Exported for tests only; not part of the pi command surface. */
export const __internals = {
  agentDir,
  boundReport,
  buildEffective,
  buildRefCatalog,
  buildResultReport,
  helpText,
  MAX_REPORT_CHARS,
  sanitize,
};

export { sanitize };
