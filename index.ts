/**
 * pi extension: /model-doctor
 *
 * ONE command, no subcommands, no arguments. A single `/model-doctor` invocation
 * reconciles ALL supported providers (openrouter, opencode, opencode-go,
 * opencode-go-2) in one safe pass and writes models.json once, atomically, with a
 * timestamped backup, compare-and-swap, and rollback. Then it prints a clear
 * per-model result (added / updated / skipped) with characteristics and
 * per-provider totals — it never dumps hundreds of unchanged models.
 *
 * What it does:
 *  - fetch official sources (OpenRouter, models.dev) + read Pi's effective
 *    catalog + user models.json/settings.json;
 *  - UPDATE drifted authoritative fields on existing managed entries;
 *  - ADD official-catalog models that are absent from Pi's effective registry
 *    (genuinely new models; never duplicates pi built-ins into models.json);
 *  - fully MIRROR opencode-go -> opencode-go-2;
 *  - skip unsafe/ambiguous models/fields and report them as warnings;
 *  - preserve credentials, base URLs, headers, unrelated providers, unknown
 *    fields; never print secrets.
 *
 * Runtime deps: node:* only. The pi package is imported for TYPES only (erased
 * at runtime; resolved by the host's extension loader).
 */
import { homedir } from "node:os";
import { dirname, join } from "node:path";

import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";

import {
  runReconcile,
  PI_DEFAULT_MAX_OUTPUT_TOKENS,
  type RefCatalog,
  type ReconcileReport,
  type ModelResult,
  type ProviderTotal,
} from "./doctor.ts";
import type {
  ApplyResult,
  DoctorPaths,
  EffectiveSnapshot,
  Evidence,
  FetchFn,
  ManagedProvider,
  ModelEntry,
  ModelsConfig,
  MutationPath,
  NormalizedModel,
  ProviderEntry,
} from "./types.ts";
import {
  MANAGED_PROVIDERS,
  isManagedProvider,
  isCompositePath,
  allowlistedSubfields,
} from "./types.ts";
import { open } from "node:fs/promises";
import { randomBytes } from "node:crypto";

// ─── Path resolution (mirrors pi's getAgentDir without a runtime host import) ─

function agentDir(): string {
  const env = process.env.PI_CODING_AGENT_DIR;
  if (env && env.length > 0) return env;
  const home = process.env.HOME || process.env.USERPROFILE || homedir();
  return join(home, ".pi", "agent");
}

function defaultPaths(): DoctorPaths {
  const modelsPath = join(agentDir(), "models.json");
  return {
    modelsPath,
    settingsPath: join(agentDir(), "settings.json"),
    backupDir: dirname(modelsPath),
    authPath: join(agentDir(), "auth.json"),
  };
}

// ─── Effective catalog + reference catalog from the live registry ────────────

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

/** Structural slice of pi 0.84.1 ModelRegistry (getAll/find/hasConfiguredAuth). */
interface RegistryLike {
  getAll(): readonly EffectiveLike[];
  find(provider: string, modelId: string): EffectiveLike | undefined;
  hasConfiguredAuth(model: EffectiveLike): boolean;
}

function normalizeEffective(provider: string, model: EffectiveLike): NormalizedModel {
  const source = "pi:effective";
  const ev = <T>(value: T, plannable = true): Evidence<T> => ({ value, source, confidence: "authoritative", plannable });
  const out: NormalizedModel = { provider, id: model.id };
  out.name = ev(model.name);
  out.api = ev(model.api);
  out.reasoning = ev(model.reasoning);
  if (model.thinkingLevelMap) out.thinkingLevelMap = ev(model.thinkingLevelMap);
  out.input = ev([...model.input] as ("text" | "image")[]);
  out.cost = ev(model.cost);
  out.contextWindow = ev(model.contextWindow);
  // pi's provider-composer fills missing maxTokens with its tiny 16384 default;
  // without an explicit value the entry really is UNKNOWN. Emitting the default
  // here would let the go-2 mirror stamp 16384 as authoritative, so treat
  // default-valued limits as missing instead of a real ceiling.
  if (model.maxTokens !== PI_DEFAULT_MAX_OUTPUT_TOKENS) out.maxTokens = ev(model.maxTokens);
  if (model.compat) out.compat = ev(model.compat);
  return out;
}

/** Snapshot of EVERY provider Pi serves — built-in catalog entries included.
 *  Generic sync uses it to avoid duplicating models Pi already provides. */
function buildEffective(registry: RegistryLike): EffectiveSnapshot {
  let all: readonly EffectiveLike[];
  try {
    all = registry.getAll();
  } catch {
    all = [];
  }
  const providers: Record<string, NormalizedModel[]> = {};
  for (const p of MANAGED_PROVIDERS) providers[p] = [];
  for (const m of all) {
    if (!m || typeof m !== "object" || typeof m.provider !== "string" || typeof m.id !== "string") continue;
    (providers[m.provider] ??= []).push(normalizeEffective(m.provider, m));
  }
  return { providers };
}

function buildRefCatalog(registry: RegistryLike): RefCatalog {
  return {
    modelIds(provider: string): readonly string[] {
      try {
        return registry
          .getAll()
          .filter((m) => m.provider === provider && typeof m.id === "string")
          .map((m) => m.id);
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

// ─── Secret-safe + terminal-safe formatting ─────────────────────────────────

const FIELD_LABEL: Record<MutationPath, string> = {
  name: "name",
  api: "api",
  baseUrl: "baseUrl",
  reasoning: "reasoning",
  thinkingLevelMap: "thinking",
  input: "input",
  contextWindow: "contextWindow",
  maxTokens: "maxTokens",
  cost: "cost",
  compat: "compat",
};

/**
 * Strip ALL C0/C1/DEL control bytes — including LF (\n, 0x0A), TAB (0x09),
 * CR (0x0D), ESC (0x1B), DEL (0x7F) and the C1 range — from an UNTRUSTED
 * field before it reaches the UI/console/session log. Replaces each with
 * U+FFFD so a malicious model id / error / characteristic cannot inject terminal
 * escape sequences, carriage-return overwrites, or forge report lines.
 */
const CONTROL_RE = /[\u0000-\u001F\u007F-\u009F]/g;
export function sanitize(text: string): string {
  return text.replace(CONTROL_RE, "\uFFFD");
}

/** Sanitize an assembled multi-line report line-by-line, preserving layout \n. */
function reportSanitized(text: string): string {
  return text.split("\n").map(sanitize).join("\n");
}

/** Render a value for display. Only allowlisted metadata is ever passed here. */
function renderValue(v: unknown): string {
  if (v === undefined) return "(absent)";
  if (v === null) return "(null)";
  return sanitize(JSON.stringify(v));
}

/** Escape a sanitized string for a Markdown table cell. */
function mdCell(text: string): string {
  return sanitize(text).replace(/\|/g, "\\|");
}

/** Render a value for a Markdown table cell (compact, pipe-escaped). */
function mdValue(v: unknown): string {
  if (v === undefined) return "—";
  if (v === null) return "(null)";
  return mdCell(JSON.stringify(v));
}

/** Raw JSON equality for subfield comparison (values are plain JSON). */
function rawEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}
function asObj(v: unknown): Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {};
}

/** Composite-field subfield diffing lives in fmtUpdatedRows (Markdown rows).
 *  As there, the complete prior composite value is NEVER disclosed (it may
 *  carry unknown/secret members) - only allowlisted subfield-level diffs. */

/** One "Added" table row: model | thinking levels | ctx/max out | input. */
function fmtAddedRow(r: ModelResult): string {
  const c = r.characteristics;
  const model = mdCell(`${r.provider}/${r.modelId}`);
  const map = c?.thinkingLevelMap;
  const thinking =
    map && typeof map === "object" && Object.keys(map).length > 0
      ? Object.keys(map).sort().join(", ")
      : "—";
  const ctx = c?.contextWindow !== undefined ? String(c.contextWindow) : "?";
  const max = c?.maxTokens !== undefined ? String(c.maxTokens) : "?";
  const input =
    Array.isArray(c?.input) && c.input.length > 0
      ? (c!.input as readonly string[]).join("+")
      : "—";
  return `| \`${model}\` | ${mdCell(thinking)} | ${mdCell(ctx)} / ${mdCell(max)} | ${mdCell(input)} |`;
}

/** One or more "Updated" table rows: model | field | was → now. */
function fmtUpdatedRows(r: ModelResult): string[] {
  const model = mdCell(`${r.provider}/${r.modelId}`);
  const rows: string[] = [];
  for (const ch of r.changes ?? []) {
    if (isCompositePath(ch.path)) {
      const before = asObj(ch.before);
      const after = asObj(ch.after);
      let emitted = false;
      for (const k of allowlistedSubfields(ch.path)) {
        if (Object.prototype.hasOwnProperty.call(after, k)) {
          const bv = Object.prototype.hasOwnProperty.call(before, k) ? before[k] : undefined;
          if (!rawEqual(bv, after[k])) {
            rows.push(`| \`${model}\` | ${mdCell(`${FIELD_LABEL[ch.path]}.${k}`)} | ${mdValue(bv)} → ${mdValue(after[k])} |`);
            emitted = true;
          }
        }
      }
      if (!emitted) rows.push(`| \`${model}\` | ${mdCell(FIELD_LABEL[ch.path])} | updated |`);
    } else {
      rows.push(`| \`${model}\` | ${mdCell(FIELD_LABEL[ch.path])} | ${mdValue(ch.before)} → ${mdValue(ch.after)} |`);
    }
  }
  if (rows.length === 0) rows.push(`| \`${model}\` | (none) | |`);
  return rows;
}

function fmtTotal(t: ProviderTotal): string {
  return `| ${mdCell(t.provider)} | ${t.added} | ${t.updated} | ${t.skipped} | ${t.unchanged} | ${mdCell(t.sourceStatus)} |`;
}

/** Build the user-facing result report from a reconcile report + apply result. */
/** Build the user-facing Markdown report from a reconcile report + apply result.
 *
 * Compact by design: per-provider totals table, an Added table (thinking /
 * ctx / max out / input), an Updated was->now table, and skipped models as a
 * bare name list. Diagnostic warnings and provenance noise are omitted from
 * the chat rendering - the reconcile still records them internally. */
export function buildResultReport(report: ReconcileReport, result: ApplyResult): string {
  if (!result.ok) {
    const lines = ["**model-doctor - update refused**", "", sanitize(result.error ?? "unknown error")];
    // Refusals carry structured findings (allowlist/schema violations, drift
    // guards). Without them the user sees only an opaque one-liner.
    if (result.findings.length > 0) {
      lines.push("", `reasons (${result.findings.length}):`);
      for (const f of result.findings) {
        lines.push(`- \`${sanitize(f.code)}\`: ${sanitize(f.message)}`);
      }
    }
    return lines.join("\n");
  }
  const lines: string[] = [
    `**model-doctor** — applied ${result.applied} change(s) to models.json · backup: ${result.backups.map((b) => sanitize(b)).join(", ") || "(none)"}`,
    "",
    "| provider | added | updated | skipped | unchanged | source |",
    "|---|---|---|---|---|---|",
  ];
  for (const t of report.totals) lines.push(fmtTotal(t));

  const added = report.results.filter((r) => r.action === "added");
  const updated = report.results.filter((r) => r.action === "updated");
  const skipped = report.results.filter((r) => r.action === "skipped");

  if (added.length > 0) {
    lines.push("", "### Added (" + added.length + ")", "");
    lines.push("| model | thinking | ctx / max out | input |", "|---|---|---|---|");
    for (const r of added) lines.push(fmtAddedRow(r));
  }
  if (updated.length > 0) {
    lines.push("", "### Updated (" + updated.length + ")", "");
    lines.push("| model | field | was → now |", "|---|---|---|");
    for (const r of updated) lines.push(...fmtUpdatedRows(r));
  }
  if (skipped.length > 0) {
    // Reasons matter now that "no published limits" is a skip: group them so a
    // long listing stays readable instead of repeating one reason per line.
    lines.push("", "**Skipped (" + skipped.length + "):**");
    const byReason = new Map<string, string[]>();
    for (const r of skipped) {
      const name = "`" + mdCell(r.provider) + "/" + mdCell(r.modelId) + "`";
      const key = sanitize(r.reason ?? "skipped");
      const list = byReason.get(key) ?? [];
      list.push(name);
      byReason.set(key, list);
    }
    for (const [reason, names] of byReason) {
      lines.push("- " + reason + ": " + names.join(", "));
    }
  }
  return lines.join("\n");
}

/** Maximum characters emitted to the TUI/console. Beyond this the full sanitized
 *  report is written to a restrictive (0600) local file and its path is shown. */
const MAX_REPORT_CHARS = 20_000;

/** Bound a sanitized report: if it exceeds the limit, write the FULL sanitized
 *  report to a 0600 file and return a capped version that keeps totals/path + a
 *  truncation marker pointing at the file. Never sends an unbounded blob to the
 *  TUI or console. */
async function boundReport(safeText: string): Promise<{ shown: string }> {
  if (safeText.length <= MAX_REPORT_CHARS) return { shown: safeText };
  const stamp = new Date().toISOString().replace(/[^0-9]/g, "").slice(0, 17);
  const filePath = join(agentDir(), `model-doctor-report-${stamp}-${randomBytes(3).toString("hex")}.txt`);
  try {
    const fh = await open(filePath, "wx", 0o600);
    try {
      await fh.writeFile(safeText);
      await fh.sync();
    } finally {
      await fh.close();
    }
  } catch {
    // Could not write the full report file: still cap the inline output.
    return {
      shown:
        safeText.slice(0, MAX_REPORT_CHARS) +
        `\n…[truncated: ${safeText.length} chars; full report could not be written]`,
    };
  }
  return {
    shown:
      safeText.slice(0, MAX_REPORT_CHARS) +
      `\n…[truncated: ${safeText.length} chars; full sanitized report written to ${filePath} (mode 0600)]`,
  };
}

// ─── Command handler ─────────────────────────────────────────────────────────

function defaultFetch(): FetchFn {
  return globalThis.fetch as unknown as FetchFn;
}

export default function (pi: ExtensionAPI): void {
  pi.registerCommand("model-doctor", {
    description:
      "Reconcile all model catalogs (official vs pi vs models.json) in one safe pass and write models.json once. No subcommands or arguments.",
    handler: async (args: string, ctx: ExtensionCommandContext) => {
      const paths = defaultPaths();
      const fetchFn = defaultFetch();

      // Real pi 0.84.1 ModelRegistry surface (getAll/find/hasConfiguredAuth).
      const registry = ctx.modelRegistry as unknown as RegistryLike;
      const effective = buildEffective(registry);
      const refCatalog = buildRefCatalog(registry);

      const emit = (text: string, type: "info" | "warning" | "error" = "info"): void => {
        if (ctx.hasUI) {
          ctx.ui.notify(text, type);
        }
        // eslint-disable-next-line no-console
        console.log(text);
      };

      // Reports go to the chat transcript as a Markdown custom message (tables
      // render properly) instead of the dim one-line status text of notify().
      const sendReport = (text: string, type: "info" | "warning" | "error" = "info"): void => {
        if (ctx.hasUI && typeof pi.sendMessage === "function") {
          try {
            pi.sendMessage(
              { customType: "model-doctor-report", content: text, display: true },
              { triggerTurn: false },
            );
            return;
          } catch {
            // fall through to the notify fallback below
          }
        }
        emit(text, type);
      };

      // Blocker: any non-empty argument is rejected WITHOUT mutation. The command
      // takes no subcommands and no provider/model argument.
      if ((args ?? "").trim().length > 0) {
        emit(reportSanitized(
          `model-doctor: this command takes no arguments or subcommands (got ${JSON.stringify(args)}). Usage: /model-doctor\n${helpText()}`,
        ), "warning");
        return;
      }

      try {
        const { result, report: rec } = await runReconcile({ paths, fetchFn, effective, refCatalog });
        const full = reportSanitized(buildResultReport(rec, result));
        const { shown } = await boundReport(full);
        sendReport(shown, result.ok ? "info" : "warning");
      } catch (err) {
        sendReport(reportSanitized(`model-doctor: error — ${sanitize(messageOf(err))}`), "error");
      }
    },
  });
}

function helpText(): string {
  return [
    "model-doctor — one-shot model catalog reconciliation",
    "",
    "Usage:",
    "  /model-doctor     reconcile all managed providers and write models.json once",
    "",
    "One invocation fetches the official sources (OpenRouter, models.dev), reads",
    "Pi's effective catalog + your models.json/settings.json, then:",
 "  • updates drifted authoritative fields on existing managed entries;",
    "  • adds official models absent from Pi's effective registry (genuinely new),",
    "    only with published limits — never with guessed ctx / max out;",
    "  • mirrors opencode-go into opencode-go-2;",
    "  • skips unsafe/ambiguous models/fields and unknown-limits models, and reports them.",
    "",
    "Then it writes models.json atomically (timestamped backup + CAS + rollback)",
    "and prints a per-model result (added/updated/skipped) with characteristics",
    "and per-provider totals. No preview/confirmation step.",
    "",
    "Managed providers: openrouter, opencode, opencode-go, opencode-go-2.",
    "Only models.json is ever written; settings/models-store/auth are read-only.",
    "Provider apiKey/baseUrl/headers/auth are never changed or echoed.",
  ].join("\n");
}

function messageOf(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

// Exported for tests / programmatic use. Not part of the pi command surface.
export const __internals = {
  buildEffective,
  buildRefCatalog,
  buildResultReport,
  renderValue,
  sanitize,
  agentDir,
  defaultPaths,
  helpText,
  boundReport,
  MAX_REPORT_CHARS,
};

// Re-export key types/values for test imports.
export type {
  ModelsConfig,
  ProviderEntry,
  ModelEntry,
  ManagedProvider,
};
