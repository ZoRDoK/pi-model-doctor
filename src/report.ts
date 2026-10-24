/**
 * Markdown report rendering for /model-doctor, plus the terminal-safe
 * sanitizer every untrusted string passes through.
 */
import { randomBytes } from "node:crypto";
import { open } from "node:fs/promises";
import { join } from "node:path";

import { agentDir } from "./paths.ts";
import type { ApplyResult, MutationPath } from "./types.ts";
import { allowlistedSubfields, isCompositePath } from "./types.ts";
import type { ModelResult, ProviderTotal, ReconcileReport } from "./doctor.ts";
import { isObject } from "./util.ts";

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

/** Untrusted strings (model ids, provider payloads, error text) must never
 *  reach the terminal or session log with control bytes: replace them with
 *  U+FFFD so they cannot inject escapes or forge report lines. */
const CONTROL_RE = /[\u0000-\u001F\u007F-\u009F]/g;

export function sanitize(text: string): string {
  return text.replace(CONTROL_RE, "\uFFFD");
}

export function reportSanitized(text: string): string {
  return text.split("\n").map(sanitize).join("\n");
}

/** Render a value for display. Only allowlisted metadata is passed here. */
function renderValue(value: unknown): string {
  if (value === undefined) return "(absent)";
  if (value === null) return "(null)";
  return sanitize(JSON.stringify(value));
}

/** Escape for a Markdown table cell. */
function mdCell(text: string): string {
  return sanitize(text).replace(/\|/g, "\\|");
}

function mdValue(value: unknown): string {
  if (value === undefined) return "—";
  if (value === null) return "(null)";
  return mdCell(JSON.stringify(value));
}

function rawEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

/** Added row: model | thinking levels | ctx / max out | input. */
function fmtAddedRow(result: ModelResult): string {
  const c = result.characteristics;
  const model = mdCell(`${result.provider}/${result.modelId}`);
  const levels = c?.thinkingLevelMap ? Object.keys(c.thinkingLevelMap).sort().join(", ") : "—";
  const ctx = c?.contextWindow !== undefined ? String(c.contextWindow) : "?";
  const max = c?.maxTokens !== undefined ? String(c.maxTokens) : "?";
  const input = Array.isArray(c?.input) && c.input.length > 0 ? c.input.join("+") : "—";

  return `| \`${model}\` | ${mdCell(levels)} | ${mdCell(ctx)} / ${mdCell(max)} | ${mdCell(input)} |`;
}

/** Updated rows: model | field | was → now. Composite values are diffed per
 *  allowlisted subfield only, so unknown members are never disclosed. */
function fmtUpdatedRows(result: ModelResult): string[] {
  const model = mdCell(`${result.provider}/${result.modelId}`);
  const rows: string[] = [];

  for (const change of result.changes ?? []) {
    const label = FIELD_LABEL[change.path];

    if (!isCompositePath(change.path)) {
      rows.push(`| \`${model}\` | ${mdCell(label)} | ${mdValue(change.before)} → ${mdValue(change.after)} |`);
      continue;
    }

    const before = isObject(change.before) ? change.before : {};
    const after = isObject(change.after) ? change.after : {};
    let emitted = false;

    for (const key of allowlistedSubfields(change.path)) {
      if (!(key in after)) continue;

      const was = key in before ? before[key] : undefined;

      if (!rawEqual(was, after[key])) {
        rows.push(`| \`${model}\` | ${mdCell(`${label}.${key}`)} | ${mdValue(was)} → ${mdValue(after[key])} |`);
        emitted = true;
      }
    }
    if (!emitted) rows.push(`| \`${model}\` | ${mdCell(label)} | updated |`);
  }

  if (rows.length === 0) rows.push(`| \`${model}\` | (none) | |`);
  return rows;
}

function fmtTotal(total: ProviderTotal): string {
  return `| ${mdCell(total.provider)} | ${total.added} | ${total.updated} | ${total.skipped} | ${total.unchanged} | ${mdCell(total.sourceStatus)} |`;
}

/** Chat-sized report: totals, added, updated, and skipped grouped by reason.
 *  Internal warnings and provenance noise stay out of the transcript. */
export function buildResultReport(report: ReconcileReport, result: ApplyResult): string {
  if (!result.ok) return refusalReport(result);

  const lines: string[] = [
    `**model-doctor** — applied ${result.applied} change(s) to models.json · backup: ${result.backups.map(sanitize).join(", ") || "(none)"}`,
    "",
    "| provider | added | updated | skipped | unchanged | source |",
    "|---|---|---|---|---|---|",
    ...report.totals.map(fmtTotal),
  ];

  const added = report.results.filter((r) => r.action === "added");
  const updated = report.results.filter((r) => r.action === "updated");
  const skipped = report.results.filter((r) => r.action === "skipped");

  if (added.length > 0) {
    lines.push("", `### Added (${added.length})`, "");
    lines.push("| model | thinking | ctx / max out | input |", "|---|---|---|---|");
    for (const r of added) lines.push(fmtAddedRow(r));
  }

  if (updated.length > 0) {
    lines.push("", `### Updated (${updated.length})`, "");
    lines.push("| model | field | was → now |", "|---|---|---|");
    for (const r of updated) lines.push(...fmtUpdatedRows(r));
  }

  if (skipped.length > 0) {
    lines.push("", `**Skipped (${skipped.length}):**`);
    for (const [reason, names] of groupByReason(skipped)) {
      lines.push(`- ${reason}: ${names.join(", ")}`);
    }
  }

  return lines.join("\n");
}

/** A refused write explains itself: without the findings the user sees an
 *  opaque one-liner. */
function refusalReport(result: ApplyResult): string {
  const lines = ["**model-doctor - update refused**", "", sanitize(result.error ?? "unknown error")];

  if (result.findings.length > 0) {
    lines.push("", `reasons (${result.findings.length}):`);
    for (const f of result.findings) {
      lines.push(`- \`${sanitize(f.code)}\`: ${sanitize(f.message)}`);
    }
  }

  return lines.join("\n");
}

function groupByReason(skipped: readonly ModelResult[]): Map<string, string[]> {
  const groups = new Map<string, string[]>();

  for (const result of skipped) {
    const name = `\`${mdCell(result.provider)}/${mdCell(result.modelId)}\``;
    const reason = sanitize(result.reason ?? "skipped");
    groups.set(reason, [...(groups.get(reason) ?? []), name]);
  }

  return groups;
}

/** Inline output cap; beyond it the full report goes to a 0600 file. */
export const MAX_REPORT_CHARS = 20_000;

export async function boundReport(safeText: string): Promise<{ shown: string }> {
  if (safeText.length <= MAX_REPORT_CHARS) return { shown: safeText };

  const stamp = new Date().toISOString().replace(/[^0-9]/g, "").slice(0, 17);
  const filePath = join(agentDir(), `model-doctor-report-${stamp}-${randomBytes(3).toString("hex")}.txt`);
  const marker = `\n…[truncated: ${safeText.length} chars`;

  try {
    const handle = await open(filePath, "wx", 0o600);
    try {
      await handle.writeFile(safeText);
      await handle.sync();
    } finally {
      await handle.close();
    }
  } catch {
    return { shown: safeText.slice(0, MAX_REPORT_CHARS) + marker + "; full report could not be written]" };
  }

  return { shown: safeText.slice(0, MAX_REPORT_CHARS) + marker + `; full sanitized report written to ${filePath} (mode 0600)]` };
}

export const __reportInternals = { renderValue };
