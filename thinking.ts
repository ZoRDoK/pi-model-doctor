/**
 * Cross-provider thinking-level resolution.
 *
 * Neither the OpenCode Zen/Go live catalogs nor models.dev publish per-model
 * reasoning-effort levels, so a managed model's thinkingLevelMap is resolved
 * from two machine-readable catalogs that do:
 *
 *  - OpenRouter `reasoning.supported_efforts` — the popular provider; wins.
 *  - Charm Hyper `reasoning.effort_levels` — fills gaps when OpenRouter has
 *    nothing for the model (Hyper's data is sometimes incomplete/broken, so it
 *    never overrides OpenRouter evidence).
 *
 * Policy (fail-closed): when both sources describe the same model and the level
 * sets disagree, the map is NOT auto-applied — the evidence is emitted
 * non-plannable with both sets in the note, so the model surfaces as "skipped
 * (ambiguous)" instead of receiving a possibly wrong map.
 *
 * Evidence already carried by the model (a direct OpenRouter list map, or a
 * pi:effective built-in map on the go-2 mirror source) always wins and is never
 * replaced by the resolver.
 */
import { slugOf } from "./sources.ts";
import type {
  Diagnosis,
  Evidence,
  ManagedProvider,
  NormalizedModel,
} from "./types.ts";

/** Level-set provenance used in evidence sources and disagreement notes. */
const SOURCE_OPENROUTER = "openrouter:efforts";
const SOURCE_HYPER = "hyper:efforts";
const SOURCE_AGREED = "openrouter+hyper:efforts";

interface LevelSet {
  readonly map: Readonly<Record<string, string | null>>;
  readonly source: string;
}

/** OpenRouter lookup outcome: a level set, no data, or internal ambiguity
 *  (several same-slug models with differing level sets). */
type OpenRouterLookup =
  | { readonly kind: "set"; readonly set: LevelSet }
  | { readonly kind: "none" }
  | { readonly kind: "ambiguous" };

/** Resolve the thinkingLevelMap evidence for one managed model from
 *  OpenRouter + Hyper. Undefined when neither source knows the model. */
export function resolveThinkingLevels(
  provider: ManagedProvider,
  modelId: string,
  diagnosis: Diagnosis,
): Evidence<Readonly<Record<string, string | null>>> | undefined {
  const fromOpenRouter = levelsFromOpenRouter(provider, modelId, diagnosis);
  const fromHyper = levelsFromHyper(provider, modelId, diagnosis);

  // Neither source reports levels: unknown, never an empty map.
  if (fromOpenRouter.kind === "none" && fromHyper === undefined) return undefined;

  if (fromOpenRouter.kind === "ambiguous") {
    // Several OpenRouter models share the slug with differing level sets:
    // nothing is trustworthy — never fall back to the less-authoritative
    // hyper catalog for this model.
    return {
      value: {},
      source: SOURCE_OPENROUTER,
      confidence: "reported",
      plannable: false,
      note: "multiple openrouter models match the slug with differing effort levels; not auto-applied",
    };
  }

  if (fromOpenRouter.kind === "none") {
    return fromHyper === undefined ? undefined : evPlannable(fromHyper);
  }

  if (fromHyper === undefined) {
    return evPlannable(fromOpenRouter.set);
  }

  if (mapsEqual(fromOpenRouter.set.map, fromHyper.map)) {
    return {
      value: fromOpenRouter.set.map,
      source: SOURCE_AGREED,
      confidence: "reported",
      plannable: true,
    };
  }
  // Both describe the model but disagree: never auto-apply either map.
  return {
    value: fromOpenRouter.set.map,
    source: SOURCE_OPENROUTER,
    confidence: "reported",
    plannable: false,
    note: `thinking levels disagree — openrouter ${describeLevels(fromOpenRouter.set.map)} vs hyper ${describeLevels(fromHyper.map)}; not auto-applied`,
  };
}

/** Attach resolved thinking levels (and the implied reasoning flag) to a
 *  source/effective model. Existing evidence is preserved as-is. */
export function enrichThinkingLevels(
  provider: ManagedProvider,
  nm: NormalizedModel,
  diagnosis: Diagnosis,
): NormalizedModel {
  if (nm.thinkingLevelMap !== undefined) return nm;
  const resolved = resolveThinkingLevels(provider, nm.id, diagnosis);
  if (resolved === undefined) return nm;
  const out: NormalizedModel = { ...nm, thinkingLevelMap: resolved };
  if (
    resolved.plannable &&
    (nm.reasoning === undefined || !nm.reasoning.plannable || nm.reasoning.value !== true)
  ) {
    out.reasoning = {
      value: true,
      source: resolved.source,
      confidence: "reported",
      plannable: true,
    };
  }
  return out;
}

/** OpenRouter lookup: exact id for openrouter-managed models, unique slug
 *  match for zen/go bare ids. Several models with the same slug must agree,
 *  otherwise the OpenRouter side itself is ambiguous. */
function levelsFromOpenRouter(
  provider: ManagedProvider,
  modelId: string,
  diagnosis: Diagnosis,
): OpenRouterLookup {
  const source = diagnosis.sources.find((s) => s.id === "openrouter");
  if (!source || source.status !== "ok") return { kind: "none" };
  const matches = source.models.filter((m) => {
    if (m.thinkingLevelMap === undefined) return false;
    return provider === "openrouter" ? m.id === modelId : slugOf(m.id) === modelId;
  });
  if (matches.length === 0) return { kind: "none" };
  const first = matches[0]?.thinkingLevelMap;
  if (first === undefined) return { kind: "none" };
  for (const m of matches) {
    const levels = m.thinkingLevelMap;
    if (levels === undefined || !mapsEqual(first.value, levels.value)) {
      return { kind: "ambiguous" }; // slug collision with differing levels
    }
  }
  return { kind: "set", set: { map: first.value, source: SOURCE_OPENROUTER } };
}

/** Hyper lookup: exact bare id for zen/go models, slug match for
 *  openrouter-managed (vendor-prefixed) ids. Duplicate hyper ids are rejected
 *  at normalization, so at most one match exists. */
function levelsFromHyper(
  provider: ManagedProvider,
  modelId: string,
  diagnosis: Diagnosis,
): LevelSet | undefined {
  const source = diagnosis.sources.find((s) => s.id === "hyper");
  if (!source || source.status !== "ok") return undefined;
  const matches = source.models.filter((m) => {
    if (m.thinkingLevelMap === undefined) return false;
    return provider === "openrouter" ? slugOf(modelId) === m.id : m.id === modelId;
  });
  const levels = matches[0]?.thinkingLevelMap;
  if (levels === undefined) return undefined;
  return { map: levels.value, source: SOURCE_HYPER };
}

function evPlannable(set: LevelSet): Evidence<Readonly<Record<string, string | null>>> {
  return { value: set.map, source: set.source, confidence: "reported", plannable: true };
}

/** Canonical equality for level maps: same non-null level->effort bindings. */
function mapsEqual(
  a: Readonly<Record<string, string | null>>,
  b: Readonly<Record<string, string | null>>,
): boolean {
  return canonicalKey(a) === canonicalKey(b);
}

function canonicalKey(map: Readonly<Record<string, string | null>>): string {
  const entries = Object.entries(map)
    .filter(([, v]) => v !== null)
    .sort(([x], [y]) => (x < y ? -1 : x > y ? 1 : 0));
  return JSON.stringify(entries);
}

/** Compact human-readable level set for disagreement notes (non-null only,
 *  in canonical pi order). */
const PI_LEVEL_ORDER: readonly string[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

function describeLevels(map: Readonly<Record<string, string | null>>): string {
  const levels = Object.entries(map)
    .filter(([, v]) => v !== null)
    .map(([k]) => k)
    .sort((x, y) => PI_LEVEL_ORDER.indexOf(x) - PI_LEVEL_ORDER.indexOf(y));
  return `[${levels.join(",")}]`;
}
