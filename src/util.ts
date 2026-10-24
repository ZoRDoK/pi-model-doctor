/**
 * Small pure helpers shared by the doctor modules. No io, no state.
 */
import type { Evidence } from "./types.ts";

export function messageOf(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function asString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

/** Positive finite number, or undefined. Provider payloads often carry strings. */
export function asPositiveNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : undefined;
}

/** Metadata value plus where it came from. "plannable" is false for values
 *  that must be reported but never written to models.json. */
export function evidence<T>(value: T, source: string, plannable = true, note?: string): Evidence<T> {
  if (note === undefined) {
    return { value, source, plannable };
  }
  return { value, source, plannable, note };
}
