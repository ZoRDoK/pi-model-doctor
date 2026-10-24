# Changelog

## 1.1.0 — 2026-10-24

### Fixed

- Generic providers no longer duplicate models pi already serves from its own
  catalog (z.ai's glm models were copied into models.json as metadata-less
  duplicates).
- A model is added only when the source publishes **both** the context window
  and the max output. The invented 65536 fallback is gone; unknown-limits models
  are reported as skipped instead of silently running on pi's defaults.
- OpenAI-compatible listings now read published limits
  (`context_length`/`context_window`, `max_completion_tokens`/`max_output_tokens`).

### Changed

- Sources live in `src/`; thinking-level resolution moved into `sources.ts`;
  report rendering extracted to `report.ts`; env/path access isolated in
  `paths.ts`.
- Removed dead code (the unused `confidence` evidence field, unused exports,
  the duplicate-ops pass) and moved repeated helpers into `util.ts`.
- Skipped models are grouped by reason in the report.
- Tests, typecheck and CI are unchanged in behavior (176 tests, Node 24/25).

## 1.0.0

Initial release.
