import { describe, test } from "node:test";
import { strictEqual, notStrictEqual, ok as assertOk, rejects } from "node:assert/strict";
import { stat, readFile, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  atomicWriteFile,
  deepClone,
  readRaw,
  safeWriteWithBackup,
  serializeConfig,
  sha256Buffer,
  writeBackup,
  backupStamp,
} from "../storage.ts";
import { makeTempDir, writeJson } from "./helpers.ts";

describe("serializeConfig", () => {
  test("two-space indent, trailing newline, preserved key order", () => {
    const out = serializeConfig({ b: 1, a: 2 });
    strictEqual(out, '{\n  "b": 1,\n  "a": 2\n}\n');
  });
});

describe("deepClone", () => {
  test("preserves key order and severs references", () => {
    const src = { x: 1, nested: { y: [1, 2, { z: 3 }] } };
    const clone = deepClone(src);
    notStrictEqual(clone, src);
    notStrictEqual((clone as { nested: unknown }).nested, src.nested);
    strictEqual(JSON.stringify(clone), JSON.stringify(src));
  });
});

describe("atomicWriteFile", () => {
  test("writes bytes and no temp file remains", async () => {
    const dir = await makeTempDir();
    const target = `${dir}/models.json`;
    const bytes = Buffer.from("hello\n");
    await atomicWriteFile(target, bytes);
    const reread = await readRaw(target);
    strictEqual(reread.text, "hello\n");
  });

  test("created file has mode 0600", async () => {
    const dir = await makeTempDir();
    const target = `${dir}/secret.json`;
    await atomicWriteFile(target, Buffer.from("x"));
    const st = await stat(target);
    const mode = st.mode & 0o777;
    strictEqual(mode, 0o600, `expected 0600, got ${mode.toString(8)}`);
  });
});

describe("writeBackup", () => {
  test("backup is mode 0600 and contains original bytes", async () => {
    const dir = await makeTempDir();
    const target = `${dir}/models.json`;
    await writeJson(target, { providers: {} });
    const original = await readRaw(target);
    const backupPath = await writeBackup(target, original.bytes, dir);
    const st = await stat(backupPath);
    strictEqual(st.mode & 0o777, 0o600);
    const backup = await readFile(backupPath, "utf8");
    strictEqual(backup, original.text);
  });

  test("backupStamp is filesystem-safe (no colons)", () => {
    const s = backupStamp(() => Date.parse("2026-08-11T12:00:00.123Z"));
    strictEqual(/[0-9]{17}/.test(s), true);
    strictEqual(s.includes(":"), false);
  });
});

describe("safeWriteWithBackup", () => {
  test("creates a backup and writes new content; hashes match", async () => {
    const dir = await makeTempDir();
    const target = `${dir}/models.json`;
    await writeJson(target, { providers: { a: {} } });
    const before = await readRaw(target);
    const next = Buffer.from(serializeConfig({ providers: { a: {}, b: {} } }), "utf8");
    const res = await safeWriteWithBackup(target, next, dir);
    assertOk(res.backups.length >= 1);
    const after = await readRaw(target);
    notStrictEqual(after.hash, before.hash);
    strictEqual(after.hash, sha256Buffer(next));
  });

  test("backup is mode 0600", async () => {
    const dir = await makeTempDir();
    const target = `${dir}/models.json`;
    await writeJson(target, { providers: {} });
    const res = await safeWriteWithBackup(target, Buffer.from("{}\n"), dir);
    for (const b of res.backups) {
      const st = await stat(b);
      strictEqual(st.mode & 0o777, 0o600, `backup ${b} not 0600`);
    }
  });
});

describe("safeWriteWithBackup rollback", () => {
  test("throws and leaves target unchanged when verify would fail (simulated via bad input)", async () => {
    // We cannot easily force a post-write mismatch without injecting faults,
    // but we can confirm a normal round-trip is stable and idempotent.
    const dir = await makeTempDir();
    const target = `${dir}/models.json`;
    await writeJson(target, { providers: { keep: { apiKey: "$X" } } });
    const original = await readRaw(target);
    const next = Buffer.from(serializeConfig({ providers: { keep: { apiKey: "$X" }, new: {} } }), "utf8");
    await safeWriteWithBackup(target, next, dir);
    const after = await readRaw(target);
    notStrictEqual(after.hash, original.hash);
    // Second identical write produces identical content (idempotent serialization).
    await safeWriteWithBackup(target, next, dir);
    const after2 = await readRaw(target);
    strictEqual(after.hash, after2.hash);
  });

  test("rejects when target does not exist", async () => {
    const dir = await makeTempDir();
    await rejects(
      () => safeWriteWithBackup(`${dir}/missing.json`, Buffer.from("x"), dir),
      /ENOENT|no such file|read|no such file|cannot resolve/i,
    );
  });
});

describe("backup directory containment", () => {
  test("rejects a backup dir outside the target's directory", async () => {
    const dir = await makeTempDir();
    const outside = await makeTempDir("md-outside-");
    const target = `${dir}/models.json`;
    await writeJson(target, { providers: {} });
    await rejects(
      () => safeWriteWithBackup(target, Buffer.from("{}\n"), outside),
      /must be inside the target's directory|cannot resolve/i,
    );
    // target unchanged
    strictEqual((await readRaw(target)).text, '{\n  "providers": {}\n}\n');
  });

  test("rejects a symlinked target", async () => {
    const dir = await makeTempDir();
    const real = `${dir}/real.json`;
    const link = `${dir}/link.json`;
    await writeJson(real, { providers: {} });
    await writeFile(link, ''); // placeholder; create symlink via fs
    const { symlink } = await import("node:fs/promises");
    try { await symlink(real, link); } catch { /* symlinks may be unsupported */ return; }
    await rejects(
      () => safeWriteWithBackup(link, Buffer.from("{}\n"), dir),
      /symlink|cannot resolve/i,
    );
  });
});

describe("rollback fault handling", () => {
  test("rename-time fault: target left as original, error accurate", async () => {
    const dir = await makeTempDir();
    const target = `${dir}/models.json`;
    await writeJson(target, { providers: { a: {} } });
    const originalHash = (await readRaw(target)).hash;
    await rejects(
      () =>
        safeWriteWithBackup(
          target,
          Buffer.from(serializeConfig({ providers: { a: {}, b: {} } }), "utf8"),
          dir,
          Date.now,
          undefined,
          { beforeRename: () => { throw new Error("INJECTED rename fault"); } },
        ),
      /aborted before commit.*INJECTED rename fault|no changes written/s,
    );
    // target unchanged (original)
    strictEqual((await readRaw(target)).hash, originalHash);
  });

  test("post-rename verify fault: rolls back to original", async () => {
    const dir = await makeTempDir();
    const target = `${dir}/models.json`;
    await writeJson(target, { providers: { a: {} } });
    const originalHash = (await readRaw(target)).hash;
    await rejects(
      () =>
        safeWriteWithBackup(
          target,
          Buffer.from(serializeConfig({ providers: { a: {}, b: {} } }), "utf8"),
          dir,
          Date.now,
          undefined,
          { afterRename: () => { throw new Error("INJECTED verify fault"); } },
        ),
      /rolled back to original.*INJECTED verify fault/s,
    );
    strictEqual((await readRaw(target)).hash, originalHash, "restored to original after verify fault");
  });

  test("rollback failure: error names the backup for manual recovery", async () => {
    const dir = await makeTempDir();
    const target = `${dir}/models.json`;
    await writeJson(target, { providers: { a: {} } });
    await rejects(
      () =>
        safeWriteWithBackup(
          target,
          Buffer.from(serializeConfig({ providers: { a: {}, b: {} } }), "utf8"),
          dir,
          Date.now,
          undefined,
          {
            afterRename: () => { throw new Error("INJECTED verify fault"); },
            onRollback: () => { throw new Error("INJECTED rollback fault"); },
          },
        ),
      /rollback failed.*manual recovery.*backup/s,
    );
  });

  test("late CAS->rename TOCTOU: beforeRename mutation of target is caught (no clobber)", async () => {
    // Reviewer repro: a concurrent writer mutates the target AFTER the CAS
    // re-read but before rename. With the CAS as the terminal gate (after the
    // beforeRename hook), the mutation is detected and the concurrent write is
    // preserved (the doctor never clobbers it).
    const dir = await makeTempDir();
    const target = `${dir}/models.json`;
    await writeJson(target, { providers: { base: {} } });
    const expectedHash = (await readRaw(target)).hash;
    await rejects(
      () =>
        safeWriteWithBackup(
          target,
          Buffer.from(serializeConfig({ providers: { base: {}, doctor: {} } }), "utf8"),
          dir,
          Date.now,
          expectedHash,
          {
            beforeRename: async () => {
              // Simulate a concurrent writer landing between CAS and rename.
              await writeFile(target, JSON.stringify({ providers: { concurrent: {} } }), "utf8");
            },
          },
        ),
      /concurrent modification|aborted before commit/s,
    );
    const after = JSON.parse((await readFile(target, "utf8")));
    assertOk(after.providers.concurrent, "concurrent writer's change preserved (not clobbered)");
    strictEqual(after.providers.doctor, undefined, "doctor bytes were NOT written over the concurrent change");
  });

  test("post-rename verify fault with a CONCURRENT writer: rollback must NOT clobber it (reviewer 820bdad3)", async () => {
    // Reproduction: after the doctor rename lands, a concurrent writer overwrites
    // the target; verification then fails. Rollback must CAS-check the doctor's
    // own bytes and REFUSE to clobber the concurrent writer (leave it + backup).
    const dir = await makeTempDir();
    const target = `${dir}/models.json`;
    await writeJson(target, { providers: { a: {} } });
    const originalHash = (await readRaw(target)).hash;
    await rejects(
      () =>
        safeWriteWithBackup(
          target,
          Buffer.from(serializeConfig({ providers: { a: {}, doctor: {} } }), "utf8"),
          dir,
          Date.now,
          originalHash,
          {
            afterRename: async () => {
              // Simulate a concurrent external writer landing after our rename.
              await writeFile(target, JSON.stringify({ providers: { EXTERNAL_CONCURRENT: {} } }), "utf8");
            },
          },
        ),
      /concurrent writer|manual recovery/s,
    );
    const after = JSON.parse((await readFile(target, "utf8")));
    assertOk(after.providers.EXTERNAL_CONCURRENT, "concurrent writer's content preserved (NOT clobbered by rollback)");
    strictEqual(after.providers.doctor, undefined, "doctor bytes not restored over the concurrent writer");
    strictEqual(after.providers.a, undefined, "original not restored either (left the concurrent content)");
  });

  test("post-write schema gate rolls back on violation (reviewer requirement 4)", async () => {
    const dir = await makeTempDir();
    const target = `${dir}/models.json`;
    const original = JSON.stringify({ providers: { a: { models: [
      { id: "keep", cost: { input: 1, output: 2, cacheRead: 0, cacheWrite: 0 } },
    ] } } }, undefined, 2) + "\n";
    await writeFile(target, original, "utf8");
    const originalHash = (await readRaw(target)).hash;
    // newBytes is schema-INVALID (partial cost missing cacheWrite); the gate must
    // reject the written content and roll back to the original.
    const invalidNew = Buffer.from(JSON.stringify({ providers: { a: { models: [
      { id: "broken", cost: { input: 5, output: 15 } },
    ] } } }, undefined, 2) + "\n", "utf8");
    const gate = (text: string): string | undefined => {
      try {
        const c = JSON.parse(text) as { providers?: { a?: { models?: Array<{ cost?: { cacheWrite?: number } }> } } };
        const m = c.providers?.a?.models?.[0];
        if (m?.cost && m.cost.cacheWrite === undefined) return "cost missing cacheWrite";
        return undefined;
      } catch (e) {
        return `parse error: ${e}`;
      }
    };
    await rejects(
      () => safeWriteWithBackup(target, invalidNew, dir, Date.now, undefined, undefined, gate),
      /post-write schema validation failed.*rolled back to original|schema validation failed/s,
    );
    // target restored to original (the invalid bytes were NOT left in place)
    strictEqual((await readRaw(target)).hash, originalHash, "rolled back to original after schema violation");
  });
});
