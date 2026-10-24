/**
 * Storage primitives: hashing, deterministic JSON, 0600 backups and an atomic
 * write with verify + rollback. Every path is injected, so tests run in temp
 * dirs. node:* imports only.
 */
import { createHash, randomBytes } from "node:crypto";
import { lstat, mkdir, open, readFile, realpath, rename, unlink } from "node:fs/promises";
import { basename, dirname, join } from "node:path";

import { messageOf } from "./util.ts";

export interface RawRead {
  readonly bytes: Buffer;
  readonly text: string;
  readonly hash: string;
}

/** Read a file's raw bytes, decoded text, and sha256 hash. */
export async function readRaw(path: string): Promise<RawRead> {
  const bytes = await readFile(path);
  return { bytes, text: bytes.toString("utf8"), hash: sha256Buffer(bytes) };
}

/** sha256 hex of raw bytes. */
export function sha256Buffer(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

/**
 * Deterministic serialization matching pi's models.json format: object key
 * insertion order preserved (JSON.parse keeps order), two-space indent, and a
 * trailing newline. Numbers and booleans are emitted unchanged.
 */
export function serializeConfig(value: unknown): string {
  return JSON.stringify(value, undefined, 2) + "\n";
}

/** Parse JSON text; throws on malformed input (caller decides severity). */
export function parseConfig(text: string): unknown {
  return JSON.parse(text);
}

/** Deep clone that preserves key insertion order. */
export function deepClone<T>(value: T): T {
  return structuredClone(value);
}

/** Build a filesystem-safe timestamp for backup names. */
export function backupStamp(now: () => number = Date.now): string {
  // YYYYMMDDHHMMSSmmm in UTC, no colons (cross-filesystem safe).
  const d = new Date(now());
  const p = (n: number, w = 2) => String(n).padStart(w, "0");
  return (
    d.getUTCFullYear().toString() +
    p(d.getUTCMonth() + 1) +
    p(d.getUTCDate()) +
    p(d.getUTCHours()) +
    p(d.getUTCMinutes()) +
    p(d.getUTCSeconds()) +
    p(d.getUTCMilliseconds(), 3)
  );
}

/** Hooks injected by tests to simulate faults at specific points. */
export interface WriteHooks {
  /** After temp write+fsync, before rename. Throw to simulate a rename-time fault. */
  beforeRename?: () => void;
  /** After a successful rename, before verify. Throw/corrupt to simulate a verify fault. */
  afterRename?: (target: string) => void;
  /** Inside rollback. Throw to simulate rollback failure. */
  onRollback?: () => void;
}

export interface AtomicWriteOptions {
  /** If set, the target bytes are re-read and hashed right before rename; a mismatch aborts (compare-and-swap). */
  readonly expectedHash?: string;
  readonly hooks?: WriteHooks;
}

/**
 * Write `bytes` to `targetPath` atomically:
 *   open same-dir temp with O_EXCL + mode 0600 -> write -> fsync -> close
 *   -> (optional CAS re-read) -> rename over target -> fsync the parent directory.
 * Returns the temp path used (for diagnostics). Throws on any failure.
 */
export async function atomicWriteFile(
  targetPath: string,
  bytes: Uint8Array,
  options?: AtomicWriteOptions,
): Promise<string> {
  const dir = dirname(targetPath);
  await mkdir(dir, { recursive: true });
  const tempPath = join(
    dir,
    `.${basename(targetPath)}.${backupStamp()}.${randomBytes(4).toString("hex")}.tmp`,
  );

  const fh = await open(tempPath, "wx", 0o600);
  try {
    await fh.writeFile(bytes);
    await fh.sync(); // fsync file data before rename
  } finally {
    await fh.close();
  }

  try {
    // Fault-injection point FIRST, so any mutation it introduces is observed
    // by the CAS gate below (closing the CAS->rename TOCTOU window: the CAS
    // re-read is the last step before rename, with no observable gap).
    options?.hooks?.beforeRename?.();
    // CAS: terminal gate immediately before rename. Re-read the target bytes
    // here (not earlier) and refuse to clobber a target whose bytes changed
    // since the caller's snapshot. A concurrent writer (or a fault hook that
    // mutates the target) is caught here and the temp is discarded untouched.
    if (options?.expectedHash !== undefined) {
      const cur = await readRaw(targetPath);
      if (cur.hash !== options.expectedHash) {
        await safeUnlink(tempPath);
        throw new Error("models.json changed during apply (concurrent modification); aborted before write");
      }
    }
    await rename(tempPath, targetPath);
  } catch (err) {
    await safeUnlink(tempPath);
    throw err;
  }
  await fsyncDirectory(dir);
  return tempPath;
}

/** fsync a directory so the rename is durable. Best-effort on non-Linux. */
async function fsyncDirectory(dirPath: string): Promise<void> {
  let dirHandle;
  try {
    dirHandle = await open(dirPath, "r");
    await dirHandle.sync();
  } catch {
    // Some platforms/filesystems cannot fsync a directory fd; durability is
    // best-effort. The rename has already happened.
  } finally {
    if (dirHandle !== undefined) {
      try {
        await dirHandle.close();
      } catch {
        /* ignore */
      }
    }
  }
}

async function safeUnlink(path: string): Promise<void> {
  await unlink(path).catch(() => undefined);
}

/**
 * Create a timestamped backup of `bytes` in the target's own directory, written
 * with mode 0600. The backup directory must be contained within (resolve to)
 * the target's directory; symlinks are rejected. Backup files are never pruned.
 */
export async function writeBackup(
  targetPath: string,
  bytes: Uint8Array,
  backupDir: string,
  now: () => number = Date.now,
): Promise<string> {
  await assertBackupDirContained(targetPath, backupDir);
  const name = `${basename(targetPath)}.bak.${backupStamp(now)}`;
  const backupPath = join(backupDir, name);
  const fh = await open(backupPath, "wx", 0o600);
  try {
    await fh.writeFile(bytes);
    await fh.sync();
  } finally {
    await fh.close();
  }
  return backupPath;
}

/** Require backupDir to resolve inside the target's directory; reject symlinks. */
async function assertBackupDirContained(targetPath: string, backupDir: string): Promise<void> {
  let targetReal: string;
  try {
    const tst = await lstat(targetPath);
    if (tst.isSymbolicLink()) throw new Error("target is a symlink; refusing");
    targetReal = await realpath(targetPath);
  } catch (err) {
    throw new Error(`cannot resolve target for backup containment: ${messageOf(err)}`);
  }
  await mkdir(backupDir, { recursive: true });
  let backupReal: string;
  try {
    const bst = await lstat(backupDir);
    if (bst.isSymbolicLink()) throw new Error("backup dir is a symlink; refusing");
    backupReal = await realpath(backupDir);
  } catch (err) {
    throw new Error(`cannot resolve backup dir for containment: ${messageOf(err)}`);
  }
  const targetDir = dirname(targetReal);
  const contained = backupReal === targetDir || backupReal.startsWith(targetDir + "/");
  if (!contained) {
    throw new Error(`backup directory must be inside the target's directory; got ${backupReal} (target dir ${targetDir})`);
  }
}

export interface SafeWriteResult {
  readonly backups: readonly string[];
  readonly writtenBytes: number;
  readonly verifiedHash: string;
}

/**
 * Full safe apply for a single file:
 *   1. verify backup directory is contained within the target's directory
 *   2. read current raw bytes (target must exist)
 *   3. write a 0600 timestamped backup of the ORIGINAL bytes
 *   4. atomic-write the NEW bytes with a compare-and-swap on `expectedHash`
 *      (aborts without clobbering if the file changed since the snapshot)
 *   5. re-read the target and verify it matches the new bytes exactly
 *   6. on a post-rename verify failure, atomically restore the original bytes;
 *      if rollback also fails, surface the backup path for manual recovery.
 *
 * Rollback is ONLY attempted when the doctor's own write landed (rename
 * succeeded); a pre-rename abort never clobbers an external modification.
 */
export async function safeWriteWithBackup(
  targetPath: string,
  newBytes: Uint8Array,
  backupDir: string,
  now: () => number = Date.now,
  expectedHash?: string,
  hooks?: WriteHooks,
  /** Post-write schema gate: run on the re-read content; returning an error
   *  string triggers rollback (same path as a byte mismatch). */
  validateWritten?: (text: string) => string | undefined,
): Promise<SafeWriteResult> {
  await assertBackupDirContained(targetPath, backupDir);
  const original = await readRaw(targetPath);
  const backups: string[] = [];
  const backupPath = await writeBackup(targetPath, original.bytes, backupDir, now);
  backups.push(backupPath);

  // Write with CAS. A failure here means the target was NOT overwritten by us
  // (CAS aborted pre-rename, or rename failed atomically) -> do NOT rollback,
  // so we never clobber a concurrent external modification.
  try {
    await atomicWriteFile(targetPath, newBytes, { expectedHash, hooks });
  } catch (writeErr) {
    throw new Error(
      `apply aborted before commit: ${messageOf(writeErr)}; no changes written to ${targetPath} (re-run plan)`,
    );
  }

  // Post-write verify (hooks.afterRename may corrupt to simulate a fault).
  try {
    if (hooks?.afterRename) hooks.afterRename(targetPath);
    const reread = await readRaw(targetPath);
    if (!buffersEqual(reread.bytes, newBytes)) {
      throw new Error("post-write verification mismatch");
    }
    // Schema readback gate: re-validate the written content; rollback on violation.
    if (validateWritten) {
      const schemaErr = validateWritten(reread.text);
      if (schemaErr) {
        throw new Error(`post-write schema validation failed: ${schemaErr}`);
      }
    }
    return { backups, writtenBytes: newBytes.byteLength, verifiedHash: reread.hash };
  } catch (verifyErr) {
    // Our write landed (rename succeeded) but verification failed. Before
    // restoring the original, CAS-check that the target STILL holds OUR bytes:
    // a concurrent writer that changed the target after our rename must NOT be
    // clobbered by our rollback (we leave its content + preserve the backup).
    let cur: RawRead | undefined;
    try {
      cur = await readRaw(targetPath);
    } catch {
      cur = undefined;
    }
    if (!cur || !buffersEqual(cur.bytes, newBytes)) {
      throw new Error(
        `post-write verify failed AND target changed by a concurrent writer; refusing to clobber it (no rollback); original bytes preserved in backup ${backupPath}; manual recovery required (verify error: ${messageOf(verifyErr)})`,
      );
    }
    // Target still holds our bytes -> safe to restore the original (CAS-gated
    // on our bytes so a last-instant concurrent writer still cannot be clobbered).
    const rb = await tryRollback(targetPath, original.bytes, hooks, cur.hash);
    if (rb.ok) {
      throw new Error(`post-write verify failed, rolled back to original: ${messageOf(verifyErr)}`);
    }
    throw new Error(
      `post-write verify failed AND rollback failed; manual recovery required (rollback error: ${rb.error}); original bytes preserved in backup ${backupPath}`,
    );
  }
}

/** Restore a file to known-good bytes via atomic write; verifies the result.
 *  If `expectedHash` is given, the restore is CAS-gated on it so a concurrent
 *  writer that landed between the caller's check and this rename is NOT clobbered. */
async function tryRollback(
  targetPath: string,
  goodBytes: Uint8Array,
  hooks?: WriteHooks,
  expectedHash?: string,
): Promise<{ ok: true } | { ok: false; error: string }> {
  try {
    if (hooks?.onRollback) hooks.onRollback();
    const opts: AtomicWriteOptions = expectedHash !== undefined ? { expectedHash, hooks } : { hooks };
    await atomicWriteFile(targetPath, goodBytes, opts);
    const reread = await readRaw(targetPath);
    if (!buffersEqual(reread.bytes, goodBytes)) {
      return { ok: false, error: "rollback verify mismatch" };
    }
    return { ok: true };
  } catch (err) {
    return { ok: false, error: messageOf(err) };
  }
}

function buffersEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.byteLength !== b.byteLength) return false;
  const av = a instanceof Uint8Array ? a : new Uint8Array(a);
  const bv = b instanceof Uint8Array ? b : new Uint8Array(b);
  let diff = 0;
  for (let i = 0; i < av.byteLength; i++) diff |= av[i]! ^ bv[i]!;
  return diff === 0;
}


