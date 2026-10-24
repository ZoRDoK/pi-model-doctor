/**
 * The only module that reads pi's environment: agent directory and file paths.
 */
import { homedir } from "node:os";
import { dirname, join } from "node:path";

import type { DoctorPaths } from "./types.ts";

export function agentDir(): string {
  const fromEnv = process.env.PI_CODING_AGENT_DIR;

  if (fromEnv && fromEnv.length > 0) return fromEnv;

  const home = process.env.HOME || process.env.USERPROFILE || homedir();
  return join(home, ".pi", "agent");
}

export function defaultPaths(): DoctorPaths {
  const modelsPath = join(agentDir(), "models.json");

  return {
    modelsPath,
    settingsPath: join(agentDir(), "settings.json"),
    backupDir: dirname(modelsPath),
    authPath: join(agentDir(), "auth.json"),
  };
}
