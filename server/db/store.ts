import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import type {
  AssStyleConfig,
  ConversionMode,
  PhoneticStyle,
  ProcessingOptions,
  SavedProject,
  TriLineCue,
} from '../../src/types/subtitle.ts';

interface LineCacheEntry {
  baseText: string;
  targetText: string;
  phoneticText: string;
  updatedAt: string;
}

interface DatabaseSchema {
  version: number;
  projects: SavedProject[];
  lineCache: Record<string, LineCacheEntry>;
}

const DATA_DIR = path.resolve(process.cwd(), 'data');
const DB_PATH = path.join(DATA_DIR, 'triline-db.json');

let memoryDb: DatabaseSchema | null = null;
let writeQueue: Promise<void> = Promise.resolve();

async function ensureDbLoaded(): Promise<DatabaseSchema> {
  if (memoryDb) return memoryDb;

  try {
    await fs.mkdir(DATA_DIR, { recursive: true });
    const raw = await fs.readFile(DB_PATH, 'utf8');
    const parsed = JSON.parse(raw) as DatabaseSchema;
    memoryDb = {
      version: parsed.version || 1,
      projects: Array.isArray(parsed.projects) ? parsed.projects : [],
      lineCache: parsed.lineCache && typeof parsed.lineCache === 'object' ? parsed.lineCache : {},
    };
  } catch {
    memoryDb = {
      version: 1,
      projects: [],
      lineCache: {},
    };
  }

  return memoryDb;
}

async function persistDb(): Promise<void> {
  if (!memoryDb) return;
  const snapshot = JSON.stringify(memoryDb);
  writeQueue = writeQueue.then(async () => {
    await fs.mkdir(DATA_DIR, { recursive: true });
    const tmpPath = `${DB_PATH}.tmp.${Date.now()}`;
    await fs.writeFile(tmpPath, snapshot, 'utf8');
    await fs.rename(tmpPath, DB_PATH);
  });
  await writeQueue;
}

export function buildProjectCacheKey(params: {
  movieId?: string;
  sourceHash: string;
  secondaryHash?: string;
  baseLang: string;
  targetLang: string;
  mode: ConversionMode;
  phoneticStyle: PhoneticStyle;
  processingOptions: ProcessingOptions;
}): string {
  const payload = JSON.stringify({
    movieId: params.movieId || 'manual',
    sourceHash: params.sourceHash,
    secondaryHash: params.secondaryHash || '',
    baseLang: params.baseLang,
    targetLang: params.targetLang,
    mode: params.mode,
    phoneticStyle: params.phoneticStyle,
    processingOptions: params.processingOptions,
  });
  return crypto.createHash('sha256').update(payload).digest('hex').slice(0, 24);
}

export function buildLineCacheKey(params: {
  mode: ConversionMode;
  baseLang: string;
  targetLang: string;
  phoneticStyle: PhoneticStyle;
  sourcePrimary: string;
  sourceSecondary?: string;
}): string {
  const normA = params.sourcePrimary.trim().replace(/\s+/g, ' ');
  const normB = (params.sourceSecondary || '').trim().replace(/\s+/g, ' ');
  const raw = `${params.mode}|${params.baseLang}|${params.targetLang}|${params.phoneticStyle}|${normA}|${normB}`;
  return crypto.createHash('sha256').update(raw).digest('hex').slice(0, 24);
}

export async function getCachedLine(key: string): Promise<LineCacheEntry | null> {
  const db = await ensureDbLoaded();
  return db.lineCache[key] || null;
}

export async function setCachedLines(entries: Array<{ key: string; value: LineCacheEntry }>): Promise<void> {
  if (entries.length === 0) return;
  const db = await ensureDbLoaded();
  for (const { key, value } of entries) {
    db.lineCache[key] = value;
  }
  await persistDb();
}

export async function findProjectByCacheKey(cacheKey: string): Promise<SavedProject | null> {
  const db = await ensureDbLoaded();
  return db.projects.find((p) => p.cacheKey === cacheKey) || null;
}

export async function listProjects(): Promise<SavedProject[]> {
  const db = await ensureDbLoaded();
  return [...db.projects].sort(
    (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime(),
  );
}

export async function getProjectById(id: string): Promise<SavedProject | null> {
  const db = await ensureDbLoaded();
  return db.projects.find((p) => p.id === id) || null;
}

export async function saveProject(project: SavedProject): Promise<SavedProject> {
  const db = await ensureDbLoaded();
  const existingIdx = db.projects.findIndex(
    (p) => p.id === project.id || p.cacheKey === project.cacheKey,
  );
  if (existingIdx >= 0) {
    db.projects[existingIdx] = project;
  } else {
    db.projects.unshift(project);
  }
  await persistDb();
  return project;
}

export async function updateProject(
  id: string,
  patch: { cues?: TriLineCue[]; styleConfig?: AssStyleConfig; title?: string },
): Promise<SavedProject | null> {
  const db = await ensureDbLoaded();
  const idx = db.projects.findIndex((p) => p.id === id);
  if (idx === -1) return null;

  const current = db.projects[idx];
  const updated: SavedProject = {
    ...current,
    ...(patch.title !== undefined ? { title: patch.title } : {}),
    ...(patch.cues !== undefined ? { cues: patch.cues } : {}),
    ...(patch.styleConfig !== undefined ? { styleConfig: patch.styleConfig } : {}),
    updatedAt: new Date().toISOString(),
  };
  db.projects[idx] = updated;
  await persistDb();
  return updated;
}

export async function deleteProject(id: string): Promise<boolean> {
  const db = await ensureDbLoaded();
  const before = db.projects.length;
  db.projects = db.projects.filter((p) => p.id !== id);
  if (db.projects.length !== before) {
    await persistDb();
    return true;
  }
  return false;
}

export async function getCacheStats(): Promise<{
  savedProjectCount: number;
  cachedLineCount: number;
}> {
  const db = await ensureDbLoaded();
  return {
    savedProjectCount: db.projects.length,
    cachedLineCount: Object.keys(db.lineCache).length,
  };
}
