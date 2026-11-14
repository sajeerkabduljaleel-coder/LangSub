import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';
import type {
  ConversionMode,
  PhoneticStyle,
  TriLineCue,
} from '../../src/types/subtitle.ts';
import {
  buildLineCacheKey,
  getCachedLine,
  setCachedLines,
} from '../db/store.ts';

const BATCH_SIZE = 20;
const MAX_RETRIES = 3;

// Unicode ranges / symbols strictly belonging to IPA that must NOT appear in English intuitive phonetics
const IPA_FORBIDDEN_REGEX = /[ɪʃθðŋəɛɔʊʌʒʔˈˌːæɑɒɜɝɾɲʎɕʑɯɤɨʉ]/;

function getAiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error(
      'GEMINI_API_KEY is not configured. Please verify your API key in Settings > Secrets.',
    );
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function getLanguagePhoneticRules(targetLang: string): string {
  const lower = targetLang.toLowerCase();
  if (lower.includes('german')) {
    return 'German rules: ch after a/o/u/au = "kh" (ich = "ikh"); sch = "sh"; sp/st at start = "shp/sht"; ei = "eye"; eu/äu = "oy"; ie = "ee"; v = "f"; w = "v"; z = "ts"; final b/d/g devoiced to p/t/k.';
  }
  if (lower.includes('japanese')) {
    return 'Japanese rules: clear syllables (koh-nee-chee-wah); long vowels doubled or marked cleanly (oh, oo, eh); final -u in desu/masu often soft/elided (dess, mahss); r = crisp light r.';
  }
  if (lower.includes('spanish')) {
    return 'Spanish rules: ll = "y"; ñ = "ny"; j/ge/gi = "kh/h"; qu = "k"; h is silent; soft intervocalic d = light "th/d"; clear vowels (a=ah, e=eh, i=ee, o=oh, u=oo).';
  }
  if (lower.includes('french')) {
    return 'French rules: account for spoken liaisons and silent final consonants; nasal vowels written naturally like "oh(n)", "ah(n)", "eh(n)"; u = "ew"; ou = "oo"; j = "zh"; qu = "k".';
  }
  if (lower.includes('korean')) {
    return 'Korean rules: account for batchim consonant assimilation between syllables; eo = "uh"; eu = "eu"; ae/e = "eh"; si = "shee"; crisp syllable breaks.';
  }
  if (lower.includes('mandarin') || lower.includes('chinese')) {
    return 'Mandarin rules: convert pinyin/characters into English-intuitive sounds: zh="j", q="ch", x="sh", c="ts", z="dz", ü="ew"; natural syllable breaks.';
  }
  if (lower.includes('italian')) {
    return 'Italian rules: c before e/i = "ch", ch = "k"; g before e/i = "j", gh = "g"; gl = "ly"; gn = "ny"; sc before e/i = "sh"; clear vowels.';
  }
  return 'Use intuitive English spelling rules consistently: ah, eh, ee, oh, oo, kh, zh, sh, ch, ny. Never use IPA symbols.';
}

function buildSystemInstruction(params: {
  mode: ConversionMode;
  baseLang: string;
  targetLang: string;
  phoneticStyle: PhoneticStyle;
}): string {
  const styleDesc =
    params.phoneticStyle === 'hyphenated'
      ? 'Separate multi-syllable words with hyphens and capitalize the primary stressed syllable in each word, e.g. "ikh fer-SHTAY-uh nikht".'
      : 'Use smooth, easy-to-read natural English-phonetic word spacing with gentle stress capitalization only where helpful, e.g. "ikh fer-shtay-uh nikht".';

  const modeTask =
    params.mode === 'target-first'
      ? `Each input cue provides "targetText" in ${params.targetLang}. Preserve "targetText" faithfully, translate it into natural conversational ${params.baseLang} for "baseText", and generate intuitive English-reader phonetics of "targetText" for "phoneticText".`
      : params.mode === 'base-first'
        ? `Each input cue provides "baseText" in ${params.baseLang} (and possibly "targetText" if already matched). Preserve "baseText", translate into natural spoken movie-dialogue ${params.targetLang} for "targetText" (if empty), and generate intuitive English-reader phonetics of "targetText" for "phoneticText".`
        : `Each input cue provides authentic "targetText" in ${params.targetLang} and paired "baseText" in ${params.baseLang} (if "baseText" is empty due to an unmatched cue, translate "targetText" into ${params.baseLang}). Preserve all provided dialogue faithfully and generate intuitive English-reader phonetics of "targetText" for "phoneticText".`;

  return [
    'You are an expert bilingual cinema subtitler and applied phonetics specialist.',
    modeTask,
    'PHONETIC RULES (CRITICAL):',
    '1. NEVER output International Phonetic Alphabet (IPA) symbols, slashes (/.../), or diacritics unfamiliar to English readers.',
    `2. ${styleDesc}`,
    `3. ${getLanguagePhoneticRules(params.targetLang)}`,
    '4. Account for contextual pronunciation, contractions, elisions, and natural spoken cadence.',
    '5. Keep lines concise and faithful to film dialogue without adding commentary or brackets.',
    '6. Return EXACTLY one JSON item per input cue with matching numeric "index".',
  ].join('\n');
}

export interface BatchInputCue {
  index: number;
  startMs: number;
  endMs: number;
  baseText: string;
  targetText: string;
}

interface GeminiBatchItem {
  index: number;
  baseText: string;
  targetText: string;
  phoneticText: string;
}

async function callGeminiBatchWithRetry(
  ai: GoogleGenAI,
  systemInstruction: string,
  batch: BatchInputCue[],
  contextBefore: BatchInputCue[],
): Promise<GeminiBatchItem[]> {
  const promptPayload = {
    readOnlyPrecedingContext: contextBefore.map((c) => ({
      baseText: c.baseText,
      targetText: c.targetText,
    })),
    cuesToProcess: batch.map((c) => ({
      index: c.index,
      baseText: c.baseText,
      targetText: c.targetText,
    })),
  };

  let lastError: unknown;

  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const response = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: JSON.stringify(promptPayload),
        config: {
          systemInstruction,
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              items: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    index: { type: Type.INTEGER },
                    baseText: { type: Type.STRING },
                    targetText: { type: Type.STRING },
                    phoneticText: { type: Type.STRING },
                  },
                  required: ['index', 'baseText', 'targetText', 'phoneticText'],
                },
              },
            },
            required: ['items'],
          },
        },
      });

      const rawText = (response.text ?? '').trim();
      if (!rawText) {
        throw new Error('Empty response returned from Gemini API.');
      }

      const parsed = JSON.parse(rawText) as { items?: GeminiBatchItem[] };
      if (!parsed.items || !Array.isArray(parsed.items)) {
        throw new Error('Malformed JSON schema returned from Gemini API.');
      }

      // Validate completeness & sanitize accidental IPA
      const byIndex = new Map<number, GeminiBatchItem>();
      for (const item of parsed.items) {
        if (typeof item.index !== 'number') continue;
        const cleanPhonetic = (item.phoneticText || '')
          .replace(/^\/+|\/+$/g, '')
          .trim();

        if (IPA_FORBIDDEN_REGEX.test(cleanPhonetic)) {
          throw new Error(`IPA character detected in phonetic output for cue ${item.index}`);
        }

        byIndex.set(item.index, {
          index: item.index,
          baseText: (item.baseText || '').trim(),
          targetText: (item.targetText || '').trim(),
          phoneticText: cleanPhonetic,
        });
      }

      // Ensure every input cue is accounted for without silent drops
      const validated: GeminiBatchItem[] = batch.map((orig) => {
        const matched = byIndex.get(orig.index);
        if (!matched) {
          throw new Error(`Missing output item for cue index ${orig.index}`);
        }
        return {
          index: orig.index,
          baseText: orig.baseText ? orig.baseText : matched.baseText,
          targetText: orig.targetText ? orig.targetText : matched.targetText,
          phoneticText: matched.phoneticText || matched.targetText,
        };
      });

      return validated;
    } catch (err: unknown) {
      lastError = err;
      const msg = err instanceof Error ? err.message : String(err);
      // Do not retry hard auth errors
      if (
        msg.includes('API_KEY_INVALID') ||
        msg.includes('PERMISSION_DENIED') ||
        msg.includes('403')
      ) {
        throw err;
      }
      if (attempt < MAX_RETRIES - 1) {
        const delayMs = 800 * Math.pow(2, attempt) + Math.floor(Math.random() * 350);
        await new Promise((r) => setTimeout(r, delayMs));
      }
    }
  }

  throw lastError instanceof Error
    ? lastError
    : new Error('Gemini batch processing failed after retries.');
}

export interface ProgressSnapshot {
  processedCues: number;
  totalCues: number;
  cacheHits: number;
  apiBatchesCompleted: number;
  apiBatchesTotal: number;
  estimatedRemainingSec: number;
}

export async function processTriLineConversion(params: {
  cues: BatchInputCue[];
  mode: ConversionMode;
  baseLang: string;
  targetLang: string;
  phoneticStyle: PhoneticStyle;
  skipLineCache?: boolean;
  onProgress: (progress: ProgressSnapshot) => void;
}): Promise<{ cues: TriLineCue[]; cacheHits: number }> {
  const totalCues = params.cues.length;
  const results: Array<TriLineCue | null> = new Array(totalCues).fill(null);
  const uncachedIndices: number[] = [];
  let cacheHits = 0;

  // Step 1: Check persistent line-level cache
  for (let i = 0; i < totalCues; i++) {
    const cue = params.cues[i];
    const cacheKey = buildLineCacheKey({
      mode: params.mode,
      baseLang: params.baseLang,
      targetLang: params.targetLang,
      phoneticStyle: params.phoneticStyle,
      sourcePrimary: params.mode === 'base-first' ? cue.baseText : cue.targetText,
      sourceSecondary: params.mode === 'dual-merge' ? cue.baseText : '',
    });

    if (!params.skipLineCache) {
      const cached = await getCachedLine(cacheKey);
      if (cached && cached.baseText && cached.targetText && cached.phoneticText) {
        results[i] = {
          id: `cue-${cue.index}`,
          index: cue.index,
          startMs: cue.startMs,
          endMs: cue.endMs,
          baseText: cue.baseText || cached.baseText,
          targetText: cue.targetText || cached.targetText,
          phoneticText: cached.phoneticText,
          cacheHit: true,
        };
        cacheHits++;
        continue;
      }
    }

    uncachedIndices.push(i);
  }

  const totalBatches = Math.ceil(uncachedIndices.length / BATCH_SIZE);
  params.onProgress({
    processedCues: cacheHits,
    totalCues,
    cacheHits,
    apiBatchesCompleted: 0,
    apiBatchesTotal: totalBatches,
    estimatedRemainingSec: totalBatches * 3,
  });

  if (uncachedIndices.length > 0) {
    const ai = getAiClient();
    const systemInstruction = buildSystemInstruction({
      mode: params.mode,
      baseLang: params.baseLang,
      targetLang: params.targetLang,
      phoneticStyle: params.phoneticStyle,
    });

    const batchDurations: number[] = [];

    for (let bStart = 0; bStart < uncachedIndices.length; bStart += BATCH_SIZE) {
      const batchSlotIndices = uncachedIndices.slice(bStart, bStart + BATCH_SIZE);
      const batchInput = batchSlotIndices.map((idx) => params.cues[idx]);

      const firstIdx = batchSlotIndices[0];
      const contextBefore = params.cues.slice(Math.max(0, firstIdx - 2), firstIdx);

      const startedAt = Date.now();
      const generated = await callGeminiBatchWithRetry(
        ai,
        systemInstruction,
        batchInput,
        contextBefore,
      );
      batchDurations.push((Date.now() - startedAt) / 1000);

      const cacheWrites: Array<{
        key: string;
        value: { baseText: string; targetText: string; phoneticText: string; updatedAt: string };
      }> = [];

      for (let j = 0; j < batchSlotIndices.length; j++) {
        const slotIdx = batchSlotIndices[j];
        const orig = params.cues[slotIdx];
        const gen = generated[j];

        const triCue: TriLineCue = {
          id: `cue-${orig.index}`,
          index: orig.index,
          startMs: orig.startMs,
          endMs: orig.endMs,
          baseText: gen.baseText,
          targetText: gen.targetText,
          phoneticText: gen.phoneticText,
          cacheHit: false,
        };
        results[slotIdx] = triCue;

        const cacheKey = buildLineCacheKey({
          mode: params.mode,
          baseLang: params.baseLang,
          targetLang: params.targetLang,
          phoneticStyle: params.phoneticStyle,
          sourcePrimary: params.mode === 'base-first' ? orig.baseText : orig.targetText,
          sourceSecondary: params.mode === 'dual-merge' ? orig.baseText : '',
        });

        cacheWrites.push({
          key: cacheKey,
          value: {
            baseText: triCue.baseText,
            targetText: triCue.targetText,
            phoneticText: triCue.phoneticText,
            updatedAt: new Date().toISOString(),
          },
        });
      }

      await setCachedLines(cacheWrites);

      const batchesDone = Math.floor(bStart / BATCH_SIZE) + 1;
      const avgBatchSec =
        batchDurations.reduce((a, b) => a + b, 0) / batchDurations.length;
      const remainingBatches = Math.max(0, totalBatches - batchesDone);

      params.onProgress({
        processedCues: Math.min(totalCues, cacheHits + bStart + batchSlotIndices.length),
        totalCues,
        cacheHits,
        apiBatchesCompleted: batchesDone,
        apiBatchesTotal: totalBatches,
        estimatedRemainingSec: Math.round(remainingBatches * avgBatchSec),
      });
    }
  }

  const finalCues = results.filter((c): c is TriLineCue => c !== null);
  return { cues: finalCues, cacheHits };
}
