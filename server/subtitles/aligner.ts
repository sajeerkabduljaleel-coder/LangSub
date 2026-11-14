import type { AlignmentStats, RawCue } from '../../src/types/subtitle.ts';

export interface AlignedCuePair {
  index: number;
  startMs: number;
  endMs: number;
  targetText: string;
  baseText: string;
  confidence: number;
}

export interface AlignmentResult {
  aligned: AlignedCuePair[];
  stats: AlignmentStats;
}

function intervalIoU(aStart: number, aEnd: number, bStart: number, bEnd: number): number {
  const interStart = Math.max(aStart, bStart);
  const interEnd = Math.min(aEnd, bEnd);
  const inter = Math.max(0, interEnd - interStart);
  const union = Math.max(aEnd, bEnd) - Math.min(aStart, bStart);
  if (union <= 0) return 0;
  return inter / union;
}

function estimateGlobalOffsetMs(targetCues: RawCue[], baseCues: RawCue[]): number {
  if (targetCues.length < 3 || baseCues.length < 3) return 0;

  const sampleCount = Math.min(40, targetCues.length);
  const step = Math.max(1, Math.floor(targetCues.length / sampleCount));
  const deltas: number[] = [];

  for (let i = 0; i < targetCues.length; i += step) {
    const tMid = (targetCues[i].startMs + targetCues[i].endMs) / 2;
    let bestDelta = Infinity;
    for (const b of baseCues) {
      const bMid = (b.startMs + b.endMs) / 2;
      const diff = tMid - bMid;
      if (Math.abs(diff) < Math.abs(bestDelta)) {
        bestDelta = diff;
      }
    }
    if (Math.abs(bestDelta) <= 8000) {
      deltas.push(bestDelta);
    }
  }

  if (deltas.length < 3) return 0;
  deltas.sort((a, b) => a - b);
  const median = deltas[Math.floor(deltas.length / 2)];
  // Only apply offset if there is a meaningful consistent shift (> 250ms)
  return Math.abs(median) > 250 ? Math.round(median) : 0;
}

function lengthCompatibility(targetText: string, baseText: string): number {
  const tLen = Math.max(1, targetText.length);
  const bLen = Math.max(1, baseText.length);
  const ratio = Math.min(tLen, bLen) / Math.max(tLen, bLen);
  return 0.5 + 0.5 * ratio;
}

/**
 * Aligns base-language and target-language subtitle cues using global offset
 * estimation, temporal IoU, midpoint proximity, and multi-cue merging while
 * preserving the authentic target-language timestamps and dialogue.
 */
export function alignDualSubtitles(
  targetCues: RawCue[],
  baseCues: RawCue[],
): AlignmentResult {
  const estimatedOffsetMs = estimateGlobalOffsetMs(targetCues, baseCues);
  const shiftedBase = baseCues.map((b) => ({
    ...b,
    shiftedStart: b.startMs + estimatedOffsetMs,
    shiftedEnd: b.endMs + estimatedOffsetMs,
  }));

  const usedBaseIndices = new Set<number>();
  const aligned: AlignedCuePair[] = [];
  let exactMatches = 0;
  let interpolatedMatches = 0;
  let unmatchedTarget = 0;

  let baseCursor = 0;

  for (let tIdx = 0; tIdx < targetCues.length; tIdx++) {
    const t = targetCues[tIdx];
    const tMid = (t.startMs + t.endMs) / 2;

    // Advance cursor for efficiency while preserving search window
    while (
      baseCursor < shiftedBase.length - 1 &&
      shiftedBase[baseCursor].shiftedEnd < t.startMs - 2500
    ) {
      baseCursor++;
    }

    const candidates: Array<{ bIdx: number; score: number; iou: number }> = [];

    for (let bIdx = Math.max(0, baseCursor - 2); bIdx < shiftedBase.length; bIdx++) {
      const b = shiftedBase[bIdx];
      if (b.shiftedStart > t.endMs + 2500) break;

      const iou = intervalIoU(t.startMs, t.endMs, b.shiftedStart, b.shiftedEnd);
      const bMid = (b.shiftedStart + b.shiftedEnd) / 2;
      const midDist = Math.abs(tMid - bMid);
      const proximity = Math.exp(-(midDist * midDist) / (2 * 900 * 900));
      const lenScore = lengthCompatibility(t.text, b.text);

      const score = iou * 0.65 + proximity * 0.25 + lenScore * 0.1;
      if (iou > 0.08 || midDist <= 1400) {
        candidates.push({ bIdx, score, iou });
      }
    }

    if (candidates.length === 0) {
      unmatchedTarget++;
      aligned.push({
        index: t.index,
        startMs: t.startMs,
        endMs: t.endMs,
        targetText: t.text,
        baseText: '',
        confidence: 0,
      });
      continue;
    }

    candidates.sort((a, b) => b.score - a.score);
    const primary = candidates[0];

    // Check if 2 adjacent base cues clearly split inside this single target cue
    const secondary = candidates.find(
      (c) =>
        c.bIdx === primary.bIdx + 1 &&
        !usedBaseIndices.has(c.bIdx) &&
        intervalIoU(
          t.startMs,
          t.endMs,
          shiftedBase[c.bIdx].shiftedStart,
          shiftedBase[c.bIdx].shiftedEnd,
        ) > 0.2,
    );

    const matchedTexts: string[] = [];
    if (!usedBaseIndices.has(primary.bIdx) || primary.iou > 0.35) {
      matchedTexts.push(shiftedBase[primary.bIdx].text);
      usedBaseIndices.add(primary.bIdx);
    }

    if (secondary && secondary.iou > 0.25 && shiftedBase[primary.bIdx].text.length < 48) {
      matchedTexts.push(shiftedBase[secondary.bIdx].text);
      usedBaseIndices.add(secondary.bIdx);
      interpolatedMatches++;
    } else if (primary.iou >= 0.45) {
      exactMatches++;
    } else {
      interpolatedMatches++;
    }

    const combinedBase = matchedTexts.join(' ').replace(/\s{2,}/g, ' ').trim();
    if (!combinedBase) {
      unmatchedTarget++;
    }

    aligned.push({
      index: t.index,
      startMs: t.startMs,
      endMs: t.endMs,
      targetText: t.text,
      baseText: combinedBase,
      confidence: Number(Math.min(1, primary.score).toFixed(2)),
    });
  }

  return {
    aligned,
    stats: {
      exactMatches,
      interpolatedMatches,
      unmatchedTarget,
      unmatchedBase: Math.max(0, baseCues.length - usedBaseIndices.size),
      estimatedOffsetMs,
    },
  };
}
