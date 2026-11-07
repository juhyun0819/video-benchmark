import type { Catalog } from "./catalog.js";
import { remapAnnotations } from "./remap.js";
import {
  DatasetSchema,
  ManifestSchema,
  type AnnotationLevel,
  type Composition,
  type Dataset,
  type Manifest,
  type ManifestBlock,
  type ManifestCaption,
  type TimeSpan,
} from "./schema.js";
import { resolveTimeline } from "./timeline.js";

export const LEVEL_ORDER = ["L5", "L4", "L3", "L2", "L1", "raw", "summary", "keystep", "narration", "atomic"] as const;

export const LICENSES: Record<Dataset, { license: string; note?: string }> = {
  nextme800: { license: "CC BY-NC 4.0", note: "Research/evaluation only; gated (mmm8383/NextMe-800)" },
  hdepic: { license: "HD-EPIC dataset terms", note: "https://hd-epic.github.io/site/" },
  ego4d: { license: "Ego4D License Agreement", note: "https://ego4d-data.org/" },
  egoexo4d: { license: "Ego-Exo4D License Agreement", note: "https://ego-exo4d-data.org/" },
};

const round3 = (x: number): number => Math.round(x * 1000) / 1000;

function roundTime(t: TimeSpan): TimeSpan {
  return t.kind === "interval"
    ? { kind: "interval", start: round3(t.start), end: round3(t.end) }
    : { kind: "point", t: round3(t.t) };
}

const startOf = (t: TimeSpan): number => (t.kind === "interval" ? t.start : t.t);
const levelRank = (l: AnnotationLevel): number => LEVEL_ORDER.indexOf(l);

function compareCaptions(a: ManifestCaption, b: ManifestCaption): number {
  const ds = startOf(a.time) - startOf(b.time);
  if (ds !== 0) return ds;
  const dl = levelRank(a.level) - levelRank(b.level);
  if (dl !== 0) return dl;
  return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
}

/**
 * Builds the export manifest. Throws CompositionError (BASE_DAY_NOT_FOUND,
 * MISSING_CANDIDATES) via resolveTimeline. Only composed values are rounded;
 * source values (src_start/src_end, provenance.src_time) are copied verbatim.
 */
export function buildManifest(comp: Composition, catalog: Catalog, now: Date): Manifest {
  const timeline = resolveTimeline(comp, catalog);

  const blocks: ManifestBlock[] = [];
  const captions: ManifestCaption[] = [];
  const used = new Set<Dataset>();

  for (const tb of timeline) {
    used.add(tb.source.dataset);
    blocks.push({
      block_id: tb.block_id,
      role: tb.role,
      category: tb.category,
      summary: tb.summary,
      composed_start: round3(tb.composed_start),
      composed_end: round3(tb.composed_end),
      source: tb.source,
      src_start: tb.src_start,
      src_end: tb.src_end,
    });
    const annotations = catalog.getAnnotations(tb.source.dataset, tb.source.video_id);
    for (const c of remapAnnotations(tb, annotations)) {
      captions.push({ ...c, time: roundTime(c.time) });
    }
  }
  captions.sort(compareCaptions);

  const last = blocks[blocks.length - 1];
  const licenses = DatasetSchema.options
    .filter((d) => used.has(d))
    .map((dataset) => ({ dataset, ...LICENSES[dataset] }));

  return ManifestSchema.parse({
    schema_version: 1,
    composition_id: comp.id,
    composition_revision: comp.revision,
    created_at: now.toISOString(),
    duration: last ? last.composed_end : 0,
    blocks,
    tracks: { captions },
    licenses,
  });
}
