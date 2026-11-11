import { ClipContractRowSchema, type Annotation, type Clip, type Dataset } from "@vb/core";
import { bump, IngestFormatError, newReport, type IngestReport } from "./jsonl.js";

/**
 * Parse content-contract clip rows (spec 4.1). A row with `category: null` is an unlabeled draft and is
 * skipped (counted as skipped.unlabeled). Captions become annotations of the clip's source video, merged
 * per video and deduplicated by annotation id.
 */
export function ingestBank(
  text: string,
  file: string,
  dataset: Dataset,
): { clips: Clip[]; annotations: Map<string, Annotation[]>; report: IngestReport } {
  const report = newReport();
  const clips: Clip[] = [];
  const annotations = new Map<string, Annotation[]>();
  const seenAnnotationIds = new Set<string>();
  const clipIds = new Set<string>();

  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];
    if (raw === undefined || raw.trim() === "") continue;
    const lineNo = i + 1;
    let json: unknown;
    try {
      json = JSON.parse(raw);
    } catch (e) {
      throw new IngestFormatError(file, lineNo, `invalid JSON: ${(e as Error).message}`);
    }
    if (typeof json === "object" && json !== null && (json as Record<string, unknown>).category === null) {
      bump(report.skipped, "unlabeled");
      continue;
    }
    const parsed = ClipContractRowSchema.safeParse(json);
    if (!parsed.success) {
      const issue = parsed.error.issues
        .map((iss) => `${iss.path.join(".") || "(row)"}: ${iss.message}`)
        .join("; ");
      throw new IngestFormatError(file, lineNo, issue);
    }
    const { captions, ...clip } = parsed.data;
    if (clip.source.dataset !== dataset) {
      throw new IngestFormatError(
        file,
        lineNo,
        `source.dataset "${clip.source.dataset}" does not match --dataset "${dataset}"`,
      );
    }
    if (clipIds.has(clip.clip_id)) throw new IngestFormatError(file, lineNo, `duplicate clip_id "${clip.clip_id}"`);
    clipIds.add(clip.clip_id);
    clips.push(clip);

    const videoId = clip.source.video_id;
    const list = annotations.get(videoId) ?? [];
    for (const c of captions) {
      const id = `${dataset}:${videoId}:${c.id}`;
      if (seenAnnotationIds.has(id)) continue;
      seenAnnotationIds.add(id);
      list.push({ id, track: "captions", level: c.level, time: c.time, text: c.text, original: c });
    }
    annotations.set(videoId, list);
    bump(report.ingested, clip.category);
  }
  return { clips, annotations, report };
}
