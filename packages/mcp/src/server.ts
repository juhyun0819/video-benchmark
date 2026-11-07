import { randomUUID } from "node:crypto";
import { resolve } from "node:path";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import {
  assertRevision,
  buildManifest,
  CompositionError,
  createComposition,
  insertEvent,
  moveEvent,
  removeEvent,
  resolveTimeline,
  type Annotation,
  type BlockCandidate,
  type Composition,
  type Dataset,
  type TimeSpan,
} from "@vb/core";
import { CatalogMissingError, FsCatalog } from "./fsCatalog.js";
import { renderTimetable, total as formatTotal } from "./render.js";
import { CompositionStore } from "./store.js";

export type ServerOptions = {
  now?: () => Date;
  genId?: () => string;
};

/** Context shared by tool handlers (edit/export tools are added in later tasks). */
export type ToolContext = {
  dataDir: string;
  now: () => Date;
  genId: () => string;
  /** Constructed per call so a missing catalog becomes a tool error rather than a startup crash. */
  catalog: () => FsCatalog;
  store: CompositionStore;
};

export function toToolError(e: unknown): CallToolResult {
  const text = errorText(e);
  return { isError: true, content: [{ type: "text", text }] };
}

function errorText(e: unknown): string {
  if (e instanceof CatalogMissingError) return "Catalog not found. Run npm run ingest first";
  if (e instanceof CompositionError) {
    const d = e.details;
    switch (e.code) {
      case "SLOT_OUT_OF_RANGE":
        return `slot ${String(d["slot"])} out of range (0..${String(d["max"])})`;
      case "CANDIDATE_NOT_FOUND":
        return `candidate_id not found: ${String(d["candidateId"])} — check with list_event_candidates`;
      case "POSITION_OUT_OF_RANGE":
        return `position ${String(d["position"])} out of range (0..${String(d["max"])})`;
      case "NOT_EVENT":
        return `not an event candidate: ${String(d["candidateId"])}`;
      case "REF_NOT_FOUND":
        return `ref_id not found: ${String(d["refId"])} — check with get_composition`;
      case "COMPOSITION_NOT_FOUND":
        return `composition not found: ${String(d["id"])} — check with list_compositions`;
      case "BASE_DAY_NOT_FOUND":
        return `base day not found: ${String(d["day"])} — check with list_base_days`;
      case "BASE_DAY_CHANGED":
        return `base day ${String(d["day"])} block layout changed (slots: ${String(d["actual"])}, required now: ${String(d["expected"])}) — create a new composition`;
      case "REVISION_MISMATCH":
        return `revision mismatch (current ${String(d["current"])})`;
      case "MISSING_CANDIDATES": {
        const ids = Array.isArray(d["ids"]) ? d["ids"].map(String) : [];
        return `candidates not in catalog: ${ids.join(", ")}`;
      }
      default:
        return `${e.code}: ${e.message}`;
    }
  }
  return e instanceof Error ? e.message : String(e);
}

export function ok(text: string, structured: Record<string, unknown>): CallToolResult {
  return { content: [{ type: "text", text }], structuredContent: structured };
}

/** Runs a handler and converts thrown errors into `isError` tool results. */
export function guard<A>(fn: (args: A) => CallToolResult): (args: A) => CallToolResult {
  return (args) => {
    try {
      return fn(args);
    } catch (e) {
      return toToolError(e);
    }
  };
}

export function fmtDuration(sec: number): string {
  const s = Math.max(0, Math.round(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  const p = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${p(m)}:${p(r)}` : `${m}:${p(r)}`;
}

function candidateView(c: BlockCandidate) {
  return {
    id: c.id,
    role: c.role,
    category: c.category,
    summary: c.summary,
    dataset: c.source.dataset,
    video_id: c.source.video_id,
    src_start: c.src_start,
    src_end: c.src_end,
    duration: c.src_end - c.src_start,
  };
}

/** Same overlap/point rules as caption remapping, against the window [start, end). */
function overlapsWindow(time: TimeSpan, start: number, end: number): boolean {
  if (time.kind === "point") return time.t >= start && time.t < end;
  if (time.start === time.end) return time.start >= start && time.start < end;
  return time.start < end && time.end > start;
}

const spanStart = (t: TimeSpan) => (t.kind === "point" ? t.t : t.start);

function previewOf(annotations: Annotation[], c: BlockCandidate, limit: number) {
  return annotations
    .filter((a) => overlapsWindow(a.time, c.src_start, c.src_end))
    .sort((a, b) => spanStart(a.time) - spanStart(b.time))
    .slice(0, limit)
    .map((a) => ({
      id: a.id,
      level: a.level,
      kind: a.time.kind,
      start: spanStart(a.time),
      ...(a.time.kind === "interval" ? { end: a.time.end } : {}),
      text: a.text,
    }));
}

export function createServer(dataDirArg: string, opts: ServerOptions = {}): McpServer {
  // Resolve once so every store/catalog path (and the export path returned to callers) is absolute.
  const dataDir = resolve(dataDirArg);
  const ctx: ToolContext = {
    dataDir,
    now: opts.now ?? (() => new Date()),
    genId: opts.genId ?? (() => randomUUID().slice(0, 8)),
    catalog: () => new FsCatalog(dataDir),
    store: new CompositionStore(dataDir),
  };
  const server = new McpServer({ name: "video-composer", version: "0.0.0" });
  registerBrowseTools(server, ctx);
  registerEditTools(server, ctx);
  registerExportTools(server, ctx);
  return server;
}

function registerBrowseTools(server: McpServer, ctx: ToolContext): void {
  server.registerTool(
    "list_base_days",
    { description: "List ingested base days with block count and total duration." },
    guard(() => {
      const days = ctx.catalog().listBaseDays();
      const text =
        days.length === 0
          ? "(no base days)"
          : days.map((d) => `${d.day}  ${d.blocks} blocks  ${fmtDuration(d.duration)}`).join("\n");
      return ok(text, { days });
    }),
  );

  server.registerTool(
    "get_base_day",
    {
      description: "Show a base day's blocks: index, length, summary and source time range.",
      inputSchema: { day: z.string().describe("Base day, e.g. 2026-04-15") },
    },
    guard(({ day }: { day: string }) => {
      const d = ctx.catalog().getBaseDay(day);
      if (!d) {
        throw new CompositionError("BASE_DAY_NOT_FOUND", `Base day ${day} not found — check with list_base_days`, { day });
      }
      const blocks = d.blocks.map((b, index) => ({ index, ...candidateView(b) }));
      const text = [
        `${d.day}  ${blocks.length} blocks`,
        ...blocks.map(
          (b) =>
            `B${b.index}  ${fmtDuration(b.duration)}  ${b.video_id} [${b.src_start}–${b.src_end}]  ${b.summary}`,
        ),
      ].join("\n");
      return ok(text, { day: d.day, blocks });
    }),
  );

  server.registerTool(
    "list_categories",
    { description: "List event categories with candidate counts." },
    guard(() => {
      const categories = ctx.catalog().listCategories();
      const text =
        categories.length === 0
          ? "(no categories)"
          : categories.map((c) => `${c.category}  ${c.count}`).join("\n");
      return ok(text, { categories });
    }),
  );

  server.registerTool(
    "list_event_candidates",
    {
      description: "List event candidates in a category, optionally filtered by summary text and duration.",
      inputSchema: {
        category: z.string(),
        query: z.string().optional().describe("Case-insensitive substring of the summary"),
        min_sec: z.number().min(0).optional(),
        max_sec: z.number().min(0).optional(),
        limit: z.number().int().min(1).max(200).default(20),
        offset: z.number().int().min(0).default(0),
      },
    },
    guard(
      (a: {
        category: string;
        query?: string | undefined;
        min_sec?: number | undefined;
        max_sec?: number | undefined;
        limit: number;
        offset: number;
      }) => {
        const { total, items } = ctx.catalog().listCandidates({
          category: a.category,
          ...(a.query !== undefined ? { query: a.query } : {}),
          ...(a.min_sec !== undefined ? { minSec: a.min_sec } : {}),
          ...(a.max_sec !== undefined ? { maxSec: a.max_sec } : {}),
          limit: a.limit,
          offset: a.offset,
        });
        const views = items.map(candidateView);
        const text = [
          `${a.category}: ${total} total, showing ${views.length} from ${a.offset}`,
          ...views.map((v) => `${v.id}  ${fmtDuration(v.duration)}  ${v.dataset}  ${v.summary}`),
        ].join("\n");
        return ok(text, { total, offset: a.offset, limit: a.limit, items: views });
      },
    ),
  );

  server.registerTool(
    "get_candidate",
    {
      description: "Show a candidate (event or base block) with a preview of its captions.",
      inputSchema: {
        candidate_id: z.string(),
        preview: z.number().int().min(0).max(200).default(10),
      },
    },
    guard(({ candidate_id, preview }: { candidate_id: string; preview: number }) => {
      const catalog = ctx.catalog();
      const c = catalog.getCandidate(candidate_id);
      if (!c) {
        throw new CompositionError("CANDIDATE_NOT_FOUND", `Candidate ${candidate_id} not found`, {
          candidateId: candidate_id,
        });
      }
      const rows = previewOf(catalog.getAnnotations(c.source.dataset, c.source.video_id), c, preview);
      const text = [
        `${c.id}  ${c.category}  ${fmtDuration(c.src_end - c.src_start)}  ${c.summary}`,
        ...rows.map((r) => `  [${r.start}${"end" in r ? `–${r.end}` : ""}] ${r.level}: ${r.text}`),
      ].join("\n");
      return ok(text, { candidate: candidateView(c), source: c.source, preview: rows });
    }),
  );
}

/**
 * Resolves the timeline and returns the rendered timetable plus structured composition/timeline.
 * Called after `save`: a resolve failure is reported with the saved revision so the caller knows the edit persisted.
 */
function timetableResult(comp: Composition, catalog: FsCatalog): CallToolResult {
  try {
    const timeline = resolveTimeline(comp, catalog);
    return ok(renderTimetable(comp, timeline), { composition: comp, timeline });
  } catch (e) {
    return {
      isError: true,
      content: [{ type: "text", text: `${errorText(e)} (changes saved as rev ${comp.revision})` }],
    };
  }
}

function registerEditTools(server: McpServer, ctx: ToolContext): void {
  const compositionId = z.string().describe("Composition id from create_composition / list_compositions");
  const expectedRevision = z
    .number()
    .int()
    .optional()
    .describe("Reject the edit unless the composition is at this revision");

  server.registerTool(
    "create_composition",
    {
      description: "Create an empty composition on a base day.",
      inputSchema: { day: z.string().describe("Base day, e.g. 2026-04-15"), name: z.string().optional() },
    },
    guard(({ day, name }: { day: string; name?: string | undefined }) => {
      const catalog = ctx.catalog();
      const baseDay = catalog.getBaseDay(day);
      if (!baseDay) {
        throw new CompositionError("BASE_DAY_NOT_FOUND", `Base day ${day} not found — check with list_base_days`, { day });
      }
      const comp = createComposition(baseDay, { genId: ctx.genId, ...(name !== undefined ? { name } : {}) });
      ctx.store.save(comp);
      return timetableResult(comp, catalog);
    }),
  );

  server.registerTool(
    "list_compositions",
    { description: "List saved compositions with id, name, base day and revision." },
    guard(() => {
      const compositions = ctx.store.list();
      const text =
        compositions.length === 0
          ? "(no compositions)"
          : compositions
              .map((c) => `${c.id}  ${c.base_day}  rev ${c.revision}${c.name !== undefined ? `  ${c.name}` : ""}`)
              .join("\n");
      return ok(text, { compositions });
    }),
  );

  server.registerTool(
    "get_composition",
    {
      description: "Show a composition's current timetable.",
      inputSchema: { composition_id: compositionId },
    },
    guard(({ composition_id }: { composition_id: string }) => {
      const comp = ctx.store.load(composition_id);
      const catalog = ctx.catalog();
      const timeline = resolveTimeline(comp, catalog);
      return ok(renderTimetable(comp, timeline), { composition: comp, timeline });
    }),
  );

  server.registerTool(
    "insert_event",
    {
      description: "Insert an event candidate into a slot (appended unless position is given).",
      inputSchema: {
        composition_id: compositionId,
        slot: z.number().int().describe("Slot index; slot i sits before base block i"),
        candidate_id: z.string(),
        position: z.number().int().optional().describe("Index within the slot"),
        expected_revision: expectedRevision,
      },
    },
    guard(
      (a: {
        composition_id: string;
        slot: number;
        candidate_id: string;
        position?: number | undefined;
        expected_revision?: number | undefined;
      }) => {
        const catalog = ctx.catalog();
        const comp = ctx.store.load(a.composition_id);
        assertRevision(comp, a.expected_revision);
        const next = insertEvent(
          comp,
          catalog,
          { slot: a.slot, candidateId: a.candidate_id, ...(a.position !== undefined ? { position: a.position } : {}) },
          ctx.genId,
        );
        ctx.store.save(next);
        return timetableResult(next, catalog);
      },
    ),
  );

  server.registerTool(
    "move_event",
    {
      description: "Move an event (by ref_id) to another slot/position.",
      inputSchema: {
        composition_id: compositionId,
        ref_id: z.string(),
        to_slot: z.number().int(),
        to_position: z.number().int().optional().describe("Index within the target slot"),
        expected_revision: expectedRevision,
      },
    },
    guard(
      (a: {
        composition_id: string;
        ref_id: string;
        to_slot: number;
        to_position?: number | undefined;
        expected_revision?: number | undefined;
      }) => {
        const catalog = ctx.catalog();
        const comp = ctx.store.load(a.composition_id);
        assertRevision(comp, a.expected_revision);
        const next = moveEvent(comp, {
          refId: a.ref_id,
          toSlot: a.to_slot,
          ...(a.to_position !== undefined ? { toPosition: a.to_position } : {}),
        });
        ctx.store.save(next);
        return timetableResult(next, catalog);
      },
    ),
  );

  server.registerTool(
    "remove_event",
    {
      description: "Remove an event (by ref_id) from the composition.",
      inputSchema: { composition_id: compositionId, ref_id: z.string(), expected_revision: expectedRevision },
    },
    guard(
      (a: { composition_id: string; ref_id: string; expected_revision?: number | undefined }) => {
        const catalog = ctx.catalog();
        const comp = ctx.store.load(a.composition_id);
        assertRevision(comp, a.expected_revision);
        const next = removeEvent(comp, { refId: a.ref_id });
        ctx.store.save(next);
        return timetableResult(next, catalog);
      },
    ),
  );
}

function registerExportTools(server: McpServer, ctx: ToolContext): void {
  server.registerTool(
    "export_manifest",
    {
      description:
        "Export the composition as a manifest file under data/exports/<id>-r<revision>.manifest.json. " +
        "Fails (writing nothing) if any referenced candidate is missing from the catalog.",
      inputSchema: { composition_id: z.string().describe("Composition id from create_composition / list_compositions") },
    },
    guard(({ composition_id }: { composition_id: string }) => {
      const comp = ctx.store.load(composition_id);
      const manifest = buildManifest(comp, ctx.catalog(), ctx.now());
      const path = ctx.store.writeExport(manifest);
      const captions_by_dataset: Record<Dataset, number> = { nextme800: 0, hdepic: 0, ego4d: 0, egoexo4d: 0 };
      for (const c of manifest.tracks.captions) captions_by_dataset[c.provenance.dataset] += 1;
      const blocks = manifest.blocks.length;
      const text = [
        `manifest saved: ${path}`,
        `total ${formatTotal(manifest.duration)} · blocks: ${blocks} · rev ${manifest.composition_revision}`,
        `caption: nextme800 ${captions_by_dataset.nextme800} · hdepic ${captions_by_dataset.hdepic} · ego4d ${captions_by_dataset.ego4d} · egoexo4d ${captions_by_dataset.egoexo4d}`,
      ].join("\n");
      return ok(text, { path, duration: manifest.duration, blocks, captions_by_dataset });
    }),
  );
}
