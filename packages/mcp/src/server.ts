import { randomUUID } from "node:crypto";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import { CompositionError, type Annotation, type BlockCandidate, type TimeSpan } from "@vb/core";
import { CatalogMissingError, FsCatalog } from "./fsCatalog.js";
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
  if (e instanceof CatalogMissingError) return "카탈로그가 없습니다. 먼저 npm run ingest를 실행하세요";
  if (e instanceof CompositionError) {
    const d = e.details;
    switch (e.code) {
      case "SLOT_OUT_OF_RANGE":
        return `slot ${String(d["slot"])}은 범위 밖 (0..${String(d["max"])})`;
      case "CANDIDATE_NOT_FOUND":
        return "candidate_id 없음 — list_event_candidates로 확인";
      case "REVISION_MISMATCH":
        return `revision 불일치 (현재 ${String(d["current"])})`;
      case "MISSING_CANDIDATES": {
        const ids = Array.isArray(d["ids"]) ? d["ids"].map(String) : [];
        return `카탈로그에 없는 후보: ${ids.join(", ")}`;
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

export function createServer(dataDir: string, opts: ServerOptions = {}): McpServer {
  const ctx: ToolContext = {
    dataDir,
    now: opts.now ?? (() => new Date()),
    genId: opts.genId ?? (() => randomUUID().slice(0, 8)),
    catalog: () => new FsCatalog(dataDir),
    store: new CompositionStore(dataDir),
  };
  const server = new McpServer({ name: "video-composer", version: "0.0.0" });
  registerBrowseTools(server, ctx);
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
          ? "(base day 없음)"
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
        throw new CompositionError("BASE_DAY_NOT_FOUND", `Base day ${day} not found — list_base_days로 확인`, { day });
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
          ? "(카테고리 없음)"
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
