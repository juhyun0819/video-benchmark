import { randomUUID } from "node:crypto";
import { resolve } from "node:path";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import {
  RULE_IDS,
  TimelineError,
  assertRevision,
  assignClip,
  buildTimelineManifest,
  candidatesFor,
  createTimeline,
  fillTimeline,
  generateSchedule,
  preflightCapacity,
  rankSelector,
  unassignClip,
  verifyTimeline,
  type RankedCandidate,
  type Slot,
  type Timeline,
  type TimelineManifest,
  type World,
} from "@vb/core";
import { loadConfig, regenFor, scheduleInput, type Dirs, type GeneratorConfig } from "./config.js";
import { FsClipCatalog } from "./fsCatalog.js";
import { TimelineStore, WorldStore } from "./store.js";
import { renderDay, renderOverview } from "./view.js";

export type ServerOptions = {
  now?: () => Date;
  /** Suffix of new timeline ids (`<person_id>-<suffix>`). */
  genId?: () => string;
};

type ToolContext = {
  now: () => Date;
  genId: () => string;
  worlds: WorldStore;
  timelines: TimelineStore;
  /** Read per call, so re-ingested catalogs and edited configs are picked up without a restart. */
  catalog: () => FsClipCatalog;
  config: () => GeneratorConfig;
};

export type ErrorReport = { code: string; message: string; details: Record<string, unknown> };

/** TimelineError keeps its code and details; anything else is INTERNAL. */
export function describeError(e: unknown): ErrorReport {
  if (e instanceof TimelineError) return { code: e.code, message: e.message, details: e.details };
  return { code: "INTERNAL", message: e instanceof Error ? e.message : String(e), details: {} };
}

export function toToolError(e: unknown): CallToolResult {
  const { code, message, details } = describeError(e);
  return { isError: true, content: [{ type: "text", text: `${code}: ${message}\n${JSON.stringify(details, null, 2)}` }] };
}

function ok(text: string, structured: Record<string, unknown>): CallToolResult {
  return { content: [{ type: "text", text }], structuredContent: structured };
}

/** Runs a handler and converts thrown errors into `isError` tool results. */
function guard<A>(fn: (args: A) => CallToolResult): (args: A) => CallToolResult {
  return (args) => {
    try {
      return fn(args);
    } catch (e) {
      return toToolError(e);
    }
  };
}

const round3 = (x: number): number => Math.round(x * 1000) / 1000;

export type ManifestSummary = { days: number; clips: number; captions: number; coverage_mean: number };

export function manifestSummary(m: TimelineManifest): ManifestSummary {
  const clips = m.days.reduce((n, d) => n + d.clips.length, 0);
  const coverage = m.days.reduce((s, d) => s + d.coverage_ratio, 0);
  return {
    days: m.days.length,
    clips,
    captions: m.captions.length,
    coverage_mean: m.days.length > 0 ? round3(coverage / m.days.length) : 0,
  };
}

/** A fresh timeline for the world: schedule, empty slots, bank-capacity preflight (§5, D7). Nothing is saved. */
export function newTimeline(
  world: World,
  config: GeneratorConfig,
  catalog: FsClipCatalog,
  genId: () => string,
  name?: string,
): Timeline {
  const schedule = generateSchedule(scheduleInput(config, world));
  const t = createTimeline(world, schedule, () => `${world.person_id}-${genId()}`);
  const named = name !== undefined ? { ...t, name } : t;
  preflightCapacity(named, catalog);
  return named;
}

function locate(t: Timeline, slotId: string): { day: number; slot: Slot } | undefined {
  for (const d of t.schedule.days) {
    const slot = d.slots.find((s) => s.slot_id === slotId);
    if (slot) return { day: d.day_index, slot };
  }
  return undefined;
}

function candidateView(c: RankedCandidate) {
  return {
    clip_id: c.clip.clip_id,
    score: c.score,
    reasons: c.reasons,
    activity: c.clip.activity,
    dataset: c.clip.source.dataset,
    video_id: c.clip.source.video_id,
    day_group: c.clip.day_group,
    duration_s: c.clip.source_end_s - c.clip.source_start_s,
    time_of_day: c.clip.time_of_day,
  };
}

export function createServer(dirs: Dirs, opts: ServerOptions = {}): McpServer {
  // Resolve once so every returned path is absolute.
  const dataDir = resolve(dirs.dataDir);
  const configDir = resolve(dirs.configDir);
  const ctx: ToolContext = {
    now: opts.now ?? (() => new Date()),
    genId: opts.genId ?? (() => randomUUID().slice(0, 8)),
    worlds: new WorldStore(dataDir),
    timelines: new TimelineStore(dataDir),
    catalog: () => new FsClipCatalog(dataDir),
    config: () => loadConfig(configDir, dataDir),
  };
  const server = new McpServer({ name: "video-composer", version: "0.0.0" });
  registerTools(server, ctx);
  return server;
}

function registerTools(server: McpServer, ctx: ToolContext): void {
  const timelineId = z.string().describe("Timeline id from create_timeline / list_timelines");
  const slotId = z.string().describe("Slot id such as d3-s2, from get_timeline");
  const expectedRevision = z
    .number()
    .int()
    .min(0)
    .describe("Current revision of the timeline; the edit is rejected when it differs");
  const relax = z
    .array(z.enum(["ignore_tags", "widen_window_30m"]))
    .optional()
    .describe("Relaxation ladder steps (§6.4): ignore_tags, widen_window_30m");

  server.registerTool(
    "list_worlds",
    { description: "List the worlds (one synthetic person each) under data/worlds." },
    guard(() => {
      const worlds = ctx.worlds.list().map((w) => ({
        person_id: w.person_id,
        length_days: w.dials.length_days,
        seed: w.seed,
        nodes: Object.keys(w.nodes),
      }));
      const text =
        worlds.length === 0
          ? "(no worlds)"
          : worlds.map((w) => `${w.person_id}  ${w.length_days} days  seed ${w.seed}  nodes ${w.nodes.join(",")}`).join("\n");
      return ok(text, { worlds });
    }),
  );

  server.registerTool(
    "create_timeline",
    {
      description:
        "Generate the schedule for a world, check that the clip bank can cover it, and save an empty timeline. Returns its id and the day 1 view.",
      inputSchema: { person_id: z.string().describe("World person_id from list_worlds"), name: z.string().optional() },
    },
    guard(({ person_id, name }: { person_id: string; name?: string | undefined }) => {
      const world = ctx.worlds.load(person_id);
      const catalog = ctx.catalog();
      const t = newTimeline(world, ctx.config(), catalog, ctx.genId, name);
      ctx.timelines.save(t);
      const head = `created timeline ${t.id} revision ${t.revision} (${t.schedule.days.length} days)`;
      return ok(`${head}\n${renderDay(t, catalog, 1)}`, {
        timeline_id: t.id,
        revision: t.revision,
        days: t.schedule.days.length,
      });
    }),
  );

  server.registerTool(
    "list_timelines",
    { description: "List saved timelines with id, name, person and revision." },
    guard(() => {
      const timelines = ctx.timelines.list();
      const text =
        timelines.length === 0
          ? "(no timelines)"
          : timelines
              .map((t) => `${t.id}${t.name ? ` (${t.name})` : ""}  ${t.person_id}  r${t.revision}  ${t.days} days`)
              .join("\n");
      return ok(text, { timelines });
    }),
  );

  server.registerTool(
    "get_timeline",
    {
      description:
        "Show a timeline. With `day`: one line per slot (slot_id window role node clip_ids relaxations). Without: one line per day.",
      inputSchema: { timeline_id: timelineId, day: z.number().int().min(1).optional() },
    },
    guard(({ timeline_id, day }: { timeline_id: string; day?: number | undefined }) => {
      const t = ctx.timelines.load(timeline_id);
      const text = day === undefined ? renderOverview(t) : renderDay(t, ctx.catalog(), day);
      return ok(text, { timeline_id: t.id, revision: t.revision });
    }),
  );

  server.registerTool(
    "list_slot_candidates",
    {
      description:
        "Ranked clips that pass every hard filter for a slot (§6.3). Prefer the same day_group for slots at the same node on one day; follow routine strength when choosing between clips similar to yesterday's and different ones.",
      inputSchema: {
        timeline_id: timelineId,
        slot_id: slotId,
        relax,
        limit: z.number().int().min(1).max(100).default(10),
      },
    },
    guard(
      ({ timeline_id, slot_id, relax: r, limit }: { timeline_id: string; slot_id: string; relax?: Relaxable[] | undefined; limit: number }) => {
        const t = ctx.timelines.load(timeline_id);
        const ranked = candidatesFor(t, ctx.catalog(), slot_id, r ?? []);
        const candidates = ranked.slice(0, limit).map(candidateView);
        const slot = locate(t, slot_id)!.slot;
        const head = `${slot.slot_id} ${slot.window[0]}-${slot.window[1]} ${slot.role} ${slot.spec.node} ${slot.spec.category}/${slot.spec.setting} "${slot.spec.activity}" — ${ranked.length} candidates`;
        const lines = candidates.map(
          (c) =>
            `${c.clip_id}  ${c.score.toFixed(3)}  "${c.activity}"  ${c.day_group}  ${c.duration_s}s  ${c.time_of_day}${c.reasons.length > 0 ? `  [${c.reasons.join("; ")}]` : ""}`,
        );
        return ok([head, ...lines].join("\n"), {
          timeline_id: t.id,
          revision: t.revision,
          slot,
          relax: r ?? [],
          total: ranked.length,
          candidates,
        });
      },
    ),
  );

  server.registerTool(
    "assign_clip",
    {
      description: "Add a clip to a slot after checking the hard filters at the given relaxations.",
      inputSchema: { timeline_id: timelineId, slot_id: slotId, clip_id: z.string(), expected_revision: expectedRevision, relax },
    },
    guard(
      (a: { timeline_id: string; slot_id: string; clip_id: string; expected_revision: number; relax?: Relaxable[] | undefined }) => {
        const t = ctx.timelines.load(a.timeline_id);
        assertRevision(t, a.expected_revision);
        const catalog = ctx.catalog();
        const next = assignClip(t, catalog, { slotId: a.slot_id, clipId: a.clip_id, relaxations: a.relax ?? [] });
        ctx.timelines.save(next);
        return editResult(next, catalog, a.slot_id, `assigned ${a.clip_id} to ${a.slot_id}`);
      },
    ),
  );

  server.registerTool(
    "unassign_clip",
    {
      description: "Remove one clip from a slot, or every clip of the slot when clip_id is omitted.",
      inputSchema: { timeline_id: timelineId, slot_id: slotId, clip_id: z.string().optional(), expected_revision: expectedRevision },
    },
    guard((a: { timeline_id: string; slot_id: string; clip_id?: string | undefined; expected_revision: number }) => {
      const t = ctx.timelines.load(a.timeline_id);
      assertRevision(t, a.expected_revision);
      const next = unassignClip(t, { slotId: a.slot_id, ...(a.clip_id !== undefined ? { clipId: a.clip_id } : {}) });
      ctx.timelines.save(next);
      const what = a.clip_id !== undefined ? `removed ${a.clip_id} from ${a.slot_id}` : `cleared ${a.slot_id}`;
      return editResult(next, ctx.catalog(), a.slot_id, what);
    }),
  );

  server.registerTool(
    "auto_fill",
    {
      description:
        "Fill every empty slot with the top-ranked candidate, walking the relaxation ladder and regenerating days that cannot be filled (§6.4).",
      inputSchema: { timeline_id: timelineId, expected_revision: expectedRevision },
    },
    guard(({ timeline_id, expected_revision }: { timeline_id: string; expected_revision: number }) => {
      const t = ctx.timelines.load(timeline_id);
      assertRevision(t, expected_revision);
      const next = fillTimeline(t, ctx.catalog(), rankSelector, regenFor(ctx.config(), t));
      ctx.timelines.save(next);
      const entries = Object.values(next.slots);
      const filled_slots = entries.filter((e) => e.clip_ids.length > 0).length;
      const dropped_optional = entries.filter((e) => e.relaxations.includes("dropped_optional")).length;
      const regenerated_days = next.schedule.days
        .filter((d, i) => d.attempt !== t.schedule.days[i]?.attempt)
        .map((d) => d.day_index);
      const text = `filled timeline ${next.id} revision ${next.revision}: ${filled_slots}/${entries.length} slots filled, ${dropped_optional} optional slots dropped, regenerated days: ${regenerated_days.join(",") || "-"}`;
      return ok(text, { timeline_id: next.id, revision: next.revision, filled_slots, dropped_optional, regenerated_days });
    }),
  );

  server.registerTool(
    "verify_timeline",
    {
      description: "Run the rule checks (§7.1) and write data/timelines/<id>.verifier_report.json.",
      inputSchema: { timeline_id: timelineId },
    },
    guard(({ timeline_id }: { timeline_id: string }) => {
      const t = ctx.timelines.load(timeline_id);
      const report = verifyTimeline(t, ctx.catalog());
      const path = ctx.timelines.writeReport(t.id, report);
      const failed_rules = RULE_IDS.filter((r) => !report.rules[r].pass);
      const lines = [
        `${report.pass ? "pass" : "FAIL"} revision ${t.revision}: ${report.flags.length} swap-point flags, ${report.seams.length} seams; report ${path}`,
        ...failed_rules.flatMap((r) =>
          report.rules[r].violations.slice(0, 10).map((v) => `${r}: ${v.slot_id ?? (v.day_index !== undefined ? `day ${v.day_index}` : "-")} ${v.detail}`),
        ),
      ];
      return ok(lines.join("\n"), {
        timeline_id: t.id,
        revision: t.revision,
        path,
        pass: report.pass,
        failed_rules,
        flags: report.flags.length,
        seams: report.seams.length,
      });
    }),
  );

  server.registerTool(
    "export_manifest",
    {
      description: "Build the timeline manifest (§8) from a verified timeline and write data/exports/<id>-r<revision>.manifest.json.",
      inputSchema: { timeline_id: timelineId },
    },
    guard(({ timeline_id }: { timeline_id: string }) => {
      const t = ctx.timelines.load(timeline_id);
      const manifest = buildTimelineManifest(t, ctx.catalog(), ctx.now());
      const path = ctx.timelines.writeExport(manifest);
      const s = manifestSummary(manifest);
      const text = `wrote ${path}: ${s.days} days, ${s.clips} clips, ${s.captions} captions, coverage mean ${s.coverage_mean}`;
      return ok(text, { path, ...s });
    }),
  );
}

type Relaxable = "ignore_tags" | "widen_window_30m";

function editResult(t: Timeline, catalog: FsClipCatalog, slotId: string, what: string): CallToolResult {
  const day = locate(t, slotId)!.day;
  return ok(`${what}; revision ${t.revision}\n${renderDay(t, catalog, day)}`, { timeline_id: t.id, revision: t.revision });
}
