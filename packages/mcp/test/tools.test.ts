import { readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { TimelineManifestSchema, TimelineSchema, WorldSchema } from "@vb/core";
import {
  EXAMPLE_WORLD,
  PERSON_ID,
  POOL_REL,
  connect,
  dataOf,
  filesIn,
  makeBank,
  setupDataDir,
  textOf,
  writeDataset,
} from "./helpers.js";

type Session = Awaited<ReturnType<typeof connect>>;
let session: Session | undefined;
afterEach(async () => {
  await session?.close();
  session = undefined;
});

async function open(dataDir: string): Promise<Session> {
  session = await connect(dataDir);
  return session;
}

type Created = { timeline_id: string; revision: number };

async function createAndFill(s: Session): Promise<Created> {
  const created = dataOf<Created>(await s.call("create_timeline", { person_id: PERSON_ID }));
  const filled = dataOf<Created>(
    await s.call("auto_fill", { timeline_id: created.timeline_id, expected_revision: created.revision }),
  );
  return filled;
}

const readTimeline = (dataDir: string, id: string) =>
  TimelineSchema.parse(JSON.parse(readFileSync(join(dataDir, "timelines", `${id}.json`), "utf8")));

describe("example world", () => {
  it("parses with WorldSchema", () => {
    const world = WorldSchema.parse(JSON.parse(readFileSync(EXAMPLE_WORLD, "utf8")));
    expect(world.person_id).toBe(PERSON_ID);
    expect(world.attributes.country).toBe("HK");
    expect(world.allow_country_fallback).toBe(true);
    expect(world.dials.length_days).toBe(28);
  });
});

describe("timeline tools", () => {
  it("full flow on fixture: create → auto_fill → verify → export writes manifest", async () => {
    const dataDir = setupDataDir();
    const s = await open(dataDir);

    const worlds = dataOf<{ worlds: { person_id: string }[] }>(await s.call("list_worlds"));
    expect(worlds.worlds.map((w) => w.person_id)).toEqual([PERSON_ID]);

    const createRes = await s.call("create_timeline", { person_id: PERSON_ID, name: "flow" });
    const created = dataOf<Created>(createRes);
    expect(created.revision).toBe(0);
    expect(textOf(createRes)).toMatch(/^d1-s1 /m);

    const filled = dataOf<Created & { filled_slots: number }>(
      await s.call("auto_fill", { timeline_id: created.timeline_id, expected_revision: 0 }),
    );
    expect(filled.revision).toBe(1);
    expect(filled.filled_slots).toBeGreaterThan(0);

    const listed = dataOf<{ timelines: { id: string; name?: string; revision: number }[] }>(
      await s.call("list_timelines"),
    );
    expect(listed.timelines).toEqual([expect.objectContaining({ id: created.timeline_id, name: "flow", revision: 1 })]);

    const day2 = await s.call("get_timeline", { timeline_id: created.timeline_id, day: 2 });
    expect(textOf(day2)).toMatch(/^d2-s1 .*nextme800:nm-g\d\d:/m);

    const verified = dataOf<{ path: string; pass: boolean; failed_rules: string[]; flags: number; seams: number }>(
      await s.call("verify_timeline", { timeline_id: created.timeline_id }),
    );
    expect(verified.pass).toBe(true);
    expect(verified.failed_rules).toEqual([]);
    expect(verified.path).toBe(join(dataDir, "timelines", `${created.timeline_id}.verifier_report.json`));
    expect(JSON.parse(readFileSync(verified.path, "utf8")).pass).toBe(true);

    const exported = dataOf<{ path: string; days: number; clips: number; captions: number; coverage_mean: number }>(
      await s.call("export_manifest", { timeline_id: created.timeline_id }),
    );
    expect(exported.path).toBe(join(dataDir, "exports", `${created.timeline_id}-r1.manifest.json`));
    const manifest = TimelineManifestSchema.parse(JSON.parse(readFileSync(exported.path, "utf8")));
    expect(exported.days).toBe(28);
    expect(manifest.days).toHaveLength(28);
    expect(exported.clips).toBe(manifest.days.reduce((n, d) => n + d.clips.length, 0));
    expect(exported.captions).toBe(manifest.captions.length);
    expect(exported.captions).toBeGreaterThan(0);
    expect(exported.coverage_mean).toBeGreaterThan(0);
  });

  it("assign with stale expected_revision returns REVISION_MISMATCH and does not save", async () => {
    const dataDir = setupDataDir();
    const s = await open(dataDir);
    const { timeline_id } = dataOf<Created>(await s.call("create_timeline", { person_id: PERSON_ID }));
    const { candidates } = dataOf<{ candidates: { clip_id: string }[] }>(
      await s.call("list_slot_candidates", { timeline_id, slot_id: "d1-s1" }),
    );
    const before = readFileSync(join(dataDir, "timelines", `${timeline_id}.json`), "utf8");

    const r = await s.call("assign_clip", {
      timeline_id,
      slot_id: "d1-s1",
      clip_id: candidates[0]!.clip_id,
      expected_revision: 3,
    });
    expect(r.isError).toBe(true);
    expect(textOf(r)).toMatch(/^REVISION_MISMATCH: /);
    expect(textOf(r)).toContain('"actual": 0');
    expect(readFileSync(join(dataDir, "timelines", `${timeline_id}.json`), "utf8")).toBe(before);

    const ok = dataOf<Created>(
      await s.call("assign_clip", { timeline_id, slot_id: "d1-s1", clip_id: candidates[0]!.clip_id, expected_revision: 0 }),
    );
    expect(ok.revision).toBe(1);
    expect(readTimeline(dataDir, timeline_id).slots["d1-s1"]!.clip_ids).toEqual([candidates[0]!.clip_id]);

    const cleared = dataOf<Created>(await s.call("unassign_clip", { timeline_id, slot_id: "d1-s1", expected_revision: 1 }));
    expect(cleared.revision).toBe(2);
    expect(readTimeline(dataDir, timeline_id).slots["d1-s1"]!.clip_ids).toEqual([]);
  });

  it("list_slot_candidates returns ≤ limit, sorted", async () => {
    const dataDir = setupDataDir();
    const s = await open(dataDir);
    const { timeline_id } = dataOf<Created>(await s.call("create_timeline", { person_id: PERSON_ID }));
    type Cand = {
      clip_id: string;
      score: number;
      reasons: string[];
      activity: string;
      dataset: string;
      video_id: string;
      day_group: string;
      duration_s: number;
      time_of_day: string;
    };
    const res = dataOf<{ relax: string[]; total: number; candidates: Cand[] }>(
      await s.call("list_slot_candidates", { timeline_id, slot_id: "d1-s1", relax: ["ignore_tags"], limit: 3 }),
    );
    expect(res.relax).toEqual(["ignore_tags"]);
    expect(res.total).toBeGreaterThan(3);
    expect(res.candidates).toHaveLength(3);
    const scores = res.candidates.map((c) => c.score);
    expect(scores).toEqual([...scores].sort((a, b) => b - a));
    expect(Object.keys(res.candidates[0]!).sort()).toEqual(
      ["activity", "clip_id", "dataset", "day_group", "duration_s", "reasons", "score", "time_of_day", "video_id"],
    );
    expect(res.candidates[0]).toMatchObject({ dataset: "nextme800", duration_s: 300, time_of_day: "unknown" });
  });

  it("list_slot_candidates shows the previous days' picks and the routine strength (§6.3)", async () => {
    const dataDir = setupDataDir();
    const s = await open(dataDir);
    const { timeline_id } = dataOf<Created>(await s.call("create_timeline", { person_id: PERSON_ID }));
    const first = dataOf<{ candidates: { clip_id: string; activity: string; day_group: string }[] }>(
      await s.call("list_slot_candidates", { timeline_id, slot_id: "d1-s1" }),
    ).candidates[0]!;
    await s.call("assign_clip", { timeline_id, slot_id: "d1-s1", clip_id: first.clip_id, expected_revision: 0 });
    const role = readTimeline(dataDir, timeline_id).schedule.days[0]!.slots[0]!.role;
    const routine = WorldSchema.parse(JSON.parse(readFileSync(EXAMPLE_WORLD, "utf8"))).dials.routine_strength;

    const res = await s.call("list_slot_candidates", { timeline_id, slot_id: "d2-s1" });
    const data = dataOf<{ recent: unknown[]; routine_strength: number }>(res);
    expect(data.routine_strength).toBe(routine);
    expect(data.recent).toEqual([
      { day: 1, role, clip_id: first.clip_id, activity: first.activity, day_group: first.day_group },
    ]);
    expect(textOf(res)).toContain(`routine_strength ${routine}`);
    expect(textOf(res)).toContain(`day 1 ${role} ${first.clip_id}`);
  });

  it("create_timeline with a 10-day_group bank returns BANK_CAPACITY and saves nothing", async () => {
    const dataDir = setupDataDir({ groups: 10 });
    const s = await open(dataDir);
    const r = await s.call("create_timeline", { person_id: PERSON_ID });
    expect(r.isError).toBe(true);
    expect(textOf(r)).toMatch(/^BANK_CAPACITY: /);
    expect(textOf(r)).toContain('"distinct_day_groups": 10');
    expect(filesIn(dataDir, "timelines")).toEqual([]);
  });

  it("export after re-ingest removed a clip returns MISSING_CLIPS listing it, no file written", async () => {
    const dataDir = setupDataDir();
    const s = await open(dataDir);
    const { timeline_id } = await createAndFill(s);
    const used = readTimeline(dataDir, timeline_id).slots["d1-s1"]!.clip_ids[0]!;
    expect(used).toMatch(/^nextme800:/);

    writeDataset(dataDir, "nextme800", makeBank().nextme800.filter((c) => c.clip_id !== used));

    const r = await s.call("export_manifest", { timeline_id });
    expect(r.isError).toBe(true);
    expect(textOf(r)).toMatch(/^MISSING_CLIPS: /);
    expect(textOf(r)).toContain(used);
    expect(filesIn(dataDir, "exports")).toEqual([]);
  });

  it("missing ATUS pool file names its path", async () => {
    const dataDir = setupDataDir({ pool: false });
    const s = await open(dataDir);
    const r = await s.call("create_timeline", { person_id: PERSON_ID });
    expect(r.isError).toBe(true);
    expect(textOf(r)).toContain(join(dataDir, POOL_REL));
    expect(filesIn(dataDir, "timelines")).toEqual([]);
  });

  it("unknown world returns WORLD_NOT_FOUND", async () => {
    const dataDir = setupDataDir();
    const s = await open(dataDir);
    const r = await s.call("create_timeline", { person_id: "nobody" });
    expect(r.isError).toBe(true);
    expect(textOf(r)).toMatch(/^WORLD_NOT_FOUND: /);
    expect(filesIn(dataDir, "timelines")).toEqual([]);
  });
});
