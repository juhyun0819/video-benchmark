import { describe, expect, it } from "vitest";
import {
  TimelineError,
  TimelineSchema,
  assertRevision,
  assignClip,
  bindingAccepts,
  createTimeline,
  fillTimeline,
  rankSelector,
  unassignClip,
} from "../src/index.js";
import { clip, genId, makeClips, makeSchedule, makeWorld, memoryCatalog, type SlotDef } from "./fixtures.js";

const catalog = memoryCatalog(makeClips());
const fresh = () => createTimeline(makeWorld(), makeSchedule(28), genId);

function codeOf(fn: () => unknown): { code: string; details: Record<string, unknown> } {
  try {
    fn();
  } catch (e) {
    if (e instanceof TimelineError) return { code: e.code, details: e.details };
    throw e;
  }
  throw new Error("expected a TimelineError");
}

describe("createTimeline", () => {
  it("starts at revision 0 with an empty entry for every slot", () => {
    const t = fresh();
    expect(TimelineSchema.parse(t)).toEqual(t);
    expect(t.schema_version).toBe(1);
    expect(t.revision).toBe(0);
    const ids = t.schedule.days.flatMap((d) => d.slots.map((s) => s.slot_id));
    expect(Object.keys(t.slots)).toEqual(ids);
    for (const id of ids) expect(t.slots[id]).toEqual({ clip_ids: [], relaxations: [] });
  });
});

describe("assignClip", () => {
  it("assign increments revision", () => {
    const t0 = fresh();
    const t1 = assignClip(t0, catalog, { slotId: "d1-s1", clipId: "g01-a-morning" });
    expect(t1.revision).toBe(1);
    expect(t1.slots["d1-s1"]).toEqual({ clip_ids: ["g01-a-morning"], relaxations: [] });
    expect(t0.revision).toBe(0);
    expect(t0.slots["d1-s1"]!.clip_ids).toEqual([]);
  });

  it('assign rejects clip of another node with CLIP_INELIGIBLE ["category_setting_node"]', () => {
    // A morning work clip in the morning home slot: only the node check fails.
    const err = codeOf(() => assignClip(fresh(), catalog, { slotId: "d1-s1", clipId: "g01-b-work-am" }));
    expect(err).toEqual({ code: "CLIP_INELIGIBLE", details: expect.objectContaining({ failed: ["category_setting_node"] }) });
  });

  it("assign rejects second use of a clip (already_used)", () => {
    const t = assignClip(fresh(), catalog, { slotId: "d1-s1", clipId: "g01-a-morning" });
    const err = codeOf(() => assignClip(t, catalog, { slotId: "d1-s1", clipId: "g01-a-morning" }));
    expect(err.code).toBe("CLIP_INELIGIBLE");
    expect(err.details.failed).toEqual(["already_used"]);
  });

  it("assign rejects a clip whose source span overlaps an assigned clip of the same video (already_used)", () => {
    const a = clip({ clip_id: "walk-a", source_start_s: 0, source_end_s: 600 });
    const b = clip({ clip_id: "walk-b", source: a.source, source_start_s: 300, source_end_s: 900, day_group: a.day_group });
    const c = clip({ clip_id: "walk-c", source: a.source, source_start_s: 600, source_end_s: 900, day_group: a.day_group });
    const cat = memoryCatalog([a, b, c]);
    const t = assignClip(fresh(), cat, { slotId: "d1-s1", clipId: "walk-a" });
    expect(codeOf(() => assignClip(t, cat, { slotId: "d1-s1", clipId: "walk-b" })).details.failed).toEqual(["already_used"]);
    expect(assignClip(t, cat, { slotId: "d1-s1", clipId: "walk-c" }).slots["d1-s1"]!.clip_ids).toEqual(["walk-a", "walk-c"]);
  });

  it("assign rejects a day_group used on another day", () => {
    const t = assignClip(fresh(), catalog, { slotId: "d1-s1", clipId: "g01-a-morning" });
    const err = codeOf(() => assignClip(t, catalog, { slotId: "d2-s6", clipId: "g01-f-evening" }));
    expect(err.code).toBe("CLIP_INELIGIBLE");
    expect(err.details.failed).toEqual(["day_group_other_day"]);
    // Same day is fine.
    expect(assignClip(t, catalog, { slotId: "d1-s6", clipId: "g01-f-evening" }).revision).toBe(2);
  });

  it("records relaxations and checks the filters at those relaxations", () => {
    const t = createTimeline(makeWorld(), makeSchedule(1, () => [
      { role: "chores", category: "home", setting: "home_indoor", node: "home", activity: "doing chores", tags: ["chores"], required: true, window: ["07:00", "08:00"] },
    ]), genId);
    expect(codeOf(() => assignClip(t, catalog, { slotId: "d1-s1", clipId: "g01-a-morning" })).details.failed).toEqual(["tags"]);
    const ok = assignClip(t, catalog, { slotId: "d1-s1", clipId: "g01-a-morning", relaxations: ["ignore_tags"] });
    expect(ok.slots["d1-s1"]).toEqual({ clip_ids: ["g01-a-morning"], relaxations: ["ignore_tags"] });
  });

  it("assigning into a slot that fillTimeline dropped removes dropped_optional", () => {
    const diner: SlotDef = { role: "dinner", category: "meal", setting: "venue_indoor", node: "diner", activity: "eating dinner", required: false, window: ["18:30", "19:30"] };
    const dinerClip = clip({ clip_id: "diner-1", category: "meal", setting: "venue_indoor", node: "diner", activity: "eating dinner", time_of_day: "evening" });
    const dropped = fillTimeline(createTimeline(makeWorld(), makeSchedule(1, () => [diner]), genId), catalog, rankSelector, () => {
      throw new Error("unexpected regen");
    });
    expect(dropped.slots["d1-s1"]).toEqual({ clip_ids: [], relaxations: ["dropped_optional"] });
    const t = assignClip(dropped, memoryCatalog([dinerClip]), { slotId: "d1-s1", clipId: "diner-1", relaxations: ["ignore_tags"] });
    expect(t.slots["d1-s1"]).toEqual({ clip_ids: ["diner-1"], relaxations: ["ignore_tags"] });
  });

  it("rejects dropped_optional as an input relaxation", () => {
    const err = codeOf(() =>
      assignClip(fresh(), catalog, { slotId: "d1-s1", clipId: "g01-a-morning", relaxations: ["dropped_optional"] }),
    );
    expect(err.code).toBe("CLIP_INELIGIBLE");
    expect(err.details.failed).toEqual(["invalid_relaxation"]);
  });

  it("throws SLOT_NOT_FOUND and CLIP_NOT_FOUND", () => {
    expect(codeOf(() => assignClip(fresh(), catalog, { slotId: "d99-s1", clipId: "g01-a-morning" })).code).toBe("SLOT_NOT_FOUND");
    expect(codeOf(() => assignClip(fresh(), catalog, { slotId: "d1-s1", clipId: "nope" })).code).toBe("CLIP_NOT_FOUND");
  });
});

describe("unassignClip", () => {
  it("unassign frees it", () => {
    let t = assignClip(fresh(), catalog, { slotId: "d1-s1", clipId: "g01-a-morning" });
    t = unassignClip(t, { slotId: "d1-s1", clipId: "g01-a-morning" });
    expect(t.revision).toBe(2);
    expect(t.slots["d1-s1"]).toEqual({ clip_ids: [], relaxations: [] });
    // Clip and its day_group are free again, on another day.
    t = assignClip(t, catalog, { slotId: "d2-s1", clipId: "g01-a-morning" });
    expect(t.slots["d2-s1"]!.clip_ids).toEqual(["g01-a-morning"]);
  });

  it("without clipId clears the slot", () => {
    let t = assignClip(fresh(), catalog, { slotId: "d1-s6", clipId: "g01-f-evening" });
    t = assignClip(t, catalog, { slotId: "d1-s6", clipId: "g02-f-evening" });
    t = unassignClip(t, { slotId: "d1-s6" });
    expect(t.slots["d1-s6"]).toEqual({ clip_ids: [], relaxations: [] });
    expect(t.revision).toBe(3);
  });
});

describe("assertRevision", () => {
  it("passes when expected matches or is absent, throws REVISION_MISMATCH otherwise", () => {
    const t = assignClip(fresh(), catalog, { slotId: "d1-s1", clipId: "g01-a-morning" });
    expect(() => assertRevision(t)).not.toThrow();
    expect(() => assertRevision(t, 1)).not.toThrow();
    expect(codeOf(() => assertRevision(t, 0))).toEqual({ code: "REVISION_MISMATCH", details: { expected: 0, actual: 1 } });
  });
});

describe("bindingAccepts", () => {
  const world = makeWorld();
  it("checks every present binding field; kitchen delegates to its default node", () => {
    const homeClip = clip({ clip_id: "x" });
    expect(bindingAccepts(world, "home", homeClip)).toBe(true);
    expect(bindingAccepts(world, "kitchen", homeClip)).toBe(true);
    expect(bindingAccepts(world, "home", { ...homeClip, person_source: "p2" })).toBe(false);
    expect(bindingAccepts(world, "home", { ...homeClip, node: "work" })).toBe(false);
    expect(bindingAccepts(world, "home", { ...homeClip, source: { ...homeClip.source, dataset: "ego4d" } })).toBe(false);
    const climb = clip({ clip_id: "y", source: { dataset: "ego4d", video_id: "v", media: { kind: "video", fps: 30 } }, person_source: "any" });
    expect(bindingAccepts(world, "climbing", { ...climb, tags: ["climbing"] })).toBe(true);
    expect(bindingAccepts(world, "climbing", { ...climb, tags: ["walking"] })).toBe(false);
    expect(bindingAccepts(world, "nowhere", homeClip)).toBe(false);
  });
});
