import type { Catalog } from "./catalog.js";
import { CompositionError } from "./errors.js";
import type { BaseDay, Composition, EventRef } from "./schema.js";

const isIndex = (n: number): boolean => Number.isInteger(n) && n >= 0;

function checkSlot(comp: Composition, slot: number): EventRef[] {
  const target = isIndex(slot) ? comp.slots[slot] : undefined;
  if (!target) {
    const max = comp.slots.length - 1;
    throw new CompositionError("SLOT_OUT_OF_RANGE", `Slot ${slot} is out of range (0..${max})`, { slot, max });
  }
  return target;
}

function checkPosition(position: number, max: number): void {
  if (!isIndex(position) || position > max) {
    throw new CompositionError("POSITION_OUT_OF_RANGE", `Position ${position} is out of range (0..${max})`, {
      position,
      max,
    });
  }
}

function cloneSlots(comp: Composition): EventRef[][] {
  return comp.slots.map((s) => s.map((r) => ({ ...r })));
}

function findRef(comp: Composition, refId: string): { slot: number; index: number } {
  for (let slot = 0; slot < comp.slots.length; slot++) {
    const index = comp.slots[slot]!.findIndex((r) => r.ref_id === refId);
    if (index >= 0) return { slot, index };
  }
  throw new CompositionError("REF_NOT_FOUND", `Event ref ${refId} not found`, { refId });
}

export function createComposition(
  baseDay: BaseDay,
  opts: { genId: () => string; name?: string },
): Composition {
  const comp: Composition = {
    schema_version: 1,
    id: opts.genId(),
    base_day: baseDay.day,
    revision: 0,
    slots: Array.from({ length: baseDay.blocks.length + 1 }, () => []),
  };
  if (opts.name !== undefined) comp.name = opts.name;
  return comp;
}

export function insertEvent(
  comp: Composition,
  catalog: Catalog,
  args: { slot: number; candidateId: string; position?: number },
  genId: () => string,
): Composition {
  const target = checkSlot(comp, args.slot);
  const position = args.position ?? target.length;
  checkPosition(position, target.length);

  const candidate = catalog.getCandidate(args.candidateId);
  if (!candidate) {
    throw new CompositionError("CANDIDATE_NOT_FOUND", `Candidate ${args.candidateId} not found`, {
      candidateId: args.candidateId,
    });
  }
  if (candidate.role !== "event") {
    throw new CompositionError("NOT_EVENT", `Candidate ${args.candidateId} is not an event block`, {
      candidateId: args.candidateId,
    });
  }

  const slots = cloneSlots(comp);
  slots[args.slot]!.splice(position, 0, { ref_id: genId(), candidate_id: args.candidateId });
  return { ...comp, revision: comp.revision + 1, slots };
}

export function moveEvent(
  comp: Composition,
  args: { refId: string; toSlot: number; toPosition?: number },
): Composition {
  const from = findRef(comp, args.refId);
  checkSlot(comp, args.toSlot);

  const slots = cloneSlots(comp);
  const [ref] = slots[from.slot]!.splice(from.index, 1);
  const target = slots[args.toSlot]!;
  const position = args.toPosition ?? target.length;
  checkPosition(position, target.length);
  target.splice(position, 0, ref!);
  return { ...comp, revision: comp.revision + 1, slots };
}

export function removeEvent(comp: Composition, args: { refId: string }): Composition {
  const from = findRef(comp, args.refId);
  const slots = cloneSlots(comp);
  slots[from.slot]!.splice(from.index, 1);
  return { ...comp, revision: comp.revision + 1, slots };
}

export function assertRevision(comp: Composition, expected?: number): void {
  if (expected !== undefined && expected !== comp.revision) {
    throw new CompositionError(
      "REVISION_MISMATCH",
      `Revision mismatch: expected ${expected}, current ${comp.revision}`,
      { expected, current: comp.revision },
    );
  }
}
