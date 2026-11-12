export type TimelineErrorCode =
  | "COUNTRY_SOURCE_MISSING"
  | "DONOR_POOL_EMPTY"
  | "SCHEDULE_INFEASIBLE"
  | "GAP_MISSING"
  | "EXCEPTION_OVERLAP"
  | "BANK_CAPACITY"
  | "UNFILLABLE"
  | "SLOT_NOT_FOUND"
  | "CLIP_NOT_FOUND"
  | "CLIP_INELIGIBLE"
  | "REVISION_MISMATCH"
  | "TIMELINE_NOT_FOUND"
  | "TIMELINE_INVALID"
  | "WORLD_NOT_FOUND"
  | "WORLD_INVALID"
  | "MISSING_CLIPS"
  | "VERIFY_FAILED"
  | "BLOCK_PLAN_INVALID";

export class TimelineError extends Error {
  readonly code: TimelineErrorCode;
  readonly details: Record<string, unknown>;

  constructor(code: TimelineErrorCode, message: string, details: Record<string, unknown> = {}) {
    super(message);
    this.name = "TimelineError";
    this.code = code;
    this.details = details;
  }
}
