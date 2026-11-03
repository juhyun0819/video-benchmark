export type CompositionErrorCode =
  | "BASE_DAY_NOT_FOUND"
  | "SLOT_OUT_OF_RANGE"
  | "POSITION_OUT_OF_RANGE"
  | "CANDIDATE_NOT_FOUND"
  | "NOT_EVENT"
  | "REF_NOT_FOUND"
  | "REVISION_MISMATCH"
  | "MISSING_CANDIDATES"
  | "COMPOSITION_NOT_FOUND"
  | "BASE_DAY_CHANGED";

export class CompositionError extends Error {
  readonly code: CompositionErrorCode;
  readonly details: Record<string, unknown>;

  constructor(code: CompositionErrorCode, message: string, details: Record<string, unknown> = {}) {
    super(message);
    this.name = "CompositionError";
    this.code = code;
    this.details = details;
  }
}
