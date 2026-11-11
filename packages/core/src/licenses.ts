import type { Dataset } from "./schema.js";

export const LICENSES: Record<Dataset, { license: string; note?: string }> = {
  nextme800: { license: "CC BY-NC 4.0", note: "Research/evaluation only; gated (mmm8383/NextMe-800)" },
  ego4d: { license: "Ego4D License Agreement", note: "https://ego4d-data.org/" },
  egoexo4d: { license: "Ego-Exo4D License Agreement", note: "https://ego-exo4d-data.org/" },
  youtube: { license: "YouTube creator content, not redistributed", note: "Manifest lists video ids and spans only" },
};
