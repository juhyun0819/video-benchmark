/** FNV-1a 32-bit hash of the UTF-16 code units of `s`. */
export function hash32(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** Deterministic mulberry32 generator, stable across Node versions. */
export class Rng {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  /** Uniform float in [0, 1). */
  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Uniform integer in [lo, hiExcl). */
  int(lo: number, hiExcl: number): number {
    return lo + Math.floor(this.next() * (hiExcl - lo));
  }

  /** Normal draw via Box-Muller. */
  normal(mean: number, sd: number): number {
    const u1 = 1 - this.next(); // (0, 1]
    const u2 = this.next();
    return mean + sd * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  bernoulli(p: number): boolean {
    return this.next() < p;
  }

  pickWeighted<T>(items: readonly { value: T; weight: number }[]): T {
    let total = 0;
    for (const it of items) total += it.weight;
    if (items.length === 0 || !(total > 0)) throw new Error("pickWeighted needs positive total weight");
    let x = this.next() * total;
    for (const it of items) {
      x -= it.weight;
      if (x < 0) return it.value;
    }
    return items[items.length - 1]!.value;
  }
}

/** Independent stream per (seed, keys). */
export function rngFor(seed: number, ...keys: (string | number)[]): Rng {
  return new Rng(hash32(`${seed}|${keys.join("|")}`));
}
