import type { Group } from "../types";

export const GROUP_COLOR: Record<Group, string> = {
  struct: "#5d7183",
  node: "#f0b429",
  edge: "#3ddad7",
  scratch: "#9d7cff",
  flag: "#ff5c8a",
};

export const GROUP_LABEL: Record<Group, string> = {
  struct: "structural",
  node: "node payload",
  edge: "edge payload",
  scratch: "per-round scratch",
  flag: "cycle detection",
};

/** Human-facing description of every sublayer in a round. */
export const LAYER_DOC: Record<string, { title: string; detail: string }> = {
  input: {
    title: "Round input",
    detail: "State entering the block. All scratch registers are zero — the previous round cleaned up after itself, which is what lets the same weights run again.",
  },
  "attn:edge_reads_source": {
    title: "Edge reads its source",
    detail: "Each edge token attends to the node matching its EDGE_FROM one-hot and copies that node's DIST into SCRATCH_SRC_DIST. Logit gaps of 2500 make the softmax exactly one-hot.",
  },
  "mlp:candidate": {
    title: "Candidate = dist[u] + w",
    detail: "A ReLU pair adds source distance and edge weight into CAND, plus the INF guard: an unreachable source pushes the candidate above INF_B instead of leaking a finite value.",
  },
  "mlp:clamp_inf": {
    title: "Clamp to INF_B",
    detail: "CAND += -ReLU(CAND - INF_B). Guarded candidates land exactly on INF_B, so 'unreachable' never drifts round over round.",
  },
  "attn:node_gathers": {
    title: "Node gathers candidates",
    detail: "D_MAX heads. Head k routes node v's slot-k in-edge candidate into SCRATCH_GATHER[k]. Nodes with fewer in-edges have their spare heads fall back to self-attention, returning dist[v] — a no-op under min.",
  },
  "mlp:min_init": {
    title: "Seed the accumulator",
    detail: "SCRATCH_MIN += DIST. An exact two-ReLU copy, correct for either sign.",
  },
  "mlp:writeback": {
    title: "Delta write-back + cleanup",
    detail: "DIST += -ReLU(DIST - SCRATCH_MIN) — a difference, never an overwrite, because the residual stream is additive. Every scratch register is zeroed in the same layer.",
  },
};

export function layerDoc(name: string) {
  if (LAYER_DOC[name]) return LAYER_DOC[name];
  const m = name.match(/^mlp:min_fold\[(\d+)\]$/);
  if (m) {
    return {
      title: `Min fold ${m[1]}`,
      detail: `SCRATCH_MIN ← min(SCRATCH_MIN, SCRATCH_GATHER[${m[1]}]) via the identity min(a,b) = a − ReLU(a − b). One layer per in-edge slot; the fold is exact, not soft.`,
    };
  }
  return { title: name, detail: "" };
}

export const KIND_COLOR = {
  sink: "#35485a",
  node: "#f0b429",
  edge: "#3ddad7",
} as const;

/** Diverging value ramp: cyan for negatives, amber for positives, magenta at INF_B. */
export function valueColor(v: number, infB: number): string {
  if (v === 0) return "rgba(20, 30, 40, 0.55)";
  if (Math.abs(v - infB) < 0.5) return "#ff5c8a";
  if (v > infB) return "#ff9ec0";
  const t = Math.min(1, Math.log1p(Math.abs(v)) / Math.log1p(infB));
  if (v > 0) {
    // ink -> amber
    const r = Math.round(40 + t * 200);
    const g = Math.round(50 + t * 130);
    const b = Math.round(60 - t * 19);
    return `rgb(${r},${g},${b})`;
  }
  const r = Math.round(35 - t * 20);
  const g = Math.round(55 + t * 163);
  const b = Math.round(65 + t * 150);
  return `rgb(${r},${g},${b})`;
}

export function attnColor(a: number): string {
  if (a < 1e-9) return "rgba(16, 24, 33, 0.6)";
  const t = Math.pow(a, 0.45);
  return `rgba(240, 180, 41, ${0.12 + t * 0.88})`;
}

export function weightColor(w: number, scale: number): string {
  if (w === 0) return "rgba(16, 24, 33, 0.5)";
  const t = Math.min(1, Math.pow(Math.abs(w) / scale, 0.4));
  return w > 0
    ? `rgba(240, 180, 41, ${0.15 + t * 0.85})`
    : `rgba(61, 218, 215, ${0.15 + t * 0.85})`;
}

export function fmt(v: number | null, infB: number): string {
  if (v === null) return "∞";
  if (Math.abs(v - infB) < 0.5) return "∞";
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(2);
}
