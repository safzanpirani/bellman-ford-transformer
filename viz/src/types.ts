export type Group = "struct" | "node" | "edge" | "scratch" | "flag";

export interface RegisterSpec {
  name: string;
  start: number;
  stop: number;
  size: number;
  group: Group;
}

export interface TokenSpec {
  row: number;
  kind: "sink" | "node" | "edge";
  label: string;
  node?: number;
  isSource?: boolean;
  u?: number;
  v?: number;
  w?: number;
  edgeIndex?: number;
}

export interface Step {
  name: string;
  state: number[][];
  attn: number[][][] | null;
}

export interface RoundTrace {
  round: number;
  steps: Step[];
  reference: (number | null)[];
}

export interface Case {
  id: string;
  name: string;
  blurb: string;
  n: number;
  source: number;
  edges: { u: number; v: number; w: number }[];
  layout: { x: number; y: number }[];
  tokens: TokenSpec[];
  initial: number[][];
  rounds: RoundTrace[];
  referenceInit: (number | null)[];
  truth: (number | null)[];
  negativeCycle: boolean;
  flags: number[];
  flagged: boolean;
}

export interface AttentionLayerWeights {
  name: string;
  kind: "attention";
  dk: number;
  heads: { wq: number[][]; wk: number[][]; wv: number[][]; wo: number[][] }[];
}

export interface MlpLayerWeights {
  name: string;
  kind: "mlp";
  w1: number[][];
  b1: number[];
  w2: number[][];
  hidden: number;
}

export type LayerWeights = AttentionLayerWeights | MlpLayerWeights;

export interface Trace {
  config: {
    nMax: number; wMax: number; dMax: number; infB: number;
    cHard: number; cMatch: number; rounds: number; seqLen: number;
    eMax: number; dModel: number; kPush: number;
  };
  stats: { params: number; trainedParams: number; sublayers: number; layerNames: string[] };
  registers: RegisterSpec[];
  weights: LayerWeights[];
  logitEvidence: {
    labels: string[];
    logits: number[][];
    cMatch: number;
    underflowThreshold: number;
  };
  cases: Case[];
  softSweep: {
    maxRounds: number;
    graphs: number;
    series: { label: string; cSoft: number | null; errors: number[] }[];
  };
}
