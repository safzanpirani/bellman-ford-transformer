import { useEffect, useMemo, useRef, useState } from "react";
import type { Case, RegisterSpec, Trace } from "../types";

export function useTraceData() {
  const [trace, setTrace] = useState<Trace | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}trace.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`trace.json → HTTP ${r.status}`);
        return r.json();
      })
      .then(setTrace)
      .catch((e) => setError(String(e)));
  }, []);
  return { trace, error };
}

/** Register lookup helpers derived once from the exported map. */
export function useRegisters(registers: RegisterSpec[]) {
  return useMemo(() => {
    const byName = new Map(registers.map((r) => [r.name, r]));
    const perDim: RegisterSpec[] = [];
    for (const r of registers) {
      for (let i = r.start; i < r.stop; i++) perDim[i] = r;
    }
    const dimLabel = (d: number) => {
      const r = perDim[d];
      if (!r) return `dim ${d}`;
      return r.size === 1 ? r.name : `${r.name}[${d - r.start}]`;
    };
    return { byName, perDim, dimLabel, col: (n: string) => byName.get(n)!.start };
  }, [registers]);
}

export interface Transport {
  round: number;          // 0 = initial state, 1..R after each round
  step: number;           // index into rounds[round-1].steps
  playing: boolean;
  setRound: (r: number) => void;
  setStep: (s: number) => void;
  setPlaying: (p: boolean) => void;
  stepForward: () => void;
  stepBack: () => void;
  totalSteps: number;
  /** the residual-stream state currently selected */
  state: number[][];
  stepName: string;
  attn: number[][][] | null;
}

/**
 * One shared playhead over (round, sublayer) that every panel reads.
 * Round 0 is the encoder output; rounds 1..R each expose their sublayers.
 */
export function useTransport(kase: Case, speedMs = 620): Transport {
  const [round, setRound] = useState(0);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timer = useRef<number | null>(null);

  const rounds = kase.rounds;
  const totalSteps = round === 0 ? 1 : rounds[round - 1].steps.length;

  // reset when the graph changes
  useEffect(() => {
    setRound(0);
    setStep(0);
    setPlaying(false);
  }, [kase.id]);

  const stepForward = () => {
    if (round === 0) {
      setRound(1);
      setStep(0);
      return;
    }
    if (step + 1 < rounds[round - 1].steps.length) setStep(step + 1);
    else if (round < rounds.length) {
      setRound(round + 1);
      setStep(0);
    } else setPlaying(false);
  };

  const stepBack = () => {
    if (round === 0) return;
    if (step > 0) setStep(step - 1);
    else if (round > 1) {
      setRound(round - 1);
      setStep(rounds[round - 2].steps.length - 1);
    } else {
      setRound(0);
      setStep(0);
    }
  };

  useEffect(() => {
    if (!playing) {
      if (timer.current) window.clearTimeout(timer.current);
      return;
    }
    timer.current = window.setTimeout(stepForward, speedMs);
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
    };
  });

  const cur =
    round === 0
      ? { name: "encoded input", state: kase.initial, attn: null }
      : rounds[round - 1].steps[Math.min(step, rounds[round - 1].steps.length - 1)];

  return {
    round, step, playing, setRound, setStep, setPlaying,
    stepForward, stepBack, totalSteps,
    state: cur.state, stepName: cur.name, attn: cur.attn,
  };
}

/** Distances currently held in the DIST register, per node. */
export function currentDistances(kase: Case, state: number[][], distCol: number) {
  const out: number[] = [];
  kase.tokens.forEach((t, i) => {
    if (t.kind === "node") out[t.node!] = state[i][distCol];
  });
  return out;
}
