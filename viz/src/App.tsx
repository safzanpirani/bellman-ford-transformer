import { useMemo, useState } from "react";
import { useRegisters, useTraceData, useTransport, currentDistances } from "./lib/useTrace";
import { Panel, Stat, Note, Mono } from "./components/Chrome";
import { TransportBar } from "./components/Transport";
import { GraphCanvas } from "./components/GraphCanvas";
import { ResidualStream } from "./components/ResidualStream";
import { AttentionMatrix, LogitEvidence } from "./components/AttentionView";
import { CircuitDiagram } from "./components/CircuitDiagram";
import { MinGadget } from "./components/MinGadget";
import { WeightInspector } from "./components/WeightInspector";
import { SoftVsExact, ConvergencePlot } from "./components/Charts";
import { RegisterRibbon } from "./components/RegisterRibbon";
import { fmt } from "./lib/theme";
import type { Case } from "./types";

export default function App() {
  const { trace, error } = useTraceData();
  const [caseId, setCaseId] = useState("chain");
  const [highlight, setHighlight] = useState<string | null>(null);
  const [hoverNode, setHoverNode] = useState<number | null>(null);

  if (error) return <Fatal msg={error} />;
  if (!trace) return <Booting />;

  const kase = trace.cases.find((c) => c.id === caseId) ?? trace.cases[0];
  return (
    <Main
      key={kase.id}
      trace={trace}
      kase={kase}
      caseId={caseId}
      setCaseId={setCaseId}
      highlight={highlight}
      setHighlight={setHighlight}
      hoverNode={hoverNode}
      setHoverNode={setHoverNode}
    />
  );
}

function Main({
  trace, kase, caseId, setCaseId, highlight, setHighlight, hoverNode, setHoverNode,
}: {
  trace: NonNullable<ReturnType<typeof useTraceData>["trace"]>;
  kase: Case;
  caseId: string;
  setCaseId: (s: string) => void;
  highlight: string | null;
  setHighlight: (s: string | null) => void;
  hoverNode: number | null;
  setHoverNode: (n: number | null) => void;
}) {
  const t = useTransport(kase);
  const regs = useRegisters(trace.registers);
  const distCol = regs.col("DIST");
  const candCol = regs.col("CAND");
  const cfg = trace.config;

  const dist = currentDistances(kase, t.state, distCol);

  // state one sublayer earlier, for the delta outlines
  const prevState = useMemo(() => {
    if (t.round === 0) return null;
    const steps = kase.rounds[t.round - 1].steps;
    if (t.step > 0) return steps[t.step - 1].state;
    return t.round > 1
      ? kase.rounds[t.round - 2].steps.at(-1)!.state
      : kase.initial;
  }, [kase, t.round, t.step]);

  const prevDist = prevState ? currentDistances(kase, prevState, distCol) : null;

  // which in-edges are currently supplying the winning candidate
  const activeEdges = useMemo(() => {
    const s = new Set<number>();
    if (t.round === 0) return s;
    const best = new Map<number, number>();
    kase.tokens.forEach((tok, i) => {
      if (tok.kind !== "edge") return;
      const c = t.state[i][candCol];
      if (c === 0) return;
      const cur = best.get(tok.v!);
      if (cur === undefined || c < t.state[cur][candCol]) best.set(tok.v!, i);
    });
    best.forEach((i, node) => {
      const c = t.state[i][candCol];
      if (c < (dist[node] ?? cfg.infB) + 1e-9 && c < cfg.infB - 0.5) {
        s.add(kase.tokens[i].edgeIndex!);
      }
    });
    return s;
  }, [kase, t.state, candCol, dist, cfg.infB, t.round]);

  // per-round DIST history for the convergence plot
  const distsByRound = useMemo(() => {
    const out = [currentDistances(kase, kase.initial, distCol)];
    for (const r of kase.rounds) {
      out.push(currentDistances(kase, r.steps.at(-1)!.state, distCol));
    }
    return out;
  }, [kase, distCol]);

  // DIST is only touched by the write-back, which is the last sublayer of the
  // round. Until then the register still holds the previous round's values, so
  // that is what we must compare against.
  const refNow = (() => {
    if (t.round === 0) return kase.referenceInit;
    const atEnd = t.step === t.totalSteps - 1;
    if (atEnd) return kase.rounds[t.round - 1].reference;
    return t.round === 1 ? kase.referenceInit : kase.rounds[t.round - 2].reference;
  })();
  const agrees = refNow.every((v, i) => {
    const got = dist[i];
    const isInf = Math.abs(got - cfg.infB) < 0.5 || got > cfg.infB;
    return v === null ? isInf : Math.abs(got - v) < 1e-4;
  });

  const attnStepIdx = t.attn ? t.step : null;

  return (
    <div className="min-h-screen">
      <Hero trace={trace} />
      <TransportBar t={t} kase={kase} />

      <main className="mx-auto max-w-[1400px] space-y-4 px-4 py-6 sm:px-6">
        <CasePicker cases={trace.cases} caseId={caseId} onPick={setCaseId} kase={kase} />

        {/* ---- the live execution row ---- */}
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
          <Panel
            title="The graph, as the model sees it"
            index="01"
            kicker={`source n${kase.source} · ${kase.n} nodes · ${kase.edges.length} edges`}
            right={
              <span
                className="whitespace-nowrap border px-1.5 py-0.5 text-[10px] tracking-wider uppercase"
                style={
                  agrees
                    ? { borderColor: "#9ee493", color: "#9ee493" }
                    : { borderColor: "#ff5c8a", color: "#ff5c8a" }
                }
              >
                {agrees ? "matches reference" : "diverged"}
              </span>
            }
          >
            <GraphCanvas
              kase={kase}
              trace={trace}
              dist={dist}
              prevDist={prevDist}
              activeEdges={activeEdges}
              hoverNode={hoverNode}
              onHoverNode={setHoverNode}
            />
            <div className="mt-2 grid grid-cols-2 gap-2 border-t border-ink-700 pt-3 sm:grid-cols-4">
              {kase.truth.map((v, i) => (
                <div key={i} className="flex items-baseline justify-between border-l border-ink-600 pl-2">
                  <span className="text-[10px] text-slatedim">n{i} true</span>
                  <span className="text-[12px] text-lime">
                    {kase.negativeCycle ? "−∞" : fmt(v, cfg.infB)}
                  </span>
                </div>
              ))}
            </div>
            <div className="mt-3">
              <Note>
                Cyan edges are the ones whose candidate currently wins the min at
                their destination. Node values are read straight out of the{" "}
                <Mono>DIST</Mono> register at the selected sublayer — nothing is
                recomputed in the browser.
              </Note>
            </div>
          </Panel>

          <Panel
            title="Residual stream"
            index="02"
            kicker={`${kase.tokens.length} live tokens × ${cfg.dModel} dims`}
          >
            <ResidualStream
              kase={kase}
              trace={trace}
              state={t.state}
              prevState={prevState}
              registers={trace.registers}
              dimLabel={regs.dimLabel}
              highlight={highlight}
              onHighlight={setHighlight}
            />
          </Panel>
        </div>

        {/* ---- circuit + attention ---- */}
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <Panel title="One round, sublayer by sublayer" index="03"
                 kicker={`${trace.stats.sublayers} residual sublayers, weight-tied across all ${cfg.rounds} rounds`}>
            <CircuitDiagram
              layerNames={trace.stats.layerNames}
              active={t.stepName}
              onSelect={(i) => {
                t.setPlaying(false);
                if (i === 0 && t.round === 0) return;
                if (t.round === 0) t.setRound(1);
                t.setStep(i);
              }}
              registers={trace.registers}
            />
          </Panel>

          <Panel title="Attention patterns" index="04"
                 kicker={attnStepIdx !== null ? "live at this sublayer" : "step onto an attention sublayer"}>
            {t.attn ? (
              <div className="flex gap-5 overflow-x-auto no-scrollbar">
                {t.attn.slice(0, 4).map((p, h) => (
                  <AttentionMatrix
                    key={h}
                    kase={kase}
                    pattern={p}
                    title={t.attn!.length > 1 ? `head ${h}` : "single head"}
                    subtitle={t.attn!.length > 1 ? `slot ${h}` : "edge → source"}
                  />
                ))}
              </div>
            ) : (
              <div className="space-y-3">
                <Note>
                  The current sublayer is an MLP — it has no attention pattern.
                  Step to <Mono>edge_reads_source</Mono> or{" "}
                  <Mono>node_gathers</Mono> to see routing.
                </Note>
                <button
                  onClick={() => {
                    t.setPlaying(false);
                    if (t.round === 0) t.setRound(1);
                    t.setStep(0);
                  }}
                  className="border border-cyan px-2.5 py-1 text-[11px] text-cyan transition-colors hover:bg-cyan/12"
                >
                  jump to the routing head →
                </button>
              </div>
            )}
          </Panel>
        </div>

        {/* ---- convergence ---- */}
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
          <Panel title="Convergence vs the reference algorithm" index="05"
                 kicker="FR-V, drawn">
            <ConvergencePlot kase={kase} trace={trace} distsByRound={distsByRound} round={t.round} />
            <div className="mt-3">
              <Note>
                Each line is one node's <Mono>DIST</Mono> after k rounds. The test
                suite asserts these values equal a reference synchronous
                Bellman-Ford after <em>every</em> round, for 100 random graphs —
                the induction proof, executable. Distances only ever move down:
                the write-back adds <Mono>−ReLU(dist − min)</Mono>, which is zero
                whenever nothing improved.
              </Note>
            </div>
          </Panel>

          <Panel title="Negative cycles" index="06"
                 kicker="one extra round + a comparison layer"
                 right={
                   <span className="whitespace-nowrap border px-1.5 py-0.5 text-[10px] tracking-wider uppercase"
                         style={kase.flagged
                           ? { borderColor: "#ff5c8a", color: "#ff5c8a" }
                           : { borderColor: "#5d7183", color: "#5d7183" }}>
                     {kase.flagged ? "flag hot" : "flag cold"}
                   </span>
                 }>
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-4">
                {kase.flags.map((f, i) => (
                  <div key={i} className="border border-ink-600 bg-ink-900 p-2">
                    <div className="tick-label">n{i} FLAG_DEC</div>
                    <div className="font-display text-2xl"
                         style={{ color: f > 1e-3 ? "#ff5c8a" : "#5d7183" }}>
                      {f > 1e-3 ? f.toFixed(1) : "0"}
                    </div>
                  </div>
                ))}
              </div>
              <Note>
                After the {cfg.rounds} standard rounds the model snapshots{" "}
                <Mono>DIST</Mono>, runs the same block once more, and writes{" "}
                <Mono>ReLU(prev − dist)</Mono> into a flag register. Any hot flag
                means some distance was still falling in round {cfg.rounds + 1} —
                a negative cycle reachable from the source. networkx agrees on all
                100 random negative-weight graphs in the suite
                {kase.negativeCycle
                  ? ", including this one, where it raises NetworkXUnbounded."
                  : "; this graph has none."}
              </Note>
            </div>
          </Panel>
        </div>

        {/* ---- mechanism explainers ---- */}
        <Panel title="Why the softmax is exactly one-hot, not approximately"
               index="07" kicker="pre-softmax logits, in units of C = C_HARD/4">
          <div className="grid gap-6 lg:grid-cols-[auto_minmax(0,1fr)]">
            <LogitEvidence
              labels={trace.logitEvidence.labels}
              logits={trace.logitEvidence.logits}
              cMatch={trace.logitEvidence.cMatch}
              underflow={trace.logitEvidence.underflowThreshold}
            />
            <div className="space-y-3 self-start">
              <Note>
                Every query's intended target beats every competitor by at least{" "}
                <Mono>C_HARD/4 = {cfg.cMatch.toLocaleString()}</Mono>. After
                softmax subtracts the row max, each off-target term is{" "}
                <Mono>exp(−2500)</Mono> — which is not merely small, it{" "}
                <em>underflows float32 to exactly 0.0</em>. So the attention
                weights are literally {"{1, 0, 0, …}"} and the value copy is
                bit-exact.
              </Note>
              <Note>
                This is the one soft component in the construction, and its error
                is bounded once rather than compounded: the values it routes are
                afterwards consumed by exact ReLU arithmetic. Tokens with nothing
                to read fall back to the <Mono>SINK</Mono> token and copy a clean
                zero, so no junk ever enters a live register.
              </Note>
            </div>
          </div>
        </Panel>

        <Panel title="The min gadget" index="08"
               kicker="min(a, b) = a − ReLU(a − b) — drag the operands">
          <MinGadget />
        </Panel>

        <Panel title="The register map" index="09"
               kicker="one source of truth for the d_model axis">
          <RegisterRibbon
            registers={trace.registers}
            highlight={highlight}
            onHighlight={setHighlight}
            dModel={cfg.dModel}
          />
        </Panel>

        <Panel title="Every weight in the model" index="10"
               kicker="the block is weight-tied — this is the entire network">
          <WeightInspector
            weights={trace.weights}
            registers={trace.registers}
            dimLabel={regs.dimLabel}
          />
        </Panel>

        <Panel title="Exact reduction vs attention soft-min" index="11"
               kicker="FR8 — the empirical payoff">
          <SoftVsExact trace={trace} />
        </Panel>

        <Footer trace={trace} />
      </main>
    </div>
  );
}

function Hero({ trace }: { trace: NonNullable<ReturnType<typeof useTraceData>["trace"]> }) {
  const c = trace.config;
  return (
    <header className="relative overflow-hidden border-b border-ink-700">
      <div className="bg-graticule absolute inset-0 opacity-45" />
      <div className="bg-grain absolute inset-0" />
      <div className="relative mx-auto max-w-[1400px] px-4 pt-14 pb-9 sm:px-6">
        <div className="rise flex items-center gap-3" style={{ animationDelay: "0ms" }}>
          <span className="h-px w-10 bg-amber" />
          <span className="tick-label">a transformer with no trained parameters</span>
        </div>
        <h1 className="rise mt-4 max-w-[19ch] font-display text-6xl leading-[0.94] text-slatebright sm:text-8xl"
            style={{ animationDelay: "70ms" }}>
          Bellman–Ford,
          <br />
          <span className="text-amber italic">by hand</span>
        </h1>
        <p className="rise mt-6 max-w-[62ch] text-[13.5px] leading-relaxed text-slatemid"
           style={{ animationDelay: "140ms" }}>
          Every weight in this network was written down, not learned. One
          transformer block is one relaxation round; attention routes each edge to
          its source node, and a chain of ReLUs takes the exact minimum. Stack the
          same block {c.rounds} times and the forward pass <em>is</em> the
          algorithm — matching networkx bit for bit on integer weights.
        </p>
        <div className="rise mt-9 grid grid-cols-2 gap-y-5 sm:grid-cols-3 lg:grid-cols-6"
             style={{ animationDelay: "210ms" }}>
          <Stat label="trained params" value={0} accent="cyan" />
          <Stat label="constructed weights" value={trace.stats.params.toLocaleString()} />
          <Stat label="d_model" value={c.dModel} accent="slate" />
          <Stat label="sublayers / round" value={trace.stats.sublayers} accent="slate" />
          <Stat label="rounds" value={c.rounds} accent="slate" />
          <Stat label="max error" value="0" unit="exact" accent="cyan" />
        </div>
      </div>
    </header>
  );
}

function CasePicker({
  cases, caseId, onPick, kase,
}: {
  cases: Case[]; caseId: string; onPick: (s: string) => void; kase: Case;
}) {
  return (
    <div className="panel bg-grain px-4 py-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="tick-label mr-1">graph</span>
        {cases.map((c) => (
          <button key={c.id} onClick={() => onPick(c.id)}
                  className="border px-2.5 py-1 text-[11px] transition-colors"
                  style={{
                    borderColor: c.id === caseId ? "#f0b429" : "#1a2531",
                    color: c.id === caseId ? "#f0b429" : "#5d7183",
                    background: c.id === caseId ? "#f0b4291a" : "transparent",
                  }}>
            {c.name}
          </button>
        ))}
      </div>
      <p className="mt-2 max-w-[86ch] text-[12px] leading-relaxed text-slatemid">
        {kase.blurb}
      </p>
    </div>
  );
}

function Footer({ trace }: { trace: NonNullable<ReturnType<typeof useTraceData>["trace"]> }) {
  const c = trace.config;
  return (
    <footer className="panel bg-grain px-5 py-6">
      <div className="grid gap-6 md:grid-cols-3">
        <div>
          <div className="tick-label mb-2">construction limits</div>
          <ul className="space-y-1 text-[11.5px] text-slatemid">
            <li>N_MAX = {c.nMax} nodes</li>
            <li>W_MAX = {c.wMax} max |weight|</li>
            <li>D_MAX = {c.dMax} in-degree ceiling</li>
            <li>INF_B = {c.infB} bounded infinity</li>
          </ul>
        </div>
        <div>
          <div className="tick-label mb-2">what is not here</div>
          <ul className="space-y-1 text-[11.5px] text-slatemid">
            <li>no LayerNorm — it would rescale real distances</li>
            <li>no dropout, no positional embeddings</li>
            <li>no gradients: every tensor requires_grad=False</li>
            <li>no tokenizer — inputs are structured tensors</li>
          </ul>
        </div>
        <div>
          <div className="tick-label mb-2">verification</div>
          <ul className="space-y-1 text-[11.5px] text-slatemid">
            <li>1,000-graph property suite, bit-exact</li>
            <li>round-by-round induction check (FR-V)</li>
            <li>fixed-point test on converged states</li>
            <li>raw-index lint over the whole package</li>
          </ul>
        </div>
      </div>
      <p className="mt-6 border-t border-ink-700 pt-4 text-[11px] text-slatedim">
        Every number rendered on this page was exported from a real forward pass
        of the model — residual streams, attention patterns and weight matrices
        alike. Nothing is illustrative.
      </p>
    </footer>
  );
}

function Booting() {
  return (
    <div className="flex min-h-screen items-center justify-center">
      <div className="space-y-2 text-center">
        <div className="font-display text-3xl text-amber">loading trace</div>
        <div className="tick-label">reading forward-pass export</div>
      </div>
    </div>
  );
}

function Fatal({ msg }: { msg: string }) {
  return (
    <div className="flex min-h-screen items-center justify-center p-6">
      <div className="panel max-w-[52ch] p-6">
        <div className="font-display text-2xl text-magenta">trace unavailable</div>
        <p className="mt-2 text-[12px] text-slatemid">{msg}</p>
        <p className="mt-3 text-[12px] text-slatedim">
          Regenerate it with{" "}
          <code className="text-cyan">
            python -m experiments.export_traces --out viz/public/trace.json
          </code>
        </p>
      </div>
    </div>
  );
}
