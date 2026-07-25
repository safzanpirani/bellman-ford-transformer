import type { RegisterSpec } from "../types";
import { GROUP_COLOR, GROUP_LABEL } from "../lib/theme";

const REG_DOC: Record<string, string> = {
  ONES: "Constant 1 on every real token; 0 on padding. Supplies bias-like terms inside attention logits.",
  SINK: "Marks token 0. Any query with nothing to read lands here and copies an all-zero payload — that is what keeps junk out of live registers.",
  IS_NODE: "Token-type flag for node tokens.",
  IS_EDGE: "Token-type flag for edge tokens; also acts as a baseline penalty in the gather heads.",
  NODE_ID: "One-hot node identity. The key side of the edge-reads-source match.",
  DIST: "The distance estimate itself. The only register the decoder reads, and the only one written across rounds.",
  EDGE_FROM: "One-hot source node of an edge. The query side of the edge-reads-source match.",
  EDGE_TO: "One-hot destination node. The query/key match that decides which node gathers this candidate.",
  EDGE_W: "Scalar edge weight, added to the source distance to form the candidate.",
  SLOT: "One-hot in-edge slot index, assigned per destination at encode time so each gather head sees at most one match.",
  CAND: "dist[u] + w for this edge. Written this round, cleared before the next.",
  SCRATCH_SRC_DIST: "Landing pad for dist[u] copied off the source node token.",
  SCRATCH_GATHER: "One dim per in-edge slot: the candidate that head k routed to this node.",
  SCRATCH_MIN: "Running minimum accumulator for the ReLU fold.",
  SCRATCH_PREV_DIST: "Snapshot of DIST taken before the extra negative-cycle round.",
  FLAG_DEC: "Hot iff this node's distance fell during the detection round — a reachable negative cycle.",
  SCRATCH_SPARE: "Unused spare dims, kept for gadget unit tests to scribble in.",
};

export function RegisterRibbon({
  registers, highlight, onHighlight, dModel,
}: {
  registers: RegisterSpec[];
  highlight: string | null;
  onHighlight: (r: string | null) => void;
  dModel: number;
}) {
  const sel = registers.find((r) => r.name === highlight);
  return (
    <div className="space-y-3">
      <div className="flex h-14 w-full overflow-hidden border border-ink-600">
        {registers.map((r) => {
          const active = highlight === r.name;
          return (
            <button
              key={r.name}
              onMouseEnter={() => onHighlight(r.name)}
              onFocus={() => onHighlight(r.name)}
              onClick={() => onHighlight(active ? null : r.name)}
              className="group relative flex min-w-0 items-end overflow-hidden transition-all"
              style={{
                flexGrow: r.size,
                flexBasis: 0,
                background: active ? `${GROUP_COLOR[r.group]}2e` : `${GROUP_COLOR[r.group]}12`,
                borderRight: "1px solid #0a0e14",
              }}
            >
              <span className="absolute inset-x-0 top-0 h-[3px]"
                    style={{ background: GROUP_COLOR[r.group], opacity: active ? 1 : 0.55 }} />
              <span className="w-full truncate px-1 pb-1 text-left text-[9px] leading-tight"
                    style={{ color: active ? GROUP_COLOR[r.group] : "#5d7183" }}>
                {r.size > 2 ? r.name : ""}
              </span>
            </button>
          );
        })}
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
        {(Object.keys(GROUP_LABEL) as (keyof typeof GROUP_LABEL)[]).map((g) => (
          <span key={g} className="flex items-center gap-1.5 text-[10.5px] text-slatedim">
            <span className="inline-block h-2.5 w-2.5" style={{ background: GROUP_COLOR[g] }} />
            {GROUP_LABEL[g]}
          </span>
        ))}
        <span className="ml-auto text-[10.5px] text-slatedim">
          d_model = <span className="text-amber">{dModel}</span>, derived from the map
        </span>
      </div>

      <div className="min-h-[76px] border-l-2 pl-3"
           style={{ borderColor: sel ? GROUP_COLOR[sel.group] : "#24313f" }}>
        {sel ? (
          <>
            <div className="flex flex-wrap items-baseline gap-2">
              <span className="text-[13px]" style={{ color: GROUP_COLOR[sel.group] }}>
                {sel.name}
              </span>
              <span className="tick-label">
                dims {sel.start}–{sel.stop - 1} · width {sel.size}
              </span>
            </div>
            <p className="mt-1 max-w-[78ch] text-[12px] leading-relaxed text-slatemid">
              {REG_DOC[sel.name]}
            </p>
          </>
        ) : (
          <p className="max-w-[78ch] text-[12px] leading-relaxed text-slatemid">
            Hover a band. Every weight-construction function addresses the stream
            through these names — a lint test in the suite fails the build if any
            raw integer index appears. Changing N_MAX resizes the map and nothing
            else.
          </p>
        )}
      </div>
    </div>
  );
}
