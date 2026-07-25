import type { RegisterSpec } from "../types";
import { GROUP_COLOR, layerDoc } from "../lib/theme";

/** Which registers each sublayer reads and writes — the round's dataflow. */
const WIRING: Record<string, { reads: string[]; writes: string[] }> = {
  input: { reads: [], writes: [] },
  "attn:edge_reads_source": { reads: ["EDGE_FROM", "NODE_ID", "DIST", "SINK"], writes: ["SCRATCH_SRC_DIST"] },
  "mlp:candidate": { reads: ["SCRATCH_SRC_DIST", "EDGE_W"], writes: ["CAND", "SCRATCH_SRC_DIST"] },
  "mlp:clamp_inf": { reads: ["CAND"], writes: ["CAND"] },
  "attn:node_gathers": { reads: ["NODE_ID", "EDGE_TO", "SLOT", "CAND", "DIST"], writes: ["SCRATCH_GATHER"] },
  "mlp:min_init": { reads: ["DIST"], writes: ["SCRATCH_MIN"] },
  "mlp:writeback": { reads: ["DIST", "SCRATCH_MIN"], writes: ["DIST", "SCRATCH_MIN", "CAND", "SCRATCH_GATHER"] },
};

export function wiringFor(name: string) {
  if (WIRING[name]) return WIRING[name];
  if (name.startsWith("mlp:min_fold")) {
    return { reads: ["SCRATCH_MIN", "SCRATCH_GATHER"], writes: ["SCRATCH_MIN"] };
  }
  return { reads: [], writes: [] };
}

export function CircuitDiagram({
  layerNames, active, onSelect, registers,
}: {
  layerNames: string[];
  active: string;
  onSelect: (i: number) => void;
  registers: RegisterSpec[];
}) {
  const names = ["input", ...layerNames];
  const regByName = new Map(registers.map((r) => [r.name, r]));

  return (
    <div className="space-y-3">
      <div className="flex gap-1 overflow-x-auto no-scrollbar pb-1">
        {names.map((n, i) => {
          const isActive = n === active;
          const kind = n.startsWith("attn") ? "attn" : n === "input" ? "in" : "mlp";
          const color = kind === "attn" ? "#3ddad7" : kind === "in" ? "#5d7183" : "#9d7cff";
          return (
            <button
              key={n + i}
              onClick={() => onSelect(i)}
              title={layerDoc(n).title}
              className="group relative shrink-0 border px-2 py-2 text-left transition-colors"
              style={{
                minWidth: 62,
                borderColor: isActive ? color : "#1a2531",
                background: isActive ? `${color}1a` : "transparent",
              }}
            >
              <div className="text-[9px] tracking-wider uppercase"
                   style={{ color: isActive ? color : "#5d7183" }}>
                {kind === "in" ? "in" : kind}
              </div>
              <div className="mt-0.5 text-[10px] leading-tight"
                   style={{ color: isActive ? "#c9d8e4" : "#5d7183" }}>
                {n === "input"
                  ? "encode"
                  : n.replace(/^(attn|mlp):/, "").replace(/[[\]]/g, " ").trim()}
              </div>
              <div className="absolute inset-x-0 bottom-0 h-[2px]"
                   style={{ background: isActive ? color : "transparent" }} />
            </button>
          );
        })}
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {(["reads", "writes"] as const).map((dir) => {
          const list = wiringFor(active)[dir];
          return (
            <div key={dir} className="border border-ink-600 bg-ink-900 p-3">
              <div className="tick-label mb-2">
                {dir === "reads" ? "reads from" : "writes into (additively)"}
              </div>
              <div className="flex flex-wrap gap-1.5">
                {list.length === 0 && (
                  <span className="text-[11px] text-slatedim">—</span>
                )}
                {list.map((r) => {
                  const spec = regByName.get(r);
                  const c = spec ? GROUP_COLOR[spec.group] : "#5d7183";
                  return (
                    <span key={r}
                          className="border px-1.5 py-0.5 text-[10px] tracking-wide"
                          style={{ borderColor: `${c}66`, color: c, background: `${c}0f` }}>
                      {r}
                    </span>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

      <div className="border-l-2 pl-3" style={{ borderColor: "#f0b429" }}>
        <div className="text-[12px] text-slatebright">{layerDoc(active).title}</div>
        <p className="mt-1 max-w-[76ch] text-[12px] leading-relaxed text-slatemid">
          {layerDoc(active).detail}
        </p>
      </div>
    </div>
  );
}
