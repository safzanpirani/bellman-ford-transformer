import { useMemo, useState } from "react";
import type { LayerWeights, RegisterSpec } from "../types";
import { GROUP_COLOR, weightColor } from "../lib/theme";

/**
 * Every number in the model. Because the block is weight-tied, these matrices
 * ARE the whole transformer — there is nothing else to look at.
 */
export function WeightInspector({
  weights, registers, dimLabel,
}: {
  weights: LayerWeights[]; registers: RegisterSpec[]; dimLabel: (d: number) => string;
}) {
  const [li, setLi] = useState(0);
  const [head, setHead] = useState(0);
  const layer = weights[li];

  const mats = useMemo(() => {
    if (layer.kind === "attention") {
      const h = layer.heads[Math.min(head, layer.heads.length - 1)];
      return [
        { name: "W_Q", m: h.wq, note: "d_model × d_k — builds the query" },
        { name: "W_K", m: h.wk, note: "d_model × d_k — builds the key" },
        { name: "W_V", m: h.wv, note: "d_v × d_model (transposed) — what gets copied" },
        { name: "W_O", m: h.wo, note: "d_v × d_model — where it lands" },
      ];
    }
    return [
      { name: "W_1", m: layer.w1, note: `d_model × ${layer.hidden} hidden ReLUs` },
      { name: "b_1", m: [layer.b1], note: "hidden bias — the thresholds" },
      { name: "W_2", m: layer.w2, note: `${layer.hidden} × d_model — the residual delta` },
    ];
  }, [layer, head]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-1">
        {weights.map((w, i) => (
          <button key={i} onClick={() => { setLi(i); setHead(0); }}
                  className="border px-2 py-1 text-[10px] transition-colors"
                  style={{
                    borderColor: i === li ? (w.kind === "attention" ? "#3ddad7" : "#9d7cff") : "#1a2531",
                    color: i === li ? "#c9d8e4" : "#5d7183",
                    background: i === li ? (w.kind === "attention" ? "#3ddad71a" : "#9d7cff1a") : "transparent",
                  }}>
            {w.name.replace(/^(attn|mlp):/, "")}
          </button>
        ))}
      </div>

      {layer.kind === "attention" && layer.heads.length > 1 && (
        <div className="flex items-center gap-2">
          <span className="tick-label">head</span>
          {layer.heads.map((_, h) => (
            <button key={h} onClick={() => setHead(h)}
                    className="h-6 w-6 border text-[10px] tabular-nums transition-colors"
                    style={{
                      borderColor: h === head ? "#3ddad7" : "#1a2531",
                      color: h === head ? "#3ddad7" : "#5d7183",
                      background: h === head ? "#3ddad71a" : "transparent",
                    }}>
              {h}
            </button>
          ))}
          <span className="text-[11px] text-slatedim">
            gathers in-edge slot {head}
          </span>
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        {mats.map((mm) => (
          <MatrixPlot key={mm.name} {...mm} registers={registers} dimLabel={dimLabel} />
        ))}
      </div>

      <div className="flex items-center gap-4 text-[10.5px] text-slatedim">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3" style={{ background: "rgba(240,180,41,.9)" }} /> positive
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3" style={{ background: "rgba(61,218,215,.9)" }} /> negative
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 border border-ink-600" style={{ background: "rgba(16,24,33,.5)" }} /> exactly zero
        </span>
      </div>
    </div>
  );
}

function MatrixPlot({
  name, m, note, registers, dimLabel,
}: {
  name: string; m: number[][]; note: string;
  registers: RegisterSpec[]; dimLabel: (d: number) => string;
}) {
  const [hover, setHover] = useState<{ r: number; c: number } | null>(null);
  const rows = m.length;
  const cols = m[0]?.length ?? 0;
  const scale = Math.max(...m.flat().map(Math.abs), 1e-9);
  // width and height scale independently so a 55×9 strip stays legible
  const cellW = Math.max(2, Math.min(24, Math.floor(430 / cols)));
  const cellH = Math.max(2, Math.min(24, Math.floor(300 / rows)));
  const isDModelRows = rows === registers[registers.length - 1].stop;

  return (
    <div className="border border-ink-600 bg-ink-900 p-3">
      <div className="mb-2 flex items-baseline gap-2">
        <span className="text-[12px] text-amber">{name}</span>
        <span className="tick-label">{rows}×{cols}</span>
        <span className="ml-auto text-[10px] text-slatedim">max |w| {scale}</span>
      </div>
      <div className="overflow-auto no-scrollbar">
        <svg width={cols * cellW + (isDModelRows ? 6 : 0) + 2} height={rows * cellH + 2}>
          {isDModelRows &&
            registers.map((r) => (
              <rect key={r.name} x={0} y={r.start * cellH} width={4}
                    height={r.size * cellH - 1} fill={GROUP_COLOR[r.group]} opacity={0.85}>
                <title>{r.name}</title>
              </rect>
            ))}
          <g transform={`translate(${isDModelRows ? 6 : 0} 0)`}>
            {m.map((row, ri) =>
              row.map((v, ci) => (
                <rect key={`${ri}-${ci}`} x={ci * cellW} y={ri * cellH}
                      width={Math.max(1, cellW - 0.6)} height={Math.max(1, cellH - 0.6)}
                      fill={weightColor(v, scale)}
                      onMouseEnter={() => setHover({ r: ri, c: ci })}
                      onMouseLeave={() => setHover(null)} />
              ))
            )}
          </g>
        </svg>
      </div>
      <div className="mt-2 h-8 text-[10.5px] leading-tight text-slatedim">
        {hover ? (
          <>
            <span className="text-slatebright">
              {isDModelRows ? dimLabel(hover.r) : `row ${hover.r}`}
            </span>
            {" → col "}{hover.c}{" = "}
            <span className="text-amber">{m[hover.r][hover.c]}</span>
          </>
        ) : (
          note
        )}
      </div>
    </div>
  );
}
