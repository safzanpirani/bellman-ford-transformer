import { useMemo, useState } from "react";

/**
 * Interactive proof-by-play of min(a,b) = a - ReLU(a - b), the exact reducer.
 * Drag the operands; the ReLU curve, the algebra, and the running fold all
 * update together.
 */
export function MinGadget() {
  const [a, setA] = useState(7);
  const [b, setB] = useState(3);
  const relu = Math.max(0, a - b);
  const out = a - relu;

  const W = 420, H = 210, PAD = 34;
  const dom = 12;
  const sx = (v: number) => PAD + ((v + dom) / (2 * dom)) * (W - PAD - 12);
  const sy = (v: number) => H - PAD - ((v + 2) / (dom + 4)) * (H - PAD - 14);

  const curve = useMemo(() => {
    const pts: string[] = [];
    for (let x = -dom; x <= dom; x += 0.5) pts.push(`${sx(x)},${sy(Math.max(0, x))}`);
    return pts.join(" ");
  }, []);

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
      <div>
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
          <line x1={PAD} y1={sy(0)} x2={W - 12} y2={sy(0)} stroke="#24313f" />
          <line x1={sx(0)} y1={14} x2={sx(0)} y2={H - PAD} stroke="#24313f" />
          <polyline points={curve} fill="none" stroke="#3ddad7" strokeWidth={2} />
          <text x={W - 14} y={sy(0) + 14} fontSize={9} fill="#5d7183" textAnchor="end">
            a − b
          </text>
          <text x={sx(0) + 6} y={20} fontSize={9} fill="#5d7183">ReLU(a − b)</text>

          <line x1={sx(a - b)} y1={sy(0)} x2={sx(a - b)} y2={sy(relu)}
                stroke="#f0b429" strokeDasharray="3 3" />
          <line x1={sx(0)} y1={sy(relu)} x2={sx(a - b)} y2={sy(relu)}
                stroke="#f0b429" strokeDasharray="3 3" />
          <circle cx={sx(a - b)} cy={sy(relu)} r={5} fill="#f0b429" className="glow-amber" />
          <text x={sx(a - b) + 8} y={sy(relu) - 8} fontSize={10} fill="#f0b429">
            {relu.toFixed(1)}
          </text>
        </svg>

        <div className="mt-3 space-y-3">
          {([["a", a, setA], ["b", b, setB]] as const).map(([name, val, set]) => (
            <label key={name} className="flex items-center gap-3 text-[11px]">
              <span className="w-4 text-amber">{name}</span>
              <input
                type="range" min={-10} max={10} step={0.5} value={val}
                onChange={(e) => set(Number(e.target.value))}
                className="h-1 flex-1 cursor-pointer appearance-none bg-ink-600 accent-amber"
              />
              <span className="w-10 text-right tabular-nums text-slatebright">{val}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="space-y-2.5 self-start border border-ink-600 bg-ink-900 p-4">
        <div className="tick-label">one hidden unit, one output weight</div>
        <div className="space-y-1.5 text-[12.5px] leading-relaxed">
          <Row label="a" value={a} color="#f0b429" />
          <Row label="b" value={b} color="#f0b429" />
          <div className="my-1 border-t border-ink-700" />
          <Row label="a − b" value={a - b} color="#8ea3b5" />
          <Row label="ReLU(a − b)" value={relu} color="#3ddad7" />
          <div className="my-1 border-t border-ink-700" />
          <Row label="a − ReLU(a − b)" value={out} color="#9ee493" big />
          <Row label="min(a, b)" value={Math.min(a, b)} color="#9ee493" big />
        </div>
        <div
          className={`mt-2 border px-2 py-1.5 text-[11px] ${
            out === Math.min(a, b)
              ? "border-lime/40 bg-lime/5 text-lime"
              : "border-magenta/40 bg-magenta/5 text-magenta"
          }`}
        >
          {out === Math.min(a, b) ? "✓ exact — bit for bit" : "mismatch"}
        </div>
        <p className="pt-1 text-[11.5px] leading-relaxed text-slatemid">
          The block chains D_MAX of these, folding one gathered candidate per
          layer. No approximation enters: the only nonlinearity is the ReLU
          that the identity is built from.
        </p>
      </div>
    </div>
  );
}

function Row({ label, value, color, big = false }: {
  label: string; value: number; color: string; big?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-4">
      <span className="text-slatedim">{label}</span>
      <span className={big ? "font-display text-xl" : "tabular-nums"} style={{ color }}>
        {value.toFixed(1)}
      </span>
    </div>
  );
}
