import { useState } from "react";
import type { Case, Trace } from "../types";
import { fmt } from "../lib/theme";

const SERIES_COLORS = ["#9ee493", "#ff5c8a", "#ff9ec0", "#f0b429", "#3ddad7", "#9d7cff"];

/** FR8: max |error| per round, exact reducer vs soft-min at several temperatures. */
export function SoftVsExact({ trace }: { trace: Trace }) {
  const { series, maxRounds, graphs } = trace.softSweep;
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const W = 760, H = 340, L = 62, B = 40, T = 16, Rt = 14;

  const floor = 1e-6;
  const all = series.flatMap((s) => s.errors).filter((e) => e > 0);
  const maxE = Math.max(...all, 1);
  const ly = (v: number) => Math.log10(Math.max(v, floor));
  const y0 = ly(floor), y1 = ly(maxE * 1.6);
  const sx = (i: number) => L + (i / (maxRounds - 1)) * (W - L - Rt);
  const sy = (v: number) => T + (1 - (ly(v) - y0) / (y1 - y0)) * (H - T - B);

  const ticks = [1e-6, 1e-4, 1e-2, 1, 100];

  return (
    <div className="space-y-3">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={L} y1={sy(t)} x2={W - Rt} y2={sy(t)} stroke="#1a2531" />
            <text x={L - 8} y={sy(t)} fontSize={9} fill="#5d7183" textAnchor="end"
                  dominantBaseline="central">
              {t === 1e-6 ? "0" : t < 1 ? `1e${Math.log10(t)}` : t}
            </text>
          </g>
        ))}
        {Array.from({ length: maxRounds }, (_, i) => (
          <text key={i} x={sx(i)} y={H - B + 15} fontSize={9} fill="#5d7183"
                textAnchor="middle">{i + 1}</text>
        ))}
        <text x={(L + W) / 2} y={H - 6} fontSize={9.5} fill="#5d7183" textAnchor="middle">
          rounds applied
        </text>
        <text x={14} y={H / 2} fontSize={9.5} fill="#5d7183" textAnchor="middle"
              transform={`rotate(-90 14 ${H / 2})`}>
          max |distance error|
        </text>
        <line x1={L} y1={sy(trace.config.nMax * 0)} x2={L} y2={H - B} stroke="#24313f" />

        {series.map((s, si) => {
          if (hidden.has(s.label)) return null;
          const color = SERIES_COLORS[si % SERIES_COLORS.length];
          const pts = s.errors.map((e, i) => `${sx(i)},${sy(e)}`).join(" ");
          const isExact = s.cSoft === null;
          return (
            <g key={s.label}>
              <polyline points={pts} fill="none" stroke={color}
                        strokeWidth={isExact ? 2.4 : 1.5}
                        strokeDasharray={isExact ? undefined : "4 3"}
                        className={isExact ? "glow-cyan" : ""} />
              {s.errors.map((e, i) => (
                <circle key={i} cx={sx(i)} cy={sy(e)} r={isExact ? 3 : 2.2} fill={color}>
                  <title>{`${s.label} · round ${i + 1} · err ${e}`}</title>
                </circle>
              ))}
            </g>
          );
        })}
      </svg>

      <div className="flex flex-wrap gap-2">
        {series.map((s, si) => {
          const color = SERIES_COLORS[si % SERIES_COLORS.length];
          const off = hidden.has(s.label);
          return (
            <button key={s.label}
                    onClick={() => {
                      const n = new Set(hidden);
                      off ? n.delete(s.label) : n.add(s.label);
                      setHidden(n);
                    }}
                    className="flex items-center gap-1.5 border px-2 py-1 text-[10.5px] transition-opacity"
                    style={{ borderColor: off ? "#1a2531" : `${color}66`, opacity: off ? 0.4 : 1,
                             color: off ? "#5d7183" : color }}>
              <span className="inline-block h-[2px] w-4" style={{ background: color }} />
              {s.label}
              <span className="text-slatedim">
                → {s.errors[s.errors.length - 1] === 0 ? "0" : s.errors[s.errors.length - 1].toExponential(1)}
              </span>
            </button>
          );
        })}
      </div>
      <p className="text-[11.5px] leading-relaxed text-slatemid">
        {graphs} random graphs. The exact reducer hits zero error and stays there;
        every soft-min temperature plateaus on a floor it can never leave, because
        a softmax average of candidates is not their minimum.
      </p>
    </div>
  );
}

/** Model DIST vs the reference algorithm, round by round — FR-V, drawn. */
export function ConvergencePlot({
  kase, trace, distsByRound, round,
}: {
  kase: Case; trace: Trace; distsByRound: number[][]; round: number;
}) {
  const infB = trace.config.infB;
  const R = kase.rounds.length;
  const W = 560, H = 230, L = 42, B = 30, T = 12, Rt = 84;
  const finite = distsByRound.flat().filter((v) => Math.abs(v - infB) > 0.5);
  const maxD = Math.max(...finite, 1);
  const sx = (i: number) => L + (i / R) * (W - L - Rt);
  const sy = (v: number) =>
    Math.abs(v - infB) < 0.5 || v > infB
      ? T + 6
      : H - B - (v / (maxD * 1.15)) * (H - T - B - 14);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      <line x1={L} y1={T + 6} x2={W - Rt} y2={T + 6} stroke="#ff5c8a" strokeWidth={1}
            strokeDasharray="2 4" opacity={0.6} />
      <text x={L - 6} y={T + 6} fontSize={9} fill="#ff5c8a" textAnchor="end"
            dominantBaseline="central">∞</text>
      <line x1={L} y1={H - B} x2={W - Rt} y2={H - B} stroke="#24313f" />
      <line x1={L} y1={T} x2={L} y2={H - B} stroke="#24313f" />
      {Array.from({ length: R + 1 }, (_, i) => (
        <g key={i}>
          <line x1={sx(i)} y1={H - B} x2={sx(i)} y2={H - B + 4} stroke="#24313f" />
          <text x={sx(i)} y={H - B + 15} fontSize={9} fill="#5d7183" textAnchor="middle">{i}</text>
        </g>
      ))}
      <text x={(L + W - Rt) / 2} y={H - 4} fontSize={9} fill="#5d7183" textAnchor="middle">
        rounds
      </text>
      <rect x={sx(round)} y={T} width={2} height={H - B - T} fill="#f0b429" opacity={0.28} />

      {Array.from({ length: kase.n }, (_, node) => {
        const pts = distsByRound.map((d, i) => `${sx(i)},${sy(d[node])}`).join(" ");
        const hue = `hsl(${(node * 47) % 360} 70% 62%)`;
        const last = distsByRound[distsByRound.length - 1][node];
        return (
          <g key={node}>
            <polyline points={pts} fill="none" stroke={hue} strokeWidth={1.8} opacity={0.9} />
            {distsByRound.map((d, i) => (
              <circle key={i} cx={sx(i)} cy={sy(d[node])} r={i === round ? 4 : 2.2}
                      fill={hue} opacity={i === round ? 1 : 0.75} />
            ))}
            <text x={W - Rt + 6} y={sy(last)} fontSize={10} fill={hue}
                  dominantBaseline="central">
              n{node} = {fmt(Math.abs(last - infB) < 0.5 ? null : last, infB)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
