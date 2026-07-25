import { useState } from "react";
import type { Case, RegisterSpec, Trace } from "../types";
import { GROUP_COLOR, KIND_COLOR, valueColor } from "../lib/theme";

interface Props {
  kase: Case;
  trace: Trace;
  state: number[][];
  prevState: number[][] | null;
  registers: RegisterSpec[];
  dimLabel: (d: number) => string;
  highlight: string | null;
  onHighlight: (r: string | null) => void;
}

const CELL = 18;
const LABEL_W = 60;
const HEAD_H = 74;

/**
 * The residual stream itself: one row per token, one column per d_model dim,
 * banded and coloured by register. This is the artefact everything else is
 * describing — deltas from the previous sublayer are outlined.
 */
export function ResidualStream({
  kase, trace, state, prevState, registers, dimLabel, highlight, onHighlight,
}: Props) {
  const [hover, setHover] = useState<{ r: number; c: number } | null>(null);
  const d = trace.config.dModel;
  const rows = kase.tokens.length;
  const w = LABEL_W + d * CELL + 8;
  const h = HEAD_H + rows * CELL + 10;
  const infB = trace.config.infB;

  return (
    <div className="space-y-3">
      <div className="overflow-x-auto no-scrollbar">
        <svg width={w} height={h} className="block" style={{ minWidth: w }}>
          {/* register bands */}
          {registers.map((r) => {
            const x = LABEL_W + r.start * CELL;
            const rw = r.size * CELL;
            const active = highlight === r.name;
            return (
              <g key={r.name}
                 onMouseEnter={() => onHighlight(r.name)}
                 onMouseLeave={() => onHighlight(null)}
                 style={{ cursor: "pointer" }}>
                <rect x={x} y={HEAD_H - 6} width={rw} height={rows * CELL + 6}
                      fill={active ? `${GROUP_COLOR[r.group]}14` : "transparent"} />
                <rect x={x} y={HEAD_H - 12} width={rw - 1} height={4}
                      fill={GROUP_COLOR[r.group]} opacity={active ? 1 : 0.65} />
                <text
                  x={x + 2} y={HEAD_H - 18}
                  fontSize={9} letterSpacing="0.04em"
                  fill={active ? GROUP_COLOR[r.group] : "#5d7183"}
                  transform={`rotate(-90 ${x + 2} ${HEAD_H - 18})`}
                  style={{ transition: "fill .15s" }}
                >
                  {r.name}
                </text>
              </g>
            );
          })}

          {/* token rows */}
          {kase.tokens.map((t, ri) => {
            const y = HEAD_H + ri * CELL;
            return (
              <g key={t.row}>
                <text x={LABEL_W - 6} y={y + CELL / 2} textAnchor="end"
                      dominantBaseline="central" fontSize={10}
                      fill={KIND_COLOR[t.kind]}>
                  {t.label}
                </text>
                {Array.from({ length: d }, (_, ci) => {
                  const v = state[ri][ci];
                  const changed =
                    prevState && Math.abs(prevState[ri][ci] - v) > 1e-6;
                  const reg = registers.find((r) => ci >= r.start && ci < r.stop)!;
                  const dimmed = highlight && reg.name !== highlight;
                  return (
                    <rect
                      key={ci}
                      x={LABEL_W + ci * CELL}
                      y={y}
                      width={CELL - 1.2}
                      height={CELL - 1.2}
                      fill={valueColor(v, infB)}
                      opacity={dimmed ? 0.18 : 1}
                      stroke={changed ? "#3ddad7" : "transparent"}
                      strokeWidth={changed ? 1.4 : 0}
                      onMouseEnter={() => setHover({ r: ri, c: ci })}
                      onMouseLeave={() => setHover(null)}
                      style={{ transition: "opacity .15s" }}
                    />
                  );
                })}
              </g>
            );
          })}
        </svg>
      </div>

      <div className="flex min-h-[42px] flex-wrap items-center gap-x-5 gap-y-1 border-t border-ink-700 pt-2.5 text-[11px]">
        {hover ? (
          <>
            <span className="text-slatedim">token</span>
            <span style={{ color: KIND_COLOR[kase.tokens[hover.r].kind] }}>
              {kase.tokens[hover.r].label}
            </span>
            <span className="text-slatedim">register</span>
            <span className="text-slatebright">{dimLabel(hover.c)}</span>
            <span className="text-slatedim">value</span>
            <span className="font-display text-lg text-amber">
              {(() => {
                const v = state[hover.r][hover.c];
                return Math.abs(v - infB) < 0.5 ? "INF_B" : v;
              })()}
            </span>
            {prevState &&
              Math.abs(prevState[hover.r][hover.c] - state[hover.r][hover.c]) > 1e-6 && (
                <span className="text-cyan">
                  Δ {(state[hover.r][hover.c] - prevState[hover.r][hover.c]).toFixed(3)}
                </span>
              )}
          </>
        ) : (
          <span className="text-slatedim">
            Hover a cell to inspect · cyan outline = written by this sublayer ·
            magenta = INF_B ({infB})
          </span>
        )}
      </div>
    </div>
  );
}
