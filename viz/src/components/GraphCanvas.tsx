import { useMemo } from "react";
import type { Case, Trace } from "../types";
import { fmt } from "../lib/theme";

interface Props {
  kase: Case;
  trace: Trace;
  dist: number[];
  prevDist: number[] | null;
  /** edge indices whose candidate is currently the winning one, if known */
  activeEdges: Set<number>;
  hoverNode: number | null;
  onHoverNode: (n: number | null) => void;
}

const W = 560;
const H = 460;
const R = 26;

export function GraphCanvas({
  kase, trace, dist, prevDist, activeEdges, hoverNode, onHoverNode,
}: Props) {
  const pts = useMemo(
    () =>
      kase.layout.map((p) => ({
        x: W / 2 + p.x * (W / 2 - 78),
        y: H / 2 + p.y * (H / 2 - 68),
      })),
    [kase]
  );

  // group parallel / bidirectional edges so they don't overlap
  const arcs = useMemo(() => {
    const seen = new Map<string, number>();
    return kase.edges.map((e, i) => {
      const key = [Math.min(e.u, e.v), Math.max(e.u, e.v)].join("-");
      const nth = seen.get(key) ?? 0;
      seen.set(key, nth + 1);
      return { ...e, i, nth, self: e.u === e.v };
    });
  }, [kase]);

  const infB = trace.config.infB;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full select-none" role="img"
         aria-label="Graph with live distance estimates">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#35485a" />
        </marker>
        <marker id="arrow-hot" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#3ddad7" />
        </marker>
        <radialGradient id="src-halo">
          <stop offset="0%" stopColor="#f0b429" stopOpacity="0.45" />
          <stop offset="100%" stopColor="#f0b429" stopOpacity="0" />
        </radialGradient>
      </defs>

      {arcs.map((e) => {
        const a = pts[e.u];
        const b = pts[e.v];
        const hot = activeEdges.has(e.i);
        const dim =
          hoverNode !== null && e.u !== hoverNode && e.v !== hoverNode;
        let d: string;
        let labelPos: { x: number; y: number };

        if (e.self) {
          const cx = a.x;
          const cy = a.y;
          const k = 46 + e.nth * 13;
          d = `M ${cx - 11} ${cy - R + 3} C ${cx - k} ${cy - k}, ${cx + k} ${cy - k}, ${cx + 11} ${cy - R + 3}`;
          labelPos = { x: cx, y: cy - R - 34 - e.nth * 9 };
        } else {
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const len = Math.hypot(dx, dy) || 1;
          const ux = dx / len;
          const uy = dy / len;
          const off = (e.nth % 2 === 0 ? 1 : -1) * Math.ceil(e.nth / 2) * 34;
          const mx = (a.x + b.x) / 2 - uy * off;
          const my = (a.y + b.y) / 2 + ux * off;
          const sx = a.x + ux * R;
          const sy = a.y + uy * R;
          const tx = b.x - ux * (R + 7);
          const ty = b.y - uy * (R + 7);
          d = `M ${sx} ${sy} Q ${mx} ${my} ${tx} ${ty}`;
          labelPos = { x: (sx + tx) / 2 - uy * (off * 0.62) , y: (sy + ty) / 2 + ux * (off * 0.62) };
        }

        return (
          <g key={e.i} opacity={dim ? 0.16 : 1} style={{ transition: "opacity .2s" }}>
            <path
              d={d}
              fill="none"
              stroke={hot ? "#3ddad7" : "#35485a"}
              strokeWidth={hot ? 2.4 : 1.4}
              markerEnd={e.self ? undefined : `url(#${hot ? "arrow-hot" : "arrow"})`}
              className={hot ? "anim-dash glow-cyan" : ""}
            />
            <g transform={`translate(${labelPos.x} ${labelPos.y})`}>
              <rect x={-13} y={-8} width={26} height={16} fill="#0a0e14"
                    stroke={hot ? "#3ddad7" : "#1a2531"} />
              <text textAnchor="middle" dominantBaseline="central"
                    fontSize={10.5} fill={hot ? "#3ddad7" : "#8ea3b5"}>
                {e.w}
              </text>
            </g>
          </g>
        );
      })}

      {pts.map((p, i) => {
        const v = dist[i] ?? infB;
        const isInf = Math.abs(v - infB) < 0.5 || v > infB;
        const changed = prevDist ? Math.abs((prevDist[i] ?? infB) - v) > 1e-6 : false;
        const isSrc = i === kase.source;
        const dim = hoverNode !== null && hoverNode !== i;
        return (
          <g key={i} opacity={dim ? 0.3 : 1}
             onMouseEnter={() => onHoverNode(i)}
             onMouseLeave={() => onHoverNode(null)}
             style={{ cursor: "pointer", transition: "opacity .2s" }}>
            {isSrc && <circle cx={p.x} cy={p.y} r={44} fill="url(#src-halo)" />}
            {changed && (
              <circle cx={p.x} cy={p.y} r={R} fill="none" stroke="#3ddad7" strokeWidth={2}>
                <animate attributeName="r" from={R} to={R + 22} dur="0.9s" repeatCount="indefinite" />
                <animate attributeName="opacity" from="0.8" to="0" dur="0.9s" repeatCount="indefinite" />
              </circle>
            )}
            <circle
              cx={p.x} cy={p.y} r={R}
              fill={isInf ? "#0d131a" : isSrc ? "#f0b429" : "#111922"}
              stroke={isInf ? "#35485a" : isSrc ? "#f0b429" : "#f0b429"}
              strokeWidth={isSrc ? 2.5 : 1.6}
              style={{ transition: "fill .3s, stroke .3s" }}
            />
            <text x={p.x} y={p.y - 5} textAnchor="middle" dominantBaseline="central"
                  fontSize={15} fontWeight={500}
                  fill={isSrc ? "#06090d" : isInf ? "#5d7183" : "#c9d8e4"}>
              {fmt(isInf ? null : v, infB)}
            </text>
            <text x={p.x} y={p.y + 11} textAnchor="middle" dominantBaseline="central"
                  fontSize={9} letterSpacing="0.08em"
                  fill={isSrc ? "rgba(6,9,13,.72)" : "#5d7183"}>
              n{i}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
