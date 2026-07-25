import { useState } from "react";
import type { Case } from "../types";
import { KIND_COLOR, attnColor } from "../lib/theme";

/** Attention pattern of one head: query rows × key columns, exactly one-hot. */
export function AttentionMatrix({
  kase, pattern, size = 13, title, subtitle,
}: {
  kase: Case; pattern: number[][]; size?: number; title: string; subtitle?: string;
}) {
  const [hover, setHover] = useState<{ q: number; k: number } | null>(null);
  const n = kase.tokens.length;
  const pad = 46;
  const w = pad + n * size + 4;
  const h = pad + n * size + 4;

  const offTarget = Math.max(
    ...pattern.map((row) => {
      const mx = Math.max(...row);
      return Math.max(...row.filter((v) => v !== mx), 0);
    })
  );

  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline gap-2">
        <span className="text-[11px] text-slatebright">{title}</span>
        {subtitle && <span className="tick-label">{subtitle}</span>}
      </div>
      <svg width={w} height={h} style={{ minWidth: w }}>
        {kase.tokens.map((t, ki) => (
          <text key={`k${ki}`} x={pad + ki * size + size / 2} y={pad - 6}
                fontSize={8} fill={KIND_COLOR[t.kind]} textAnchor="start"
                transform={`rotate(-90 ${pad + ki * size + size / 2} ${pad - 6})`}
                opacity={hover && hover.k !== ki ? 0.35 : 1}>
            {t.label}
          </text>
        ))}
        {kase.tokens.map((t, qi) => (
          <g key={`q${qi}`}>
            <text x={pad - 5} y={pad + qi * size + size / 2} fontSize={8}
                  fill={KIND_COLOR[t.kind]} textAnchor="end" dominantBaseline="central"
                  opacity={hover && hover.q !== qi ? 0.35 : 1}>
              {t.label}
            </text>
            {kase.tokens.map((_, ki) => (
              <rect key={ki}
                    x={pad + ki * size} y={pad + qi * size}
                    width={size - 1} height={size - 1}
                    fill={attnColor(pattern[qi][ki])}
                    stroke={hover?.q === qi && hover?.k === ki ? "#3ddad7" : "transparent"}
                    onMouseEnter={() => setHover({ q: qi, k: ki })}
                    onMouseLeave={() => setHover(null)} />
            ))}
          </g>
        ))}
      </svg>
      <div className="text-[10.5px] leading-snug text-slatedim">
        {hover ? (
          <span>
            <span style={{ color: KIND_COLOR[kase.tokens[hover.q].kind] }}>
              {kase.tokens[hover.q].label}
            </span>
            {" attends "}
            <span className="text-amber">{pattern[hover.q][hover.k].toFixed(6)}</span>
            {" to "}
            <span style={{ color: KIND_COLOR[kase.tokens[hover.k].kind] }}>
              {kase.tokens[hover.k].label}
            </span>
          </span>
        ) : (
          <span>
            max off-target weight ={" "}
            <span className={offTarget === 0 ? "text-lime" : "text-magenta"}>
              {offTarget === 0 ? "0 (exactly)" : offTarget.toExponential(2)}
            </span>
          </span>
        )}
      </div>
    </div>
  );
}

/** Pre-softmax logits: the reason the pattern above is exactly one-hot. */
export function LogitEvidence({
  labels, logits, cMatch, underflow,
}: {
  labels: string[]; logits: number[][]; cMatch: number; underflow: number;
}) {
  const n = labels.length;
  const size = 26;
  const pad = 52;
  const gaps: number[] = logits.map((row) => {
    const s = [...row].sort((a, b) => b - a);
    return s[0] - s[1];
  });
  const minGap = Math.min(...gaps);

  return (
    <div className="space-y-3">
      <div className="overflow-x-auto no-scrollbar">
        <svg width={pad + n * size + 4} height={pad + n * size + 4}>
          {labels.map((l, ki) => (
            <text key={ki} x={pad + ki * size + size / 2} y={pad - 8} fontSize={9}
                  fill="#5d7183" textAnchor="start"
                  transform={`rotate(-90 ${pad + ki * size + size / 2} ${pad - 8})`}>
              {l}
            </text>
          ))}
          {labels.map((l, qi) => (
            <g key={qi}>
              <text x={pad - 6} y={pad + qi * size + size / 2} fontSize={9}
                    fill="#5d7183" textAnchor="end" dominantBaseline="central">
                {l}
              </text>
              {labels.map((_, ki) => {
                const v = logits[qi][ki];
                const t = v / (4 * cMatch);
                return (
                  <g key={ki}>
                    <rect x={pad + ki * size} y={pad + qi * size}
                          width={size - 1} height={size - 1}
                          fill={v === 0 ? "rgba(16,24,33,.6)" : `rgba(240,180,41,${0.12 + t * 0.85})`} />
                    <text x={pad + ki * size + size / 2 - 0.5}
                          y={pad + qi * size + size / 2}
                          fontSize={7.5} textAnchor="middle" dominantBaseline="central"
                          fill={t > 0.55 ? "#06090d" : "#8ea3b5"}>
                      {v === 0 ? "0" : `${(v / cMatch).toFixed(0)}C`}
                    </text>
                  </g>
                );
              })}
            </g>
          ))}
        </svg>
      </div>
      <div className="grid gap-2 text-[11.5px] sm:grid-cols-3">
        <div className="border border-ink-600 bg-ink-900 p-2.5">
          <div className="tick-label">smallest winning gap</div>
          <div className="font-display text-2xl text-amber">{minGap.toLocaleString()}</div>
        </div>
        <div className="border border-ink-600 bg-ink-900 p-2.5">
          <div className="tick-label">float32 underflow at</div>
          <div className="font-display text-2xl text-cyan">exp(−{underflow})</div>
        </div>
        <div className="border border-ink-600 bg-ink-900 p-2.5">
          <div className="tick-label">margin</div>
          <div className="font-display text-2xl text-lime">
            {(minGap / underflow).toFixed(0)}×
          </div>
        </div>
      </div>
    </div>
  );
}
