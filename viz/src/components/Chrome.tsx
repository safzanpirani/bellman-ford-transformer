import type { ReactNode } from "react";

export function Panel({
  title, index, kicker, children, className = "", right,
}: {
  title: string;
  index?: string;
  kicker?: string;
  children: ReactNode;
  className?: string;
  right?: ReactNode;
}) {
  return (
    <section className={`panel bg-grain relative ${className}`}>
      <header className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-ink-700 px-4 py-2.5">
        {index && (
          <span className="text-[10px] tracking-[0.2em] text-amber/70 tabular-nums">{index}</span>
        )}
        <h3 className="text-[12px] font-medium tracking-[0.16em] uppercase text-slatebright">
          {title}
        </h3>
        {kicker && <span className="tick-label hidden sm:block">{kicker}</span>}
        <div className="ml-auto shrink-0">{right}</div>
      </header>
      <div className="p-4">{children}</div>
    </section>
  );
}

export function Stat({ label, value, unit, accent = "amber" }: {
  label: string; value: string | number; unit?: string; accent?: "amber" | "cyan" | "magenta" | "slate";
}) {
  const color = {
    amber: "text-amber", cyan: "text-cyan", magenta: "text-magenta", slate: "text-slatemid",
  }[accent];
  return (
    <div className="border-l border-ink-600 pl-3">
      <div className="tick-label">{label}</div>
      <div className={`font-display text-3xl leading-tight ${color}`}>
        {value}
        {unit && <span className="ml-1 font-mono text-xs text-slatedim">{unit}</span>}
      </div>
    </div>
  );
}

export function Tag({ children, color = "#5d7183", filled = false }: {
  children: ReactNode; color?: string; filled?: boolean;
}) {
  return (
    <span
      className="inline-flex items-center gap-1.5 border px-1.5 py-0.5 text-[10px] tracking-wider uppercase"
      style={{
        borderColor: color,
        color: filled ? "#06090d" : color,
        background: filled ? color : "transparent",
      }}
    >
      {children}
    </span>
  );
}

export function Note({ children }: { children: ReactNode }) {
  return (
    <p className="max-w-[68ch] text-[12.5px] leading-relaxed text-slatemid">{children}</p>
  );
}

export function Mono({ children }: { children: ReactNode }) {
  return (
    <code className="border border-ink-600 bg-ink-900 px-1 py-px text-[11.5px] text-cyan">
      {children}
    </code>
  );
}
