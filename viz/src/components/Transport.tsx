import type { Case } from "../types";
import type { Transport as T } from "../lib/useTrace";
import { layerDoc } from "../lib/theme";

export function TransportBar({ t, kase }: { t: T; kase: Case }) {
  const R = kase.rounds.length;
  const doc = layerDoc(t.stepName);

  return (
    <div className="sticky top-0 z-30 border-b border-ink-700 bg-ink-950/92 backdrop-blur-md">
      <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-x-4 gap-y-2 px-4 py-2.5 sm:px-6">
        <div className="flex items-center gap-1">
          <Btn onClick={() => { t.setPlaying(false); t.setRound(0); t.setStep(0); }} label="⏮" title="reset" />
          <Btn onClick={() => { t.setPlaying(false); t.stepBack(); }} label="◀" title="previous sublayer" />
          <button
            onClick={() => t.setPlaying(!t.playing)}
            className="h-7 border border-amber bg-amber/12 px-3 text-[11px] text-amber transition-colors hover:bg-amber/22"
          >
            {t.playing ? "❚❚ pause" : "▶ run"}
          </button>
          <Btn onClick={() => { t.setPlaying(false); t.stepForward(); }} label="▶" title="next sublayer" />
        </div>

        <div className="flex items-center gap-2">
          <span className="tick-label">round</span>
          <div className="flex gap-0.5">
            {Array.from({ length: R + 1 }, (_, r) => (
              <button
                key={r}
                onClick={() => { t.setPlaying(false); t.setRound(r); t.setStep(0); }}
                className="h-6 w-6 border text-[10px] tabular-nums transition-colors"
                style={{
                  borderColor: r === t.round ? "#f0b429" : "#1a2531",
                  background: r === t.round ? "#f0b42922" : r < t.round ? "#f0b4290a" : "transparent",
                  color: r === t.round ? "#f0b429" : r < t.round ? "#8a6712" : "#5d7183",
                }}
              >
                {r}
              </button>
            ))}
          </div>
          <span className="text-[10.5px] text-slatedim">of {R}</span>
        </div>

        {t.round > 0 && (
          <div className="flex min-w-[220px] flex-1 items-center gap-2">
            <span className="tick-label shrink-0">sublayer</span>
            <input
              type="range" min={0} max={t.totalSteps - 1} value={t.step}
              onChange={(e) => { t.setPlaying(false); t.setStep(Number(e.target.value)); }}
              className="h-1 flex-1 cursor-pointer appearance-none bg-ink-600 accent-cyan"
            />
            <span className="w-12 shrink-0 text-right text-[10.5px] tabular-nums text-slatedim">
              {t.step + 1}/{t.totalSteps}
            </span>
          </div>
        )}

        <div className="ml-auto flex items-center gap-2 text-[11px]">
          <span
            className="border px-1.5 py-0.5 text-[10px] tracking-wider uppercase"
            style={
              t.stepName.startsWith("attn")
                ? { borderColor: "#3ddad7", color: "#3ddad7" }
                : t.stepName.startsWith("mlp")
                ? { borderColor: "#9d7cff", color: "#9d7cff" }
                : { borderColor: "#5d7183", color: "#5d7183" }
            }
          >
            {t.stepName.startsWith("attn") ? "attention" : t.stepName.startsWith("mlp") ? "mlp" : "input"}
          </span>
          <span className="text-slatebright">{doc.title}</span>
        </div>
      </div>
    </div>
  );
}

function Btn({ onClick, label, title }: { onClick: () => void; label: string; title: string }) {
  return (
    <button onClick={onClick} title={title}
            className="h-7 w-7 border border-ink-600 text-[10px] text-slatemid transition-colors hover:border-ink-500 hover:text-slatebright">
      {label}
    </button>
  );
}
