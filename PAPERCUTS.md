# PAPERCUTS

Small, non-blocking frictions encountered by agents while working. Review this file periodically and sand them down.

## 2026-07-26T05:38:55.044Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/bellmanford`
- **Tags:** `remote-ops`

Running the v3 perturbation diagnostic on the Windows maints host via fleet: the staged source archive did not contain `.venv`, but the launch command assumed a repo-local interpreter. PowerShell first reported the path as a module name and then, with the call operator, confirmed it did not exist. Check `Get-Command python` after staging and use the resolved host interpreter.
## 2026-07-26T07:08:24.740Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/bellmanford`
- **Tags:** `gpu-ops`

Running three independent Bellman-Ford v4 saturation jobs concurrently on the RTX 3090 after nvidia-smi showed ample free memory: the width-128 LR=1e-3 job hit CUDA out-of-memory at step 15000 even though earlier aggregate reported memory use was low. Do not infer safe concurrency from a single nvidia-smi snapshot; run long diagnostic jobs sequentially or cap concurrency and preserve partial outputs as operational failures.

## 2026-07-26T08:31:39.098Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/bellmanford`
- **Tags:** `remote-ops`

While monitoring a long fleet-launched Windows training job, a Get-CimInstance Win32_Process -Filter query failed as Invalid query because nested quote escaping was brittle. Use unfiltered Get-CimInstance piped to Where-Object or a simpler Get-Process command for fleet PowerShell diagnostics.

## 2026-07-26T11:53:33.298Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/bellmanford`
- **Tags:** `remote-ops`

While checking whether a second configured remote host could run an independent diagnostic, fleet list failed because the CLI has no list subcommand. Use fleet help first and the documented inventory/status command instead of assuming list.

## 2026-07-26T11:54:59.884Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/bellmanford`
- **Tags:** `remote-ops`

While staging a committed git archive on a second Windows GPU host, Test-Path printed True but returned exit 0, so an && chain extracted into an existing directory. Never use Test-Path as a shell guard without an explicit if/exit; stage archives into a newly validated unique directory and verify its hash.

## 2026-07-26T12:44:20.356Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/bellmanford`
- **Tags:** `remote-ops`

While parallelizing independent GPU diagnostics, fleet aliases main-win and maints appeared as distinct hosts in fleet ls but resolved to the same physical Windows machine and GPU (same hostname and hardware UUID). Verify host identity before cross-host GPU parallelism; distinct fleet names do not guarantee distinct hardware.

## 2026-07-26T14:55:06.870Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/bellmanford`
- **Tags:** `remote-ops`

A 3-hour Windows training run launched with blocking fleet exec died at step 49k when SSH exited 255, leaving no run.json or optimizer state. Long remote experiments must use fleet spawn plus fleet jobs wait/tail so they outlive SSH; blocking exec is not durable enough.

## 2026-07-26T17:10:09.881Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/bellmanford`
- **Tags:** `shell-quoting`

A long jq command for packaging diagnostic curves failed with zsh unmatched quote because the single-quoted jq program contained an apostrophe in prose. Avoid apostrophes inside single-quoted jq programs or put the filter in a file.
