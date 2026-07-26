# PAPERCUTS

Small, non-blocking frictions encountered by agents while working. Review this file periodically and sand them down.

## 2026-07-26T05:38:55.044Z — codex — gpt-5.6-sol

- **Directory:** `/Users/safzan/Development/projects/bellmanford`
- **Tags:** `remote-ops`

Running the v3 perturbation diagnostic on the Windows maints host via fleet: the staged source archive did not contain `.venv`, but the launch command assumed a repo-local interpreter. PowerShell first reported the path as a module name and then, with the call operator, confirmed it did not exist. Check `Get-Command python` after staging and use the resolved host interpreter.
