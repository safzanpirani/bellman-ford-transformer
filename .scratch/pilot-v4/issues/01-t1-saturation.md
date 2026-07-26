# T1 saturation and size climb

Status: needs-triage
Lifecycle: completed

Train the favourable diagnostic recipe with cosine decay to zero until the
pre-stated two-window plateau criterion is met. Explore learning rate and width,
then climb the selected recipe from \(n \leq 4\) to \(n \leq 8\).

## Comments

- V3's best checkpoints were at steps 12,000 and 11,250 of 12,000, so those
  runs are treated as budget-limited.
- Width 128 with LayerNorm saturated at 80,000 steps on \(n \leq 4\).
- The \(n \leq 8\) climb was dropped under the final four-hour cut after two
  operationally interrupted attempts; no partial result was reported.
