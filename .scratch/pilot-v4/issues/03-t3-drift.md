# T3 perturbation drift

Status: needs-triage
Lifecycle: completed

Audit loss and gradient at the exact construction, then run perturbed
G1-compatible trajectories with zero terminal learning rate and a smaller peak
learning rate. Jointly record objective, exactness, and parameter distance.

## Resolution

The exact construction has zero loss and zero gradient. Zero-floor schedules
recover exactness, and no logged interval showed exactness falling while
validation loss also fell. The evidence supports optimizer-step drift.
