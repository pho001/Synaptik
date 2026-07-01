# prepare

The top-level `prepare` package separates shared backend prepare inputs and validation
from the composition root that builds a prepared execution.

Package ownership:

- `prepare.context` owns `BackendPrepareContext`, immutable prepare inputs, and
  package-private indexes over prepared metadata, selected plans, partition roles,
  and lowered regions.
- `prepare.validation` owns backend-neutral validation shared by backend preparers.
- `prepare.orchestration` owns `PreparedExecutionBuilder`, backend dispatch, and
  prepare trace contribution.

Concrete preparation logic remains backend-owned:

- `backend.cpu.prepare`
- `backend.cpu1.prepare`
- `backend.metal.prepare`
- `backend.cuda.prepare`

These backend-specific prepare packages are backend compilers: they consume backend-neutral
plans and shared context, resolve concrete storage, kernel, provider and launch choices, and
emit immutable executable artifacts. They never import `prepare.orchestration`. Runtime applies
the compiled choice and does not redo static prepare decisions.

Shared accelerator preparation helpers belong under `backend.accelerator.prepare`.
