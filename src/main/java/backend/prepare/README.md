# backend.prepare

`backend.prepare` owns backend-neutral preparation orchestration.

It may coordinate:

- `PreparedExecutionBuilder`
- `BackendPrepareDispatcher`
- prepared metadata indexes
- lowered-region and backend-plan lookup indexes

It must not own concrete backend implementation logic.
Concrete preparers belong under backend-specific packages:

- `backend.cpu.prepare`
- `backend.metal.prepare`
- `backend.cuda.prepare`

Shared accelerator preparation helpers belong under `backend.accelerator.prepare`.

The current concrete preparers in this package are migration leftovers scheduled for removal in phase 39.
Do not add new concrete backend preparers here.
