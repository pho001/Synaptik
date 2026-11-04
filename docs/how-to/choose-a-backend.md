# Choose a backend deliberately

## Outcome

Choose an execution owner that supports your actual program without treating a dtype name,
GPU presence or preferred native API as a capability guarantee. Start with the
[CPU learning project](../examples/README.md) and its JDK26/private-artifact prerequisites.

## Decide from the program, not the device name

1. List the operations, exact data types, layouts, shapes and required derivative order. Check
   them against the [current support matrix](../backend-guide/production-support.md).
2. Keep standard CPU for the first working result. `Engine.standard()` provides a CPU composition;
   it is not automatic CPU/GPU selection. The tutorial needs no external BLAS or oneDNN library.
3. If selecting Metal, obtain and verify the documented macOS arm64 native package and supply its
   absolute library path. Register the opened integration explicitly through Engine's ownership
   builder. Use the [complete composition reference](../reference/engine/lifecycle.md), not a
   replacement hidden dependency or service loader.
4. For an owned Model, call `withEngine` before its first valid compute/training setup attempt.
   That Engine is borrowed: keep it open for the Model's dependent prepared work and close it
   yourself afterward. A failed setup can still freeze the choice.
5. Compare representative outputs before timing. Measure preparation and steady-state runs
   separately; retain a prepared owner when that is the workload you intend to measure.

| Choice | What it means | What it does not mean |
|---|---|---|
| Standard CPU | Existing CPU realization and occurrence-qualified optional OpenBLAS probe | A required native install, GPU discovery or a universal speed claim |
| Explicit oneDNN CPU | An exact external supported library/API configuration, qualified per occurrence | Native FP8 availability, F64 support or proof that every unit used oneDNN |
| Explicit Metal | A supported Apple-silicon device/package and qualified Metal program | CPU parity, FP8 support or general FLOAT64 arithmetic |
| Explicit CPU+Metal | Planning across registered owners with checked scheduled transfers | Automatic runtime recovery on another backend or implicit dtype conversion |

## Check what actually happened

Configured preference and selected realization differ. Use the
[optional preparation diagnostics](../developer-guide/debugging-trace.md) when a route matters.
A CPU `GRAPH` preference does not prove that oneDNN Graph ran. Preparing multiple registered
backends does not prove that both received a partition. The Metal failure record describes an
actual failing invocation, not a replay.

The observable result should first be equivalent model output in the qualified domain. A new
backend can be slower on small tensors because initialization, dispatch, transfer or
materialization dominates. The [benchmarking guide](../developer-guide/benchmarking.md) explains
bounded measurements; no tutorial timing is a cross-machine performance promise.

## Common errors and useful variation

| Symptom | Cause to check | Action |
|---|---|---|
| Unsupported operation/dtype/layout | The particular occurrence is outside the selected backend's domain | Consult the exact matrix row; change the program or explicitly choose a supported owner |
| Metal rejects FP8 | FP8 is not a Metal capability | Retain CPU, or explicitly redesign storage/types and revalidate the changed program |
| Model Engine binding is frozen | Selection followed a valid setup attempt | Select before setup on a fresh Model; do not retry by mutating prepared ownership |
| Native load/version/package error | Wrong file, target, dependencies or verification | Follow the owning package instructions; do not suppress it as a numerical fallback |

Useful variation: keep the learning model and input unchanged while selecting an explicitly opened
CPU Engine before its first compute. Its predictions should remain unchanged; observer events can
then explain preparation without changing the learning objective. For Metal, use the qualified
composition instructions and actual package rather than assuming the same source is sufficient.
