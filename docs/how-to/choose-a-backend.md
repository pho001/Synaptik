# Choose a [backend](../glossary.md#backend) deliberately

## Outcome

Choose an execution [owner](../glossary.md#term-owner) that supports your actual program without treating a [dtype](../glossary.md#data-type--datatype) name,
[GPU](../glossary.md#term-gpu) presence or preferred native [API](../glossary.md#term-application-programming-interface-api) as a capability guarantee. Start with the
[CPU learning project](../examples/README.md) and its JDK26/private-[artifact](../glossary.md#term-artifact) prerequisites.

A **[backend](../glossary.md#backend)** supplies execution on a supported processor or [device](../glossary.md#term-device); a **route** is one concrete
implementation within it. [CPU](../glossary.md#term-cpu) is the central processor, while [Metal](../glossary.md#term-metal) provides the supported Apple
GPU (graphics processor) integration. A **dtype** is an [element](../glossary.md#term-element)'s numerical type, such as [FLOAT32](../glossary.md#term-float32).
An **[operation](../glossary.md#operation) [occurrence](../glossary.md#term-occurrence)** is one particular operation with its actual inputs and attributes,
not every possible use of its name. **[Derivative](../glossary.md#term-derivative) order** distinguishes a first derivative from
a derivative of a derivative. Check the needed order, not only forward [prediction](../glossary.md#term-prediction) support.

## Decide from the program, not the [device](../glossary.md#term-device) name

1. List the operations, exact data types, layouts, shapes and required derivative order. Check
   them against the <span data-glossary-navigation="true"> ([layouts glossary](../glossary.md#layout); [shapes glossary](../glossary.md#shape); [derivative glossary](../glossary.md#term-derivative))</span>[current support matrix](../backend-guide/production-support.md)<span data-glossary-navigation="true"> ([support matrix glossary](../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([matrix glossary](../glossary.md#term-review-matrix))</span>.
2. Keep standard [CPU](../glossary.md#term-cpu) for the first working result. `Engine.standard()` provides a CPU composition;
   it is not automatic CPU/GPU selection. The tutorial needs no external BLAS or oneDNN library.<span data-glossary-navigation="true"> ([BLAS glossary](../glossary.md#term-blas))</span>
3. If selecting Metal, obtain and verify the documented macOS arm64 native package and supply its
   absolute library path. Register the opened integration explicitly through Engine's ownership
   builder. Use the <span data-glossary-navigation="true"> ([arm64 glossary](../glossary.md#native-packaging-and-platform-vocabulary); [Engine glossary](../glossary.md#term-engine); [ownership glossary](../glossary.md#term-ownership-meanings))</span>[complete composition reference](../reference/engine/lifecycle.md), not a
   replacement hidden dependency or service loader.<span data-glossary-navigation="true"> ([dependency glossary](../glossary.md#term-dependency))</span>
4. For an [owned Model](../glossary.md#model-module-buffer-and-scan-distinctions), call `withEngine` before its first valid compute/training setup attempt.
   That Engine is borrowed: keep it open for the Model's dependent prepared work and close it
   yourself afterward. A failed setup can still freeze the choice.<span data-glossary-navigation="true"> ([borrowed glossary](../glossary.md#ownership-borrowing-and-lease); [Model glossary](../glossary.md#term-model))</span>
5. Compare representative outputs before timing. Measure preparation and steady-state runs
   separately; retain a prepared owner when that is the workload you intend to measure.

| Choice | What it means | What it does not mean |
|---|---|---|
| Standard CPU | Existing CPU realization and [occurrence](../glossary.md#term-occurrence)-qualified optional [OpenBLAS](../glossary.md#term-openblas) probe | A required native install, [GPU](../glossary.md#term-gpu) discovery or a universal speed claim |
| Explicit [oneDNN](../glossary.md#term-onednn) CPU | An exact external supported library/[API](../glossary.md#term-application-programming-interface-api) configuration, qualified per occurrence | Native [FP8](../glossary.md#term-fp8) availability, [F64](../glossary.md#floating-point-precision-and-rounding) support or proof that every unit used oneDNN |
| Explicit [Metal](../glossary.md#term-metal) | A supported Apple-silicon device/package and qualified Metal program | CPU parity, FP8 support or general [FLOAT64](../glossary.md#term-float64) arithmetic |
| Explicit CPU+Metal | [Planning](../glossary.md#term-planning) across registered [owners](../glossary.md#term-owner) with checked scheduled transfers | Automatic [runtime](../glossary.md#term-runtime) recovery on another [backend](../glossary.md#backend) or implicit [dtype](../glossary.md#data-type--datatype) conversion |

**FP8**, **F64** and **FLOAT64** name numerical representations, not processors: FP8 stores a
[floating-point](../glossary.md#term-floating-point) value in eight bits, while F64/FLOAT64 uses 64 bits. A **[partition](../glossary.md#partition)** groups work
assigned to one backend. A **transfer** moves a value across the boundary between owners;
it does not implicitly change its type. **[Steady state](../glossary.md#term-steady-state)** is the repeated-work period after setup.
The [ownership picture](../explanation/ownership.md#a-result-is-not-always-an-independent-value)
shows why retaining a prepared [owner](../glossary.md#term-owner) and retaining [detached](../glossary.md#detached-model-results) output have different lifetimes.

## Check what actually happened

Configured preference and selected realization differ. Use the
[optional preparation diagnostics](../developer-guide/debugging-trace.md) when a route matters.
A [CPU](../glossary.md#term-cpu) [`GRAPH`](../glossary.md#graph) preference does not prove that [oneDNN](../glossary.md#term-onednn) Graph ran. Preparing multiple registered
[backends](../glossary.md#backend) does not prove that both received a [partition](../glossary.md#partition). The [Metal](../glossary.md#term-metal) failure record describes an
actual failing invocation, not a replay.

The observable result should first be equivalent [model](../glossary.md#term-model) output in the qualified domain. A new
backend can be slower on small [tensors](../glossary.md#tensor) because initialization, [dispatch](../glossary.md#term-dispatch), transfer or
[materialization](../glossary.md#materialization) dominates. The [benchmarking guide](../developer-guide/benchmarking.md)<span data-glossary-navigation="true"> ([benchmarking glossary](../glossary.md#term-benchmarking))</span> explains
bounded measurements; no tutorial timing is a cross-machine performance promise.

## Common errors and useful variation

| Symptom | Cause to check | Action |
|---|---|---|
| Unsupported [operation](../glossary.md#operation)/[dtype](../glossary.md#data-type--datatype)/[layout](../glossary.md#layout) | The particular [occurrence](../glossary.md#term-occurrence) is outside the selected [backend](../glossary.md#backend)'s domain | Consult the exact [matrix](../glossary.md#term-matrix-meanings) row; change the program or explicitly choose a supported [owner](../glossary.md#term-owner) |
| [Metal](../glossary.md#term-metal) rejects [FP8](../glossary.md#term-fp8) | FP8 is not a Metal capability | Retain [CPU](../glossary.md#term-cpu), or explicitly redesign storage/types and revalidate the changed program |
| [Model](../glossary.md#term-model) [Engine](../glossary.md#term-engine) [binding](../glossary.md#term-binding) is frozen | Selection followed a valid setup attempt | Select before setup on a fresh Model; do not retry by mutating prepared [ownership](../glossary.md#term-ownership-meanings) |
| Native load/version/package error | Wrong file, [target](../glossary.md#term-target), dependencies or verification | Follow the owning package instructions; do not suppress it as a numerical fallback |

Useful variation: keep the learning model and input unchanged while selecting an explicitly opened
CPU Engine before its first compute. Its [predictions](../glossary.md#term-prediction) should remain unchanged; [observer](../glossary.md#term-observer) events can
then explain preparation without changing the learning [objective](../glossary.md#term-objective). For Metal, use the qualified
composition instructions and actual package rather than assuming the same source is sufficient.
