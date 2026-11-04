# 7. Choose [backend](../glossary.md#backend) and performance work deliberately

## Outcome and prerequisites

Finish with a reproducible correctness baseline and choose the next focused guide without
confusing a new [backend](../glossary.md#backend) with a new mathematical [model](../glossary.md#term-model). Complete
[setup](../examples/README.md#build-the-complete-project) and
[reusable execution](reusable-execution.md). Commands run in the example project with [JDK](../glossary.md#term-java-development-kit-jdk) 26
and `LEARNING_CP`; no native [accelerator](../glossary.md#term-accelerator) is needed for this chapter's baseline.

## Keep meaning separate from the route

A **[backend](../glossary.md#backend)** provides execution for supported [operation](../glossary.md#operation) occurrences. A **route** is a concrete
implementation choice within one backend. [CPU](../glossary.md#term-cpu) [scalar](../glossary.md#cpu-portable-execution-strategy)/generated [JVM](../glossary.md#term-java-virtual-machine-jvm)/[Vector API](../glossary.md#term-vector-api)/[native library](../glossary.md#term-native-library)
choices are not separate mathematical models. The [Model](../glossary.md#term-model) operation families define the meaning;
[backend capability](../glossary.md#backend-capability) still depends on actual [dtype](../glossary.md#data-type--datatype), [shape](../glossary.md#shape), [layout](../glossary.md#layout) and requested [derivatives](../glossary.md#term-derivative).
A [graph](../glossary.md#graph) that can be described or saved is not thereby executable on every backend.

**CPU** means the central processor running ordinary Java and supported native routes;
**[GPU](../glossary.md#term-gpu)** means a graphics processor that can also execute numerical work. **JVM** is the [Java
Virtual Machine](../glossary.md#java-platform-and-build-tools). Java's **Vector API** expresses operations on several numerical [lanes](../glossary.md#term-lane) together.
[OpenBLAS](../glossary.md#term-openblas) and [oneDNN](../glossary.md#term-onednn) are external CPU numerical libraries; [Metal](../glossary.md#term-metal) is Apple's [device](../glossary.md#term-device)-computation
interface. None of those names replaces checking whether the exact operation is supported.

The application uses `Engine.standard()`, which creates a CPU-only Engine. Model borrows the
explicit Engine in `reuse`; the application closes it after Model. Engine in turn owns its
opened backend integration. Adding native-access permission neither selects Metal nor promises
OpenBLAS use. Standard CPU composition can probe optional OpenBLAS even when the actual graph
uses portable Java. The project requires no Vector API module; explicit Vector API routes have
separate setup requirements.

The tiny workload remains FLOAT32 input `[2,1]`, private weight `[1,1]`, no bias, SUM loss and
`Sgd(0.05)`. Its pre-update objective $L(0)=40$, derivative $L'(0)=-40$ and updated weight
$w_{\mathrm{new}}=0-0.05(-40)=2$ refer to the
[defined one-weight calculation](train-and-evaluate.md#work-through-the-update), not timings.
Nothing here changes the candidate, adds a dependency, changes numerical semantics or enables
a global backend switch.

## Run the baseline, not a timing claim

```bash
# Keep correctness and lifetime checks before attempting any timing experiment.
java --enable-native-access=ALL-UNNAMED -cp "$LEARNING_CP" \
  io.github.pho001.synaptik.examples.LearningJourney reuse
```

Expected application stdout, derived from the source rather than an observed benchmark:

```text
== reuse ==
reuse.model-first=[2.0, 6.0]
reuse.model-fresh=[4.0, 8.0]
reuse.prepared=[2.0, 6.0]
reuse.prepared=[4.0, 8.0]
reuse.after-all-owners-close=[4.0, 8.0]
```

This checks numerical and [ownership](../glossary.md#term-ownership-meanings) expectations, not speed. A process invocation includes
[JVM](../glossary.md#term-java-virtual-machine-jvm) startup; `Engine.backward` is a one-shot [compile](../glossary.md#compile)/[prepare](../glossary.md#prepare) path; printing and materializing
copy output. None of those costs isolates repeated [kernel](../glossary.md#kernel) execution. Two multiplications are
also far too small to infer an [accelerator](../glossary.md#term-accelerator) benefit. Do not turn this chapter into a benchmark
by timing one shell command and calling the number kernel [latency](../glossary.md#term-latency).

A **kernel** is the concrete numerical routine performing a piece of work. **Latency** is the
time for one measured [operation](../glossary.md#operation), so its measurement must say where timing starts and ends.
**[Warmup](../glossary.md#term-warmup)** runs work before measurement so startup and [compilation](../glossary.md#compiler-engine-and-execution-stage-names) effects are not mistaken for
steady-state execution. **Materializing** results adds a copy. A correctness baseline establishes
expected answers; a benchmark measures a stated workload under stated conditions.

## Choose the next question

| Your question | Read next | What to keep explicit |
|---|---|---|
| Which packages explain these calls? | [Whole-framework explanation](../explanation/framework.md) and [package atlas](../explanation/packages/index.md) | Public entry points versus unsupported internal implementation packages |
| Which current [CPU](../glossary.md#term-cpu) operations/[dtypes](../glossary.md#data-type--datatype)/[layouts](../glossary.md#layout) execute and differentiate? | [CPU backend](../backend-guide/cpu-backend.md) and [production support](../backend-guide/production-support.md) | Forward support and actual backward support are separate checks |
| How do I compose [Metal](../glossary.md#term-metal) on supported Apple hardware? | [Metal registration and composition](../backend-guide/metal-backend.md#registration-and-composition) | An explicit integration, platform/native prerequisites and [owner](../glossary.md#term-owner) lifetimes, not `Engine.standard()` |
| How do I configure native CPU acceleration? | [CPU backend](../backend-guide/cpu-backend.md), including its explicit [oneDNN](../glossary.md#term-onednn) composition and [OpenBLAS](../glossary.md#term-openblas) sections | Native installation/configuration and route [admission](../glossary.md#term-admission); oneDNN and OpenBLAS remain CPU routes |
| How should I measure setup versus repeated work? | [Benchmarking](../developer-guide/benchmarking.md)<span data-glossary-navigation="true"> ([Benchmarking glossary](../glossary.md#term-benchmarking))</span> | Workload, [warmup](../glossary.md#term-warmup), measurement boundary, [materialization](../glossary.md#materialization), cleanup and reported environment |
| How do I control public execution directly? | [Prepared-graph guide](../user-guide/running-models.md) and [API reference](../reference/index.md)<span data-glossary-navigation="true"> ([API glossary](../glossary.md#term-application-programming-interface-api))</span> | [Compile](../glossary.md#compile) owner, prepared owner, per-[run](../glossary.md#term-run) results and [detached](../glossary.md#detached-model-results) copies |

Follow the relevant guide's actual prerequisites before changing composition. Keep the toy
correctness checks as a baseline, but consult current support for your real [graph](../glossary.md#graph) rather than
extrapolating from one [FLOAT32](../glossary.md#term-float32) [matrix multiplication](../glossary.md#matrix-multiplication--matmul). [Backend](../glossary.md#backend)/native [modules](../glossary.md#term-module) not used by this
program are explained in the atlas; absence from this journey is not evidence of missing support.

## Useful variation and relevant error

Run the [gradient](../glossary.md#term-gradient) boundary independently before evaluating another supported configuration:

```bash
# Forward support alone does not establish support for the selected derivative.
java --enable-native-access=ALL-UNNAMED -cp "$LEARNING_CP" \
  io.github.pho001.synaptik.examples.LearningJourney gradients
```

Its expected [objective](../glossary.md#term-objective) is `[40.0]`, [weight](../glossary.md#term-weight) gradient `[-40.0]`, unchanged stored weight `0.0`,
and the [detached](../glossary.md#detached-model-results) gradient remains `[-40.0]` after [Engine](../glossary.md#term-engine) close. See the
[full expected lines and derivative interpretation](loss-and-gradients.md#run-and-interpret).
This separates correct forward [prediction](../glossary.md#term-prediction) from actual differentiation; it is not a broad
[backend](../glossary.md#backend) [conformance](../glossary.md#term-conformance) suite.

A backend/configuration that cannot [prepare](../glossary.md#prepare) the actual [graph](../glossary.md#graph) must fail [admission](../glossary.md#term-admission); do not
silently substitute a random [model](../glossary.md#term-model), change [dtype](../glossary.md#data-type--datatype)/[loss](../glossary.md#loss), remove a [derivative](../glossary.md#term-derivative) check or assume
[runtime](../glossary.md#term-runtime) will migrate the work. Preserve the diagnostic and compare dtype/[shape](../glossary.md#shape)/[layout](../glossary.md#layout)/native
prerequisites with the selected guide. An unavailable optional [BLAS](../glossary.md#term-blas) library is not the same
problem as a malformed `[1,2]` input, whose real rejection is already covered by `invalid-shape`.

## Source and boundaries

The baseline's complete
[Java source](../examples/learning-journey/src/main/java/io/github/pho001/synaptik/examples/LearningJourney.java)
uses public [`Engine`](../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java)<span data-glossary-navigation="true"> ([Engine glossary](../glossary.md#term-engine))</span>
and owned [`Model`](../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Model.java)<span data-glossary-navigation="true"> ([Model glossary](../glossary.md#term-model))</span>.
[Explicit Engine composition](../architecture/contracts/runtime-prepare-engine.md#public-explicit-composition)
is the authoritative boundary. This chapter changes no [backend](../glossary.md#backend) policy, [API](../glossary.md#term-application-programming-interface-api) or performance
[qualification](../glossary.md#term-qualification) and claims no execution evidence beyond the expectations supplied for comparison.

Return to the [journey map](index.md) or keep the
[complete expected transcript](../examples/learning-journey/expected-stdout.txt) with your [run](../glossary.md#term-run)
when comparing the candidate in your own environment.
