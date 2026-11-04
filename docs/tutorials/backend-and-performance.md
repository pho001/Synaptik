# 7. Choose backend and performance work deliberately

## Outcome and prerequisites

Finish with a reproducible correctness baseline and choose the next focused guide without
confusing a new backend with a new mathematical model. Complete
[setup](../examples/README.md#build-the-complete-project) and
[reusable execution](reusable-execution.md). Commands run in the example project with JDK 26
and `LEARNING_CP`; no native accelerator is needed for this chapter's baseline.

## Keep meaning separate from the route

A **backend** provides execution for supported operation occurrences. A **route** is a concrete
implementation choice within one backend. CPU scalar/generated JVM/Vector API/native library
choices are not separate mathematical models. The Model operation families define the meaning;
backend capability still depends on actual dtype, shape, layout and requested derivatives.
A graph that can be described or saved is not thereby executable on every backend.

The canonical program uses `Engine.standard()`, which is CPU-only. Adding native-access
permission neither selects Metal nor promises OpenBLAS use. The standard CPU owner can probe
optional OpenBLAS even for a graph whose actual work uses portable Java. The project requires
no Vector API module; explicit Vector API routes have separate setup requirements.

The tiny workload remains FLOAT32 input `[2,1]`, private weight `[1,1]`, no bias, SUM loss,
`Sgd(0.05)`, with the same `40 -> -40 -> w=2` arithmetic. Nothing here changes the candidate,
adds a dependency, rewrites numerical semantics or enables a global backend switch.

## Run the baseline, not a timing claim

```bash
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

This checks numerical and ownership expectations, not speed. A process invocation includes
JVM startup; `Engine.backward` is a one-shot compile/prepare path; printing and materializing
copy output. None of those costs isolates repeated kernel execution. Two multiplications are
also far too small to infer an accelerator benefit. Do not turn this chapter into a benchmark
by timing one shell command and calling the number kernel latency.

## Choose the next question

| Your question | Read next | What to keep explicit |
|---|---|---|
| Which packages explain these calls? | [Whole-framework explanation](../explanation/framework.md) and [package atlas](../explanation/packages/index.md) | Public entry points versus unsupported internal implementation packages |
| Which current CPU operations/dtypes/layouts execute and differentiate? | [CPU backend](../backend-guide/cpu-backend.md) and [production support](../backend-guide/production-support.md) | Forward support and actual backward support are separate checks |
| How do I compose Metal on supported Apple hardware? | [Metal registration and composition](../backend-guide/metal-backend.md#registration-and-composition) | An explicit integration, platform/native prerequisites and owner lifetimes, not `Engine.standard()` |
| How do I configure native CPU acceleration? | [CPU backend](../backend-guide/cpu-backend.md), including its explicit oneDNN composition and OpenBLAS sections | Native installation/configuration and route admission; oneDNN and OpenBLAS remain CPU routes |
| How should I measure setup versus repeated work? | [Benchmarking](../developer-guide/benchmarking.md) | Workload, warmup, measurement boundary, materialization, cleanup and reported environment |
| How do I control public execution directly? | [Prepared-graph guide](../user-guide/running-models.md) and [API reference](../reference/index.md) | Compile owner, prepared owner, per-run results and detached copies |

Follow the relevant guide's actual prerequisites before changing composition. Keep the toy
correctness checks as a baseline, but consult current support for your real graph rather than
extrapolating from one FLOAT32 matrix multiplication. Backend/native modules not used by this
program are explained in the atlas; absence from this journey is not evidence of missing support.

## Useful variation and relevant error

Run the gradient boundary independently before evaluating another supported configuration:

```bash
java --enable-native-access=ALL-UNNAMED -cp "$LEARNING_CP" \
  io.github.pho001.synaptik.examples.LearningJourney gradients
```

Its expected objective is `[40.0]`, weight gradient `[-40.0]`, unchanged stored weight `0.0`,
and the detached gradient remains `[-40.0]` after Engine close. See the
[full expected lines and derivative interpretation](loss-and-gradients.md#run-and-interpret).
This separates correct forward prediction from actual differentiation; it is not a broad
backend conformance suite.

A backend/configuration that cannot prepare the actual graph must fail admission; do not
silently substitute a random model, change dtype/loss, remove a derivative check or assume
runtime will migrate the work. Preserve the diagnostic and compare dtype/shape/layout/native
prerequisites with the selected guide. An unavailable optional BLAS library is not the same
problem as a malformed `[1,2]` input, whose real rejection is already covered by `invalid-shape`.

## Source and boundaries

The baseline's complete
[Java source](../examples/learning-journey/src/main/java/io/github/pho001/synaptik/examples/LearningJourney.java)
uses public [`Engine`](../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java)
and owned [`Model`](../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Model.java).
[Explicit Engine composition](../architecture/contracts/runtime-prepare-engine.md#public-explicit-composition)
is the authoritative boundary. This chapter changes no backend policy, API or performance
qualification and claims no execution evidence beyond the expectations supplied for comparison.

Return to the [journey map](index.md) or keep the
[complete expected transcript](../examples/learning-journey/expected-stdout.txt) with your run
when comparing the candidate in your own environment.
