# Ordinary and advanced [compile](../../glossary.md#compile) entry points

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

An entry point is the [API](../../glossary.md#term-application-programming-interface-api) through which a caller requests [compilation](../../glossary.md#compiler-engine-and-execution-stage-names). Ordinary [Engine](../../glossary.md#term-engine) supplies
a fixed convenient policy; Advanced Engine exposes more explicit request data. A compiled handle
identifies a logical recipe belonging to one Engine, not runnable storage. Preparation must still
check that the selected [backend](../../glossary.md#backend) can turn that recipe into executable work.

## The caller changes at the integration boundary

The application calls `engine.compile(outputs)` with Tensor expressions describing desired
results. Engine then calls `GraphCompilationPort.compile(...)` with those expressions and its
frozen policy/provider inputs; the port delegates to the package-private compiler. Compiler
captures dependencies and returns logical metadata, not the input arrays' current values.
Engine wraps that result in its own owner-bound handle. Compile reads no host-storage bytes and
does not acquire a lease that would keep an application Arena alive. At execution, the application
must separately supply usable stored inputs under [run binding and result lifetimes](../runtime/engine-boundary.md#current-ordinary-engine-boundary).
The [artifact structure](artifacts.md#current-complete-artifact-compilation-and-integration-port)
is useful to integration authors, but ordinary callers do not receive it directly.

<a id="current-ordinary-and-advanced-engine-compile-boundaries"></a>
## Current ordinary and advanced [Engine](../../glossary.md#term-engine) [compile](../../glossary.md#compile) boundaries

The ordinary `Engine` exposes these two compile forms:

```java
// These calls describe forward/derivative work; neither reads stored inputs or updates parameters.
CompiledGraph forward = engine.compile(List.of(output));
CompiledGraph differentiated = engine.compile(
        List.of(output), List.of(cotangentSeed), List.of(target));
```

The first form selects `FORWARD_ONLY`. The second selects `FORWARD_AND_BACKWARD` and constructs
exactly one functional-[gradient](../../glossary.md#term-gradient) stage with one explicit non-null [seed](../../glossary.md#term-seed) per forward output,
[identity](../../glossary.md#term-identity-meanings)-unique non-empty [targets](../../glossary.md#term-target), `createGraph == false`, and disconnected-target policy
`ERROR`. [Seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings) may repeat and may themselves be expressions. Neither form infers targets, creates
an implicit [scalar](../../glossary.md#term-scalar) seed, adds a [second derivative](../../glossary.md#term-second-derivative) stage, selects `TRAINING_STEP`, or supplies
no-[argument](../../glossary.md#term-argument) backward.

Both forms use standard optional [graph optimization](../../glossary.md#forward-graph-optimization), unconstrained [backend intent](../../glossary.md#backend-intent--backendintent), and neutral
[partition](../../glossary.md#partition) scoring. They return a fresh [immutable](../../glossary.md#term-immutable) `CompiledGraph` owned by the exact Engine. Its
`inputs()` list contains only final caller-bindable [`TensorId`](../../glossary.md#tensorid) and [descriptor](../../glossary.md#term-descriptor) [metadata](../../glossary.md#term-metadata) in [Compiler](../../glossary.md#term-compiler)
[binding](../../glossary.md#term-binding) order. Repeated use of one [leaf](../../glossary.md#term-leaf) yields one input; identity-distinct equal-descriptor leaves
remain distinct; compiler [constants](../../glossary.md#term-constant) are omitted; surviving explicit seed leaves remain ordinary
inputs. The handle retains no caller [Tensor](../../glossary.md#tensor), request list, [provenance](../../glossary.md#provenance), or [host](../../glossary.md#term-host)-storage reference,
and compile does not read `Tensor.hostStorage()`.

The ordinary handle deliberately exposes no `CompileArtifacts`, [graph](../../glossary.md#graph)-local [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot), partition,
diagnostic, or advanced handle. It is not closeable. Its metadata remains readable after Engine
closure, although the closed Engine rejects preparation and all other new work. Completed Engine
task 0004 adds [materialization](../../glossary.md#materialization) at the later [run](../../glossary.md#term-run)-result boundary without changing [compilation](../../glossary.md#compiler-engine-and-execution-stage-names).
Completed task 0005A replaces task 0005's explicit-input one-shot methods with four ordinary
`compute(...)` overloads. Each transiently inventories reachable provenance-free [Model](../../glossary.md#term-model) Tensor
leaves, invokes this same fixed forward-only compilation afresh, and uses final
`CompiledGraph.inputs()` metadata as the sole input-membership and ordering authority before
[prepare](../../glossary.md#prepare), run, complete [publication](../../glossary.md#term-publication)/aggregate-byte [preflight](../../glossary.md#term-preflight), ordered materialization, and cleanup.
This is Model expression-provenance traversal, not Compiler [IR](../../glossary.md#term-intermediate-representation-ir) traversal or Engine [liveness](../../glossary.md#term-liveness)
reconstruction. It adds no compile [cache](../../glossary.md#term-cache), reusable hidden handle, new compile option, or Compiler
[API](../../glossary.md#term-application-programming-interface-api).

`Engine.backward(objective, targets, maximumTotalBytes)` is the current narrower one-shot
Compiler mapping. It inventories reachable leaves, then issues exactly one
`FORWARD_AND_BACKWARD` stage for the [singleton](../../glossary.md#term-singleton) [objective](../../glossary.md#term-objective) with one absent seed,
`createGraph == false`, and `DisconnectedPolicy.ERROR`. Compiler interprets that absent seed as
exact typed positive one only when the objective is scalar, floating, and gradient-eligible; it
also remains authoritative for target membership, [differentiable](../../glossary.md#derivative-and-chain-rule) connectivity, supported
operations, and final bindable inputs. Targets are explicit, ordered, non-empty, and unique by
exact Tensor [identity](../../glossary.md#term-reference-identity); accepted targets may be leaves, intermediates, or the objective. Engine
does not infer targets, inputs, liveness, or gradient semantics and retains no Tensor state.

The call freshly compiles and prepares each time. Explicit seeds, multiple forward outputs,
reusable execution, and selective publication materialization remain available through the
ordinary three-list compile [lifecycle](../../glossary.md#lifecycle). `ZERO`, a second stage, higher-order construction, and the
complete request policy remain available through `AdvancedEngine.compile(...)` with an explicit
`FunctionalGradientRequest`.

`AdvancedEngine.compile(...)` is the current public consumer of `GraphCompilationPort`. It
requires a non-empty ordered output list and receives [`BackendIntent`](../../glossary.md#backend-intent--backendintent), [`CompileMode`](../../glossary.md#compile-mode--compilemode),
[`GraphOptimizationConfig`](../../glossary.md#graph-optimization-configuration--graphoptimizationconfig), [`PartitionScoringConfig`](../../glossary.md#partition-scoring-configuration--partitionscoringconfig), and an optional
`FunctionalGradientRequest` directly. It snapshots the output list, retains no caller
collection, and returns a fresh `AdvancedCompiledGraph` bound to the exact open Engine that
compiled it. The handle is immutable and [opaque](../../glossary.md#term-opaque): callers cannot inspect its model, partitions,
[publication plan](../../glossary.md#publication-plan), constants, or diagnostics, and it owns no independently closeable resource.

`AdvancedEngine` supplies one [CPU](../../glossary.md#term-cpu) capability provider to the compiler. Ordinary `Engine` supplies
the frozen provider and availability inventory captured by its builder. Neither path discovers
[backends](../../glossary.md#backend), builds a public reusable capability [matrix](../../glossary.md#term-review-matrix), creates `CompileConfig`, or promises that
every successfully compiled [artifact](../../glossary.md#term-artifact) is preparable. Preparation is a separate boundary: ordinary
Engine requires a non-empty plan with one exact registered [owner](../../glossary.md#term-owner) before applying that backend's
constraints; Advanced Engine retains its narrower CPU restriction. Compile success is therefore
not an execution promise.


---

[Previous topic](index.md) · [Next topic](capture.md)
