# Ordinary and advanced compile entry points

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="current-ordinary-and-advanced-engine-compile-boundaries"></a>
## Current ordinary and advanced Engine compile boundaries

The ordinary `Engine` exposes these two compile forms:

```java
CompiledGraph forward = engine.compile(List.of(output));
CompiledGraph differentiated = engine.compile(
        List.of(output), List.of(cotangentSeed), List.of(target));
```

The first form selects `FORWARD_ONLY`. The second selects `FORWARD_AND_BACKWARD` and constructs
exactly one functional-gradient stage with one explicit non-null seed per forward output,
identity-unique non-empty targets, `createGraph == false`, and disconnected-target policy
`ERROR`. Seeds may repeat and may themselves be expressions. Neither form infers targets, creates
an implicit scalar seed, adds a second derivative stage, selects `TRAINING_STEP`, or supplies
no-argument backward.

Both forms use standard optional graph optimization, unconstrained backend intent, and neutral
partition scoring. They return a fresh immutable `CompiledGraph` owned by the exact Engine. Its
`inputs()` list contains only final caller-bindable `TensorId` and descriptor metadata in Compiler
binding order. Repeated use of one leaf yields one input; identity-distinct equal-descriptor leaves
remain distinct; compiler constants are omitted; surviving explicit seed leaves remain ordinary
inputs. The handle retains no caller Tensor, request list, provenance, or host-storage reference,
and compile does not read `Tensor.hostStorage()`.

The ordinary handle deliberately exposes no `CompileArtifacts`, graph-local `ValueId`, partition,
diagnostic, or advanced handle. It is not closeable. Its metadata remains readable after Engine
closure, although the closed Engine rejects preparation and all other new work. Completed Engine
task 0004 adds materialization at the later run-result boundary without changing compilation.
Completed task 0005A replaces task 0005's explicit-input one-shot methods with four ordinary
`compute(...)` overloads. Each transiently inventories reachable provenance-free Model Tensor
leaves, invokes this same fixed forward-only compilation afresh, and uses final
`CompiledGraph.inputs()` metadata as the sole input-membership and ordering authority before
prepare, run, complete publication/aggregate-byte preflight, ordered materialization, and cleanup.
This is Model expression-provenance traversal, not Compiler IR traversal or Engine liveness
reconstruction. It adds no compile cache, reusable hidden handle, new compile option, or Compiler
API.

`Engine.backward(objective, targets, maximumTotalBytes)` is the current narrower one-shot
Compiler mapping. It inventories reachable leaves, then issues exactly one
`FORWARD_AND_BACKWARD` stage for the singleton objective with one absent seed,
`createGraph == false`, and `DisconnectedPolicy.ERROR`. Compiler interprets that absent seed as
exact typed positive one only when the objective is scalar, floating, and gradient-eligible; it
also remains authoritative for target membership, differentiable connectivity, supported
operations, and final bindable inputs. Targets are explicit, ordered, non-empty, and unique by
exact Tensor identity; accepted targets may be leaves, intermediates, or the objective. Engine
does not infer targets, inputs, liveness, or gradient semantics and retains no Tensor state.

The call freshly compiles and prepares each time. Explicit seeds, multiple forward outputs,
reusable execution, and selective publication materialization remain available through the
ordinary three-list compile lifecycle. `ZERO`, a second stage, higher-order construction, and the
complete request policy remain available through `AdvancedEngine.compile(...)` with an explicit
`FunctionalGradientRequest`.

`AdvancedEngine.compile(...)` is the current public consumer of `GraphCompilationPort`. It
requires a non-empty ordered output list and receives `BackendIntent`, `CompileMode`,
`GraphOptimizationConfig`, `PartitionScoringConfig`, and an optional
`FunctionalGradientRequest` directly. It snapshots the output list, retains no caller
collection, and returns a fresh `AdvancedCompiledGraph` bound to the exact open Engine that
compiled it. The handle is immutable and opaque: callers cannot inspect its model, partitions,
publication plan, constants, or diagnostics, and it owns no independently closeable resource.

`AdvancedEngine` supplies one CPU capability provider to the compiler. Ordinary `Engine` supplies
the frozen provider and availability inventory captured by its builder. Neither path discovers
backends, builds a public reusable capability matrix, creates `CompileConfig`, or promises that
every successfully compiled artifact is preparable. Preparation is a separate boundary: ordinary
Engine requires a non-empty plan with one exact registered owner before applying that backend's
constraints; Advanced Engine retains its narrower CPU restriction. Compile success is therefore
not an execution promise.


---

[Previous topic](index.md) · [Next topic](capture.md)
