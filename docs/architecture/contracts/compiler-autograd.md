# Compiler and automatic-differentiation contract

> This scoped contract is incorporated by the [authoritative architecture root](../../../ARCHITECTURE.md).
> It is normative only within the scope stated below and has no independent authority. Root global
> invariants and dependency rules apply everywhere and take precedence. Any overlap, contradiction,
> missing applicable scope, or ambiguity requires an explicit architecture update; do not choose
> between contracts silently.

## Scope

This contract owns the detailed partition-scoring inputs and output boundary, Compiler
responsibilities, compile artifacts, training-graph construction, compiler-owned automatic
differentiation, and the compile lifecycle. General Planning module responsibilities remain with
the foundational contract. This file does not own physical resources, backend lowering, prepared
execution, or optimizer orchestration.

## Partition scoring

Planning includes backend-neutral partition scoring.

Partition scoring may use compile-time information only, such as:

- graph metadata
- op kind
- data type
- shape
- estimated element count
- estimated byte size
- backend capabilities
- backend intent
- graph phase
- producer/consumer ownership candidates
- logical materialization estimates
- transfer estimates
- boundary penalties
- accelerator bonuses
- small-region penalties
- backend-neutral planning cost profiles

Partition scoring must not use:

- current runtime residency
- current device buffers
- concrete kernel classes
- concrete MPSGraph executables
- concrete CUDA kernels
- concrete OpenBLAS calls
- current `RunState`
- physical buffer addresses
- prepared executables
- backend-local workload-cache entries, route configurations, or other model-autotuning values

Partition scoring decides backend ownership at node or segment level before maximal same-owner partitioning.

The output of scoring is ownership, not executable implementation.

Backend-specific lowering and kernel selection belong to backend prepare.

A planning cost model is not a model-autotuning parameter set. It may estimate ownership
cost from backend-neutral graph and transfer facts, but it must not interpret vector, thread,
tile, route, kernel, or other backend implementation vocabulary.

## `modules/compiler`

`modules/compiler` owns graph compilation.

Allowed:

- graph capture
- fail-closed autograd preflight over Tensor expression occurrences
- compiler-owned gradient-rule dispatch expressed through public Tensor operations
- per-compile identity-based gradient-contribution accumulation
- topological sorting
- producer/use indexing
- canonicalization
- shape inference
- data type inference
- validation
- dead-code elimination
- common subexpression elimination
- constant folding
- algebraic simplification
- autograd expansion
- backward graph construction
- combined forward/backward graph optimization
- publication binding
- partition planning orchestration
- logical memory planning orchestration
- compile diagnostics
- `CompileArtifacts`

Forbidden:

- physical buffers
- `PreparedSchedule`
- `PreparedExecution`
- backend executables
- concrete kernel selection
- backend-specific lowering
- runtime workspace state
- runtime residency
- concrete backend dependencies

The compiler produces immutable compile-time artifacts.

It must not construct runtime execution units.

## Compile artifacts

`CompileArtifacts` is one immutable compile-time recipe containing exactly these eight semantic
components, in current record order:

1. the exact graph-scope `CompileMode`;
2. the exact final `CompiledGraphModel`;
3. immutable graph-order `List<PlannedPartition>` membership for the maximal partitions;
4. the `LogicalMemoryPlan` derived from that graph and partition list;
5. the `PublicationPlan` for the exact final graph;
6. the complete graph-input source classification in `CompileConstantPlan`;
7. successful-compile deferred diagnostics in `CompileDiagnostics`; and
8. the exact-graph derivative-order metadata in `DerivativeGraphMetadata`.

These components are logical compile output, not physical, prepared, backend-executable, or
runtime state.

`CompileArtifacts` must not contain:

- physical buffers
- prepared executables
- backend executable objects
- concrete kernel routes
- runtime workspaces
- runtime residency state
- mutable run state

## Training graph model

In backward-capable compile modes, the compiler constructs backward Tensor expressions before
graph capture. It then captures the forward outputs and requested gradient roots together exactly
once into a combined forward + backward graph.

The combined graph is immutable compile-time graph state. Public Tensors remain expression model
state and never become graph nodes or values.

Runtime may expose separate forward and backward schedules, or a single training-step schedule, depending on prepare-time decisions.

Backends must not implement global autograd.

Backends execute prepared regions only.

Optimizer updates are either:

- backend-agnostic runtime/training steps, or
- graph operations generated by the training extension and lowered by backend prepare

Training must not depend on concrete backend modules.

## Compiler-owned automatic differentiation

The compiler owns reverse-mode automatic differentiation. The required compile-time flow is:

```text
original forward Tensor expression DAG
  -> fail-closed backward-reachable operation/attribute/policy preflight
  -> compiler-owned reverse traversal and Tensor gradient-expression construction
  -> combined forward + gradient Tensor expression DAG
  -> one phase-aware graph capture
  -> immutable combined compiler graph
  -> inference and validation
  -> canonicalization and exact combined-graph optimization
  -> final validation
  -> publication and planning
```

`FORWARD_ONLY` skips autograd. `FORWARD_AND_BACKWARD` and `TRAINING_STEP` construct the combined
Tensor expression DAG before capture. `TRAINING_STEP` does not add optimizer updates until a later
architecture decision and implementation task explicitly introduces them.

Gradient-rule dispatch belongs to named compiler components such as
`ElementwiseGradientRules`. A rule constructs formulas only by calling existing public Tensor
operations such as `mul`, `add`, `sumToShape`, and `transpose`. The compiler must not add
model-owned derivative rules, a second low-level algebra language, direct generated graph-node
formula construction, a public gradient registry or facade, or Tensor gradient/backward state.

For one compile request, the compiler may use identity-based maps from exact Tensor objects to
ordered gradient contributions and accumulated gradients. This is ephemeral reverse-accumulation
bookkeeping, not public Tensor state and not another graph representation. Multiple contributions
are accumulated with ordinary `Tensor.add`.

Before constructing any backward expression, the compiler must inventory every
backward-reachable operation occurrence and its exact attributes and derivative policies. Any
unsupported or ambiguous occurrence fails closed. Full inference and validation still occur only
after the one combined capture, so a later construction, capture, inference, validation, or
optimization failure may consume temporary model-level `TensorId` values. Tensor IDs are opaque
and are never rolled back or reused.

Seeds and derivative constants are storage-free Tensor leaves or expressions explicitly
registered as compile-time constant splats. Tensor storage, labels, descriptor shape, factory
history, or provenance absence must never silently imply a compile-time constant.

Phase-aware capture receives the ordered forward outputs, requested gradient roots and their
target roles, the identity set of original forward producers, and explicit constant facts. It
assigns every `NodeId` and `ValueId` exactly once. Nodes whose producer identity belongs to the
original forward set have phase `FORWARD`; generated derivative producers have phase `BACKWARD`.
Per-node `GraphPhase` remains authoritative and must not be replaced by only a positional
backward-start index.

Distinct differentiation targets may legitimately resolve to the same captured gradient
`ValueId`. Gradient result roles map each target independently, while the graph's public output
boundary contains each distinct gradient value once. The compiler must not create identity nodes
solely to make those result values distinct.

Optimization operates on the immutable combined graph. The exact arithmetic rules, constant
folding, dead-code elimination, and common-subexpression elimination already established by
compiler tasks 0003, 0003A, and 0003B must be reassessed for both phases and applied only where
their existing semantic guards remain valid. Common-subexpression elimination is phase-local
unless a later architecture update and proof establishes a broader safe rule. Every changed
candidate is revalidated through the compiler's inference-and-validation boundary. This contract
does not authorize new algebraic rewrites.

Generated gradient formulas are ordinary differentiable Tensor expressions. A functional request
contains exactly one or two ordered reverse-mode stages. Every stage has non-empty ordered output
references, an output-aligned list of optional cotangent seeds, and a non-empty ordered target
list whose exact Tensor references are unique and belong to the complete original forward
inventory. Stage one selects exact Tensors from the requested forward-output boundary through
forward references. Stage two, when present, selects only generated first-stage gradients through
a stage-one target index.

An absent seed means an exact typed positive one only for a scalar, floating,
gradient-eligible output. A present seed must have the output's exact Shape and floating data type
and must not itself request gradients. `DisconnectedPolicy.ERROR` rejects a valid target that has
no differentiable route from the selected outputs. `DisconnectedPolicy.ZERO` instead returns an
ordinary exact typed zero expression; several target roles may share that same final graph value.

A one-stage request requires `createGraph == false`. A two-stage request requires
`createGraph == true` for stage one and `false` for stage two. `createGraph` retains first-stage
formulas only for the immediate second stage inside the same compile. It does not establish an
arbitrary nested or persistent derivative chain. A third stage and derivative order greater than
two are unsupported.

The compiler captures the original forward outputs and all requested stage-one and stage-two
gradient roots together once. `DerivativeGraphMetadata` augments, but does not replace,
`GraphPhase`: original forward producers have derivative order zero, producers first owned by the
first reverse stage have order one, and producers first owned by the second reverse stage have
order two. Both derivative orders one and two remain `BACKWARD` graph phase.

Gradient publication bindings exist only for derivative orders one and two. They are ordered by
derivative order and then stage-local target index, retain the target identity and final gradient
value, and remain target-distinct when several targets share one captured value. The graph output
boundary starts with the ordered forward values and then lists each previously unseen gradient
value once in binding order.

The bounded lifecycle creates no persistent or runtime tape, mutable Tensor gradient state,
`Tensor.backward()`, model-owned derivative rules, backend-owned global autograd, or second
gradient algebra. The ordinary `Engine.backward(...)` convenience is narrower than the complete
functional request boundary: it always requests one first-order stage for one scalar objective,
uses an absent seed, sets `createGraph == false`, and uses `DisconnectedPolicy.ERROR`. The full
bounded request surface is available through the advanced compile boundary.


## Compile lifecycle

Compile lifecycle:

```text
Tensor forward outputs
  -> if backward is requested:
     - fail-closed autograd preflight
     - reverse accumulation through public Tensor operations
     - combined forward + gradient Tensor expression DAG
  -> one phase-aware GraphCapture
  -> topological sort
  -> producer/use index
  -> shape and data type inference
  -> validation
  -> canonicalization
  -> combined-graph optimization
     - DCE
     - phase-local CSE
     - constant folding
     - algebraic simplification
  -> final validation
  -> publication binding
  -> backend intent propagation
  -> capability analysis
  -> partition scoring
  -> ownership decision
  -> maximal same-owner partitioning
  -> logical memory/materialization requirements
  -> CompileArtifacts
```

Compile must not create:

- prepared schedules
- prepared units
- prepared executions
- backend executables
- physical buffers
- kernel routes
- runtime workspaces
- backend-specific DAGs
