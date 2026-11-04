# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0009: [Compiler](../../glossary.md#term-compiler)-owned [pre-capture Tensor-expression autograd](../../glossary.md#pre-capture-tensor-expression-autograd)


## Editorial reading context

The application describes a forward computation with Tensor operations. Compiler then calls ordinary Tensor methods to construct derivative expressions before capturing the combined graph. A leaf has no producing expression and may be storage-free, as with an explicitly registered derivative seed. Leaf status alone neither proves constant values nor transfers storage ownership.

Use the [autograd seeds, capture and saved outputs](../../architecture/training-graph.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

### Mathematical reading aid: one derivative contribution

For differentiable real-valued functions $y=f(x)$ and scalar objective $L(y)$, let
$g=\partial L/\partial y$ be the incoming sensitivity. The chain rule gives
$\partial L/\partial x=g f'(x)$. For $f(x)=x^2$, $x=3$ and $g=2$, the
contribution is $2\cdot 2\cdot3=12$. This explains a contribution, not an
optimizer update. Compiler expresses it with Tensor operations; their actual rounded
floating evaluation and special-value rules, not real-number calculus alone, govern execution.

## Status

Accepted — 2026-07-26

## Context

The previous [compiler](../../glossary.md#term-compiler) plan captured and optimized an [immutable](../../glossary.md#term-immutable) [forward graph](../../glossary.md#forward-graph), reconstructed
placeholder [Tensors](../../glossary.md#tensor) for [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot) values, expressed [derivative](../../glossary.md#term-derivative) formulas over those placeholders,
and captured only the backward fragment. That design introduced a conversion boundary between
public Tensor expressions and captured [graph](../../glossary.md#graph) values, obscured hidden auxiliary output [identity](../../glossary.md#term-identity-meanings),
and split graph-local identity assignment across forward and backward construction.

The current [model](../../glossary.md#term-model) already provides a [backend](../../glossary.md#backend)-independent Tensor expression vocabulary capable of
expressing adjoints. It also represents one [producer](../../glossary.md#term-producer) with ordered multi-output slots, including
the [dropout](../../glossary.md#dropout--inverted-dropout) [mask](../../glossary.md#term-boolean-mask) and [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../glossary.md#term-normalization) saved statistics required by backward formulas. The
remaining model gap is that a producer retains [descriptors](../../glossary.md#term-descriptor), but not the [canonical](../../glossary.md#term-canonical-form) Tensor wrapper
for every slot.

## Decision drivers

- keep [derivative](../../glossary.md#term-derivative)-rule [ownership](../../glossary.md#term-ownership-meanings) in the [compiler](../../glossary.md#term-compiler);
- use one existing [Tensor](../../glossary.md#tensor) [operation](../../glossary.md#operation) vocabulary rather than a second algebra language;
- assign [`NodeId`](../../glossary.md#nodeid) and [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot) once for the combined computation;
- preserve exact same-[occurrence](../../glossary.md#term-occurrence) hidden outputs without reconstructing wrappers;
- fail before partial backward construction when an operation or policy is unsupported;
- optimize and validate the complete [immutable](../../glossary.md#term-immutable) [graph](../../glossary.md#graph);
- keep Tensor free of [gradient](../../glossary.md#term-gradient)/backward [lifecycle](../../glossary.md#lifecycle) state; and
- preserve an explicit path to higher derivatives.

## Options considered

### Captured-[IR](../../glossary.md#term-intermediate-representation-ir) [autograd](../../glossary.md#autograd) with placeholder [Tensors](../../glossary.md#tensor)

[Capture](../../glossary.md#term-capture-meanings) and optimize [forward graph](../../glossary.md#forward-graph) state first, create placeholder Tensors for forward
[`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot) values, stage [gradients](../../glossary.md#term-gradient), and translate the backward fragment back into [graph](../../glossary.md#graph) state. This
retains the completed forward pipeline directly, but creates a placeholder/[binding](../../glossary.md#term-binding) subsystem,
requires two capture-like [identity](../../glossary.md#term-identity-meanings) domains, and cannot obtain exact hidden output wrappers from
the current [producer](../../glossary.md#term-producer) contract.

### Direct [immutable](../../glossary.md#term-immutable)-[graph](../../glossary.md#graph) algebra

Build [derivative](../../glossary.md#term-derivative) [`Operation`](../../glossary.md#operation), `CompiledNode`, and `GraphValue` objects directly through a new
[compiler](../../glossary.md#term-compiler) algebra. This avoids temporary [Tensors](../../glossary.md#tensor), but duplicates the public operation construction
vocabulary and its [Shape](../../glossary.md#shape), [descriptor](../../glossary.md#term-descriptor), [provenance](../../glossary.md#provenance), and [validation](../../glossary.md#term-validation) behavior.

### [Compiler](../../glossary.md#term-compiler)-owned [pre-capture Tensor-expression autograd](../../glossary.md#pre-capture-tensor-expression-autograd)

[Preflight](../../glossary.md#term-preflight) the original forward [Tensor](../../glossary.md#tensor) expression, construct [gradients](../../glossary.md#term-gradient) with ordinary public Tensor
operations, and [capture](../../glossary.md#term-capture-meanings) the forward outputs and gradient roots together once. Retain exact
[canonical](../../glossary.md#term-canonical-form) output wrappers on each [producer](../../glossary.md#term-producer) so hidden saved outputs participate by [identity](../../glossary.md#term-identity-meanings).

## Decision

Synaptik adopts [compiler](../../glossary.md#term-compiler)-owned [pre-capture Tensor-expression autograd](../../glossary.md#pre-capture-tensor-expression-autograd).

For `FORWARD_AND_BACKWARD` and the initial `TRAINING_STEP`, the compiler inventories every
backward-reachable [producer](../../glossary.md#term-producer) [occurrence](../../glossary.md#term-occurrence) and exact [derivative](../../glossary.md#term-derivative)-relevant attributes or policies. It
fails closed before constructing backward expressions if any required rule is absent or
ambiguous. `FORWARD_ONLY` skips this work.

Named compiler components such as `ElementwiseGradientRules` [dispatch](../../glossary.md#term-dispatch) formulas and call only
ordinary public [Tensor](../../glossary.md#tensor) methods. One [compile](../../glossary.md#compile) request may use [identity](../../glossary.md#term-identity-meanings)-based maps to collect
contributions and accumulated [gradients](../../glossary.md#term-gradient); accumulation uses `Tensor.add`. These maps are
short-lived compiler bookkeeping, not Tensor state or a [graph](../../glossary.md#graph) representation.

[Seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings) and derivative [constants](../../glossary.md#term-constant) are storage-free Tensor [leaves](../../glossary.md#term-leaf) or expressions registered
explicitly as compile-time constant splats. Storage, factory history, and [provenance](../../glossary.md#provenance) absence never
imply constant status.

One phase-aware [capture](../../glossary.md#term-capture-meanings) receives forward outputs, gradient roots and [target](../../glossary.md#term-target) roles, the original
forward-producer identity set, and explicit constant facts. It assigns graph-local IDs once and
classifies every captured producer as `FORWARD` or `BACKWARD`. Result roles may map multiple
targets to one gradient [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot); the output boundary retains each distinct gradient value once.

[Inference](../../glossary.md#term-inference) and [validation](../../glossary.md#term-validation) follow capture. [Canonicalization](../../glossary.md#occurrence-identity-and-canonical-form), the exact rewrites and folding already
selected by compiler tasks 0003A and 0003B, [dead-code elimination](../../glossary.md#term-dead-code-elimination-dce), and phase-local
[common-subexpression elimination](../../glossary.md#term-common-subexpression-elimination-cse) then operate on the [immutable](../../glossary.md#term-immutable) combined graph where their
existing guards are proved safe. Changed candidates are revalidated through Compiler 0002. No new
rewrite is implied by this [ADR](../../glossary.md#term-architecture-decision-record-adr).

[`TensorProducer`](../../glossary.md#producer--tensorproducer) must retain and return the [canonical](../../glossary.md#term-canonical-form) exact Tensor wrapper for every output slot.
`TensorFactory` atomically constructs the shared producer, wrappers, and indexed provenance.
This introduces a collectable object cycle, not [mutable](../../glossary.md#term-mutable) [ownership](../../glossary.md#term-ownership-meanings): final fields and unpublished
construction provide safe [publication](../../glossary.md#term-publication), and ordinary [garbage collection](../../glossary.md#term-garbage-collection-gc) reclaims the whole
unreachable occurrence. No wrapper is reconstructed.

## Rationale

This option keeps one semantic construction language and one [graph capture](../../glossary.md#capture-graph-ports-and-reconstruction). Exact [Tensor](../../glossary.md#tensor) [identity](../../glossary.md#term-reference-identity)
is available when reverse traversal needs a saved auxiliary, while [graph](../../glossary.md#graph)-local [identity](../../glossary.md#term-identity-meanings) remains a
[compiler](../../glossary.md#term-compiler) concern. Compiler [ownership](../../glossary.md#term-ownership-meanings) is explicit without adding public registries, [model](../../glossary.md#term-model)
[derivative](../../glossary.md#term-derivative) rules, ThreadLocal [scope](../../glossary.md#term-scope), or [mutable](../../glossary.md#term-mutable) Tensor [gradients](../../glossary.md#term-gradient).

[Preflight](../../glossary.md#term-preflight) avoids predictable partial backward construction, while post-[capture](../../glossary.md#term-capture-meanings) [inference](../../glossary.md#term-inference) remains
the authoritative semantic check. Consequently, later failures may consume temporary Tensor IDs;
that cost is compatible with their existing [opaque](../../glossary.md#term-opaque), monotonic, non-reusable contract.

## Consequences

### Positive

- forward and backward [nodes](../../glossary.md#node) receive [graph](../../glossary.md#graph)-local [identities](../../glossary.md#term-identity-meanings) once;
- all optimization and logical [planning](../../glossary.md#term-planning) can see one [immutable](../../glossary.md#term-immutable) combined graph;
- hidden same-[occurrence](../../glossary.md#term-occurrence) outputs are exact [Tensor](../../glossary.md#tensor) expressions available to [compiler](../../glossary.md#term-compiler) rules;
- generated [gradients](../../glossary.md#term-gradient) remain ordinary [differentiable](../../glossary.md#derivative-and-chain-rule) Tensor expressions; and
- public Tensor methods and ergonomic multi-output result carriers remain unchanged.

### Negative and risks

- the [model](../../glossary.md#term-model) [producer](../../glossary.md#term-producer) now retains its result wrappers, creating an intentional object cycle;
- [compiler](../../glossary.md#term-compiler) task 0004 must adapt completed forward-only orchestration for combined phases;
- [fail-closed](../../glossary.md#term-fail-closed) [preflight](../../glossary.md#term-preflight) and later full [validation](../../glossary.md#term-validation) are separate checks; and
- failed [compilation](../../glossary.md#compiler-engine-and-execution-stage-names) can consume temporary [Tensor](../../glossary.md#tensor) IDs.

### Migration, testing, and follow-up

[Model](../../glossary.md#term-model) task 0025 is the next implementation frontier and adds [canonical](../../glossary.md#term-canonical-form) [producer](../../glossary.md#term-producer) outputs with
focused [identity](../../glossary.md#term-identity-meanings), construction, [publication](../../glossary.md#term-publication), ID-side-effect, and hidden-output tests. [Compiler](../../glossary.md#term-compiler)
task 0004 remains Draft until that prerequisite is complete. It must remove the obsolete
placeholder/[`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot) conversion design and implement the one-[capture](../../glossary.md#term-capture-meanings) pipeline.

Higher [derivatives](../../glossary.md#term-derivative) remain future. A later task must define create-[graph](../../glossary.md#graph) or derivative-order
[lifecycle](../../glossary.md#lifecycle), cover every [operation](../../glossary.md#operation) used in [gradient](../../glossary.md#term-gradient) formulas, and represent derivative order
alongside [node](../../glossary.md#node) phase. It must not add `Tensor.gradient`, `Tensor.backward`, [mutable](../../glossary.md#term-mutable) gradient
fields, or hidden thread-local state.

This decision changes no [module](../../glossary.md#term-module) [dependency direction](../../glossary.md#java-platform-and-build-tools). Existing architecture tests therefore need
no [dependency](../../glossary.md#term-dependency)-rule update; implementation tasks add focused model/compiler behavior tests.

## Related documentation

- [Architecture contract](../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../glossary.md#architecture-contract))</span>
- [Lifecycle](../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../glossary.md#lifecycle))</span>
- [Training graph](../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../glossary.md#training-graph))</span>
- [Autograd strategy](../notes/autograd-strategy.md)<span data-glossary-navigation="true"> ([Autograd glossary](../../glossary.md#autograd))</span>
- [Model master plan](../../planning/modules/model/master-plan.md)<span data-glossary-navigation="true"> ([Model glossary](../../glossary.md#term-model))</span>
- [Compiler master plan](../../planning/modules/compiler/master-plan.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../glossary.md#term-compiler))</span>
- [Superseded ADR 0005](0005-training-combined-forward-backward-graph.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>
