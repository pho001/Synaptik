# Task 0003: [Canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) and Forward Optimization

> **Editorial reading key — not part of the historical record.**
>
> GraphCanonicalization changes local numbering and representation; optional dead-code elimination
> removes unneeded work, while common-subexpression elimination shares equivalent occurrences only
> under its guards. These passes consume verified graph descriptions and return revalidated
> descriptions, not computed tensor values. Config permits optional work but cannot turn off
> correctness checks. Requested output identity and multi-output occurrence boundaries explain why
> apparently duplicate work can remain.
>
> Current reading: [exact graph transformations and final logical layouts](../../../../reference/compiler/optimization.md#current-package-private-canonicalization-and-whole-graph-optimization); [hard intent, compile scope and optional optimization](../../../../reference/compiler/configuration.md#current-compile-configuration-inputs).

## Status

Complete

## Goal

Add the smallest complete [compiler](../../../../glossary.md#term-compiler)-owned transformation pipeline over a successful
`ValidatedGraph`:

```text
successful captured-graph validation
  -> mandatory deterministic graph-local canonicalization
  -> task-0002 inference and validation
  -> optional forward dead-code elimination
  -> task-0002 inference and validation when changed
  -> optional exact forward common-subexpression elimination
  -> task-0002 inference and validation when changed
  -> optional forward dead-code elimination cleanup
  -> task-0002 inference and validation when changed
  -> successful canonical ValidatedGraph
```

Mandatory [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) gives every accepted [graph](../../../../glossary.md#graph) candidate one deterministic dense
graph-local [identity](../../../../glossary.md#term-identity-meanings) and storage order without changing its [operation](../../../../glossary.md#operation) occurrences or boundaries.
The optional pipeline is controlled only by the existing [`GraphOptimizationConfig`](../../../../glossary.md#graph-optimization-configuration--graphoptimizationconfig) permission and
contains exactly one bounded sequence whose safety follows from current [immutable](../../../../glossary.md#term-immutable) [model](../../../../glossary.md#term-model)
contracts: forward [dead-code elimination](../../../../glossary.md#term-dead-code-elimination-dce) ([DCE](../../../../glossary.md#compiler-transformations-and-dominance)), exact [common-subexpression elimination](../../../../glossary.md#term-common-subexpression-elimination-cse) ([CSE](../../../../glossary.md#compiler-transformations-and-dominance)) for
internal forward occurrences, and one final forward DCE cleanup. Each transform is invoked once in
that fixed order. The pipeline does not iterate to a fixed point.

The result remains package-private compiler pass state. This task creates no public compiler
[facade](../../../../glossary.md#term-facade), [compile](../../../../glossary.md#compile) aggregate, candidate collection, diagnostics [schema](../../../../glossary.md#term-schema), [backward graph](../../../../glossary.md#backward-graph), [publication
plan](../../../../glossary.md#publication-plan), [planning](../../../../glossary.md#term-planning) orchestration, prepared state, [backend](../../../../glossary.md#backend) behavior, or execution.

Arithmetic and algebraic rewriting remain wholly outside this task. Draft Compiler 0003A receives
that follow-up capability under the current strict operation semantics; task 0003 neither depends
on it nor changes its reviewed canonicalization and one-shot `DCE -> CSE -> DCE` implementation
[scope](../../../../glossary.md#term-scope).

Compile-time [constant](../../../../glossary.md#term-constant) representation and [constant folding](../../../../glossary.md#term-constant-folding) also remain wholly outside this task.
Draft Compiler 0003B follows 0003A and must define a compiler-owned immutable constant fact/ingress
representation plus exact deterministic folding. It must not read [mutable](../../../../glossary.md#term-mutable) public [Tensor](../../../../glossary.md#tensor) [host
storage](../../../../glossary.md#host-storage) as authoritative compile-time data, and every changed graph candidate must be revalidated
through Compiler 0002.

## [Scope](../../../../glossary.md#term-scope)

- Add one package-private stateless pipeline entry point with this exact [shape](../../../../glossary.md#shape):

  ```java
  static ValidatedGraph optimize(
          ValidatedGraph validatedGraph,
          GraphOptimizationConfig optimizationConfig)
  ```

- Return the final successful `ValidatedGraph` directly. Add no second optimization-result,
  candidate, pass-report, statistics, diagnostics, or provenance type.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Give the three focused helpers these exact package-private static shapes:

  ```java
  static CompiledGraphModel canonicalize(CompiledGraphModel graph)
  static CompiledGraphModel eliminate(CompiledGraphModel graph)
  static CompiledGraphModel eliminate(CompiledGraphModel graph)
  ```

  The first method belongs to `GraphCanonicalization`; the second and third belong to
  `ForwardCommonSubexpressionElimination` and `ForwardDeadCodeElimination`, respectively. Each
  rejects a null graph with <span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>`NullPointerException("graph")` and returns only an immutable graph,
  with the optional helpers retaining the exact argument on no change.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [argument glossary](../../../../glossary.md#term-argument))</span>
- Require the input to be the successful package-private result of Compiler 0002. The pipeline
  does not trust or copy its constraint list; task-0002 validation regenerates constraints after
  mandatory canonicalization and after every changed optional candidate.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Apply mandatory [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) regardless of
  `optionalOptimizationsEnabled()`.
- Interpret `GraphOptimizationConfig.disabled()` as canonicalization plus [validation](../../../../glossary.md#term-validation) only.
- Interpret `GraphOptimizationConfig.standard()` as canonicalization plus the exact standard
  optional order `DCE -> CSE -> DCE`, with validation at each changed boundary and no fixed-point
  iteration.
- Keep all four new top-level production types and their four named static methods package-private
  in `io.github.pho001.synaptik.compiler`. Each type is final and stateless, with no instance
  fields and a private constructor; implementation-only nested records and methods remain private.
- Add meaningful source Javadocs for every new contract-relevant type and method, including
  inputs, immutability and identity behavior, results, ordering, and expected failures.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Add focused tests for the pipeline, canonicalization, exact CSE, DCE, transformed-candidate
  validation, identity allocation, phases, boundaries, multi-output and explicit RNG-state
  semantics, immutability, determinism, configuration interpretation, and visibility.<span data-glossary-navigation="true"> ([CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- Update the Compile API, targeted Tensor API current/planned statements, glossary, config
  Javadocs, and config/compiler planning status after implementation.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Tensor glossary](../../../../glossary.md#tensor); [planning glossary](../../../../glossary.md#term-planning))</span>
- Complete the required separate documentation-focused review in the same overall change.

### Mandatory [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) contract

Canonicalization is correctness-preserving [normalization](../../../../glossary.md#term-normalization), not optional optimization. It rebuilds
the accepted [immutable](../../../../glossary.md#term-immutable) [graph](../../../../glossary.md#graph) with one deterministic graph-local numbering and storage form while
preserving its complete computation and boundary structure.

[Canonical](../../../../glossary.md#term-canonical-form) allocation and storage order is exact:

1. allocate every graph input in `graph.inputs()` order as `ValueId(0)` through
   `ValueId(inputCount - 1)`;
2. visit [nodes](../../../../glossary.md#node) in stored [topological](../../../../glossary.md#term-topological-order) `graph.nodes()` order;
3. allocate each visited node's outputs in its stored output-position order;
4. allocate that node's [`NodeId`](../../../../glossary.md#nodeid) from zero in the same stored node order;
5. store `GraphValue` instances in that value-allocation order and `CompiledNode` instances in
   that node-allocation order; and
6. remap graph outputs in exact existing `graph.outputs()` order and phases by their owning
   remapped node.

Canonicalization must:

- retain the exact [`Operation`](../../../../glossary.md#operation) reference for every [occurrence](../../../../glossary.md#term-occurrence);
- retain the exact `TensorDescriptor` reference for every graph input and every node output;
- retain exact ordered node input positions, including repeated positions;
- retain every node and every output slot, including unused, auxiliary, saved-value, and opaque
  RNG-state outputs;<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- retain every graph input, including an unused input, in exact boundary order;
- retain every graph output as one distinct ordered boundary position;
- retain every node's exact `GraphPhase` value;
- allocate no ID from an original numeric ID, Tensor ID, identity hash, map iteration order, or
  process-global state;<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [hash glossary](../../../../glossary.md#term-hash))</span>
- produce equal canonical graphs for equal accepted graph structure regardless of sparse or
  non-monotonic original graph-local ID values; and
- mutate none of the input result, graph, values, nodes, collections, operations, descriptors,
  phases, predicates, or constraints.

The canonical graph is a new immutable [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel). No [identity](../../../../glossary.md#term-identity-meanings) promise is made for the
new graph, node, value, ID, or list containers. Exact immutable operation and [descriptor](../../../../glossary.md#term-descriptor) [element](../../../../glossary.md#term-element)
references are preserved because canonicalization changes graph-local identity only.

Immediately pass this canonical graph to
`CapturedGraphInference.inferAndValidate(CompiledGraphModel)`. The returned constraints therefore
refer to canonical `NodeId` values and are derived from canonical descriptors rather than copied
from the input result.

### Optional forward [dead-code elimination](../../../../glossary.md#term-dead-code-elimination-dce)

The same focused [DCE](../../../../glossary.md#compiler-transformations-and-dominance) helper is invoked exactly twice when optional optimization is enabled: once
before [CSE](../../../../glossary.md#term-common-subexpression-elimination-cse) and once as final cleanup after CSE. Each invocation computes [liveness](../../../../glossary.md#term-liveness) from:

- every ordered [graph](../../../../glossary.md#graph) output; and
- every non-`FORWARD` node, whose occurrence and complete outputs are roots outside this task's
  forward-only transformation authority.

Walk [producer](../../../../glossary.md#term-producer) dependencies backward without recursion proportional to graph depth. A [node](../../../../glossary.md#node) is live
when any output is needed. Once a node is live, retain the whole [occurrence](../../../../glossary.md#term-occurrence), make every ordered
input live, and retain every output slot even when only one slot is used. This indivisible-node
rule preserves top-K, attention, [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../../../glossary.md#term-normalization), [dropout](../../../../glossary.md#dropout--inverted-dropout) [mask](../../../../glossary.md#term-boolean-mask)/next-state, and future saved
values.

DCE may remove only a `FORWARD` node that is unreachable from graph outputs and all retained
non-forward work. It removes that node's complete output set. It must retain:

- every graph input and its [descriptor](../../../../glossary.md#term-descriptor) in exact boundary order, even when unused after DCE;
- every graph output in exact order;
- every non-forward node and its [dependency](../../../../glossary.md#term-dependency) closure;
- every output slot of each retained node;
- exact retained [operation](../../../../glossary.md#operation), descriptor, and phase references; and
- [topological order](../../../../glossary.md#term-topological-order) of retained nodes.

This is safe because [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) observes computation through graph outputs and phase-
classified retained work, current [model](../../../../glossary.md#term-model) operations contain no hidden [mutable](../../../../glossary.md#term-mutable) [side effect](../../../../glossary.md#term-side-effect), and [RNG
state](../../../../glossary.md#graph-rng-state--graphrngstate) and advancement are explicit graph values. Removing an unreachable explicit-state branch
cannot alter a retained state value or output.

Build a changed DCE candidate in the same dense [canonical](../../../../glossary.md#term-canonical-form) allocation order. Return the exact input
graph reference when no node is removed. Otherwise return the new [immutable](../../../../glossary.md#term-immutable) candidate to the
orchestrator, which immediately validates it through [Compiler](../../../../glossary.md#term-compiler) 0002 before the next pass or final
return.

### Optional exact forward [common-subexpression elimination](../../../../glossary.md#term-common-subexpression-elimination-cse)

After the first [DCE](../../../../glossary.md#term-dead-code-elimination-dce) and any required [validation](../../../../glossary.md#term-validation), visit the current [canonical](../../../../glossary.md#term-canonical-form) [nodes](../../../../glossary.md#node) once in stored
[topological order](../../../../glossary.md#term-topological-order). Only `GraphPhase.FORWARD` nodes are eligible. The first eligible [occurrence](../../../../glossary.md#term-occurrence) for
an exact key is the deterministic representative.

The [CSE](../../../../glossary.md#compiler-transformations-and-dominance) key contains exactly:

- the node's `GraphPhase.FORWARD` value;
- value equality of its [immutable](../../../../glossary.md#term-immutable) [`Operation`](../../../../glossary.md#operation), covering the typed operation kind and complete
  immutable attributes;
- the complete ordered list of already-remapped input [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) positions, including repeats;
  and
- the complete ordered list of output `TensorDescriptor` values, including output count and
  multi-output position.

A later node may merge only when that complete key equals the representative's key. Merge is
all-or-nothing: output position `i` maps to representative output position `i` for every slot. A
multi-output node is never partially merged, split, or reduced to its publicly wrapped outputs.

To preserve output and future [publication](../../../../glossary.md#term-publication) boundaries, a node whose output list contains any [graph](../../../../glossary.md#graph)
output is ineligible both to merge and to serve as a representative. This conservative rule keeps
every requested graph-output occurrence distinct and ordered instead of trying to represent two
publication roots with one `ValueId`, which [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) deliberately forbids.

The current contracts prove exact CSE safety:

- equal `Operation` values describe equal [backend](../../../../glossary.md#backend)-independent semantics and immutable attributes;
- equal ordered input IDs mean equal logical operands in equal semantic roles;
- equal complete output [descriptors](../../../../glossary.md#term-descriptor) mean equal logical result contracts for every output slot;
- only equal forward phases may merge;
- current operations retain no live service, storage, hidden generator, runtime state, or mutable
  side effect; and<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [mutable glossary](../../../../glossary.md#term-mutable); [side effect glossary](../../../../glossary.md#term-side-effect))</span>
- graph randomness is explicit. Equal RNG initializer attributes denote the same abstract stream
  position, and equal dropout input value, Shape, probability, state value, and conforming prepared
  implementation determine equal output, mask, and next-state values. Slotwise merging therefore
  preserves state branching and advancement semantics without inventing a random algorithm.<span data-glossary-navigation="true"> ([RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [Shape glossary](../../../../glossary.md#shape); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>

No operation-family whitelist, rewrite [registry](../../../../glossary.md#term-registry), annotation, reflection, class-name [dispatch](../../../../glossary.md#term-dispatch),
service loader, custom equivalence [callback](../../../../glossary.md#term-callback), or backend support query is added. [Compiler](../../../../glossary.md#term-compiler) 0002 has
already failed closed for unsupported operation kinds before this pass can receive a successful
input.

The CSE candidate must be rebuilt in the same dense allocation order as [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form): all
graph inputs first, then retained representative-node outputs and nodes in first-occurrence order.
All graph inputs remain present. Graph outputs are remapped in exact order. Retained operations,
descriptors, and phases retain their exact references. Return the exact input graph reference when
no merge occurs; otherwise return the new immutable candidate to the orchestrator, which
immediately validates it through Compiler 0002 before final DCE may consume it.

### Bounded pass sequence

The optional sequence is exactly one first [DCE](../../../../glossary.md#term-dead-code-elimination-dce) invocation, one [topological](../../../../glossary.md#term-topological-order) [CSE](../../../../glossary.md#term-common-subexpression-elimination-cse) invocation, and one
final DCE invocation. The first DCE removes unreachable occurrences before CSE chooses its first
eligible representatives. The sequence never loops, compares candidate scores, or seeks a fixed
point. CSE uses already-remapped inputs while scanning, so an earlier exact merge is visible to
later keys in the same pass. The final DCE closes [liveness](../../../../glossary.md#term-liveness) once after CSE; no pass is repeated
after that cleanup.

### [Validation](../../../../glossary.md#term-validation) and failure order

The pipeline validates top-level [arguments](../../../../glossary.md#term-argument) in this exact order:

1. reject a null `validatedGraph` with `NullPointerException("validatedGraph")`;
2. reject a null `optimizationConfig` with `NullPointerException("optimizationConfig")`;
3. canonicalize `validatedGraph.graph()`;
4. invoke `CapturedGraphInference.inferAndValidate` on the [canonical](../../../../glossary.md#term-canonical-form) candidate;
5. when optional work is disabled, return that exact validated canonical result;
6. when enabled, run first forward DCE; if it returns a changed graph, validate it and use that
   exact successful result as the next input;
7. run exact forward CSE; if it returns a changed graph, validate it and use that exact successful
   result as the next input;<span data-glossary-navigation="true"> ([CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>
8. [run](../../../../glossary.md#term-run) final forward [DCE](../../../../glossary.md#term-dead-code-elimination-dce); if it returns a changed [graph](../../../../glossary.md#graph), validate it; and
9. return the last exact successful `ValidatedGraph`.

[Canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) and both optional transforms consume structurally closed [immutable](../../../../glossary.md#term-immutable) graphs and are
not second structural validators. A transformed [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) constructor failure indicates
an implementation defect and propagates. Candidate [inference](../../../../glossary.md#term-inference), [descriptor](../../../../glossary.md#term-descriptor), or constraint failures
retain [Compiler](../../../../glossary.md#term-compiler) 0002's existing deterministic exception type, context, failure ordering, and
cause behavior. This task adds no public [compile](../../../../glossary.md#compile) exception or error translation.

The optional helpers return their exact graph argument on a no-change result. The pipeline does
not repeat inference for that identical graph. Every newly constructed candidate is validated
exactly once before another pass or caller can consume it.

## Out of [scope](../../../../glossary.md#term-scope)

- graph capture changes, Tensor/provenance traversal, a public capture or optimization method, a
  public <span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>`GraphCompiler`, `CompiledGraph`, [engine](../../../../glossary.md#term-engine) [facade](../../../../glossary.md#term-facade), or [compiler](../../../../glossary.md#term-compiler) service
- a `CompileConfig` aggregate, compile-mode interpretation, defaults for a future aggregate, or
  consumption of backend intent and partition-scoring configuration<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [backend intent glossary](../../../../glossary.md#backend-intent--backendintent); [partition glossary](../../../../glossary.md#partition))</span>
- rewriting, [binding](../../../../glossary.md#term-binding), solving, serializing, or publishing deferred [graph](../../../../glossary.md#graph) constraints
- a compiler-owned immutable constant fact/ingress representation, constant folding, or any
  inspection, import, evaluation, propagation, hashing, or storage of Tensor/graph element values;
  Draft Compiler 0003B owns that boundary and must not treat mutable public Tensor host storage as
  authoritative compile-time data<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [constant glossary](../../../../glossary.md#term-constant); [constant folding glossary](../../../../glossary.md#term-constant-folding); [evaluation glossary](../../../../glossary.md#term-evaluation); [element glossary](../../../../glossary.md#term-element); [mutable glossary](../../../../glossary.md#term-mutable); [host storage glossary](../../../../glossary.md#host-storage))</span>
- arithmetic or algebraic simplification, neutral/absorbing-element rules, reassociation,
  commutation, distributivity, strength reduction, or reciprocal substitution; exact candidates
  are handed off to Draft Compiler 0003A and remain outside task 0003<span data-glossary-navigation="true"> ([reassociation glossary](../../../../glossary.md#term-reassociation); [commutation glossary](../../../../glossary.md#term-commutation); [distributivity glossary](../../../../glossary.md#term-distributivity))</span>
- approximate mathematics, relaxed floating reassociation, or exceptional floating-value
  assumptions; these remain excluded from 0003A until a future numerical-permission contract
  exists
- redundant same-type cast removal or cast-chain folding because current cast contracts do not
  define all numerical conversion and chained-rounding behavior, and bypass can change the exact
  unresolved-layout and gradient descriptor contract<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- reshape/view/permute/slice/contiguous folding, inverse-pair cancellation, layout
  materialization, physical aliasing, copy selection, or descriptor replacement<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [slice glossary](../../../../glossary.md#slice); [contiguous glossary](../../../../glossary.md#term-contiguous); [cancellation glossary](../../../../glossary.md#term-cancellation); [materialization glossary](../../../../glossary.md#materialization); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
- operation decomposition, high-level operation lowering, graph fusion, backend fusion,
  specialization, route/kernel selection, or backend-specific graph construction<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [decomposition glossary](../../../../glossary.md#term-decomposition); [lowering glossary](../../../../glossary.md#lowering); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend); [specialization glossary](../../../../glossary.md#term-specialization); [kernel glossary](../../../../glossary.md#kernel))</span>
- a pass registry, rewrite-rule registry, generic optimizer framework, visitor framework, plugin
  API, annotations, reflection, string dispatch, custom callback, pass list, or public pass enum<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [optimizer glossary](../../../../glossary.md#term-optimizer); [plugin glossary](../../../../glossary.md#term-plugin); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [dispatch glossary](../../../../glossary.md#term-dispatch); [callback glossary](../../../../glossary.md#term-callback))</span>
- a graph-candidate collection, model-autotuning boundary, search space, cost model, score,
  measurement, cache, serialization, or tuning orchestration<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [cost model glossary](../../../../glossary.md#compiler-transformations-and-dominance); [cache glossary](../../../../glossary.md#term-cache); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- autograd, adjoint rules, saved-value selection, gradient accumulation, backward graph
  construction, phase creation, post-autograd optimization, or task 0004<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [adjoint glossary](../../../../glossary.md#term-adjoint); [gradient accumulation glossary](../../../../glossary.md#term-gradient-accumulation); [backward graph glossary](../../../../glossary.md#backward-graph))</span>
- [publication](../../../../glossary.md#term-publication) binding derivation, `PublicationPlan`, diagnostics, trace payloads/emission,
  capability analysis, ownership selection, partitioning, logical-memory orchestration, or<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
  `CompileArtifacts`
- preparation, physical memory, transfers, schedules, executables, runtime state/residency,
  backend behavior, engine behavior, or execution<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency))</span>
- changes to model semantics, graph DTOs, config behavior/API shape, planning/trace/runtime/
  prepare/engine/backend source or tests, module dependencies, Gradle/build configuration,
  architecture rules/docs/tests, backend conformance, or integration tests<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- a detailed task 0003A, 0003B, 0004, or 0005 specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially compiler responsibilities,
  immutable compile-time state, dependency rules, and compile lifecycle<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [immutable glossary](../../../../glossary.md#term-immutable); [compile glossary](../../../../glossary.md#compile); [dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)
- [Compiler task 0001: graph capture](0001-tensor-expression-graph-capture.md)<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction))</span>
- [Compiler task 0002: inference and validation](0002-captured-graph-inference-and-validation.md)<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation))</span>
- [Model master plan](../../model/master-plan.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model capability baseline](../../model/capabilities.md)
- [Model capability closure audit](../../model/model-capability-contract-closure-audit.md)
- [Model task 0009: compiled graph model](../../model/tasks/0009-compiled-graph-model.md)<span data-glossary-navigation="true"> ([compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel))</span>
- [Model task 0018K: operation signature hardening](../../model/tasks/0018k-operation-signature-and-construction-hardening.md)<span data-glossary-navigation="true"> ([operation signature glossary](../../../../glossary.md#operation-signature))</span>
- [Model task 0018L: shared multi-output provenance](../../model/tasks/0018l-shared-multi-output-tensor-provenance.md)<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- [Model task 0019B: explicit graph RNG state](../../model/tasks/0019b-explicit-graph-rng-state-foundation.md)<span data-glossary-navigation="true"> ([graph RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- [Model task 0019B1: explicit graph dropout](../../model/tasks/0019b1-explicit-graph-dropout-construction.md)<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- [Config master plan](../../config/master-plan.md)
- [Config task 0002: compile modes and optimization configuration](../../config/tasks/0002-compile-modes-and-graph-optimization-configuration.md)
- [Planning master plan](../../planning/master-plan.md)
- [Planning contract closure audit](../../planning/planning-contract-closure-audit.md)
- [Trace master plan](../../trace/master-plan.md)<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace))</span>
- [Compile API](../../../../api/compile-api.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Glossary](../../../../glossary.md)

## Architecture constraints

- [`Tensor`](../../../../glossary.md#tensor) remains public mutable API state and is not graph IR. Transformation consumes only the
  successful immutable graph and compiler pass state; it retains no Tensor, Tensor ID, producer,
  provenance, label, or storage reference.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [immutable glossary](../../../../glossary.md#term-immutable); [compiler glossary](../../../../glossary.md#term-compiler); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- [`Operation`](../../../../glossary.md#operation) remains model-owned immutable backend-independent semantics. The compiler compares
  its value contract but adds no support query, executable behavior, or compiler-owned operation
  kind.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [backend glossary](../../../../glossary.md#backend))</span>
- [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel), `CompiledNode`, and `GraphValue` remain model-owned immutable graph state.
  Transformations construct new valid instances; they do not add fields, mutate instances, or
  introduce a second public graph model.
- [`NodeId`](../../../../glossary.md#nodeid) and [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) are graph-local. Numeric identity may change across a transformation;
  deterministic remapping and boundary position preserve meaning before publication bindings are
  later derived.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [bindings glossary](../../../../glossary.md#term-binding))</span>
- [Graph](../../../../glossary.md#graph) outputs are the current logical preservation/[publication](../../../../glossary.md#term-publication) roots. This task derives no
  `PublicationBinding` or `PublicationPlan` and never collapses distinct graph-output
  occurrences.
- Every node is indivisible across its complete ordered outputs. Hidden, auxiliary, saved-value,
  multi-output, and explicit RNG-state positions are retained or remapped together.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- Forward optimization never transforms a non-forward occurrence. Mandatory ID canonicalization
  retains every phase; DCE treats non-forward work as live roots.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce))</span>
- Every constructed graph candidate passes the same Compiler 0002 inference and validation before
  advancing. Deferred constraints are regenerated and remain tied to the candidate's node IDs.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation))</span>
- Compiler output is compile-time state only. No physical, prepared, executable, backend, or
  runtime object enters the pipeline.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- The existing compiler-to-config and compiler-to-model dependencies are sufficient. No module
  edge or visibility widening is authorized.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [widening glossary](../../../../glossary.md#term-widening))</span>
- `ARCHITECTURE.md` remains unchanged. Stop if implementation requires changed operation
  semantics, a public binding/publication decision, another module's executable behavior, a new
  dependency, or a different lifecycle owner.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle); [owner glossary](../../../../glossary.md#term-owner))</span>

## Package impact

Existing package used:

- `io.github.pho001.synaptik.compiler` — remains the cohesive package-private compiler front-end
  and forward-transformation boundary. Keeping task-0003 types here lets them consume the existing
  package-private <span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler))</span>`ValidatedGraph` and `CapturedGraphInference` without [widening](../../../../glossary.md#term-widening) Java visibility.

Packages added or changed:

- No Java package is added. The compiler root gains one narrow orchestrator and three focused
  package-private transforms. It must not become a generic optimizer or pass registry. Task 0005
  still owns reassessment of a public/cross-package orchestration boundary when a concrete
  consumer exists.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [registry glossary](../../../../glossary.md#term-registry))</span>

Type placement:

- `io.github.pho001.synaptik.compiler.ForwardGraphOptimization` — package-private stateless entry
  point that interprets only [`GraphOptimizationConfig`](../../../../glossary.md#graph-optimization-configuration--graphoptimizationconfig), orders mandatory/optional work, and invokes
  Compiler 0002 validation at candidate boundaries.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- `io.github.pho001.synaptik.compiler.GraphCanonicalization` — package-private deterministic graph
  reindexing and immutable rebuild with no semantic simplification.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- `io.github.pho001.synaptik.compiler.ForwardCommonSubexpressionElimination` — package-private
  exact-key, first-representative, all-output forward CSE with graph-output exclusions.<span data-glossary-navigation="true"> ([CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>
- `io.github.pho001.synaptik.compiler.ForwardDeadCodeElimination` — package-private nonrecursive
  backward liveness and whole-node forward DCE.<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness); [node glossary](../../../../glossary.md#node); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce))</span>

Tests mirror the production package and one production concern each. No generic shared production
rewriter, candidate [DTO](../../../../glossary.md#data-transfer-object--dto), key DTO, or pass interface is authorized. Small private records inside the
owning source file are permitted only when they make that one implementation readable and do not
escape as a contract.

## Affected files

Expected [compiler](../../../../glossary.md#term-compiler) production paths:

- add
  `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardGraphOptimization.java`
- add
  `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCanonicalization.java`
- add
  `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardCommonSubexpressionElimination.java`
- add
  `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardDeadCodeElimination.java`

Expected compiler test paths:

- add
  `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ForwardGraphOptimizationTest.java`
- add
  `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCanonicalizationTest.java`
- add
  `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ForwardCommonSubexpressionEliminationTest.java`
- add
  `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ForwardDeadCodeEliminationTest.java`

Expected config [Javadoc](../../../../glossary.md#term-javadoc) paths:

- update
  `modules/config/src/main/java/io/github/pho001/synaptik/config/compile/GraphOptimizationConfig.java`
  only to replace future-consumer wording with the current package-private compiler consumer while
  preserving its exact record/API/behavior contract<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- update
  `modules/config/src/main/java/io/github/pho001/synaptik/config/compile/package-info.java`
  only to distinguish current optimization-permission consumption from still-unimplemented
  aggregate, mode, intent, and scoring orchestration

Expected documentation and [planning](../../../../glossary.md#term-planning) paths:

- update [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- update targeted current/planned boundaries in [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- update [Glossary](../../../../glossary.md)
- create and finalize this task specification
- update [Compiler master plan](../master-plan.md)
- update [Config master plan](../../config/master-plan.md) only to remove the stale claim that no
  compiler consumes [`GraphOptimizationConfig`](../../../../glossary.md#graph-optimization-configuration--graphoptimizationconfig); do not change any config task status or frontier
- update [Roadmap](../../../roadmap.md)

Review without modification unless a documented conflict requires stopping: existing compiler
source/tests, config tests and other config source, all [model](../../../../glossary.md#term-model) source/tests and planning records,
Public/Training APIs, architecture documents/tests, planning/[trace](../../../../glossary.md#trace)/[runtime](../../../../glossary.md#term-runtime)/[prepare](../../../../glossary.md#prepare)/[engine](../../../../glossary.md#term-engine)/training/
[backend](../../../../glossary.md#backend) source and tests, [Gradle](../../../../glossary.md#term-gradle)/build files, backend [conformance](../../../../glossary.md#term-conformance), and integration tests.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most the exact seventeen paths listed under
[Affected files](#affected-files): four [compiler](../../../../glossary.md#term-compiler) production files, four compiler test files, two
config [Javadoc](../../../../glossary.md#term-javadoc) files, and seven documentation/[planning](../../../../glossary.md#term-planning) files.

The scope is larger than the normal guide because one cohesive first pipeline requires the three
separately testable [graph](../../../../glossary.md#graph) transformations, one orchestrator, and synchronized current-consumer
documentation in config. It remains inside existing compiler dependencies and changes executable
behavior in only one [module](../../../../glossary.md#term-module).

Do not modify any existing compiler implementation/test, config test/[API](../../../../glossary.md#term-application-programming-interface-api) [shape](../../../../glossary.md#shape), [model](../../../../glossary.md#term-model) contract,
build file, architecture path, or later task. If implementation needs an eighteenth path, a
different package, another [operation](../../../../glossary.md#operation) contract, or a new public/cross-package type, stop and report
the concrete need rather than expanding scope.

## Acceptance criteria

- The [compiler](../../../../glossary.md#term-compiler) adds exactly one package-private static
  `ForwardGraphOptimization.optimize(ValidatedGraph, GraphOptimizationConfig)` entry point and no
  public declaration.
- The four planned production types are package-private final stateless utilities with private
  constructors; their four named transform/orchestration methods are package-private static, and
  no other non-private production method is added.
- The method rejects null [arguments](../../../../glossary.md#term-argument) in the specified order/messages and returns only a
  `ValidatedGraph`.
- Disabled optimization still performs deterministic mandatory canonicalization and post-
  canonical task-0002 validation, but performs no CSE or DCE.<span data-glossary-navigation="true"> ([canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [canonical glossary](../../../../glossary.md#term-canonical-form); [validation glossary](../../../../glossary.md#term-validation); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce))</span>
- Standard optimization performs only `canonicalize -> validate -> forward DCE -> validate if
  changed -> exact forward CSE -> validate if changed -> forward DCE cleanup -> validate if
  changed` in that fixed order, invoking each listed pass once without a fixed-point loop.
- Every newly constructed candidate is passed exactly once through
  `CapturedGraphInference.inferAndValidate` before advancing; an unchanged graph is not
  redundantly revalidated.
- Canonical node/value IDs are dense from zero and use exactly the specified input, node, and
  output-slot allocation order independent of original numeric IDs and map iteration.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node))</span>
- Canonicalization preserves all nodes, inputs, outputs, repeated operands, output slots, phases,
  exact operations, and exact descriptors without mutation.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- Successful transformed constraints are regenerated for final node IDs, remain immutable and
  deterministically ordered, and are never copied from the incoming result.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- CSE uses only the exact complete key above, retains the first eligible representative, remaps
  every output slot positionally, and permits a graph-output producer neither to merge nor to
  serve as a representative; non-forward nodes are never eligible.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- CSE distinguishes different operation kinds/attributes, input order or repetition, output
  count/descriptors, and phases; hash collisions cannot create equality.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [hash glossary](../../../../glossary.md#term-hash))</span>
- Equal explicit RNG initializer/dropout semantics may merge only under the same exact key and
  all-output rules; state branching, mask, and next-state positions remain intact.<span data-glossary-navigation="true"> ([RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>
- DCE removes only unreachable forward occurrences, retains all graph inputs, graph outputs,
  non-forward nodes and dependencies, and every output slot of each live node, and does not use
  Java recursion proportional to graph depth.
- Every changed CSE/DCE graph is structurally closed, canonical, immutable, topologically ordered,
  phase-complete, and valid under Compiler 0002.
- Tests cover sparse/non-monotonic graph-local IDs, zero-node pass-through, repeated inputs,
  fan-out, exact-reference retention, mixed forward/backward phase classification,
  deferred-constraint ID regeneration,
  deterministic repeat results, and input immutability.<span data-glossary-navigation="true"> ([fan-out glossary](../../../../glossary.md#term-fan-out))</span>
- Tests cover positive and negative CSE keys, first-representative order, requested-output
  exclusion with the output producer encountered both before and after an equal internal node,
  multi-output all-or-nothing remapping, explicit RNG-state replay, and no-change exact graph
  return.
- Tests cover dead/live branches, unused preserved graph inputs, multi-output/RNG slots, retained
  non-forward work and dependencies, a deep chain without stack overflow, and no-change exact
  graph return.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow))</span>
- Pipeline tests lock the [graph](../../../../glossary.md#graph)-visible `DCE -> CSE -> DCE` order, including a dead earlier
  duplicate that first DCE removes before CSE chooses the first live representative, changed-
  candidate deferred-constraint regeneration, deterministic repeat results, and exact no-change
  helper identity. Source inspection locks one invocation of each listed pass, no validation call
  for an unchanged helper result, and absence of iteration.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Tests lock package-private visibility and absence of a pass registry, public optimizer surface,
  candidate/result DTO, and config API-shape change.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [optimizer glossary](../../../../glossary.md#term-optimizer); [DTO glossary](../../../../glossary.md#data-transfer-object--dto); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape))</span>
- Compile API and targeted Tensor API text identify the exact current internal pipeline without
  claiming public compilation, binding, numerical execution, cast/arithmetic/algebraic/view
  rewrites, autograd, publication/planning orchestration, preparation, backend behavior, or
  runtime.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Tensor glossary](../../../../glossary.md#tensor); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [binding glossary](../../../../glossary.md#term-binding); [view glossary](../../../../glossary.md#term-view-meanings); [autograd glossary](../../../../glossary.md#autograd); [publication glossary](../../../../glossary.md#term-publication); [planning glossary](../../../../glossary.md#term-planning); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- [`GraphOptimizationConfig`](../../../../glossary.md#graph-optimization-configuration--graphoptimizationconfig) source/package Javadocs and config master-plan notes identify its one
  current package-private compiler consumer while preserving every public semantic boundary and
  keeping `CompileConfig` Draft.
- The glossary defines the reusable distinction between mandatory graph canonicalization and
  optional forward optimization, including exact CSE/DCE and current internal status.<span data-glossary-navigation="true"> ([graph canonicalization glossary](../../../../glossary.md#graph-canonicalization))</span>
- Existing compiler/config/model APIs, task-0001/0002 behavior, architecture, dependencies, Gradle,
  planning/trace/runtime/prepare/backend behavior, conformance, and integration remain unchanged
  for recorded reasons.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Gradle glossary](../../../../glossary.md#term-gradle); [trace glossary](../../../../glossary.md#trace); [prepare glossary](../../../../glossary.md#prepare); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Exactly the seventeen authorized paths change. Task 0003, its compiler master-plan row, and the
  roadmap become `Complete` only after implementation, documentation, and validation finish.
  Tasks 0003A, 0003B, 0004, and 0005 remain Draft rows without detailed specifications.
- A separate documentation-focused agent pass has finalized affected Javadocs, APIs, glossary,
  planning status/evidence, terminology, links, and no-change conclusions in this same overall
  change.

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, [run](../../../../glossary.md#term-run) focused transformation tests as needed:

```bash
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.GraphCanonicalizationTest
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.ForwardCommonSubexpressionEliminationTest
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.ForwardDeadCodeEliminationTest
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.ForwardGraphOptimizationTest
```

After executable Java stabilizes, the implementation context runs exactly one final [compiler](../../../../glossary.md#term-compiler)
[module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :modules:compiler:test
```

This is task-tier validation. No [dependency](../../../../glossary.md#term-dependency), architecture boundary, shared build configuration,
public [API](../../../../glossary.md#term-application-programming-interface-api) [shape](../../../../glossary.md#shape), or second module's executable behavior changes. Repository-wide and [architecture
test](../../../../glossary.md#term-architecture-test) validation is deferred to the named compiler transformation-and-[autograd](../../../../glossary.md#autograd) capability
[checkpoint](../../../../glossary.md#term-checkpoint) after task 0004 or [continuous integration](../../../../glossary.md#term-continuous-integration-ci), unless implementation reveals a concrete
cross-module risk.

The implementation context hands exact focused/final commands and XML counts to the documentation
context. The documentation pass does not repeat successful Java tests unless it changes executable
Java behavior or records a concrete stale-evidence risk.

Documentation pass:

```bash
./gradlew :modules:compiler:javadoc :modules:config:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

The documentation pass must also verify:

- all four new top-level types and four named methods are package-private, while implementation-
  only members remain private and no public declaration is added;
- complete meaningful `@param`, `@return`, and expected `@throws` coverage;
- generated Javadocs render the four compiler types and updated config status accurately;
- local Markdown links, anchors, balanced fences, final newlines, and trailing whitespace;
- exact seventeen-path [scope](../../../../glossary.md#term-scope);
- deterministic ID and one-shot `DCE -> CSE -> DCE` order plus every transformed-candidate
  validation call in source/tests;
- no public [optimizer](../../../../glossary.md#term-optimizer), pass/candidate/result type, [registry](../../../../glossary.md#term-registry), reflection, or string [dispatch](../../../../glossary.md#term-dispatch);
- no model value/storage inspection, cast/arithmetic/algebraic/view/constant rewrite, autograd,
  publication, planning, trace, prepare/runtime/backend/engine behavior, dependency/build, or
  architecture change;<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [view glossary](../../../../glossary.md#term-view-meanings); [constant glossary](../../../../glossary.md#term-constant); [publication glossary](../../../../glossary.md#term-publication); [planning glossary](../../../../glossary.md#term-planning); [trace glossary](../../../../glossary.md#trace); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [engine glossary](../../../../glossary.md#term-engine))</span>
- task/master/roadmap `Complete` [synchronization](../../../../glossary.md#term-synchronization) only after all evidence is final; and
- no task-0003A-or-later detailed specification.

## Dependencies

- [Compiler](../../../../glossary.md#term-compiler) task 0001, deterministic structural [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction) — Complete.
- Compiler task 0002, complete captured-graph inference/validation and typed deferred constraints
  — Complete.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation))</span>
- [Model](../../../../glossary.md#term-model) task 0009, [immutable](../../../../glossary.md#term-immutable) structurally closed [compiled graph](../../../../glossary.md#compiled-graph--compiledgraphmodel) — Complete.
- Model task 0018K, stable typed [operation](../../../../glossary.md#operation) [signatures](../../../../glossary.md#term-signature) and [occurrence](../../../../glossary.md#term-occurrence) [cardinality](../../../../glossary.md#term-cardinality) — Complete.
- Model task 0018L, whole-occurrence multi-output [provenance](../../../../glossary.md#provenance) — Complete.
- Model tasks 0019B/0019B1, explicit [RNG](../../../../glossary.md#graph-rng-state--graphrngstate)-state and [dropout](../../../../glossary.md#dropout--inverted-dropout) replay/advancement semantics — Complete.
- Model capability and contract closure — Complete.
- Config task 0002, stable one-[boolean](../../../../glossary.md#data-type--datatype) optional-optimization permission — Complete.
- Current compiler dependencies on model and config — present and sufficient.

[Planning](../../../../glossary.md#term-planning), [trace](../../../../glossary.md#trace) payloads, [runtime](../../../../glossary.md#term-runtime), [prepare](../../../../glossary.md#prepare), training, and concrete [backends](../../../../glossary.md#backend) are not dependencies of
this task. Their source and incomplete public orchestration contracts are not needed to transform
and revalidate internal immutable [graph](../../../../glossary.md#graph) state.

## Follow-up tasks

Future Draft [compiler](../../../../glossary.md#term-compiler) rows, in order:

- 0003A — operation-aware and data-type-aware exact arithmetic rewriting under current strict
  semantics, with Compiler 0002 revalidation and deterministic bounded cleanup; relaxed or
  fast-math transformations remain excluded until a future numerical-permission contract exists.
- 0003B — a compiler-owned immutable constant fact/ingress representation and exact deterministic
  constant folding after 0003A, without treating mutable public Tensor host storage as
  authoritative compile-time data; every changed candidate is revalidated through Compiler 0002,
  and runtime/backend execution, physical allocation, broad partial evaluation,
  relaxed/fast-math, and architecture changes remain excluded.<span data-glossary-navigation="true"> ([constant folding glossary](../../../../glossary.md#term-constant-folding); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- 0004 — autograd and backward graph construction after 0003A and 0003B, including separately
  specified post-autograd optimization and the transformation-and-autograd capability checkpoint.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [backward graph glossary](../../../../glossary.md#backward-graph); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- 0005 — publication, planning orchestration, diagnostics, and immutable compile artifacts after
  their concrete boundaries are stable.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [planning glossary](../../../../glossary.md#term-planning); [compile artifacts glossary](../../../../glossary.md#compile-artifacts))</span>

Draft task 0003A is the explicit [handoff](../../../../glossary.md#term-handoff) for exact arithmetic rewrites; it must derive every rule
from current [operation](../../../../glossary.md#operation) and data-type contracts rather than relaxed [identities](../../../../glossary.md#term-identity-meanings). Draft task 0003B
is the explicit handoff for [compile](../../../../glossary.md#compile)-time [constants](../../../../glossary.md#term-constant) and exact deterministic folding; it must first
define compiler-owned [immutable](../../../../glossary.md#term-immutable) constant facts and their ingress representation rather than read
[mutable](../../../../glossary.md#term-mutable) public [Tensor](../../../../glossary.md#tensor) [host storage](../../../../glossary.md#host-storage) as authoritative compile-time data, and it must route every
changed candidate back through Compiler 0002. Other future work may broaden [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) or
optional optimization only from explicit operation contracts and separately specified safety
proofs. Cast and [view](../../../../glossary.md#term-view-meanings) rewrites require their own numerical/[descriptor](../../../../glossary.md#term-descriptor) rules; [graph](../../../../glossary.md#graph) candidate
generation for [model autotuning](../../../../glossary.md#autotuning--model-autotuning) requires a stable orchestration consumer. Do not create those
detailed tasks or implement them opportunistically here.

## Architecture impact

Expected impact: None.

This task implements the architecture-defined [compiler](../../../../glossary.md#term-compiler) [ownership](../../../../glossary.md#term-ownership-meanings) of [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), [CSE](../../../../glossary.md#term-common-subexpression-elimination-cse), and [DCE](../../../../glossary.md#term-dead-code-elimination-dce)
using [immutable](../../../../glossary.md#term-immutable) [compile](../../../../glossary.md#compile)-time [graph](../../../../glossary.md#graph) state and the existing config permission. It changes no [module](../../../../glossary.md#term-module)
ownership, [dependency direction](../../../../glossary.md#java-platform-and-build-tools), public [lifecycle](../../../../glossary.md#lifecycle), [artifact](../../../../glossary.md#term-artifact) [shape](../../../../glossary.md#shape), or execution boundary. If a
safe transformation needs changed [model](../../../../glossary.md#term-model) semantics, public [publication](../../../../glossary.md#term-publication)/[binding](../../../../glossary.md#term-binding) behavior, another
module's executable logic, or a [dependency](../../../../glossary.md#term-dependency)/architecture update, stop and report the issue.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the compiler master plan, and
docs/planning/modules/compiler/tasks/0003-canonicalization-and-forward-optimization.md. Read the
directly referenced compiler/config/model source, tests, APIs, glossary, and completed contracts
needed to verify transformation safety.

Implement Compiler 0003 exactly within its seventeen authorized paths. Do not add any other
canonicalization or optimization, public compiler/pass/candidate/result surface, registry,
constant/value execution, cast/arithmetic/algebraic/view rewrite, autograd/backward work,
publication/planning/diagnostic orchestration, CompileArtifacts, trace emission, prepare/runtime/
backend/engine work, or dependency/build/architecture change. Stop on a scope, safety-proof, or
architecture conflict.
The optional standard pipeline is the one-shot `DCE -> CSE -> DCE` sequence in the task; do not
reorder it, iterate it to a fixed point, or skip validation of any changed candidate.

After executable implementation and the final compiler test evidence, hand the actual diff and
exact evidence to a separate documentation-focused agent or thread with clean context. That pass
must follow docs/developer-guide/documentation-rules.md, finalize package-private and config
Javadocs, Compile/Tensor APIs, glossary, planning status/evidence, terminology, links, and no-change
conclusions in the same overall change, and must not repeat successful Java tests unless executable
behavior changes or a concrete stale-evidence risk is recorded.

Update this task with local decisions, exact validation evidence, implementation notes, completion
summary, and final status. Do not mark it Complete before that pass and every acceptance criterion
finish.
```

## Local decisions

- Keep [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) separate from optional optimization so `disabled()` still produces one
  dense, revalidated graph and the config boolean remains only an optimization permission.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [boolean glossary](../../../../glossary.md#data-type--datatype))</span>
- Rebuild changed CSE and DCE candidates directly in canonical allocation order. Optional helpers
  return the exact input graph when unchanged, allowing the orchestrator to avoid redundant
  validation without adding a result or change-report type.<span data-glossary-navigation="true"> ([CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce); [canonical glossary](../../../../glossary.md#term-canonical-form); [validation glossary](../../../../glossary.md#term-validation))</span>
- Use the complete operation/ordered-remapped-input/phase/output-descriptor key for exact CSE and
  exclude every graph-output producer from both sides of representation. Whole-node slotwise
  merging preserves multi-output and explicit RNG-state occurrences.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [descriptor glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer); [node glossary](../../../../glossary.md#node); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- Treat every non-forward node as a DCE root and walk value dependencies iteratively. This keeps
  forward-only authority explicit and avoids Java recursion proportional to graph depth.
- Keep all four types and their named methods package-private. No current external consumer
  justifies a public compiler, pass, registry, candidate, result, or diagnostics surface.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [registry glossary](../../../../glossary.md#term-registry))</span>

## Known limitations

- The only optional pipeline is the one-shot `DCE -> CSE -> DCE` sequence. It does not iterate to a
  fixed point or expose pass selection.
- Exact arithmetic/algebraic rewriting remains Draft task 0003A; compiler-owned immutable constant
  ingress and folding remain Draft task 0003B.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [immutable glossary](../../../../glossary.md#term-immutable); [constant glossary](../../../../glossary.md#term-constant))</span>
- Cast, view, layout, decomposition, autograd, publication, planning, diagnostics, trace,
  preparation, runtime, backend, engine, numerical execution, and public compilation surfaces are
  not implemented by this task.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [layout glossary](../../../../glossary.md#layout); [decomposition glossary](../../../../glossary.md#term-decomposition); [autograd glossary](../../../../glossary.md#autograd); [publication glossary](../../../../glossary.md#term-publication); [planning glossary](../../../../glossary.md#term-planning); [trace glossary](../../../../glossary.md#trace); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [engine glossary](../../../../glossary.md#term-engine); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- Graph-output producers are conservatively excluded from CSE even when their exact keys match, so
  distinct requested publication roots remain distinct.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([Graph glossary](../../../../glossary.md#graph); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- The implementation context ran the focused canonicalization and exact-CSE command. The final XML
  evidence recorded 4 <span data-glossary-navigation="true"> ([canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>`GraphCanonicalizationTest` tests and 7
  `ForwardCommonSubexpressionEliminationTest` tests, all passing with no failures, errors, or
  skips.
- The implementation context ran the focused forward-DCE command. Its final XML evidence recorded
  6 <span data-glossary-navigation="true"> ([DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce))</span>`ForwardDeadCodeEliminationTest` tests, all passing with no failures, errors, or skips.
- The implementation context ran the focused optimization-pipeline command. Its final XML evidence
  recorded 9 `ForwardGraphOptimizationTest` tests, all passing with no failures, errors, or skips.
- After executable Java stabilized, the implementation context ran exactly one final
  `./gradlew :modules:compiler:test`. It passed 68 tests across 11 suites with zero failures,
  errors, or skips. Its pre-handoff <span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>`git diff --check` also passed.
- The mandatory separate clean-context documentation-focused review was performed by Codex task
  identity <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>`/root` on 2026-07-15. It applied the General, API/Javadoc, and Planning profiles and
  independently reviewed the actual four production files, four tests, config Javadocs and master
  plan, Compile and targeted Tensor APIs, glossary, compiler tasks 0001/0002, and directly relevant
  model multi-output and explicit RNG-state contracts.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Planning glossary](../../../../glossary.md#term-planning); [Compile glossary](../../../../glossary.md#compile); [Tensor glossary](../../../../glossary.md#tensor); [compiler glossary](../../../../glossary.md#term-compiler); [model glossary](../../../../glossary.md#term-model); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- The documentation pass changed no executable Java behavior or tests. It therefore reused the
  implementation context's focused and final Java-test evidence and did not rerun those suites.
- `./gradlew :modules:compiler:javadoc :modules:config:javadoc` passed with `BUILD SUCCESSFUL in
  1s`; 8 actionable tasks were considered, 4 executed and 4 up-to-date. Generated compiler pages
  render all four package-private types and named methods, with complete parameter, return, and
  expected-failure documentation. Generated config pages render the current package-private
  consumer while preserving the public boolean permission contract.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [boolean glossary](../../../../glossary.md#data-type--datatype))</span>
- Source and generated-[Javadoc](../../../../glossary.md#term-javadoc) inspection confirmed all four top-level types are package-private
  `final`, all four named methods are package-private `static`, constructors and implementation
  helpers remain private, and no public compiler declaration was added.
- `python3 /tmp/validate_synaptik_markdown.py` was present and passed after documentation edits: 234
  Markdown files, 4,242 local links, 260 local anchors, 2,962 fence markers, and final-newline and
  trailing-whitespace validation passed.
- Manual source/test inspection confirmed mandatory input-first then topological/output-slot dense
  canonicalization; the one-shot DCE/CSE/DCE source order; validation after canonicalization and
  every changed optional candidate; no validation for unchanged helper identity; complete exact
  CSE keys; graph-output representative exclusion; whole-node output retention; and graph-output
  plus non-forward DCE roots.<span data-glossary-navigation="true"> ([topological glossary](../../../../glossary.md#term-topological-order); [graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node))</span>
- Manual scope inspection found no constant/value execution or cast, arithmetic, algebraic, view,
  decomposition, fusion, autograd, publication, planning, diagnostics, <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [constant glossary](../../../../glossary.md#term-constant); [view glossary](../../../../glossary.md#term-view-meanings); [decomposition glossary](../../../../glossary.md#term-decomposition); [fusion glossary](../../../../glossary.md#term-fusion); [autograd glossary](../../../../glossary.md#autograd); [publication glossary](../../../../glossary.md#term-publication))</span>`CompileArtifacts`, trace,
  prepare, runtime, backend, engine, dependency, or build behavior. No pass registry, reflection,
  string dispatch, public optimizer, or candidate/result DTO was added.<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [engine glossary](../../../../glossary.md#term-engine); [dependency glossary](../../../../glossary.md#term-dependency); [registry glossary](../../../../glossary.md#term-registry); [dispatch glossary](../../../../glossary.md#term-dispatch); [optimizer glossary](../../../../glossary.md#term-optimizer); [DTO glossary](../../../../glossary.md#data-transfer-object--dto))</span>
- Public API and Training API required no change because the implementation adds no public Java
  declaration, callable compiler lifecycle, training, gradient, or optimizer behavior. Existing
  compiler, config, and model APIs and completed compiler tasks 0001/0002 required no change because
  task 0003 consumes their contracts without changing signatures or established behavior.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [gradient glossary](../../../../glossary.md#term-gradient); [signatures glossary](../../../../glossary.md#term-signature))</span>
- Architecture documents/tests required no change because compiler-owned canonicalization, CSE,
  and DCE are already authorized and no module boundary or dependency changed. Planning, trace,
  runtime, prepare, backend, engine, Gradle, conformance, and integration paths required no change
  because this task adds no behavior in those owners and no cross-module executable contract.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- The config master plan changed only its stale consumer-status wording; all config task statuses
  remain unchanged. Compiler tasks 0003A, 0003B, 0004, and 0005 remain Draft rows without detailed
  specifications.
- Final `git diff --check` passed. The combined
  `{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u` audit returned
  exactly the seventeen authorized paths, and `git status --short` reported only those paths.
  Status, forbidden-surface, package-private visibility, source-order, changed-candidate
  validation, and no-later-detailed-spec checks passed after completion synchronization.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>

## Implementation notes

- Added four package-private stateless compiler types: one orchestrator, mandatory canonicalizer,
  exact forward CSE, and iterative forward DCE.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce))</span>
- Canonicalization always creates a new immutable graph with dense graph-local IDs while retaining
  exact operation and descriptor references, phases, boundaries, repeated inputs, and output slots.<span data-glossary-navigation="true"> ([Canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [immutable glossary](../../../../glossary.md#term-immutable); [graph glossary](../../../../glossary.md#graph); [operation glossary](../../../../glossary.md#operation); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Standard optimization invokes DCE, CSE, and cleanup DCE once each. Changed candidates are rebuilt
  canonically and revalidated through Compiler 0002; unchanged optional results retain identity.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Exact CSE uses already-remapped inputs during one topological scan and merges complete
  multi-output/RNG occurrences slotwise. DCE walks liveness iteratively from graph outputs and
  non-forward work and retains every input plus all slots of each live node.<span data-glossary-navigation="true"> ([topological glossary](../../../../glossary.md#term-topological-order); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [liveness glossary](../../../../glossary.md#term-liveness); [node glossary](../../../../glossary.md#node))</span>
- Finalized all four production Javadocs, current [`GraphOptimizationConfig`](../../../../glossary.md#graph-optimization-configuration--graphoptimizationconfig) consumer wording,
  Compile/Tensor API boundaries, glossary terminology, and synchronized planning records without
  modifying executable behavior or tests.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: implemented and documented mandatory dense canonicalization and the optional
  one-shot forward DCE/CSE/DCE pipeline with validation at every changed candidate boundary.<span data-glossary-navigation="true"> ([canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [validation glossary](../../../../glossary.md#term-validation))</span>
- Files changed or created: exactly the seventeen authorized paths—four production files, four
  focused tests, two config Javadoc files, Compile API, targeted Tensor API, glossary, this task,
  compiler master plan, config master plan, and roadmap.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Tensor glossary](../../../../glossary.md#tensor); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Tests and validation: reused the implementation context's passing 4-test canonicalization,
  7-test CSE, 6-test DCE, 9-test pipeline, and final 68-test/11-suite compiler evidence; compiler
  and config Javadoc, generated-page inspection, Markdown validation, exact-scope/status audits,
  forbidden-surface checks, and whitespace checks passed.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: the mandatory clean-context pass finalized all affected Javadocs,
  APIs, glossary, planning status/evidence, terminology, and no-change conclusions without
  changing executable Java or repeating successful tests.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: current internal canonicalization and exact forward optimization are now
  distinguished from still-planned operation-specific rewrites and every public/downstream
  lifecycle surface.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Javadoc review: all new package-private type/method contracts and updated config consumer wording
  render correctly with complete tags and boundaries.
- Glossary impact: added the reusable mandatory graph-canonicalization versus optional forward-
  optimization distinction, including exact CSE/DCE and current internal status.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- Unresolved issues: None.
- Follow-up required: None for task 0003. Draft tasks 0003A, 0003B, 0004, and 0005 require separate
  future planning and implementation.

Status: Complete
