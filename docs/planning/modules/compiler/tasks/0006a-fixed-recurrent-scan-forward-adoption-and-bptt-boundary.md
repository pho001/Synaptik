# Task 0006A: Fixed Recurrent-[Scan](../../../../glossary.md#term-cumulative-scan) Forward Adoption and Explicit [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) Boundary

## Status

Complete

## Goal

Adopt the current [Model](../../../../glossary.md#term-model) fixed recurrent-[scan](../../../../glossary.md#term-cumulative-scan) family in [Compiler](../../../../glossary.md#term-compiler)'s forward-only [graph](../../../../glossary.md#graph) pipeline
without introducing graph regions, recurrence execution, or [backpropagation through time](../../../../glossary.md#term-backpropagation-through-time-bptt) ([BPTT](../../../../glossary.md#backpropagation-through-time-and-recurrent-state)).

For each [identity](../../../../glossary.md#term-identity-meanings)-distinct `RNN_TANH`, `GRU_RESET_AFTER`, or [`LSTM`](../../../../glossary.md#term-long-short-term-memory-lstm) [producer](../../../../glossary.md#term-producer) constructed through
the low-level `model.tensor.RecurrentScan` namespace, Compiler must preserve one ordinary flat
multi-output `CompiledNode`, independently infer and validate its complete static [descriptor](../../../../glossary.md#term-descriptor)
contract, retain its kind and `FORWARD` or `REVERSE` direction, and hand that same ordinary [node](../../../../glossary.md#node)
to the existing [publication](../../../../glossary.md#term-publication) and [Planning](../../../../glossary.md#term-planning) orchestration.

The selected compiled representation is the existing Model-owned graph representation:

```text
one RecurrentScan TensorProducer occurrence
  -> one ordinary CompiledNode
     operation = exact RecurrentScanKind + exact RecurrentDirection
     inputs    = exact ordered 5/6 or 6/7 ValueIds
     outputs   = exact ordered 2 or 3 ValueIds
  -> existing OperationCapabilityQuery when ordinary Planning orchestration is invoked
```

There is no recurrent-specific compiler [intermediate representation](../../../../glossary.md#term-intermediate-representation-ir), body, nested graph, region,
loop node, callable graph, or per-time-step expansion. Graph construction and compiled [topology](../../../../glossary.md#term-topology)
for one [occurrence](../../../../glossary.md#term-occurrence) remain [constant](../../../../glossary.md#term-constant) in `time`.

Backward-capable [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) remains intentionally unavailable for every [forward graph](../../../../glossary.md#forward-graph) containing
a recurrent occurrence. Compiler must reject that complete forward inventory during allocation-
free [autograd](../../../../glossary.md#autograd) [preflight](../../../../glossary.md#term-preflight), before constructing a [seed](../../../../glossary.md#term-seed), [derivative](../../../../glossary.md#term-derivative) constant, formula [Tensor](../../../../glossary.md#tensor), or
partial [gradient](../../../../glossary.md#term-gradient) graph. The closed first-order gradient inventory therefore continues to exclude
the exact three recurrent [signatures](../../../../glossary.md#term-signature) even though forward [inference](../../../../glossary.md#term-inference) adopts them.

## Motivation

[Model](../../../../glossary.md#term-model) tasks 0025E and 0025F now supply the stable semantic family, exact [signatures](../../../../glossary.md#term-signature), [canonical](../../../../glossary.md#term-canonical-form)
multi-output wrappers, and corrected stateless `RecurrentScan.rnn/gru/lstm` construction seam.
Before this task, generic [Compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings) already saw those expressions as one flat [node](../../../../glossary.md#node), but
`CapturedGraphInference` rejected `RecurrentScanKind` before optimization, [publication](../../../../glossary.md#term-publication), or
[Planning](../../../../glossary.md#term-planning). This task closes only that forward Compiler gap.

The distinction between forward and [gradient](../../../../glossary.md#term-gradient) inventories is essential:

| Inventory or stage | Result after 0006A |
|---|---|
| Model [operation](../../../../glossary.md#operation) signatures | The three recurrent signatures remain current structural Model declarations. |
| Compiler capture | Each exact [producer](../../../../glossary.md#term-producer) [identity](../../../../glossary.md#term-identity-meanings) becomes one flat node with every ordered output slot. |
| Compiler forward [inference](../../../../glossary.md#term-inference)/final [validation](../../../../glossary.md#term-validation) | The three recurrent signatures are supported and independently revalidated. |
| Compiler first-order gradient coverage | The existing 128 supported signatures remain unchanged; the three recurrent signatures remain the exact disjoint deferred set. |
| Backward-capable [compile](../../../../glossary.md#compile) [preflight](../../../../glossary.md#term-preflight) | Any recurrent [occurrence](../../../../glossary.md#term-occurrence) in the complete original forward inventory fails before [derivative](../../../../glossary.md#term-derivative) [Tensor](../../../../glossary.md#tensor) allocation. |
| Planning/[backend capability](../../../../glossary.md#backend-capability) | The unchanged ordinary query carries the node; no provider gains support in this task. |

This is the smallest coherent forward adoption because it uses the existing flat [graph](../../../../glossary.md#graph) and
[lifecycle](../../../../glossary.md#lifecycle) boundaries while preserving the explicitly deferred [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) decision.

## [Scope](../../../../glossary.md#term-scope)

- Add one package-private, final, stateless `RecurrentScanInference` helper in
  `io.github.pho001.synaptik.compiler`.
- Route every `RecurrentScanKind` from `CapturedGraphInference` to that helper.
- Independently derive and validate all three kinds, both directions, and biased or bias-free
  cardinalities from ordered graph descriptors.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- Preserve the existing ordinary `CompiledNode` as the sole compiled representation.
- Preserve exact [producer](../../../../glossary.md#term-producer)-[identity](../../../../glossary.md#term-identity-meanings) [capture](../../../../glossary.md#term-capture-meanings) behavior:
  - one producer reached through multiple sibling wrappers is captured once;
  - structurally equal but identity-distinct recurrent producers remain distinct [nodes](../../../../glossary.md#node);
  - repeated reachability never duplicates a producer;
  - all two or three output slots remain present on a live recurrent node.
- Exclude recurrent occurrences from common-subexpression elimination (CSE), so equivalent
  recurrent expressions cannot erase their Model occurrence identity. Other current CSE
  eligibility and exact merging behavior remain unchanged.<span data-glossary-navigation="true"> ([CSE glossary](../../../../glossary.md#compiler-transformations-and-dominance))</span><span data-glossary-navigation="true"> ([common-subexpression elimination glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [Model glossary](../../../../glossary.md#term-model); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Preserve whole-node dead-code elimination (DCE): an entirely unreachable recurrent occurrence
  may be removed, but if any output is live, the node and every declared sibling output slot stay
  structurally present.<span data-glossary-navigation="true"> ([DCE glossary](../../../../glossary.md#compiler-transformations-and-dominance))</span><span data-glossary-navigation="true"> ([dead-code elimination glossary](../../../../glossary.md#term-dead-code-elimination-dce))</span>
- Preserve caller order for multi-root graph outputs, including roots from different slots of one
  occurrence and roots from multiple identity-distinct occurrences.
- Admit the family only through `CompileMode.FORWARD_ONLY` in the current [Compiler](../../../../glossary.md#term-compiler) [graph](../../../../glossary.md#graph) pipeline.
- Add a deterministic allocation-free autograd-preflight guard over the complete original forward
  producer inventory for <span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [preflight glossary](../../../../glossary.md#term-preflight); [guard glossary](../../../../glossary.md#term-guard))</span>`FORWARD_AND_BACKWARD` and `TRAINING_STEP`.
- Preserve the exact three recurrent [signature](../../../../glossary.md#term-signature) fingerprints as deferred from
  `FirstOrderGradientCoverage`; do not add a recurrent formula owner or classify valid lengths as
  differentiable.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule))</span>
- Prove that successful forward graph compilation, canonicalization, optional optimization,
  repeated final inference/validation, publication binding, and the existing ordinary Planning
  query preserve recurrent node semantics and ordered descriptors.<span data-glossary-navigation="true"> ([forward graph glossary](../../../../glossary.md#forward-graph); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [publication glossary](../../../../glossary.md#term-publication); [binding glossary](../../../../glossary.md#term-binding); [Planning glossary](../../../../glossary.md#term-planning))</span>
- Update the Compile API and fixed-recurrent glossary entry to mark only Compiler forward
  adoption current and keep execution, capability, and BPTT claims planned.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [BPTT glossary](../../../../glossary.md#term-backpropagation-through-time-bptt))</span>
- Complete a separate documentation-focused review in the same overall change.

## Forward [inference](../../../../glossary.md#term-inference) and [validation](../../../../glossary.md#term-validation) contract

### Accepted [operation](../../../../glossary.md#operation) and cardinalities

The helper accepts only an `Operation` whose kind is one of:

```text
RNN_TANH
GRU_RESET_AFTER
LSTM
```

Its exact attributes object is the current `RecurrentDirection.FORWARD` or
`RecurrentDirection.REVERSE`. `Operation` and `CompiledNode` already enforce the family-owned
structural [signature](../../../../glossary.md#term-signature), but recurrent [inference](../../../../glossary.md#term-inference) still dispatches by the closed typed family and
must not accept a custom kind or alternate attributes type.

Accepted ordered inputs are exactly:

```text
RNN_TANH / GRU_RESET_AFTER, bias-free:
  [input, validLengths, initialHidden, inputWeight, hiddenWeight]

RNN_TANH / GRU_RESET_AFTER, biased:
  [input, validLengths, initialHidden, inputWeight, hiddenWeight, bias]

LSTM, bias-free:
  [input, validLengths, initialHidden, initialCell, inputWeight, hiddenWeight]

LSTM, biased:
  [input, validLengths, initialHidden, initialCell, inputWeight, hiddenWeight, bias]
```

Accepted and independently derived output order is exactly:

```text
RNN_TANH / GRU_RESET_AFTER:
  [outputs, finalHidden]

LSTM:
  [outputs, finalHidden, finalCell]
```

### [Descriptor](../../../../glossary.md#term-descriptor) derivation

[Inference](../../../../glossary.md#term-inference) revalidates the current [Model](../../../../glossary.md#term-model) descriptor contract in semantic role order without
reading [Tensor](../../../../glossary.md#tensor) storage or [runtime](../../../../glossary.md#term-runtime) valid-length values:

1. `input` has one floating [data type](../../../../glossary.md#data-type--datatype) and a fully static [rank](../../../../glossary.md#term-rank)-three [Shape](../../../../glossary.md#shape)
   `[time, batch, inputSize]`; `inputSize` is positive while `time` and [`batch`](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) may be zero.
2. `validLengths` is fully static [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `INT64[batch]` and has
   `requiresGrad == false`.
3. `initialHidden` has the exact common floating type and fully static Shape
   `[batch, hiddenSize]`; `hiddenSize` is positive.
4. For [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm), `initialCell` has the exact common type and Shape
   `[batch, hiddenSize]`.
5. `gateCount * hiddenSize` is computed with checked arithmetic, where gate count is one, three,
   or four for RNN, GRU, or LSTM.<span data-glossary-navigation="true"> ([RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru))</span>
6. `inputWeight` has the common type and Shape
   `[gateCount * hiddenSize, inputSize]`.
7. `hiddenWeight` has the common type and Shape
   `[gateCount * hiddenSize, hiddenSize]`.
8. When present, [`bias`](../../../../glossary.md#term-bias) has the common type and Shape `[gateCount * hiddenSize]`.
9. `requiresGrad` is the OR of `input`, initial state, weight, and optional bias roles. The
   valid-length role never contributes.<span data-glossary-navigation="true"> ([weight glossary](../../../../glossary.md#term-weight))</span>
10. Derived outputs have unresolved [layout](../../../../glossary.md#layout) and exact descriptors:
    - dense `outputs`: common type, `[time, batch, hiddenSize]`, derived [gradient](../../../../glossary.md#term-gradient) eligibility;
    - `finalHidden`: common type, `[batch, hiddenSize]`, derived gradient eligibility;
    - LSTM `finalCell`: common type, `[batch, hiddenSize]`, derived gradient eligibility.

Every Shape is fully static, so this family produces no [deferred graph constraint](../../../../glossary.md#deferred-graph-constraint). Direction
changes transition traversal meaning but does not change descriptors.

`CapturedGraphInference` retains its existing deterministic outer behavior: it derives the
complete descriptor list, compares each ordered result with the stored `GraphValue` descriptor,
and reports [node](../../../../glossary.md#node) position, [`NodeId`](../../../../glossary.md#nodeid), kind, output position, [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), expected descriptor, and
stored descriptor on mismatch. The same pass runs after [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) and after every changed
optional optimization candidate, providing final [validation](../../../../glossary.md#term-validation) without a recurrent-only validation
phase.

### [Validation](../../../../glossary.md#term-validation) [ownership](../../../../glossary.md#term-ownership-meanings) and exclusions

[Model](../../../../glossary.md#term-model) construction continues to perform its current local [descriptor](../../../../glossary.md#term-descriptor) validation before creating
the [producer](../../../../glossary.md#term-producer). [Compiler](../../../../glossary.md#term-compiler) independently checks the captured [occurrence](../../../../glossary.md#term-occurrence) because [graph](../../../../glossary.md#graph) [inference](../../../../glossary.md#term-inference) may
also receive manually assembled [immutable](../../../../glossary.md#term-immutable) graph values and every transformed candidate must be
validated from graph state. Compiler adds no check for facts that require [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) values or a
later [lifecycle](../../../../glossary.md#lifecycle) [owner](../../../../glossary.md#term-owner).

In particular, Compiler does not:

- read, [snapshot](../../../../glossary.md#term-snapshot), bind, or validate the [scalar](../../../../glossary.md#term-scalar) contents of `validLengths`;
- require [host storage](../../../../glossary.md#host-storage) or infer lengths from input values, [padding](../../../../glossary.md#padding), labels, or [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings);
- choose traversal loops, padding writes, gates, accumulators, algorithms, saved state, or
  numerical lowering;<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering))</span>
- add dynamic or [binding](../../../../glossary.md#term-binding)-dependent [Shape](../../../../glossary.md#shape) support; or
- duplicate Model expression-construction checks in another public [API](../../../../glossary.md#term-application-programming-interface-api).

[Runtime](../../../../glossary.md#term-runtime) length bounds `[0, time]`, complete pre-write validation, positive-zero padding, and
skipped recurrent arithmetic remain concrete-[backend](../../../../glossary.md#backend) execution responsibilities.

## [Capture](../../../../glossary.md#term-capture-meanings), optimization, and downstream [handoff](../../../../glossary.md#term-handoff)

### [Identity](../../../../glossary.md#term-identity-meanings) and [topology](../../../../glossary.md#term-topology)

`GraphCapture` remains unchanged. Its existing identity maps already provide the required
representation:

- exact [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) [reference identity](../../../../glossary.md#term-reference-identity) controls [capture](../../../../glossary.md#term-capture-meanings) deduplication;
- one [producer](../../../../glossary.md#term-producer) allocates one [`NodeId`](../../../../glossary.md#nodeid) and all output [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) slots in declared order;
- exact `TensorProvenance.outputIndex()` maps [canonical](../../../../glossary.md#term-canonical-form) wrappers to those slots;
- structurally equal distinct producers remain distinct at capture; and
- topology contains no step, gate, [mask](../../../../glossary.md#term-boolean-mask), [padding](../../../../glossary.md#padding), state-update, or body [node](../../../../glossary.md#node).

The focused task tests must lock this behavior with `RecurrentScan`-constructed producers rather
than a synthetic custom [operation](../../../../glossary.md#operation).

### [CSE](../../../../glossary.md#term-common-subexpression-elimination-cse) and [DCE](../../../../glossary.md#term-dead-code-elimination-dce)

Current generic CSE would otherwise consider two equal internal recurrent [nodes](../../../../glossary.md#node) mergeable by
[operation](../../../../glossary.md#operation) value, remapped inputs, [descriptors](../../../../glossary.md#term-descriptor), phase, and [derivative](../../../../glossary.md#term-derivative) order. That conflicts with
the selected [identity](../../../../glossary.md#term-identity-meanings)-distinct recurrent [occurrence](../../../../glossary.md#term-occurrence) contract. CSE eligibility must therefore
exclude every `RecurrentScanKind` occurrence before creating an expression key or representative.
This is a narrow family-specific exclusion; no other CSE key, ordering, output-[producer](../../../../glossary.md#term-producer) exclusion,
phase rule, or derivative-order rule changes.

DCE remains whole-node structural [liveness](../../../../glossary.md#term-liveness). When any recurrent output slot is a [graph](../../../../glossary.md#graph) root or
feeds live work, the node remains with every two or three output [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) positions. An unused
sibling is not removed from the node and its slots are not renumbered. If no output is live, DCE
may remove the whole occurrence under its current rules.

### [Publication](../../../../glossary.md#term-publication) and [Planning](../../../../glossary.md#term-planning) [handoff](../../../../glossary.md#term-handoff)

[Compiler](../../../../glossary.md#term-compiler) introduces no new handoff record. After final [validation](../../../../glossary.md#term-validation), existing publication [bindings](../../../../glossary.md#term-binding)
refer to the final recurrent output [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) values exactly as for every other [graph](../../../../glossary.md#graph) root.
Existing graph-wide Planning orchestration creates one ordinary [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery) per
final [node](../../../../glossary.md#node) from:

- the exact retained [`Operation`](../../../../glossary.md#operation) reference, including recurrent kind and direction;
- the exact ordered input [descriptor](../../../../glossary.md#term-descriptor) list; and
- the exact ordered two- or three-output descriptor list.

Planning remains responsible only for asking providers and selecting [ownership](../../../../glossary.md#term-ownership-meanings). This task changes
no capability provider and advertises no recurrent support. A Compiler [unit test](../../../../glossary.md#term-unit-test) may use an
explicit recording provider that accepts the query solely to prove unchanged handoff and
publication/[partition](../../../../glossary.md#partition) structure; that test provider is not a production capability claim. With
the current real providers unchanged, complete [lifecycle](../../../../glossary.md#lifecycle) availability remains false until a
later [backend](../../../../glossary.md#backend) task truthfully advertises and implements the family.

## [Autograd](../../../../glossary.md#autograd) and effect order

### Forward-only path

`GraphCompiler` retains its current order for `FORWARD_ONLY`:

```text
top-level request validation
  -> one identity-based capture
  -> recurrent-aware inference and validation
  -> mandatory canonicalization and validation
  -> optional exact rewrite/fold/DCE/CSE/DCE, validating each changed candidate
  -> forward publication roles
  -> existing ordinary Planning query when complete artifacts are requested
```

No stage reads [runtime](../../../../glossary.md#term-runtime) values or allocates physical state.

### Backward-capable path

`AutogradPreflight` must inspect the already allocation-free complete original forward [producer](../../../../glossary.md#term-producer)
inventory immediately after that inventory is built. For both `FORWARD_AND_BACKWARD` and
`TRAINING_STEP`, the first recurrent [occurrence](../../../../glossary.md#term-occurrence) in deterministic producer postorder fails with
occurrence context and an explanation that fixed recurrent [scan](../../../../glossary.md#term-cumulative-scan) is forward-only until [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) is
implemented.

The [guard](../../../../glossary.md#term-guard) applies even when a recurrent output is a separate forward root and the requested
[gradient](../../../../glossary.md#term-gradient) [targets](../../../../glossary.md#term-target) lie on another branch. This keeps the architecture statement exact:
`FORWARD_ONLY` is the sole current [compile mode](../../../../glossary.md#compile-mode--compilemode) that adopts recurrent scan. Request/null/
[canonical](../../../../glossary.md#term-canonical-form)-wrapper [validation](../../../../glossary.md#term-validation) that necessarily precedes construction of the complete inventory
retains its existing order; after the inventory exists, recurrent rejection precedes [seed](../../../../glossary.md#term-seed) checks,
stage selection, occurrence formula validation, and all [derivative](../../../../glossary.md#term-derivative) [Tensor](../../../../glossary.md#tensor) construction.

Failure must consume no new Tensor ID and must construct no default seed, explicit-seed
[normalization](../../../../glossary.md#term-normalization), typed splat, `ARGSORT`, local formula, gradient [accumulator](../../../../glossary.md#accumulator-type), combined [capture](../../../../glossary.md#term-capture-meanings), or
partial [backward graph](../../../../glossary.md#backward-graph).

### [Closed inventory](../../../../glossary.md#term-closed-inventory) boundary

Do not add `RecurrentScanKind` rows to `FirstOrderGradientCoverage.SIGNATURES`, do not add a
recurrent `FamilyOwner`, and do not change the exact 128 supported-row count. The coordinator-
authorized `FirstOrderGradientCoverageTest` boundary remains authoritative and unchanged:

- the exact recurrent RNN, GRU, and LSTM fingerprints are disjoint from the 128 supported
  signatures;<span data-glossary-navigation="true"> ([RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm))</span>
- their 131-row union equals the complete 38-family, 110-kind [Model](../../../../glossary.md#term-model) [signature](../../../../glossary.md#term-signature) inventory; and
- every recurrent output/input boundary classification remains `FC` with exact reason
  `unknown or unclassified operation kind/attributes pairing`, without [Tensor](../../../../glossary.md#tensor)-ID allocation.

This direct classifier boundary and the new complete-forward-inventory [preflight](../../../../glossary.md#term-preflight) [guard](../../../../glossary.md#term-guard) test
different contracts. Forward [inference](../../../../glossary.md#term-inference) support must not be mistaken for [gradient](../../../../glossary.md#term-gradient) coverage.
Valid lengths remain non-[differentiable](../../../../glossary.md#derivative-and-chain-rule), but marking only that role [`ND`](../../../../glossary.md#term-nd) would be insufficient:
all [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) formulas and saved-state policy remain absent.

## Out of [scope](../../../../glossary.md#term-scope)

- any change to `RecurrentScan`, [`Tensor`](../../../../glossary.md#tensor), [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), recurrent result records,
  `RecurrentScanKind`, `RecurrentDirection`, Model descriptors, signatures, semantics, tests,
  Javadocs, or Tensor IDs<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [descriptors glossary](../../../../glossary.md#term-descriptor); [signatures glossary](../../../../glossary.md#term-signature))</span>
- any `Tensor` recurrent receiver method, generic `scan(body)`, callback, lambda, cell body,
  nested <span data-glossary-navigation="true"> ([callback glossary](../../../../glossary.md#term-callback))</span>[`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel), graph region, block, loop IR, free-variable capture, or region
  ownership rule<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [capture glossary](../../../../glossary.md#term-capture-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- any change to current `RnnSequence`, `GruSequence`, `LstmSequence`, bidirectional containers,
  cells, parameters, static Java <span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter))</span>`long[]` length snapshots, compact output lists, static unrolling,
  or NN tests/documentation<span data-glossary-navigation="true"> ([unrolling glossary](../../../../glossary.md#term-unrolling))</span>
- BPTT, derivative formulas, recurrent gradient family owner, gradient signature adoption,
  saved gates, saved-state output slots, tape, checkpointing, recomputation, backward operations,
  second-order recurrent gradients, or Training behavior<span data-glossary-navigation="true"> ([BPTT glossary](../../../../glossary.md#term-backpropagation-through-time-bptt); [derivative glossary](../../../../glossary.md#term-derivative); [gradient glossary](../../../../glossary.md#term-gradient); [owner glossary](../../../../glossary.md#term-owner))</span>
- value reads, valid-length bounds checks, runtime input binding, dynamic Shapes, arbitrary masks,
  holes, active-row compaction, packed batches, sorting, padded-output materialization, or skipped-
  arithmetic implementation<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [binding glossary](../../../../glossary.md#term-binding); [Shapes glossary](../../../../glossary.md#shape); [masks glossary](../../../../glossary.md#term-boolean-mask); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [materialization glossary](../../../../glossary.md#materialization))</span>
- changes to Planning, Prepare, Runtime, Engine, backend-contract, any concrete backend, backend
  capability matrices, providers, routes, lowering, kernels, executables, memory declarations,
  schedules, tracing, serialization, or state dictionaries<span data-glossary-navigation="true"> ([matrices glossary](../../../../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [concrete backend glossary](../../../../glossary.md#concrete-backend); [backend capability glossary](../../../../glossary.md#backend-capability); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- capability advertisement or a claim that any current backend can plan, prepare, or execute a
  recurrent node<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node))</span>
- changes to dependencies, Gradle, architecture contracts or explanations, ADR 0012, architecture
  tests, backend conformance tests, integration tests, Tensor API, Training API, Model
  capabilities, or the global roadmap<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- redirecting or decomposing current static [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [graphs](../../../../glossary.md#graph) through the new [operation](../../../../glossary.md#operation)
- [Compiler](../../../../glossary.md#term-compiler) task 0007 [identities](../../../../glossary.md#term-identity-meanings) or any relaxed numerical rewrite
- a detailed specification for BPTT, a backend implementation, NN runtime-length integration, or
  any later task

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially fixed recurrent scan,
  Compiler ownership, flat graph identity, and lifecycle invariants<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [Compiler glossary](../../../../glossary.md#term-compiler); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [graph glossary](../../../../glossary.md#graph); [identity glossary](../../../../glossary.md#term-identity-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [ADR 0012: Fixed recurrent scan without graph regions](../../../../design/decisions/0012-fixed-recurrent-scan-without-regions.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Runtime, Prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)
- [Compiler 0001: Tensor expression graph capture](0001-tensor-expression-graph-capture.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction))</span>
- [Compiler 0002: Captured-graph inference and validation](0002-captured-graph-inference-and-validation.md)<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation))</span>
- [Compiler 0003: Canonicalization and forward optimization](0003-canonicalization-and-forward-optimization.md)<span data-glossary-navigation="true"> ([Canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- [Compiler 0004: Pre-capture autograd and combined graph compilation](0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md)<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [autograd glossary](../../../../glossary.md#autograd); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- [Compiler 0005: Publication, Planning orchestration, and artifacts](0005-publication-planning-orchestration-and-compile-artifacts.md)<span data-glossary-navigation="true"> ([Publication glossary](../../../../glossary.md#term-publication); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
- [Compiler 0005E: First-order coverage closure](0005e-first-order-gradient-coverage-closure-checkpoint.md)
- [Compiler 0006: Explicit functional gradients](0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md)<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient))</span>
- [Model 0025: Canonical producer outputs](../../model/tasks/0025-canonical-tensor-producer-outputs.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Canonical glossary](../../../../glossary.md#term-canonical-form); [producer glossary](../../../../glossary.md#term-producer))</span>
- [Model 0025E: Recurrent semantic family](../../model/tasks/0025e-fixed-recurrent-scan-semantic-family-and-tensor-expressions.md)
- [Model 0025F: Recurrent namespace correction](../../model/tasks/0025f-recurrent-scan-expression-namespace-correction.md)
- [NN 0021A: Fixed recurrent-scan architecture decision](../../../extensions/nn/tasks/0021a-fixed-recurrent-scan-architecture-decision.md)<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Glossary](../../../../glossary.md)

## Architecture constraints

- One fixed recurrent producer remains one ordinary flat compiled node. Compiler must not create
  a recurrent-specific graph representation or unroll by <span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [node glossary](../../../../glossary.md#node); [Compiler glossary](../../../../glossary.md#term-compiler))</span>`time`.
- The current flat meanings of [`Tensor`](../../../../glossary.md#tensor), [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), [`Operation`](../../../../glossary.md#operation), `CompiledNode`, and
  [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) remain unchanged.
- Model owns fixed transition semantics and descriptor-visible expression construction. Compiler
  owns independent captured inference, final validation, graph transformations, and fail-closed
  autograd.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [autograd glossary](../../../../glossary.md#autograd))</span>
- Recurrent direction and exact ordered inputs/outputs remain [immutable](../../../../glossary.md#term-immutable) [Model](../../../../glossary.md#term-model) operation facts.
- [Identity](../../../../glossary.md#term-identity-meanings)-distinct recurrent occurrences remain distinct through the final optimized [graph](../../../../glossary.md#graph).
- Compiler may remove an entirely dead occurrence but must preserve the complete structural output
  signature of every live multi-output occurrence.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [signature glossary](../../../../glossary.md#term-signature))</span>
- `FORWARD_ONLY` is the only current mode that may contain a recurrent occurrence. Backward-
  capable modes reject it during allocation-free preflight.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight))</span>
- Compiler output remains immutable compile-time state. Compiler allocates no physical buffer,
  prepared recipe, executable, runtime state, or backend-specific loop plan.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [buffer glossary](../../../../glossary.md#term-buffer); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Planning](../../../../glossary.md#term-planning) receives only the existing ordinary operation query and does not interpret recurrence.
- No current capability provider or [backend](../../../../glossary.md#backend) support claim changes.
- No dependency or module-boundary change is authorized. Compiler continues to avoid Runtime,
  Prepare, Engine, NN, Training, Data, and concrete-backend dependencies.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- `ARCHITECTURE.md` remains unchanged. Stop if implementation requires an architecture change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.compiler` — owns package-private graph inference, validation,
  optimization, autograd preflight, compile orchestration, and focused same-package tests.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [autograd glossary](../../../../glossary.md#autograd); [preflight glossary](../../../../glossary.md#term-preflight); [compile glossary](../../../../glossary.md#compile))</span>
- `io.github.pho001.synaptik.model.operation.recurrent` — supplies the current immutable kind and
  direction semantics; read but not changed.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- `io.github.pho001.synaptik.model.tensor` — supplies `RecurrentScan`, canonical Tensor producer
  outputs, and descriptors for boundary tests; only the <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [Tensor glossary](../../../../glossary.md#tensor); [producer glossary](../../../../glossary.md#term-producer); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>`RecurrentScan` type Javadoc changes,
  while its executable body and public surface remain unchanged.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- `io.github.pho001.synaptik.model.graph` — supplies the existing immutable ordinary compiled graph
  representation; read but not changed.<span data-glossary-navigation="true"> ([compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel))</span>
- `io.github.pho001.synaptik.planning.capability` — receives the existing ordinary operation query
  through current Compiler orchestration; read but not changed.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>

Packages added or changed:

- No package is added. The existing Compiler root package gains one cohesive family-owned
  inference helper and retains all implementation types package-private.

Type placement:

- `io.github.pho001.synaptik.compiler.RecurrentScanInference` — package-private stateless
  descriptor derivation for the closed recurrent family; it belongs beside the other
  family-specific inference helpers because Compiler owns graph validation.
- `io.github.pho001.synaptik.compiler.RecurrentScanInferenceTest` — same-package focused valid and
  invalid descriptor tests for the helper and outer final-validation context.
- `io.github.pho001.synaptik.compiler.RecurrentScanCompilerTest` — same-package vertical boundary
  tests for capture identity, CSE/DCE, forward compilation, Planning query handoff, and allocation-
  free backward rejection without widening production visibility.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [identity glossary](../../../../glossary.md#term-identity-meanings); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [Planning glossary](../../../../glossary.md#term-planning); [handoff glossary](../../../../glossary.md#term-handoff); [widening glossary](../../../../glossary.md#term-widening))</span>

`CapturedGraphInference`, `ForwardCommonSubexpressionElimination`, and `AutogradPreflight` remain
their existing package-private [owners](../../../../glossary.md#term-owner). No public [Compiler](../../../../glossary.md#term-compiler) type or package is added.

## Affected files

Expected production source:

- update
  `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CapturedGraphInference.java`
- add
  `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/RecurrentScanInference.java`
- update
  `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardCommonSubexpressionElimination.java`
- update
  `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java`

Expected tests:

- add
  `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/RecurrentScanInferenceTest.java`
- add
  `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/RecurrentScanCompilerTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning):

- update [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- update [Glossary](../../../../glossary.md), limited to the existing fixed-recurrent-[scan](../../../../glossary.md#term-cumulative-scan) entry
- update the `RecurrentScan` type Javadoc only; its executable body and public surface remain
  unchanged<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- update [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>, [Training API](../../../../api/training-api.md),
  and [Model capabilities](../../model/capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>, limited to stale [Compiler](../../../../glossary.md#term-compiler)-adoption status
- update [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span> and
  [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>, and
  [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>, limited to stale implementation-status
  wording; no architecture rule changes
- update and finalize this task specification
- update [Compiler master plan](../master-plan.md)
- update [Model master plan](../../model/master-plan.md), limited to stale Compiler 0006A status

Review without modification unless a conflict requires stopping:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCapture.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCompiler.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FirstOrderGradientCoverage.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FirstOrderGradientCoverageTest.java`
- other Model recurrent source, tests, and Javadocs; NN source/tests; other architecture/ADR
  documents and tests; Planning/Prepare/Runtime/Engine/backend source/tests and plans;
  dependencies; Gradle; conformance/integration tests; and the global roadmap<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most the exact eighteen paths listed under
[Affected files](#affected-files): four [Compiler](../../../../glossary.md#term-compiler) production paths, two Compiler test paths, one
[Model](../../../../glossary.md#term-model) [Javadoc](../../../../glossary.md#term-javadoc) path, and eleven documentation/[planning](../../../../glossary.md#term-planning) paths.

The original ten-path boundary omitted directly affected current-status statements. During the
independent documentation pass, the coordinator first authorized six documentation corrections,
then authorized the separately discovered focused [lifecycle](../../../../glossary.md#lifecycle) correction and the final stale Model
master-plan status correction. The resulting eighteen-path boundary remains one cohesive
Compiler capability:
[inference](../../../../glossary.md#term-inference) [dispatch](../../../../glossary.md#term-dispatch) and family logic, [identity](../../../../glossary.md#term-identity-meanings)-preserving optimization, allocation-free backward
gating, focused vertical tests, and the documentation/planning surfaces that state current
Compiler adoption. If implementation requires a nineteenth path, executable Model/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Planning/
[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime)/[Engine](../../../../glossary.md#term-engine)/[backend](../../../../glossary.md#backend) work, or another public type, stop and report the concrete gap
instead of expanding scope.

## Acceptance criteria

- `CapturedGraphInference` accepts every current `RecurrentScanKind`, both directions, and every
  bias-free/biased cardinality through one dedicated package-private inference helper.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [cardinality glossary](../../../../glossary.md#term-cardinality))</span>
- The helper independently derives the exact ordered two or three descriptors and produces no
  deferred constraint for valid fully static occurrences.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- Invalid manually assembled occurrences fail in deterministic semantic-role order for floating
  type, static rank/Shape, positive feature/hidden extent, valid-length type/gradient/Shape,
  state agreement, checked packed extent, weight/bias type and Shape, output count, and stored
  output descriptor mismatches.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [Shape glossary](../../../../glossary.md#shape); [feature glossary](../../../../glossary.md#term-feature); [extent glossary](../../../../glossary.md#term-extent); [weight glossary](../../../../glossary.md#term-weight))</span>
- [Inference](../../../../glossary.md#term-inference) never reads [Tensor](../../../../glossary.md#tensor) storage or valid-length values and never validates [runtime](../../../../glossary.md#term-runtime) bounds.
- Each `RecurrentScan` [producer](../../../../glossary.md#term-producer) [captures](../../../../glossary.md#term-capture-meanings) as exactly one ordinary flat `CompiledNode` with the exact
  operation reference, direction, ordered 5/6 or 6/7 input IDs, ordered 2/3 output IDs, exact
  descriptors, and no time-dependent topology.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [topology glossary](../../../../glossary.md#term-topology))</span>
- Two canonical output wrappers from one producer deduplicate to one node; two structurally equal
  but producer-identity-distinct calls remain two nodes.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [node glossary](../../../../glossary.md#node); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Recurrent nodes are ineligible for CSE in every phase/order path. Other exact CSE cases retain
  their existing behavior.<span data-glossary-navigation="true"> ([CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>
- DCE removes a wholly dead recurrent occurrence but retains a live occurrence and every declared
  sibling slot when only one output is a root or consumer input.<span data-glossary-navigation="true"> ([DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Multi-root capture and final compilation preserve caller output order, output-slot identity,
  and distinct producer occurrences.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- `FORWARD_ONLY` graph compilation succeeds through capture, initial inference, canonicalization,
  optional optimization, and final validation for RNN, GRU, and LSTM examples, including zero
  time and zero batch descriptors.<span data-glossary-navigation="true"> ([canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [validation glossary](../../../../glossary.md#term-validation); [RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Complete Compiler orchestration with an explicit recording test provider passes the exact
  recurrent operation and descriptor order through the existing [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery),
  publication binding, owner map, partition, logical memory, and diagnostics without adding a
  production capability claim.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [binding glossary](../../../../glossary.md#term-binding); [partition glossary](../../../../glossary.md#partition))</span>
- Every backward-capable compile whose complete original forward inventory contains a recurrent
  producer fails in deterministic producer postorder during <span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>`AutogradPreflight`, including when
  the requested gradient lies on a separate non-recurrent branch.
- Backward rejection leaves the next Tensor ID unchanged and occurs before any seed, derivative
  constant, formula Tensor, matching <span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [derivative glossary](../../../../glossary.md#term-derivative); [constant glossary](../../../../glossary.md#term-constant))</span>`ARGSORT`, accumulated [gradient](../../../../glossary.md#term-gradient), or combined [graph](../../../../glossary.md#graph) exists.
- `FirstOrderGradientCoverage.SIGNATURES`, its 128-row contract, its family-[owner](../../../../glossary.md#term-owner) enum, and
  `FirstOrderGradientCoverageTest` remain unchanged. The exact three recurrent signatures remain
  deferred and keep their exact unknown/unclassified no-allocation assertions.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- Current static NN recurrent containers, Model public API/semantics, and every existing capability
  provider remain unchanged.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Model glossary](../../../../glossary.md#term-model); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Compiler and Model Javadocs, Compile/Tensor/Training API, Model capabilities, glossary, and the
  three focused architecture explanations state that forward Compiler adoption is current while
  BPTT, execution, runtime length reads, backend capability, and NN delegation remain planned.<span data-glossary-navigation="true"> ([BPTT glossary](../../../../glossary.md#term-backpropagation-through-time-bptt); [backend capability glossary](../../../../glossary.md#backend-capability))</span>
- Production Javadocs explain recurrent inference, CSE identity exclusion, and allocation-free
  preflight behavior with complete <span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight))</span>`@param`, `@return`, and expected `@throws` tags on new or
  changed contract-relevant methods.
- The documentation pass records reasoned no-change conclusions for other Model recurrent
  Javadocs and tests, the authoritative architecture contract and ADR 0012, other NN/Planning/
  Prepare/Runtime/Engine/backend docs, architecture/conformance/integration tests, Gradle,
  dependencies, and the global roadmap.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Exactly the eighteen authorized paths change. No file is staged, and concurrent CPU/backend/Data or
  global-roadmap changes are neither touched nor incorporated.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Task 0006A and its [Compiler](../../../../glossary.md#term-compiler) master-plan row are [`Ready`](../../../../glossary.md#term-ready) before implementation and become
  `Complete` only after implementation, focused/module validation, and the separate documentation
  pass. Compiler 0007 remains Draft without a detailed specification; no BPTT task specification
  is created.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- A separate documentation-focused agent pass has independently finalized affected Javadocs,
  Compile API, glossary impact, planning status/evidence, links, and terminology in the same
  overall change.

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, [run](../../../../glossary.md#term-run) the focused [Compiler](../../../../glossary.md#term-compiler) capability tests:

```bash
./gradlew :modules:compiler:test \
  --tests io.github.pho001.synaptik.compiler.RecurrentScanInferenceTest \
  --tests io.github.pho001.synaptik.compiler.RecurrentScanCompilerTest \
  --tests io.github.pho001.synaptik.compiler.FirstOrderGradientCoverageTest
```

The focused command must prove all valid kinds/directions/cardinalities, invalid [descriptor](../../../../glossary.md#term-descriptor)
categories and final mismatch context, [identity](../../../../glossary.md#term-identity-meanings)-distinct [capture](../../../../glossary.md#term-capture-meanings), multi-root order, [CSE](../../../../glossary.md#term-common-subexpression-elimination-cse)/[DCE](../../../../glossary.md#term-dead-code-elimination-dce),
forward-only optimized [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names), ordinary [Planning](../../../../glossary.md#term-planning)-query [handoff](../../../../glossary.md#term-handoff), exact deferred-[gradient](../../../../glossary.md#term-gradient)
[partition](../../../../glossary.md#partition), and allocation-free rejection for both backward-capable modes.

Run the existing [Model](../../../../glossary.md#term-model) boundary tests once to verify that Compiler consumes the committed public
construction seam and [signature](../../../../glossary.md#term-signature)/result contracts without changing Model:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.RecurrentScanExpressionTest
```

After executable Compiler code stabilizes, run one final affected-[module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :modules:compiler:test
```

Record [Gradle](../../../../glossary.md#term-gradle) outcomes and [JUnit](../../../../glossary.md#term-junit) XML suite/test/skip/failure/error counts. Do not repeat the full
Model suite because no Model executable source changes. Repository-wide and architecture-test
validation is deferred to [continuous integration](../../../../glossary.md#term-continuous-integration-ci): this task changes one module, no [dependency](../../../../glossary.md#term-dependency),
no architecture rule, no shared build configuration, and no executable [backend](../../../../glossary.md#backend) or end-to-end
path.

The implementation context hands the exact focused, Model-boundary, and final Compiler evidence
to a separate clean documentation context. That pass must not rerun successful Java tests unless
it changes executable behavior or records a concrete stale-evidence risk.

Documentation pass:

```bash
./gradlew :modules:compiler:javadoc
./gradlew :modules:model:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git diff --cached --name-only
git status --short
```

The documentation pass must also verify generated package-private Compiler Javadocs and rendered
`RecurrentScan` [Javadoc](../../../../glossary.md#term-javadoc); complete `@param`, `@return`, and expected `@throws` coverage; local
Markdown links and anchors; balanced fences; heading uniqueness; final newlines; trailing
whitespace; exact eighteen-path [scope](../../../../glossary.md#term-scope); unchanged Model [API](../../../../glossary.md#term-application-programming-interface-api) and 128-plus-three gradient boundary;
no recurrent region/body/receiver spelling; no capability/backend claim; no changes outside the
authorized Compiler, Model Javadoc, and explanatory/planning pages;
task/master status [synchronization](../../../../glossary.md#term-synchronization); Compiler 0007 Draft with no specification; no [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) task
specification; and an empty Git index.

## Dependencies

- Accepted [ADR 0012](../../../../design/decisions/0012-fixed-recurrent-scan-without-regions.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
  and completed [NN 0021A](../../../extensions/nn/tasks/0021a-fixed-recurrent-scan-architecture-decision.md)<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
  supply the fixed-family/no-region architecture — Complete.
- [Model 0025](../../model/tasks/0025-canonical-tensor-producer-outputs.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span> supplies canonical
  producer output wrappers — Complete.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [producer glossary](../../../../glossary.md#term-producer))</span>
- [Model 0025E](../../model/tasks/0025e-fixed-recurrent-scan-semantic-family-and-tensor-expressions.md)
  supplies fixed semantics, signatures, descriptors, and results — Complete.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- [Model 0025F](../../model/tasks/0025f-recurrent-scan-expression-namespace-correction.md) supplies
  the corrected public stateless `RecurrentScan` construction seam and exact supported/deferred
  boundary test — Complete and the final Model prerequisite.
- Compiler 0001–0003B supply identity-based capture, inference/final-validation orchestration,
  canonicalization, exact optimization, DCE/CSE, and constant sidecars — Complete.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [constant glossary](../../../../glossary.md#term-constant))</span>
- Compiler 0004–0005E supply allocation-free autograd preflight, combined capture, complete
  artifacts/Planning orchestration, and the closed first-order coverage checkpoint — Complete.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [preflight glossary](../../../../glossary.md#term-preflight); [artifacts glossary](../../../../glossary.md#term-artifact); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- [Compiler 0006](0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md)
  supplies the current one/two-stage functional request and backward-capable entry behavior —
  Complete.
- Existing Compiler dependencies on Model, Config, Planning, Backend Contract, and Trace are
  sufficient. No new dependency is required.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Trace glossary](../../../../glossary.md#trace); [dependency glossary](../../../../glossary.md#term-dependency))</span>

Within the [Compiler](../../../../glossary.md#term-compiler) master plan, 0006A is the first unfinished task and the next applicable
[module](../../../../glossary.md#term-module) frontier. The repository roadmap still records the separate [CPU](../../../../glossary.md#term-cpu) 0007A1G execution
frontier; this user-requested [planning](../../../../glossary.md#term-planning) pass does not edit that concurrent roadmap or itself
authorize Compiler implementation ahead of the coordinator's execution order.

## Follow-up tasks

- A later explicit Compiler architecture and implementation task must design BPTT, including
  derivative equations, valid-length gradient routing, saved-state versus recomputation policy,
  any necessary canonical auxiliaries, higher-order boundary, and complete first-order inventory
  adoption. No ID or detailed specification is created here.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [BPTT glossary](../../../../glossary.md#term-backpropagation-through-time-bptt); [derivative glossary](../../../../glossary.md#term-derivative); [gradient glossary](../../../../glossary.md#term-gradient); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- A later concrete-backend task must truthfully advertise exact supported recurrent kind,
  direction, type, and Shape combinations and implement one-time lowering, resources, complete
  runtime length validation, dense padded outputs, final states, and skipped invalid-coordinate
  arithmetic.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [Shape glossary](../../../../glossary.md#shape); [lowering glossary](../../../../glossary.md#lowering); [runtime glossary](../../../../glossary.md#term-runtime); [validation glossary](../../../../glossary.md#term-validation); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Existing future Engine work owns typed logical input and publication binding; later NN/Data work
  owns runtime-valid-length convenience and compatibility with static sequence containers.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [publication glossary](../../../../glossary.md#term-publication); [binding glossary](../../../../glossary.md#term-binding); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Compiler 0007 remains the separate Draft owner of exact constant identities and permission-aware
  algebra. It is not advanced or specified by this task.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [constant glossary](../../../../glossary.md#term-constant))</span>
- Dynamic Shapes, arbitrary masks, active-row compaction, serialization, general graph regions,
  and static-NN migration remain separate later decisions.<span data-glossary-navigation="true"> ([masks glossary](../../../../glossary.md#term-boolean-mask); [serialization glossary](../../../../glossary.md#term-serialization); [graph glossary](../../../../glossary.md#graph))</span>

## Architecture impact

Expected impact: None.

This task implements the already accepted [Compiler](../../../../glossary.md#term-compiler) portion of [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0012 using the existing flat
[graph](../../../../glossary.md#graph), [inference](../../../../glossary.md#term-inference), optimization, [publication](../../../../glossary.md#term-publication), and [Planning](../../../../glossary.md#term-planning) boundaries. It changes no architecture
rule, [ownership](../../../../glossary.md#term-ownership-meanings), [dependency direction](../../../../glossary.md#java-platform-and-build-tools), public [Model](../../../../glossary.md#term-model) surface, or execution contract. If correct
implementation requires a new graph representation, public Compiler [API](../../../../glossary.md#term-application-programming-interface-api), shared loop/body
contract, another [module](../../../../glossary.md#term-module) change, or capability advertisement, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, ADR 0012, docs/planning/planning-guide.md, the Compiler master
plan, and docs/planning/modules/compiler/tasks/0006a-fixed-recurrent-scan-forward-adoption-and-bptt-boundary.md.
Read the task's directly referenced Model recurrent, Compiler capture/inference/optimization/
autograd/artifact, Planning handoff, source, tests, and documentation contracts.

Implement Compiler 0006A exactly within its eighteen authorized paths. Preserve the existing ordinary
CompiledNode representation and the exact 128-supported-plus-three-deferred gradient boundary.
Do not implement BPTT, a region/body/loop IR, executable Model/NN/API changes, runtime value reads, capability
advertisement, backend execution, another module, dependencies, Gradle, architecture, or global-
roadmap changes. Stop on any architecture, inventory, or scope conflict.

After executable implementation and the recorded focused, Model-boundary, and final Compiler
validation, hand the actual diff and exact evidence to a separate documentation-focused agent or
thread with clean context. That pass must follow docs/developer-guide/documentation-rules.md and
finalize affected Javadocs, Compile API, glossary impact, planning status/evidence, and
documentation validation in the same overall change without repeating successful Java tests
unless executable behavior changes or a concrete stale-evidence risk is recorded.

Update this task with local decisions, exact validation evidence, implementation notes,
completion summary, and final status. Do not mark it Complete before every acceptance criterion
and the documentation pass finish.
```

## Local decisions

- The existing `CompiledNode` is the complete recurrent compiled representation. No new IR type
  or operation wrapper is authorized.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [operation glossary](../../../../glossary.md#operation))</span>
- Recurrent inference is a dedicated family helper because its ordered multi-state/parameter
  descriptor contract is cohesive and should not enlarge the already broad structured helper.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [parameter glossary](../../../../glossary.md#term-parameter); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Recurrent occurrences are excluded from CSE to preserve the architecture's identity-distinct
  occurrence contract; DCE retains its existing whole-node liveness rules.<span data-glossary-navigation="true"> ([CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [identity glossary](../../../../glossary.md#term-identity-meanings); [occurrence glossary](../../../../glossary.md#term-occurrence); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce); [node glossary](../../../../glossary.md#node); [liveness glossary](../../../../glossary.md#term-liveness))</span>
- Backward-capable modes reject any recurrent occurrence in the complete original forward
  inventory, not only a selected gradient route, because the architecture makes <span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>`FORWARD_ONLY`
  the sole initial adopting mode.
- The first-order production gradient inventory and its exact boundary test remain unchanged.
  Forward inference inventory and gradient coverage inventory intentionally differ.
- A recording test provider may prove the unchanged ordinary Planning handoff but cannot create a
  production capability or backend claim.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [handoff glossary](../../../../glossary.md#term-handoff); [backend glossary](../../../../glossary.md#backend))</span>

## Known limitations

- No current production capability provider is expected to accept a recurrent query, so this task
  does not make the operation executable through a real backend or public Engine facade.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [backend glossary](../../../../glossary.md#backend); [Engine glossary](../../../../glossary.md#term-engine); [facade glossary](../../../../glossary.md#term-facade))</span>
- Runtime valid-length contents remain unchecked until a concrete backend owns complete pre-write
  validation.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [concrete backend glossary](../../../../glossary.md#concrete-backend); [validation glossary](../../../../glossary.md#term-validation))</span>
- Every recurrent [Shape](../../../../glossary.md#shape) is fully static; dynamic or [binding](../../../../glossary.md#term-binding)-dependent Shapes remain unsupported.
- [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) and every recurrent [derivative](../../../../glossary.md#term-derivative) role remain [fail-closed](../../../../glossary.md#term-fail-closed).
- Existing [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) sequences continue to construct static [unrolled](../../../../glossary.md#backpropagation-through-time-and-recurrent-state) [graphs](../../../../glossary.md#graph) and are not redirected.

## [Validation](../../../../glossary.md#term-validation) evidence

- Focused [Compiler](../../../../glossary.md#term-compiler) validation passed after one test-[fixture](../../../../glossary.md#term-fixture) correction:

  ```text
  ./gradlew :modules:compiler:test \
    --tests io.github.pho001.synaptik.compiler.RecurrentScanInferenceTest \
    --tests io.github.pho001.synaptik.compiler.RecurrentScanCompilerTest \
    --tests io.github.pho001.synaptik.compiler.FirstOrderGradientCoverageTest
  BUILD SUCCESSFUL; 3 suites, 21 tests, 0 skipped, 0 failures, 0 errors
  ```

  The first invocation exposed one invalid test setup that attempted to represent a gradient-
  eligible <span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>[`INT64`](../../../../glossary.md#data-type--datatype) [descriptor](../../../../glossary.md#term-descriptor), which `TensorDescriptor` itself correctly rejects. The test was
  corrected to preserve that upstream invariant, and the exact focused command then passed.<span data-glossary-navigation="true"> ([invariant glossary](../../../../glossary.md#term-invariant))</span>
- The required committed [Model](../../../../glossary.md#term-model) boundary passed without changing Model:

  ```text
  ./gradlew :modules:model:test \
    --tests io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanSemanticsTest \
    --tests io.github.pho001.synaptik.model.tensor.RecurrentScanExpressionTest
  BUILD SUCCESSFUL; 2 suites, 15 tests, 0 skipped, 0 failures, 0 errors
  ```

- The one authoritative affected-[module](../../../../glossary.md#term-module) suite passed after executable freeze:

  ```text
  ./gradlew :modules:compiler:test
  BUILD SUCCESSFUL; 33 suites, 224 tests, 0 skipped, 0 failures, 0 errors
  ```

- Final documentation and generated-surface validation passed:

  ```text
  ./gradlew :modules:compiler:javadoc
  BUILD SUCCESSFUL
  ./gradlew :modules:model:javadoc
  BUILD SUCCESSFUL
  javap -classpath modules/compiler/build/classes/java/main -p \
    io.github.pho001.synaptik.compiler.RecurrentScanInference
  javac external six-overload RecurrentScan surface probe
  javac/java package-local Compiler inventory/reflection probe
  supported=128, deferred=3, inference=final, inferMethods=1
  ```

  `javap` shows one package-private final type, one private constructor, and one package-private
  static `infer(Operation, List<TensorDescriptor>, int)` entry. Generated package-private
  Javadocs contain the recurrent inference contract, CSE identity exclusion, and allocation-free
  BPTT boundary, and rendered Model Javadocs contain the corrected <span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [identity glossary](../../../../glossary.md#term-identity-meanings); [BPTT glossary](../../../../glossary.md#term-backpropagation-through-time-bptt))</span>`RecurrentScan` forward/backward
  status. The external compile accepts all six unchanged public biased or bias-free overloads.
  The reflection/inventory probe confirms the package-private final inference owner, one <span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [bias glossary](../../../../glossary.md#term-bias); [owner glossary](../../../../glossary.md#term-owner))</span>`infer`
  entry, 128 supported gradient rows, and exactly three recurrent signatures. The production
  coverage source and corrected <span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>`FirstOrderGradientCoverageTest` have no diff.
- Initial implementation scope and hygiene checks found the original ten authorized unstaged
  paths, an empty Git index, no forbidden production imports or added dependency/build/
  architecture/backend path, no Compiler 0007 specification, and no BPTT task specification.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [dependency glossary](../../../../glossary.md#term-dependency); [backend glossary](../../../../glossary.md#backend))</span>
  `git diff --check` passed. The independent documentation pass then found six stale current-
  status statements directly invalidated by forward adoption; the coordinator authorized those
  exact documentation-only corrections and expanded scope to sixteen paths. A later complete
  focused-doc scan found the same stale statement in the lifecycle explanation; the coordinator
  authorized that exact seventeenth path. The final exhaustive relevant-surface scan found stale
  current-status wording in the Model master plan; the coordinator authorized that exact
  eighteenth path. No other current claim says 0006A is Draft, forward recurrent inference
  rejects, or Compiler adoption is future. The completed Model 0025F row remains an accurate
  historical summary of what that earlier task preserved at its own completion.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Final documentation and repository-hygiene gates passed:

  ```text
  python3 /tmp/validate_synaptik_markdown.py
  validated 11 Markdown files: links, anchors, headings, fences, whitespace
  git diff --check
  exactly 18 changed or untracked task paths
  git diff --cached --name-only
  <empty>
  ```

  The exact path probe finds four Compiler production files, two Compiler tests, one Model
  Javadoc file, and eleven documentation/planning files. Forbidden-import, unchanged-contract,
  status, no-0007-specification, and no-separate-BPTT-specification probes also pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>
- Repository-wide, architecture-test, backend-conformance, integration-test, and full Model-suite
  validation remain intentionally deferred to continuous integration under the task's tier rules:
  no architecture boundary, dependency, shared build, backend behavior, or end-to-end executable
  path changed.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [continuous integration glossary](../../../../glossary.md#term-continuous-integration-ci))</span>

## Implementation notes

- `CapturedGraphInference` now dispatches the closed recurrent family to the new stateless
  `RecurrentScanInference`. The helper validates exact kinds, direction attributes,
  cardinalities, common floating type, fully static role Shapes, positive feature/state extents,
  checked packed gate extent, non-gradient <span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [feature glossary](../../../../glossary.md#term-software-feature); [extent glossary](../../../../glossary.md#term-extent); [gradient glossary](../../../../glossary.md#term-gradient))</span>`INT64[batch]` valid lengths, weights/biases, and
  ordered output descriptors without reading values or creating constraints.<span data-glossary-navigation="true"> ([weights glossary](../../../../glossary.md#term-weight); [biases glossary](../../../../glossary.md#term-bias); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- [Capture](../../../../glossary.md#term-capture-meanings) continues to emit one ordinary flat `CompiledNode` per Model producer. Recurrent nodes
  are explicitly excluded from CSE, while existing whole-node DCE retains every sibling output
  when any slot is live and removes only fully dead occurrences. Multi-root slot identity and
  caller order remain intact.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [nodes glossary](../../../../glossary.md#node); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Ordinary publication, owner selection, partitioning, logical-memory planning, and diagnostics
  receive the exact recurrent operation and ordered descriptors through the existing<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [owner glossary](../../../../glossary.md#term-owner); [planning glossary](../../../../glossary.md#term-planning); [operation glossary](../../../../glossary.md#operation))</span>
  [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery). No production provider advertises recurrence and no execution path
  was added.
- `AutogradPreflight` inventories the complete original forward graph and rejects the first
  recurrent producer in deterministic producer postorder before request-stage, seed, or formula
  validation. Both backward-capable modes preserve the next Tensor ID on rejection, including
  when the requested gradient belongs to a separate non-recurrent branch.<span data-glossary-navigation="true"> ([forward graph glossary](../../../../glossary.md#forward-graph); [seed glossary](../../../../glossary.md#term-seed); [validation glossary](../../../../glossary.md#term-validation))</span>
- Focused tests cover all three kinds, both directions, both bias arities, exact capture and query
  order, zero time/batch, invalid descriptors, final-validation context, producer identity, CSE,
  whole-node DCE, multi-root slots, unrelated CSE behavior, both backward modes, deterministic
  diagnostics, and no-allocation rejection.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- [Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api), [Tensor](../../../../glossary.md#tensor) API, Training API, [Model](../../../../glossary.md#term-model) capabilities, glossary, `RecurrentScan` Javadoc,
  module boundaries, training graph, lifecycle, and Model master-plan wording describe current
  forward adoption and the unchanged backward/execution boundary. The independent documentation
  pass added only the eight coordinator-authorized stale-status corrections beyond the original
  ten paths. Other Model recurrent Javadocs/tests and planning records, the authoritative
  architecture contract and ADR 0012, NN, Planning, Prepare, Runtime, Engine, backend,
  conformance/integration tests, Gradle, dependencies, and the global roadmap need no change
  because their contracts, historical records, or current status were not altered by this task.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [module glossary](../../../../glossary.md#term-module); [training graph glossary](../../../../glossary.md#training-graph); [lifecycle glossary](../../../../glossary.md#lifecycle); [architecture contract glossary](../../../../glossary.md#architecture-contract); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Completion summary

Implemented, executable-tested, independently documented, and validated the fixed recurrent-[scan](../../../../glossary.md#term-cumulative-scan)
forward [Compiler](../../../../glossary.md#term-compiler) boundary in exactly eighteen authorized paths: four Compiler production files,
two focused Compiler tests, one [Model](../../../../glossary.md#term-model) [Javadoc](../../../../glossary.md#term-javadoc) file, and eleven documentation/[planning](../../../../glossary.md#term-planning) files. The
successful Java evidence remains the frozen focused 21-test, Model-boundary 15-test, and full
Compiler 224-test results; no executable Java changed during the documentation pass, so those
suites were not repeated. Both Javadoc tasks, generated-page inspection, public external [compile](../../../../glossary.md#compile),
package-local reflection/inventory, Markdown, stale-status, import, forbidden-[scope](../../../../glossary.md#term-scope), exact-path,
whitespace, and empty-index gates pass. No blocker or follow-up remains within task 0006A.

Status: Complete
