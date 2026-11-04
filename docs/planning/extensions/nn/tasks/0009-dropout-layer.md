# Task 0009: [Dropout](../../../../glossary.md#dropout--inverted-dropout) Layer

> **Editorial reading context — not a status update.** The application explicitly threads
> graph random-number state into Dropout.forward and keeps the returned next state for a later
> construction. The layer owns neither a generator nor a hidden advancing counter. A supplied
> ForwardContext chooses training construction or evaluation's exact-reference bypass.
> Training's result describes output and next-state positions of one producer; no draw happens
> merely because forward returns. Today's [explicit-state dropout](../../../../reference/tensor/graph-randomness.md#explicit-state-dropout-construction)
> explains those positions and [state advancement](../../../../reference/tensor/graph-randomness.md#state-advancement-branching-and-edge-cases)
> explains branching versus threading, without extending this historical layer's execution evidence.

## Status

Complete

## Goal

Add one final public mode-sensitive [`Dropout`](../../../../glossary.md#dropout--inverted-dropout) [module](../../../../glossary.md#term-module) that composes the existing explicit-state
[Model](../../../../glossary.md#term-model) dropout [operation](../../../../glossary.md#operation) without owning random state. Every forward call receives the input
[`Tensor`](../../../../glossary.md#tensor), the caller's [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate), and an [immutable](../../../../glossary.md#term-immutable) `ForwardContext`. Training creates one
Model inverted-dropout [occurrence](../../../../glossary.md#term-occurrence) and returns its output and next state. [Evaluation](../../../../glossary.md#term-evaluation) creates no
Model operation and returns the exact input and exact incoming state references.

The existing Model `DropoutResult` is deliberately training-occurrence-specific: its contract
says that output and next state select slots zero and two of one shared dropout [producer](../../../../glossary.md#term-producer). An
evaluation bypass has no such producer. This task therefore adds one narrow [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)-owned
`DropoutForwardResult` that truthfully represents both branches without broadening or weakening
the Model contract.

[Mental model](../../../../glossary.md#term-mental-model):

```text
stored validated drop probability
  + caller input
  + caller graph RNG state
  + immutable forward-mode snapshot
    -> TRAINING:   input.dropout(probability, state)
                     -> exact Model output and next-state references
                     -> fresh NN DropoutForwardResult
    -> EVALUATION: no Model operation
                     -> fresh NN DropoutForwardResult(input, same state)
```

This is symbolic expression composition and explicit caller-owned [state threading](../../../../glossary.md#state-threading). It is not an
[eager](../../../../glossary.md#term-eager-evaluation) draw, execution, hidden generator, [mutable](../../../../glossary.md#term-mutable) counter, [parameter](../../../../glossary.md#term-parameter)/[buffer](../../../../glossary.md#term-buffer) transition, [optimizer
step](../../../../glossary.md#optimizer-and-stochastic-gradient-descent), or [serialization](../../../../glossary.md#term-serialization) contract.

## [Scope](../../../../glossary.md#term-scope)

- Add final public `io.github.pho001.synaptik.nn.layers.Dropout` extending [`Module`](../../../../glossary.md#term-module).
- Add public final record
  `io.github.pho001.synaptik.nn.layers.DropoutForwardResult(Tensor output,
  GraphRngState nextState)` as the exact typed result of both forward branches.
- Add exactly the constructor and forward method in the public API table. Add no overload,
  default probability, probability/rate alias, accessor, builder, options object, functional
  convenience, or generic layer interface.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Represent the fraction of positions to drop as one primitive `double probability`. Validate it
  through the existing public Model <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>`DropoutAttrs` intrinsic contract: it must be finite and
  numerically in <span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite))</span>`[0.0, 1.0)`. Preserve either signed zero exactly and store the validated
  primitive without normalization or conversion.<span data-glossary-navigation="true"> ([signed zero glossary](../../../../glossary.md#term-signed-zero); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Make the layer parameterless and bufferless. It declares no [`Parameter`](../../../../glossary.md#term-parameter), [`Buffer`](../../../../glossary.md#term-buffer), or child and
  adds no entry to direct or recursive module-state discovery.
- Receive a non-null [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate) explicitly on every forward call and return a non-null next
  state explicitly. The layer must never construct, seed, split, clone, retain between calls,
  advance, replace, register, or consult a hidden random state.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed))</span>
- Receive a non-null [immutable](../../../../glossary.md#term-immutable) `ForwardContext` on every call. Its captured mode alone selects the
  branch, even when it came from another module or differs from this layer's current inherited
  mode.
- In training, call exactly `input.dropout(probability, state)` once. Return a fresh NN result
  retaining the exact <span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>`output()` and `nextState()` references of that one Model result. Do not
  expose or reconstruct the keep mask, output Tensors, state Tensor, producer, or attributes.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask); [Tensors glossary](../../../../glossary.md#tensor); [producer glossary](../../../../glossary.md#term-producer))</span>
- In evaluation, construct no dropout or other Model occurrence. Return a fresh NN result whose
  output is the exact input Tensor reference and whose next state is the exact incoming<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
  `GraphRngState` reference.
- Let evaluation accept every non-null Tensor data type and Shape because bypass performs no
  dropout operation. Training retains the Model requirement that input data type be floating.
  Do not add a false common floating-input precondition to the identity branch.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [operation glossary](../../../../glossary.md#operation); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Preserve inherited `Module` mode behavior: a new layer starts in training mode, `train()` and
  `eval()` update inherited module mode, and `forwardContext()` snapshots it. The forward method
  never reads `mode()` implicitly; only the supplied context selects that call.
- Finalize the affected layer/package Javadocs and glossary terminology in the required separate
  clean documentation context before the implementation task becomes Complete.

## Out of [scope](../../../../glossary.md#term-scope)

- Any change to `Tensor.dropout`, `DropoutAttrs`, `DropoutResult`, [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate),
  `ForwardContext`, `ForwardMode`, or [`Module`](../../../../glossary.md#term-module).
- Broadening [Model](../../../../glossary.md#term-model) `DropoutResult` to describe a no-[producer](../../../../glossary.md#term-producer) [evaluation](../../../../glossary.md#term-evaluation) bypass.
- A hidden or retained [seed](../../../../glossary.md#term-seed), generator, key, counter, state [Tensor](../../../../glossary.md#tensor), `RandomGenerator`, thread-local
  source, process-global source, default state, or automatic state split.
- Calling `GraphRngState.initial`, inspecting its private Tensor, inventing public key/counter
  access, or adding an arbitrary-wrap/copy/advance API.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- A parameter, buffer, child module, mutable state binding, registration name, state-dictionary
  entry, checkpoint field, serialization token, or optimizer state.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [buffer glossary](../../../../glossary.md#term-buffer); [mutable glossary](../../../../glossary.md#term-mutable); [binding glossary](../../../../glossary.md#term-binding); [serialization glossary](../../../../glossary.md#term-serialization); [token glossary](../../../../glossary.md#term-token); [optimizer state glossary](../../../../glossary.md#term-optimizer-state))</span>
- An eager random draw, host storage access, Tensor mutation, sampling algorithm, portable
  bitstream, numerical scaling/masking, compiler capture, lowering, preparation, backend route,
  runtime publication, or execution.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [host storage glossary](../../../../glossary.md#host-storage); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [lowering glossary](../../../../glossary.md#lowering); [backend route glossary](../../../../glossary.md#backend-route); [runtime glossary](../../../../glossary.md#term-runtime); [publication glossary](../../../../glossary.md#term-publication))</span>
- A public keep mask, saved mask, producer, raw Model result, mode flag, probability field, or
  branch tag in <span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask))</span>`DropoutForwardResult`.
- A `forward(Tensor)` overload that reads implicit module mode, a no-state overload, a
  training-only API, a generic stochastic-module interface, or the unary composition contract
  reserved for task 0011.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary))</span>
- Thread-safety, atomicity, synchronization, rollback, identifier rollback, batching, session,
  gradient publication, optimizer, parameter-group, or training-step claims.<span data-glossary-navigation="true"> ([atomicity glossary](../../../../glossary.md#term-atomicity); [synchronization glossary](../../../../glossary.md#term-synchronization); [rollback glossary](../../../../glossary.md#term-rollback); [gradient glossary](../../../../glossary.md#term-gradient); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- Any new gradient formula, autograd rule, operation kind, Tensor semantics, backend support,
  conformance test, integration test, dependency, or architecture rule.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [operation glossary](../../../../glossary.md#operation); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [integration test glossary](../../../../glossary.md#term-integration-test); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- State dictionary/[checkpoint](../../../../glossary.md#term-checkpoint) work from task 0010 or `Sequential`/shared unary composition from
  task 0011.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [NN master plan](../master-plan.md)<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- [Model master plan](../../../modules/model/master-plan.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model task 0019B: Explicit graph RNG state foundation](../../../modules/model/tasks/0019b-explicit-graph-rng-state-foundation.md)<span data-glossary-navigation="true"> ([graph RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- [Model task 0019B1: Explicit graph dropout construction](../../../modules/model/tasks/0019b1-explicit-graph-dropout-construction.md)<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>

## Architecture constraints

- `extensions/nn` owns train/eval layer composition and depends only on `modules/model`.
- Generic dropout meaning, attributes, explicit graph RNG state, producer topology, and the
  training-occurrence result stay in <span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [graph RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [producer glossary](../../../../glossary.md#term-producer); [topology glossary](../../../../glossary.md#term-topology); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>`modules/model`.
- The layer must not make Model depend on NN and must not depend on Training, Compiler, Runtime,
  Prepare, Engine, CPU, Metal, CUDA, or another backend.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Compiler glossary](../../../../glossary.md#term-compiler); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda); [backend glossary](../../../../glossary.md#backend))</span>
- NN module state consists only of declared parameters, buffers, and children. This layer has
  none; graph RNG state remains a caller-threaded Model value rather than module state.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [parameters glossary](../../../../glossary.md#term-parameter); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- `ForwardContext` is composition metadata, not runtime state. The supplied immutable snapshot is
  authoritative for a call and need not originate from the receiving layer.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [immutable glossary](../../../../glossary.md#term-immutable); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Every successful training call creates one fresh Model semantic occurrence. Evaluation is a
  composition bypass and must preserve exact input/state identity without manufacturing a Model
  no-op.<span data-glossary-navigation="true"> ([Evaluation glossary](../../../../glossary.md#term-evaluation); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Existing [Model](../../../../glossary.md#term-model) [state threading](../../../../glossary.md#state-threading) remains explicit. Passing the returned `nextState` to a later
  call expresses sequential consumption; reusing one incoming state expresses an intentional
  branch. The layer adds no coordination policy.
- Tensor IDs are opaque monotonic construction identifiers. Training may consume a prefix before
  identifier exhaustion; there is no rollback. Evaluation consumes none.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [opaque glossary](../../../../glossary.md#term-opaque); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- [Planning](../../../../glossary.md#term-planning) is non-authoritative. If implementation discovers a conflict with `ARCHITECTURE.md` or
  a final API cannot express this contract, stop before editing and report the exact uncertainty.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Public [API](../../../../glossary.md#term-application-programming-interface-api)

All declared public/protected surface added by this task is final and exact:

| Type | [Signature](../../../../glossary.md#term-signature) | Contract |
|---|---|---|
| `public final class Dropout extends Module` | `public Dropout(double probability)` | Validates and stores the exact [finite](../../../../glossary.md#term-finite) drop probability in `[0.0, 1.0)`; [signed zero](../../../../glossary.md#term-signed-zero) is retained; declares no [module](../../../../glossary.md#term-module) state. |
| [`Dropout`](../../../../glossary.md#dropout--inverted-dropout) | `public DropoutForwardResult forward(Tensor input, GraphRngState state, ForwardContext context)` | Validates non-null [arguments](../../../../glossary.md#term-argument) in [parameter](../../../../glossary.md#term-parameter) order, reads the context mode once, delegates exactly once to [Model](../../../../glossary.md#term-model) in training, and performs exact-reference bypass in [evaluation](../../../../glossary.md#term-evaluation). |
| `public record DropoutForwardResult(Tensor output, GraphRngState nextState)` | `public DropoutForwardResult` compact [canonical](../../../../glossary.md#term-canonical-form) constructor | Rejects null output then null next state, retains exact references, and uses ordinary record value equality. |
| `DropoutForwardResult` | generated `output()` | Returns the exact retained non-null output reference. |
| `DropoutForwardResult` | generated `nextState()` | Returns the exact retained non-null next-state reference. |

There is no declared public/protected probability accessor, mode accessor, state accessor,
training/evaluation convenience, [mask](../../../../glossary.md#term-boolean-mask) accessor, constructor overload, nested type, or interface.
Inherited final `Module` discovery/mode methods remain unchanged.

## Result-carrier decision and [ownership](../../../../glossary.md#term-ownership-meanings)

`io.github.pho001.synaptik.model.tensor.DropoutResult` cannot serve as the [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) forward result
without changing its current truth. Its public [Javadoc](../../../../glossary.md#term-javadoc) defines output and next state as output
slots zero and two of one training-[dropout](../../../../glossary.md#dropout--inverted-dropout) [producer](../../../../glossary.md#term-producer). [Evaluation](../../../../glossary.md#term-evaluation) intentionally has no producer and
must return the original input and state. Reusing that carrier would violate its documented
[invariant](../../../../glossary.md#term-invariant) even though its public constructor mechanically accepts the two references.

`DropoutForwardResult` therefore belongs in `io.github.pho001.synaptik.nn.layers`, beside the
mode-sensitive layer whose two branches it represents. It is a shallow [immutable](../../../../glossary.md#term-immutable) record with
only the two values every caller must thread. It does not own, copy, mutate, inspect, or execute
either value. It records neither branch, probability, [mask](../../../../glossary.md#term-boolean-mask), source layer, context, nor producer.

- Every successful `forward` call returns a fresh NN result record.
- Evaluation retains the exact caller `input` and exact caller `state` references.
- Training retains the exact `output()` and exact `nextState()` references returned by the one
  Model call; it does not return the Model carrier itself or preserve it as layer state.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- Independently constructed equal records use ordinary record equality. This does not change
  Tensor or <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>[`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate) [identity](../../../../glossary.md#term-identity-meanings) semantics.
- The record's [canonical](../../../../glossary.md#term-canonical-form) constructor checks `output` then `nextState`, with those parameter names
  as null messages, and causes no Tensor identifier or producer side effect.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [side effect glossary](../../../../glossary.md#term-side-effect))</span>

## Probability, [module](../../../../glossary.md#term-module) state, and mode contract

Construction performs exactly one intrinsic [validation](../../../../glossary.md#term-validation) by creating `new
DropoutAttrs(probability)` and stores `attrs.probability()` in one private final primitive field.
This reuses the [Model](../../../../glossary.md#term-model)-owned domain and error wording without retaining the attributes object.
Both positive and [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero) remain distinct raw `double` representations. Probability zero
does not turn training into [evaluation](../../../../glossary.md#term-evaluation): training still delegates and creates one state-consuming
[dropout](../../../../glossary.md#dropout--inverted-dropout) [occurrence](../../../../glossary.md#term-occurrence).

The layer declares no [parameter](../../../../glossary.md#term-parameter), [buffer](../../../../glossary.md#term-buffer), or child. Direct and recursive parameter/buffer
discovery are empty, and state-dictionary task 0010 will therefore have no Dropout entry under
the current state categories. [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate) is a per-call Model input/output value, not a module
buffer merely because it is threaded between calls.

Inherited local mode remains useful only for producing a context [snapshot](../../../../glossary.md#term-snapshot). The forward method
does not read the layer's current mode, mutate it, or require that the supplied context originated
from this layer. A captured training context remains training after `eval()`, and a captured
evaluation context remains evaluation after `train()`.

## [Validation](../../../../glossary.md#term-validation) and side-effect order

### Construction

1. Construct `DropoutAttrs` from the supplied probability.
2. If intrinsic [validation](../../../../glossary.md#term-validation) fails, propagate its exact `IllegalArgumentException`. No layer is
   returned and no Tensor, state, producer, storage, random source, or identifier is created.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [producer glossary](../../../../glossary.md#term-producer))</span>
3. Store the exact validated primitive probability. Retain no `DropoutAttrs` instance or other
   caller-owned object.

Accepted boundary cases include `0.0d`, `-0.0d`, and the greatest [finite](../../../../glossary.md#term-finite) `double` below `1.0d`.
Reject [NaN](../../../../glossary.md#term-nan), either [infinity](../../../../glossary.md#term-infinity), every numerical negative value, `1.0d`, and every greater value.

### Forward common prefix

Each call completes the following checks before reading mode or entering either branch:

1. Reject null `input` with `NullPointerException("input")`.
2. Reject null `state` with `NullPointerException("state")`.
3. Reject null `context` with `NullPointerException("context")`.
4. Read `context.mode()` exactly once. `ForwardContext` construction already guarantees a
   non-null enum value.

These local failures consume no [Tensor](../../../../glossary.md#tensor) identifier, create no result carrier or [producer](../../../../glossary.md#term-producer), and
change no [module](../../../../glossary.md#term-module)/context/state value. Null checking before branch selection deliberately gives
the layer one stable public [argument](../../../../glossary.md#term-argument) order independent of mode.

### [Evaluation](../../../../glossary.md#term-evaluation) branch

After the common prefix, evaluation constructs only `new DropoutForwardResult(input, state)` and
returns it. It performs no floating-type, [Shape](../../../../glossary.md#shape), [layout](../../../../glossary.md#layout), [gradient](../../../../glossary.md#term-gradient), storage, probability, or [Model](../../../../glossary.md#term-model)
[operation](../../../../glossary.md#operation) [validation](../../../../glossary.md#term-validation). The exact input object already carries all its [metadata](../../../../glossary.md#term-metadata). The result is
fresh, but `result.output() == input` and `result.nextState() == state`. There is no [Tensor](../../../../glossary.md#tensor) ID,
[producer](../../../../glossary.md#term-producer), [mask](../../../../glossary.md#term-boolean-mask), next-state expression, counter advancement, or hidden draw.

### Training branch

After the common prefix, training calls `input.dropout(probability, state)` exactly once. The
[Model](../../../../glossary.md#term-model) helper validates floating input then the already-valid stored probability then state before
allocation. With the layer's non-null and constructor invariants already established, a non-
floating input is the remaining ordinary local Model [validation](../../../../glossary.md#term-validation) failure. A successful Model call
creates exactly three output wrappers/IDs in output, hidden [mask](../../../../glossary.md#term-boolean-mask), next-state order under one
[producer](../../../../glossary.md#term-producer). The layer then reads the Model result's output and next state and constructs one fresh
[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) result from those exact references.

If Model allocation fails after consuming a prefix of output identifiers, identifiers are not
rolled back and no NN result is returned. The layer retains no partial result, state, or producer.
Record construction cannot fail for a valid Model result because both Model components are
non-null by contract.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.layers` — concrete layer [ownership](../../../../glossary.md#term-ownership-meanings) and the new [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) forward result.
- `io.github.pho001.synaptik.nn.module` — unchanged [`Module`](../../../../glossary.md#term-module) and `ForwardContext` contracts.
- `io.github.pho001.synaptik.model.tensor` — unchanged Tensor, explicit state, and training result
  contracts.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- `io.github.pho001.synaptik.model.operation.random` — existing intrinsic probability [validation](../../../../glossary.md#term-validation).

No package is added. The exact new type placement is:

- `io.github.pho001.synaptik.nn.layers.Dropout` — parameterless/bufferless mode-sensitive layer.
- `io.github.pho001.synaptik.nn.layers.DropoutForwardResult` — two-reference result for both NN
  branches.
- `io.github.pho001.synaptik.nn.layers.DropoutTest` — public surface, state/mode, result,
  provenance, validation, identity, and side-effect tests.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

## Affected files

Expected production files:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/Dropout.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/DropoutForwardResult.java`.

Expected test file:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/DropoutTest.java`.

Expected documentation and [planning](../../../../glossary.md#term-planning) files:

- `docs/glossary.md` — extend existing Dropout and NN module/forward-context entries with explicit
  NN branch behavior, result ownership, caller-threaded state, and no registered module state;
  do not duplicate Model mathematics.<span data-glossary-navigation="true"> ([Dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Model glossary](../../../../glossary.md#term-model))</span>
- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify exactly the seven paths listed above: three production Java files,
one [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) test file, and three documentation/[planning](../../../../glossary.md#term-planning) files. If implementation needs a [Model](../../../../glossary.md#term-model) edit,
second test [owner](../../../../glossary.md#term-owner), new [module](../../../../glossary.md#term-module) [API](../../../../glossary.md#term-application-programming-interface-api), API-guide edit, [dependency](../../../../glossary.md#term-dependency), [architecture test](../../../../glossary.md#term-architecture-test), [conformance](../../../../glossary.md#term-conformance) or
[integration test](../../../../glossary.md#term-integration-test), result field, generic interface, or eighth path, stop and propose a focused
follow-up instead of expanding this task.

## Acceptance criteria

- Public final `Dropout extends Module` declares exactly `Dropout(double)` and
  `forward(Tensor, GraphRngState, ForwardContext)` as public members, with no other declared
  public/protected member, overload, nested type, or public/protected field.
- Public final record `DropoutForwardResult` has exactly `Tensor output` and `GraphRngState
  nextState` in that order, no nested type or added component, and exact-reference retention with
  null validation in component order.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Construction accepts and bit-preserves both signed zeros and every other valid probability,
  rejects the complete invalid domain through `DropoutAttrs`, stores only one primitive
  probability, and causes no Tensor/state/identifier/random-source side effect.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [side effect glossary](../../../../glossary.md#term-side-effect))</span>
- Direct/recursive parameter and buffer collections and child discovery remain empty. No
  registration name or hidden graph RNG state appears in layer fields.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [buffer glossary](../../../../glossary.md#term-buffer); [graph RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- Evaluation is selected only by the exact supplied context snapshot, returns one fresh NN result
  with exact input/state references, accepts non-floating as well as floating inputs, and consumes
  no Tensor ID or producer occurrence.<span data-glossary-navigation="true"> ([Evaluation glossary](../../../../glossary.md#term-evaluation); [snapshot glossary](../../../../glossary.md#term-snapshot); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Training is selected only by the exact supplied context snapshot and delegates exactly once to
  Model dropout with the exact input, stored probability representation, and supplied state.
- Training tests prove one [Model](../../../../glossary.md#term-model) [`DROPOUT`](../../../../glossary.md#dropout--inverted-dropout) [producer](../../../../glossary.md#term-producer), ordered exact `[input, state]` provenance,
  output slots zero/one/two, hidden BOOL mask, exact <span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [BOOL glossary](../../../../glossary.md#data-type--datatype); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>`DropoutAttrs`, output metadata, next-state
  metadata, three-ID order, and exact output/next-state references in the fresh NN result without
  exposing the mask publicly.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- Probability zero in training still constructs a fresh Model occurrence and next-state
  expression; evaluation remains the only bypass.
- Repeated training with the same incoming state constructs distinct branches; explicitly
  threading the returned next state supplies it to the next producer. The layer mutates neither
  state nor module state.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- Captured contexts remain authoritative across later `train()`/`eval()` calls and may originate
  from another module. Forward changes neither the supplied context nor inherited mode.
- Null arguments fail in exact input/state/context order before mode selection. Integral/BOOL
  input fails only in training and succeeds by exact identity in evaluation. Local failures consume
  no identifiers; partial Model identifier exhaustion follows the documented no-rollback rule.<span data-glossary-navigation="true"> ([arguments glossary](../../../../glossary.md#term-argument); [identity glossary](../../../../glossary.md#term-identity-meanings); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Public/package Javadocs document purpose, probability semantics, explicit state ownership,
  context authority, both branch results, exact reference retention, result freshness,
  validation/failure order, Model delegation, state discovery, and no-execution/no-hidden-RNG
  boundaries with complete <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>`@param`, `@return`, and `@throws` tags.
- A separate clean-context documentation pass finalizes Javadocs, package docs, glossary,
  planning evidence, no-change conclusions, generated Javadoc, Markdown, exact scope/status, and
  whitespace before the task becomes Complete.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- No Model, Training, build, dependency, architecture, architecture-test, Tensor/Compile/Training
  API, compiler/runtime/prepare/Engine/backend, CPU, global-roadmap, task 0010/0011, or unrelated
  path enters the task diff.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Validation tier: task validation for the single affected `extensions/nn` [module](../../../../glossary.md#term-module) plus the targeted
documentation pass. This task changes neither [dependency direction](../../../../glossary.md#java-platform-and-build-tools) nor end-to-end/[backend](../../../../glossary.md#backend)
behavior, so repository-wide, architecture, backend-[conformance](../../../../glossary.md#term-conformance), and integration validation are
not required.

Implementation pass runs focused tests while developing:

```text
./gradlew :extensions:nn:test \
  --tests io.github.pho001.synaptik.nn.layers.DropoutTest
```

Then [run](../../../../glossary.md#term-run) the affected module once as the authoritative Java validation:

```text
./gradlew :extensions:nn:test
```

The clean documentation pass reuses a successful unchanged Java result and does not rerun it. It
runs:

```text
./gradlew :extensions:nn:javadoc
```

Final validation also checks:

- reflection/import surface for both public types and absence of extra members;
- no [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) import of Training, [Compiler](../../../../glossary.md#term-compiler), [Runtime](../../../../glossary.md#term-runtime), [Prepare](../../../../glossary.md#prepare), [Engine](../../../../glossary.md#term-engine), or backend packages;
- unchanged `extensions/nn` [dependency](../../../../glossary.md#term-dependency) set;
- local Markdown links, anchors, fences, terminal newlines, and trailing whitespace;
- exactly the seven task-owned paths and no task 0010/0011 specification;
- task 0009 remains In progress through implementation and documentation review, becomes Complete
  only after every final gate passes, and tasks 0010 and 0011 remain concise Draft rows;
- `git diff --check`, including a no-index whitespace check for every untracked new file.

Do not run [Model](../../../../glossary.md#term-model), Compiler, [CPU](../../../../glossary.md#term-cpu), root, architecture, conformance, or integration suites merely to
repeat stable prerequisite evidence. Escalate only if implementation changes a prerequisite or
discovers a concrete cross-module failure.

## Dependencies

- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) tasks 0001–0008 are Complete.
- Model tasks 0019B and 0019B1 are Complete and provide the final explicit state, probability,
  operation, Tensor convenience, and training-result contracts.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [Tensor glossary](../../../../glossary.md#tensor))</span>
- [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate), [Model](../../../../glossary.md#term-model) `DropoutResult`, `Tensor.dropout`, `DropoutAttrs`, `ForwardContext`, and
  [`Module`](../../../../glossary.md#term-module) are final enough to implement this task without changing them.
- The existing model-only NN Gradle dependency and architecture test already permit every needed
  import.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [architecture test glossary](../../../../glossary.md#term-architecture-test))</span>
- Numerical/backend support is not a prerequisite because this task constructs or bypasses
  symbolic expressions only.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>

## Follow-up tasks

- NN 0010: State dictionary and checkpoint contract. Dropout contributes no parameter or buffer
  path and must not cause graph RNG state to become checkpointed module state.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [Dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [parameter glossary](../../../../glossary.md#term-parameter); [buffer glossary](../../../../glossary.md#term-buffer); [graph RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0011: [Unary](../../../../glossary.md#term-unary) [Tensor](../../../../glossary.md#tensor) [module](../../../../glossary.md#term-module) composition and `Sequential`. It must account for Dropout's explicit
  state/result signature rather than forcing this layer into a false unary Tensor-only contract.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- Future training/session orchestration may own how a caller carries graph RNG state between
  steps, but it must consume this explicit contract and may not retrofit hidden layer state.
- Compiler/backend/runtime work may capture and realize existing Model dropout semantics in its
  own plans; it does not belong in this NN task.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [capture glossary](../../../../glossary.md#term-capture-meanings); [Model glossary](../../../../glossary.md#term-model))</span>

## Documentation and no-change review

Document types for this task:

- Java [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc): General plus API/Javadoc profiles.
- `layers/package-info.java`: API/Javadoc package profile.
- glossary entry: General reference style with first-use explanation and links.
- task/master [planning](../../../../glossary.md#term-planning) records: General plus Planning profiles.

Required documentation changes:

- Finalize complete Javadocs for [`Dropout`](../../../../glossary.md#dropout--inverted-dropout), `DropoutForwardResult`, constructor, forward method,
  and record components/canonical constructor.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Extend `layers/package-info.java` with the explicit-state, context-selected Dropout contract.
- Extend the existing glossary Dropout entry and NN module/forward-context entry without copying
  the full Model formula or claiming execution support.
- Synchronize the [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) master-plan row/status/decisions and this task's evidence/completion summary.

Reasoned no-change conclusions to record in the completion summary:

- `ARCHITECTURE.md`, focused architecture docs, and ADR 0007 remain accurate because ownership and
  dependency direction do not change.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>
- `docs/api/tensor-api.md` remains accurate because [Model](../../../../glossary.md#term-model) `Tensor.dropout`, [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate), and
  Model `DropoutResult` do not change.
- `docs/api/compile-api.md` remains accurate because this task adds no capture, compilation, or
  gradient request.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- `docs/api/training-api.md` and `docs/architecture/training-graph.md` remain accurate because no
  optimizer, session, gradient publication, or state orchestration is implemented.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [publication glossary](../../../../glossary.md#term-publication))</span>
- Model capabilities/master-plan/tasks remain accurate because the NN wrapper adds no Model
  semantic or backend capability.<span data-glossary-navigation="true"> ([backend capability glossary](../../../../glossary.md#backend-capability))</span>
- `ForwardContext`, [`Module`](../../../../glossary.md#term-module), [`Operation`](../../../../glossary.md#operation), [`Shape`](../../../../glossary.md#shape), [`DataType`](../../../../glossary.md#data-type--datatype), [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue), and related
  normalization/layer contracts remain accurate without edits.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization))</span>
- Build files and dependency architecture tests remain accurate because the NN model-only edge is
  unchanged.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [NN model glossary](../../../../glossary.md#model-module-buffer-and-scan-distinctions))</span>
- Backend conformance and integration tests remain unnecessary because no numerical execution or
  end-to-end path changes.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- The global roadmap and unrelated CPU planning/implementation remain untouched because this is
  the recorded isolated NN frontier.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>

## Architecture impact

No architecture rule changes. This task implements the existing [ownership](../../../../glossary.md#term-ownership-meanings) split: [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) owns
mode-selected layer composition, while [Model](../../../../glossary.md#term-model) owns explicit [graph](../../../../glossary.md#graph) [RNG](../../../../glossary.md#graph-rng-state--graphrngstate) and [dropout](../../../../glossary.md#dropout--inverted-dropout) semantics. No
[ADR](../../../../glossary.md#term-architecture-decision-record-adr), architecture document, [dependency](../../../../glossary.md#term-dependency) rule, or [architecture test](../../../../glossary.md#term-architecture-test) changes.

## Implementation prompt

Implement [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) task 0009 exactly from this specification in a clean implementation context. Read
the root `AGENTS.md`, `ARCHITECTURE.md`, focused architecture/[planning](../../../../glossary.md#term-planning) rules, NN master and tasks
0001–0009, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007, [Model](../../../../glossary.md#term-model) tasks 0019B–0019B1, final [dropout](../../../../glossary.md#dropout--inverted-dropout)/state/[Tensor](../../../../glossary.md#tensor) APIs and tests, current
[`Module`](../../../../glossary.md#term-module)/`ForwardContext` and concrete layer patterns, glossary, and [dependency](../../../../glossary.md#term-dependency) tests before
editing. Do not use legacy design except read-only behavioral context, do not use a GSD workflow,
and do not [commit](../../../../glossary.md#term-commit) or push.

Add only final public `Dropout`, public record `DropoutForwardResult`, their single focused test,
and the specified package/glossary/planning updates. Use `DropoutAttrs` once for constructor
[validation](../../../../glossary.md#term-validation) and retain only its exact primitive probability. Forward must validate input, state,
and context in that order, read the supplied context once, delegate exactly once to
`Tensor.dropout` in training, and return exact-[reference identity](../../../../glossary.md#term-reference-identity) bypass in [evaluation](../../../../glossary.md#term-evaluation). Never
retain/create/[seed](../../../../glossary.md#term-seed)/split/advance hidden [RNG state](../../../../glossary.md#graph-rng-state--graphrngstate), declare module state, expose a [mask](../../../../glossary.md#term-boolean-mask), invent
execution behavior, or broaden the [API](../../../../glossary.md#term-application-programming-interface-api).

[Run](../../../../glossary.md#term-run) the focused and one authoritative NN module test pass. Then hand the unchanged implementation
to a separate clean documentation-focused context to independently finalize Javadocs, package
docs, glossary, and planning evidence, run generated [Javadoc](../../../../glossary.md#term-javadoc) and document/[scope](../../../../glossary.md#term-scope)/whitespace gates,
and record no-change conclusions. Preserve every unrelated dirty [CPU](../../../../glossary.md#term-cpu)/global planning path exactly.
If any architectural uncertainty, eighth path, or need to alter a final prerequisite appears,
stop and report it rather than inventing a contract.

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

After Java/tests are stable, give the required clean documentation context:

- this task goal, exact seven-path limit, public [API](../../../../glossary.md#term-application-programming-interface-api) table, branch/[validation](../../../../glossary.md#term-validation)/result decisions;
- the final implementation diff and exact successful Java commands/counts;
- the relevant architecture, ADR, documentation rules/profiles, Model contracts, NN module/layer
  contracts, glossary entries, and planning files;<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Model glossary](../../../../glossary.md#term-model); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [module glossary](../../../../glossary.md#term-module); [planning glossary](../../../../glossary.md#term-planning))</span>
- the mandate to finalize Javadocs, package docs, glossary, master/task evidence and no-change
  conclusions without changing executable behavior;
- the instruction not to repeat successful Java tests unless it changes executable Java or finds
  a concrete reason;
- generated-Javadoc, Markdown, reflection/import/dependency, exact-scope/status, and whitespace
  gates; and<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [dependency glossary](../../../../glossary.md#term-dependency); [scope glossary](../../../../glossary.md#term-scope))</span>
- the completion-summary and `Status` format required by `AGENTS.md`.

## Local decisions

- Use the term **drop probability**, not ambiguous rate/keep probability. It is the fraction
  dropped and matches Model naming exactly.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- Use primitive `double` because that is the final Model attribute/Tensor API representation. Do
  not introduce <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>[`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue), a [Tensor](../../../../glossary.md#tensor) [scalar](../../../../glossary.md#term-scalar), or a probability value object.
- Order forward arguments as input, graph RNG state, then context. Input and state are the two
  caller-threaded semantic values; the immutable context is the final branch selector. Common
  null validation follows that same public parameter order.<span data-glossary-navigation="true"> ([arguments glossary](../../../../glossary.md#term-argument); [graph RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [immutable glossary](../../../../glossary.md#term-immutable); [validation glossary](../../../../glossary.md#term-validation); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- Reuse `DropoutAttrs` only as the intrinsic constructor validator; retain no attrs object and
  expose no getter because no current consumer requires configuration introspection.
- Name the [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) carrier `DropoutForwardResult`, not `DropoutResult`, to avoid collision with and false
  equivalence to the Model training-occurrence carrier.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Put the carrier beside the layer in `nn.layers`, not in `nn.module` or Model: it exists only
  because the concrete mode-sensitive layer has two result origins.
- Return one carrier shape for both modes so callers always receive the state they must thread.
  Returning only Tensor during evaluation would force a mode-dependent Java return type or hide
  state preservation.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Preserve the exact incoming state reference in evaluation. Creating an equivalent new state
  would allocate an occurrence and falsely imply advancement or replay-state reconstruction.
- Preserve the exact input reference in evaluation. A Model no-op would consume an ID and change
  expression identity without semantic need.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Accept non-floating input in evaluation because bypass performs no dropout. Training alone owns
  Model floating eligibility.<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- Keep probability zero as a training occurrence because Model explicitly assigns it one draw per
  logical element; only evaluation bypasses state advancement.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>
- Add no universal forward interface. This explicit-state result signature is evidence that task
  0011 must design composition from real consumers rather than prematurely forcing every module
  into <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [module glossary](../../../../glossary.md#term-module))</span>`Tensor -> Tensor`.

## Implementation notes

- Added final public [`Dropout`](../../../../glossary.md#dropout--inverted-dropout) with one [Model](../../../../glossary.md#term-model)-validated primitive probability and the exact
  `forward(Tensor, GraphRngState, ForwardContext)` signature. Forward performs common null checks
  in parameter order, treats the supplied context as authoritative, bypasses Model entirely in
  evaluation, and delegates once to <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [parameter glossary](../../../../glossary.md#term-parameter); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>`Tensor.dropout` in training.
- Added public record `DropoutForwardResult` with exact output/next-state components, ordered null
  validation, and exact-reference retention for both branches.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Extended the existing layers package draft and added one focused test owner covering the exact
  surface, validation, module-state absence, context behavior, identity, provenance, explicit
  state threading, and failure effects. No Model, dependency, execution, or unrelated source was
  changed.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [module glossary](../../../../glossary.md#term-module); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance); [state threading glossary](../../../../glossary.md#state-threading); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Known limitations

- The layer builds expressions or bypasses them; it does not execute [dropout](../../../../glossary.md#dropout--inverted-dropout).
- [Model](../../../../glossary.md#term-model) selects no portable random algorithm or cross-[backend](../../../../glossary.md#backend) bitstream.
- The public [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) result does not expose the keep [mask](../../../../glossary.md#term-boolean-mask) or [producer](../../../../glossary.md#term-producer).
- The layer does not manage one RNG stream across calls; callers must thread or branch state
  explicitly.<span data-glossary-navigation="true"> ([RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- [Evaluation](../../../../glossary.md#term-evaluation) and training accept different [Tensor](../../../../glossary.md#tensor) type domains by design.
- No [checkpoint](../../../../glossary.md#term-checkpoint)/session contract records caller-owned [graph RNG state](../../../../glossary.md#graph-rng-state--graphrngstate).
- Inherited module mode is mutable, and this task adds no thread-safety guarantee around mode
  changes, context capture, or concurrent forward construction.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [mutable glossary](../../../../glossary.md#term-mutable); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- Tensor identifier exhaustion may consume a training output-ID prefix without [rollback](../../../../glossary.md#term-rollback).

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) validation performed before implementation:

- [x] This detailed task is the only [Ready](../../../../glossary.md#term-ready) [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) task, and its master-plan row is linked and Ready.
- [x] NN tasks 0010–0011 remain concise Draft rows and have no detailed task files.
- [x] Local Markdown links, anchors, fences, terminal newlines, and trailing whitespace pass.
- [x] Planning [scope](../../../../glossary.md#term-scope) contains exactly this task specification and the NN master plan.
- [x] Whole-worktree `git diff --check` and the new-task no-index whitespace check pass while
      unrelated work is preserved.

No Java test or generated-[Javadoc](../../../../glossary.md#term-javadoc) command was [run](../../../../glossary.md#term-run) for this planning-only change.

Final implementation and documentation evidence:

- Clean implementation context `/root/nn_0009_implementation` added only the planned Dropout
  production/test surface and synchronized planning evidence. Independent clean documentation
  context `/root/nn_0009_docs` then reviewed the final implementation and tests against the
  architecture, task, Model prerequisites, and final NN APIs without changing executable Java or
  test behavior.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- Focused test command/result: final
  `./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.DropoutTest`
  passed with `BUILD SUCCESSFUL`; the XML report contains one suite and 9 tests with zero
  failures, errors, or skips.
- Authoritative `:extensions:nn:test` command/result: after executable Java and tests stabilized,
  `./gradlew :extensions:nn:test` passed with `BUILD SUCCESSFUL`; XML reports contain 15 suites
  and 83 tests with zero failures, errors, or skips. No executable Java or test changed afterward
  in this implementation context.
- Generated Javadoc command/result: after the documentation review,
  `./gradlew :extensions:nn:javadoc` passed with `BUILD SUCCESSFUL`; all three actionable tasks
  were up to date. Inspection of generated [`Dropout`](../../../../glossary.md#dropout--inverted-dropout), `DropoutForwardResult`, and layers package
  pages confirmed the complete probability, ownership, context authority, branch identity,
  freshness, failure, and non-execution contracts. The existing production and package Javadocs
  required no source revision because they were already complete and accurate.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [identity glossary](../../../../glossary.md#term-identity-meanings); [source revision glossary](../../../../glossary.md#term-source-revision))</span>
- Public surface/import/[dependency](../../../../glossary.md#term-dependency) checks: final `javap -public` showed `Dropout extends Module`
  with exactly its one constructor and one forward method, and final record
  `DropoutForwardResult` with exactly its canonical constructor, two accessors, and standard
  record members. A separate reflection program independently confirmed finality, superclass,
  one primitive field, exact declarations/components, and absence of nested or extra visible API.
  Production import scans found only Model, the existing NN module package, and JDK imports; the
  unchanged NN Gradle project retains its sole Model dependency and no forbidden Training,
  Compiler, Runtime, Prepare, Engine, or backend import.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [Gradle glossary](../../../../glossary.md#term-gradle); [Compiler glossary](../../../../glossary.md#term-compiler); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend))</span>
- Focused coverage proves complete probability validation and signed-zero retention; the sole
  primitive configuration field and absent module/random state; record null order, identity,
  freshness, and equality; context authority across later and foreign mode changes; exact
  evaluation input/state identity with no ID allocation for all six data types; exact training
  kind, attributes, ordered state/input provenance, three slots/IDs, hidden BOOL mask, descriptors,
  and wrapped output/next-state references; probability-zero producer creation; same-state
  branching and explicit next-state threading; local null/type failure order with no ID effects;
  and partial Model identifier exhaustion with no retained layer state. Tests inspect metadata and
  provenance only and make no eager or numerical-execution claim.<span data-glossary-navigation="true"> ([signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [evaluation glossary](../../../../glossary.md#term-evaluation); [provenance glossary](../../../../glossary.md#provenance); [BOOL glossary](../../../../glossary.md#data-type--datatype); [mask glossary](../../../../glossary.md#term-boolean-mask); [descriptors glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer); [state threading glossary](../../../../glossary.md#state-threading); [metadata glossary](../../../../glossary.md#term-metadata); [eager glossary](../../../../glossary.md#term-eager-evaluation))</span>
- Documentation validation passed for `docs/glossary.md`, the NN master plan, and this task: 335
  local links and 293 anchored destinations resolved, fences balanced, terminal newlines present,
  and no trailing whitespace was found. The glossary now distinguishes Model's same-producer
  training carrier from the NN branch carrier and gives a small declarative evaluation/training
  identity table without repeating Model mathematics or claiming execution.
- Exact final scope contains only the seven task-owned paths: `Dropout.java`,
  `DropoutForwardResult.java`, layers `package-info.java`, `DropoutTest.java`, `docs/glossary.md`,
  the NN master plan, and this task. No task 0010/0011 specification or unrelated Model, build,
  dependency, architecture, API, CPU, global-roadmap, compiler/runtime/prepare/Engine/backend,
  conformance, or integration path entered the task diff. NN tasks 0001–0009 are Complete; tasks
  0010–0011 remain Draft rows with no detailed files; no NN task is Ready.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Final `git diff --check` and explicit no-index checks for every untracked new path produced no
  whitespace diagnostics.
- No-change review confirmed that `ARCHITECTURE.md`, focused architecture documents, ADR 0007,
  architecture tests, Tensor/Compile/Training API guides, the training graph, Model
  capabilities/master/tasks, Model and related operation contracts, build files, conformance and
  integration tests, runtime/backends, CPU work, the global roadmap, and later NN tasks remain
  accurate without edits. This NN wrapper changes no ownership, dependency, Model semantic,
  gradient/optimizer, execution, persistence, or backend contract.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [training graph glossary](../../../../glossary.md#training-graph); [operation glossary](../../../../glossary.md#operation); [gradient glossary](../../../../glossary.md#term-gradient); [optimizer glossary](../../../../glossary.md#term-optimizer); [persistence glossary](../../../../glossary.md#term-persistence))</span>

## Completion summary

- Completed changes: The exact Dropout executable surface, NN result carrier, package Javadocs,
  focused coverage, glossary explanation, and synchronized planning evidence are complete without
  hidden state or out-of-scope behavior.<span data-glossary-navigation="true"> ([Dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [planning glossary](../../../../glossary.md#term-planning); [scope glossary](../../../../glossary.md#term-scope))</span>
- Files changed or created: Exactly the seven task-owned paths: three production Java paths, one
  focused NN test, the glossary, NN master plan, and this task record.
- Tests or validation performed: Focused one-suite/9-test and authoritative NN 15-suite/83-test
  runs passed with zero failures, errors, or skips and were not repeated after no executable
  change. Final NN Javadoc, generated-page inspection, <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`javap`, independent reflection,
  import/dependency, Markdown, exact-scope/status, newline, trailing-whitespace, and diff checks
  passed.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- Documentation review: Clean documentation context `/root/nn_0009_docs` independently finalized
  the glossary and planning evidence and confirmed that existing type/method/record/package
  Javadocs required no edit.
- Unresolved issues: None.
- Required follow-up: None for task 0009. [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0010 and 0011 remain Draft future capabilities.

Status: Complete
