# Task 0009: Dropout Layer

## Status

Complete

## Goal

Add one final public mode-sensitive `Dropout` module that composes the existing explicit-state
Model dropout operation without owning random state. Every forward call receives the input
`Tensor`, the caller's `GraphRngState`, and an immutable `ForwardContext`. Training creates one
Model inverted-dropout occurrence and returns its output and next state. Evaluation creates no
Model operation and returns the exact input and exact incoming state references.

The existing Model `DropoutResult` is deliberately training-occurrence-specific: its contract
says that output and next state select slots zero and two of one shared dropout producer. An
evaluation bypass has no such producer. This task therefore adds one narrow NN-owned
`DropoutForwardResult` that truthfully represents both branches without broadening or weakening
the Model contract.

Mental model:

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

This is symbolic expression composition and explicit caller-owned state threading. It is not an
eager draw, execution, hidden generator, mutable counter, parameter/buffer transition, optimizer
step, or serialization contract.

## Scope

- Add final public `io.github.pho001.synaptik.nn.layers.Dropout` extending `Module`.
- Add public final record
  `io.github.pho001.synaptik.nn.layers.DropoutForwardResult(Tensor output,
  GraphRngState nextState)` as the exact typed result of both forward branches.
- Add exactly the constructor and forward method in the public API table. Add no overload,
  default probability, probability/rate alias, accessor, builder, options object, functional
  convenience, or generic layer interface.
- Represent the fraction of positions to drop as one primitive `double probability`. Validate it
  through the existing public Model `DropoutAttrs` intrinsic contract: it must be finite and
  numerically in `[0.0, 1.0)`. Preserve either signed zero exactly and store the validated
  primitive without normalization or conversion.
- Make the layer parameterless and bufferless. It declares no `Parameter`, `Buffer`, or child and
  adds no entry to direct or recursive module-state discovery.
- Receive a non-null `GraphRngState` explicitly on every forward call and return a non-null next
  state explicitly. The layer must never construct, seed, split, clone, retain between calls,
  advance, replace, register, or consult a hidden random state.
- Receive a non-null immutable `ForwardContext` on every call. Its captured mode alone selects the
  branch, even when it came from another module or differs from this layer's current inherited
  mode.
- In training, call exactly `input.dropout(probability, state)` once. Return a fresh NN result
  retaining the exact `output()` and `nextState()` references of that one Model result. Do not
  expose or reconstruct the keep mask, output Tensors, state Tensor, producer, or attributes.
- In evaluation, construct no dropout or other Model occurrence. Return a fresh NN result whose
  output is the exact input Tensor reference and whose next state is the exact incoming
  `GraphRngState` reference.
- Let evaluation accept every non-null Tensor data type and Shape because bypass performs no
  dropout operation. Training retains the Model requirement that input data type be floating.
  Do not add a false common floating-input precondition to the identity branch.
- Preserve inherited `Module` mode behavior: a new layer starts in training mode, `train()` and
  `eval()` update inherited module mode, and `forwardContext()` snapshots it. The forward method
  never reads `mode()` implicitly; only the supplied context selects that call.
- Finalize the affected layer/package Javadocs and glossary terminology in the required separate
  clean documentation context before the implementation task becomes Complete.

## Out of scope

- Any change to `Tensor.dropout`, `DropoutAttrs`, `DropoutResult`, `GraphRngState`,
  `ForwardContext`, `ForwardMode`, or `Module`.
- Broadening Model `DropoutResult` to describe a no-producer evaluation bypass.
- A hidden or retained seed, generator, key, counter, state Tensor, `RandomGenerator`, thread-local
  source, process-global source, default state, or automatic state split.
- Calling `GraphRngState.initial`, inspecting its private Tensor, inventing public key/counter
  access, or adding an arbitrary-wrap/copy/advance API.
- A parameter, buffer, child module, mutable state binding, registration name, state-dictionary
  entry, checkpoint field, serialization token, or optimizer state.
- An eager random draw, host storage access, Tensor mutation, sampling algorithm, portable
  bitstream, numerical scaling/masking, compiler capture, lowering, preparation, backend route,
  runtime publication, or execution.
- A public keep mask, saved mask, producer, raw Model result, mode flag, probability field, or
  branch tag in `DropoutForwardResult`.
- A `forward(Tensor)` overload that reads implicit module mode, a no-state overload, a
  training-only API, a generic stochastic-module interface, or the unary composition contract
  reserved for task 0011.
- Thread-safety, atomicity, synchronization, rollback, identifier rollback, batching, session,
  gradient publication, optimizer, parameter-group, or training-step claims.
- Any new gradient formula, autograd rule, operation kind, Tensor semantics, backend support,
  conformance test, integration test, dependency, or architecture rule.
- State dictionary/checkpoint work from task 0010 or `Sequential`/shared unary composition from
  task 0011.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Planning guide](../../../planning-guide.md)
- [NN master plan](../master-plan.md)
- [Model master plan](../../../modules/model/master-plan.md)
- [Model task 0019B: Explicit graph RNG state foundation](../../../modules/model/tasks/0019b-explicit-graph-rng-state-foundation.md)
- [Model task 0019B1: Explicit graph dropout construction](../../../modules/model/tasks/0019b1-explicit-graph-dropout-construction.md)

## Architecture constraints

- `extensions/nn` owns train/eval layer composition and depends only on `modules/model`.
- Generic dropout meaning, attributes, explicit graph RNG state, producer topology, and the
  training-occurrence result stay in `modules/model`.
- The layer must not make Model depend on NN and must not depend on Training, Compiler, Runtime,
  Prepare, Engine, CPU, Metal, CUDA, or another backend.
- NN module state consists only of declared parameters, buffers, and children. This layer has
  none; graph RNG state remains a caller-threaded Model value rather than module state.
- `ForwardContext` is composition metadata, not runtime state. The supplied immutable snapshot is
  authoritative for a call and need not originate from the receiving layer.
- Every successful training call creates one fresh Model semantic occurrence. Evaluation is a
  composition bypass and must preserve exact input/state identity without manufacturing a Model
  no-op.
- Existing Model state threading remains explicit. Passing the returned `nextState` to a later
  call expresses sequential consumption; reusing one incoming state expresses an intentional
  branch. The layer adds no coordination policy.
- Tensor IDs are opaque monotonic construction identifiers. Training may consume a prefix before
  identifier exhaustion; there is no rollback. Evaluation consumes none.
- Planning is non-authoritative. If implementation discovers a conflict with `ARCHITECTURE.md` or
  a final API cannot express this contract, stop before editing and report the exact uncertainty.

## Public API

All declared public/protected surface added by this task is final and exact:

| Type | Signature | Contract |
|---|---|---|
| `public final class Dropout extends Module` | `public Dropout(double probability)` | Validates and stores the exact finite drop probability in `[0.0, 1.0)`; signed zero is retained; declares no module state. |
| `Dropout` | `public DropoutForwardResult forward(Tensor input, GraphRngState state, ForwardContext context)` | Validates non-null arguments in parameter order, reads the context mode once, delegates exactly once to Model in training, and performs exact-reference bypass in evaluation. |
| `public record DropoutForwardResult(Tensor output, GraphRngState nextState)` | `public DropoutForwardResult` compact canonical constructor | Rejects null output then null next state, retains exact references, and uses ordinary record value equality. |
| `DropoutForwardResult` | generated `output()` | Returns the exact retained non-null output reference. |
| `DropoutForwardResult` | generated `nextState()` | Returns the exact retained non-null next-state reference. |

There is no declared public/protected probability accessor, mode accessor, state accessor,
training/evaluation convenience, mask accessor, constructor overload, nested type, or interface.
Inherited final `Module` discovery/mode methods remain unchanged.

## Result-carrier decision and ownership

`io.github.pho001.synaptik.model.tensor.DropoutResult` cannot serve as the NN forward result
without changing its current truth. Its public Javadoc defines output and next state as output
slots zero and two of one training-dropout producer. Evaluation intentionally has no producer and
must return the original input and state. Reusing that carrier would violate its documented
invariant even though its public constructor mechanically accepts the two references.

`DropoutForwardResult` therefore belongs in `io.github.pho001.synaptik.nn.layers`, beside the
mode-sensitive layer whose two branches it represents. It is a shallow immutable record with
only the two values every caller must thread. It does not own, copy, mutate, inspect, or execute
either value. It records neither branch, probability, mask, source layer, context, nor producer.

- Every successful `forward` call returns a fresh NN result record.
- Evaluation retains the exact caller `input` and exact caller `state` references.
- Training retains the exact `output()` and exact `nextState()` references returned by the one
  Model call; it does not return the Model carrier itself or preserve it as layer state.
- Independently constructed equal records use ordinary record equality. This does not change
  Tensor or `GraphRngState` identity semantics.
- The record's canonical constructor checks `output` then `nextState`, with those parameter names
  as null messages, and causes no Tensor identifier or producer side effect.

## Probability, module state, and mode contract

Construction performs exactly one intrinsic validation by creating `new
DropoutAttrs(probability)` and stores `attrs.probability()` in one private final primitive field.
This reuses the Model-owned domain and error wording without retaining the attributes object.
Both positive and negative zero remain distinct raw `double` representations. Probability zero
does not turn training into evaluation: training still delegates and creates one state-consuming
dropout occurrence.

The layer declares no parameter, buffer, or child. Direct and recursive parameter/buffer
discovery are empty, and state-dictionary task 0010 will therefore have no Dropout entry under
the current state categories. `GraphRngState` is a per-call Model input/output value, not a module
buffer merely because it is threaded between calls.

Inherited local mode remains useful only for producing a context snapshot. The forward method
does not read the layer's current mode, mutate it, or require that the supplied context originated
from this layer. A captured training context remains training after `eval()`, and a captured
evaluation context remains evaluation after `train()`.

## Validation and side-effect order

### Construction

1. Construct `DropoutAttrs` from the supplied probability.
2. If intrinsic validation fails, propagate its exact `IllegalArgumentException`. No layer is
   returned and no Tensor, state, producer, storage, random source, or identifier is created.
3. Store the exact validated primitive probability. Retain no `DropoutAttrs` instance or other
   caller-owned object.

Accepted boundary cases include `0.0d`, `-0.0d`, and the greatest finite `double` below `1.0d`.
Reject NaN, either infinity, every numerical negative value, `1.0d`, and every greater value.

### Forward common prefix

Each call completes the following checks before reading mode or entering either branch:

1. Reject null `input` with `NullPointerException("input")`.
2. Reject null `state` with `NullPointerException("state")`.
3. Reject null `context` with `NullPointerException("context")`.
4. Read `context.mode()` exactly once. `ForwardContext` construction already guarantees a
   non-null enum value.

These local failures consume no Tensor identifier, create no result carrier or producer, and
change no module/context/state value. Null checking before branch selection deliberately gives
the layer one stable public argument order independent of mode.

### Evaluation branch

After the common prefix, evaluation constructs only `new DropoutForwardResult(input, state)` and
returns it. It performs no floating-type, Shape, layout, gradient, storage, probability, or Model
operation validation. The exact input object already carries all its metadata. The result is
fresh, but `result.output() == input` and `result.nextState() == state`. There is no Tensor ID,
producer, mask, next-state expression, counter advancement, or hidden draw.

### Training branch

After the common prefix, training calls `input.dropout(probability, state)` exactly once. The
Model helper validates floating input then the already-valid stored probability then state before
allocation. With the layer's non-null and constructor invariants already established, a non-
floating input is the remaining ordinary local Model validation failure. A successful Model call
creates exactly three output wrappers/IDs in output, hidden mask, next-state order under one
producer. The layer then reads the Model result's output and next state and constructs one fresh
NN result from those exact references.

If Model allocation fails after consuming a prefix of output identifiers, identifiers are not
rolled back and no NN result is returned. The layer retains no partial result, state, or producer.
Record construction cannot fail for a valid Model result because both Model components are
non-null by contract.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.layers` — concrete layer ownership and the new NN forward result.
- `io.github.pho001.synaptik.nn.module` — unchanged `Module` and `ForwardContext` contracts.
- `io.github.pho001.synaptik.model.tensor` — unchanged Tensor, explicit state, and training result
  contracts.
- `io.github.pho001.synaptik.model.operation.random` — existing intrinsic probability validation.

No package is added. The exact new type placement is:

- `io.github.pho001.synaptik.nn.layers.Dropout` — parameterless/bufferless mode-sensitive layer.
- `io.github.pho001.synaptik.nn.layers.DropoutForwardResult` — two-reference result for both NN
  branches.
- `io.github.pho001.synaptik.nn.layers.DropoutTest` — public surface, state/mode, result,
  provenance, validation, identity, and side-effect tests.

## Affected files

Expected production files:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/Dropout.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/DropoutForwardResult.java`.

Expected test file:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/DropoutTest.java`.

Expected documentation and planning files:

- `docs/glossary.md` — extend existing Dropout and NN module/forward-context entries with explicit
  NN branch behavior, result ownership, caller-threaded state, and no registered module state;
  do not duplicate Model mathematics.
- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

## Maximum scope

This task may create or modify exactly the seven paths listed above: three production Java files,
one NN test file, and three documentation/planning files. If implementation needs a Model edit,
second test owner, new module API, API-guide edit, dependency, architecture test, conformance or
integration test, result field, generic interface, or eighth path, stop and propose a focused
follow-up instead of expanding this task.

## Acceptance criteria

- Public final `Dropout extends Module` declares exactly `Dropout(double)` and
  `forward(Tensor, GraphRngState, ForwardContext)` as public members, with no other declared
  public/protected member, overload, nested type, or public/protected field.
- Public final record `DropoutForwardResult` has exactly `Tensor output` and `GraphRngState
  nextState` in that order, no nested type or added component, and exact-reference retention with
  null validation in component order.
- Construction accepts and bit-preserves both signed zeros and every other valid probability,
  rejects the complete invalid domain through `DropoutAttrs`, stores only one primitive
  probability, and causes no Tensor/state/identifier/random-source side effect.
- Direct/recursive parameter and buffer collections and child discovery remain empty. No
  registration name or hidden graph RNG state appears in layer fields.
- Evaluation is selected only by the exact supplied context snapshot, returns one fresh NN result
  with exact input/state references, accepts non-floating as well as floating inputs, and consumes
  no Tensor ID or producer occurrence.
- Training is selected only by the exact supplied context snapshot and delegates exactly once to
  Model dropout with the exact input, stored probability representation, and supplied state.
- Training tests prove one Model `DROPOUT` producer, ordered exact `[input, state]` provenance,
  output slots zero/one/two, hidden BOOL mask, exact `DropoutAttrs`, output metadata, next-state
  metadata, three-ID order, and exact output/next-state references in the fresh NN result without
  exposing the mask publicly.
- Probability zero in training still constructs a fresh Model occurrence and next-state
  expression; evaluation remains the only bypass.
- Repeated training with the same incoming state constructs distinct branches; explicitly
  threading the returned next state supplies it to the next producer. The layer mutates neither
  state nor module state.
- Captured contexts remain authoritative across later `train()`/`eval()` calls and may originate
  from another module. Forward changes neither the supplied context nor inherited mode.
- Null arguments fail in exact input/state/context order before mode selection. Integral/BOOL
  input fails only in training and succeeds by exact identity in evaluation. Local failures consume
  no identifiers; partial Model identifier exhaustion follows the documented no-rollback rule.
- Public/package Javadocs document purpose, probability semantics, explicit state ownership,
  context authority, both branch results, exact reference retention, result freshness,
  validation/failure order, Model delegation, state discovery, and no-execution/no-hidden-RNG
  boundaries with complete `@param`, `@return`, and `@throws` tags.
- A separate clean-context documentation pass finalizes Javadocs, package docs, glossary,
  planning evidence, no-change conclusions, generated Javadoc, Markdown, exact scope/status, and
  whitespace before the task becomes Complete.
- No Model, Training, build, dependency, architecture, architecture-test, Tensor/Compile/Training
  API, compiler/runtime/prepare/Engine/backend, CPU, global-roadmap, task 0010/0011, or unrelated
  path enters the task diff.

## Tests / validation

Validation tier: task validation for the single affected `extensions/nn` module plus the targeted
documentation pass. This task changes neither dependency direction nor end-to-end/backend
behavior, so repository-wide, architecture, backend-conformance, and integration validation are
not required.

Implementation pass runs focused tests while developing:

```text
./gradlew :extensions:nn:test \
  --tests io.github.pho001.synaptik.nn.layers.DropoutTest
```

Then run the affected module once as the authoritative Java validation:

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
- no NN import of Training, Compiler, Runtime, Prepare, Engine, or backend packages;
- unchanged `extensions/nn` dependency set;
- local Markdown links, anchors, fences, terminal newlines, and trailing whitespace;
- exactly the seven task-owned paths and no task 0010/0011 specification;
- task 0009 remains In progress through implementation and documentation review, becomes Complete
  only after every final gate passes, and tasks 0010 and 0011 remain concise Draft rows;
- `git diff --check`, including a no-index whitespace check for every untracked new file.

Do not run Model, Compiler, CPU, root, architecture, conformance, or integration suites merely to
repeat stable prerequisite evidence. Escalate only if implementation changes a prerequisite or
discovers a concrete cross-module failure.

## Dependencies

- NN tasks 0001–0008 are Complete.
- Model tasks 0019B and 0019B1 are Complete and provide the final explicit state, probability,
  operation, Tensor convenience, and training-result contracts.
- `GraphRngState`, Model `DropoutResult`, `Tensor.dropout`, `DropoutAttrs`, `ForwardContext`, and
  `Module` are final enough to implement this task without changing them.
- The existing model-only NN Gradle dependency and architecture test already permit every needed
  import.
- Numerical/backend support is not a prerequisite because this task constructs or bypasses
  symbolic expressions only.

## Follow-up tasks

- NN 0010: State dictionary and checkpoint contract. Dropout contributes no parameter or buffer
  path and must not cause graph RNG state to become checkpointed module state.
- NN 0011: Unary Tensor module composition and `Sequential`. It must account for Dropout's explicit
  state/result signature rather than forcing this layer into a false unary Tensor-only contract.
- Future training/session orchestration may own how a caller carries graph RNG state between
  steps, but it must consume this explicit contract and may not retrofit hidden layer state.
- Compiler/backend/runtime work may capture and realize existing Model dropout semantics in its
  own plans; it does not belong in this NN task.

## Documentation and no-change review

Document types for this task:

- Java API/Javadoc: General plus API/Javadoc profiles.
- `layers/package-info.java`: API/Javadoc package profile.
- glossary entry: General reference style with first-use explanation and links.
- task/master planning records: General plus Planning profiles.

Required documentation changes:

- Finalize complete Javadocs for `Dropout`, `DropoutForwardResult`, constructor, forward method,
  and record components/canonical constructor.
- Extend `layers/package-info.java` with the explicit-state, context-selected Dropout contract.
- Extend the existing glossary Dropout entry and NN module/forward-context entry without copying
  the full Model formula or claiming execution support.
- Synchronize the NN master-plan row/status/decisions and this task's evidence/completion summary.

Reasoned no-change conclusions to record in the completion summary:

- `ARCHITECTURE.md`, focused architecture docs, and ADR 0007 remain accurate because ownership and
  dependency direction do not change.
- `docs/api/tensor-api.md` remains accurate because Model `Tensor.dropout`, `GraphRngState`, and
  Model `DropoutResult` do not change.
- `docs/api/compile-api.md` remains accurate because this task adds no capture, compilation, or
  gradient request.
- `docs/api/training-api.md` and `docs/architecture/training-graph.md` remain accurate because no
  optimizer, session, gradient publication, or state orchestration is implemented.
- Model capabilities/master-plan/tasks remain accurate because the NN wrapper adds no Model
  semantic or backend capability.
- `ForwardContext`, `Module`, `Operation`, `Shape`, `DataType`, `ScalarValue`, and related
  normalization/layer contracts remain accurate without edits.
- Build files and dependency architecture tests remain accurate because the NN model-only edge is
  unchanged.
- Backend conformance and integration tests remain unnecessary because no numerical execution or
  end-to-end path changes.
- The global roadmap and unrelated CPU planning/implementation remain untouched because this is
  the recorded isolated NN frontier.

## Architecture impact

No architecture rule changes. This task implements the existing ownership split: NN owns
mode-selected layer composition, while Model owns explicit graph RNG and dropout semantics. No
ADR, architecture document, dependency rule, or architecture test changes.

## Implementation prompt

Implement NN task 0009 exactly from this specification in a clean implementation context. Read
the root `AGENTS.md`, `ARCHITECTURE.md`, focused architecture/planning rules, NN master and tasks
0001–0009, ADR 0007, Model tasks 0019B–0019B1, final dropout/state/Tensor APIs and tests, current
`Module`/`ForwardContext` and concrete layer patterns, glossary, and dependency tests before
editing. Do not use legacy design except read-only behavioral context, do not use a GSD workflow,
and do not commit or push.

Add only final public `Dropout`, public record `DropoutForwardResult`, their single focused test,
and the specified package/glossary/planning updates. Use `DropoutAttrs` once for constructor
validation and retain only its exact primitive probability. Forward must validate input, state,
and context in that order, read the supplied context once, delegate exactly once to
`Tensor.dropout` in training, and return exact-reference identity bypass in evaluation. Never
retain/create/seed/split/advance hidden RNG state, declare module state, expose a mask, invent
execution behavior, or broaden the API.

Run the focused and one authoritative NN module test pass. Then hand the unchanged implementation
to a separate clean documentation-focused context to independently finalize Javadocs, package
docs, glossary, and planning evidence, run generated Javadoc and document/scope/whitespace gates,
and record no-change conclusions. Preserve every unrelated dirty CPU/global planning path exactly.
If any architectural uncertainty, eighth path, or need to alter a final prerequisite appears,
stop and report it rather than inventing a contract.

## Documentation-agent handoff

After Java/tests are stable, give the required clean documentation context:

- this task goal, exact seven-path limit, public API table, branch/validation/result decisions;
- the final implementation diff and exact successful Java commands/counts;
- the relevant architecture, ADR, documentation rules/profiles, Model contracts, NN module/layer
  contracts, glossary entries, and planning files;
- the mandate to finalize Javadocs, package docs, glossary, master/task evidence and no-change
  conclusions without changing executable behavior;
- the instruction not to repeat successful Java tests unless it changes executable Java or finds
  a concrete reason;
- generated-Javadoc, Markdown, reflection/import/dependency, exact-scope/status, and whitespace
  gates; and
- the completion-summary and `Status` format required by `AGENTS.md`.

## Local decisions

- Use the term **drop probability**, not ambiguous rate/keep probability. It is the fraction
  dropped and matches Model naming exactly.
- Use primitive `double` because that is the final Model attribute/Tensor API representation. Do
  not introduce `ScalarValue`, a Tensor scalar, or a probability value object.
- Order forward arguments as input, graph RNG state, then context. Input and state are the two
  caller-threaded semantic values; the immutable context is the final branch selector. Common
  null validation follows that same public parameter order.
- Reuse `DropoutAttrs` only as the intrinsic constructor validator; retain no attrs object and
  expose no getter because no current consumer requires configuration introspection.
- Name the NN carrier `DropoutForwardResult`, not `DropoutResult`, to avoid collision with and false
  equivalence to the Model training-occurrence carrier.
- Put the carrier beside the layer in `nn.layers`, not in `nn.module` or Model: it exists only
  because the concrete mode-sensitive layer has two result origins.
- Return one carrier shape for both modes so callers always receive the state they must thread.
  Returning only Tensor during evaluation would force a mode-dependent Java return type or hide
  state preservation.
- Preserve the exact incoming state reference in evaluation. Creating an equivalent new state
  would allocate an occurrence and falsely imply advancement or replay-state reconstruction.
- Preserve the exact input reference in evaluation. A Model no-op would consume an ID and change
  expression identity without semantic need.
- Accept non-floating input in evaluation because bypass performs no dropout. Training alone owns
  Model floating eligibility.
- Keep probability zero as a training occurrence because Model explicitly assigns it one draw per
  logical element; only evaluation bypasses state advancement.
- Add no universal forward interface. This explicit-state result signature is evidence that task
  0011 must design composition from real consumers rather than prematurely forcing every module
  into `Tensor -> Tensor`.

## Implementation notes

- Added final public `Dropout` with one Model-validated primitive probability and the exact
  `forward(Tensor, GraphRngState, ForwardContext)` signature. Forward performs common null checks
  in parameter order, treats the supplied context as authoritative, bypasses Model entirely in
  evaluation, and delegates once to `Tensor.dropout` in training.
- Added public record `DropoutForwardResult` with exact output/next-state components, ordered null
  validation, and exact-reference retention for both branches.
- Extended the existing layers package draft and added one focused test owner covering the exact
  surface, validation, module-state absence, context behavior, identity, provenance, explicit
  state threading, and failure effects. No Model, dependency, execution, or unrelated source was
  changed.

## Known limitations

- The layer builds expressions or bypasses them; it does not execute dropout.
- Model selects no portable random algorithm or cross-backend bitstream.
- The public NN result does not expose the keep mask or producer.
- The layer does not manage one RNG stream across calls; callers must thread or branch state
  explicitly.
- Evaluation and training accept different Tensor type domains by design.
- No checkpoint/session contract records caller-owned graph RNG state.
- Inherited module mode is mutable, and this task adds no thread-safety guarantee around mode
  changes, context capture, or concurrent forward construction.
- Tensor identifier exhaustion may consume a training output-ID prefix without rollback.

## Validation evidence

Planning validation performed before implementation:

- [x] This detailed task is the only Ready NN task, and its master-plan row is linked and Ready.
- [x] NN tasks 0010–0011 remain concise Draft rows and have no detailed task files.
- [x] Local Markdown links, anchors, fences, terminal newlines, and trailing whitespace pass.
- [x] Planning scope contains exactly this task specification and the NN master plan.
- [x] Whole-worktree `git diff --check` and the new-task no-index whitespace check pass while
      unrelated work is preserved.

No Java test or generated-Javadoc command was run for this planning-only change.

Final implementation and documentation evidence:

- Clean implementation context `/root/nn_0009_implementation` added only the planned Dropout
  production/test surface and synchronized planning evidence. Independent clean documentation
  context `/root/nn_0009_docs` then reviewed the final implementation and tests against the
  architecture, task, Model prerequisites, and final NN APIs without changing executable Java or
  test behavior.
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
  were up to date. Inspection of generated `Dropout`, `DropoutForwardResult`, and layers package
  pages confirmed the complete probability, ownership, context authority, branch identity,
  freshness, failure, and non-execution contracts. The existing production and package Javadocs
  required no source revision because they were already complete and accurate.
- Public surface/import/dependency checks: final `javap -public` showed `Dropout extends Module`
  with exactly its one constructor and one forward method, and final record
  `DropoutForwardResult` with exactly its canonical constructor, two accessors, and standard
  record members. A separate reflection program independently confirmed finality, superclass,
  one primitive field, exact declarations/components, and absence of nested or extra visible API.
  Production import scans found only Model, the existing NN module package, and JDK imports; the
  unchanged NN Gradle project retains its sole Model dependency and no forbidden Training,
  Compiler, Runtime, Prepare, Engine, or backend import.
- Focused coverage proves complete probability validation and signed-zero retention; the sole
  primitive configuration field and absent module/random state; record null order, identity,
  freshness, and equality; context authority across later and foreign mode changes; exact
  evaluation input/state identity with no ID allocation for all six data types; exact training
  kind, attributes, ordered state/input provenance, three slots/IDs, hidden BOOL mask, descriptors,
  and wrapped output/next-state references; probability-zero producer creation; same-state
  branching and explicit next-state threading; local null/type failure order with no ID effects;
  and partial Model identifier exhaustion with no retained layer state. Tests inspect metadata and
  provenance only and make no eager or numerical-execution claim.
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
  0010–0011 remain Draft rows with no detailed files; no NN task is Ready.
- Final `git diff --check` and explicit no-index checks for every untracked new path produced no
  whitespace diagnostics.
- No-change review confirmed that `ARCHITECTURE.md`, focused architecture documents, ADR 0007,
  architecture tests, Tensor/Compile/Training API guides, the training graph, Model
  capabilities/master/tasks, Model and related operation contracts, build files, conformance and
  integration tests, runtime/backends, CPU work, the global roadmap, and later NN tasks remain
  accurate without edits. This NN wrapper changes no ownership, dependency, Model semantic,
  gradient/optimizer, execution, persistence, or backend contract.

## Completion summary

- Completed changes: The exact Dropout executable surface, NN result carrier, package Javadocs,
  focused coverage, glossary explanation, and synchronized planning evidence are complete without
  hidden state or out-of-scope behavior.
- Files changed or created: Exactly the seven task-owned paths: three production Java paths, one
  focused NN test, the glossary, NN master plan, and this task record.
- Tests or validation performed: Focused one-suite/9-test and authoritative NN 15-suite/83-test
  runs passed with zero failures, errors, or skips and were not repeated after no executable
  change. Final NN Javadoc, generated-page inspection, `javap`, independent reflection,
  import/dependency, Markdown, exact-scope/status, newline, trailing-whitespace, and diff checks
  passed.
- Documentation review: Clean documentation context `/root/nn_0009_docs` independently finalized
  the glossary and planning evidence and confirmed that existing type/method/record/package
  Javadocs required no edit.
- Unresolved issues: None.
- Required follow-up: None for task 0009. NN 0010 and 0011 remain Draft future capabilities.

Status: Complete
