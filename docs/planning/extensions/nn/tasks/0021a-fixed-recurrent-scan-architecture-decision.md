# Task 0021A: Fixed Recurrent-Scan Architecture Decision

## Status

Complete

## Goal

Make the explicit architecture decision required before Synaptik publishes or implements a
runtime-valid-length recurrent API.

The decision must authorize one fixed first-class recurrent-scan operation family as an ordinary
flat multi-output Model operation occurrence. It must explicitly decline a general user-defined
control-flow region, callback body, captured subgraph, or loop intermediate representation at this
frontier.

The resulting architecture contract and ADR must make the later implementation sequence
executable without leaving ownership or semantics to implementation-time invention:

```text
one ordinary flat Tensor producer / compiled node
  kind: RNN_TANH, GRU_RESET_AFTER, or LSTM
  attrs: FORWARD or REVERSE
  inputs: dense time-major values, runtime valid lengths, explicit initial state,
          and explicit parameter tensors
  outputs: dense original-time-aligned hidden outputs and explicit final state(s)

Compiler and Planning see one ordinary operation occurrence.
Backend Prepare lowers that occurrence once to a bounded recurrent loop plan.
Runtime invokes only the prepared executable and never interprets recurrence.
```

This task is architecture and documentation only. It creates no Java API, graph node, compiler
rule, prepared recipe, runtime step, Engine binding, backend capability, kernel, numerical result,
NN overload, Data adapter, build edge, or executable claim.

## Motivation

The current architecture authorizes Model-owned operation semantics and flat immutable graphs,
Compiler-owned capture/inference/autograd, backend-owned lowering, and Runtime execution of
prepared schedules. It does not authorize a nested graph region, loop body, body capture, region
identity, region ownership, or Runtime graph interpretation.

Publishing a Model body/region type now would therefore invent architecture and leave every
consumer undefined. Publishing only a new operation kind would also be incomplete: Compiler's
closed first-order inventory would reject the new family, captured inference would not adopt it,
and no backend could execute it. Combining all of Model, Compiler, Engine, Runtime, and CPU
implementation in one task would violate the repository's cross-layer task boundary.

The smallest coherent prerequisite is consequently one focused architecture decision that fixes
the semantic and lifecycle contract, after which owner-specific tasks can proceed in order.

## Selected architecture decision

### Fixed operation, not general control flow

- The first runtime recurrent scan is one fixed first-class Model operation family.
- Its mathematical transition is part of the operation semantics. It is not supplied through a
  Java callback, lambda, `Tensor` body, nested `CompiledGraphModel`, region, block, callable graph,
  or other user-defined body value.
- One public call creates exactly one identity-distinct multi-output `TensorProducer`. Compiler
  capture emits exactly one ordinary flat `CompiledNode` for that producer.
- The operation has no captured Tensor references beyond its ordered ordinary inputs. It owns no
  nested graph-local IDs, body inputs, body outputs, free-variable capture, region identity, or
  cross-graph ownership rule.
- `Tensor`, `TensorProducer`, `Operation`, `CompiledNode`, and `CompiledGraphModel` retain their
  current flat meanings. A general control-flow or region system remains forbidden until another
  explicit architecture update.

This decision distinguishes two concepts:

```text
declarative Model operation
  = immutable functional meaning and ordered Tensor inputs/outputs

runtime control flow
  = backend-internal prepared loop implementing that meaning
```

Runtime never receives `Operation`, `CompiledNode`, a body graph, or a loop condition. It invokes
one already prepared backend-owned bound action through the existing Runtime contracts.

### Exact semantic family

The future Model task owns one family with exactly these semantic variants:

```text
RNN_TANH
GRU_RESET_AFTER
LSTM
```

All three accept one immutable direction attribute:

```text
FORWARD
REVERSE
```

No bidirectional, stacked, arbitrary-cell, configurable-gate, configurable-activation,
peephole, projection, recurrent-dropout, residual, stateful, sparse, or quantized variant belongs
to the first family.

The fixed cell equations and parameter packing exactly match the current NN cells:

- `RNN_TANH` uses one tanh state transition, separate input and hidden weights, and one optional
  shared input-side bias;
- `GRU_RESET_AFTER` uses reset, update, candidate gate order, applies reset after the recurrent
  candidate projection, uses `candidate + update * (hidden - candidate)`, and has one optional
  packed input-side bias; and
- `LSTM` uses input, forget, candidate, output gate order, explicit hidden and cell carried
  states, the current fixed sigmoid/tanh equations, and one optional packed input-side bias.

### Exact future public Model surface

The later Model task must publish exactly one direction enum, one two-output result, one
three-output LSTM result, and these six receiver methods in the existing Tensor package:

```java
RecurrentScanResult rnnScan(
        Tensor validLengths,
        Tensor initialHidden,
        Tensor inputWeight,
        Tensor hiddenWeight,
        RecurrentDirection direction)

RecurrentScanResult rnnScan(
        Tensor validLengths,
        Tensor initialHidden,
        Tensor inputWeight,
        Tensor hiddenWeight,
        Tensor bias,
        RecurrentDirection direction)

RecurrentScanResult gruScan(
        Tensor validLengths,
        Tensor initialHidden,
        Tensor inputWeight,
        Tensor hiddenWeight,
        RecurrentDirection direction)

RecurrentScanResult gruScan(
        Tensor validLengths,
        Tensor initialHidden,
        Tensor inputWeight,
        Tensor hiddenWeight,
        Tensor bias,
        RecurrentDirection direction)

LstmRecurrentScanResult lstmScan(
        Tensor validLengths,
        Tensor initialHidden,
        Tensor initialCell,
        Tensor inputWeight,
        Tensor hiddenWeight,
        RecurrentDirection direction)

LstmRecurrentScanResult lstmScan(
        Tensor validLengths,
        Tensor initialHidden,
        Tensor initialCell,
        Tensor inputWeight,
        Tensor hiddenWeight,
        Tensor bias,
        RecurrentDirection direction)
```

The receiver is the time-major input Tensor. `RecurrentScanResult` exposes exactly
`outputs, finalHidden`; `LstmRecurrentScanResult` exposes exactly
`outputs, finalHidden, finalCell`. The result components are canonical output wrappers from one
exact shared producer in that order. This is a planned surface, not current runnable API.

The exact operation input order is:

```text
RNN/GRU without bias:
  [input, validLengths, initialHidden, inputWeight, hiddenWeight]

RNN/GRU with bias:
  [input, validLengths, initialHidden, inputWeight, hiddenWeight, bias]

LSTM without bias:
  [input, validLengths, initialHidden, initialCell, inputWeight, hiddenWeight]

LSTM with bias:
  [input, validLengths, initialHidden, initialCell, inputWeight, hiddenWeight, bias]
```

The exact operation output order is:

```text
RNN/GRU: [outputs, finalHidden]
LSTM:    [outputs, finalHidden, finalCell]
```

### Shape and type boundary

The first executable capability is deliberately static-Shape and runtime-value-dynamic:

- input is fully static rank three `[time, batch, inputSize]`;
- valid lengths are a fully static rank-one `INT64` Tensor `[batch]` with
  `requiresGrad == false`;
- initial hidden is `[batch, hiddenSize]`;
- initial cell, for LSTM, is `[batch, hiddenSize]`;
- input weight is `[gateCount * hiddenSize, inputSize]`;
- hidden weight is `[gateCount * hiddenSize, hiddenSize]`;
- optional bias is `[gateCount * hiddenSize]`;
- `gateCount` is one for RNN, three for GRU, and four for LSTM;
- `inputSize` and `hiddenSize` are positive; `time` and `batch` may be zero; and
- input, states, weights, and optional bias use one exact common floating data type.

The dense output Shape is `[time, batch, hiddenSize]`. Final-state Shapes are
`[batch, hiddenSize]`. Layout is unresolved at Model construction. Output gradient eligibility is
the OR of the differentiable floating input/state/parameter roles; valid lengths never contribute.

Dynamic or binding-dependent `time`, `batch`, `inputSize`, `hiddenSize`, parameter Shapes, or
output Shapes are outside this first program. A later dynamic-Shape lifecycle must not be hidden
inside runtime valid lengths. Prepared reuse is required across different valid-length values for
the same compatible static descriptors, not across arbitrary Shapes.

### Valid-length and traversal semantics

For each original batch row `b`, execution validates one runtime length `L[b]` in
`[0, time]`. Values are never inferred from input contents, padding values, zero, NaN, token IDs,
labels, or storage.

For `FORWARD`, row `b` applies the fixed transition at original time coordinates
`0 .. L[b]-1`. For `REVERSE`, it applies the same transition at coordinates
`L[b]-1 .. 0`. In both directions, the dense output is aligned to original time coordinates:

- every valid coordinate stores the next hidden state produced after consuming that coordinate;
- every padded coordinate `t >= L[b]` stores the exact positive zero of the common data type;
- a zero-length row has positive-zero output at every time and returns its exact initial hidden
  value semantically, plus its exact initial cell value for LSTM; and
- a zero-time request therefore requires every valid length to be zero and returns empty dense
  outputs plus the initial final-state values semantically.

The operation never treats padding as a recurrent input. Reverse means reverse traversal of only
the valid prefix, never whole-axis reversal through the padded suffix.

Before mutating any output representation, an executable backend must validate the complete
length vector, including every bound and any representation-specific access preconditions.
Invalid values fail the run without partially written published results. The architecture task
does not select the public exception translation owned by the future Engine facade.

### Purity, ownership, and side effects

- The operation is functionally pure and has no hidden module, compiler, prepared, or runtime
  state.
- All carried states are explicit operation inputs and final outputs.
- It owns no RNG, mode, counter, Buffer, Parameter, mutation, callback, I/O, or external resource.
- Different executions of one prepared recipe use isolated `RunState` instances under the current
  Runtime contract.
- NN continues to own parameter wrappers and state paths; the operation consumes their current
  Tensor bindings as ordinary inputs.
- NN state dictionaries contain parameter and persistent-buffer Tensor bindings. Future model
  checkpoint artifacts may persist the materialized values of those entries. The scan operation,
  compiler graph, prepared executable, runtime state, and backend artifacts are rebuilt and are
  not serialized by this decision.

### Compiler, Planning, Prepare, Runtime, Engine, and backend boundary

- Model owns kind/attributes, exact semantics, validation visible from descriptors, result
  metadata, canonical multi-output provenance, and the public Tensor surface.
- Compiler captures the producer as one ordinary flat node, owns descriptor inference and final
  validation, updates the exact operation inventory, and initially rejects every backward-capable
  request reaching the family before constructing any gradient Tensor.
- Backpropagation through time is a later explicit Compiler task. This task chooses no saved-gate
  outputs, tape, checkpointing policy, recomputation policy, backward operation, or derivative
  formula.
- Planning uses the existing ordinary operation capability query and selects backend ownership.
  It does not interpret recurrence, direction, valid lengths, active rows, loop parameters, or a
  route.
- Prepare uses the existing static-Shape partition projection, staged backend analysis,
  shared-resource declaration, slot assignment, and finalization lifecycle. The fixed operation
  introduces no shared loop-body or control-flow contract.
- Runtime uses the existing caller-input representation, prepared executable, cold binding,
  schedule, isolated run state, and bound invocation contracts. It does not select loop count,
  inspect valid lengths, compact rows, or interpret the operation.
- Engine owns the public mapping from typed logical caller inputs to ordered Runtime caller-input
  representations and from publication positions to typed outputs. Engine must not specialize or
  rebuild the graph from valid-length values.
- A concrete backend advertises support only for exact implemented variant/type/Shape/direction
  combinations. Backend analysis lowers the occurrence once, declares all resources before slot
  assignment, and finalization constructs the reusable executable.

The existing Planning, Prepare, and Runtime contracts are sufficient for the selected static-
Shape ordinary-node design. No task in those modules is required merely to name recurrence.
Implementation must stop and produce evidence if a later backend proves an actual shared-contract
gap instead of adding speculative control-flow types now.

### Performance contract

- One Model scan occurrence remains one compiled graph node regardless of `time` and valid-length
  values. Graph construction and compiler graph size are `O(1)` in `time` for that occurrence.
- Backend analysis compiles or otherwise prepares the transition implementation once per selected
  occurrence/plan. It must not construct, compile, or retain one graph/node/executable body per
  time step.
- The runtime hot path performs no reflection, string dispatch, boxing per scalar element, graph
  inspection, operation dispatch, backend lookup, route selection, or per-step object-graph
  growth.
- The first capability requires genuine skipped transition arithmetic for invalid coordinates.
  It does not require physical active-row compaction, sorting, a packed buffer, or a promise that
  zero-initialization and length checks perform no work.
- A backend may use a bounded branch-based row/time traversal and still satisfy the first
  capability if invalid coordinates execute no recurrent dot products, gates, or state update.
  Documentation must say exactly that; it must not claim compact memory traffic or skipped all
  physical work without evidence.
- Physical active-row compaction, vectorized packed batches, workspace reuse, or other optimized
  execution is a later backend route decision with truthful resource and performance evidence.

### NN compatibility and migration

- Current `RnnSequence`, `GruSequence`, `LstmSequence`, and the three bidirectional containers keep
  their current snapshotted Java `long[]`, compact-output-list, static-unroll, exact provenance,
  and final-state contracts unchanged throughout 0021A and 0021B.
- Model and Compiler work does not silently redirect those APIs to the new operation.
- NN 0022 later decides whether to add distinct runtime-length overloads, migrate current methods,
  retain them as compatibility APIs, or deprecate them. That decision must acknowledge the result
  shape difference: the new scan returns dense zero-padded original-time-aligned outputs, whereas
  current static containers expose a list of compact per-step outputs.
- Bidirectional migration must preserve independent parameter ownership, valid-prefix-only
  reverse traversal, forward-first final-axis concatenation, original-time alignment, and
  type-specific final hidden/cell states.
- A host `long[]` adapter that constructs a different graph for each length pattern is not the
  target runtime API and cannot be presented as implementation of this decision.

## Scope

- Update the authoritative architecture contract with the fixed-operation/no-region decision,
  exact owner boundaries, static-Shape/runtime-value distinction, Runtime hot-path rule,
  explicit-state/purity rule, initial BPTT boundary, and performance/migration invariants.
- Add ADR 0012 recording alternatives, decision, consequences, rejected general-region and dense-
  masking approaches, and follow-up owner order.
- Synchronize the focused lifecycle, module-boundary, training-graph, and Runtime/Prepare/backend
  explanations with the same current-versus-planned distinction.
- Link the ADR from the current architecture index and design-decision index.
- Keep the NN, Model, and Compiler master plans synchronized with one Ready architecture task and
  Draft owner-specific follow-ups only.
- Record reasoned no-change conclusions for dependency rules, architecture tests, Planning,
  Prepare, Runtime, Engine code, backend code, configuration, trace, Data/Text plans, public APIs,
  glossary, build files, and the global roadmap.

## Out of scope

- Any production or test Java, Javadoc, package documentation, generated documentation, build
  script, settings file, module dependency, or public API.
- A Model operation kind, attributes type, direction enum, Tensor method, result carrier,
  descriptor helper, producer, graph node, or source-backed capability update.
- Compiler capture, inference, validation, optimization, first-order inventory, gradient rule,
  BPTT, publication, or compile-artifact behavior.
- Planning capability logic, scoring, ownership rules, partitioning, or logical memory changes.
- Prepare projection, analysis, declaration, assignment, finalization, or schedule assembly.
- Runtime representation, input, slot, executable, bound invocation, schedule, runner,
  publication, result, or state behavior.
- Engine facade or typed caller-input/output binding.
- CPU or another backend capability, lowering, IR, generation, reference kernel, execution,
  conformance, or performance evidence.
- NN or Data runtime-valid-length API, migration, adapter, module behavior, state path, or static
  container change.
- Dynamic Shapes, general loops, conditionals, arbitrary masks with holes, user-defined bodies,
  nested regions, callable graphs, or serialization formats.
- Editing the concurrent CPU master/task/source/test/guide work or the global roadmap.
- Creating a detailed specification for Model 0025E, Compiler 0006A, NN 0021B, Engine, backend,
  NN 0022, or any later task.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Training graph](../../../../architecture/training-graph.md)
- [ADR 0002: Backend-owned lowering](../../../../design/decisions/0002-backend-owned-lowering.md)
- [ADR 0007: NN and Training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)
- [ADR 0009: Compiler-owned pre-capture autograd](../../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011: Per-run Runtime ownership](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [Planning guide](../../../planning-guide.md)
- [NN master plan](../master-plan.md)
- [Model master plan](../../../modules/model/master-plan.md)
- [Compiler master plan](../../../modules/compiler/master-plan.md)
- [Prepare master plan](../../../modules/prepare/master-plan.md)
- [Runtime master plan](../../../modules/runtime/master-plan.md)
- [Engine master plan](../../../modules/engine/master-plan.md)
- [CPU master plan](../../../backends/cpu/master-plan.md)
- [Data master plan](../../data/master-plan.md)
- [Text master plan](../../text/master-plan.md)
- [Training master plan](../../training/master-plan.md)
- [Completed static RNN task](0015-static-packed-rnn-sequence.md)
- [Completed static GRU task](0016-static-packed-gru-sequence.md)
- [Completed static LSTM task](0017-static-packed-lstm-sequence.md)
- [Completed automatic recurrent task](0020-automatic-recurrent-initialization-and-sequence-defaults.md)
- [Completed bidirectional task](0020c-bidirectional-static-recurrent-composition.md)

## Architecture constraints

- This task explicitly changes and clarifies the architecture contract through the coordinated
  architecture process. Planning text alone cannot authorize the decision.
- Existing dependency directions remain unchanged. In particular, NN still depends only on
  Model; Runtime still does not depend on Model, Prepare, Engine, or concrete backends; and
  concrete backends still do not depend on Engine.
- The new rule must preserve `Tensor` as public model state rather than graph IR and preserve the
  prohibition on `Operation` and `CompiledNode` in Runtime hot paths.
- The ADR and focused explanations must agree with the authoritative contract exactly. If a
  coherent decision would require nested graph regions, Runtime graph inspection, a new module,
  or a dependency reversal, stop and report the architecture blocker instead of weakening the
  selected boundary.
- No implementation status claim may be added. Every described Java name and method remains
  explicitly planned until its owner task completes.

## Package impact

Existing packages used: none. This task changes no Java package.

Packages added, moved, renamed, or removed: none.

The architecture decision reserves ownership, not Java source placement. Model 0025E must later
confirm exact package placement before becoming Ready.

## Affected files

Expected exact implementation scope:

1. `ARCHITECTURE.md`
2. `docs/architecture/current-architecture-plan.md`
3. `docs/architecture/lifecycle.md`
4. `docs/architecture/module-boundaries.md`
5. `docs/architecture/runtime-prepare-backend-boundary.md`
6. `docs/architecture/training-graph.md`
7. `docs/design/README.md`
8. `docs/design/decisions/0012-fixed-recurrent-scan-without-regions.md`
9. `docs/planning/extensions/nn/tasks/0021a-fixed-recurrent-scan-architecture-decision.md`
10. `docs/planning/extensions/nn/master-plan.md`
11. `docs/planning/modules/model/master-plan.md`
12. `docs/planning/modules/compiler/master-plan.md`

No other path is permitted. In particular, do not edit the global roadmap or the concurrent CPU
master plan.

## Maximum scope

This task may create or modify exactly the twelve paths above. Stop and propose a later focused
follow-up if another path is necessary.

## Acceptance criteria

- `ARCHITECTURE.md` explicitly authorizes the fixed recurrent operation and explicitly does not
  authorize a general body/region system.
- The exact variants, direction, input/output order, static Shape/type rules, dense output and
  final-state semantics, runtime valid-length validation, zero-length, zero-time, reverse, LSTM
  second state, purity, and side effects are unambiguous and consistent across the contract, ADR,
  and focused explanations.
- The exact planned Model public surface and the non-runnable status are recorded without adding
  Java or claiming implementation.
- Model, Compiler, Planning, Prepare, Runtime, Engine, concrete backend, NN, Data, Training, and
  checkpoint/serialization ownership are each explicit.
- Compiler forward adoption and the initial fail-closed BPTT boundary are explicit; no derivative
  implementation or saved-state strategy is selected.
- One-node/no-time-growth, one prepared transition, hot-path, genuine invalid-coordinate skip,
  and no-active-compaction-promise performance rules are explicit.
- Current static Java-`long[]` APIs and result-shape differences remain truthful and unchanged;
  migration is deferred to NN 0022.
- Model 0025E and Compiler 0006A remain concise Draft rows without detailed specs. NN 0021A is the
  sole Ready NN task; NN 0021B and later rows remain Draft.
- Architecture links, headings, anchors, fences, final newlines, exact scope, no-index whitespace,
  and whole-worktree whitespace checks pass.
- Existing architecture dependency tests pass and no test change is required because dependency
  directions and source inventories do not change.
- The documentation-focused architecture context records reasoned no-change conclusions and the
  final completion summary before marking the task Complete.

## Tests / validation

Run in the clean architecture/documentation implementation context:

```bash
./gradlew :testing:architecture-tests:test

git diff --check
git status --short
```

Also validate:

- every local Markdown link and anchor in the twelve-path set;
- unique headings and balanced fenced code blocks;
- the ADR index and current-architecture index links;
- exact terminology consistency for `recurrent scan`, `valid lengths`, `fixed operation`,
  `body`, `region`, `FORWARD`, `REVERSE`, `BPTT`, and `active-row compaction`;
- exact twelve-path scope while preserving concurrent CPU/backend/global-roadmap changes;
- exactly one new task file and exactly one Ready NN row;
- no detailed Model 0025E, Compiler 0006A, NN 0021B, backend, or later task file;
- no production/test/build/settings diff;
- no-index whitespace for the new ADR and this task file;
- terminal newline and trailing-whitespace checks; and
- whole-worktree `git diff --check`.

Repository-wide tests, Model/Compiler/Runtime/Prepare/Engine/backend tests, Javadocs,
backend-conformance, and integration tests are not required because this task changes no Java,
build, dependency, or executable behavior. The architecture test is the proportionate focused
guard for unchanged dependency directions.

## Dependencies

- NN 0020C is Complete and fixes concrete RNN/GRU/LSTM directional/state semantics.
- Current Model multi-output provenance and flat graph contracts are Complete.
- Compiler 0006 and its closed first-order inventory are Complete and reveal why a new kind cannot
  land without an ordered Compiler adoption task.
- Prepare 0003 and Runtime 0010/0014 are Complete and provide the current static partition,
  caller-input representation, prepared executable, schedule, run-state, and hot-path boundaries.
- Engine remains Draft; its existing 0001–0002 rows own lifecycle composition and typed input/output
  mapping needed before a public runnable scan claim.
- No CPU, backend, Data, Text, Training, or Checkpoint implementation is a prerequisite for this
  architecture decision.

## Follow-up tasks

- Model 0025E, Draft only: implement the fixed semantic family and exact Tensor surface.
- Compiler 0006A, Draft only: adopt forward capture/inference/inventory and fail BPTT closed.
- A later concrete-backend task, not yet detailed: truthfully lower and execute supported static
  variants with complete length prevalidation and skipped invalid-coordinate transition work.
- Engine 0001–0002, already Draft: compose the lifecycle and map typed caller inputs/publications.
- NN 0021B, Draft program row only: coordinate the owner-specific implementation checkpoint.
- A later Compiler BPTT task only after the forward executable path is stable and its saved-versus-
  recompute contract is selected.
- NN 0022, Draft only: add the Data-owned runtime-valid-length NN API and deliberate compatibility
  migration after the complete prerequisite is executable.
- Dynamic Shapes, active-row compaction, arbitrary masks with holes, and a general region system
  remain separate future decisions.

## Architecture impact

Expected impact: explicit architecture clarification and significant design decision.

The task updates the authoritative contract, focused architecture explanations, and ADR together.
It does not change a dependency direction or add a module, so architecture dependency tests
require execution but no source update. If implementation discovers that the fixed ordinary-node
design cannot preserve the current lifecycle boundaries, stop and report the conflicting rule and
required decision instead of silently introducing regions or Runtime graph access.

## Implementation prompt

Use this prompt in a separate clean architecture/documentation-focused agentic task/thread:

```text
You are the clean architecture/documentation implementation agent for Synaptik NN task 0021A.
Do not use any GSD skill or workflow. Do not commit or push. Preserve all concurrent CPU/backend
and global-roadmap changes.

Read AGENTS.md, ARCHITECTURE.md, docs/architecture/current-architecture-plan.md,
docs/planning/planning-guide.md, docs/planning/roadmap.md,
docs/planning/extensions/nn/tasks/0021a-fixed-recurrent-scan-architecture-decision.md, every
architecture/ADR/master/task/source contract directly linked from that task, and the final dirty
worktree status in full.

Implement only the exact twelve-path architecture/documentation scope. Ratify the selected fixed
ordinary multi-output recurrent operation and explicit no-region boundary exactly as specified.
Do not add Java, build changes, public APIs, executable claims, a detailed later task, a global-
roadmap edit, or a CPU-master edit. Stop on any architecture conflict or need for a thirteenth
path.

Run the specified architecture and documentation validation, record exact evidence and reasoned
no-change conclusions, update the task completion summary, and mark it Complete only when every
gate passes.
```

## Local decisions

- The fixed operation is the sole authorized recurrent control-flow representation at this
  frontier. Its backend-internal bounded loop is execution strategy, not graph structure.
- The exact family, directions, ordered signatures, fully static descriptor boundary, runtime
  valid-length meaning, dense aligned output, final-state semantics, purity, failure atomicity,
  owner boundaries, and performance rules are normative in `ARCHITECTURE.md` and summarized by
  accepted ADR 0012.
- Existing Planning, Prepare, and Runtime contracts require no source or plan task merely to name
  recurrence. Later implementation must report evidence of a real shared-contract gap before
  proposing a change.
- Model 0025E and Compiler 0006A remain concise Draft rows. NN 0021B remains the Draft coordinating
  implementation program. No later detailed task was created.

## Known limitations

- This task authorizes and specifies architecture only; it adds no runnable Java API or execution
  support.
- The first selected capability supports runtime-changing valid-length values only for compatible
  fully static descriptors.
- BPTT, dynamic Shapes, physical active-row compaction, arbitrary masks with holes, and general
  user-defined control flow remain deferred.
- Public Engine exception translation for an invalid runtime length remains intentionally
  unselected.

## Validation evidence

- Clean architecture/documentation context `/root/nn_0021a_architecture` read the authoritative
  contract, all five focused architecture documents, directly related ADRs, General/
  Architecture/ADR/Planning documentation profiles, planning guide and roadmap, relevant owner
  master plans and recurrent tasks, flat Model/Compiler graph contracts, Planning capability,
  staged Prepare, Runtime cold-binding/run-state, Engine frontier, concrete-backend capability,
  and architecture-test sources before finalizing the decision.
- Independent final review context `/root/nn_0021a_docs_review` rechecked those contracts against
  current Model, Compiler, Planning, Prepare, Runtime, Engine, backend-capability, recurrent-NN,
  Data, state-dictionary, checkpoint, and architecture-test sources. It corrected one inaccurate
  parameter-only persistence sentence: current NN state dictionaries contain both parameter and
  persistent-buffer bindings, and future model checkpoints may persist materialized values from
  both entry kinds. No other semantic, ownership, status, or scope defect remained.
- `./gradlew :testing:architecture-tests:test` passed in the implementation context before the
  independent wording correction: `BUILD SUCCESSFUL`; all three Gradle tasks were up to date. The
  retained result set has four suites and six tests with zero failures, errors, or skips. The
  independent review reused this evidence because its documentation-only state/checkpoint wording
  correction changes no dependency direction, Runtime source inventory, or executable contract;
  no architecture-test source changed.
- A Ruby twelve-file documentation check passed: unique headings, balanced fenced code blocks,
  terminal newlines, and no trailing whitespace. A Ruby local-link/anchor check passed for every
  Markdown link in the exact twelve-file set, including the ADR and current-architecture index
  links. The independent review reran the combined final check after its correction and received
  `PASS: 12 files; headings, fences, newlines, whitespace, local links, and anchors`. Its first
  two invocations exposed only validator compatibility defects, which were corrected outside the
  repository before the final pass.
- Terminology inspection confirmed consistent use of `recurrent scan`, `valid lengths`, `fixed
  operation`, `body`, `region`, `FORWARD`, `REVERSE`, `BPTT`, and `active-row compaction` across
  the contract, ADR, focused explanations, task, and owner plans.
- `git diff --no-index --check --quiet /dev/null` for the new ADR and this new task returned the
  expected status 1 for a present diff and emitted no whitespace diagnostic. Final whole-worktree
  `git diff --check` passed.
- Scope inspection against the captured initial `git status --short` confirmed this task changed
  exactly the twelve authorized paths. It added exactly ADR 0012 and this task file; no Model
  0025E, Compiler 0006A, NN 0021B, backend, or later detailed task was created. Existing
  concurrent CPU source/test/guide/planning work, Data planning, and the global roadmap remained
  outside this task and were neither staged nor altered by this context.
- Status/frontier inspection confirmed NN 0021A transitioned `Ready` to `In progress` to
  `Complete`; NN 0021B–0024, Model 0025E, and Compiler 0006A remain concise Draft rows with no
  detailed specifications and no executable implementation claim.
- No production Java, tests, Javadoc, package documentation, Gradle/settings, dependency,
  Tensor/Compile/Training API guide, glossary, Trace, conformance, or integration path changed.
  No module/root/backend suite or Javadoc generation was required for this architecture-only
  documentation task.

## Implementation notes

- Canonical implementation context: `/root/nn_0021a_architecture`; canonical independent final
  review context: `/root/nn_0021a_docs_review`.
- Updated the authoritative contract, five focused architecture explanations, current-
  architecture index, design-decision index, and new ADR 0012 as one coordinated architecture
  change.
- Synchronized only the NN, Model, and Compiler planning frontiers. Existing concurrent CPU/
  backend work and `docs/planning/roadmap.md` were preserved unchanged by this task.
- Dependency directions, Gradle structure, production APIs, Compiler/Planning/Prepare/Runtime/
  Engine/backend Java, NN behavior, Training/Data/Text behavior, Trace schemas, conformance tests,
  and integration tests do not change. Architecture-test sources remain accurate because no
  dependency edge or reviewed Runtime source inventory changed.
- Public Tensor, Compile, and Training API guides and Javadoc remain current because all named
  recurrent Java types and methods are explicitly planned and no declaration or executable
  behavior exists. The glossary remains unchanged because the architecture documents define
  `recurrent scan`, valid lengths, BPTT, and active-row compaction locally and the task's fixed
  operation decision does not change an existing reusable glossary term.
- NN state dictionaries retain their existing parameter-and-buffer boundary; future model
  checkpoints may persist materialized values from both entry kinds. No checkpoint format or
  serialization contract is added. The CPU plan and global roadmap remain outside this task's
  maximum scope.

## Completion summary

- Completed changes: accepted the fixed ordinary flat multi-output recurrent-scan architecture,
  exact semantics and planned Model surface, static-Shape/runtime-valid-length boundary, dense
  alignment and final states, complete pre-mutation validation, purity, lifecycle ownership,
  fail-closed initial BPTT boundary, one-node/one-prepared-transition performance contract, and
  unchanged static-NN migration boundary; added ADR 0012 and synchronized focused explanations.
- Files changed or created: exactly the twelve paths listed under Affected files.
- Tests and validation: architecture tests and all specified Markdown, link/anchor, heading,
  fence, newline, whitespace, terminology, scope, inventory, and status checks passed as recorded
  above.
- Documentation-agent review: clean implementation context `/root/nn_0021a_architecture` and
  independent final review context `/root/nn_0021a_docs_review` completed the targeted passes; the
  latter corrected the state-dictionary/checkpoint wording. There was no executable implementation
  context or Java behavior to hand off or retest.
- Documentation impact: the authoritative contract, five focused explanations, two indexes, ADR,
  and three owner plans are synchronized. Public API guides remain unchanged because the surface
  is planned and non-runnable.
- Javadoc review: no Java declaration or implementation changed; existing Javadoc remains
  accurate.
- Glossary impact: no change; decision-specific terms are defined locally and no existing reusable
  glossary meaning changed.
- Unresolved issues: BPTT, dynamic Shapes, public Engine exception translation, physical active-
  row compaction, arbitrary masks with holes, general regions, and static-API migration remain
  intentionally deferred.
- Follow-up required: NN 0021B must coordinate Model 0025E, Compiler 0006A, Engine 0001–0002, and
  truthful concrete-backend implementation in owner order before any runnable recurrent-scan or
  NN runtime-length claim.

Status: Complete
