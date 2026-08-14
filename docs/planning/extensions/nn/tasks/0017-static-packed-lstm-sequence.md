# Task 0017: Static Packed LSTM Sequence

## Status

Complete

## Goal

Add one final public long short-term memory (LSTM) sequence container that statically compacts and
threads both recurrent states through the concrete `LstmCell` signature. The public sequence
output is each step's exact compact next-hidden Tensor. Compact next-cell Tensors remain explicit
internal carried values, and the result restores both final hidden and final cell rows to original
batch order. Explicit copied Java lengths alone determine activity; Tensor values never do.

```text
static time-major input + initial hidden + initial cell + copied Java lengths
  -> compact input, hidden, and cell active batches
  -> one LstmCell.forward call per non-empty step
  -> expose exact compact next-hidden outputs
  -> carry exact compact next-cell values to the next step
  -> restore finalHidden and finalCell from each row's exit step
```

This task was implemented simultaneously with NN 0016 by explicit user request. Both independently
reuse completed NN 0015 and write disjoint implementation paths. No shared recurrent abstraction
or shared helper was introduced; one joint documentation context integrated their public story.

## Scope

- Add final public `io.github.pho001.synaptik.nn.layers.LstmSequence` extending `Module` directly.
- Add final public record `LstmSequenceForwardResult` with exactly `List<Tensor> packedOutputs`,
  `Tensor finalHidden`, and `Tensor finalCell`, in that order.
- Declare exactly `LstmSequence(LstmCell cell)`, `LstmCell cell()`, and
  `LstmSequenceForwardResult forward(Tensor input, Tensor initialHidden, Tensor initialCell,
  long[] lengths)`.
- Permanently own the exact supplied `LstmCell` under child name `cell`; declare and retain no
  direct parameter, buffer, recurrent state, length, index, output, or step state.
- Require fully static floating input `[time, batch, inputSize]` and fully static floating initial
  hidden and initial cell Shapes `[batch, hiddenSize]`, plus exactly one copied length in
  `[0, time]` per original batch row.
- Preserve ascending original-row order without sorting. Construct exactly `max(lengths)` steps,
  gather only active input/hidden/cell rows, and call `LstmCell.forward` once per step.
- At step zero, share one active-original INT64 index leaf across the input, initial-hidden, and
  initial-cell gathers. At later steps, share one relative-survivor index leaf across previous
  compact next-hidden and next-cell gathers.
- Append only each exact `LstmCellForwardResult.nextHidden()` to public `packedOutputs`. Carry the
  exact `nextCell()` in an internal temporary list solely for later recurrence and final-cell exit
  selection; do not expose all cell states or retain either list after return beyond references
  reachable from the result expressions.
- Restore final hidden and cell rows independently. A zero-length row uses its matching initial
  state row; a positive row uses the exact hidden/cell row from its last active step. Stack both in
  original order.
- For empty batch or all-zero lengths, return an empty packed list plus the exact `initialHidden`
  and `initialCell` references without Tensor creation or cell invocation.
- Complete all caller-controlled current-cell-schema, descriptor, length, promotion, broadcast,
  gate/state, gather/select, and both final-stack checks before the first new Tensor identity.
- Preserve mode, ownership, recursive state paths, replacement observation, and no-retained-state
  behavior through existing contracts.
- Add focused exact-surface/result, ownership, validation-order, two-state packing/carrying,
  provenance, restoration, replacement, mode, zero-length, and exclusion tests.
- Draft complete Javadocs in the two new production types and hand off to the one joint clean
  documentation pass after both executable tasks stabilize.

## Out of scope

- `RnnSequence`, `GruSequence`, cell changes, or a shared recurrent abstraction/helper/state tuple.
- Public per-step packed cell-state collection, dense padded output, flattened packed Tensor,
  unpacking metadata, batch-first, sorting/permutations, reverse/bidirectional or stacked layers.
- Runtime Tensor lengths/masks, Java masks, zero-value inference, dense masking, padding values,
  or a recurrent Model scan/control-flow body.
- Hidden/cell Buffer state, defaults, initializers, mutation, reset/detach, stateful forward, or
  unary/Sequential adaptation.
- Peepholes, projections, clipping, recurrent dropout, alternate LSTM equations or checkpoint
  packing, compiler/autograd/training/runtime/backend/execution work.
- Model, build, dependency, architecture, ADR/test, global roadmap, CPU, shared documentation, or
  task-0016 edits by the implementation agent.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [ADR 0007](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)
- [Training API](../../../../api/training-api.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Planning guide](../../../planning-guide.md)
- [NN master plan](../master-plan.md)
- [Completed GRU cell task](0013-gru-cell.md)
- [Completed LSTM cell task](0014-lstm-cell.md)
- [Completed static packed RNN sequence task](0015-static-packed-rnn-sequence.md)
- [Parallel GRU sequence task](0016-static-packed-gru-sequence.md)

## Architecture constraints

- NN continues to depend only on Model. Model owns all Tensor operations, type/Shape/layout
  semantics, eager leaves, identity, and provenance.
- Both recurrent states remain caller-threaded Tensor data, never module persistent state.
- `LstmSequence` is a direct Module; no unary or `Sequential` adapter may erase two state inputs.
- Java lengths may control static expression construction. Runtime Tensor data cannot control
  active Shape or loop count without a future Model scan/control-flow contract.
- `packedOutputs` means visible next-hidden sequence outputs. Internal compact next-cell values
  are necessary carried state, not another public output sequence. Both restored final states are
  explicit because omitting cell state would make continuation impossible.
- The result is an NN composition carrier, not a Model multi-output producer or runtime result.
- Do not add a common helper/source/test or recurrent interface during parallel work. Stop if a
  new Model API, dependency, shared path, hidden state, or unlisted file is required.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.layers`
- `io.github.pho001.synaptik.nn.module`
- existing Model Tensor, type, Shape, layout, and descriptor packages

Packages added or changed:

- `io.github.pho001.synaptik.nn.layers` — two public LSTM-sequence types; no package is added

Type placement:

- `LstmSequence` — NN owns concrete two-state cell composition and static packing policy.
- `LstmSequenceForwardResult` — NN owns visible compact outputs and both restored final states.

## Public API

```java
public final class LstmSequence extends Module {
    public LstmSequence(LstmCell cell)
    public LstmCell cell()
    public LstmSequenceForwardResult forward(
            Tensor input,
            Tensor initialHidden,
            Tensor initialCell,
            long[] lengths)
}

public record LstmSequenceForwardResult(
        List<Tensor> packedOutputs,
        Tensor finalHidden,
        Tensor finalCell)
```

No other declared public/protected API is allowed. The record checks null packed list, first null
element while snapshotting, null final hidden, then null final cell. It retains exact references,
performs no descriptor relationship validation, and creates no Tensor.

## State schema and ownership

| Kind | Local name | Value |
|---|---|---|
| Child | `cell` | exact supplied `LstmCell` |

There is no direct state. Recursive paths preserve exact LSTM cell order:
`cell.inputWeight`, `cell.hiddenWeight`, optional `cell.bias`. Constructor ownership and mode
propagation follow `Module`. No caller state or compact result enters discovery/state dictionary.

## Forward and packing contract

With copied `L[b]` and `active(t)` defined as in NN 0015, each step performs:

1. `input.select(0, t)` and an active-original INT64 index leaf;
2. axis-zero gather of compact input;
3. at step zero, axis-zero gathers of both initial states using that same active leaf;
4. later, one relative-survivor INT64 leaf shared by axis-zero gathers from prior compact hidden
   and prior compact cell;
5. one `cell.forward(compactInput, compactHidden, compactCell)` call;
6. append the exact `nextHidden` to public packed outputs and the exact `nextCell` to a temporary
   carried-cell list.

For row `b`, final hidden/cell each come from their corresponding initial row when `L[b] == 0`,
or from the corresponding packed hidden/cell result at step `L[b] - 1` and its stable exit
position. Stack hidden rows first, then cell rows, both in original order. This ordering fixes
observable Tensor identity allocation and delegated failure prefixes.

## Validation and side-effect order

`forward` performs:

1. null checks for `input`, `initialHidden`, `initialCell`, then `lengths`;
2. clone lengths once;
3. read current input weight, hidden weight, and optional bias once for descriptor preflight;
4. revalidate the complete packed LSTM schema and checked gate bounds;
5. validate input floating/rank-three/fully-static, then hidden floating/rank-two/fully-static,
   then cell floating/rank-two/fully-static;
6. validate input feature, hidden feature, cell feature, and both state batch extents against the
   input batch, in that order;
7. validate copied length count and elements in original order, deriving a maximum no greater
   than `Integer.MAX_VALUE`;
8. for every distinct active extent, prevalidate compact input/hidden/cell projections, eight gate
   slices, four activations, next-cell equation, next-hidden equation, and exact
   `[activeCount, hiddenSize]` Shapes/types for both returned states;
9. prevalidate every final hidden and final cell selection and the two independent exact-type/
   exact-Shape STACK requests, including zero-length rows; and
10. only then construct expressions in the fixed order above and restore hidden before cell.

Every caller-controlled failure consumes no new Tensor ID or storage. Later unexpected failures
retain completed prefixes with no partial carrier and no module mutation. Parameter preflight does
not make multi-step calls atomic with replacement; caller synchronization is required for a
consistent binding snapshot.

For one row, one step, and no bias, creation order is SELECT; shared index; input GATHER; hidden
GATHER; cell GATHER; the exact twenty-five-Tensor LSTM cell chain; final-hidden SELECT; final-cell
SELECT; final-hidden STACK; final-cell STACK: 34 new Tensor identities. A biased cell creates 35.
The result carrier itself creates no Tensor identity. Tests lock exact source associations,
including shared gather indices and hidden-before-cell restoration order.

## Test requirements

- Reflectively verify finality, direct `Module` inheritance, exact constructor/accessor/four-input
  forward surface, exact three-component record order, and absence of public/protected extras or
  retained length/input/output/index/hidden/cell fields.
- Verify record null order, structural packed-list snapshot, exact Tensor references,
  unmodifiable access, and ordinary generated record equality.
- Verify sole-child ownership, null/already-owned failures, recursive LSTM state paths/order,
  state dictionary, train/eval propagation, forward mode independence, and compatible parameter
  replacement affecting later construction only.
- For `[5,3,1]`, unsorted `[1,3,2]`, and `[0,2,0]`, inspect active and survivor index values,
  compact input/hidden/cell Shapes, shared state-gather index identity, exact cell-result hidden
  and cell provenance, and both original-order exit-state STACK inputs.
- Verify zero-valued storage-free data remains active and no cumulative scan appears. All-zero,
  zero-time, and empty-batch requests must return an empty list plus both exact initial references
  without consuming an ID.
- Cover null order; non-floating input/hidden/cell; wrong rank and every dynamic axis; input,
  hidden, and cell feature mismatches; both state batch mismatches; length count/range/list limit;
  ordinary current-cell schema; every gate/state broadcast; and separate hidden-stack/cell-stack
  exact-type failures. Every caller-controlled failure precedes expression creation.
- Lock eager index metadata, step-zero and later index sharing, exact next-hidden public list,
  internal next-cell carrying, hidden-before-cell final restoration, one-step 34/35-ID order, and
  late-failure no-rollback behavior.
- Prove no public packed-cell list, hidden module state, mask/scan/dense padding, unary/Sequential
  adapter, direct Model operation construction, shared recurrent abstraction, or new dependency.

## Affected files

Implementation-agent ownership is exactly:

1. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/LstmSequence.java`
2. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/LstmSequenceForwardResult.java`
3. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LstmSequenceTest.java`
4. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LstmSequencePackingTest.java`
5. `docs/planning/extensions/nn/tasks/0017-static-packed-lstm-sequence.md`

The one later joint documentation pass may finalize the two production Javadocs and owns:

6. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`
7. `docs/api/training-api.md`
8. `docs/glossary.md`
9. `docs/planning/extensions/nn/master-plan.md`

It may finalize this task evidence after handoff. No other path is permitted. The LSTM
implementation agent must not edit paths 6–9 or any task-0016 path.

## Maximum scope

The implementation phase may modify only its five exact paths. The complete joint task surface is
nine paths within the exact fourteen-path union. Stop before a fifteenth path or shared helper.

## Acceptance criteria

- The exact direct-Module class and three-component immutable result APIs exist without extras.
- The sole LSTM child owns all module state; neither recurrent state is retained.
- Each active step shares indices exactly, exposes next hidden, carries next cell, and omits every
  padded row using explicit lengths only.
- Final hidden and cell rows are independently restored from exact exit-step or initial sources;
  all-zero and empty requests return both exact initial references with no expressions.
- Preflight, two-state type/Shape behavior, provenance/ID order, mode, replacement, failure
  effects, and result ownership are covered by focused tests.
- No public packed cell sequence, shared abstraction, runtime scan/mask, dense padding,
  dependency, execution claim, or unlisted path is introduced.
- Focused LSTM sequence tests and the one combined NN module suite pass.
- The joint clean documentation pass finalizes all Javadocs/shared docs and synchronizes both
  tasks/master before Complete.

## Tests / validation

The LSTM implementation agent runs:

```bash
./gradlew :extensions:nn:test \
  --tests io.github.pho001.synaptik.nn.layers.LstmSequenceTest \
  --tests io.github.pho001.synaptik.nn.layers.LstmSequencePackingTest
```

After both parallel executable diffs stabilize, exactly one coordinator-designated implementation
or validation context runs `./gradlew :extensions:nn:test` once for their union. The joint docs
context reuses it and runs final `./gradlew :extensions:nn:javadoc`, generated-page, exact surface,
private-state, import/dependency, Markdown, fourteen-path union, status/frontier, newline, and
`git diff --check` validation. Broader suites remain at owning checkpoints or CI.

## Dependencies

- NN 0014 provides the exact two-state LSTM cell and result contract.
- NN 0015 proves stable static active-set compaction and exit restoration.
- Completed Model SELECT/GATHER/STACK/eager-leaf/Shape/promotion/provenance APIs.
- No technical dependency on NN 0016; simultaneous execution is the recorded explicit exception.

## Follow-up tasks

- Reassess a shared recurrent sequence abstraction only after the three concrete containers and
  results are complete and a real consumer can preserve LSTM's second state without erasure.
- Runtime-dynamic lengths/masks need a separate Model recurrent scan/control-flow design.
- Dense unpacking, bidirectionality, stacking, and recurrent dropout remain separate consumers.

## Architecture impact

Expected impact: None. Stop on any required architecture/dependency change.

## Implementation prompt

```text
You are the clean-context implementation agent for Synaptik NN task 0017. Do not use GSD, commit,
or push. Preserve concurrent CPU and task-0016 work.

Read AGENTS.md, ARCHITECTURE.md, the current architecture plan, planning guide/roadmap, NN master
plan, this task, completed NN 0014 and 0015, final Module/Parameter/LstmCell/result/RnnSequence
source and tests, and directly relevant Model Tensor/Shape/layout/provenance contracts in full.

Implement exactly task 0017 in its five implementation-owned paths. Draft complete Javadocs in
the two production types, run only the focused LSTM sequence suites, record exact evidence in this
task, and report executable stability. Do not edit package-info, Training API, glossary, NN
master, roadmap, task 0016, or a shared helper/source/test. The coordinator will arrange one
combined NN suite after both parallel implementations and one later clean joint documentation
pass. Stop on architecture, API, scope, or shared-file uncertainty.
```

## Documentation-agent handoff

The single later clean context receives both specs/diffs, both focused results, and the combined
NN result. It independently validates behavior, finalizes all four new type Javadocs plus shared
package-info, Training API, glossary, task/master evidence, and reasoned no-change conclusions.
It must not rerun successful Java tests unless executable code changes.

## Local decisions

- Public packed outputs are next-hidden Tensors, matching the visible output of an LSTM step.
  Compact next-cell Tensors are carried internally and only their restored final state is public.
- Return both final states in hidden-then-cell order, matching `LstmCellForwardResult` and allowing
  truthful continuation.
- Share active and survivor index leaves across both state gathers to express one stable active
  set rather than duplicate logically equal metadata.
- Restore and stack hidden before cell to make failure prefixes and Tensor IDs deterministic.
- Duplicate bounded private packing/preflight logic during parallel work; no shared abstraction is
  justified or safely owned yet.

## Known limitations

- Lengths are Java construction metadata and all participating Shapes are fully static.
- Only hidden outputs are exposed per step; intermediate cell states have no public list.
- Construction is not atomic with concurrent parameter replacement.
- No numerical, gradient, compile, backend, scheduling, or execution support is claimed.

## Validation evidence

- Planning context `/root/nn_0016_0017_planning` selected the explicit three-component result and
  two-state carrying/restoration contract after reviewing current cell, sequence, Model, state,
  training, and glossary boundaries.
- The user-authorized parallel exception is safe because implementation paths are disjoint and
  both tasks depend independently on completed NN 0015; shared documentation has one later owner.
- Planning-only validation resolved every local link in the master plan and both new tasks,
  confirmed balanced fences and final newlines, passed separate no-index whitespace checks for
  both new files, and passed whole-worktree `git diff --check`. Status inspection found exactly
  the two explicitly justified Ready NN rows and exactly the three intended NN planning paths.
  Concurrent CPU source, test, planning, and roadmap changes remained untouched.
- Implementation context `/root/nn_0017_implementation` added the exact direct-Module sequence,
  three-component result, and two focused suites in only the four planned Java paths. The
  sequence owns only the exact `cell` child, snapshots Java lengths, preserves original active-row
  order, shares step-zero and survivor indices across both state gathers, exposes only exact
  next-hidden results, carries exact next-cell results internally, and restores hidden then cell.
- The first `./gradlew :extensions:nn:compileJava` found one duplicate local variable name in
  metadata preflight; the name was corrected and the rerun passed. The first
  `./gradlew :extensions:nn:testClasses` found a test-only non-effectively-final lambda capture;
  the fixture was corrected. The first focused two-suite run then found one incorrect test
  expectation for the fourth input gate-slice identifier; inspection confirmed the implemented
  fixed LSTM order, the assertion was corrected from offset 15 to 12, and no production change
  resulted from that failure.
- Final implementation-context command
  `./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.LstmSequenceTest
  --tests io.github.pho001.synaptik.nn.layers.LstmSequencePackingTest` passed 2 suites and 15 tests
  with zero failures, errors, or skips after executable and test code stabilized. It covers the
  exact surface/result, ownership/mode/state paths, null and descriptor order, all dynamic axes,
  features/batches/lengths, separate final-state stack types, replacement, `[5,3,1]`, unsorted
  `[1,3,2]`, zero-length rows, stable and shared active/survivor indices, exact hidden publication
  and cell carrying, initial/exit restoration, no zero inference/scan, and exact 34/35-ID cases.
- Preliminary `./gradlew :extensions:nn:javadoc` passed after production source stabilized. Both
  generated type pages exist. No production source changed after that pass; the only later Java
  change expanded one focused test with the biased 35-ID case and missing dynamic-axis fixtures.
- `javap -public` confirmed only the planned constructor, `cell()`, four-input `forward`, and
  ordinary three-component record API. `javap -private` confirmed the sequence's sole field is
  final `LstmCell cell`; all packing/preflight values are method-local and no nested type exists.
  The focused reflective tests independently lock these surfaces and exclusions.
- Import inspection found only Model, NN Module/Parameter, and JDK imports. There is no Operation
  construction, shared recurrent helper/abstraction, mask/scan API, unary/Sequential adaptation,
  Gradle/dependency change, or edit outside the five implementation-owned paths. Concurrent GRU,
  CPU, shared documentation, master-plan, and roadmap work remained untouched.
- Final whole tracked-worktree `git diff --check` passed. Separate
  `git diff --check --no-index /dev/null <path>` checks for each of the five untracked task-owned
  files produced no whitespace diagnostics; exit one was the expected no-index difference status.
  A forbidden-import/construction scan produced no match. Final status inspection showed exactly
  the four new LSTM Java paths and this task for context 0017 alongside the separately owned GRU,
  CPU, shared-documentation, master-plan, and roadmap paths.
- Joint clean documentation context `/root/nn_0016_0017_docs` independently reviewed the final
  LSTM and GRU implementations, focused tests, shared module/state and Model composition
  contracts, generated Javadocs, Training API, glossary, and planning diff. It found no
  executable, public-API, architecture, dependency, or scope defect and changed no executable
  Java or tests.
- The context reused the stable LSTM focused 2-suite/15-test result and GRU focused
  2-suite/14-test result. Its one coordinated `./gradlew :extensions:nn:test` over the union
  passed 29 suites and 197 tests with zero failures, errors, or skips. No executable Java or test
  changed afterward.
- Final `./gradlew :extensions:nn:javadoc` passed after the shared documentation edits. Generated
  pages for both sequences, both results, and the layers package accurately document defensive
  Java lengths, stable original-order active batches, compact hidden outputs, internal LSTM cell
  carrying, both restored final states, zero-value data, all-zero exact-reference behavior,
  ownership, failures, and construction-versus-execution boundaries.
- Final public/private `javap` confirmed the exact LSTM constructor/accessor/four-input forward,
  ordinary three-component record, direct `Module` inheritance, and sole retained
  `LstmCell cell` field. The authoritative suite's reflection tests lock the same surface. Manual
  JShell reflection produced the expected superclass, field, method, and record-component output;
  its later preferences-flush warning was an environment-only shutdown limitation.
- Production imports remain Java, Model, and existing NN only; the NN Gradle module retains only
  its Model implementation dependency. The five-file Markdown check passed 320 local links, 250
  anchors, balanced fences, and final newlines. Exact-scope inspection found the intended
  fourteen NN paths while preserving concurrent CPU source/test/planning, glossary CPU hunks, and
  global-roadmap work.
- Architecture/ADRs/tests, Tensor and Compile APIs, Training Java API/training graph, Model
  operations/capabilities, Gradle/dependencies, compiler/runtime/prepare/engine, conformance and
  integration tests, backends, other modules, CPU, and the global roadmap need no change: the
  task adds only concrete static NN expression composition and no operation, execution,
  dependency, or architecture contract.
- Final status/later-specification, exact-scope, new-file newline/whitespace, and whole-worktree
  `git diff --check` gates passed. Both tasks/master rows are Complete, no Ready row remains, and
  no later NN task specification was created.

## Implementation notes

- Added `LstmSequence` with complete validate-before-create preflight, stable active compaction,
  exact shared indices for both state gathers, explicit two-state recurrence, compact hidden
  publication, and original-order exit restoration for both states.
- Added `LstmSequenceForwardResult` with defensive structural output-list snapshotting and exact
  null order for packed outputs, final hidden, then final cell.
- Added focused surface/ownership/validation/replacement tests and two-state
  packing/provenance/identity-order tests. Production Javadocs are complete drafts for independent
  joint documentation review.

## Completion summary

- Completed changes: the two-state static packed LSTM sequence, immutable result, focused tests,
  finalized public/package documentation, Training API, glossary, and synchronized planning
  records are complete.
- Files changed or created: the four LSTM Java paths and this task record, plus the four shared
  documentation/planning paths within the exact fourteen-path joint change.
- Tests and validation: focused LSTM 15/15 and coordinated NN 197/197 passed; final Javadoc,
  generated-page, bytecode/reflection, imports/dependency, Markdown, exact-scope/status,
  newline/whitespace, and diff gates passed.
- Documentation-agent review: joint clean context `/root/nn_0016_0017_docs` completed the
  independent review without changing executable Java or tests.
- Documentation impact: shared sequence documentation now distinguishes one-state GRU from
  two-state LSTM carrying and restoration while preserving their common static packing policy.
- Javadoc review: the production drafts were accurate and needed no source change; the shared
  package Javadoc and final generated output were finalized and inspected.
- Glossary impact: the existing recurrent packing terms now cover LSTM hidden-output publication,
  internal cell carrying, and restoration of both final states.
- Unresolved issues: None.
- Follow-up required: None for NN 0017.

Status: Complete
