# Task 0016: Static Packed GRU Sequence

## Status

Complete

## Goal

Add one final public gated recurrent unit (GRU) sequence container that reuses the completed NN
0015 construction-time packing policy through the concrete `GruCell` signature. A fully static
time-major input and a defensively copied Java length array determine a stable compact active
batch for each represented step. The container exposes each exact compact next-hidden Tensor and
restores final hidden rows to original batch order without ever treating numeric zero as padding.

```text
static [time, batch, inputSize] input + initial hidden + copied Java lengths
  -> stable active original rows for each time step
  -> compact input and carried hidden through SELECT/GATHER
  -> one concrete GruCell.forward call per non-empty step
  -> immutable compact next-hidden outputs by time
  -> exit-row selection and STACK in original batch order
```

NN 0016 and NN 0017 were implemented simultaneously by explicit user request. They are
technically independent after completed NN 0015: this task owns only GRU-specific types, tests,
and evidence, while task 0017 owns only LSTM-specific counterparts. One joint documentation pass
finalized their shared documentation and master-plan status after both executable diffs stabilized.

## Scope

- Add final public `io.github.pho001.synaptik.nn.layers.GruSequence` extending `Module` directly.
- Add final public record `GruSequenceForwardResult` with exactly
  `List<Tensor> packedOutputs` then `Tensor finalHidden`.
- Declare exactly `GruSequence(GruCell cell)`, `GruCell cell()`, and
  `GruSequenceForwardResult forward(Tensor input, Tensor initialHidden, long[] lengths)`.
- Permanently own the exact supplied cell under local child name `cell`; declare no direct
  parameter, buffer, recurrent state, length, index, or output state.
- Require fully static floating input `[time, batch, inputSize]`, fully static floating initial
  hidden `[batch, hiddenSize]`, and exactly one copied length in `[0, time]` per original row.
- Preserve stable ascending original-row order without sorting. Construct exactly
  `max(lengths)` steps, gather only active rows, and invoke `GruCell.forward` once per step.
- At step zero, share one active-original INT64 index leaf between input and initial-hidden
  gathers. At later steps, use one active-original index for input and one relative-survivor index
  for the prior compact hidden.
- Expose the exact `GruCell.forward` Tensor at each time in an immutable list. Restore each final
  hidden row from its exit step, or from initial hidden when its length is zero, then stack in
  original order.
- When the batch is empty or every length is zero, return an empty packed list and the exact
  initial-hidden reference without allocating a Tensor identity or calling the cell.
- Complete every caller-controlled descriptor, current-cell-schema, length, promotion, broadcast,
  gather, slice, gate, interpolation, final-selection, and stack check before creating the first
  index leaf or expression.
- Keep forward mode-insensitive and preserve existing child ownership, parameter replacement,
  recursive discovery, state-dictionary, and mode-propagation behavior.
- Add focused exact-surface, ownership, result, validation-order, packing, provenance, exit-state,
  replacement, mode, zero-length, and exclusion tests.
- Draft complete Javadocs in the two production types. Hand them and all evidence to the one joint
  clean documentation-focused pass after both tasks' executable work stabilizes.

For lengths `[5, 3, 1]`, five Java cell calls receive compact batch extents `[3, 2, 2, 1, 1]` and
represent nine logical recurrent rows rather than fifteen dense padded rows.

## Out of scope

- `RnnSequence`, `LstmSequence`, changes to any recurrent cell, or a shared recurrent cell,
  sequence, state tuple, packing helper, base class, interface, or public `PackedSequence`.
- Runtime Tensor lengths or masks, a Java mask, zero-value inference, dense post-cell masking,
  synthesized padding, sorting, permutations, or inverse permutations.
- Dense padded outputs, unpacking, batch-first layout, reverse or bidirectional traversal, stacked
  recurrent layers, residuals, recurrent dropout, attention, or state detachment.
- A recurrent Model scan/control-flow body or reinterpretation of `CUM_SUM`/`CUM_PROD`.
- Default or retained hidden state, hidden Buffers, state initialization, mutation, reset, or an
  adapter into `UnaryTensorModule` or `Sequential`.
- Gradient-rule, compiler, Runtime, Prepare, Engine, backend, numerical execution, checkpoint
  transport, optimizer, or training-session behavior.
- Model, Gradle, dependency, architecture, ADR, architecture-test, conformance, integration,
  global-roadmap, CPU, shared documentation, or task-0017 changes by the implementation agent.

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

## Architecture constraints

- `extensions/nn` retains only its Model dependency. Model owns Tensor identity, descriptors,
  SELECT, GATHER, STACK, eager leaves, gate expressions, Shape algebra, and provenance.
- Recurrent hidden state remains caller-threaded Tensor data and never becomes module state.
- `GruSequence` is a direct `Module`, not a unary module or `Sequential` participant.
- Java lengths are construction metadata. A runtime Tensor cannot determine Java loop count or
  active Shapes under the current expression model.
- The result is an NN composition value, not a Model producer result, checkpoint, or runtime
  result. Construction makes metadata only and proves no execution or skipped physical kernel.
- Do not add a shared packing helper or recurrent abstraction. The current concrete LSTM result
  still has a different two-state contract, and the explicit parallel file partition forbids a
  shared implementation path.
- Stop if the task needs a new Model API, dependency, runtime scan, hidden state, shared source or
  test file, or any path outside its authorized ownership.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.layers`
- `io.github.pho001.synaptik.nn.module`
- existing Model Tensor, type, Shape, layout, and descriptor packages

Packages added or changed:

- `io.github.pho001.synaptik.nn.layers` — two public GRU-sequence types; no package is added

Type placement:

- `GruSequence` — NN owns concrete cell composition and static packing policy.
- `GruSequenceForwardResult` — NN owns the cell-specific compact-output/final-state value.

## Public API

```java
public final class GruSequence extends Module {
    public GruSequence(GruCell cell)
    public GruCell cell()
    public GruSequenceForwardResult forward(
            Tensor input, Tensor initialHidden, long[] lengths)
}

public record GruSequenceForwardResult(
        List<Tensor> packedOutputs,
        Tensor finalHidden)
```

The sequence declares no other public/protected constructor, method, field, nested type,
interface, or overload. The record canonical constructor rejects null `packedOutputs`, then the
first null element while structurally snapshotting the list, then null `finalHidden`. It retains
every exact Tensor reference and performs no descriptor relationship check or Tensor creation.

## State schema and ownership

The sequence has one exact child and no direct state:

| Kind | Local name | Value |
|---|---|---|
| Child | `cell` | exact supplied `GruCell` |

Recursive state paths are `cell.inputWeight`, `cell.hiddenWeight`, and optional `cell.bias` in
existing declaration order. A null cell fails before registration. Existing `Module.child`
semantics reject an already owned cell without partial ownership. Forward retains no caller or
intermediate value.

## Forward and packing contract

Define copied length `L[b]`, `steps = max(L)`, and:

```text
active(t) = original b in ascending order where L[b] > t
```

For every `t` in `[0, steps)`:

1. create `input.select(0, t)`;
2. create one dense contiguous, unlabeled, provenance-free, gradient-ineligible INT64 leaf for
   `active(t)`;
3. gather the time slice on axis zero;
4. at step zero gather `initialHidden` with that same leaf; later create one relative-survivor
   INT64 leaf and gather the previous packed output;
5. call `cell.forward(compactInput, compactHidden)` once; and
6. append that exact next-hidden Tensor.

Final row `b` comes from `initialHidden.select(0, b)` when `L[b] == 0`; otherwise it comes from
the row for `b` in `packedOutputs.get(L[b] - 1)`. A nontrivial result stacks these selections on
axis zero. Input values and host storage are never inspected.

## Validation and side-effect order

`forward` performs:

1. null checks for `input`, `initialHidden`, then `lengths`;
2. one clone of `lengths`, used exclusively thereafter;
3. one descriptor-preflight read of current input weight, hidden weight, and optional bias in cell
   declaration order;
4. complete current GRU packed-schema revalidation and checked derivation of `hiddenSize`;
5. input floating/rank-three/fully-static validation;
6. initial-hidden floating/rank-two/fully-static validation;
7. input/hidden feature and batch equality checks;
8. copied length-count and encounter-order range checks while deriving a maximum no greater than
   `Integer.MAX_VALUE`;
9. non-constructing preflight of every distinct active-count projection, six gate slices,
   reset-after gate equation, interpolation, and exact `[activeCount, hiddenSize]` result;
10. final row SELECT and exact STACK Shape/type preflight, including rejection when a zero-length
    initial row type differs from recurrent row type; and
11. only then the fixed construction order above.

All caller-controlled failures precede the first Tensor ID, expression, or eager storage. Later
allocation, identifier, or unexpected delegated failure preserves successful prefixes without
returning a partial result or mutating the module. Descriptor preflight is not a transaction with
concurrent parameter replacement: every actual cell call observes its then-current compatible
bindings, so callers coordinate replacement when a multi-step snapshot matters.

For a one-row, one-step, no-bias call, creation order is SELECT; shared index leaf; input GATHER;
hidden GATHER; the exact twenty-Tensor GRU cell chain; final-row SELECT; final STACK: 26 new Tensor
identities. A biased cell adds the cell's one bias-ADD identity for 27 total. Tests inspect the
exact order and associations without relying on numerical execution.

## Test requirements

- Reflectively verify finality, direct `Module` inheritance, the exact three-member sequence API,
  the exact two-component record surface, and absence of public/protected extras or retained
  length/input/output/index/state fields.
- Verify record null order, exact element references, structural list snapshot, unmodifiable
  access, and ordinary generated record equality.
- Verify sole-child ownership, null/already-owned failures, direct-empty state, recursive paths,
  state dictionary, train/eval propagation, forward mode independence, and replacement affecting
  only later constructions.
- Inspect `[5,3,1]`, unsorted `[1,3,2]`, and `[0,2,0]` active/survivor index host data, compact
  Shapes, exact GRU input/hidden provenance, exit-row sources, stable order, and final STACK.
- Verify all-zero lengths, zero time, and empty batch return the exact no-expression result; verify
  storage-free/all-zero-valued input remains active and no cumulative-scan kind appears.
- Cover null order, non-floating inputs, wrong ranks, every dynamic axis, both feature mismatches,
  batch/length-count mismatch, first invalid length, list/index limits, current-cell-schema
  validation through ordinary current bindings, gate/interpolation broadcast failure, and mixed
  final-stack type. Every caller-controlled failure must precede a new Tensor ID.
- Lock eager index type, Shape, layout, label, storage, gradient eligibility, provenance absence,
  step-zero sharing, later survivor semantics, one-step 26/27-ID order, and documented late-
  failure non-rollback behavior.
- Prove absence of masks, retained recurrent state, dense padded result, direct `Operation`
  construction, unary/Sequential compatibility, shared recurrent types, and non-Model dependency.

## Affected files

Implementation-agent ownership is exactly:

1. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/GruSequence.java`
2. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/GruSequenceForwardResult.java`
3. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/GruSequenceTest.java`
4. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/GruSequencePackingTest.java`
5. `docs/planning/extensions/nn/tasks/0016-static-packed-gru-sequence.md`

The later joint documentation pass may additionally finalize Javadocs in the two production
files and owns these shared paths exactly:

6. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`
7. `docs/api/training-api.md`
8. `docs/glossary.md`
9. `docs/planning/extensions/nn/master-plan.md`

It may also finalize this task evidence after the implementation handoff. No agent may edit an
unlisted path. The GRU implementation agent must not edit paths 6–9 or any task-0017 path.

## Maximum scope

The implementation phase may modify at most its five exact owned paths. The complete joint change
may touch this task's nine listed paths, within the fourteen-path union recorded by tasks 0016 and
0017. Stop if another path is required.

## Acceptance criteria

- Exact final direct-Module class and two-component result surfaces exist without extra API.
- One exact GRU child owns all recursive state; the container retains no recurrent state.
- Explicit copied lengths alone create stable compact steps and final-row restoration.
- Packed outputs are exact GRU next-hidden results; zero-length/all-zero/empty behavior matches
  the contract and consumes no unexpected IDs.
- Full local preflight, fixed provenance/ID order, no-zero inference, mode, replacement,
  concurrency, and failure-prefix behavior are tested.
- No LSTM/RNN/shared abstraction, runtime scan/mask, dense padding, dependency, execution claim,
  or out-of-scope path is introduced.
- Focused GRU sequence tests and the one coordinated final NN module suite pass.
- One joint clean documentation pass finalizes Javadocs/shared docs, records no-change
  conclusions, and synchronizes both tasks/master before either becomes Complete.

## Tests / validation

The GRU implementation agent runs its focused suites after its owned executable files stabilize:

```bash
./gradlew :extensions:nn:test \
  --tests io.github.pho001.synaptik.nn.layers.GruSequenceTest \
  --tests io.github.pho001.synaptik.nn.layers.GruSequencePackingTest
```

After both implementation agents report executable stability, the coordinator designates exactly
one implementation/validation context to run once:

```bash
./gradlew :extensions:nn:test
```

The joint documentation context reuses those results, runs final
`./gradlew :extensions:nn:javadoc`, validates generated pages, public/private surfaces, imports,
the unchanged Model-only dependency, Markdown links/anchors/fences, exact fourteen-path union,
status, final newlines, and `git diff --check`. Repository-wide, architecture, Model,
backend-conformance, and integration suites remain deferred because no shared contract,
dependency, operation, or execution behavior changes.

## Dependencies

- NN 0013 provides the exact GRU cell/state/formula contract.
- NN 0015 proves the construction-time one-hidden-state packing/restoration policy.
- Completed Model SELECT, GATHER, STACK, eager INT64 leaf, Shape, promotion, and provenance APIs.
- No dependency on task 0017; both run in parallel under the recorded user-authorized exception.

## Follow-up tasks

- NN 0017 proceeds simultaneously and independently for the concrete two-state LSTM signature.
- Reassess a shared recurrent sequence abstraction only after all three concrete sequence results
  are complete and a consumer proves a type-safe benefit.
- Runtime-dynamic masks/lengths remain blocked on a future genuine Model recurrent scan/control-
  flow contract.

## Architecture impact

Expected impact: None. Stop and report if implementation requires an architecture or dependency
change.

## Implementation prompt

```text
You are the clean-context implementation agent for Synaptik NN task 0016. Do not use GSD, commit,
or push. Preserve concurrent CPU and task-0017 work.

Read AGENTS.md, ARCHITECTURE.md, the current architecture plan, planning guide/roadmap, NN master
plan, this task, completed NN 0013 and 0015, final Module/Parameter/GruCell/RnnSequence source and
tests, and directly relevant Model Tensor/Shape/layout/provenance contracts in full.

Implement exactly task 0016 in its five implementation-owned paths. Draft complete Javadocs in
the two production types, run only the focused GRU sequence suites, record exact evidence in this
task, and report executable stability. Do not edit package-info, Training API, glossary, NN
master, roadmap, task 0017, or any shared helper/source/test. The coordinator will arrange one
combined NN suite after both parallel implementations and one later clean joint documentation
pass. Stop on architecture, API, scope, or shared-file uncertainty.
```

## Documentation-agent handoff

The later one joint clean context receives both task specs, both executable diffs, both focused
results, and the one combined NN result. It independently reviews behavior/tests, finalizes the
four new types' Javadocs plus shared package-info, Training API, glossary, both task records, and
the NN master plan. It records reasoned no-change conclusions for architecture/ADR/tests, Tensor
and Compile APIs, Training Java/graph, Model capabilities, Gradle/dependencies, compiler/runtime/
prepare/engine, backend conformance/integration, global roadmap, CPU, and other modules. It must
not repeat successful Java tests unless executable Java changes.

## Local decisions

- Keep a cell-specific result even though its component structure matches RNN; structural
  similarity alone does not prove interchangeable cell ownership or a useful abstraction.
- Preserve the exact NN 0015 active-set, index-leaf, survivor, and final-row policy.
- Return packed next-hidden outputs by time, not a dense padded Tensor or flattened value.
- Duplicate bounded private preflight/packing logic rather than create a shared source file during
  parallel work. Refactoring awaits a concrete post-LSTM consumer and separate task.

## Known limitations

- Lengths and all input/state Shapes are construction-time/static.
- Results are compact by time and have no dense unpacking metadata.
- Multi-step construction is not atomic with parameter replacement.
- The task constructs metadata and claims no gradient, compiler, backend, or execution support.

## Validation evidence

- Planning context `/root/nn_0016_0017_planning` selected this concrete API after reviewing the
  architecture/planning contracts, completed recurrent cells and RNN sequence, Module/state,
  relevant Model composition APIs/tests, Training API, and glossary.
- The explicit user request authorizes simultaneous 0016/0017 execution. Dependencies and the
  five-path implementation ownership sets are disjoint; shared documentation is deferred to one
  later clean context.
- Planning-only validation resolved every local link in the master plan and both new tasks,
  confirmed balanced fences and final newlines, passed separate no-index whitespace checks for
  both new files, and passed whole-worktree `git diff --check`. Status inspection found exactly
  the two explicitly justified Ready NN rows and exactly the three intended NN planning paths.
  Concurrent CPU source, test, planning, and roadmap changes remained untouched.
- Implementation, focused tests, combined NN validation, and documentation evidence are pending.
- Clean implementation context `/root/nn_0016_implementation` added the exact final direct
  `GruSequence`, two-component `GruSequenceForwardResult`, and both focused test classes in the
  four implementation-owned Java paths. The implementation duplicates only the bounded
  cell-specific static-packing logic authorized by this task and introduces no shared recurrent
  abstraction, dependency, runtime Tensor mask, dense padded output, or retained recurrent state.
- The first focused run compiled production and tests and ran 14 tests, with one test failure. The
  failure was an incorrect ID-counter assertion in the current-schema corruption test: that test
  created two ordinary input leaves after capturing the counter. Both leaves were moved before
  the snapshot; production behavior was unchanged.
- After that test-only correction and final schema-order alignment with `GruCell`, the stabilized
  focused command
  `./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.GruSequenceTest --tests io.github.pho001.synaptik.nn.layers.GruSequencePackingTest`
  passed 2 suites and 14 tests with no failures, errors, or skips. Production and focused tests
  did not change after this run.
- Preliminary `./gradlew :extensions:nn:javadoc` passed (`BUILD SUCCESSFUL`; 3 actionable tasks,
  1 executed and 2 up-to-date). The later joint clean documentation context still owns final
  Javadoc review and generation after both sequence implementations stabilize.
- `javap -public` confirmed the exact constructor, `cell()`, and
  `forward(Tensor, Tensor, long[])` surface plus the ordinary two-component record surface.
  `javap -private` confirmed the sequence retains only one `GruCell cell` field; all packing,
  validation, and index routines are private static implementation details.
- Production import inspection found only Java, Model, and existing NN types. `git diff --check`
  passed for the combined worktree. The implementation touched only its four new Java paths and
  this task record; concurrent CPU, shared documentation/master/roadmap, task 0017, and LSTM work
  were left untouched.
- Joint clean documentation context `/root/nn_0016_0017_docs` independently reviewed the final
  GRU and LSTM implementations, focused tests, shared module/state and Model composition
  contracts, generated Javadocs, Training API, glossary, and planning diff. It found no
  executable, public-API, architecture, dependency, or scope defect and changed no executable
  Java or tests.
- The documentation context reused the stable GRU focused 2-suite/14-test result and LSTM focused
  2-suite/15-test result. It ran the one coordinated authoritative
  `./gradlew :extensions:nn:test` over the combined diff: 29 suites and 197 tests passed with no
  failures, errors, or skips. No executable Java or test changed afterward.
- The same context finalized the four new type/member Javadocs by review, updated the shared
  layers package Javadoc, Training API, glossary, both task records, and NN master plan, and ran
  final `./gradlew :extensions:nn:javadoc` successfully. Generated pages for both sequences,
  both results, and the layers package contain the exact static-packing, state, ownership,
  length-snapshot, zero-value, all-zero, failure, and non-execution boundaries.
- Final `javap -public` and `javap -private` confirmed the exact two GRU types, direct `Module`
  inheritance, constructor/accessor/forward surface, ordinary record surface, and sole retained
  `GruCell cell` field. The authoritative suite's reflection tests independently lock the same
  surface. A manual JShell reflection inspection produced the expected superclass, field, method,
  and record-component results; JShell then reported only a sandbox preferences-flush warning at
  shutdown.
- Production imports remain limited to Java, Model, and existing NN types, and
  `extensions/nn/build.gradle.kts` still contains only the Model implementation dependency. The
  five-file Markdown validator passed 320 local links, 250 anchors, balanced fences, and final
  newlines. Exact-scope inspection found the intended fourteen NN paths; concurrent CPU, CPU
  planning, glossary CPU hunks, and global-roadmap work were preserved.
- `ARCHITECTURE.md`, focused architecture explanations, ADR 0007, and architecture tests require
  no change because this is concrete Model-only NN composition on the existing dependency edge.
  Tensor and Compile APIs, Training Java source and training graph, Model operations/capabilities,
  Gradle/dependencies, compiler/runtime/prepare/engine, backend conformance/integration, other
  modules, CPU work, and the global roadmap require no change because no semantic operation,
  executable lifecycle, module boundary, or cross-module capability changed.
- Final status, later-specification absence, exact-scope, new-file whitespace/newline, and
  whole-worktree `git diff --check` gates passed. Tasks 0016 and 0017 and their master rows are
  Complete; no Ready row or later NN task specification was introduced.

## Implementation notes

- Added `GruSequence` with one exact owned `GruCell`, defensive length snapshotting, full
  non-constructing current-cell/input/state/length/type/Shape preflight, stable original-order
  active-row gathers, relative survivor gathers, exact cell-result exposure, exit-position
  capture, and original-order final-hidden stacking.
- Added `GruSequenceForwardResult` with ordered null checks, a structural immutable list snapshot,
  and exact Tensor-reference retention.
- Added focused surface, result ownership, module ownership/state/mode, replacement,
  caller-failure/no-ID, current-schema, zero-step, stable packing, index metadata/provenance,
  zero-value exclusion, exit restoration, and exact 26/27-ID-order tests.
- Drafted complete type, constructor, accessor, forward, result-component, nullability, ownership,
  failure, side-effect, concurrency, mode, and non-execution Javadocs in the two production files
  for independent finalization by the joint documentation context.

## Completion summary

- Completed changes: the GRU-specific static packed sequence, immutable result, focused tests,
  finalized public/package documentation, Training API, glossary, and synchronized planning
  records are complete.
- Files changed or created: the four GRU Java paths and this task record, plus the four shared
  documentation/planning paths within the exact fourteen-path joint change.
- Tests and validation: focused GRU 14/14 and coordinated NN 197/197 passed; final Javadoc,
  generated-page, bytecode/reflection, imports/dependency, Markdown, exact-scope/status,
  newline/whitespace, and diff gates passed.
- Documentation-agent review: joint clean context `/root/nn_0016_0017_docs` completed the
  independent review without changing executable Java or tests.
- Documentation impact: shared package/API/glossary text now describes all three concrete static
  sequence containers and preserves the runtime-scan boundary.
- Javadoc review: the implementation drafts were accurate and required no production-source edit;
  shared package Javadoc was finalized and generated output passed inspection.
- Glossary impact: existing construction-time length, active batch, static packed recurrent
  sequence, and recurrent scan definitions now cover RNN, GRU, and LSTM without zero inference.
- Unresolved issues: None.
- Follow-up required: None for NN 0016.

Status: Complete
