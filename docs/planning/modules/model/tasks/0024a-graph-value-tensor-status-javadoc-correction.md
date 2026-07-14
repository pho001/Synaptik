# Task 0024A: GraphValue Tensor-Status Javadoc Correction

## Status

Complete

## Goal

Resolve the sole blocking finding from the completed model capability-and-contract closure audit.
Correct the stale `GraphValue` type Javadoc that calls the already-current public mutable
`Tensor` API planned, prove that the edit changes documentation only, record closure in the audit
artifact, and close the selected `modules/model` capability milestone.

## Scope

- Change only the stale status wording in the `GraphValue` type Javadoc from planned to current.
- Preserve the complete declaration, constructor, accessors, validation, messages, imports,
  bytecode-visible API, record value semantics, graph ownership, and all surrounding Javadoc
  contracts.
- Review generated Javadoc and confirm that `GraphValue` distinguishes the current public mutable
  Tensor expression object from immutable graph-local values without implying graph capture,
  storage, runtime residency, or compiler behavior.
- Add a closure addendum to the task-0024 audit artifact. Preserve its historical
  `BLOCKING_GAP` verdict as the result at audit time, then record that 0024A resolved its sole
  blocker and that the model milestone is now closed.
- Synchronize model capabilities, master plan, and roadmap to mark 0024A and the selected model
  milestone Complete.
- Run focused documentation and scope validation after the wording and planning status are stable.

## Out of scope

- any Java declaration, constructor, method, field, import, validation, behavior, or test change
- rewriting unrelated `GraphValue`, graph-model, Tensor, producer/provenance, or compiled-graph
  Javadocs
- a new API, operation, capability, architecture rule, ADR, dependency, Gradle change, or module
- compiler capture, graph construction, autograd, prepare, runtime, backend, or execution work
- changing task 0024's completed historical verdict or erasing its evidence
- creating a detailed task 0024B, task 0025, or next-project-area specification
- unrelated documentation cleanup

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Documentation profile index](../../../../developer-guide/documentation/README.md)
- [General style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Model capabilities](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0024](0024-model-capability-and-contract-closure-audit.md)
- [Task-0024 audit artifact](../model-capability-contract-closure-audit.md)

## Architecture constraints

- `ARCHITECTURE.md` remains authoritative and unchanged.
- `Tensor` remains current public mutable API state and is not graph IR.
- `GraphValue` remains immutable graph-local logical value metadata and does not gain a Tensor
  reference, producer/consumer links, storage, compiler, or runtime state.
- This task corrects documentation to match already-completed architecture and source; it selects
  no new behavior or ownership rule.

## Package impact

No package or dependency changes. The only Java path receives one Javadoc-only wording
correction; its compiled contract must remain unchanged.

## Affected files

Exactly six paths:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/graph/GraphValue.java`
- add and finalize this task
- `docs/planning/modules/model/model-capability-contract-closure-audit.md`
- `docs/planning/modules/model/capabilities.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification: `GraphValueTest`, `Tensor`, graph-model source/tests, Tensor/Compile/
Training/Runtime/Public APIs, glossary, architecture documents/tests, Gradle, other modules,
backend conformance, and integration tests.

## Maximum scope

At most the exact six paths above. Stop if accurate closure requires any test, additional Java,
API/glossary, architecture, build, dependency, or cross-module change.

## Acceptance criteria

- The stale phrase no longer calls the public mutable Tensor API planned and instead describes it
  as current.
- The edited paragraph remains meaningful and preserves the distinction among Tensor,
  `GraphValue`, `CompiledNode`, storage, prepared memory, device buffers, and runtime residency.
- A source diff proves that `GraphValue.java` changed only in Javadoc text.
- Generated model Javadoc contains the corrected current-status wording and no stale phrase.
- The task-0024 artifact preserves its original `BLOCKING_GAP` verdict and adds an explicit dated
  closure addendum naming 0024A, the correction, validation, and final closed milestone state.
- Capabilities, master plan, and roadmap agree that task 0024A is Complete and the selected model
  capability milestone is closed.
- No next detailed task specification is created.
- Exactly the six authorized paths change; no Java behavior, tests, Gradle, architecture, another
  module, conformance, or integration path changes.
- Markdown, links, anchors, fences, final newlines, trailing whitespace, and `git diff --check`
  pass.

## Tests / validation

This is a Javadoc-only Java edit after task 0024's successful 1,017-test repository checkpoint.
Do not repeat successful Java test suites. Run one model Javadoc validation after the source and
documentation are stable:

```bash
./gradlew :modules:model:javadoc
```

Then inspect the generated `GraphValue` Javadoc and run:

```bash
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

Also compare the `GraphValue.java` diff directly to prove that no declaration, import, executable
statement, or surrounding contract changed. Confirm exactly six authorized paths, synchronized
Complete status, preserved task-0024 history, and no detailed later task file.

## Dependencies

- Completed task 0024 and its `BLOCKING_GAP` audit artifact.
- The already-current public `Tensor` contract and completed immutable graph model.

## Follow-up tasks

None inside the model milestone. Successful completion closes the selected model capability
milestone. Selection and planning of the next project area remain separate work without a detailed
specification in this change.

## Architecture impact

None. The correction makes Javadoc agree with the existing authoritative architecture.

## Implementation prompt

Use this prompt in one separate clean documentation-focused task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, documentation/planning rules and profiles, roadmap, model
capabilities/master plan, completed task 0024 and its audit artifact, task 0024A, GraphValue and
its focused test, the current Tensor contract, and affected graph-model Javadocs.

Implement task 0024A exactly inside its six authorized paths. Correct only the stale GraphValue
type-Javadoc status wording, preserve every declaration and behavior, add the historical closure
addendum, and synchronize the model milestone as Complete. Stop on scope or architecture
conflict. Run model Javadoc once after the documentation is stable, inspect the generated output,
and run all specified Markdown/scope/status/whitespace checks. Reuse task 0024's successful Java
test evidence; do not repeat executable suites. Do not create a later detailed specification.
```

## Local decisions

- Use one bounded Javadoc-only follow-up rather than reopen task 0024 or expand its historical
  scope.
- Preserve the audit's original verdict and add a closure addendum, so the record distinguishes
  the audit-time finding from the post-follow-up milestone state.
- Treat model Javadoc generation as the executable compilation/rendering check; repeating the
  successful model and repository tests would add no evidence for a comment-only edit.
- Close the model milestone only after source-diff, generated-Javadoc, Markdown, scope, and status
  checks all pass.

## Known limitations

- This task does not assert compiler, runtime, backend, or numerical execution completeness.
- Closing the selected model milestone does not reject future model capabilities justified by a
  concrete downstream need.

## Validation evidence

- Clean documentation-focused context `/root/execute_0024a` read the repository instructions,
  authoritative architecture, focused architecture index, documentation rules, General,
  API/Javadoc, and Planning profiles, planning guide, roadmap, model capabilities/master plan,
  completed task 0024 and its audit artifact, this task, `GraphValue`, `GraphValueTest`, the
  current `Tensor` contract, and the affected graph-model Javadocs.
- The direct `GraphValue.java` diff contains exactly one Javadoc word change from `planned` to
  `current`. Imports, declaration, record components, constructor, accessors, executable
  statements, validation, messages, and all surrounding Javadoc remain byte-for-byte unchanged.
- `./gradlew :modules:model:javadoc` passed once after the source and documentation were stable.
  The generated `GraphValue` page contains `current public mutable Tensor API object used to build
  expressions`, contains no stale `planned public mutable` phrase, and retains the graph-local,
  storage-free, compiler-free, and runtime-free boundaries.
- `python3 /tmp/validate_synaptik_markdown.py` passed local Markdown links, anchors, fences, final
  newlines, and trailing whitespace. `git diff --check` passed.
- Task-specific scope inspection confirmed exactly the six authorized paths changed relative to
  the shared starting tree: `GraphValue.java`, this task, the task-0024 audit artifact, model
  capabilities, model master plan, and roadmap. The other three dirty paths belong to the
  preserved completed task-0024 change that preceded this task.
- Status scans confirm task 0024A, the model master-plan row, the roadmap row, and the selected
  model capability milestone are Complete. The audit's original `BLOCKING_GAP` verdict and open
  decision remain intact as historical text, followed by the dated closure addendum. No detailed
  0024B, 0025, or next-project-area task file was created.
- Task 0024's successful repository checkpoint evidence was reused: 1,017 tests across 128 suites
  with zero failures, errors, or skips. No executable Java changed after that result, so no Java
  test suite was repeated.
- API references, glossary, architecture documentation/tests, Gradle, other modules, backend
  conformance, and integration tests required no change because this task introduces no term,
  behavior, ownership rule, dependency, or executable contract.

## Implementation notes

Changed only the stale status adjective in the `GraphValue` type Javadoc, appended a dated closure
record without rewriting task 0024's historical verdict, and synchronized the capability,
master-plan, and roadmap records. The shared task-0024 changes were preserved. No declaration,
behavior, test, API, glossary, architecture, build, dependency, or cross-module path changed.

## Completion summary

- Completed changes: corrected the stale `GraphValue` Tensor-status Javadoc wording, added the
  historical closure addendum, and closed the selected model capability milestone.
- Files changed or created: exactly six task-authorized paths — `GraphValue.java`, this task, the
  task-0024 audit artifact, model capabilities, model master plan, and roadmap.
- Tests and validation: model Javadoc passed once and its generated `GraphValue` page was
  inspected; Markdown links/anchors/fences/newlines/whitespace, exact task scope, synchronized
  status, preserved audit history, absence of a later detailed task, and `git diff --check`
  passed. Task 0024's successful 1,017-test checkpoint was reused; executable suites were not
  repeated.
- Documentation-agent review: clean context `/root/execute_0024a` executed and finalized this
  documentation-only task.
- Documentation impact: the Javadoc and five planning records now agree that Tensor is current
  and the selected model milestone is closed; no API, glossary, architecture, or other
  documentation change is needed.
- Javadoc review: `GraphValue` continues to distinguish public mutable Tensor state from immutable
  graph-local logical values and makes no capture, storage, compiler, backend, or runtime claim.
- Glossary impact: no change; the task changes status wording only and introduces no reusable
  domain term or altered definition.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
