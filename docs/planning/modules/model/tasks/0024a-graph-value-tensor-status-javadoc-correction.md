# Task 0024A: GraphValue [Tensor](../../../../glossary.md#tensor)-Status [Javadoc](../../../../glossary.md#term-javadoc) Correction

## Status

Complete

## Goal

Resolve the sole blocking finding from the completed [model](../../../../glossary.md#term-model) capability-and-contract closure audit.
Correct the stale `GraphValue` type [Javadoc](../../../../glossary.md#term-javadoc) that calls the already-current public [mutable](../../../../glossary.md#term-mutable)
[`Tensor`](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) planned, prove that the edit changes documentation only, record closure in the audit
[artifact](../../../../glossary.md#term-artifact), and close the selected `modules/model` capability milestone.

## [Scope](../../../../glossary.md#term-scope)

- Change only the stale status wording in the `GraphValue` type [Javadoc](../../../../glossary.md#term-javadoc) from planned to current.
- Preserve the complete declaration, constructor, accessors, validation, messages, imports,
  bytecode-visible API, record value semantics, graph ownership, and all surrounding Javadoc
  contracts.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [graph glossary](../../../../glossary.md#graph); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Review generated Javadoc and confirm that `GraphValue` distinguishes the current public mutable
  Tensor expression object from immutable graph-local values without implying graph capture,
  storage, runtime residency, or compiler behavior.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [Tensor glossary](../../../../glossary.md#tensor); [immutable glossary](../../../../glossary.md#term-immutable); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Add a closure addendum to the task-0024 audit [artifact](../../../../glossary.md#term-artifact). Preserve its historical
  `BLOCKING_GAP` verdict as the result at audit time, then record that 0024A resolved its sole
  blocker and that the model milestone is now closed.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Synchronize model capabilities, master plan, and roadmap to mark 0024A and the selected model
  milestone Complete.
- [Run](../../../../glossary.md#term-run) focused documentation and scope [validation](../../../../glossary.md#term-validation) after the wording and [planning](../../../../glossary.md#term-planning) status are stable.

## Out of [scope](../../../../glossary.md#term-scope)

- any Java declaration, constructor, method, field, import, [validation](../../../../glossary.md#term-validation), behavior, or test change
- rewriting unrelated `GraphValue`, graph-model, Tensor, producer/provenance, or compiled-graph
  Javadocs<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- a new [API](../../../../glossary.md#term-application-programming-interface-api), [operation](../../../../glossary.md#operation), capability, architecture rule, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), [dependency](../../../../glossary.md#term-dependency), [Gradle](../../../../glossary.md#term-gradle) change, or [module](../../../../glossary.md#term-module)
- [compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings), [graph](../../../../glossary.md#graph) construction, [autograd](../../../../glossary.md#autograd), [prepare](../../../../glossary.md#prepare), [runtime](../../../../glossary.md#term-runtime), [backend](../../../../glossary.md#backend), or execution work
- changing task 0024's completed historical verdict or erasing its evidence
- creating a detailed task 0024B, task 0025, or next-project-area specification
- unrelated documentation cleanup

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Documentation profile index](../../../../developer-guide/documentation/README.md)
- [General style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Planning style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Model capabilities](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Task 0024](0024-model-capability-and-contract-closure-audit.md)
- [Task-0024 audit artifact](../model-capability-contract-closure-audit.md)<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>

## Architecture constraints

- `ARCHITECTURE.md` remains authoritative and unchanged.
- [`Tensor`](../../../../glossary.md#tensor) remains current public [mutable](../../../../glossary.md#term-mutable) [API](../../../../glossary.md#term-application-programming-interface-api) state and is not [graph](../../../../glossary.md#graph) [IR](../../../../glossary.md#term-intermediate-representation-ir).
- `GraphValue` remains immutable graph-local logical value metadata and does not gain a Tensor
  reference, producer/consumer links, storage, compiler, or runtime state.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [metadata glossary](../../../../glossary.md#term-metadata); [producer glossary](../../../../glossary.md#term-producer); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- This task corrects documentation to match already-completed architecture and source; it selects
  no new behavior or ownership rule.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Package impact

No package or [dependency](../../../../glossary.md#term-dependency) changes. The only Java path receives one [Javadoc](../../../../glossary.md#term-javadoc)-only wording
correction; its compiled contract must remain unchanged.

## Affected files

Exactly six paths:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/graph/GraphValue.java`
- add and finalize this task
- `docs/planning/modules/model/model-capability-contract-closure-audit.md`
- `docs/planning/modules/model/capabilities.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification: `GraphValueTest`, [`Tensor`](../../../../glossary.md#tensor), [graph](../../../../glossary.md#graph)-[model](../../../../glossary.md#term-model) source/tests, Tensor/[Compile](../../../../glossary.md#compile)/
Training/[Runtime](../../../../glossary.md#term-runtime)/Public APIs, glossary, architecture documents/tests, [Gradle](../../../../glossary.md#term-gradle), other [modules](../../../../glossary.md#term-module),
[backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance), and integration tests.

## Maximum [scope](../../../../glossary.md#term-scope)

At most the exact six paths above. Stop if accurate closure requires any test, additional Java,
[API](../../../../glossary.md#term-application-programming-interface-api)/glossary, architecture, build, [dependency](../../../../glossary.md#term-dependency), or cross-[module](../../../../glossary.md#term-module) change.

## Acceptance criteria

- The stale phrase no longer calls the public mutable Tensor API planned and instead describes it
  as current.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- The edited paragraph remains meaningful and preserves the distinction among [Tensor](../../../../glossary.md#tensor),
  `GraphValue`, `CompiledNode`, storage, prepared memory, [device](../../../../glossary.md#term-device) [buffers](../../../../glossary.md#term-buffer), and [runtime](../../../../glossary.md#term-runtime) [residency](../../../../glossary.md#residency).
- A source diff proves that `GraphValue.java` changed only in [Javadoc](../../../../glossary.md#term-javadoc) text.
- Generated [model](../../../../glossary.md#term-model) Javadoc contains the corrected current-status wording and no stale phrase.
- The task-0024 [artifact](../../../../glossary.md#term-artifact) preserves its original `BLOCKING_GAP` verdict and adds an explicit dated
  closure addendum naming 0024A, the correction, validation, and final closed milestone state.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Capabilities, master plan, and roadmap agree that task 0024A is Complete and the selected model
  capability milestone is closed.
- No next detailed task specification is created.
- Exactly the six authorized paths change; no Java behavior, tests, Gradle, architecture, another
  module, conformance, or integration path changes.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Markdown, links, anchors, fences, final newlines, trailing whitespace, and `git diff --check`
  pass.

## Tests / [validation](../../../../glossary.md#term-validation)

This is a [Javadoc](../../../../glossary.md#term-javadoc)-only Java edit after task 0024's successful 1,017-test repository [checkpoint](../../../../glossary.md#term-checkpoint).
Do not repeat successful Java test suites. [Run](../../../../glossary.md#term-run) one [model](../../../../glossary.md#term-model) Javadoc validation after the source and
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

- Completed task 0024 and its `BLOCKING_GAP` audit [artifact](../../../../glossary.md#term-artifact).
- The already-current public [`Tensor`](../../../../glossary.md#tensor) contract and completed [immutable](../../../../glossary.md#term-immutable) [graph](../../../../glossary.md#graph) [model](../../../../glossary.md#term-model).

## Follow-up tasks

None inside the [model](../../../../glossary.md#term-model) milestone. Successful completion closes the selected model capability
milestone. Selection and [planning](../../../../glossary.md#term-planning) of the next project area remain separate work without a detailed
specification in this change.

## Architecture impact

None. The correction makes [Javadoc](../../../../glossary.md#term-javadoc) agree with the existing authoritative architecture.

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
  scope.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- Preserve the audit's original verdict and add a closure addendum, so the record distinguishes
  the audit-time finding from the post-follow-up milestone state.
- Treat model Javadoc generation as the executable compilation/rendering check; repeating the
  successful model and repository tests would add no evidence for a comment-only edit.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- Close the model milestone only after source-diff, generated-Javadoc, Markdown, scope, and status
  checks all pass.

## Known limitations

- This task does not assert [compiler](../../../../glossary.md#term-compiler), [runtime](../../../../glossary.md#term-runtime), [backend](../../../../glossary.md#backend), or numerical execution completeness.
- Closing the selected model milestone does not reject future model capabilities justified by a
  concrete downstream need.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Clean documentation-focused context `/root/execute_0024a` read the repository instructions,
  authoritative architecture, focused architecture index, documentation rules, General,
  API/Javadoc, and Planning profiles, planning guide, roadmap, model capabilities/master plan,
  completed task 0024 and its audit artifact, this task, <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Planning glossary](../../../../glossary.md#term-planning); [artifact glossary](../../../../glossary.md#term-artifact))</span>`GraphValue`, `GraphValueTest`, the
  current [`Tensor`](../../../../glossary.md#tensor) contract, and the affected [graph](../../../../glossary.md#graph)-[model](../../../../glossary.md#term-model) Javadocs.
- The direct `GraphValue.java` diff contains exactly one [Javadoc](../../../../glossary.md#term-javadoc) word change from `planned` to
  `current`. Imports, declaration, record components, constructor, accessors, executable
  statements, validation, messages, and all surrounding Javadoc remain byte-for-byte unchanged.
- `./gradlew :modules:model:javadoc` passed once after the source and documentation were stable.
  The generated `GraphValue` page contains `current public mutable Tensor API object used to build
  expressions`, contains no stale `planned public mutable` phrase, and retains the graph-local,
  storage-free, compiler-free, and runtime-free boundaries.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- `python3 /tmp/validate_synaptik_markdown.py` passed local Markdown links, anchors, fences, final
  newlines, and trailing whitespace. `git diff --check` passed.
- Task-specific scope inspection confirmed exactly the six authorized paths changed relative to
  the shared starting tree: <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`GraphValue.java`, this task, the task-0024 audit artifact, model
  capabilities, model master plan, and roadmap. The other three dirty paths belong to the
  preserved completed task-0024 change that preceded this task.
- Status scans confirm task 0024A, the model master-plan row, the roadmap row, and the selected
  model capability milestone are Complete. The audit's original `BLOCKING_GAP` verdict and open
  decision remain intact as historical text, followed by the dated closure addendum. No detailed
  0024B, 0025, or next-project-area task file was created.
- Task 0024's successful repository checkpoint evidence was reused: 1,017 tests across 128 suites
  with zero failures, errors, or skips. No executable Java changed after that result, so no Java
  test suite was repeated.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- API references, glossary, architecture documentation/tests, Gradle, other modules, backend
  conformance, and integration tests required no change because this task introduces no term,
  behavior, ownership rule, dependency, or executable contract.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Implementation notes

Changed only the stale status adjective in the `GraphValue` type [Javadoc](../../../../glossary.md#term-javadoc), appended a dated closure
record without rewriting task 0024's historical verdict, and synchronized the capability,
master-plan, and roadmap records. The shared task-0024 changes were preserved. No declaration,
behavior, test, [API](../../../../glossary.md#term-application-programming-interface-api), glossary, architecture, build, [dependency](../../../../glossary.md#term-dependency), or cross-[module path](../../../../glossary.md#term-module-path) changed.

## Completion summary

- Completed changes: corrected the stale `GraphValue` Tensor-status Javadoc wording, added the
  historical closure addendum, and closed the selected model capability milestone.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Files changed or created: exactly six task-authorized paths — `GraphValue.java`, this task, the
  task-0024 audit artifact, model capabilities, model master plan, and roadmap.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
- Tests and [validation](../../../../glossary.md#term-validation): [model](../../../../glossary.md#term-model) [Javadoc](../../../../glossary.md#term-javadoc) passed once and its generated `GraphValue` page was
  inspected; Markdown links/anchors/fences/newlines/whitespace, exact task scope, synchronized
  status, preserved audit history, absence of a later detailed task, and <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`
  passed. Task 0024's successful 1,017-test checkpoint was reused; executable suites were not
  repeated.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Documentation-agent review: clean context `/root/execute_0024a` executed and finalized this
  documentation-only task.
- Documentation impact: the Javadoc and five planning records now agree that Tensor is current
  and the selected model milestone is closed; no API, glossary, architecture, or other
  documentation change is needed.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Javadoc review: `GraphValue` continues to distinguish public mutable Tensor state from immutable
  graph-local logical values and makes no capture, storage, compiler, backend, or runtime claim.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [immutable glossary](../../../../glossary.md#term-immutable); [graph glossary](../../../../glossary.md#graph); [capture glossary](../../../../glossary.md#term-capture-meanings); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Glossary impact: no change; the task changes status wording only and introduces no reusable
  domain term or altered definition.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
