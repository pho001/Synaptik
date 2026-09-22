# Task 0025M: Tensor guide and API status reconciliation

## Status

Complete

## Change class

Class B — documentation-only reconciliation of a current public Model workflow and cross-layer API
status. No Java API, behavior, lifecycle, ownership, or architecture rule changes.

## Goal

Replace the pre-`Tensor` user guide with a runnable current tensor-construction and expression
workflow, and correct the stale Tensor API sentence that still labels current Engine and CPU
execution capabilities as planned.

## Scope

- Rewrite `docs/user-guide/tensors.md` around one current public task: create a dense Tensor with
  host storage, build one expression, and inspect the resulting metadata and provenance.
- Include complete inputs, a runnable Java 26 example, expected observations, common failures,
  limitations, and focused related links.
- Correct only the stale cross-layer status paragraph near the opening of `docs/api/tensor-api.md`.
  Distinguish current raw host-storage access, Engine compile/prepare/run/materialization, and
  bounded CPU execution from absent typed Tensor element access, device-residency ownership, and
  unsupported backend or operation combinations.
- Complete a clean documentation implementation and an independent targeted review, then record
  compact result evidence and synchronize planning status.

## Non-goals

- No Java, Javadoc, test, Gradle, API-shape, behavior, numerical, storage, lifecycle, dependency,
  architecture-contract, architecture-explanation, ADR, glossary, or other guide change.
- No exhaustive rewrite of the eleven-thousand-line Tensor API reference.
- No Engine tutorial, execution example, new Tensor convenience, typed element-access API, or
  claim that every represented operation is executable.
- No change to Model 0026 or creation of another detailed task brief.

## Contracts

- [`ARCHITECTURE.md` heading `Core lifecycle`](../../../../../ARCHITECTURE.md#core-lifecycle) —
  current public compile/prepare/run ownership and closeable handles.
- [`ARCHITECTURE.md` heading `Core invariants`](../../../../../ARCHITECTURE.md#core-invariants) —
  Tensor is public Model state, not graph IR or runtime state.
- [`foundational-modules.md` heading `modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel)
  — Model owns Tensor, TensorFactory, metadata, expression meaning, and host storage.
- [`runtime-prepare-engine.md` heading `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — Engine owns current public CPU-only lifecycle composition and result access.
- [`backend-execution.md` heading `CPU backend routes`](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)
  — CPU route support remains exact and fail-closed rather than universal.

If an applicable contract is missing or ambiguous, stop and report it.

## Files and symbols

Exact implementation allowlist:

- `docs/user-guide/tensors.md` — complete current task-oriented rewrite.
- `docs/api/tensor-api.md` — only the stale opening status paragraph.
- `docs/planning/modules/model/tasks/0025m-tensor-guide-and-api-status-reconciliation.md` — status
  and compact result evidence.
- `docs/planning/modules/model/master-plan.md` — 0025M row and frontier summary only.
- `docs/planning/roadmap.md` — Model row and current-frontier summary only.

Read-only evidence includes `Tensor`, `TensorFactory`, `TensorProducer`, `TensorProvenance`,
`HostTensorStorage`, their focused tests and Javadocs, the focused current Engine API, and the
targeted Tensor/host-storage/Engine terms in `docs/glossary.md`.

No path outside the five-file implementation allowlist may change.

## Acceptance criteria

- The user guide no longer says public Tensor state, host storage, factories, provenance, or
  expression construction are planned or unavailable.
- Its example uses only current public Java 26 API, creates independent dense host-backed input,
  builds one storage-free expression without evaluation, and reports exact observations that can
  be compiled and run against the repository classes.
- The guide accurately distinguishes immutable Tensor identity/descriptor/provenance from the
  synchronized borrowed host-storage association, caller-owned raw memory, graph capture, and
  Engine execution.
- Common errors cover at least unresolved layout/allocation or incompatible expression inputs;
  limitations do not imply typed Tensor element access or universal backend execution.
- The Tensor API opening no longer calls the public compiler facade, publication delivery, and
  backend execution wholly planned. It names the current Engine/CPU boundary without claiming
  general backend coverage or moving lifecycle state into Tensor.
- Current Java/Javadoc already agrees and remains unchanged. Targeted glossary review finds no new
  or changed reusable term, or the task stops because the five-file allowlist is insufficient.
- A clean documentation implementation and a separate independent targeted review both complete.
  The review checks the final diff against source/contracts and records reasoned no-change
  conclusions for Javadoc, glossary, architecture, tests, and other guides.
- Only the five allowed paths change; task/master/roadmap status is synchronized; this brief stays
  at most 200 lines and 15 KB; links/anchors, unique headings, fences, final newlines, whitespace,
  and `git diff --check` pass.

## Validation

After the documentation is stable, copy the exact guide example to
`/tmp/TensorGuideExample.java`, then run:

```bash
./gradlew :modules:model:classes
javac -cp modules/model/build/classes/java/main \
  -d /tmp/synaptik-tensor-guide /tmp/TensorGuideExample.java
java -cp modules/model/build/classes/java/main:/tmp/synaptik-tensor-guide TensorGuideExample
python3 /tmp/validate_synaptik_markdown.py \
  docs/user-guide/tensors.md \
  docs/api/tensor-api.md \
  docs/planning/modules/model/tasks/0025m-tensor-guide-and-api-status-reconciliation.md \
  docs/planning/modules/model/master-plan.md \
  docs/planning/roadmap.md
wc -l -c docs/planning/modules/model/tasks/0025m-tensor-guide-and-api-status-reconciliation.md
git diff --name-only -- '*.java' '*.gradle' '*.gradle.kts'
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git diff --check
```

The Java/Gradle scan must be empty and the complete path audit must equal the five-file allowlist.
If the temporary Markdown validator is absent, create an equivalent validator outside the
repository for local targets/anchors, unique headings, balanced fences, final newlines, and
trailing whitespace. The independent review reruns final documentation, size, scope, example,
and diff checks, but does not add or repeat a Java test suite without executable changes.

Repository-wide validation: deferred because this task changes explanatory/status documentation
and planning only, with no executable, dependency, build, or architecture-contract change.

## Dependencies and follow-up

- Frontier verification: the user explicitly authorized planning the remaining confirmed
  explanatory/status drift from clean HEAD `a7be5679`. Model source, focused Javadocs, and the
  current Engine/public API already supply the needed authority; completed Model work through
  0025L and Engine work through 0012 satisfy the documentation prerequisites.
- 0025M was inserted before unselected Draft task 0026 as the sole authorized `Ready` frontier.
  Its clean implementation and independent targeted review are now complete. Engine, tuning,
  Prepare/backend, Runtime-boundary, and CPU-guide drift remain later separately owned candidates
  without detailed briefs.

## Documentation and review impact

- Primary profiles: General + User guide + Example format for `tensors.md`; General + API and
  Javadoc for `tensor-api.md`; Planning style for planning files.
- Javadoc impact: none expected because Java contracts are unchanged and already current.
- Glossary impact: none expected because existing terms and meanings are reused; verify by
  targeted search.
- A clean documentation implementation context and separate independent review context are
  mandatory because this Class B task changes a public user workflow and API-status explanation.

## Result

The clean documentation implementation rewrote `docs/user-guide/tensors.md` as a runnable current
flat-import and unary-expression workflow, corrected only the stale opening cross-layer paragraph
in `docs/api/tensor-api.md`, and synchronized this task, the Model plan, and the roadmap. Targeted
source, Javadoc, focused-test, Engine, contract, and glossary review found no Java/Javadoc,
glossary, architecture, test, or other-guide change: the implementation reuses current terms and
documents existing behavior only.

- `./gradlew :modules:model:classes`, exact guide-block comparison, `javac`, and `java` passed;
  the thirteen output lines exactly matched the documented observations.
- The independent targeted review checked the final diff against the named Model, Engine, and CPU
  contracts; current Tensor, storage, factory, producer/provenance, and Engine source/Javadocs;
  focused tests; and targeted glossary terms. It corrected the guide's backing-array inference
  and association-only synchronization/lifetime/threading wording, then separated compile-time
  publication requests, run-time input binding, and open-result materialization in the API status
  paragraph.
- The review reused the successful compile/run evidence because its corrections did not change
  the exact Java or output blocks. A final comparison confirmed that the guide block still matched
  `/tmp/TensorGuideExample.java` byte for byte.
- The five-file Markdown validator passed links, generated anchors, fences, final newlines, and
  whitespace; targeted opening/guide searches found none of the replaced stale claims.
- The brief remained below 200 lines and 15 KB. The Java/Gradle scan was empty, the complete path
  audit equaled the exact five-file allowlist, and `git diff --check` passed.
- Javadoc, glossary, architecture, tests, and other guides need no change: the final documentation
  describes current contracts and public behavior without changing Java, terminology, authority,
  backend capability, or workflow beyond the two selected pages.

Status: Complete
