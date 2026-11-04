# Task 0025M: [Tensor](../../../../glossary.md#tensor) guide and [API](../../../../glossary.md#term-application-programming-interface-api) status reconciliation

**Editorial reading key (current context, not a status update).**

This completed record concerns a narrow documentation reconciliation, not an expansion of execution
support. In current terminology, the caller supplies a descriptor (type, shape and layout) and an
array to TensorFactory’s flat import; values are copied into independent host storage. By contrast,
creation with existing storage retains a borrowed association, not a copy. A leaf has no producing
expression, whether or not it has storage. The [creation
contracts](../../../../reference/tensor/creation.md) explain these distinct description, allocation
and value-import boundaries.

For the recorded unary workflow, negation constructs a storage-free result description: its producer
records the operation and exact input, while provenance selects output slot zero. It neither
evaluates negation nor owns a result buffer. Follow [expression producers and indexed
provenance](../../../../reference/tensor/identity.md#expression-producers-and-indexed-provenance) to
interpret those identities. Model defines operation meaning; Engine composes compile/prepare/run and
result access through supported execution routes, not new semantics. Semantic responsibility does
not transfer ownership of borrowed storage; synchronized association access does not guarantee
raw-memory lifetime or thread safety.

## Status

Complete

## Change class

Class B — documentation-only reconciliation of a current public [Model](../../../../glossary.md#term-model) workflow and cross-layer [API](../../../../glossary.md#term-application-programming-interface-api)
status. No Java API, behavior, [lifecycle](../../../../glossary.md#lifecycle), [ownership](../../../../glossary.md#term-ownership-meanings), or architecture rule changes.

## Goal

Replace the pre-[`Tensor`](../../../../glossary.md#tensor) user guide with a runnable current tensor-construction and expression
workflow, and correct the stale Tensor [API](../../../../glossary.md#term-application-programming-interface-api) sentence that still labels current [Engine](../../../../glossary.md#term-engine) and [CPU](../../../../glossary.md#term-cpu)
execution capabilities as planned.

## [Scope](../../../../glossary.md#term-scope)

- Rewrite `docs/user-guide/tensors.md` around one current public task: create a dense Tensor with
  host storage, build one expression, and inspect the resulting metadata and provenance.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [host storage glossary](../../../../glossary.md#host-storage); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- Include complete inputs, a runnable Java 26 example, expected observations, common failures,
  limitations, and focused related links.
- Correct only the stale cross-layer status paragraph near the opening of `docs/api/tensor-api.md`.
  Distinguish current raw host-storage access, Engine compile/prepare/run/materialization, and
  bounded CPU execution from absent typed Tensor element access, device-residency ownership, and
  unsupported backend or operation combinations.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [Engine glossary](../../../../glossary.md#term-engine); [compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [materialization glossary](../../../../glossary.md#materialization); [CPU glossary](../../../../glossary.md#term-cpu); [element glossary](../../../../glossary.md#term-element); [device glossary](../../../../glossary.md#term-device); [residency glossary](../../../../glossary.md#residency); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [backend glossary](../../../../glossary.md#backend); [operation glossary](../../../../glossary.md#operation))</span>
- Complete a clean documentation implementation and an independent targeted review, then record
  compact result evidence and synchronize planning status.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Non-goals

- No Java, Javadoc, test, Gradle, API-shape, behavior, numerical, storage, lifecycle, dependency,
  architecture-contract, architecture-explanation, ADR, glossary, or other guide change.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Gradle glossary](../../../../glossary.md#term-gradle); [shape glossary](../../../../glossary.md#shape); [lifecycle glossary](../../../../glossary.md#lifecycle); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- No exhaustive rewrite of the eleven-thousand-line [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) reference.
- No Engine tutorial, execution example, new Tensor convenience, typed element-access API, or
  claim that every represented operation is executable.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [element glossary](../../../../glossary.md#term-element); [operation glossary](../../../../glossary.md#operation))</span>
- No change to [Model](../../../../glossary.md#term-model) 0026 or creation of another detailed task brief.

## Contracts

- [`ARCHITECTURE.md` heading `Core lifecycle`](../../../../../ARCHITECTURE.md#core-lifecycle) —
  current public compile/prepare/run ownership and closeable handles.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- [`ARCHITECTURE.md` heading `Core invariants`](../../../../../ARCHITECTURE.md#core-invariants) —
  Tensor is public Model state, not graph IR or runtime state.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Model glossary](../../../../glossary.md#term-model); [graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- [`foundational-modules.md` heading `modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel)
  — Model owns Tensor, TensorFactory, metadata, expression meaning, and host storage.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [host storage glossary](../../../../glossary.md#host-storage))</span>
- [`runtime-prepare-engine.md` heading `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — Engine owns current public CPU-only lifecycle composition and result access.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [`backend-execution.md` heading `CPU backend routes`](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)
  — CPU route support remains exact and fail-closed rather than universal.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>

If an applicable contract is missing or ambiguous, stop and report it.

## Files and symbols

Exact implementation allowlist:

- `docs/user-guide/tensors.md` — complete current task-oriented rewrite.
- `docs/api/tensor-api.md` — only the stale opening status paragraph.
- `docs/planning/modules/model/tasks/0025m-tensor-guide-and-api-status-reconciliation.md` — status
  and compact result evidence.
- `docs/planning/modules/model/master-plan.md` — 0025M row and frontier summary only.
- `docs/planning/roadmap.md` — [Model](../../../../glossary.md#term-model) row and current-frontier summary only.

Read-only evidence includes [`Tensor`](../../../../glossary.md#tensor), `TensorFactory`, [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), `TensorProvenance`,
`HostTensorStorage`, their focused tests and Javadocs, the focused current [Engine](../../../../glossary.md#term-engine) [API](../../../../glossary.md#term-application-programming-interface-api), and the
targeted Tensor/[host](../../../../glossary.md#term-host)-storage/Engine terms in `docs/glossary.md`.

No path outside the five-file implementation allowlist may change.

## Acceptance criteria

- The user guide no longer says public Tensor state, host storage, factories, provenance, or
  expression construction are planned or unavailable.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [host storage glossary](../../../../glossary.md#host-storage); [provenance glossary](../../../../glossary.md#provenance))</span>
- Its example uses only current public Java 26 API, creates independent dense host-backed input,
  builds one storage-free expression without evaluation, and reports exact observations that can
  be compiled and run against the repository classes.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [host glossary](../../../../glossary.md#term-host); [evaluation glossary](../../../../glossary.md#term-evaluation); [run glossary](../../../../glossary.md#term-run))</span>
- The guide accurately distinguishes immutable Tensor identity/descriptor/provenance from the
  synchronized borrowed host-storage association, caller-owned raw memory, graph capture, and
  Engine execution.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Common errors cover at least unresolved layout/allocation or incompatible expression inputs;
  limitations do not imply typed Tensor element access or universal backend execution.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [element glossary](../../../../glossary.md#term-element); [backend glossary](../../../../glossary.md#backend))</span>
- The Tensor API opening no longer calls the public compiler facade, publication delivery, and
  backend execution wholly planned. It names the current Engine/CPU boundary without claiming
  general backend coverage or moving lifecycle state into Tensor.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [facade glossary](../../../../glossary.md#term-facade); [publication glossary](../../../../glossary.md#term-publication); [CPU glossary](../../../../glossary.md#term-cpu); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Current Java/Javadoc already agrees and remains unchanged. Targeted glossary review finds no new
  or changed reusable term, or the task stops because the five-file allowlist is insufficient.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- A clean documentation implementation and a separate independent targeted review both complete.
  The review checks the final diff against source/contracts and records reasoned no-change
  conclusions for Javadoc, glossary, architecture, tests, and other guides.
- Only the five allowed paths change; task/master/roadmap status is synchronized; this brief stays
  at most 200 lines and 15 KB; links/anchors, unique headings, fences, final newlines, whitespace,
  and `git diff --check` pass.

## [Validation](../../../../glossary.md#term-validation)

After the documentation is stable, copy the exact guide example to
`/tmp/TensorGuideExample.java`, then [run](../../../../glossary.md#term-run):

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

The Java/[Gradle](../../../../glossary.md#term-gradle) scan must be empty and the complete path audit must equal the five-file allowlist.
If the temporary Markdown validator is absent, create an equivalent validator outside the
repository for local [targets](../../../../glossary.md#term-target)/anchors, unique headings, balanced fences, final newlines, and
trailing whitespace. The independent review reruns final documentation, size, [scope](../../../../glossary.md#term-scope), example,
and diff checks, but does not add or repeat a Java test suite without executable changes.

Repository-wide validation: deferred because this task changes explanatory/status documentation
and [planning](../../../../glossary.md#term-planning) only, with no executable, [dependency](../../../../glossary.md#term-dependency), build, or architecture-contract change.

## Dependencies and follow-up

- Frontier verification: the user explicitly authorized planning the remaining confirmed
  explanatory/status drift from clean HEAD <span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>`a7be5679`. Model source, focused Javadocs, and the
  current Engine/public API already supply the needed authority; completed Model work through
  0025L and Engine work through 0012 satisfy the documentation prerequisites.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Engine glossary](../../../../glossary.md#term-engine); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- 0025M was inserted before unselected Draft task 0026 as the sole authorized [`Ready`](../../../../glossary.md#term-ready) frontier.
  Its clean implementation and independent targeted review are now complete. Engine, tuning,
  Prepare/backend, Runtime-boundary, and CPU-guide drift remain later separately owned candidates
  without detailed briefs.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [Runtime glossary](../../../../glossary.md#term-runtime); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Documentation and review impact

- Primary profiles: General + User guide + Example format for `tensors.md`; General + API and
  Javadoc for <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>`tensor-api.md`; [Planning](../../../../glossary.md#term-planning) style for planning files.
- [Javadoc](../../../../glossary.md#term-javadoc) impact: none expected because Java contracts are unchanged and already current.
- Glossary impact: none expected because existing terms and meanings are reused; verify by
  targeted search.
- A clean documentation implementation context and separate independent review context are
  mandatory because this Class B task changes a public user workflow and API-status explanation.

## Result

The clean documentation implementation rewrote `docs/user-guide/tensors.md` as a runnable current
flat-import and [unary](../../../../glossary.md#term-unary)-expression workflow, corrected only the stale opening cross-layer paragraph
in `docs/api/tensor-api.md`, and synchronized this task, the [Model](../../../../glossary.md#term-model) plan, and the roadmap. Targeted
source, [Javadoc](../../../../glossary.md#term-javadoc), focused-test, [Engine](../../../../glossary.md#term-engine), contract, and glossary review found no Java/Javadoc,
glossary, architecture, test, or other-guide change: the implementation reuses current terms and
documents existing behavior only.

- `./gradlew :modules:model:classes`, exact guide-block comparison, `javac`, and `java` passed;
  the thirteen output lines exactly matched the documented observations.
- The independent targeted review checked the final diff against the named Model, Engine, and CPU
  contracts; current Tensor, storage, factory, producer/provenance, and Engine source/Javadocs;
  focused tests; and targeted glossary terms. It corrected the guide's backing-array inference
  and association-only synchronization/lifetime/threading wording, then separated compile-time
  publication requests, run-time input binding, and open-result materialization in the API status
  paragraph.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Tensor glossary](../../../../glossary.md#tensor); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [inference glossary](../../../../glossary.md#term-inference); [synchronization glossary](../../../../glossary.md#term-synchronization); [lifetime glossary](../../../../glossary.md#term-lifetime); [compile glossary](../../../../glossary.md#compile); [publication glossary](../../../../glossary.md#term-publication); [run glossary](../../../../glossary.md#term-run); [binding glossary](../../../../glossary.md#term-binding); [materialization glossary](../../../../glossary.md#materialization); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- The review reused the successful compile/run evidence because its corrections did not change
  the exact Java or output blocks. A final comparison confirmed that the guide block still matched
  `/tmp/TensorGuideExample.java` byte for byte.
- The five-file Markdown validator passed links, generated anchors, fences, final newlines, and
  whitespace; targeted opening/guide searches found none of the replaced stale claims.
- The brief remained below 200 lines and 15 KB. The Java/Gradle scan was empty, the complete path
  audit equaled the exact five-file allowlist, and <span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>`git diff --check` passed.
- Javadoc, glossary, architecture, tests, and other guides need no change: the final documentation
  describes current contracts and public behavior without changing Java, terminology, authority,
  backend capability, or workflow beyond the two selected pages.<span data-glossary-navigation="true"> ([backend capability glossary](../../../../glossary.md#backend-capability))</span>

Status: Complete
