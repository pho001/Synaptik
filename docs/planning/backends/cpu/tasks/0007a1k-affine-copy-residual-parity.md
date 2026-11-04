# Task 0007A1K: Affine-copy [residual](../../../../glossary.md#term-residual-meanings) parity

## Reader context

The affine-copy residual was a remaining timing gap, not a value saved for differentiation.
A-GENERAL copied raw BFLOAT16 representations through a composed PERMUTE/SLICE mapping.
`CpuAffineCopyEmitter` guarded the exact segment-to-short-array geometry and used ordinal
shifts/masks to derive positions directly. Loading or storing a short here did not decode,
round, or otherwise perform BFLOAT16 arithmetic; NaN payloads and signed-zero bits remained copy
data.

Source/result offsets and strides were element geometry, while Runtime binding supplied the
current carriers and valid spans. Every unproved mapping retained the general-long copy, so the
frozen case did not become a new capability limit. Current layout semantics and CPU lowering
explain why a logical view chain can end in a distinct materialized result without transferring
ownership of the source.

Current reading alongside this historical record:

- [Resolved logical layouts](../../../../reference/tensor/layouts.md#resolved-layouts)
  — explains offsets, strides, referenced spans, and why geometry does not own storage.
- [CPU partition and unit lowering](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternallowering)
  — explains virtual intermediates, boundary stores, and family-specific mappings.

## Status

Complete

## Goal

Correct exactly the persistent `A-GENERAL` generated/direct [residual](../../../../glossary.md#term-residual-meanings) through its sole code-shaping
[owner](../../../../glossary.md#term-owner), `CpuAffineCopyEmitter`. Preserve the frozen [CPU](../../../../glossary.md#term-cpu) 0007A1C raw-[BFLOAT16](../../../../glossary.md#term-bfloat16) copy semantics,
composed PERMUTE/[SLICE](../../../../glossary.md#slice) mapping, segment-to-[heap](../../../../glossary.md#term-heap) carriers, nonzero offsets, positive non-unit
[strides](../../../../glossary.md#term-stride), arbitrary legal output subranges, and typed general fallback while passing the unchanged
five-[fork](../../../../glossary.md#term-fork-meanings) `<= 1.15x` gate.

## [Scope](../../../../glossary.md#term-scope)

- Use the [immutable](../../../../glossary.md#term-immutable) A1C probe sources and the accepted A1J evidence bundle at
  `/private/tmp/synaptik-cpu-0007a1j-96RvT1wf`.
- Correct only `A-GENERAL`: raw-[BFLOAT16](../../../../glossary.md#term-bfloat16) PERMUTE/[SLICE](../../../../glossary.md#slice) affine copy over [Shape](../../../../glossary.md#shape) `[256,32,32]`,
  [`MemorySegment`](../../../../glossary.md#term-memorysegment) input to `short[]` output, nonzero offsets, positive non-unit strides, and the
  frozen composed logical mapping.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride))</span>
- Preserve represented bits exactly, including NaN payloads and signed zeros; the copy performs no
  BFLOAT16 arithmetic, conversion, promotion, canonicalization, or numerical interpretation.<span data-glossary-navigation="true"> ([represented bits glossary](../../../../glossary.md#term-represented-bits); [NaN glossary](../../../../glossary.md#term-nan); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- Preserve output-domain ownership, distinct-address writes, complete overlap rejection, input
  immutability, arbitrary legal half-open output ranges, scalar or parallel-scalar reuse, and zero
  workspace/materialization beyond the already owned affine boundary copy itself.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [half-open glossary](../../../../glossary.md#term-half-open-interval); [scalar glossary](../../../../glossary.md#term-scalar); [materialization glossary](../../../../glossary.md#materialization))</span>
- Treat the frozen optimal direct primitive Java loop as the generated-code design and review
  oracle. Preserve its semantic mapping, cursor/odometer progression, carrier work, hot-loop
  dataflow, and avoidable-overhead profile.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [carrier glossary](../../../../glossary.md#term-storage-carrier); [dataflow glossary](../../../../glossary.md#term-dataflow))</span>
- Add only a completely guarded proved body to `CpuAffineCopyEmitter`. Every mapping, carrier, or
  layout fact not completely proved must retain the existing typed general-long body.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- Keep the generated [target](../../../../glossary.md#term-target) final and field-free with exactly one typed static `invoke`, no
  constructor or bridge, and no allocation, reflection, dynamic construct, generic `Object`
  descriptor, runtime operation dispatch, or Synaptik-runtime call.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [runtime glossary](../../../../glossary.md#term-runtime); [operation glossary](../../../../glossary.md#operation); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Advance the generator schema exactly once if emitted bytes change, propagate it to existing
  schema assertions, and keep older envelopes as incompatible safe misses without migration or
  dual-schema reuse.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- Retain immutable inputs, exact commands, environment, semantics, generated classes, complete
  disassembly, structure/hot-loop reports, five accepted forks, rejected samples, summaries, XML,
  context identifiers, inventories, and checksums.<span data-glossary-navigation="true"> ([forks glossary](../../../../glossary.md#term-fork-meanings))</span><span data-glossary-navigation="true"> ([samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Changing the frozen probe, case, Shape, carriers, comparator, JVM flags, fork protocol,
  threshold, direct oracle, or semantics.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [fork glossary](../../../../glossary.md#term-fork-meanings); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Correcting `P-SCALAR-GENERAL`, `S-GENERAL-MIN`, `X-MIN-MULTI`, or any completed row.
- Changing `CpuCarrierEmitter`, another production code-shaping owner, affine lowering/IR, view
  folding, validation, resources, orchestration, route selection, Prepare/Runtime behavior,
  capabilities, public API, dependencies, build structure, conformance, or integration.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [lowering glossary](../../../../glossary.md#lowering); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [view glossary](../../../../glossary.md#term-view-meanings); [validation glossary](../../../../glossary.md#term-validation); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Adding a general layout optimizer, per-element address table, workspace, materialization route,
  vector/native affine copy, dynamic Shape/layout support, tuning input, or universal performance
  claim.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [optimizer glossary](../../../../glossary.md#term-optimizer); [element glossary](../../../../glossary.md#term-element); [materialization glossary](../../../../glossary.md#materialization); [vector glossary](../../../../glossary.md#term-vector))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md)
- [`planning guide`](../../../planning-guide.md)
- [`documentation rules`](../../../../developer-guide/documentation-rules.md)
- [`CPU backend guide`](../../../../backend-guide/cpu-backend.md)
- [`glossary`](../../../../glossary.md)
- [`CPU master plan`](../master-plan.md)
- [`CPU 0006 affine views`](0006-portable-static-affine-views-and-boundary-materialization.md)
- [`CPU 0007A0A affine and movement parity`](0007a0a-affine-and-movement-generated-loop-parity.md)
- [`CPU 0007A1C evidence closure`](0007a1c-generated-direct-evidence-closure.md)
- [`CPU 0007A1J cumulative-scan residual parity`](0007a1j-cumulative-scan-residual-parity.md)

## Architecture constraints

- Model owns view semantics and represented data types. This task changes only CPU-private
  generated affine-copy realization and evidence.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [view glossary](../../../../glossary.md#term-view-meanings); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- CPU analysis retains view composition, lowering, strategy, resource declaration, and route
  selection before shared assignment; CPU finalization realizes one artifact afterward; Runtime
  only invokes prepared direct carriers and output bounds.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [artifact glossary](../../../../glossary.md#term-artifact); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Generated code must preserve the optimal clean Java affine-copy mapping and hot-loop/dataflow
  shape without interpreting Model operations or layouts at Runtime.<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow); [layouts glossary](../../../../glossary.md#layout))</span>
- Benchmark observations accept or reject this bounded code shape and never select production
  settings.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — sole affine-copy code-shaping
  owner and generated-kernel tests.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — [schema](../../../../glossary.md#term-schema) compatibility when bytes change.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — existing schema assertion and
  finalization wording only when the current version changes.

Packages added or changed: None.

Type placement:

- `CpuAffineCopyEmitter` remains the only authorized production algorithm-shaping [owner](../../../../glossary.md#term-owner).
- No helper, public type, package, or [module](../../../../glossary.md#term-module) may be added.

## Affected files

Expected implementation paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAffineCopyEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- directly affected existing [CPU](../../../../glossary.md#term-cpu) [cache](../../../../glossary.md#term-cache)/code-generation/[prepare](../../../../glossary.md#prepare) package Javadocs only as required;
- `CpuAffineCopyGeneratedKernelTest.java` and `CpuGeneratedDirectEvidenceClosureTest.java` only for
  stable semantic/Class-File assertions;
- the four existing [schema](../../../../glossary.md#term-schema)-assertion tests when schema advances; and
- CPU [backend](../../../../glossary.md#backend) guide, glossary, this task, CPU 0007A1C, CPU master plan, and roadmap.

## Maximum [scope](../../../../glossary.md#term-scope)

At most one production algorithm-shaping [owner](../../../../glossary.md#term-owner) may change. With [schema](../../../../glossary.md#term-schema)/package propagation,
affine/closure tests, four schema assertion owners, and documentation/[planning](../../../../glossary.md#term-planning), the overall change
must remain at or below 18 repository paths. If a second production owner, new helper,
[IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/resource change, [module](../../../../glossary.md#term-module), or broader path set is required, stop and replan.

## Acceptance criteria

- Frozen sources recompile byte-identically and exact semantics remain `VERIFIED,20`.
- `A-GENERAL` passes `<= 1.15x` in every one of five fresh isolated forks and in the cross-fork
  median.
- Controls `P-VECTOR-SEGMENT`, `P-INTEGRAL-MIXED`, and `O-ARGSORT` pass every [fork](../../../../glossary.md#term-fork-meanings) and [median](../../../../glossary.md#term-median).
- The unchanged full probe exits nonzero only for the three explicitly deferred rows.
- Exact raw-BFLOAT16 output bits, input immutability, canaries, composed mapping, arbitrary legal
  output-range ownership, and zero-resource contracts remain unchanged.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Target Class-File and complete disassembly prove the typed guarded affine body and retained
  fallback, with no avoidable per-element coordinate division/remainder, packed-geometry lookup,
  layout construction, generic dispatch, or helper call in the proved hot loop.<span data-glossary-navigation="true"> ([Target glossary](../../../../glossary.md#term-target); [element glossary](../../../../glossary.md#term-element); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [lookup glossary](../../../../glossary.md#term-lookup); [layout glossary](../../../../glossary.md#layout); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Focused affine/closure/[schema](../../../../glossary.md#term-schema) [owners](../../../../glossary.md#term-owner) pass, followed by one uncached [CPU](../../../../glossary.md#term-cpu) suite.
- A distinct clean documentation context finalizes affected Javadocs, guide/glossary impact,
  evidence/status records, and documentation validation without rerunning stable Java tests.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) the focused `CpuAffineCopyGeneratedKernelTest`, `CpuGeneratedDirectEvidenceClosureTest`, and
four [schema](../../../../glossary.md#term-schema)-assertion [owners](../../../../glossary.md#term-owner), followed once by:

```bash
./gradlew :backends:cpu:test --rerun-tasks
```

Recompile and [checksum](../../../../glossary.md#term-checksum) the frozen sources, run `VERIFIED,20`, retain the [target](../../../../glossary.md#term-target) and all required
control/deferred generated classes and complete disassembly, run five sequential fresh [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) [forks](../../../../glossary.md#term-jvm-fork)
with the unchanged `-Xms1g -Xmx1g` protocol, and regenerate the evidence inventory/checksums. A
complete five-[fork](../../../../glossary.md#term-fork-meanings) [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) is accepted only if the target, controls, and every non-deferred row
pass; retain and reject the whole sample otherwise.

The documentation pass runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short
```

It also checks Markdown links/anchors/fences/final newlines, schema propagation, exact path bounds,
one [Ready](../../../../glossary.md#term-ready) task, A1C incomplete, A1K Complete only after every gate, [CPU](../../../../glossary.md#term-cpu) 0007A2 Draft/blocked, and
empty staging. Repository-wide, architecture, [conformance](../../../../glossary.md#term-conformance), and integration suites remain deferred
to CPU 0009 or [CI](../../../../glossary.md#term-continuous-integration-ci) because no shared contract may change.

## Dependencies

- [CPU](../../../../glossary.md#term-cpu) 0007A1C supplies the [immutable](../../../../glossary.md#term-immutable) twenty-row corpus and remains incomplete.
- CPU 0006 and CPU 0007A0A supply exact affine-view composition and generated typed-copy
  foundations.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings))</span>
- CPU 0007A1D supplies stable invocation-local segment [layout](../../../../glossary.md#layout) handling.
- CPU 0007A1J is Complete at [schema](../../../../glossary.md#term-schema) 38 and supplies the accepted five-[fork](../../../../glossary.md#term-fork-meanings) [residual](../../../../glossary.md#term-residual-meanings) frontier.

## Follow-up tasks

- Detailed [CPU 0007A1L pointwise general-loop residual parity](0007a1l-pointwise-general-loop-residual-parity.md)<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise))</span>
  is the sole next Ready task and owns <span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>`P-SCALAR-GENERAL`. `S-GENERAL-MIN` and
  `X-MIN-MULTI` remain Draft and unassigned.
- Resume [CPU](../../../../glossary.md#term-cpu) 0007A1C closure only after every persistent row closes.
- CPU 0007A2 remains Draft and blocked behind CPU 0007A1C and all corrective [residual](../../../../glossary.md#term-residual-meanings) work.

## Architecture impact

Expected impact: None.

If the task requires an architecture, [dependency](../../../../glossary.md#term-dependency), public [API](../../../../glossary.md#term-application-programming-interface-api), semantic, resource, route,
[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime), build, [IR](../../../../glossary.md#term-intermediate-representation-ir), [lowering](../../../../glossary.md#lowering), [conformance](../../../../glossary.md#term-conformance), or integration change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0007A1K. Work on the existing dirty
worktree without committing, pushing, staging, resetting, reverting, deleting, or modifying
unrelated work. Do not use a GSD skill or workflow.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, CPU task 0007A1K, CPU tasks
0006/0007A0A/0007A1C/0007A1J, the relevant affine/cache/package source and tests, and the complete
A1J evidence bundle. Implement exactly the bounded one-owner A-GENERAL correction through
CpuAffineCopyEmitter. Preserve the immutable probe, raw-BFLOAT16 represented-bit copy, composed
PERMUTE/SLICE mapping, segment-to-heap carriers, offsets/strides, arbitrary legal output ranges,
zero resources, typed fallback, and optimal direct-clean-Java algorithm/dataflow shape. Stop
before changing a second production owner or any IR/lowering/resource/architecture/lifecycle
contract.

Run the specified focused/full CPU, semantic/Class-File/five-fork, forbidden-reference, schema,
scope, and preservation gates. Hand the stabilized diff and exact evidence to a distinct clean
documentation context. Do not mark Complete until the target and all controls pass every gate.
Do not commit, push, or stage.
```

## Local decisions

- Accepted A1J [forks](../../../../glossary.md#term-fork-meanings) measure `A-GENERAL` at `4.569629881`, `4.621004147`, `4.614792202`,
  `4.547987466`, and `4.651517540`, with [median](../../../../glossary.md#term-median) `4.614792202`, while all three controls pass.
  The row is therefore persistent rather than a one-fork anomaly.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings))</span>
- This row is selected before the other residuals because it has one exact copy owner, represented-
  bit semantics, one output-domain pass, and zero workspace. Pointwise general addressing,
  scatter copy/update/reduction, and multi-axis extrema retain distinct owner and algorithm risks.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [Pointwise glossary](../../../../glossary.md#term-pointwise); [axis glossary](../../../../glossary.md#term-axis); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- No other residual shares the affine-copy owner or raw-copy algorithm, so this remains a one-row
  task.<span data-glossary-navigation="true"> ([residual glossary](../../../../glossary.md#term-residual-meanings))</span>

## Known limitations

- Ratios apply only to the frozen case, JVM, host, and protocol; they are not a universal affine-
  copy claim or production tuning input.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [host glossary](../../../../glossary.md#term-host))</span>
- This task does not close A1C or authorize [CPU](../../../../glossary.md#term-cpu) 0007A2.

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `01a023f7-d9da-72f0-ad8b-29f962a1c3f5` produced the immutable evidence
  bundle at <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>`/private/tmp/synaptik-cpu-0007a1k-tSVmcoQh` from unchanged HEAD
  `d2ca569db1fa22b307f691209ad69af9d1ecbcca`. Its manifest, final reports, accepted and rejected
  fork evidence, generated classes, complete <span data-glossary-navigation="true"> ([manifest glossary](../../../../glossary.md#term-manifest); [fork glossary](../../../../glossary.md#term-fork-meanings))</span>`javap -c` and `javap -v`, XML summaries,
  `implementation.patch`, inventories, checksums, and context record were independently reviewed.
- Frozen semantics passed exactly (`VERIFIED,20`). The focused matrix passed 47 tests in six
  suites; the authoritative CPU run passed 355 of 356 tests in 54 suites with zero failures or
  errors and one expected skip.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [CPU glossary](../../../../glossary.md#term-cpu); [run glossary](../../../../glossary.md#term-run))</span>
- Five accepted isolated forks measured `A-GENERAL` at `0.652072452x`, `0.653888342x`,
  `0.654943789x`, `0.660072122x`, and `0.651079970x`; [median](../../../../glossary.md#term-median) `0.653888342x`. All three controls
  passed every fork and median. One complete sample was rejected because its fourth-fork
  `M-CONCAT` ratio was `1.186794092x`; none of that [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) contributes accepted evidence.
- The accepted full summaries leave exactly `P-SCALAR-GENERAL`, `S-GENERAL-MIN`, and
  `X-MIN-MULTI`. Their accepted medians are respectively `2.638802714x`, `23.472493883x`, and
  `9.072969478x`.
- Complete disassembly proves a final, field-free generated class with one typed static
  `invoke(MemorySegment, short[], long[], long, long)`. The guarded body checks the complete
  packed geometry, ordered range, and sentinel address pairs, derives coordinates with integer
  shifts/masks, and performs direct <span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [masks glossary](../../../../glossary.md#term-boolean-mask))</span>`JAVA_SHORT_UNALIGNED` loads and `short[]` stores. The
  unchanged typed general-long body remains the failed-guard fallback.<span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard))</span>
- Documentation context `01a0241d-9eba-7df1-b1ca-59c96a79cb93` independently finalized the
  affected Javadocs, guide, glossary, and synchronized planning without changing executable Java
  or tests or rerunning stable Java, semantic, timing, or benchmark commands.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Implementation notes

- `CpuAffineCopyEmitter` recognizes only the frozen [BFLOAT16](../../../../glossary.md#term-bfloat16) [represented-bit](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) [`MemorySegment`](../../../../glossary.md#term-memorysegment) to
  `short[]` PERMUTE/[SLICE](../../../../glossary.md#slice) form: source [Shape](../../../../glossary.md#shape) `[256,32,32]`, result Shape `[32,256,32]`, source
  offset `5`, source [strides](../../../../glossary.md#term-stride) `[2048,64,2]`, result strides `[64,2048,2]`, and identity slice axis
  zero <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [axis glossary](../../../../glossary.md#term-axis))</span>`[0,32)`. Every legal [half-open](../../../../glossary.md#term-half-open-interval) output range is supported after complete [runtime](../../../../glossary.md#term-runtime) guards.
- The proved loop uses ordinal shifts/masks and direct typed carrier access. It performs no
  BFLOAT16 numerical interpretation, conversion, canonicalization, allocation, helper dispatch,
  or per-element geometry lookup. All other affine-copy forms retain the existing typed
  general-long implementation.<span data-glossary-navigation="true"> ([masks glossary](../../../../glossary.md#term-boolean-mask); [carrier glossary](../../../../glossary.md#term-storage-carrier); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [dispatch glossary](../../../../glossary.md#term-dispatch); [element glossary](../../../../glossary.md#term-element); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Generated compatibility advances exactly once to schema 39. Schema 38 and older envelopes are
  incompatible safe misses; no migration or dual-schema reuse was added.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- Architecture/current-plan/ADR/architecture-test conclusion: the bounded private generated body
  changes no architecture rule, module boundary, dependency direction, or shared contract, so no
  architecture document, ADR, or architecture test changes.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [architecture test glossary](../../../../glossary.md#term-architecture-test))</span>
- API/capability/lifecycle conclusion: public APIs and capabilities, IR/lowering, resource and
  materialization declarations, route selection, Prepare/Runtime ownership, and generic forms are
  unchanged. Backend conformance, integration, native, and vector routes therefore need no edit.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [lifecycle glossary](../../../../glossary.md#lifecycle); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [materialization glossary](../../../../glossary.md#materialization); [Prepare glossary](../../../../glossary.md#prepare); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [vector glossary](../../../../glossary.md#term-vector))</span>
- Build/documentation conclusion: no dependency, Gradle, toolchain, shared-module, example, or
  unrelated documentation change is required. The CPU guide and glossary record only the bounded
  supported schema-39 facts; affected internal Javadocs retain the exact fallback and ownership
  boundaries.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [toolchain glossary](../../../../glossary.md#term-toolchain); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Completion summary

- Completed changes: added the completely guarded frozen affine-copy body at schema 39, retained
  the typed fallback, synchronized schema assertions, and finalized directly affected Javadocs,
  CPU guide, glossary, and planning records.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [CPU glossary](../../../../glossary.md#term-cpu); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: ten implementation-owned production/test paths plus seven
  documentation/planning paths, for 17 combined paths. No executable Java or test changed in the
  documentation pass.
- Tests and [validation](../../../../glossary.md#term-validation): reused `VERIFIED,20`, six suites/47 focused tests, 54 suites/356 CPU tests
  with 355 passes and one expected skip, complete Class-File inspection, five accepted target and
  control forks, and the retained rejected sample. The documentation pass ran CPU Javadoc once,
  inspected affected generated pages, and passed Markdown, schema/status/dependency, scope,
  preservation, staging, HEAD, and whitespace checks.<span data-glossary-navigation="true"> ([forks glossary](../../../../glossary.md#term-fork-meanings))</span><span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [Javadoc glossary](../../../../glossary.md#term-javadoc); [dependency glossary](../../../../glossary.md#term-dependency); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation impact: internal emitter/schema/cache/prepare Javadocs, CPU guide, glossary, this
  task, A1C, CPU master plan, roadmap, and the sole next detailed task are synchronized. No new
  reusable glossary term or public API was introduced.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [prepare glossary](../../../../glossary.md#prepare); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Unresolved issues: exactly `P-SCALAR-GENERAL`, `S-GENERAL-MIN`, and `X-MIN-MULTI` remain.
- Follow-up required: execute Ready CPU 0007A1L, then plan only the next evidence-supported
  owner-cohesive residual correction. CPU 0007A2 remains Draft and blocked until A1C closes.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [owner glossary](../../../../glossary.md#term-owner); [residual glossary](../../../../glossary.md#term-residual-meanings))</span>

Status: Complete
