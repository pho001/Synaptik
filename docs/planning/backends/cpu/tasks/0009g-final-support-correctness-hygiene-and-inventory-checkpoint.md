# Task 0009G: Final Support, Correctness, Hygiene, and Inventory [Checkpoint](../../../../glossary.md#term-checkpoint)

## Reader context

This task records a final accounting checkpoint joining the live CPU fixture inventory to
operation-family-form-ledger-v3.tsv. `CpuGeneratedCoverageCheckpointTest` reconstructs provider,
preparer, and finalizer/generator outcomes so each finite owner is classified as generated, direct,
or exactly rejected. Requested strategies are distinguished from selected strategies; observing four
selected strategies across the inventory does not grant every strategy to every operation form.

Here, semantic witnesses invoke generated entries against independent expected behavior.
Structural-oracle and source/class-file hygiene evidence instead examine implementation shape and
forbidden hot-path machinery; neither substitutes for the other or establishes performance success.
Analysis owns route and resource decisions, finalization constructs prepared work after assignment,
and Runtime invokes it. Run-owned workspace remains temporary execution storage, not persistent
model state.

Current-topic navigation (not a replacement for the historical record):

- [current CPU preparation and invocation handoffs](../../../../explanation/packages/cpu.md#read-the-pipeline-before-the-packages) — Names the current analysis, shared assignment, finalization, cold binding, and Runtime participants.
- [current ordinary-production route policy](../../../../design/notes/cpu-kernel-strategy.md#current-ordinary-production-decision) — Prevents historical vector or parallel evidence from being read as current production-default authorization.

## Status

Complete

## Goal

Close [CPU](../../../../glossary.md#term-cpu) 0009 with one evidence-backed, [fail-closed](../../../../glossary.md#term-fail-closed) accounting [checkpoint](../../../../glossary.md#term-checkpoint). For every currently
advertised [operation](../../../../glossary.md#operation) family and every meaningful [finite](../../../../glossary.md#term-finite) data-type, [carrier](../../../../glossary.md#term-storage-carrier), [layout](../../../../glossary.md#layout)/access, and
requested/selected execution-strategy combination, record exactly one of supported direct
execution, supported generated execution, or exact rejection. Reconcile those facts with semantic
invocation, selected-route, and hot-path-hygiene evidence without requiring every operation to
implement every strategy.

## [Scope](../../../../glossary.md#term-scope)

- Reproduce the live provider-to-preparer-to-finalizer/generator fixture inventory and reconcile
  each exact owner, operation form, descriptor/data-type roles, carrier ordering, dense or
  general layout access, requested and selected scalar, vector, parallel, or vector-parallel
  strategy, materialization, selected route, generated entry, and outcome fact.<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture); [owner glossary](../../../../glossary.md#term-owner); [operation glossary](../../../../glossary.md#operation); [descriptor glossary](../../../../glossary.md#term-descriptor); [carrier glossary](../../../../glossary.md#term-storage-carrier); [layout glossary](../../../../glossary.md#layout); [scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector); [materialization glossary](../../../../glossary.md#materialization))</span>
- Reconcile `operation-family-form-ledger-v3.tsv` with that inventory as the readable family
  summary. Every family must state which meaningful combinations are generated, direct, or
  rejected. Scalar, vector, parallel, and vector-parallel apply only where a family has a
  meaningful supported request/selection; absence is not a failure when it is rejected or has no
  such execution model.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Preserve the exact generated/rejected partition and verify every generated owner has an
  existing independent semantic invocation witness, while each rejected owner stays outside
  generation and CPU ownership. Reconcile direct/composition and retained-generated routes with
  their lowerer, IR, emitter/reference, prepare/finalization, invocation, artifact/cache/schema,
  and evidence owners.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [CPU glossary](../../../../glossary.md#term-cpu); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [prepare glossary](../../../../glossary.md#prepare); [artifact glossary](../../../../glossary.md#term-artifact); [cache glossary](../../../../glossary.md#term-cache); [schema glossary](../../../../glossary.md#term-schema))</span>
- Run proportionate source/call-site/Class-File or decompilation hygiene checks for selected hot
  paths. Retained generated specializations keep their own optimal clean-Java oracle for the same
  algorithm, loop/dataflow/store shape, and avoidable-overhead profile where required; do not
  construct a universal comparator or promote partial evidence to universal proof.<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run); [decompilation glossary](../../../../glossary.md#term-decompilation); [oracle glossary](../../../../glossary.md#term-test-oracle); [dataflow glossary](../../../../glossary.md#term-dataflow); [shape glossary](../../../../glossary.md#shape))</span>
- Update only the CPU checkpoint tests and checked coverage artifacts needed to make accounting
  reproducible, plus directly affected CPU Javadocs, CPU documentation, glossary, and planning
  status records if evidence proves a genuine documentation discrepancy.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- New semantic capability, operation, data type, carrier, layout, vectorization, parallel policy,
  route migration, generic verifier, cache/schema design, or Model/Compiler/Prepare/Runtime/public
  API change.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [data type glossary](../../../../glossary.md#data-type--datatype); [carrier glossary](../../../../glossary.md#term-storage-carrier); [layout glossary](../../../../glossary.md#layout); [cache glossary](../../../../glossary.md#term-cache); [schema glossary](../../../../glossary.md#term-schema); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- A universal all-strategies matrix; fresh profiling, benchmark thresholds, five-fork campaigns,
  or reopening user-closed loss/stable-reduction performance campaigns. Existing performance
  facts, including non-passing facts, are reported accurately; fresh profiling is optional and
  non-blocking unless it identifies a correctness regression.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [fork glossary](../../../../glossary.md#term-fork-meanings); [loss glossary](../../../../glossary.md#loss); [regression glossary](../../../../glossary.md#term-regression))</span>
- Architecture, dependency, shared lifecycle, build, backend-conformance, integration, or
  architecture-test changes unless a discovered discrepancy needs a separate planned follow-up.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Architecture references

- `ARCHITECTURE.md` sections on backend preparation, CPU backend routes, generated code, and
  Runtime execution are authoritative.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [CPU glossary](../../../../glossary.md#term-cpu); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- `docs/architecture/current-architecture-plan.md` is explanatory only.
- Parent CPU 0009, the CPU master plan, and completed 0009A--0009F3 are planning evidence, not
  architecture rules.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Architecture constraints

[CPU](../../../../glossary.md#term-cpu) analysis owns [lowering](../../../../glossary.md#lowering), route selection, [specialization](../../../../glossary.md#term-specialization), and exact shared-resource
declarations; finalization constructs [immutable](../../../../glossary.md#term-immutable) prepared work after shared slot assignment;
[Runtime](../../../../glossary.md#term-runtime) cold-binds typed representations and executes prepared work without route selection.
Preserve [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) inputs, [run](../../../../glossary.md#term-run)-owned workspace, [publication](../../../../glossary.md#term-publication)/failure cleanup, and parallel range
[ownership](../../../../glossary.md#term-ownership-meanings). Selected hot loops add no avoidable allocation, boxing, reflection, [synchronization](../../../../glossary.md#term-synchronization),
map/string [dispatch](../../../../glossary.md#term-dispatch), or per-[element](../../../../glossary.md#term-element) virtual dispatch. Generated specialized paths retain the
architecture-required clean-Java [oracle](../../../../glossary.md#term-test-oracle) discipline. Stop if evidence requires an architecture,
[module](../../../../glossary.md#term-module)-boundary, [dependency](../../../../glossary.md#term-dependency), capability, public-[API](../../../../glossary.md#term-application-programming-interface-api), [lifecycle](../../../../glossary.md#lifecycle), or resource-contract change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` for [CPU](../../../../glossary.md#term-cpu) capability and unsupported-boundary tests.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` for inventory, disposition,
  evidence, semantic-witness, and hot-path-hygiene checkpoint tests/resources.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Existing `internal.lowering`, `internal.ir`, `internal.prepare`, `internal.executable`,
  `internal.cache`, and `internal.reference` packages are evidence [owners](../../../../glossary.md#term-owner) only.

Packages added or changed:

- None.

Type placement:

- Existing `CpuGeneratedCoverageCheckpointTest` and related test/resource owners retain the
  accounting logic because they reconstruct the live fixture matrix and its exact boundary.<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>

## Affected files

Expected:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneratedCoverageCheckpointTest.java`
- At most three related existing [CPU](../../../../glossary.md#term-cpu) [checkpoint](../../../../glossary.md#term-checkpoint)/evidence/unsupported-boundary test files.
- `backends/cpu/src/test/resources/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/generated-coverage-inventory.tsv`
- `backends/cpu/src/test/resources/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/operation-family-form-ledger-v3.tsv`
- At most two related disposition/evidence resources if live facts require them.
- Directly affected CPU Javadocs, CPU guide, glossary, parent 0009, CPU master plan, and roadmap
  only when completed evidence changes a documented fact.

## Maximum [scope](../../../../glossary.md#term-scope)

At most twelve [CPU](../../../../glossary.md#term-cpu) test/resource/documentation paths and the three [planning](../../../../glossary.md#term-planning) status paths may
change. Production Java must not change. If the live [matrix](../../../../glossary.md#term-matrix-meanings) exposes a missing semantic family,
provider/preparer behavior change, route migration, or shared-contract gap, stop and create one
bounded Draft follow-up; do not broaden this [checkpoint](../../../../glossary.md#term-checkpoint).

## Acceptance criteria

1. Live fixture reconstruction and the frozen inventory agree byte-for-byte, with one owner per
   meaningful supported/rejected combination and 17,463 generated plus 173 exact rejected owners
   (17,636 total), unless a source-backed correction updates all checked facts together.<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture); [owner glossary](../../../../glossary.md#term-owner))</span>
2. The family ledger and live inventory are mutually accountable: every ledger family/form maps
   to live owners or an explicit rejection, and every live operation form has one ledger family.
   It verifies meaningful type, carrier, contiguous/dense or non-contiguous/general layout, and
   selected scalar/vector/parallel/vector-parallel facts where applicable, and fails for an
   orphan, duplicate, relabelled, stale, unsupported, or silently unclassified combination.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [carrier glossary](../../../../glossary.md#term-storage-carrier); [contiguous glossary](../../../../glossary.md#term-contiguous); [layout glossary](../../../../glossary.md#layout); [scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector))</span>
3. Every generated owner has an independent direct generated-entry semantic witness and selected
   route; every rejected owner has no generated class and fails at the correct provider or
   preparer boundary. Existing direct/composition and retained-generated decisions remain
   truthful, with no fallback or dual route introduced.
4. Selected hot paths have proportionate automated source/call-site and actual Class-File or
   decompilation hygiene evidence, including the applicable family clean-Java oracle. It rejects
   new hidden allocation, boxing, reflection, synchronization, map/string dispatch, per-element
   virtual dispatch, route selection, or cache leakage in a selected loop.<span data-glossary-navigation="true"> ([decompilation glossary](../../../../glossary.md#term-decompilation); [oracle glossary](../../../../glossary.md#term-test-oracle); [synchronization glossary](../../../../glossary.md#term-synchronization); [dispatch glossary](../../../../glossary.md#term-dispatch); [element glossary](../../../../glossary.md#term-element); [cache glossary](../../../../glossary.md#term-cache))</span>
5. Existing performance evidence and its non-passing/user-closed status are reported but are not
   acceptance gates. No fresh benchmark or five-fork campaign is required.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings))</span>
6. A separate documentation-focused clean context finalizes affected Javadocs, CPU documentation,
   planning records, and glossary impact in the same change, recording explicit no-change
   conclusions when behavior and reusable terminology are unchanged.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [planning glossary](../../../../glossary.md#term-planning))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Task validation — after final [CPU](../../../../glossary.md#term-cpu) test/resource edits, [run](../../../../glossary.md#term-run) once:

```bash
./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest --tests io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedCoverageCheckpointTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedCoverageEvidenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedDirectEvidenceClosureTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedStructuralOracleCatalogTest
```

Capability [checkpoint](../../../../glossary.md#term-checkpoint) — after focused validation:

```bash
./gradlew :backends:cpu:test
./gradlew :backends:cpu:javadoc
```

Repository/[CI](../../../../glossary.md#term-continuous-integration-ci) validation — run `./gradlew test` for this final CPU 0009 capability checkpoint
unless a concrete environment failure is recorded. Architecture, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and
integration tests need no change or rerun when the checkpoint changes no [dependency](../../../../glossary.md#term-dependency) rule, backend
behavior contract, or end-to-end behavior; record that conclusion. Stop and plan the correct test
work if evidence changes one of those contracts.

Documentation pass — apply General and [Planning](../../../../glossary.md#term-planning) profiles and documentation rules; check links,
anchors, headings, fences, terminology/glossary, exact [scope](../../../../glossary.md#term-scope)/status ordering, line endings, and
whitespace, then run:

```bash
git diff --check
git status --short -uall
```

## Dependencies

- Ready parent CPU 0009 and Complete 0009A--0009F3, especially 0009F3's specialized-family
  route decisions.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Current CPU provider/preparer/finalizer, coverage inventory, family ledger, disposition/evidence
  registries, semantic closures, and generated-code hygiene evidence.
- Completed CPU 0005 strategy foundation, 0007A1C evidence closure, 0008 families, and existing
  backend capability/conformance ownership boundaries.<span data-glossary-navigation="true"> ([backend capability glossary](../../../../glossary.md#backend-capability); [conformance glossary](../../../../glossary.md#term-conformance); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Follow-up tasks

- None expected. If acceptance reveals a missing capability, route, semantics, shared contract,
  or documentation correction outside this scope, add only its immediate bounded Draft follow-up
  to the CPU master plan and keep 0009G incomplete.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Architecture impact

Expected impact: None.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU 0009G. Read AGENTS.md, ARCHITECTURE.md,
current architecture plan, Planning Guide, CPU master plan, parent CPU 0009, completed 0009A--F3
(especially F3), this task, CPU production/test inventories, capability/conformance ownership, and
generated-code catalog/checkpoint evidence. Implement only this bounded checkpoint. Account for
every meaningful supported or rejected datatype/carrier/layout/strategy combination; do not
require every operation to have every strategy. Do not migrate routes, change production Java,
reopen performance campaigns, add benchmark gates, commit, stage, push, or use GSD. Stop on an
architecture or scope conflict. Hand the stabilized diff and test evidence to a distinct clean
documentation-focused context following documentation-rules.md; do not mark Complete until its
review is recorded.
```

## Local decisions

0009G is one cohesive final [checkpoint](../../../../glossary.md#term-checkpoint), not ordered child tasks: the [canonical](../../../../glossary.md#term-canonical-form) inventory already
reconstructs the complete [finite](../../../../glossary.md#term-finite) [owner](../../../../glossary.md#term-owner) [matrix](../../../../glossary.md#term-matrix-meanings) and the completed family decisions establish every
current selected route. Splitting its cross-family join would leave all-family closure
unverifiable until a later task and violate the sole-frontier rule.

## Known limitations

This closes truthful support/correctness accounting, not universal structural or performance
promotion. Existing 0008I [loss](../../../../glossary.md#loss) [fork](../../../../glossary.md#term-fork-meanings)-0 and user-closed forks 1--4, 0008O, and 0008P performance
facts remain historical [fail-closed](../../../../glossary.md#term-fail-closed) evidence and cannot be relabelled as passing.

## [Validation](../../../../glossary.md#term-validation) evidence

The implementation context ran the specified six-class focused [CPU](../../../../glossary.md#term-cpu) command after the final
test/resource edits; it passed. The final CPU capability [checkpoint](../../../../glossary.md#term-checkpoint) also passed on 2026-09-10:
`./gradlew :backends:cpu:test` reported 931 tests, 28 skipped, and zero failures or errors, and
`./gradlew :backends:cpu:javadoc` passed. Repository validation, `./gradlew test`, passed. The
implementation context also reported `git diff --check` passing before this documentation pass.

The separate documentation-focused context
`01a08bb5-c7ad-79a1-98aa-b047cd6ffd64` reviewed the final four-file implementation diff, the
live-inventory checkpoint and its evidence [owners](../../../../glossary.md#term-owner), the parent and completed 0009A--F3
records, the CPU master plan, roadmap, CPU guide, glossary, current Javadocs, and the applicable
General, [Planning](../../../../glossary.md#term-planning), and [Javadoc](../../../../glossary.md#term-javadoc) profiles. It did not rerun Java/[Gradle](../../../../glossary.md#term-gradle) tests because its edits are
Markdown only and the supplied executable evidence is current. It verified that the new readable
ledger rows cover all 120 live [operation](../../../../glossary.md#operation) forms; dense/general [layout](../../../../glossary.md#layout) presence and the four
actually selected strategies are asserted, while unsupported or non-meaningful strategies are
not represented as universal support. The checkpoint remains proportional hygiene and
support-accounting evidence, not universal clean-Java-[oracle](../../../../glossary.md#term-test-oracle) or performance proof.

No CPU guide update is needed: no CPU capability, route, user workflow, or public behavior
changed. No Javadoc update or generation rerun is needed: production Java and its contracts did
not change, and the supplied CPU Javadoc generation passed. No glossary update is needed:
"requested strategy", "selected strategy", dense/general layout, and generated route are
existing terms whose meanings did not change. No architecture, build, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), or
integration documentation/test update is needed: the diff changes neither dependencies nor
backend or end-to-end behavior.

## Implementation notes

The readable ledger is deliberately an accountable summary of the [finite](../../../../glossary.md#term-finite) live inventory, rather
than a claim that every form supports every strategy. The 45 added rows classify previously
unrepresented live forms; the three pre-existing broad historical summary rows remain explicit.
No direct/generated route changed.

## Completion summary

[CPU](../../../../glossary.md#term-cpu) 0009G completes the final truthful CPU support/correctness [checkpoint](../../../../glossary.md#term-checkpoint). The four changed
test/resource paths reconcile all 120 live [operation](../../../../glossary.md#operation) forms with the readable ledger, preserve the
17,463 generated plus 173 exact rejected [owner](../../../../glossary.md#term-owner) [partition](../../../../glossary.md#partition), prove dense/general [layout](../../../../glossary.md#layout) presence,
and observe all four actually selected strategies. [Pointwise](../../../../glossary.md#term-pointwise) ledger [validation](../../../../glossary.md#term-validation) remains
[fail-closed](../../../../glossary.md#term-fail-closed) while accepting live pointwise forms outside the [opcode](../../../../glossary.md#term-opcode) enum. The result does not
claim universal strategy availability, universal structural-[oracle](../../../../glossary.md#term-test-oracle) coverage, or performance
success. Historical CPU 0008I [fork](../../../../glossary.md#term-fork-meanings)-0 NON_PASSING evidence (19 of 792 above 1.15; worst
1.3861164205039096) and user-closed forks 1--4 remain unchanged. No follow-up is required.

Status: Complete
