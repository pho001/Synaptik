# Planning Contract Closure Audit

## Executive conclusion and closure verdict

Verdict: `CLOSED`

The selected `modules/planning` milestone is coherent at its current boundary. The implemented
pipeline asks an immutable occurrence-level capability question, derives complete hard
eligibility, chooses one deterministic cost-free backend owner, groups maximal consecutive
same-owner partitions, and derives immutable logical value requirements. Current source and tests
cover every stage's validation, ordering, association, identity, and reference-retention rules.
No missing planning-owned semantic contract prevents later compiler-owned orchestration.

The five public declarations are sufficient and minimal. The four current evaluator/generator
operations may remain package-private until a concrete compiler consumer establishes one narrow
collaboration boundary. Compiler orchestration, publication planning, prepare, physical memory,
backend lowering, runtime residency, cost-bearing scoring, tracing payloads, and model autotuning
are explicit downstream work rather than closure gaps.

## Authority, scope, and method

`ARCHITECTURE.md` is the authority for ownership, lifecycle, and dependency direction. Current
Java source and tests are primary evidence for implemented behavior. Generated Javadocs establish
the rendered public/internal boundary. Explanatory documentation and completed task history are
supporting consistency evidence.

The audit ran in clean documentation context `/root/audit_planning_0006`. It read the repository
instructions, architecture contract and focused architecture/ADR documents, documentation rules
and profiles, planning guide and roadmap, relevant module master plans, completed Planning
0001-0005 tasks, current planning/model/config/backend-contract source and tests, generated
planning Javadocs, API and guide pages, glossary, and memory-planning note. It changed no Java,
test, Gradle, architecture contract, ADR, architecture test, conformance test, integration test,
compiler, prepare, runtime, backend, or other executable artifact.

Inventory used these exact commands:

```bash
rg --files modules/planning/src/main/java | sort
rg --files modules/planning/src/test/java | sort
rg -n '^(public )?(final )?(class|interface|record|enum|sealed interface) ' modules/planning/src/main/java
rg --files modules/planning/build/docs/javadoc | sort
```

Source and test inspection then compared constructors and operations with every focused test.
Dependency inspection used `modules/planning/build.gradle.kts`, imports, architecture rules, and
the forbidden-dependency scan recorded in section 12.

## Architecture and lifecycle boundary assessment

Planning remains backend-neutral and answers where graph work runs. Its current results contain
backend ownership identities, graph node identities, graph value identities, logical tensor
descriptors, partition relationships, and graph-output preservation. They contain no live backend
provider, device selection, backend route, kernel, lowering, executable, allocation, transfer,
residency, or run state.

Compiler remains the owner of graph-wide invocation, complete owner-map assembly, publication
planning, immutable `CompileArtifacts`, public compile failures, and diagnostics. Prepare owns the
shared transition from compile artifacts to prepared state. Concrete backends own lowering and
physical realization; runtime owns prepared execution and per-run state. The current absence of a
compiler consumer does not move orchestration into planning or require isolated internal stages
to become public prematurely.

No inspected source, dependency, or document contradicts this lifecycle allocation.

## Production, test, and generated-Javadoc inventory

The production inventory contains exactly 12 files:

```text
capability/BackendCapabilityProvider.java
capability/BackendEligibility.java
capability/BackendOwnerSelection.java
capability/OperationCapabilityQuery.java
capability/package-info.java
memory/LogicalMemoryPlan.java
memory/LogicalMemoryPlanning.java
memory/LogicalMemoryRequirement.java
memory/package-info.java
partition/MaximalSameOwnerPartitioning.java
partition/PlannedPartition.java
partition/package-info.java
```

The test inventory contains exactly eight suites:

```text
capability/BackendCapabilityContractTest.java
capability/BackendEligibilityTest.java
capability/BackendOwnerSelectionTest.java
memory/LogicalMemoryPlanTest.java
memory/LogicalMemoryPlanningTest.java
memory/LogicalMemoryRequirementTest.java
partition/MaximalSameOwnerPartitioningTest.java
partition/PlannedPartitionTest.java
```

Generated Javadocs contain the three package summaries and exactly five public planning type
pages:

```text
capability/package-summary.html
memory/package-summary.html
partition/package-summary.html
capability/BackendCapabilityProvider.html
capability/OperationCapabilityQuery.html
memory/LogicalMemoryPlan.html
memory/LogicalMemoryRequirement.html
partition/PlannedPartition.html
```

`allclasses-index.html` and `overview-tree.html` expose those five types. There is no generated
page or public-index entry for `BackendEligibility`, `BackendOwnerSelection`,
`MaximalSameOwnerPartitioning`, or `LogicalMemoryPlanning`. Inspection of all summaries and type
pages found wording consistent with current source; no Java-Javadoc correction is required.

## Capability query and provider assessment

`OperationCapabilityQuery` snapshots one exact non-null `Operation` reference and ordered copied
input/output lists after validating the list references, every descriptor element, and occurrence
signature counts. It retains the exact descriptor references inside those immutable list
snapshots. It does not decide compatibility, availability, ownership, or executability.

`BackendCapabilityProvider` supplies one stable non-null `BackendId` and a deterministic boolean
semantic-support answer for a non-null immutable query. It is intentionally backend-level. It
does not discover providers or devices, evaluate hard requirements, choose a device or route,
score candidates, prepare work, or execute work. `BackendCapabilityContractTest` verifies record
shape, snapshots, exact references, validation, signature occurrence counts, provider shape, and
the null-query contract.

## Hard-eligibility assessment

`BackendEligibility.evaluate` first validates top-level inputs, scans providers in encounter
order, obtains each provider identity once, and rejects duplicate identities. It then validates
all availability snapshots and rejects duplicate snapshot identities. Complete equal-identity
association is checked provider-first for missing snapshots and snapshot-first for missing
providers before any capability call.

After association succeeds, an empty snapshot or exact backend, device, or device-class hard
requirement mismatch skips that provider. Each remaining provider is queried exactly once in
provider order. A true answer retains that provider's exact `BackendId` reference in an immutable
provider-order list. A valid no-match yields an empty list. Provider exceptions propagate and
stop later calls. No device is selected or retained. `BackendEligibilityTest` locks these rules,
including validation/failure ordering and exact-reference behavior.

## Owner-selection and scoring assessment

`BackendOwnerSelection.select` consumes the complete hard-eligible `BackendId` list directly. It
rejects an empty list before reading snapshot elements, then validates the full snapshot input,
rejects duplicate identities, and requires an equal-identity snapshot for every eligible backend;
extra unique snapshots are permitted.

Without a preference it returns the first eligible identity. With a preferred `DeviceClass`, it
returns the first provider-order eligible backend with a matching available device, otherwise the
first eligible backend. It returns the exact eligibility-list reference, re-evaluates neither
capability nor hard eligibility, and chooses no device, route, or kernel. There is no numeric
score, workload/cost classification, estimate, or cost profile. `BackendOwnerSelectionTest`
verifies all comparison, fallback, association, failure-order, and reference rules.

## Partition recipe and generation assessment

`PlannedPartition` is the minimal immutable owner-plus-node-ID recipe. It retains the exact owner
reference and an immutable ordered node-ID snapshot after non-null/non-empty/duplicate validation.

`MaximalSameOwnerPartitioning.partition` validates graph and owner map before output construction:
null keys, unknown keys in numeric node-ID order, missing coverage in graph order, and null owners
in graph order. It then traverses only stored topological graph order and splits on unequal owner
identity values. Each maximal consecutive equal-owner run retains exact graph `NodeId` references
and the first exact owner reference in that run. Graph inputs/outputs, phase changes, fan-out,
merge, and multi-output nodes do not create extra splits. A zero-node graph produces an empty
immutable list. The two partition tests cover DTO and generator rules.

## Logical-memory recipe and derivation assessment

`LogicalMemoryRequirement` records one exact value identity, exact logical descriptor, optional
exact producing partition, ordered distinct exact consuming partitions, and graph-output flag.
`LogicalMemoryPlan` is the immutable ordered aggregate and rejects duplicate value identities.

`LogicalMemoryPlanning.plan` validates all top-level and partition elements, unknown/duplicate
node membership, complete graph coverage, flattened graph order, and adjacent equal-owner
non-maximal partitions before derivation. It emits exactly one requirement per graph value in
graph-value order. Producer association comes from the node that outputs the value; consumers are
distinct partitions in partition order; graph membership supplies output preservation. Exact
graph `ValueId`/`TensorDescriptor` and supplied `PlannedPartition` references are retained.
Dynamic and expression shapes remain representable through the descriptor; no byte count is
forced. A zero-node graph passes graph inputs/outputs through as producerless requirements. The
three logical-memory tests lock these DTO and derivation rules.

## Public DTO and package-private operation assessment

The exact public surface is:

1. `BackendCapabilityProvider` — the cross-package backend-to-planning capability collaboration.
2. `OperationCapabilityQuery` — the immutable occurrence question exposed to a provider.
3. `PlannedPartition` — the immutable partition recipe for downstream compile artifacts.
4. `LogicalMemoryRequirement` — the immutable per-value logical requirement recipe.
5. `LogicalMemoryPlan` — the ordered aggregate for downstream compile artifacts.

These five declarations are necessary, sufficient, and minimal for the selected milestone. The
provider/query pair is the only current inward collaboration. The three recipes are retained
cross-package outputs. No public capability matrix, eligibility result, owner-selection result,
owner map, planning facade, device choice, cost profile, transfer plan, or physical-memory DTO is
needed now.

The package-private declarations are `BackendEligibility`, `BackendOwnerSelection`,
`MaximalSameOwnerPartitioning`, and `LogicalMemoryPlanning`. Their operations
`BackendEligibility.evaluate`, `BackendOwnerSelection.select`,
`MaximalSameOwnerPartitioning.partition`, and `LogicalMemoryPlanning.plan` may remain internal.
They form complete tested planning semantics, but no external consumer currently needs an
isolated stage. Publishing them now would guess the future compiler collaboration and expose
intermediate mechanics rather than a consumer-driven boundary. A later compiler integration may
add one narrow public collaboration without changing these audited semantics.

## Validation, equality, ordering, identity, and reference matrix

| Stage | Validation and failure order | Equality and association | Output order | Exact references retained |
|---|---|---|---|---|
| Capability query | Top-level operation/lists, list elements, then signature counts | Operation signature occurrence counts; no descriptor compatibility decision | Input/output encounter order | Operation and every descriptor |
| Hard eligibility | Top-level inputs; complete provider scan; complete snapshot scan; both association directions; filtering; capability calls | Equal `BackendId` associates provider/snapshot and evaluates exact requirements | Provider order | Eligible provider identity |
| Owner selection | Top-level inputs; empty eligibility before snapshot elements; complete snapshots; required associations | Equal `BackendId` associates snapshots; equal `DeviceClass` tests preference | Eligible/provider order | Selected eligibility identity |
| Partition recipe | Owner; node list; elements; nonempty and distinct nodes | Record equality is value-based | Supplied node order | Owner and node identities |
| Partition generation | Graph/map; null keys; unknown keys; missing keys; null owners; then generation | Equal owner values continue a run | Stored graph topological order | Graph node identities and first owner per run |
| Logical requirement | Value; descriptor; producer optional; consumers and distinctness | Record/identifier equality validates uniqueness | Supplied consumer order | Value, descriptor, producer, consumers |
| Logical plan | Requirement list; elements; distinct value identities | Equal value identities reject duplicates | Supplied requirement order | Requirement elements |
| Logical derivation | Graph/partitions; elements; membership; coverage; graph order; maximality; then derivation | Equal node/value/owner identities associate facts | Graph-value order; consumers in partition order | Graph value/descriptor and supplied partitions |

Immutable snapshots copy list membership while preserving documented element references. Equality
is used for association and duplicate detection; it does not weaken exact-reference retention in
the results that promise it.

## Package and dependency assessment

The package map is cohesive: `capability` owns questions, providers, eligibility, and baseline
selection; `partition` owns the public partition recipe and consecutive-run generation; `memory`
owns logical requirements, plan, and derivation. There is no root facade, vague utility package,
compiler adapter, or live-provider retention in an output DTO.

`modules/planning/build.gradle.kts` has exactly:

```kotlin
api(project(":modules:model"))
implementation(project(":modules:config"))
api(project(":modules:backend-contract"))
implementation(project(":modules:trace"))
```

Model and backend-contract are public dependencies because public signatures expose their types.
Config is internal to selection input and trace is an allowed internal dependency without a
current emitted schema. This exact scan returned no forbidden production dependency:

```bash
rg -n 'io\.github\.pho001\.synaptik\.(runtime|prepare|engine)|io\.github\.pho001\.synaptik\.backends?' \
  modules/planning/src/main/java modules/planning/build.gradle.kts
```

There is no runtime, prepare, engine, concrete-backend, compiler, or tools dependency and no
architecture-test change is needed because no dependency boundary changed.

## Documentation, examples, glossary, and planning-history consistency

The public API, compile API, partition-scoring architecture explanation, memory-planning note,
backend-selection guide, and glossary correctly distinguish current recipes and internal rules
from planned compiler orchestration and physical realization. Provider and backend-selection
examples use current public contracts and make no execution claim. Completed Planning 0001-0005
task evidence remains unchanged.

One lifecycle diagram in `docs/backend-guide/capability-provider.md` still called partitioning
planned after Planning 0004 and omitted current Planning 0005 logical-memory derivation. This
documentation drift was corrected within the permitted scope. The correction describes current
internal stages and keeps public orchestration and cost-bearing scoring planned; it changes no
contract or example.

Task 0006, the planning/config/trace current-status narratives, and the roadmap are synchronized
with closure. Config 0004+, Trace 0003+, compiler, prepare, runtime, and every later task remain
Draft without a new detailed specification or Ready status.

## Compiler, CompileArtifacts, publication, and prepare handoff

The future compiler consumer owns graph-wide sequencing of the current planning rules, assembly
of the complete per-node owner map, translation of internal terminal failures into public compile
failures, diagnostics, and construction of immutable `CompileArtifacts`. Planning contributes
backend-neutral rules plus immutable partition and logical-memory recipes; it does not own the
aggregate compile lifecycle.

`PublicationBinding` remains standalone model data and is not an input to logical-memory
derivation. Compiler will own a future `PublicationPlan` that combines graph membership,
tensor-to-value publication context, policy, and diagnostics. The current graph-output flag is
only the logical preservation obligation.

Prepare will consume immutable compile artifacts through the future shared prepare boundary. It
does not need planning to predict lowering, physical allocation, transfers, or executable shape
now. This is a stable handoff boundary, not a claim that `CompileArtifacts`, publication planning,
or prepare implementation currently exists.

## Deferred and downstream-owned work

The following work is deliberately deferred and does not block the selected planning milestone:

- compiler orchestration, a possible narrow planning collaboration, `CompileArtifacts`, public
  compile failures, diagnostics, and `PublicationPlan`;
- Config 0004 and any backend-neutral cost/workload classification or numeric scoring input;
- trace compile payloads, emission, correlation, rejection taxonomy, and serialization;
- prepare-layer artifact validation and shared transition to prepared state;
- backend-owned lowering, fusion, specialization, route/kernel selection, transfers, physical
  memory, allocation, executable construction, and device realization;
- runtime residency, invocation state, execution, and publication delivery;
- engine composition, concrete provider implementations, benchmarks, and model autotuning.

Each item depends on a future concrete consumer or lifecycle owner. None supplies a missing rule
needed to interpret the five current public declarations or four current internal operations.

## Findings, severity, and disposition

| Area | Current contract | Primary evidence | Required invariant or question | Finding label | Disposition |
|---|---|---|---|---|---|
| Capability query/provider | Immutable occurrence query and stable backend-level boolean provider | `OperationCapabilityQuery`, `BackendCapabilityProvider`, `BackendCapabilityContractTest` | No discovery, availability, scoring, routing, prepare, or execution | `NO_CHANGE_CONFIRMED` | Current boundary is complete. |
| Hard eligibility | Complete provider/snapshot association and provider-order immutable eligible identities | `BackendEligibility`, `BackendEligibilityTest` | Validate before calls; exact requirements filter; no device selection | `NO_CHANGE_CONFIRMED` | Internal semantic is complete. |
| Baseline owner selection | Preferred-class-first, provider-order deterministic selection | `BackendOwnerSelection`, `BackendOwnerSelectionTest` | Complete candidate set; terminal empty input; no numeric cost | `NO_CHANGE_CONFIRMED` | Cost-free baseline closes current policy. |
| Maximal same-owner partitioning | Consecutive equal-owner runs in stored graph order | `MaximalSameOwnerPartitioning`, both partition tests | Complete map, maximality, exact graph references, zero-node result | `NO_CHANGE_CONFIRMED` | Current recipe/generator are complete. |
| Logical-memory derivation | One exact logical requirement per graph value | `LogicalMemoryPlanning`, three memory tests | Coverage/maximality before derivation; producer/consumer/output facts; dynamic shapes | `NO_CHANGE_CONFIRMED` | Current logical plan is complete. |
| Public DTO minimality | Exactly five public planning declarations | Declaration scan in section 2; public type tests | Are five declarations sufficient and minimal? Yes. | `NO_CHANGE_CONFIRMED` | Reject premature matrices, maps, facade, cost, device, transfer, and physical DTOs. |
| Package-private operation visibility | Four internal evaluator/generator operations | Source declarations, reflection/public-shape tests, generated Javadocs | May all four remain internal? Yes, until a concrete compiler consumer defines one narrow collaboration. | `NON_BLOCKING_DEFERRED` | Consumer-driven visibility decision belongs with compiler integration. |
| Validation and failure ordering | Each stage validates complete structural inputs before semantic calls or derivation | All eight focused test suites | Preserve deterministic first failure and avoid partial provider calls/results | `NO_CHANGE_CONFIRMED` | Source/tests agree. |
| Equality association and exact references | Value equality associates identities; documented results retain exact references | All production records/generators and identity assertions in focused tests | Association must not erase promised object identity | `NO_CHANGE_CONFIRMED` | Source/tests agree. |
| Package/dependency surface | Three cohesive packages; model/backend-contract public, config/trace internal | `modules/planning/build.gradle.kts`; forbidden import scan in section 12 | No outward/downward dependency or concrete-backend coupling | `NO_CHANGE_CONFIRMED` | Exact dependency surface is architecture-compliant. |
| Generated Javadocs/public indexes | Three package summaries and five public type pages only | `:modules:planning:javadoc`; page/index inspection in sections 4 and 17 | Four internal types absent from pages and public indexes | `NO_CHANGE_CONFIRMED` | Rendering and visibility are correct. |
| API/guide examples/glossary | Current/internal/planned boundaries generally match source | Reviewed API, guides, design note, glossary; capability lifecycle diagram | One stale planned-partitioning phrase required correction | `DOCUMENTATION_DRIFT` | Corrected `docs/backend-guide/capability-provider.md`. |
| Completed history and status | Planning 0001-0005 Complete; 0006 closes selected milestone | Completed tasks, master plans, roadmap, status scans | Preserve history and create no later spec/Ready task | `NO_CHANGE_CONFIRMED` | Only closure evidence/status synchronized. |
| Compiler orchestration and `CompileArtifacts` | Future compiler owns graph-wide invocation and aggregate artifacts | `ARCHITECTURE.md`, compiler master plan, compile API | Does absent orchestration leave a planning semantic gap? No. | `NON_BLOCKING_DEFERRED` | Future concrete compiler consumer owns integration. |
| Publication-plan boundary | Graph-output flag is logical preservation; publication binding stays outside planning | `CompiledGraphModel`, `PublicationBinding`, memory source/tests, compile API | Do not infer tensor publication policy in planning | `NON_BLOCKING_DEFERRED` | Future compiler-owned `PublicationPlan`. |
| Prepare/runtime/backend physical realization | Planning outputs no physical or executable state | Architecture lifecycle/boundary docs; planning DTO/source scan | Physical memory, transfers, lowering, routes, residency, execution stay downstream | `NON_BLOCKING_DEFERRED` | Prepare, concrete backends, and runtime retain ownership. |
| Cost-bearing scoring and Config 0004 | Current baseline is complete without numeric cost | Owner-selection source/tests; config/planning master plans | No speculative workload family, units, or profile | `NON_BLOCKING_DEFERRED` | Remains Draft until a concrete cost-bearing consumer exists. |
| Trace and diagnostics | No current planning trace payload or structured failure taxonomy | Planning source scan; trace master plan | Do not invent producer schema before compiler orchestration | `NON_BLOCKING_DEFERRED` | Trace 0003+ remain Draft. |

There are no unresolved planning-owned gaps and no architecture conflict. The single corrected
documentation drift and the explicitly owned downstream integrations are compatible with the
closure verdict.

## Checkpoint evidence and planning-milestone decision

The final checkpoint command is:

```bash
./gradlew test :testing:architecture-tests:test :modules:planning:javadoc
```

The final documentation and scope commands are:

```bash
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
rg --files docs/planning/modules/planning/tasks | sort
rg -n '^Ready$|\| Ready \|' docs/planning
```

The combined checkpoint passed. Gradle reported 47 actionable tasks: 2 executed, 45 up-to-date,
and 0 skipped. XML reports contain 1,137 tests across 149 repository suites, including 63 tests
across all eight planning suites and three tests across all three architecture suites. Every
reported suite has zero failures, errors, or skips.

Post-checkpoint inspection confirmed all three package summaries and five public pages. Both
public indexes list exactly the five public planning declarations, and none of the four internal
operation types has a generated page or index entry. Final
`python3 /tmp/validate_synaptik_markdown.py` passed for 231 Markdown files, 4,134 local links, 251
local anchors, 2,916 fence markers, final newlines, and trailing whitespace. Final
`git diff --check` passed with no output. The exact path scan returned 21 paths: the preserved
Planning 0005 change, the Planning 0006 specification/status work, this required artifact, and one
permitted capability-provider guide correction; this audit added no executable path.

The combined documentation shell reported exit code 1 only because its last exact command,
`rg -n '^Ready$|\| Ready \|' docs/planning`, correctly returned no match. No preceding validation
command failed. The task-directory scan contains only Planning 0001-0006, and no later task
specification exists.

The evidence supports completing task 0006 and the selected planning milestone. The roadmap
planning area becomes `Complete`. No later detailed task specification is created and no task is
made Ready. A separate frontier reassessment may choose the next global task after this closure;
this audit does not select it.
