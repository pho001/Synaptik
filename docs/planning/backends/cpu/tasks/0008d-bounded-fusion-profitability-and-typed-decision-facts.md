# Task 0008D: Bounded [Fusion](../../../../glossary.md#term-fusion) Profitability and Typed Decision Facts

## Status

Complete

## Goal

Add one cohesive [CPU](../../../../glossary.md#term-cpu)-private cold-analysis capability that enumerates the complete admitted set of
legal [pointwise](../../../../glossary.md#term-pointwise) [fused](../../../../glossary.md#specialization-and-fusion) and materialized-split [partition](../../../../glossary.md#partition) candidates, ranks every legal complete
candidate with deterministic bounded no-measurement heuristics, and retains closed typed legality,
profitability, and selection facts.

The exact CPU 0008B plan remains a mandatory reproducible comparison candidate. CPU 0008C
recognition facts, dispositions, associated baseline snapshots, and [artifact](../../../../glossary.md#term-artifact)-[identity](../../../../glossary.md#term-identity-meanings) exclusion
remain unchanged. Legality rejection is distinct from profitability rejection. Tie, uncertainty,
or incomplete bounded enumeration selects the [canonical](../../../../glossary.md#term-canonical-form) safe split. Selection finishes during CPU
analysis; [Runtime](../../../../glossary.md#term-runtime) receives only the selected prepared work.

## [Scope](../../../../glossary.md#term-scope)

### Closed typed facts and reproducible [identities](../../../../glossary.md#term-identity-meanings)

- Add [sealed](../../../../glossary.md#term-sealed) [immutable](../../../../glossary.md#term-immutable) [CPU](../../../../glossary.md#term-cpu)-private `CpuFusionDecision` with exactly `LegalCandidate`,
  `LegalityRejection`, `ProfitabilityRejection`, and `Selection` variants. Nested records/enums may
  carry common facts. Collections are defensive immutable ordered snapshots.<span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry))</span>
- `LegalCandidate` records one stable `CandidateIdentity`, complete topology/resource/hard-budget
  facts, profitability inputs, integer score, and canonical-split/0008B-baseline roles.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- `LegalityRejection` records the source topology, attempted pair, exact failed hard fact, and one
  closed reason: `SEMANTIC_BARRIER`, `PUBLICATION_BARRIER`, `FAN_OUT_BARRIER`,
  `STATE_OR_RANDOM_BARRIER`, `NUMERICAL_ORDER_BARRIER`, `ALIAS_OR_ACCESS_UNPROVED`,
  `DEPENDENCY_CYCLE`, `UNSUPPORTED_LOWERING`, `ROUTE_INELIGIBLE`, or
  `HARD_BUDGET_EXCEEDED`.
- `ProfitabilityRejection` refers only to a retained legal complete candidate and uses one closed
  reason: `INSUFFICIENT_MARGIN`, `CODE_SIZE_PRESSURE`, `LIVE_VALUE_PRESSURE`,
  `MATERIALIZATION_COST`, `SAFE_SPLIT_TIE`, or `UNCERTAIN_INPUT`.
- `Selection` records selected, canonical-split, and 0008B-baseline identities; stable rank;
  score/margin facts; and <span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>`PROFITABLE_FUSION`, `CANONICAL_SPLIT`, `TIE_FALLBACK`,
  `UNCERTAINTY_FALLBACK`, or `ENUMERATION_BUDGET_FALLBACK`.
- `CandidateIdentity` contains, in stable unit order, relative member-node positions, dependency
  indices, portable IR structural keys, specializations, strategies, access regimes, relative
  materialized-boundary roles/geometry, workspace facts, and fused/split topology. It excludes
  node/value/partition/requirement/slot identity, carrier objects, addresses, workers, runs,
  loaders, generated classes, artifact roots, timing, cache contents, and map iteration order.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [dependency glossary](../../../../glossary.md#term-dependency); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [fused glossary](../../../../glossary.md#specialization-and-fusion); [partition glossary](../../../../glossary.md#partition); [identity glossary](../../../../glossary.md#term-identity-meanings); [carrier glossary](../../../../glossary.md#term-storage-carrier); [artifact glossary](../../../../glossary.md#term-artifact); [cache glossary](../../../../glossary.md#term-cache))</span>
- Identity and tie-breaking use typed values and stable relative positions, never strings, enum
  text, object identity, or <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span>[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) magnitude. No maps, reflection, annotations, callbacks,
  registries, DSLs, or generic parameter bags are permitted.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>

### Complete bounded [fused](../../../../glossary.md#specialization-and-fusion)/split enumeration

- Narrowly refactor `CpuPartitionDagDecomposer` to expose the unchanged maximally split seeds, the
  exact unchanged CPU 0008B result, and every distinct complete topology reachable through zero or
  more existing 0008B vertical/horizontal ordinary-pointwise contractions within the budgets
  below. Its existing <span data-glossary-navigation="true"> ([seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [CPU glossary](../../../../glossary.md#term-cpu); [topology glossary](../../../../glossary.md#term-topology); [pointwise glossary](../../../../glossary.md#term-pointwise))</span>`decompose(...)` result remains exactly compatible with pre-0008D behavior.
- Enumerate breadth-first from canonical split. For each source topology, test vertical pairs in
  producer/member order, then horizontal pairs in stable left/right order. Deduplicate resulting
  candidates by typed identity and record each distinct source/pair attempt once.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [producer glossary](../../../../glossary.md#term-producer); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Rank only complete candidates that cover every node/publication exactly once and contain every
  exact resource. Partial contraction paths or local kernels are never candidates.<span data-glossary-navigation="true"> ([Rank glossary](../../../../glossary.md#term-rank); [node glossary](../../../../glossary.md#node); [publication glossary](../../../../glossary.md#term-publication); [contraction glossary](../../../../glossary.md#contraction-dimension); [kernels glossary](../../../../glossary.md#kernel))</span>
- Preserve every 0008B legality rule and hard budget. A failed contraction emits a typed legality
  rejection and leaves its source topology valid. Failure of the canonical split fails analysis
  before declarations or decision facts.
- CPU 0008C recognition is a fixed barrier: enumeration may not split, merge, reorder, or otherwise
  change a retained fact's members, at-most-two associated baseline units, disposition, or exact
  baseline snapshot. Do not rerun a looser pattern or infer equivalent mathematics.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Specialized, affine, state/random, reduction/scan, ordering, scatter/fold, softmax,
  normalization, batch-normalization, Conv1d/2d/3d, and multi-output units remain indivisible.
  Only already implemented ordinary pointwise IR contractions vary.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [normalization glossary](../../../../glossary.md#term-normalization); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>

### Exact budgets and reachable seams

| Budget | Ceiling | On exceedance |
| --- | ---: | --- |
| [Partition](../../../../glossary.md#partition) [nodes](../../../../glossary.md#node) / [seed](../../../../glossary.md#term-seed) or final units | 8 / 8 | Existing [fail-closed](../../../../glossary.md#term-fail-closed) baseline rule |
| Distinct complete candidate topologies | 64 | [Canonical](../../../../glossary.md#term-canonical-form) split; no partial [ranking](../../../../glossary.md#term-ranking) |
| Distinct source-[topology](../../../../glossary.md#term-topology) pair attempts | 256 | Canonical split; no partial ranking |
| Legal candidates ranked | 64 | Canonical split; no partial ranking |
| Retained legality rejections | 256 | Canonical split; no partial ranking |
| Retained profitability rejections | 63 | Dominated by 64 legal candidates |
| Total decision facts | 384 | Reject inconsistent construction |
| Materialized boundary positions across one candidate | 64 | Candidate illegal; per-unit cap remains 16 |
| Workspace facts across one candidate | 8 | Candidate illegal; at most one per unit |

- If a queue or untested pair remains at the 64/256 ceiling, selection is
  `ENUMERATION_BUDGET_FALLBACK`. Retained partial facts may be inspected but cannot affect choice.
- Tests reach 1/8 units, 1/64 candidates, 255/256 attempts with a pending 257th, 63 profitability
  rejections, 384 total facts, 64 candidate boundary positions, and 8 workspace facts.
- Use real supported fixtures for canonical-split-equals-baseline, a two-candidate vertical case,
  a horizontal multi-store case, and a diamond with typed fan-out rejection. Use one narrow
  package-private immutable typed-topology seam for dominated 65th-candidate/257th-attempt cases;
  it accepts no <span data-glossary-navigation="true"> ([fan-out glossary](../../../../glossary.md#term-fan-out); [immutable glossary](../../../../glossary.md#term-immutable))</span>[`Operation`](../../../../glossary.md#operation), invents no generated form, and never enters production preparation.

### Exact default profitability [heuristic](../../../../glossary.md#term-heuristic)

- Use only checked integer facts already available during analysis; read no clock, benchmark,
  profile, trace, cache, environment policy, global mutable state, or Runtime state.<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace); [cache glossary](../../../../glossary.md#term-cache); [mutable glossary](../../../../glossary.md#term-mutable); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Compute for each legal candidate:

  ```text
  unitCost = 64 * finalUnitCount
  materializationCost = sum over cross-unit materialized boundaries:
      16 + min(4096, ceilDiv(referencedBytes, 4096))
  structuralCost = sum over pointwise units:
      generatedCodeSizeUnits + indexingComplexityUnits
      + 8 * max(0, simultaneouslyLiveValues - 8)
  familyCost = 32 * indivisibleNonPointwiseUnitCount
  totalScore = unitCost + materializationCost + structuralCost + familyCost
  ```

- `referencedBytes` is checked referenced element span times data-type width. Code-size, indexing,
  and liveness use the exact 0008B definitions through one shared package-private calculation;
  do not duplicate formulas.<span data-glossary-navigation="true"> ([referenced element span glossary](../../../../glossary.md#referenced-element-span); [liveness glossary](../../../../glossary.md#term-liveness))</span>
- Missing geometry/type width, inconsistent roles, division error, or checked [overflow](../../../../glossary.md#term-overflow) produces
  `UNCERTAIN_INPUT`; never saturate or treat unknown as favorable.
- A fused candidate is profitable over canonical split only if its score is at least 32 lower,
  every pointwise unit is at most 48 generated-code-size units and 12 live values, total
  materialized referenced bytes do not increase, and route/numerical/determinism/workspace/0008C
  baseline facts remain exact.<span data-glossary-navigation="true"> ([fused glossary](../../../../glossary.md#specialization-and-fusion); [pointwise glossary](../../../../glossary.md#term-pointwise))</span>
- Rank qualifying candidates by score, materialized bytes, maximum code-size units, and maximum
  live values. Exact ties prefer more units then stable identity. If the tied best would be more
  fused than canonical split, record <span data-glossary-navigation="true"> ([Rank glossary](../../../../glossary.md#term-rank); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>`SAFE_SPLIT_TIE` and select [canonical](../../../../glossary.md#term-canonical-form) split.
- Uncertain canonical split, any incomparable legal candidate, or incomplete enumeration selects
  canonical split. The exact 0008B baseline remains a marked candidate and may win, lose, or equal
  split; it is never mutated or omitted.

### Preparation, [ownership](../../../../glossary.md#term-ownership-meanings), and generated-code discipline

- `CpuFusionProfitabilitySelector` owns stateless enumeration/ranking. CPU analysis completes it
  before deriving the selected resources once through existing 0008B assembly.<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span><span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Extend `CpuPartitionPreparationPlan` with immutable decision facts. A trivial one-unit plan has
  equal split/baseline/selected identities. Validation recomputes consistency and proves selected
  identity exactly matches retained units/resources; all rejected facts refer to the right legal
  or attempted candidate.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [Validation glossary](../../../../glossary.md#term-validation); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Finalization realizes only already selected artifacts after assignment and cannot rerank or add
  resources. Runtime, shared Prepare, Planning, Compiler, Model, Config, Trace, Engine, Backend
  Contract, and other backends gain no type or interpretation responsibility.<span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Planning glossary](../../../../glossary.md#term-planning); [Compiler glossary](../../../../glossary.md#term-compiler); [Model glossary](../../../../glossary.md#term-model); [Trace glossary](../../../../glossary.md#trace); [Engine glossary](../../../../glossary.md#term-engine); [Backend glossary](../../../../glossary.md#backend))</span>
- Decision facts remain CPU-private cold metadata, outside generated artifact/resource/cache
  identity and Runtime. Public Trace translation, tuning inspection, opaque shared candidate
  handoff, persistence, measurement, cache use, and serialization are deferred.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [cache glossary](../../../../glossary.md#term-cache); [tuning inspection glossary](../../../../glossary.md#tuning-inspection); [opaque glossary](../../../../glossary.md#term-opaque); [handoff glossary](../../../../glossary.md#term-handoff); [persistence glossary](../../../../glossary.md#term-persistence); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- Select only existing IR, specialization, entry shape, route, algorithm, and executable forms.
  Prove exact structural-key/class-byte identity and absence of decision/recognition/selector
  references in Class-File, artifact, finalization handle, and hot invocation evidence.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [specialization glossary](../../../../glossary.md#term-specialization); [shape glossary](../../../../glossary.md#shape))</span>
- Reuse owning CPU 0005-series/0008B optimal-clean-Java performance evidence when identity and hot
  code are unchanged. Run no redundant timing and make no new speedup claim.<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>
- Any new emitter branch, IR form, entry descriptor, helper call, algorithm, route, artifact key/
  schema field, composite hot behavior, or generated/hot form is out of scope. Stop and replan it
  with an optimal clean-Java oracle, complete Class-File/hidden-helper inspection, and fresh
  five-fork generated/direct <span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [schema glossary](../../../../glossary.md#term-schema); [scope glossary](../../../../glossary.md#term-scope); [oracle glossary](../../../../glossary.md#term-test-oracle); [fork glossary](../../../../glossary.md#term-fork-meanings))</span>`<= 1.15x` evidence.

### Documentation

- After stable implementation/Java evidence, use a separate clean documentation-focused context.
  Finalize detailed Javadocs and update the CPU guide with legality versus profitability, exact
  heuristics, typed ownership, fallbacks, and no-measurement/no-Runtime-selection boundaries.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Review the glossary and change it only for a reusable terminology change; otherwise record a
  reasoned no-change conclusion. Synchronize this task, CPU master plan, and roadmap.
- [CPU](../../../../glossary.md#term-cpu) 0008D becomes `Review needed` after implementation [validation](../../../../glossary.md#term-validation) and `Complete` only after the
  documentation gate. On completion CPU 0008E becomes the sole [`Ready`](../../../../glossary.md#term-ready) row; create no 0008E
  specification.

## Out of [scope](../../../../glossary.md#term-scope)

- New semantics, algorithms, family lowerers, emitters, generated forms, routes, strategies,
  capabilities, artifact schema/key fields, hot composite work, or performance claims.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [schema glossary](../../../../glossary.md#term-schema))</span>
- Any redefinition/broadening of 0008C recognition, family inventory, precedence, disposition,
  explicit-semantic-only rule, or baseline snapshots.
- CPU 0008E materialization/representation expansion; MATMUL (0008F); pooling, attention, or loss
  work (0008G–0008I).<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [materialization glossary](../../../../glossary.md#materialization); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [pooling glossary](../../../../glossary.md#term-pooling); [loss glossary](../../../../glossary.md#loss))</span>
- Measurement, benchmark-selected behavior, autotuning, tuning-cache use/mutation, persistence,
  public configuration, target discovery, Trace payloads, public/generic candidate APIs,
  registries, DSLs, callbacks, reflection, string dispatch, maps, or serialization.<span data-glossary-navigation="true"> ([autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [cache glossary](../../../../glossary.md#term-cache); [persistence glossary](../../../../glossary.md#term-persistence); [target glossary](../../../../glossary.md#term-target); [Trace glossary](../../../../glossary.md#trace); [dispatch glossary](../../../../glossary.md#term-dispatch); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- Compiler/Planning/shared-Prepare/Runtime/Engine changes, another module/backend, public API,
  architecture/dependency/Gradle changes, conformance/integration tests, or generated evidence.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [module glossary](../../../../glossary.md#term-module); [backend glossary](../../../../glossary.md#backend); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md): backend-owned fusion/selection, staged
  resources, generated-code discipline, tuning separation, and Runtime boundary.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [fusion glossary](../../../../glossary.md#term-fusion); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md).
- [`performance evidence and model autotuning`](../../../../architecture/performance-evidence-and-tuning.md).
- [`ADR 0008`](../../../../design/decisions/0008-performance-evidence-and-tuning-boundaries.md).
- [`planning guide`](../../../planning-guide.md).
- [`CPU master plan`](../master-plan.md).
- Complete [`CPU 0008B`](0008b-general-partition-dag-computation-unit-decomposition-and-bounded-fusion.md)
  and [`CPU 0008C`](0008c-typed-specialized-subgraph-and-epilogue-recognition.md).

## Architecture constraints

- Planning selects CPU ownership; CPU analysis alone enumerates, proves legality, ranks, selects,
  and declares exact resources before shared assignment.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [CPU glossary](../../../../glossary.md#term-cpu); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Legality precedes profitability and cannot be weakened by cost. Analysis is deterministic from
  explicit facts and performs no measurement/cache lookup.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- 0008C recognition/baseline snapshots remain immutable inputs. Decision facts stay outside
  generated identity, shared contracts, and Runtime hot work.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [identity glossary](../../../../glossary.md#term-identity-meanings); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Finalization realizes existing selected forms only. Any need for a public/shared type, new
  generated/hot form, schema/route/capability/module/dependency/architecture change is a stop-and-
  replan condition.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.ir` — closed facts and typed [identities](../../../../glossary.md#term-identity-meanings).
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — enumeration, shared hard facts,
  heuristic ranking, and selection.<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span><span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — selected plan/resources and [validation](../../../../glossary.md#term-validation).

Packages added or changed:

- None.

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision` — [sealed](../../../../glossary.md#term-sealed) facts and [identity](../../../../glossary.md#term-identity-meanings).
- `io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuFusionProfitabilitySelector` — sole
  bounded selector.

Tests mirror production packages. No public, tuning, [trace](../../../../glossary.md#trace), [registry](../../../../glossary.md#term-registry), [cache](../../../../glossary.md#term-cache), or new source-set
package is added.

## Affected files

Expected production/[Javadoc](../../../../glossary.md#term-javadoc) paths:

- new `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuFusionDecision.java`;
- new `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuFusionProfitabilitySelector.java`;
- `CpuPartitionDagDecomposer.java` in that [lowering](../../../../glossary.md#lowering) package;
- `CpuPartitionPreparationPlan.java` and `CpuPartitionPreparer.java` in `internal.prepare`.

Expected test paths:

- new `CpuFusionDecisionTest` and `CpuFusionProfitabilitySelectorTest`;
- existing decomposer, plan/preparer, specialized-recognizer, and artifact-identity or package-
  inventory tests only as directly required.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

Expected completion documentation paths are [CPU](../../../../glossary.md#term-cpu) guide, glossary only if justified, this task, CPU
master plan, and roadmap. No other Java/test/generated/architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/[Gradle](../../../../glossary.md#term-gradle)/[conformance](../../../../glossary.md#term-conformance)/
integration/later-task path is expected.

## Maximum [scope](../../../../glossary.md#term-scope)

- At most 5 [CPU](../../../../glossary.md#term-cpu) production/[Javadoc](../../../../glossary.md#term-javadoc) paths, including exactly 2 new production types.
- At most 6 CPU test paths, including at most 2 new test types.
- At most 5 documentation/[planning](../../../../glossary.md#term-planning) paths, including this task and conditional glossary.
- At most 16 total paths and 384 decision facts per admitted [partition](../../../../glossary.md#partition).

If another [owner](../../../../glossary.md#term-owner), package, form, path, type, or ceiling is required, stop and replan.

## Acceptance criteria

- Closed typed immutable facts distinguish legality, profitability, and selection with exact
  identities/reasons and no forbidden generic/public mechanism.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Canonical split and exact 0008B baseline are always present for supported work; compatibility
  decomposition and 0008C facts/baseline snapshots remain exact.<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- Every admitted complete topology is ranked or incomplete enumeration selects canonical split.
  No partial candidate set controls selection.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology))</span>
- All 0008B barriers/budgets remain hard and each typed legality reason is distinct from
  profitability.
- Exact score constants/formulas, 32-point margin, 48/12 soft ceilings, no-increased-
  materialization rule, ranks, tie behavior, and uncertainty fallback pass at/one-beyond seams.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [materialization glossary](../../../../glossary.md#materialization))</span>
- Tests reach 64/over-64 candidates, 256/pending-257 attempts, 63 profitability rejections, 384
  facts, 8 units/workspaces, and 64 boundary positions via real fixtures or the bounded seam.
- Repeated analysis and randomized map insertion produce identical ordered facts, choice,
  resources, structural keys, and class bytes.
- Plan validation rejects missing/duplicate/mismatched facts, identities, ranks, scores, reasons,
  baseline/split roles, or selected units/resources.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Ties, unknowns, overflow, incomparable candidates, and incomplete enumeration select canonical
  split without losing nodes, publications, edges, workspaces, or execution.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [nodes glossary](../../../../glossary.md#node))</span>
- Exact class [identity](../../../../glossary.md#term-identity-meanings) and no-leak scans pass; no new timing/performance claim is made.
- Final CPU tests, Javadoc, guide/glossary review, planning synchronization, scope/status/schema-
  52/empty-index/whitespace checks pass in the required implementation/documentation contexts.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning); [synchronization glossary](../../../../glossary.md#term-synchronization); [scope glossary](../../../../glossary.md#term-scope); [schema glossary](../../../../glossary.md#term-schema))</span>
- No public/shared/build/architecture/[conformance](../../../../glossary.md#term-conformance)/integration/0008E-specification change occurs.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused tests cover facts, [identity](../../../../glossary.md#term-identity-meanings), enumeration/deduplication, every reason, score arithmetic,
thresholds/budgets/fallbacks, 0008B/0008C preservation, plan/resources/finalization, and forbidden
leakage. After the last executable change [run](../../../../glossary.md#term-run) exactly one authoritative:

```bash
./gradlew :backends:cpu:test --rerun-tasks
```

Regenerate representative one-unit, vertical, horizontal multi-store, [Conv2d](../../../../glossary.md#convolution--conv2d), reduction-split,
and explicit-semantic existing classes; compare keys/checksums and inspect `javap -c -v -p` plus
member references. Reuse owning performance evidence.

The clean documentation context runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short -uall
```

It also validates local links/anchors, [canonical](../../../../glossary.md#term-canonical-form) headings, fences, final newlines, rendered
[Javadoc](../../../../glossary.md#term-javadoc), terminology, path/type/fact ceilings, [schema](../../../../glossary.md#term-schema) 52, 0008C preservation, 0008D/0008E status,
absence of a 0008E file, and empty staging. Repository/architecture/[conformance](../../../../glossary.md#term-conformance)/integration suites
remain [CPU](../../../../glossary.md#term-cpu) 0009/[CI](../../../../glossary.md#term-continuous-integration-ci) unless a forbidden shared change requires replanning.

## Dependencies

- Complete [CPU](../../../../glossary.md#term-cpu) 0008B and CPU 0008C.
- Complete CPU 0005A–0005J and CPU 0008 existing [IR](../../../../glossary.md#term-intermediate-representation-ir)/generation/evidence contracts.
- Existing [Planning](../../../../glossary.md#term-planning), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), Config, [Trace](../../../../glossary.md#trace), [cache](../../../../glossary.md#term-cache)/[artifact](../../../../glossary.md#term-artifact), and tuning boundaries.

No architecture or shared-contract blocker is known.

## Follow-up tasks

- CPU 0008E becomes Ready after this task completes: at-most-two external read materializations
  and representation reuse; create no specification now.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- [CPU](../../../../glossary.md#term-cpu) 0008F–0008I retain [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), [pooling](../../../../glossary.md#term-pooling), attention, and [loss](../../../../glossary.md#loss) [ownership](../../../../glossary.md#term-ownership-meanings).
- Config 0006A, Prepare 0004, CPU 0016, and Tuning 0001–0002 retain future explicit tuning inputs,
  opaque handoff, compatible cache consumption, and bounded measurement.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [opaque glossary](../../../../glossary.md#term-opaque); [handoff glossary](../../../../glossary.md#term-handoff); [cache glossary](../../../../glossary.md#term-cache))</span>
- CPU 0009 retains coverage/[conformance](../../../../glossary.md#term-conformance)/integration closure.

## Architecture impact

Expected impact: None.

No-change conclusions: architecture/ADRs/tests already authorize [backend](../../../../glossary.md#backend)-private candidates and
safe heuristics; [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler) semantics and [graph](../../../../glossary.md#graph) stay unchanged; [Planning](../../../../glossary.md#term-planning) [ownership](../../../../glossary.md#term-ownership-meanings)/Config stay
unchanged; shared [Prepare](../../../../glossary.md#prepare) sees selected declarations only; [Runtime](../../../../glossary.md#term-runtime)/[Engine](../../../../glossary.md#term-engine) execute unchanged
prepared work; [Trace](../../../../glossary.md#trace)/tuning/[cache](../../../../glossary.md#term-cache) translation is deferred; [Gradle](../../../../glossary.md#term-gradle), capability, [conformance](../../../../glossary.md#term-conformance), and
integration contracts do not change.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the isolated implementation agent for Synaptik CPU task 0008D.

Read AGENTS.md, ARCHITECTURE.md, the current architecture plan, performance-evidence/tuning page
and ADR 0008, planning guide, roadmap, CPU master plan, complete CPU 0008B/0008C tasks, this task,
and final affected production/tests/artifact-identity contracts.

Implement exactly the Ready specification. Preserve the exact 0008B compatibility baseline,
0008C facts/snapshots, schema 52, and all generated/hot forms. Do not add measurement, tuning or
cache work, Trace payloads, public/shared APIs, 0008E materialization, MATMUL, or another family.
Stop on architecture, shared-contract, generated-form, complete-enumeration, or scope conflict.

After stable CPU/identity validation, hand the same diff/evidence to a separate clean
documentation-focused agent following documentation-rules.md. It independently finalizes
Javadocs, CPU guide, glossary impact, planning evidence/status, and documentation checks without
repeating stable Java/performance evidence unless executable behavior changes or a concrete risk
is recorded.

Do not stage, commit, or push. Update implementation sections/status only from actual evidence.
```

## Local decisions

- Canonical maximally split seeds are the correctness fallback; exact 0008B maximal-legal output
  is a mandatory comparison/compatibility candidate, not a forced profitability winner.<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
- Complete enumeration is the closed existing contraction grammar within 64 candidates/256
  attempts; exceeding either invalidates partial ranking.<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span><span data-glossary-navigation="true"> ([contraction glossary](../../../../glossary.md#contraction-dimension))</span>
- Recognition is a fixed barrier, avoiding silent expansion of 0008C's two-unit association.
- Conservative integer heuristics estimate structure without claiming measured time. The 32-point
  margin and 75%-of-hard-limit code/live ceilings reject marginal complex fusion; ties split.<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion))</span>
- Facts remain outside artifact identity and Runtime. Later diagnostics/tuning require their own
  owner tasks.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [identity glossary](../../../../glossary.md#term-identity-meanings); [Runtime glossary](../../../../glossary.md#term-runtime); [owner glossary](../../../../glossary.md#term-owner))</span>
- Open questions: None; any need outside these fixed decisions requires replanning.

## Known limitations

- This is a structural [heuristic](../../../../glossary.md#term-heuristic), not measurement or a universal speedup claim.
- Only existing [pointwise](../../../../glossary.md#term-pointwise) [contraction](../../../../glossary.md#contraction-dimension) varies; recognition/family execution does not expand.
- [CPU](../../../../glossary.md#term-cpu) 0008E owns representation/[materialization](../../../../glossary.md#materialization) variants; tuning/[cache](../../../../glossary.md#term-cache)/[opaque](../../../../glossary.md#term-opaque) [handoff](../../../../glossary.md#term-handoff) is absent.
- CPU 0009 retains capability/[conformance](../../../../glossary.md#term-conformance)/integration closure.

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context: `01a03f9e-9ea6-7bf2-988e-565cf055725e`.
- The focused decision/enumeration/preparation/recognition/identity matrix passed 51 tests with no
  failures or errors. The subsequent affected carrier/composite regression matrix passed 105
  tests with no failures or errors.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [carrier glossary](../../../../glossary.md#term-storage-carrier); [regression glossary](../../../../glossary.md#term-regression))</span>
- The final non-authoritative full CPU preflight passed 532 tests in 103 suites with 3 skipped and
  no failures or errors. Java and test paths were then frozen.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight))</span>
- Main review corrections passed the final 55-test focused decision, enumeration, preparation,
  recognition, identity, and inventory matrix. It includes fail-closed forgeries of retained
  member order, structural key, specialization, dependency, boundary, workspace, PUBLICATION,
  and PARTITION_WRITE facts. After Java/tests froze, the one new authoritative<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed); [specialization glossary](../../../../glossary.md#term-specialization); [dependency glossary](../../../../glossary.md#term-dependency); [PUBLICATION glossary](../../../../glossary.md#term-publication))</span>
  `./gradlew :backends:cpu:test --rerun-tasks` invocation passed in 19 seconds with all 22 tasks
  executed: 536 tests in 103 suites, 3 skipped, no failures, and no errors.
- `decisionFactsStayOutsideStructuralAndClassFileIdentity` passed under the authoritative suite.
  The regenerated multi-store Class-File and retained 0008B control both have SHA-256
  `9c39fe6d782c90e8df951bc0553127fd36c1c3e7e9b712dcf3469a721a9ad08e`.
  `javap -v -p` reports Java 26 major version 70 and only the existing static `invoke` [descriptor](../../../../glossary.md#term-descriptor)
  `([D[D[D[D[JJJ)V`; decision, selector, recognition, plan, [Runtime](../../../../glossary.md#term-runtime), and [Trace](../../../../glossary.md#trace) names are absent.
- Fresh 0008D validation is retained at
  `/private/tmp/synaptik-cpu-0008d-evidence/run-1787811127399`: five isolated warmed forks and all
  45 valid samples per comparison, without retries or discarded ratios. Generated selected fused
  topology versus its optimal clean-Java equivalent has median <span data-glossary-navigation="true"> ([samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>`0.886422494`, range
  `0.770534550–1.020759890`, and [fork](../../../../glossary.md#term-fork-meanings) medians `[0.904498666, 0.863566684, 0.911971163,
  0.893748893, 0.844830644]`. Selected [fused](../../../../glossary.md#specialization-and-fusion) [topology](../../../../glossary.md#term-topology) versus exact [canonical](../../../../glossary.md#term-canonical-form) split has [median](../../../../glossary.md#term-median)
  `0.503477345`, range `0.455093555–0.545522309`, and fork medians `[0.500456830,
  0.504762972, 0.510064970, 0.490162694, 0.499579832]`. These validate this fixed topology on the
  evidence host and are not a universal speedup claim.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host))</span>
- The validation-only correction changed no executable, generated, emitter, or performance-test
  Java, so the stable performance evidence above remains the owning evidence. The mandatory
  authoritative suite intrinsically exercised the unchanged evidence path once and additionally
  retained `/private/tmp/synaptik-cpu-0008d-evidence/run-1787812131415`; it retained all samples,
  reported generated/direct median `0.885426214` and range `0.795689187–1.023570898`, and
  fused/split median `0.494450262` and range `0.457029703–0.535479074`, with the same three class
  hashes below.<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span>
- Selected-fused/canonical-first/canonical-second Class-File [SHA-256](../../../../glossary.md#term-sha-256) values are respectively
  `4b14ae9848fcc4c3109be3a665ff619cf2ebd1b5c0cffdee9530402be115e9ac`,
  `baf6a8c5f8cca73e098e87a0459dc9c199adc071e57ffead16fbee5b2b4098ac`, and
  `e707a5b2baf7c67cb53873c464b2916d86cc60b667e410de1bebd229748d62d1`.
  Retained Java-26 `javap -c -v -p` and [Class-File API](../../../../glossary.md#term-class-file-api) inspection show the unchanged
  `([I[I[JJJ)V` descriptor, one method, zero fields, member references, allocations, dispatch
  instructions, and string constants for all three classes.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [constants glossary](../../../../glossary.md#term-constant))</span>
- Source no-leak scans found no decision or selector reference in code generation, executable,
  cache, or finalization sources. <span data-glossary-navigation="true"> ([code generation glossary](../../../../glossary.md#term-code-generation); [cache glossary](../../../../glossary.md#term-cache))</span>`CpuGeneratorSchema.CURRENT_VERSION` remains 52. The historical
  0008B semantic/Class-File and five-fork evidence also passed unchanged.
- Clean documentation-focused review context: `01a041eb-619f-7fb3-8ab7-a99daf2da3ab`.
  It independently read the architecture, planning, documentation profiles, complete predecessor
  tasks, final changed production/tests, CPU guide, glossary, and both retained evidence runs. It
  changed Javadocs and Markdown only; executable Java and tests remained frozen.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- The implementation scope before warranted explanatory documentation is exactly 5 production
  paths (exactly 2 new), 6 test paths (exactly 2 new), and 3 planning paths: 14 paths. The planned
  documentation allowance adds `docs/backend-guide/cpu-backend.md` under the General and Backend-
  guide profiles and <span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend))</span>`docs/glossary.md` under the General/glossary rules, for 5 documentation/
  planning paths and 16 total paths, exactly at the task ceilings.
- The CPU guide required a narrow update because it still described profitability as future work;
  it now explains complete bounded enumeration, exact structural scoring and guardrails, best-only
  tie fallback, typed fact ownership, publication projection, fail-closed baseline validation, and
  the evidence boundary without a universal speedup claim. The glossary required updates to the
  existing CPU portable-route, preparation-plan, and execution-unit entries because their reusable
  implemented-boundary definitions still deferred profitability.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Final `./gradlew :backends:cpu:javadoc` passed in 2 seconds with 11 actionable tasks, 2 executed
  and 9 up-to-date. The only warnings are the two established Java 26 incubating Vector-module
  warnings. The first successful generation exposed 62 missing enum-value comments; the
  documentation pass completed those descriptions and regenerated the final warning-clean API
  surface. Rendered pages for <span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [module glossary](../../../../glossary.md#term-module); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>`CpuFusionDecision` and `CpuFusionProfitabilitySelector` exist.
- Targeted Markdown validation passed for the five documentation/planning paths: 880 local links
  and anchors resolve, fences are balanced, every file has a final newline, and the task retains
  the canonical 20-heading order. The first Ruby helper had a `#{1,6}` interpolation syntax error;
  the first compatible retry then found that this environment lacks `Array#filter_map`; and the
  next retry used an anchor normalizer that collapsed meaningful double hyphens and therefore
  reported false missing anchors. The final `map(...).compact` helper with GitHub-compatible
  space preservation passed. These were validator-script corrections, not documentation defects.
- Final [scope](../../../../glossary.md#term-scope)/status/[schema](../../../../glossary.md#term-schema)/hygiene checks passed: exactly one [CPU](../../../../glossary.md#term-cpu) [`Ready`](../../../../glossary.md#term-ready) row exists (`0008E`),
  no 0008E task file exists, schema remains 52, rendered Javadoc pages exist, exact scope is 5
  production + 6 test + 5 documentation/planning paths, <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`git diff --check` and
  `git diff --cached --check` pass, and the staged index is empty. The Java CPU test suite and
  performance benchmarks were not rerun because executable Java remained frozen.

## Implementation notes

- Added the closed [CPU](../../../../glossary.md#term-cpu)-private `CpuFusionDecision` [model](../../../../glossary.md#term-model) and stateless
  `CpuFusionProfitabilitySelector` with stable typed identities, exact checked integer scoring,
  distinct legality/profitability/selection facts, deterministic ranks, and conservative split,
  tie, uncertainty, and incomplete-enumeration outcomes.
- Corrected selection to inspect the best guardrail-compatible alternative first: a strictly
  profitable winner now wins even when a different non-winning candidate ties canonical split;<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
  `SAFE_SPLIT_TIE`/`TIE_FALLBACK` applies only when that best comparison itself ties split. The
  narrow typed `LegalCandidate` policy seam proves both outcomes without entering preparation.
- Narrowly extended the decomposer with breadth-first complete-topology enumeration bounded at 64
  candidates and 256 distinct pair attempts while leaving <span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology))</span>`decompose(...)` and its exact 0008B
  compatibility baseline unchanged. The sole typed operation-free topology seam reaches dominated
  65th-candidate and pending-257th-attempt cases.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- `decompose(...)` and enumeration now consume the same typed contraction result. Explicit
  topology and portable-IR family facts classify barriers; lower rejection is<span data-glossary-navigation="true"> ([contraction glossary](../../../../glossary.md#contraction-dimension); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
  `UNSUPPORTED_LOWERING` when no narrower typed fact exists. No exception-message, package-name,
  or other string dispatch remains.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Preparation evaluates legal complete candidates, preserves exact 0008C recognition associations
  and snapshots, projects caller carrier choices from the exact 0008B boundary view while assigning
  new internal split boundaries the existing segment form, retains decisions, and recomputes the
  selected identity during plan validation. Validation now compares every derivable retained
  recognition overlap against the immutable compatibility baseline, including member order,
  structural key, specialization, strategy, dependencies, access regime, referenced bytes,
  alignment, representable boundary role, workspace geometry/role, and pointwise/indivisible
  topology. It separately retains the typed graph-publication boundary positions projected from
  existing logical-memory requirements and recomputes PUBLICATION versus PARTITION_WRITE without
  trusting the claimed decision role.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [view glossary](../../../../glossary.md#term-view-meanings); [identity glossary](../../../../glossary.md#term-identity-meanings); [validation glossary](../../../../glossary.md#term-validation); [immutable glossary](../../../../glossary.md#term-immutable); [specialization glossary](../../../../glossary.md#term-specialization); [alignment glossary](../../../../glossary.md#term-alignment); [pointwise glossary](../../../../glossary.md#term-pointwise); [graph glossary](../../../../glossary.md#graph); [publication glossary](../../../../glossary.md#term-publication))</span>
- Focused tests cover closed inventories, immutable facts, 1/8 units, 1/64 candidates, 256 attempts
  with a pending 257th, 63 profitability rejections, 384 total facts, 64 boundary positions, 8
  workspaces, vertical fusion, publication and diamond fan-out barriers, attempt-budget fallback,
  malformed retained plans, unchanged carrier projection, composite execution, schema identity,
  Class-File leakage, the non-winning-tie regression, and fresh selected-fused/direct and
  selected-fused/canonical-split performance distributions.<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion); [fan-out glossary](../../../../glossary.md#term-fan-out); [schema glossary](../../../../glossary.md#term-schema); [regression glossary](../../../../glossary.md#term-regression); [fused glossary](../../../../glossary.md#specialization-and-fusion); [distributions glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
- No 0008E materialization, MATMUL, Trace, tuning/cache/measurement, public/shared contract,
  Runtime, emitter, schema, capability, build, architecture, conformance, or integration change was
  made. Java/tests remained frozen after the authoritative pass; the clean documentation review
  finalized Javadocs, the CPU guide, glossary impact, planning evidence/status, and documentation
  checks without repeating stable Java/performance evidence.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [Trace glossary](../../../../glossary.md#trace); [cache glossary](../../../../glossary.md#term-cache); [Runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Implementation completed within the fixed production/test/path ceilings, including bounded
  enumeration, selection, retained facts, recomputed plan validation, carrier-compatible selected
  preparation, focused tests, and retained Class-File/no-leak evidence.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Documentation-agent review: Complete in clean context
  `01a041eb-619f-7fb3-8ab7-a99daf2da3ab`; affected Javadocs, the CPU guide, glossary, this task,
  CPU master plan, and roadmap were finalized without executable behavior changes.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Documentation impact: the guide and existing glossary entries now document the implemented
  legality/profitability/selection boundary, exact bounded heuristic, safe fallbacks, typed cold
  ownership, publication projection, fail-closed retained baseline validation, and evidence limits.<span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [publication glossary](../../../../glossary.md#term-publication); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Javadoc review: finalized all affected/new production types, record components, constructors,
  and methods for invariants, defensive ownership, nullability, limits, failures, projection, and
  selection semantics.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [nullability glossary](../../../../glossary.md#term-nullability))</span>
- Architecture/public/shared/build/conformance/integration impact: None; the change remains
  CPU-private cold Prepare analysis and changes no capability, dependency, schema, generated form,
  shared lifecycle, or Runtime behavior.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Prepare glossary](../../../../glossary.md#prepare); [dependency glossary](../../../../glossary.md#term-dependency); [schema glossary](../../../../glossary.md#term-schema); [lifecycle glossary](../../../../glossary.md#lifecycle); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Unresolved issues: None.
- Follow-up required: None. CPU 0008E is the sole Ready CPU row; no detailed 0008E specification
  exists.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- No staging, [commit](../../../../glossary.md#term-commit), or push occurred.

Status: Complete
