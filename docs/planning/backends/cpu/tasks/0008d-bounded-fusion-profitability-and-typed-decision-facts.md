# Task 0008D: Bounded Fusion Profitability and Typed Decision Facts

## Status

Complete

## Goal

Add one cohesive CPU-private cold-analysis capability that enumerates the complete admitted set of
legal pointwise fused and materialized-split partition candidates, ranks every legal complete
candidate with deterministic bounded no-measurement heuristics, and retains closed typed legality,
profitability, and selection facts.

The exact CPU 0008B plan remains a mandatory reproducible comparison candidate. CPU 0008C
recognition facts, dispositions, associated baseline snapshots, and artifact-identity exclusion
remain unchanged. Legality rejection is distinct from profitability rejection. Tie, uncertainty,
or incomplete bounded enumeration selects the canonical safe split. Selection finishes during CPU
analysis; Runtime receives only the selected prepared work.

## Scope

### Closed typed facts and reproducible identities

- Add sealed immutable CPU-private `CpuFusionDecision` with exactly `LegalCandidate`,
  `LegalityRejection`, `ProfitabilityRejection`, and `Selection` variants. Nested records/enums may
  carry common facts. Collections are defensive immutable ordered snapshots.
- `LegalCandidate` records one stable `CandidateIdentity`, complete topology/resource/hard-budget
  facts, profitability inputs, integer score, and canonical-split/0008B-baseline roles.
- `LegalityRejection` records the source topology, attempted pair, exact failed hard fact, and one
  closed reason: `SEMANTIC_BARRIER`, `PUBLICATION_BARRIER`, `FAN_OUT_BARRIER`,
  `STATE_OR_RANDOM_BARRIER`, `NUMERICAL_ORDER_BARRIER`, `ALIAS_OR_ACCESS_UNPROVED`,
  `DEPENDENCY_CYCLE`, `UNSUPPORTED_LOWERING`, `ROUTE_INELIGIBLE`, or
  `HARD_BUDGET_EXCEEDED`.
- `ProfitabilityRejection` refers only to a retained legal complete candidate and uses one closed
  reason: `INSUFFICIENT_MARGIN`, `CODE_SIZE_PRESSURE`, `LIVE_VALUE_PRESSURE`,
  `MATERIALIZATION_COST`, `SAFE_SPLIT_TIE`, or `UNCERTAIN_INPUT`.
- `Selection` records selected, canonical-split, and 0008B-baseline identities; stable rank;
  score/margin facts; and `PROFITABLE_FUSION`, `CANONICAL_SPLIT`, `TIE_FALLBACK`,
  `UNCERTAINTY_FALLBACK`, or `ENUMERATION_BUDGET_FALLBACK`.
- `CandidateIdentity` contains, in stable unit order, relative member-node positions, dependency
  indices, portable IR structural keys, specializations, strategies, access regimes, relative
  materialized-boundary roles/geometry, workspace facts, and fused/split topology. It excludes
  node/value/partition/requirement/slot identity, carrier objects, addresses, workers, runs,
  loaders, generated classes, artifact roots, timing, cache contents, and map iteration order.
- Identity and tie-breaking use typed values and stable relative positions, never strings, enum
  text, object identity, or `ValueId` magnitude. No maps, reflection, annotations, callbacks,
  registries, DSLs, or generic parameter bags are permitted.

### Complete bounded fused/split enumeration

- Narrowly refactor `CpuPartitionDagDecomposer` to expose the unchanged maximally split seeds, the
  exact unchanged CPU 0008B result, and every distinct complete topology reachable through zero or
  more existing 0008B vertical/horizontal ordinary-pointwise contractions within the budgets
  below. Its existing `decompose(...)` result remains exactly compatible with pre-0008D behavior.
- Enumerate breadth-first from canonical split. For each source topology, test vertical pairs in
  producer/member order, then horizontal pairs in stable left/right order. Deduplicate resulting
  candidates by typed identity and record each distinct source/pair attempt once.
- Rank only complete candidates that cover every node/publication exactly once and contain every
  exact resource. Partial contraction paths or local kernels are never candidates.
- Preserve every 0008B legality rule and hard budget. A failed contraction emits a typed legality
  rejection and leaves its source topology valid. Failure of the canonical split fails analysis
  before declarations or decision facts.
- CPU 0008C recognition is a fixed barrier: enumeration may not split, merge, reorder, or otherwise
  change a retained fact's members, at-most-two associated baseline units, disposition, or exact
  baseline snapshot. Do not rerun a looser pattern or infer equivalent mathematics.
- Specialized, affine, state/random, reduction/scan, ordering, scatter/fold, softmax,
  normalization, batch-normalization, Conv1d/2d/3d, and multi-output units remain indivisible.
  Only already implemented ordinary pointwise IR contractions vary.

### Exact budgets and reachable seams

| Budget | Ceiling | On exceedance |
| --- | ---: | --- |
| Partition nodes / seed or final units | 8 / 8 | Existing fail-closed baseline rule |
| Distinct complete candidate topologies | 64 | Canonical split; no partial ranking |
| Distinct source-topology pair attempts | 256 | Canonical split; no partial ranking |
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
  it accepts no `Operation`, invents no generated form, and never enters production preparation.

### Exact default profitability heuristic

- Use only checked integer facts already available during analysis; read no clock, benchmark,
  profile, trace, cache, environment policy, global mutable state, or Runtime state.
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
  do not duplicate formulas.
- Missing geometry/type width, inconsistent roles, division error, or checked overflow produces
  `UNCERTAIN_INPUT`; never saturate or treat unknown as favorable.
- A fused candidate is profitable over canonical split only if its score is at least 32 lower,
  every pointwise unit is at most 48 generated-code-size units and 12 live values, total
  materialized referenced bytes do not increase, and route/numerical/determinism/workspace/0008C
  baseline facts remain exact.
- Rank qualifying candidates by score, materialized bytes, maximum code-size units, and maximum
  live values. Exact ties prefer more units then stable identity. If the tied best would be more
  fused than canonical split, record `SAFE_SPLIT_TIE` and select canonical split.
- Uncertain canonical split, any incomparable legal candidate, or incomplete enumeration selects
  canonical split. The exact 0008B baseline remains a marked candidate and may win, lose, or equal
  split; it is never mutated or omitted.

### Preparation, ownership, and generated-code discipline

- `CpuFusionProfitabilitySelector` owns stateless enumeration/ranking. CPU analysis completes it
  before deriving the selected resources once through existing 0008B assembly.
- Extend `CpuPartitionPreparationPlan` with immutable decision facts. A trivial one-unit plan has
  equal split/baseline/selected identities. Validation recomputes consistency and proves selected
  identity exactly matches retained units/resources; all rejected facts refer to the right legal
  or attempted candidate.
- Finalization realizes only already selected artifacts after assignment and cannot rerank or add
  resources. Runtime, shared Prepare, Planning, Compiler, Model, Config, Trace, Engine, Backend
  Contract, and other backends gain no type or interpretation responsibility.
- Decision facts remain CPU-private cold metadata, outside generated artifact/resource/cache
  identity and Runtime. Public Trace translation, tuning inspection, opaque shared candidate
  handoff, persistence, measurement, cache use, and serialization are deferred.
- Select only existing IR, specialization, entry shape, route, algorithm, and executable forms.
  Prove exact structural-key/class-byte identity and absence of decision/recognition/selector
  references in Class-File, artifact, finalization handle, and hot invocation evidence.
- Reuse owning CPU 0005-series/0008B optimal-clean-Java performance evidence when identity and hot
  code are unchanged. Run no redundant timing and make no new speedup claim.
- Any new emitter branch, IR form, entry descriptor, helper call, algorithm, route, artifact key/
  schema field, composite hot behavior, or generated/hot form is out of scope. Stop and replan it
  with an optimal clean-Java oracle, complete Class-File/hidden-helper inspection, and fresh
  five-fork generated/direct `<= 1.15x` evidence.

### Documentation

- After stable implementation/Java evidence, use a separate clean documentation-focused context.
  Finalize detailed Javadocs and update the CPU guide with legality versus profitability, exact
  heuristics, typed ownership, fallbacks, and no-measurement/no-Runtime-selection boundaries.
- Review the glossary and change it only for a reusable terminology change; otherwise record a
  reasoned no-change conclusion. Synchronize this task, CPU master plan, and roadmap.
- CPU 0008D becomes `Review needed` after implementation validation and `Complete` only after the
  documentation gate. On completion CPU 0008E becomes the sole `Ready` row; create no 0008E
  specification.

## Out of scope

- New semantics, algorithms, family lowerers, emitters, generated forms, routes, strategies,
  capabilities, artifact schema/key fields, hot composite work, or performance claims.
- Any redefinition/broadening of 0008C recognition, family inventory, precedence, disposition,
  explicit-semantic-only rule, or baseline snapshots.
- CPU 0008E materialization/representation expansion; MATMUL (0008F); pooling, attention, or loss
  work (0008G–0008I).
- Measurement, benchmark-selected behavior, autotuning, tuning-cache use/mutation, persistence,
  public configuration, target discovery, Trace payloads, public/generic candidate APIs,
  registries, DSLs, callbacks, reflection, string dispatch, maps, or serialization.
- Compiler/Planning/shared-Prepare/Runtime/Engine changes, another module/backend, public API,
  architecture/dependency/Gradle changes, conformance/integration tests, or generated evidence.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md): backend-owned fusion/selection, staged
  resources, generated-code discipline, tuning separation, and Runtime boundary.
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md).
- [`performance evidence and model autotuning`](../../../../architecture/performance-evidence-and-tuning.md).
- [`ADR 0008`](../../../../design/decisions/0008-performance-evidence-and-tuning-boundaries.md).
- [`planning guide`](../../../planning-guide.md).
- [`CPU master plan`](../master-plan.md).
- Complete [`CPU 0008B`](0008b-general-partition-dag-computation-unit-decomposition-and-bounded-fusion.md)
  and [`CPU 0008C`](0008c-typed-specialized-subgraph-and-epilogue-recognition.md).

## Architecture constraints

- Planning selects CPU ownership; CPU analysis alone enumerates, proves legality, ranks, selects,
  and declares exact resources before shared assignment.
- Legality precedes profitability and cannot be weakened by cost. Analysis is deterministic from
  explicit facts and performs no measurement/cache lookup.
- 0008C recognition/baseline snapshots remain immutable inputs. Decision facts stay outside
  generated identity, shared contracts, and Runtime hot work.
- Finalization realizes existing selected forms only. Any need for a public/shared type, new
  generated/hot form, schema/route/capability/module/dependency/architecture change is a stop-and-
  replan condition.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.ir` — closed facts and typed identities.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — enumeration, shared hard facts,
  heuristic ranking, and selection.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — selected plan/resources and validation.

Packages added or changed:

- None.

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision` — sealed facts and identity.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuFusionProfitabilitySelector` — sole
  bounded selector.

Tests mirror production packages. No public, tuning, trace, registry, cache, or new source-set
package is added.

## Affected files

Expected production/Javadoc paths:

- new `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuFusionDecision.java`;
- new `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuFusionProfitabilitySelector.java`;
- `CpuPartitionDagDecomposer.java` in that lowering package;
- `CpuPartitionPreparationPlan.java` and `CpuPartitionPreparer.java` in `internal.prepare`.

Expected test paths:

- new `CpuFusionDecisionTest` and `CpuFusionProfitabilitySelectorTest`;
- existing decomposer, plan/preparer, specialized-recognizer, and artifact-identity or package-
  inventory tests only as directly required.

Expected completion documentation paths are CPU guide, glossary only if justified, this task, CPU
master plan, and roadmap. No other Java/test/generated/architecture/ADR/Gradle/conformance/
integration/later-task path is expected.

## Maximum scope

- At most 5 CPU production/Javadoc paths, including exactly 2 new production types.
- At most 6 CPU test paths, including at most 2 new test types.
- At most 5 documentation/planning paths, including this task and conditional glossary.
- At most 16 total paths and 384 decision facts per admitted partition.

If another owner, package, form, path, type, or ceiling is required, stop and replan.

## Acceptance criteria

- Closed typed immutable facts distinguish legality, profitability, and selection with exact
  identities/reasons and no forbidden generic/public mechanism.
- Canonical split and exact 0008B baseline are always present for supported work; compatibility
  decomposition and 0008C facts/baseline snapshots remain exact.
- Every admitted complete topology is ranked or incomplete enumeration selects canonical split.
  No partial candidate set controls selection.
- All 0008B barriers/budgets remain hard and each typed legality reason is distinct from
  profitability.
- Exact score constants/formulas, 32-point margin, 48/12 soft ceilings, no-increased-
  materialization rule, ranks, tie behavior, and uncertainty fallback pass at/one-beyond seams.
- Tests reach 64/over-64 candidates, 256/pending-257 attempts, 63 profitability rejections, 384
  facts, 8 units/workspaces, and 64 boundary positions via real fixtures or the bounded seam.
- Repeated analysis and randomized map insertion produce identical ordered facts, choice,
  resources, structural keys, and class bytes.
- Plan validation rejects missing/duplicate/mismatched facts, identities, ranks, scores, reasons,
  baseline/split roles, or selected units/resources.
- Ties, unknowns, overflow, incomparable candidates, and incomplete enumeration select canonical
  split without losing nodes, publications, edges, workspaces, or execution.
- Exact class identity and no-leak scans pass; no new timing/performance claim is made.
- Final CPU tests, Javadoc, guide/glossary review, planning synchronization, scope/status/schema-
  52/empty-index/whitespace checks pass in the required implementation/documentation contexts.
- No public/shared/build/architecture/conformance/integration/0008E-specification change occurs.

## Tests / validation

Focused tests cover facts, identity, enumeration/deduplication, every reason, score arithmetic,
thresholds/budgets/fallbacks, 0008B/0008C preservation, plan/resources/finalization, and forbidden
leakage. After the last executable change run exactly one authoritative:

```bash
./gradlew :backends:cpu:test --rerun-tasks
```

Regenerate representative one-unit, vertical, horizontal multi-store, Conv2d, reduction-split,
and explicit-semantic existing classes; compare keys/checksums and inspect `javap -c -v -p` plus
member references. Reuse owning performance evidence.

The clean documentation context runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short -uall
```

It also validates local links/anchors, canonical headings, fences, final newlines, rendered
Javadoc, terminology, path/type/fact ceilings, schema 52, 0008C preservation, 0008D/0008E status,
absence of a 0008E file, and empty staging. Repository/architecture/conformance/integration suites
remain CPU 0009/CI unless a forbidden shared change requires replanning.

## Dependencies

- Complete CPU 0008B and CPU 0008C.
- Complete CPU 0005A–0005J and CPU 0008 existing IR/generation/evidence contracts.
- Existing Planning, Prepare, Runtime, Config, Trace, cache/artifact, and tuning boundaries.

No architecture or shared-contract blocker is known.

## Follow-up tasks

- CPU 0008E becomes Ready after this task completes: at-most-two external read materializations
  and representation reuse; create no specification now.
- CPU 0008F–0008I retain MATMUL, pooling, attention, and loss ownership.
- Config 0006A, Prepare 0004, CPU 0016, and Tuning 0001–0002 retain future explicit tuning inputs,
  opaque handoff, compatible cache consumption, and bounded measurement.
- CPU 0009 retains coverage/conformance/integration closure.

## Architecture impact

Expected impact: None.

No-change conclusions: architecture/ADRs/tests already authorize backend-private candidates and
safe heuristics; Model/Compiler semantics and graph stay unchanged; Planning ownership/Config stay
unchanged; shared Prepare sees selected declarations only; Runtime/Engine execute unchanged
prepared work; Trace/tuning/cache translation is deferred; Gradle, capability, conformance, and
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
  is a mandatory comparison/compatibility candidate, not a forced profitability winner.
- Complete enumeration is the closed existing contraction grammar within 64 candidates/256
  attempts; exceeding either invalidates partial ranking.
- Recognition is a fixed barrier, avoiding silent expansion of 0008C's two-unit association.
- Conservative integer heuristics estimate structure without claiming measured time. The 32-point
  margin and 75%-of-hard-limit code/live ceilings reject marginal complex fusion; ties split.
- Facts remain outside artifact identity and Runtime. Later diagnostics/tuning require their own
  owner tasks.
- Open questions: None; any need outside these fixed decisions requires replanning.

## Known limitations

- This is a structural heuristic, not measurement or a universal speedup claim.
- Only existing pointwise contraction varies; recognition/family execution does not expand.
- CPU 0008E owns representation/materialization variants; tuning/cache/opaque handoff is absent.
- CPU 0009 retains capability/conformance/integration closure.

## Validation evidence

- Implementation context: `01a03f9e-9ea6-7bf2-988e-565cf055725e`.
- The focused decision/enumeration/preparation/recognition/identity matrix passed 51 tests with no
  failures or errors. The subsequent affected carrier/composite regression matrix passed 105
  tests with no failures or errors.
- The final non-authoritative full CPU preflight passed 532 tests in 103 suites with 3 skipped and
  no failures or errors. Java and test paths were then frozen.
- Main review corrections passed the final 55-test focused decision, enumeration, preparation,
  recognition, identity, and inventory matrix. It includes fail-closed forgeries of retained
  member order, structural key, specialization, dependency, boundary, workspace, PUBLICATION,
  and PARTITION_WRITE facts. After Java/tests froze, the one new authoritative
  `./gradlew :backends:cpu:test --rerun-tasks` invocation passed in 19 seconds with all 22 tasks
  executed: 536 tests in 103 suites, 3 skipped, no failures, and no errors.
- `decisionFactsStayOutsideStructuralAndClassFileIdentity` passed under the authoritative suite.
  The regenerated multi-store Class-File and retained 0008B control both have SHA-256
  `9c39fe6d782c90e8df951bc0553127fd36c1c3e7e9b712dcf3469a721a9ad08e`.
  `javap -v -p` reports Java 26 major version 70 and only the existing static `invoke` descriptor
  `([D[D[D[D[JJJ)V`; decision, selector, recognition, plan, Runtime, and Trace names are absent.
- Fresh 0008D validation is retained at
  `/private/tmp/synaptik-cpu-0008d-evidence/run-1787811127399`: five isolated warmed forks and all
  45 valid samples per comparison, without retries or discarded ratios. Generated selected fused
  topology versus its optimal clean-Java equivalent has median `0.886422494`, range
  `0.770534550–1.020759890`, and fork medians `[0.904498666, 0.863566684, 0.911971163,
  0.893748893, 0.844830644]`. Selected fused topology versus exact canonical split has median
  `0.503477345`, range `0.455093555–0.545522309`, and fork medians `[0.500456830,
  0.504762972, 0.510064970, 0.490162694, 0.499579832]`. These validate this fixed topology on the
  evidence host and are not a universal speedup claim.
- The validation-only correction changed no executable, generated, emitter, or performance-test
  Java, so the stable performance evidence above remains the owning evidence. The mandatory
  authoritative suite intrinsically exercised the unchanged evidence path once and additionally
  retained `/private/tmp/synaptik-cpu-0008d-evidence/run-1787812131415`; it retained all samples,
  reported generated/direct median `0.885426214` and range `0.795689187–1.023570898`, and
  fused/split median `0.494450262` and range `0.457029703–0.535479074`, with the same three class
  hashes below.
- Selected-fused/canonical-first/canonical-second Class-File SHA-256 values are respectively
  `4b14ae9848fcc4c3109be3a665ff619cf2ebd1b5c0cffdee9530402be115e9ac`,
  `baf6a8c5f8cca73e098e87a0459dc9c199adc071e57ffead16fbee5b2b4098ac`, and
  `e707a5b2baf7c67cb53873c464b2916d86cc60b667e410de1bebd229748d62d1`.
  Retained Java-26 `javap -c -v -p` and Class-File API inspection show the unchanged
  `([I[I[JJJ)V` descriptor, one method, zero fields, member references, allocations, dispatch
  instructions, and string constants for all three classes.
- Source no-leak scans found no decision or selector reference in code generation, executable,
  cache, or finalization sources. `CpuGeneratorSchema.CURRENT_VERSION` remains 52. The historical
  0008B semantic/Class-File and five-fork evidence also passed unchanged.
- Clean documentation-focused review context: `01a041eb-619f-7fb3-8ab7-a99daf2da3ab`.
  It independently read the architecture, planning, documentation profiles, complete predecessor
  tasks, final changed production/tests, CPU guide, glossary, and both retained evidence runs. It
  changed Javadocs and Markdown only; executable Java and tests remained frozen.
- The implementation scope before warranted explanatory documentation is exactly 5 production
  paths (exactly 2 new), 6 test paths (exactly 2 new), and 3 planning paths: 14 paths. The planned
  documentation allowance adds `docs/backend-guide/cpu-backend.md` under the General and Backend-
  guide profiles and `docs/glossary.md` under the General/glossary rules, for 5 documentation/
  planning paths and 16 total paths, exactly at the task ceilings.
- The CPU guide required a narrow update because it still described profitability as future work;
  it now explains complete bounded enumeration, exact structural scoring and guardrails, best-only
  tie fallback, typed fact ownership, publication projection, fail-closed baseline validation, and
  the evidence boundary without a universal speedup claim. The glossary required updates to the
  existing CPU portable-route, preparation-plan, and execution-unit entries because their reusable
  implemented-boundary definitions still deferred profitability.
- Final `./gradlew :backends:cpu:javadoc` passed in 2 seconds with 11 actionable tasks, 2 executed
  and 9 up-to-date. The only warnings are the two established Java 26 incubating Vector-module
  warnings. The first successful generation exposed 62 missing enum-value comments; the
  documentation pass completed those descriptions and regenerated the final warning-clean API
  surface. Rendered pages for `CpuFusionDecision` and `CpuFusionProfitabilitySelector` exist.
- Targeted Markdown validation passed for the five documentation/planning paths: 880 local links
  and anchors resolve, fences are balanced, every file has a final newline, and the task retains
  the canonical 20-heading order. The first Ruby helper had a `#{1,6}` interpolation syntax error;
  the first compatible retry then found that this environment lacks `Array#filter_map`; and the
  next retry used an anchor normalizer that collapsed meaningful double hyphens and therefore
  reported false missing anchors. The final `map(...).compact` helper with GitHub-compatible
  space preservation passed. These were validator-script corrections, not documentation defects.
- Final scope/status/schema/hygiene checks passed: exactly one CPU `Ready` row exists (`0008E`),
  no 0008E task file exists, schema remains 52, rendered Javadoc pages exist, exact scope is 5
  production + 6 test + 5 documentation/planning paths, `git diff --check` and
  `git diff --cached --check` pass, and the staged index is empty. The Java CPU test suite and
  performance benchmarks were not rerun because executable Java remained frozen.

## Implementation notes

- Added the closed CPU-private `CpuFusionDecision` model and stateless
  `CpuFusionProfitabilitySelector` with stable typed identities, exact checked integer scoring,
  distinct legality/profitability/selection facts, deterministic ranks, and conservative split,
  tie, uncertainty, and incomplete-enumeration outcomes.
- Corrected selection to inspect the best guardrail-compatible alternative first: a strictly
  profitable winner now wins even when a different non-winning candidate ties canonical split;
  `SAFE_SPLIT_TIE`/`TIE_FALLBACK` applies only when that best comparison itself ties split. The
  narrow typed `LegalCandidate` policy seam proves both outcomes without entering preparation.
- Narrowly extended the decomposer with breadth-first complete-topology enumeration bounded at 64
  candidates and 256 distinct pair attempts while leaving `decompose(...)` and its exact 0008B
  compatibility baseline unchanged. The sole typed operation-free topology seam reaches dominated
  65th-candidate and pending-257th-attempt cases.
- `decompose(...)` and enumeration now consume the same typed contraction result. Explicit
  topology and portable-IR family facts classify barriers; lower rejection is
  `UNSUPPORTED_LOWERING` when no narrower typed fact exists. No exception-message, package-name,
  or other string dispatch remains.
- Preparation evaluates legal complete candidates, preserves exact 0008C recognition associations
  and snapshots, projects caller carrier choices from the exact 0008B boundary view while assigning
  new internal split boundaries the existing segment form, retains decisions, and recomputes the
  selected identity during plan validation. Validation now compares every derivable retained
  recognition overlap against the immutable compatibility baseline, including member order,
  structural key, specialization, strategy, dependencies, access regime, referenced bytes,
  alignment, representable boundary role, workspace geometry/role, and pointwise/indivisible
  topology. It separately retains the typed graph-publication boundary positions projected from
  existing logical-memory requirements and recomputes PUBLICATION versus PARTITION_WRITE without
  trusting the claimed decision role.
- Focused tests cover closed inventories, immutable facts, 1/8 units, 1/64 candidates, 256 attempts
  with a pending 257th, 63 profitability rejections, 384 total facts, 64 boundary positions, 8
  workspaces, vertical fusion, publication and diamond fan-out barriers, attempt-budget fallback,
  malformed retained plans, unchanged carrier projection, composite execution, schema identity,
  Class-File leakage, the non-winning-tie regression, and fresh selected-fused/direct and
  selected-fused/canonical-split performance distributions.
- No 0008E materialization, MATMUL, Trace, tuning/cache/measurement, public/shared contract,
  Runtime, emitter, schema, capability, build, architecture, conformance, or integration change was
  made. Java/tests remained frozen after the authoritative pass; the clean documentation review
  finalized Javadocs, the CPU guide, glossary impact, planning evidence/status, and documentation
  checks without repeating stable Java/performance evidence.

## Completion summary

- Implementation completed within the fixed production/test/path ceilings, including bounded
  enumeration, selection, retained facts, recomputed plan validation, carrier-compatible selected
  preparation, focused tests, and retained Class-File/no-leak evidence.
- Documentation-agent review: Complete in clean context
  `01a041eb-619f-7fb3-8ab7-a99daf2da3ab`; affected Javadocs, the CPU guide, glossary, this task,
  CPU master plan, and roadmap were finalized without executable behavior changes.
- Documentation impact: the guide and existing glossary entries now document the implemented
  legality/profitability/selection boundary, exact bounded heuristic, safe fallbacks, typed cold
  ownership, publication projection, fail-closed retained baseline validation, and evidence limits.
- Javadoc review: finalized all affected/new production types, record components, constructors,
  and methods for invariants, defensive ownership, nullability, limits, failures, projection, and
  selection semantics.
- Architecture/public/shared/build/conformance/integration impact: None; the change remains
  CPU-private cold Prepare analysis and changes no capability, dependency, schema, generated form,
  shared lifecycle, or Runtime behavior.
- Unresolved issues: None.
- Follow-up required: None. CPU 0008E is the sole Ready CPU row; no detailed 0008E specification
  exists.
- No staging, commit, or push occurred.

Status: Complete
