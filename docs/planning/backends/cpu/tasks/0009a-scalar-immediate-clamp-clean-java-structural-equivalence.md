# Task 0009A: Scalar-Immediate and Clamp Generated Support and Clean-Java Semantic Closure

## Status

Complete

## Goal

Close generated CPU support for every one of the 256 finite current scalar-immediate and clamp
forms. The closure proves exact finite-form provenance, generation/load/invocation, independently
authored optimal typed clean-Java semantic execution, applicable carrier/type/access/strategy
coverage, deterministic evidence identity, hot-path hygiene, and explicit rejected boundaries.
It does not make literal control-flow-graph equality a gate.

The earlier all-256 formal structural-equivalence claim is deliberately narrowed. Equivalent
generated and `javac` BFLOAT16 loops may place guard and latch branches differently, so literal CFG
equality is unsound. A general CFG-isomorphism engine, full abstract interpreter, arbitrary-bytecode
proof, or mutation suite for every formal schema dimension is outside this support-closure task.

## Scope

- Retain the exact 256-form inventory and source-derived provenance from CPU 0008Q1. For each form,
  verify the exact operation/form, typed carrier roles, access regime, selected strategy,
  materialization disposition, schema, deterministic class hash, and evidence-key identity.
- Generate, load, and invoke every form. Compare its result over normal values, meaningful ranges,
  and applicable NaN, infinity, signed-zero, subnormal, overflow, underflow, and clamp-bound edge
  values against an independently authored optimal typed clean-Java counterpart. The counterpart
  must not call generated code, CPU lowering, or a reference kernel.
- Cover every semantically and technically applicable scalar, vector, parallel-scalar, and
  parallel-vector selected mode; every current meaningful data type/carrier combination; and both
  contiguous and non-contiguous access forms. Record an exact not-applicable or rejected boundary
  where the current provider/preparer does not admit a form.
- Inspect the generated Class-File hot path for no Synaptik helper leakage, allocation, boxing,
  reflection, method-handle/dynamic invocation, string/map dispatch, generic carrier dispatch, or
  avoidable virtual/interface dispatch. Review representative generated/direct comparisons and
  decompilation proportionately against the clean-Java algorithm, hot-loop/dataflow, and
  avoidable-overhead oracle required by `ARCHITECTURE.md`.
- Promote `CURRENT_GENERATED_VS_CLEAN_JAVA_STRUCTURAL_EQUIVALENCE` only for an exact projection
  supported by a non-tautological comparator. Keep every other structural row at
  `PARTIAL_CLASSFILE_NO_CLEAN_JAVA_ORACLE`; partial structural evidence neither blocks support and
  semantic closure nor may be mislabelled complete equivalence.
- Preserve deterministic resource schemas, hashes, row accounting, and unsupported/rejected
  boundaries. Performance dispositions remain `PARTIAL_NO_REPRESENTATIVE_BENCHMARK` in this task.

## Out of scope

- Production Java, generator/lowering behavior, cache identity, capability/selection policy,
  Model/Compiler/Prepare/Runtime contracts, arbitrary immediate/bound bit enumeration, and new
  materialization policy.
- A general JVM bytecode verifier, literal CFG isomorphism, arbitrary bytecode proof, full abstract
  interpretation, or negative mutation controls for every possible structural-schema dimension.
- Full exhaustive performance validation, any performance disposition promotion, or repair of the
  unrelated failed CPU 0008I loss fork.

## Architecture references and constraints

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md): CPU prepare owns lowering and route selection;
  directly generated code must preserve the same specialized clean-Java semantic algorithm,
  hot-loop/dataflow shape, and avoidable-overhead profile. Proportionate semantic tests,
  Class-File/decompilation inspection, hygiene checks, and generated-versus-direct comparisons are
  required; byte-for-byte identity, `javac` identity, or identical JIT assembly is not.
- [CPU 0008Q1](0008q1-finite-scalar-immediate-clamp-matrix.md) supplies finite semantic fixtures
  and provenance. [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md) owns the
  wider support inventory, truthful disposition promotion, and ordered closure frontier.
- Evidence is test-only and must not change CPU production identity, provider capability, route
  admission, selection, or runtime behavior.

## Package impact

Existing test-only package:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` owns the finite inventory,
  independent clean-Java counterparts, execution/hygiene evidence, and any narrow structural
  comparator.

No production package or type changes.

## Affected files and maximum scope

Expected paths:

- Existing scalar-immediate/clamp matrix, clean-Java oracle, structural/hygiene, and coverage
  checkpoint test sources.
- Existing checked inventory, evidence ledger, disposition registry, and gap-matrix resources only
  when a row-exact result requires synchronization.
- This task, CPU 0009, CPU master plan, and roadmap.

Maximum: nine CPU test/evidence paths and four planning paths. Zero production Java, public
Javadoc, backend-guide, glossary, Gradle, architecture, conformance, integration, or performance
evidence paths. If more paths, a generator change, or a new support family is needed, stop and
create a narrow Draft follow-up.

## Acceptance criteria

- All 256 finite current forms have exact inventory/provenance identity and successful generated
  class generation, loading, and invocation.
- Every form has an independent optimal typed clean-Java semantic counterpart and passes its
  applicable ordinary, range, and edge-value execution cases.
- The evidence records the exact carrier/type/access/strategy/materialization applicability for
  scalar, vector, parallel-scalar, and parallel-vector modes, with explicit rejected or
  not-applicable boundaries rather than inferred coverage.
- Generated Class-File hot paths pass direct-hygiene checks for no helper leakage, allocation,
  boxing, reflection, method-handle/dynamic invocation, string/map dispatch, generic carrier
  dispatch, or avoidable virtual/interface dispatch.
- Hashes, schemas, evidence keys, resource parsing, and row accounting are deterministic. Exact
  promoted structural projections have a non-tautological comparator; all other rows remain
  `PARTIAL_CLASSFILE_NO_CLEAN_JAVA_ORACLE` without preventing this support/semantic closure.
- No performance row is promoted. CPU 0008I loss fork 0 remains unrelated, `NON_PASSING`, and
  fail-closed; no full performance claim is made.
- A separate documentation-focused pass finalizes the planning/evidence review and records
  reasoned no-change conclusions for public Javadoc, guides, glossary, architecture, ADRs,
  architecture/conformance/integration tests, and Gradle.

## Tests / validation

Implementation context runs focused CPU tests for the 256-form inventory, generation/load/
invocation, clean-Java semantic counterparts, edge/range cases, selected strategy/access/carrier
matrix, deterministic resource/hash accounting, unsupported boundaries, and hot-path hygiene. It
runs a narrow structural comparator only for projections it can genuinely prove. It also runs
`git diff --check`.

The documentation context reuses successful Java evidence unless it changes executable behavior;
it validates Markdown links, anchors, fences, exact path scope, status/dependency synchronization,
and whitespace. Repository-wide validation is deferred to CPU 0009G/CI. No full performance run is
required or promoted by 0009A.

## Dependencies and follow-up tasks

- Complete CPU 0008Q1 and its documentation pass; current CPU 0009 inventory/evidence resources.
- CPU 0009A is complete. CPU 0009B is the next ordered child, but remains `Draft`; under the
  Planning Guide it needs its own detailed specification only when it becomes the active frontier.

## Architecture impact

Expected impact: None. Stop for an architecture, dependency, production-selection, schema, or
generator change.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU task 0009A. Read AGENTS.md,
ARCHITECTURE.md, docs/architecture/current-architecture-plan.md, the Planning Guide, CPU master
plan, CPU 0008Q1, CPU 0009, and this task. Implement the finite scalar-immediate/clamp generated
support and clean-Java semantic closure exactly as specified. Do not alter production behavior or
claim general formal bytecode/CFG equivalence. Promote structural evidence only where a
non-tautological comparator proves its exact projection; leave all other rows partial. Do not run
or promote full performance evidence. Stop for scope or architecture conflict. Do not commit or
push. After stable focused evidence, use a separate clean documentation-focused context to finalize
documentation review and evidence before marking this task Complete.
```

## Local decisions

- CPU 0008Q1 remains the finite semantic fixture and provenance authority. This task closes support
  and clean-Java semantic execution for that exact finite set; it neither broadens the immediate
  domain nor changes production identity facts.
- A literal generated-versus-`javac` CFG comparison is not used as an equivalence test because
  equivalent BFLOAT16 loops can place guard/latch branches differently. A structural promotion is
  valid only when its comparator proves a specific projection independently of the artifact under
  test.
- The architecture oracle remains unchanged: generated code is reviewed against optimal clean Java
  for semantic algorithm, hot-loop/dataflow, and avoidable overhead, with evidence proportional to
  risk.

## Known limitations

0009A makes no arbitrary-immediate, all-backend, formal-general-bytecode, or full performance
claim. Structural rows without a genuinely supported comparator remain partial. The retained loss
performance fork is separate debt and does not block this support closure.

## Validation evidence

The independently approved implementation evidence recorded all 256 finite scalar-immediate/clamp
forms with exact inventory/provenance identity, generation, loading, invocation, independently
`javac`-compiled typed clean-Java counterparts, and bit-exact execution over full, empty,
nonzero-start, and tail ranges. Outside-range sentinels remained unchanged. Active exact-member
closed invocation/overhead allowlist checks passed for both generated and clean-Java artifacts.

All `*ScalarImmediateClamp*` tests passed (28 tests, including one existing expected skip).
`CpuGeneratedCoverageCheckpointTest` passed (8 tests), as did
`CpuGeneratedCoverageEvidenceTest` and `CpuGeneratedCoverageUnsupportedBoundaryTest`.
Inventory/disposition checks and `git diff --check` passed. Independent review found no findings
and returned `APPROVED`.

The documentation-focused pass reviewed the final dirty diff, this task, CPU 0009, the CPU master
plan, roadmap, final affected tests/resources, predecessor 0008Q1, and the required architecture,
planning, and General/Planning documentation contracts. It made no executable edit and therefore
reused the implementation Java evidence under the documentation rules.

## Implementation notes

- The 256 exact forms are a practical support and clean-Java semantic closure, not a formal
  control-flow-graph or full structural-equivalence proof. All 17,463 generated inventory rows,
  including these forms, retain `PARTIAL_CLASSFILE_NO_CLEAN_JAVA_ORACLE`; all retain
  `PARTIAL_NO_REPRESENTATIVE_BENCHMARK`.
- The exact canonical inventory is 17,636 rows: 17,463 generated rows and 173 exact rejections.
  No separate semantic-ledger row was added: the checked inventory/checkpoint and existing
  pointwise semantic manifest account for exactly the 256 forms.
- No production code, API, schema, lowering, preparation, selection, runtime behavior, or
  performance disposition changed.

## Completion summary

- Completed changes: finalized the planning record for the independently approved 256-form
  scalar-immediate/clamp generated-support and clean-Java semantic closure. The historical
  filename is retained, while the title and completion evidence make no full structural-equivalence
  claim.
- Files changed: this task, CPU 0009, the CPU master plan, and the roadmap.
- Validation: reused the successful 28-test `*ScalarImmediateClamp*` run (one expected skip),
  8-test `CpuGeneratedCoverageCheckpointTest` run, `CpuGeneratedCoverageEvidenceTest`, and
  `CpuGeneratedCoverageUnsupportedBoundaryTest`; checked inventory/dispositions, Markdown
  links/anchors/fences/newlines/whitespace, status/dependency synchronization, intended paths,
  and `git diff --check`.
- Documentation review: General and Planning profiles applied. Public/API and test Javadocs,
  glossary, architecture documents and ADRs, architecture/conformance/integration tests, Gradle,
  and guides need no change: this is test-only evidence and planning finalization, with no public
  API, behavior, terminology, architecture, build, or executable-contract change. Existing test
  comments/Javadocs accurately describe the clean-Java oracle and partial structural boundary.
- Unresolved issues: none for 0009A. CPU 0009B is the next ordered `Draft` child; its detailed
  specification is deferred until it is the active frontier. Broader structural and representative
  performance closure remain explicitly partial work for CPU 0009B--0009G.

Status: Complete
