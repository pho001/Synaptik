# Task 0008: General-scan body-bearing capability query

## Status

Complete; main accepted the bounded Planning0008 transport foundation on 2026-10-22 after final
independent Feynman APPROVE and the exact compatibility gates below. Authorized Planning frontier:
None. Compiler AD/capture is the sole next slice, Draft/not yet briefed; no successor Ready or launched.
Historical Ready verification (2026-10-22): Complete Planning0001–0007 and accepted Model0034/0035,
HEAD=origin/main below, empty index, unchanged three protected hashes and all 110 Model source/test
hashes, inactive conflicts/sole writer. Preflight Feynman and main pinned criterion 2's minimal
applicability guard without authority/backend change. The corrected Ready packet had 177 lines,
11.6 KB, all seven integration fields and dated verification. No historical Model re-audit.

## Change class

Class C — public record-component compatibility and cross-module capability/ownership boundary.
Implements accepted authority, without an architecture or dependency change.

## Goal

Deliver a tested Planning capability foundation that forwards a complete immutable compiled scan
body, ordered descriptors and demand to a whole-occurrence provider. Not scan execution.

## Scope

Extend existing `OperationCapabilityQuery` with fourth component
`Optional<CompiledScanBody> scanBody`; preserve public three-argument empty-body construction.
Update query/provider/ownership/package Javadoc and focused tests cohesively. Existing
`planning.capability` package only; no new production type, helper, module or dependency.

## Non-goals

No Model edits, Compiler capture/inference/automatic differentiation (AD), demand analysis or
rewrite propagation, real backend support/route policy/ledgers, logical-memory changes, Prepare,
Runtime, execution, NN or persistence. No second inference system or mixed-owner body partitioning.

## Contracts

Read `AGENTS.md`, this brief, affected source/tests and root scope index, then exact headings:

- Root [Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules).
- Foundational [modules/planning](../../../../architecture/contracts/foundational-modules.md#modulesplanning)
  and [Profile-free capability queries](../../../../architecture/contracts/foundational-modules.md#profile-free-capability-queries).
- Recurrent [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan),
  particularly `Source body, attachments and scoped identities`, `Canonical residual ports and demand`,
  `Traversal, empty domains and declared qualification`, `Qualification, performance and migration gates`.

Authority already names the optional body, old constructor and complete provider qualification.
Do not reopen that decision; stop only on genuine new ambiguity or required scope expansion.

## Dependencies and integration

- Depends on: Planning0001–0007 Complete; accepted Model0034/0035 Complete, uncommitted.
- Conflicts with: query/provider/ownership API and shared scan/planning document edits; none active.
  Preserve all 136 preexisting dirty paths, including Model source/tests, `THIRD_PARTY_NOTICES.md`,
  CPU cast oracle and untracked Superseded Compiler0010.
- Parallel group: None; sole writer, serial clean code executor then independent clean code/docs review.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
  plus accepted UNCOMMITTED Model0034/0035, not merged predecessors.
- Integration order: accepted Model foundation → Planning0008 code → independent review → main
  compatibility checkpoint/acceptance; successors stay Draft pending separate authorization.
- Integration validation: main's exact downstream compilation/architecture/provider gate below.
- Shared-document integration owner: main; owns brief/result, Planning master, scan program and
  roadmap. Workers do not edit shared planning/authority documents.

## Files and symbols

Under `modules/planning/src/{main,test}/java/io/github/pho001/synaptik/planning/capability/`:

- Main: `OperationCapabilityQuery.java`, `BackendCapabilityProvider.java`,
  `BackendOwnerPlanning.java` (documentation only), `package-info.java`.
- Tests: `BackendCapabilityContractTest.java`, `BackendOwnerPlanningTest.java`.
  `BackendEligibilityTest.java` is unchanged regression coverage.
- Read-only Model: `CompiledScanBody`, `CompiledNode`, `CompiledGraphModel`, `GraphValue`,
  `ValueId`, `ScanKind`, `ScanAttrs`, focused `CompiledScanBodyTest`.

No accepted Model file should need changes. Report any required additional explanatory/shared
file edits to main before touching them; do not broaden scope silently.

## Acceptance criteria

1. Exact public components: `(operation, inputs, outputs, scanBody)` with generic
   `Optional<CompiledScanBody>`, public four-argument canonical and retained three-argument
   constructor delegating to empty. Old construction equals explicit empty construction.
   Attachment participates in equality/hash/diagnostic text, not a wire format. Retain exact
   immutable body/operation/descriptor references and ordered descriptor snapshots/repeated positions.
2. Constructor validation stays structural: non-null references/elements, immutable list snapshots,
   signature occurrence counts and attachment applicability. Preserve old three-argument validation
   order/messages. Null optional throws `NullPointerException("scanBody")`; document its check order.
   After existing signature-count validation, a present body requires
   `operation.kind() == ScanKind.SCAN`; otherwise throw `IllegalArgumentException`. This is the
   same attachment-applicability boundary as `CompiledNode`, not header/body/descriptor inference.
   Test rejection for both ordinary (including ADD) and fixed recurrent kinds with present bodies,
   signature-count failure preceding applicability failure, and unchanged old three-argument failure
   order. Signature-valid missing-body `ScanKind.SCAN` questions remain representable through both
   three-argument and four-argument empty construction for provider false. Scan header/body/descriptor
   mismatches likewise remain representable for provider false; add no such inference or backend
   qualification to the query. The applicability guard prevents old CPU/Metal providers that ignore
   the attachment from accepting an ordinary supported operation carrying a scan body; no backend edits.
3. Existing Model construction owns graph closure, role coverage, increasing in-range demand and
   nested-body rejection (`CompiledScanBody`), plus attachment kind/header/formal/role cardinalities
   (`CompiledNode`). Apart from criterion 2's applicability guard, do not duplicate/weaken those checks.
   Later Compiler supplies independently
   validated descriptors/body and full ordinary-use demand; Planning computes neither.
4. Provider interface remains exactly stable `backendId()` and boolean `supports(query)`, with
   deterministic immutable-configuration answers and existing null-query contract. Document that
   true qualifies complete header, every body occurrence/descriptor, residual/ordered binding/demand
   requirements and generated topology PLUS loop binder. Unknown/missing/malformed/unsupported
   occurrences or binders return false. Supported children alone never establish scan support.
   This task uses test providers, not a production backend implementation.
5. Forward a meaningful local body with ordered carry/step/invariant formals (including unused),
   repeated role results but distinct outer ports, full outer descriptors/gradient flags and
   nonempty/empty demand through `BackendOwnerPlanning.selectOwner`. Assert exact query/body/graph
   references at providers. Local numeric IDs may coincide with independently scoped outer IDs;
   do not compare/flatten scopes or transport source Tensor, Buffer, callback or Runtime objects.
6. Fixture-bounded test providers demonstrate deterministic true for a fully qualified question;
   false for missing body, header/formal/role or descriptor mismatch, unknown/unsupported child,
   and supported children with unsupported binder. Provider-false attachment/mismatch fixtures must
   use `ScanKind.SCAN`, never an illegal present body on a non-scan kind. No reusable test inference
   engine. Structurally
   impossible malformed graph/role/demand fixtures still fail in existing Model constructors;
   never bypass their guards. T=0/empty demand does not excuse unsupported semantics. All-false
   answers retain terminal no-eligible failure; no weakened intent or fallback.
7. Preserve ordinary/fixed empty-body behavior, eligibility/availability/order/intent/preference,
   exact owner references and provider-exception propagation. Update exact reflection tests for
   four components/accessors and both constructors; retain the exact two-method provider assertion,
   rather than relaxing public-shape tests.

## Validation

Worker focused development command:

```bash
./gradlew :modules:planning:test --tests '*BackendCapabilityContractTest' --tests '*BackendOwnerPlanningTest' --tests '*BackendEligibilityTest'
```

One final affected-module run after executable stabilization; Javadoc after final review edits:

```bash
./gradlew :modules:planning:test :modules:planning:javadoc
```

Main's source-matched compatibility checkpoint, once after independent approval:

```bash
./gradlew :modules:compiler:compileTestJava :backends:cpu:compileTestJava :backends:metal:compileTestJava :modules:prepare:compileTestJava
./gradlew :testing:architecture-tests:test --tests '*PlanningDependencyContractTest' --tests '*RuntimeDependencyAndHotPathContractTest'
./gradlew :modules:compiler:test --tests '*GraphCompilationTest' --tests '*SourceGraphCapabilityTest' :backends:cpu:test --tests '*CpuCapabilityProviderTest' --tests '*CpuCapabilityProviderPublicShapeTest' :backends:metal:test --tests '*MetalCapabilityProviderTest'
```

Check owned local links/anchors, new-file whitespace and `git diff --check`. Review reuses successful
worker executable evidence unless code changes or a concrete evidence gap justifies rerun.
No root/native/performance campaign; one root test/Javadoc gate remains at the later low-level join.

## Follow-up

Compiler capture/AD/demand, Prepare projection and actual CPU support/execution are separate later
slices; this foundation neither implements them nor makes them Ready.

## Documentation and review impact

Clean implementation and independent code/docs review completed; see Result. Component/constructor/
accessor and provider/ownership/package Javadoc distinguish accepted protocol, implemented transport
and unimplemented downstream support. Documentation rules, general and API/Javadoc profiles governed
that pass; glossary and explanatory status changes were targeted. Shared planning evidence remains
main-owned. This mechanical closure changes no Java, Javadoc, terminology or authority.

## Result

- Implemented four existing production files: `OperationCapabilityQuery`, `BackendCapabilityProvider`,
  `BackendOwnerPlanning` (docs only), `package-info`; two existing tests: `BackendCapabilityContractTest`
  and `BackendOwnerPlanningTest`. No new production type/module/dependency/backend/Model change.
- Fourth `Optional<CompiledScanBody>` component, retained three-argument constructor, exact immutable
  references/ordered snapshots and post-signature non-scan-present-body guard satisfy criteria 1–3.
  Empty-body Scan and header/descriptor mismatch remain representable for provider false; complete
  body plus loop-binder qualification has no child-inference shortcut. No demand algorithm/numerical rule.
- Plato: exact focused command above PASS, 35 tests. Final combined module test/Javadoc command above
  PASS, 73 tests, zero failures/errors/skips/Javadoc warnings; executable code then frozen (six files).
- Final independent Feynman APPROVE: no executable blocker/change. Final
  `./gradlew :modules:planning:javadoc` PASS, zero warnings; 15 targeted links/anchors, edited Markdown
  whitespace, rendered Javadoc and diff checks PASS; reused 73 module tests. Implementation updated
  four production Javadocs; reviewer changed only Query/package comments, plus eight narrow Markdown
  status docs: compile-api, public-api, glossary, general-scan, module-boundaries, foundational, recurrent,
  ADR0045. Factual status only; no authority/provider-capability change. Shared plans closed here.
- Main's three exact compatibility commands above PASS: compilation 5s, 16 tasks (8 executed/8 up-to-date;
  existing unchecked `CpuPartitionFinalizerTest` note); architecture 2 suites/4 tests, 1s, 27 tasks
  (2 executed/25 up-to-date); selected Compiler 2 suites/10 tests, CPU 2/30, Metal 1/19, 2s, 28 tasks
  (4 executed/24 up-to-date; existing incubator-vector warning). All tests zero failures/errors/skips.
- Main diff/index/base/three-protected-hash/110-Model-source-test freeze checks PASS at acceptance.
  Source/IR plus Planning transport ONLY; live Compiler body queries/capture/AD/demand, Prepare,
  real backend Scan support, NN and persistence remain future. No root/native/performance campaign;
  later low-level join retains the root gate. No unresolved acceptance issue; no successor authorization.

Status: Complete
