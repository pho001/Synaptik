# Task 0008: General-[scan body](../../../../glossary.md#backpropagation-through-time-and-recurrent-state)-bearing capability query

## Status

Complete; main accepted the bounded Planning0008 transport foundation on 2026-10-22 after final
independent Feynman APPROVE and the exact compatibility gates below. Authorized [Planning](../../../../glossary.md#term-planning) frontier:
None. [Compiler](../../../../glossary.md#term-compiler) [AD](../../../../glossary.md#autograd)/[capture](../../../../glossary.md#term-capture-meanings) is the sole next [slice](../../../../glossary.md#slice), Draft/not yet briefed; no successor [Ready](../../../../glossary.md#term-ready) or launched.
Historical Ready verification (2026-10-22): Complete Planning0001–0007 and accepted Model0034/0035,
HEAD=origin/main below, empty index, unchanged three protected [hashes](../../../../glossary.md#term-hash) and all 110 [Model](../../../../glossary.md#term-model) source/test
hashes, inactive conflicts/sole writer. [Preflight](../../../../glossary.md#term-preflight) Feynman and main pinned criterion 2's minimal
applicability [guard](../../../../glossary.md#term-guard) without authority/[backend](../../../../glossary.md#backend) change. The corrected Ready packet had 177 lines,
11.6 KB, all seven integration fields and dated verification. No historical Model re-audit.

## Change class

Class C — public record-component compatibility and cross-[module](../../../../glossary.md#term-module) capability/[ownership](../../../../glossary.md#term-ownership-meanings) boundary.
Implements accepted authority, without an architecture or [dependency](../../../../glossary.md#term-dependency) change.

## Goal

Deliver a tested [Planning](../../../../glossary.md#term-planning) capability foundation that forwards a complete [immutable](../../../../glossary.md#term-immutable) compiled [scan
body](../../../../glossary.md#backpropagation-through-time-and-recurrent-state), ordered [descriptors](../../../../glossary.md#term-descriptor) and demand to a whole-[occurrence](../../../../glossary.md#term-occurrence) provider. Not [scan](../../../../glossary.md#term-cumulative-scan) execution.

## [Scope](../../../../glossary.md#term-scope)

Extend existing [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery) with fourth component
`Optional<CompiledScanBody> scanBody`; preserve public three-[argument](../../../../glossary.md#term-argument) empty-body construction.
Update query/provider/[ownership](../../../../glossary.md#term-ownership-meanings)/package [Javadoc](../../../../glossary.md#term-javadoc) and focused tests cohesively. Existing
`planning.capability` package only; no new production type, helper, [module](../../../../glossary.md#term-module) or [dependency](../../../../glossary.md#term-dependency).

## Non-goals

No [Model](../../../../glossary.md#term-model) edits, [Compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings)/[inference](../../../../glossary.md#term-inference)/[automatic differentiation](../../../../glossary.md#autograd) ([AD](../../../../glossary.md#autograd)), demand analysis or
rewrite propagation, real [backend](../../../../glossary.md#backend) support/route policy/ledgers, logical-memory changes, [Prepare](../../../../glossary.md#prepare),
[Runtime](../../../../glossary.md#term-runtime), execution, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) or [persistence](../../../../glossary.md#term-persistence). No second inference system or mixed-[owner](../../../../glossary.md#term-owner) body partitioning.

## Contracts

Read `AGENTS.md`, this brief, affected source/tests and root [scope](../../../../glossary.md#term-scope) index, then exact headings:

- Root [Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>.
- Foundational [modules/planning](../../../../architecture/contracts/foundational-modules.md#modulesplanning)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module); [planning glossary](../../../../glossary.md#term-planning))</span>
  and [Profile-free capability queries](../../../../architecture/contracts/foundational-modules.md#profile-free-capability-queries).
- Recurrent [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([General differentiable scan glossary](../../../../glossary.md#general-differentiable-scan))</span>,
  particularly `Source body, attachments and scoped identities`, `Canonical residual ports and demand`,
  `Traversal, empty domains and declared qualification`, `Qualification, performance and migration gates`.

Authority already names the optional body, old constructor and complete provider [qualification](../../../../glossary.md#term-qualification).
Do not reopen that decision; stop only on genuine new ambiguity or required scope expansion.

## Dependencies and integration

- Depends on: Planning0001–0007 Complete; accepted Model0034/0035 Complete, uncommitted.
- Conflicts with: query/provider/ownership API and shared scan/planning document edits; none active.
  Preserve all 136 preexisting dirty paths, including Model source/tests, <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [scan glossary](../../../../glossary.md#term-cumulative-scan); [planning glossary](../../../../glossary.md#term-planning); [Model glossary](../../../../glossary.md#term-model))</span>`THIRD_PARTY_NOTICES.md`,
  CPU cast oracle and untracked Superseded Compiler0010.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Parallel group: None; sole writer, serial clean code executor then independent clean code/docs review.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
  plus accepted UNCOMMITTED Model0034/0035, not merged predecessors.
- Integration order: accepted Model foundation → Planning0008 code → independent review → main
  compatibility checkpoint/acceptance; successors stay Draft pending separate authorization.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Integration [validation](../../../../glossary.md#term-validation): main's exact downstream [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names)/architecture/provider gate below.
- Shared-document integration owner: main; owns brief/result, Planning master, scan program and
  roadmap. Workers do not edit shared planning/authority documents.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>

## Files and symbols

Under `modules/planning/src/{main,test}/java/io/github/pho001/synaptik/planning/capability/`:

- Main: `OperationCapabilityQuery.java`, `BackendCapabilityProvider.java`,
  `BackendOwnerPlanning.java` (documentation only), `package-info.java`.
- Tests: `BackendCapabilityContractTest.java`, `BackendOwnerPlanningTest.java`.
  `BackendEligibilityTest.java` is unchanged [regression](../../../../glossary.md#term-regression) coverage.
- Read-only [Model](../../../../glossary.md#term-model): `CompiledScanBody`, `CompiledNode`, [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel), `GraphValue`,
  [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), `ScanKind`, `ScanAttrs`, focused `CompiledScanBodyTest`.

No accepted Model file should need changes. Report any required additional explanatory/shared
file edits to main before touching them; do not broaden [scope](../../../../glossary.md#term-scope) silently.

## Acceptance criteria

1. Exact public components: `(operation, inputs, outputs, scanBody)` with generic
   `Optional<CompiledScanBody>`, public four-argument canonical and retained three-argument
   constructor delegating to empty. Old construction equals explicit empty construction.
   Attachment participates in equality/hash/diagnostic text, not a wire format. Retain exact
   immutable body/operation/descriptor references and ordered descriptor snapshots/repeated positions.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [canonical glossary](../../../../glossary.md#term-canonical-form); [hash glossary](../../../../glossary.md#term-hash); [wire format glossary](../../../../glossary.md#term-wire-format); [immutable glossary](../../../../glossary.md#term-immutable); [operation glossary](../../../../glossary.md#operation); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
2. Constructor validation stays structural: non-null references/elements, immutable list snapshots,
   signature occurrence counts and attachment applicability. Preserve old three-argument validation
   order/messages. Null optional throws <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [elements glossary](../../../../glossary.md#term-element); [signature glossary](../../../../glossary.md#term-signature); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>`NullPointerException("scanBody")`; document its check order.
   After existing signature-count validation, a present body requires
   `operation.kind() == ScanKind.SCAN`; otherwise throw `IllegalArgumentException`. This is the
   same attachment-applicability boundary as `CompiledNode`, not header/body/descriptor inference.
   Test rejection for both ordinary (including ADD) and fixed recurrent kinds with present bodies,
   signature-count failure preceding applicability failure, and unchanged old three-argument failure
   order. Signature-valid missing-body <span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference))</span>`ScanKind.SCAN` questions remain representable through both
   three-argument and four-argument empty construction for provider false. Scan header/body/descriptor
   mismatches likewise remain representable for provider false; add no such inference or backend
   qualification to the query. The applicability guard prevents old CPU/Metal providers that ignore
   the attachment from accepting an ordinary supported operation carrying a scan body; no backend edits.<span data-glossary-navigation="true"> ([Scan glossary](../../../../glossary.md#term-cumulative-scan); [backend glossary](../../../../glossary.md#backend); [qualification glossary](../../../../glossary.md#term-qualification); [guard glossary](../../../../glossary.md#term-guard); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [scan body glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state))</span>
3. Existing Model construction owns graph closure, role coverage, increasing in-range demand and
   nested-body rejection (<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [graph glossary](../../../../glossary.md#graph))</span>`CompiledScanBody`), plus attachment kind/header/formal/role cardinalities
   (`CompiledNode`). Apart from criterion 2's applicability guard, do not duplicate/weaken those checks.
   Later Compiler supplies independently
   validated descriptors/body and full ordinary-use demand; Planning computes neither.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning))</span>
4. Provider interface remains exactly stable `backendId()` and [boolean](../../../../glossary.md#data-type--datatype) `supports(query)`, with
   deterministic immutable-configuration answers and existing null-query contract. Document that
   true qualifies complete header, every body occurrence/descriptor, residual/ordered binding/demand
   requirements and generated topology PLUS loop binder. Unknown/missing/malformed/unsupported
   occurrences or binders return false. Supported children alone never establish scan support.
   This task uses test providers, not a production backend implementation.<span data-glossary-navigation="true"> ([residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [binding glossary](../../../../glossary.md#term-binding); [topology glossary](../../../../glossary.md#term-topology))</span>
5. Forward a meaningful local body with ordered carry/step/invariant formals (including unused),
   repeated role results but distinct outer ports, full outer descriptors/gradient flags and
   nonempty/empty demand through <span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry); [invariant glossary](../../../../glossary.md#term-invariant); [gradient glossary](../../../../glossary.md#term-gradient))</span>`BackendOwnerPlanning.selectOwner`. Assert exact query/body/graph
   references at providers. Local numeric IDs may coincide with independently scoped outer IDs;
   do not compare/flatten scopes or transport source Tensor, Buffer, callback or Runtime objects.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Buffer glossary](../../../../glossary.md#term-buffer); [callback glossary](../../../../glossary.md#term-callback); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
6. Fixture-bounded test providers demonstrate deterministic true for a fully qualified question;
   false for missing body, header/formal/role or descriptor mismatch, unknown/unsupported child,
   and supported children with unsupported binder. Provider-false attachment/mismatch fixtures must
   use <span data-glossary-navigation="true"> ([Fixture glossary](../../../../glossary.md#term-fixture))</span>`ScanKind.SCAN`, never an illegal present body on a non-scan kind. No reusable test inference
   engine. Structurally
   impossible malformed graph/role/demand fixtures still fail in existing Model constructors;
   never bypass their guards. T=0/empty demand does not excuse unsupported semantics. All-false
   answers retain terminal no-eligible failure; no weakened intent or fallback.<span data-glossary-navigation="true"> ([engine glossary](../../../../glossary.md#term-engine))</span>
7. Preserve ordinary/fixed empty-body behavior, eligibility/availability/order/intent/preference,
   exact owner references and provider-exception propagation. Update exact reflection tests for
   four components/accessors and both constructors; retain the exact two-method provider assertion,
   rather than relaxing public-shape tests.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [shape glossary](../../../../glossary.md#shape))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker focused development command:

```bash
./gradlew :modules:planning:test --tests '*BackendCapabilityContractTest' --tests '*BackendOwnerPlanningTest' --tests '*BackendEligibilityTest'
```

One final affected-[module](../../../../glossary.md#term-module) [run](../../../../glossary.md#term-run) after executable stabilization; [Javadoc](../../../../glossary.md#term-javadoc) after final review edits:

```bash
./gradlew :modules:planning:test :modules:planning:javadoc
```

Main's source-matched compatibility [checkpoint](../../../../glossary.md#term-checkpoint), once after independent approval:

```bash
./gradlew :modules:compiler:compileTestJava :backends:cpu:compileTestJava :backends:metal:compileTestJava :modules:prepare:compileTestJava
./gradlew :testing:architecture-tests:test --tests '*PlanningDependencyContractTest' --tests '*RuntimeDependencyAndHotPathContractTest'
./gradlew :modules:compiler:test --tests '*GraphCompilationTest' --tests '*SourceGraphCapabilityTest' :backends:cpu:test --tests '*CpuCapabilityProviderTest' --tests '*CpuCapabilityProviderPublicShapeTest' :backends:metal:test --tests '*MetalCapabilityProviderTest'
```

Check owned local links/anchors, new-file whitespace and `git diff --check`. Review reuses successful
worker executable evidence unless code changes or a concrete evidence gap justifies rerun.
No root/native/performance campaign; one root test/Javadoc gate remains at the later low-level join.

## Follow-up

[Compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings)/[AD](../../../../glossary.md#autograd)/demand, [Prepare](../../../../glossary.md#prepare) projection and actual [CPU](../../../../glossary.md#term-cpu) support/execution are separate later
slices; this foundation neither implements them nor makes them [Ready](../../../../glossary.md#term-ready).

## Documentation and review impact

Clean implementation and independent code/docs review completed; see Result. Component/constructor/
accessor and provider/[ownership](../../../../glossary.md#term-ownership-meanings)/package [Javadoc](../../../../glossary.md#term-javadoc) distinguish accepted protocol, implemented transport
and unimplemented downstream support. Documentation rules, general and [API](../../../../glossary.md#term-application-programming-interface-api)/Javadoc profiles governed
that pass; glossary and explanatory status changes were targeted. Shared [planning](../../../../glossary.md#term-planning) evidence remains
main-owned. This mechanical closure changes no Java, Javadoc, terminology or authority.

## Result

- Implemented four existing production files: [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery), [`BackendCapabilityProvider`](../../../../glossary.md#backend-capability-provider--backendcapabilityprovider),
  `BackendOwnerPlanning` (docs only), `package-info`; two existing tests: `BackendCapabilityContractTest`
  and `BackendOwnerPlanningTest`. No new production type/[module](../../../../glossary.md#term-module)/[dependency](../../../../glossary.md#term-dependency)/[backend](../../../../glossary.md#backend)/[Model](../../../../glossary.md#term-model) change.
- Fourth `Optional<CompiledScanBody>` component, retained three-argument constructor, exact immutable
  references/ordered snapshots and post-signature non-scan-present-body guard satisfy criteria 1–3.
  Empty-body Scan and header/descriptor mismatch remain representable for provider false; complete
  body plus loop-binder qualification has no child-inference shortcut. No demand algorithm/numerical rule.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [immutable glossary](../../../../glossary.md#term-immutable); [signature glossary](../../../../glossary.md#term-signature); [scan glossary](../../../../glossary.md#term-cumulative-scan); [guard glossary](../../../../glossary.md#term-guard); [descriptor glossary](../../../../glossary.md#term-descriptor); [qualification glossary](../../../../glossary.md#term-qualification); [inference glossary](../../../../glossary.md#term-inference))</span>
- Plato: exact focused command above PASS, 35 tests. Final combined module test/Javadoc command above
  PASS, 73 tests, zero failures/errors/skips/Javadoc warnings; executable code then frozen (six files).<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Final independent Feynman APPROVE: no executable blocker/change. Final
  `./gradlew :modules:planning:javadoc` PASS, zero warnings; 15 targeted links/anchors, edited Markdown
  whitespace, rendered Javadoc and diff checks PASS; reused 73 module tests. Implementation updated
  four production Javadocs; reviewer changed only Query/package comments, plus eight narrow Markdown
  status docs: compile-api, public-api, glossary, general-scan, module-boundaries, foundational, recurrent,
  ADR0045. Factual status only; no authority/provider-capability change. Shared plans closed here.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>
- Main's three exact compatibility commands above PASS: compilation 5s, 16 tasks (8 executed/8 up-to-date;
  existing unchecked <span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>`CpuPartitionFinalizerTest` note); architecture 2 suites/4 tests, 1s, 27 tasks
  (2 executed/25 up-to-date); selected Compiler 2 suites/10 tests, CPU 2/30, Metal 1/19, 2s, 28 tasks
  (4 executed/24 up-to-date; existing incubator-vector warning). All tests zero failures/errors/skips.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [vector glossary](../../../../glossary.md#term-vector))</span>
- Main diff/index/base/three-protected-hash/110-Model-source-test freeze checks PASS at acceptance.
  Source/IR plus Planning transport ONLY; live Compiler body queries/capture/AD/demand, Prepare,
  real backend Scan support, NN and persistence remain future. No root/native/performance campaign;
  later low-level join retains the root gate. No unresolved acceptance issue; no successor authorization.<span data-glossary-navigation="true"> ([hash glossary](../../../../glossary.md#term-hash); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [Planning glossary](../../../../glossary.md#term-planning); [capture glossary](../../../../glossary.md#term-capture-meanings); [AD glossary](../../../../glossary.md#autograd); [Prepare glossary](../../../../glossary.md#prepare); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [persistence glossary](../../../../glossary.md#term-persistence))</span>

Status: Complete
