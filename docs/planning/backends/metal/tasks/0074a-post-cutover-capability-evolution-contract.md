# Task 0074A: Post-cutover capability evolution contract

## Status

Complete — the user-approved contract distinction was implemented and independently reviewed
with zero P0/P1/P2 findings. It permits later qualified provider additions but does not itself
grant [FLOAT32](../../../../glossary.md#term-float32) or low-[precision](../../../../glossary.md#term-precision) [EXP](../../../../glossary.md#term-expx) support.

## Change class

Class C — clarifies an authoritative cross-[module](../../../../glossary.md#term-module) capability and evidence contract before a new
[Metal](../../../../glossary.md#term-metal) execution route may be admitted.

## Goal

Distinguish the frozen [ACCELERATOR](../../../../glossary.md#term-accelerator) provider-answer [snapshot](../../../../glossary.md#term-snapshot) at [Model](../../../../glossary.md#term-model) 0032 cutover from the
evolving profile-free current provider and its current ledger. Define an explicit gate for a
later new [occurrence](../../../../glossary.md#term-occurrence): independently reported [dtype](../../../../glossary.md#data-type--datatype)-specific support, qualified [backend route](../../../../glossary.md#backend-route),
current-provider ledger update, [conformance](../../../../glossary.md#term-conformance) tests, and compatibility [identity](../../../../glossary.md#term-identity-meanings) where behavior
changes. Resolve the wording that presently forbids the [FLOAT32](../../../../glossary.md#term-float32)-only [EXP](../../../../glossary.md#term-expx) delta in Task 0074.

## [Scope](../../../../glossary.md#term-scope)

- Clarify the cutover-only preservation language in `ARCHITECTURE.md` and the owning backend
  contract. Preserve the historical v1 ledger unchanged; make the v2 ledger a snapshot of actual
  current provider answers, not a permanently identical projection of v1.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- State that a new FLOAT32 capability never implies BFLOAT16/FLOAT16 support. Each low answer
  requires its own true provider query and qualified custom route under the existing low-partition
  contract. Existing frozen supported low rows remain supported.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [partition glossary](../../../../glossary.md#partition))</span>
- Explain the distinction at the Prepare/backend boundary and record the decision in a concise
  ADR. Align affected capability documentation; do not create a runtime policy or certificate.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [runtime glossary](../../../../glossary.md#term-runtime); [certificate glossary](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions))</span>

## Non-goals

No production capability, provider-code, [kernel](../../../../glossary.md#kernel), native [ABI](../../../../glossary.md#term-application-binary-interface-abi), [schema](../../../../glossary.md#term-schema), numerical formula,
low-[partition](../../../../glossary.md#partition) route, ledger-row, or test-[fixture](../../../../glossary.md#term-fixture) change in this task. Task 0074 implements [F32](../../../../glossary.md#floating-point-precision-and-rounding)
[EXP](../../../../glossary.md#term-expx) only after this contract passes independent review; [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT16](../../../../glossary.md#term-float16) EXP follows separately.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants):
  authority index, Model semantics, current provider evidence, append-only compatibility, and
  low-partition route boundaries.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [partition glossary](../../../../glossary.md#partition))</span>
- [Backend contract — Profile-free backend capability and identity](../../../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [backend capability glossary](../../../../glossary.md#backend-capability); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span>:
  occurrence-specific capability, frozen cutover baseline, independently queried low types,
  fixed Prepare route, native safety.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [Prepare glossary](../../../../glossary.md#prepare))</span>
- [Runtime/Prepare contract — Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>:
  route selection precedes shared resource declarations; Runtime never repairs capability.

## Dependencies and integration

- Depends on: Model 0032 Complete; Metal 0069 Complete at its approved source checkpoint;
  explicit user decision approving the baseline/current distinction.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Metal glossary](../../../../glossary.md#term-metal); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Conflicts with: Task 0074 implementation; any concurrent root/backend contract, capability
  ledger, Metal provider/native, or shared explanatory-document edit.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [capability ledger glossary](../../../../glossary.md#capability-ledger))</span>
- Parallel group: None.
- Common base revision: N/A; serial worktree over `1cab0731b75c035f589a60f8b9922a767211e6c6`
  with preserved uncommitted 0069 evidence and partial 0074 source changes.
- Integration order: contract/ADR implementation → independent Class C documentation review →
  coordinator marks 0074A Complete and 0074 Ready → resume 0074 code.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Ready glossary](../../../../glossary.md#term-ready); [resume glossary](../../../../glossary.md#schema-serialization-and-checkpoint))</span>
- Integration validation: targeted architecture/documentation links and terminology, no executable
  code test rerun for documentation-only edits, <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`git diff --check`; Task 0074 owns provider/conformance
  and full integrated build validation of the later capability change.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- Shared-document integration owner: main coordination context; task worker owns only the
  specifically listed architecture/ADR documents, reviewer may finalize those after worker.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- Frontier verification (2026-10-09): Model 0032 and Metal 0069 are Complete at their recorded
  checkpoints, 0074 is Blocked pending this exact decision, no active Metal executor remains, and
  user direction resolves the missing policy choice. This brief is Ready at the sole serial Metal
  frontier; partial 0074 code is preserved but cannot be accepted until this contract lands.

## Files and symbols

- `ARCHITECTURE.md` — root current-versus-frozen capability [invariant](../../../../glossary.md#term-invariant).
- `docs/architecture/contracts/backend-execution.md` — owning scoped capability/[identity](../../../../glossary.md#term-identity-meanings) rule.
- `docs/architecture/runtime-prepare-backend-boundary.md` — explanatory boundary example.
- `docs/design/decisions/0027-post-cutover-capability-evolution.md` — concise rationale,
  rejected permanent freeze, consequences.

## Acceptance criteria

- Authority unambiguously treats v1 ACCELERATOR true/false answers as the immutable cutover
  baseline, while reviewed post-cutover provider additions update the current ledger and tests.<span data-glossary-navigation="true"> ([ACCELERATOR glossary](../../../../glossary.md#term-accelerator); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- New FLOAT32 support does not grant BFLOAT16/FLOAT16 support by descriptor proxy or implication;
  low-type support requires its own qualified custom implementation and query. No existing low
  capability is removed.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- The route/identity and fail-closed requirements remain unchanged. No new runtime layer or
  numerical proof/certificate is introduced.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [runtime glossary](../../../../glossary.md#term-runtime); [certificate glossary](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions))</span>
- Root, one owning scoped contract, explanatory document, ADR, glossary terminology check,
  links and independent Class C review agree. No production code or test fixture is edited.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [test fixture glossary](../../../../glossary.md#adr-ci-and-qualification))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker: inspect the exact contract headings and current ledger/provider tests, then validate
Markdown links/anchors, targeted glossary terms, document style, [scope](../../../../glossary.md#term-scope)-limited diff, and
`git diff --check`. Independent reviewer checks the authority change against current source
and task 0074 without repeating unchanged executable tests.

## Documentation and review impact

This is an architecture decision. Apply the architecture, decision-record, and general style
profiles under `docs/developer-guide/documentation/`. An independent clean review context is
mandatory; no Java [Javadoc](../../../../glossary.md#term-javadoc) changes are expected because this step edits no Java source.

## Result

Changed `ARCHITECTURE.md`, the owning `backend-execution.md` scoped contract, the explanatory
[Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare) boundary document, and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0027. The v1 cutover ledger and all production
code/tests remained untouched by this task. The worker validated scoped links, anchors,
terminology, whitespace, and `git diff --check`; the independent Class C reviewer resolved all
59 local links/anchors in the four documents, checked targeted glossary terms and v1's unchanged
diff, and independently passed `git diff --check`. No executable test was repeated for this
documentation-only decision; no Java [Javadoc](../../../../glossary.md#term-javadoc) changed. There are no unresolved 0074A findings.
Task 0074 still needs its separate provider/native/test implementation and review.

Status: Complete.
