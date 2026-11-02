# Task Model0037: Length-aware general Scan authority

## Status

Complete — authority only, accepted by MAIN on 2026-10-24 after Hume's final overall independent
Class C APPROVE, zero open P1/P2. Source implementation has not started.

## Change class

Class C — semantic/API/AD/runtime/wire boundary decision. This execution is documentation only.

## Goal and scope

Enact exactly the user-selected whole-batch length-aware design in root/owning authority,
focused explanation and significant ADR; prepare only the immediate bounded Model foundation.
No source, tests, build, Javadoc, native, commit/push or successor launch. No GSD.

## Contracts

- [Root authority/scope index](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules),
  [Documentation authority and architecture updates](../../../../../ARCHITECTURE.md#documentation-authority-and-architecture-updates),
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- [Recurrent-scan: General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan),
  every subheading including the replaced final lengths gate and new Length-aware whole-batch scan;
  this remains the sole complete semantic owner.
- Compiler/autograd [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation),
  [Compile lifecycle / Bounded general-scan cold reconstruction](../../../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction).
- Foundational [modules/model](../../../../architecture/contracts/foundational-modules.md#modulesmodel),
  [modules/planning](../../../../architecture/contracts/foundational-modules.md#modulesplanning);
  Runtime [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle);
  Backend [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes).
- Extensions [Owned tensor Models](../../../../architecture/contracts/extensions-training.md#owned-tensor-models),
  [Canonical graph-model version 4](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-4),
  new [Canonical graph-model version 5](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-5).

## Dependencies and integration

- Depends on: Model0036 proposal delivery Complete; accepted Model0035, Planning0008,
  Compiler0011/0012, Prepare0010, CPU0030–0037, CP0012/0013, NN0031 and recorded joins Complete.
- Conflicts with: every scan API/semantic/AD/cold/runtime/wire writer; NN0032 fluent authority;
  shared root/contracts/planning/explanation. None active; sole writer, read-only review allowed.
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus
  297 accepted dirty paths / 2425 Git-listed files, **uncommitted**, not merged predecessors.
- Integration order: Model0037 independent review → MAIN acceptance → promote Model0038 Ready
  after verifying base/dependencies/conflicts again → separate clean source launch by MAIN.
- Integration validation: authority-document checkpoint below; later Model0038 compatibility gate.
- Shared-document integration owner: MAIN; this context has exclusive authority/planning delegation.

Ready/frontier verification before drafting, 2026-10-24: user “ok, pust se do prace” approves the
independently reviewed selected design; current request authorizes authority and first foundation
only. The current roadmap/master-plan and MAIN's supplied accepted evidence verify all predecessors
above Complete. HEAD/origin/base/index match, conflicts inactive and no parallel writer. Model0037
was the sole newly authorized **Ready authority frontier**; subsequent Hume approval and MAIN
acceptance complete this authority task. Model0036 remains historical proposal-only completion,
unchanged. MAIN has verified Model0038 Ready as the sole Model source frontier; launch is MAIN-owned.
Backup `/private/tmp/synaptik-length-scan-implementation.Dma3c9/{baseline.json,accepted.tar,accepted/}`
contains the accepted base; MAIN verified all hashes. Preserve all nonallowlisted work.

## Files and acceptance

Write allowlist: ARCHITECTURE.md; relevant headings in the six incorporated contracts; existing
docs/architecture/general-differentiable-scan.md; new ADR0049-length-aware-general-scan.md;
Model master-plan and new 0037/0038 briefs; docs/planning/{general-differentiable-scan,
length-aware-general-scan,roadmap}.md. No NN tracker update is needed: explicit held-gate links suffice.

- Exact selected header/overload, local/actual count distinction, bounded row/terminal rules,
  original AD restriction including all unused/non-gradient sequences, history roles/pruning,
  activity-before-load/math, raw identity/+0, active batch reduction then one shared ADD/time,
  no B×P bank and full pre-write lengths validation are consistent in the semantic owner.
- Other contracts only own boundaries/links. Selective v5/SYNGRF05/version5/opcode33 is accepted
  future wire; old v1–v4 bytes/default calls/identity streams stay unchanged. Record shape/hash/text
  compatibility is explicitly not implied by constructor-call compatibility.
- Distinguish accepted future behavior from current all-valid implementation everywhere touched.
  Preserve all approved exclusions; no `.rnn` migration or higher-order waiver.
- Source brief is ≤200 lines/15KB, has exact scoped files/guards/tests/Javadoc/validation, and
  cannot launch before independent authority approval and MAIN acceptance.

## Validation and documentation review

Documentation checkpoint only: local links/fragments, EOF newlines, trailing whitespace,
`git diff --check`, review delta against accepted backup, hash every nonallowlisted Git-listed
file, compare new path set, HEAD/origin/empty index and existing native binary hash.
No Gradle, Java, native build, generated-code or performance run in this phase.
Apply General, Architecture, Decision-record and Planning profiles plus Example format.
Targeted glossary search: General differentiable scan, Scan formal and primal binding,
Scan residual port and history demand, cotangent, valid lengths. Existing meanings stay accurate
for the all-valid baseline; row notation is defined locally, not a new public type. No glossary
write in this allowlist; later source documentation review must assess its API impact.
Javadoc impact: no Java/Javadoc edited now; required in Model0038 for changed contracts.
Independent Class C review is mandatory; Hume's read-only review owns no writes and reuses these
checks. MAIN alone accepts authority and advances status. No dependency rule changed, hence
architecture executable validation is deferred to the named source compatibility gate, not run now.

## Result

Drafted root/six-owner cutover, focused explanation, ADR0049, synchronized trackers and bounded
Model0038 source/IR packet. Numbers 0037/0038 and ADR0049 were free; codec currently ends at
opcode32, so outer opcode33 is selected for future v5. Completed Model0036 is untouched.
Validation: shared `check-docs.mjs authority-paths.json` passed (15 files, 555 links,
214 fragments, zero errors; EOF/whitespace included); `audit-tree.mjs authority-paths.json`
checked all 2425 baseline hashes: 12 allowed modifications, 3 allowed additions, no missing or
outside-allowlist changes, HEAD/origin unchanged, index empty, native hash unchanged. Current
Git-listed count 2428 / dirty count 300. `git diff --check` passed. Reviewed boundary/root delta
against accepted/; the source brief remains within 200 lines / 15KB. Hume's draft corrections are included:
singleton shared-row broadcast, explicit V3/V4 fixture selectors and Tensor-ID-allocation-free
cold row checking (ordinary local heap maps/arrays are allowed). No code/build claim.

Hume's final overall independent Class C verdict is APPROVE, zero open P1/P2; MAIN accepted
Model0037 on 2026-10-24. Hume also explicitly approved the factual correction requiring one
MAIN native-enabled repository test/Javadoc checkpoint for Model0038's multi-module guards,
with the existing matching library, both performance flags false and no rebuild/campaign.
MAIN granted status/gate-only closure within the same 15 paths, verified all predecessors/base/
conflicts and promoted Model0038 Ready. All seven integration fields remain recorded there.
No semantics, Java, tests, build or Javadoc changed in closure; source/AD/runtime/v5 remain unimplemented.
No unresolved authority issue or required Model0037 follow-up. MAIN separately launches Model0038;
no other successor is Ready. Final closure uses the same documentation/preservation checkers.

Status: Complete
