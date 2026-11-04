# Task Model0037: Length-aware [general Scan](../../../../glossary.md#general-differentiable-scan) authority

**Editorial reading key (current context, not a status update).**

This records Model0037’s documentation-only authority decision, not implementation delivery; current
reference descriptions must not be read back into that checkpoint. The caller supplies actual
Tensors to `Scan.apply`, binding declared formals by position in a `ScanBody` step template. Carry
passes state between steps; sequences supply time slices; invariants stay unchanged. A leaf has no
producing expression, not necessarily stored values. Follow [source
binding](../../../../reference/tensor/general-scan.md#formals-actual-positions-and-source-closure)
for these positional boundaries: construction reads descriptions, not input values, and storage-free
expressions neither execute nor own result buffers.

ROW identifies a proved independent batch coordinate; SHARED lacks that coordinate, regardless of
equal sizes. Residuals name retainable internal values; demand selects needed histories, not
automatic storage. The semantic owner defines meaning; Compiler handles capture/AD, Prepare projects
bindings/demand, and providers realize supported routes without redefining semantics. Semantic
responsibility is not buffer ownership. Follow [length-aware
boundaries](../../../../reference/tensor/general-scan.md#length-aware-source-construction-and-qualified-execution)
for current qualification and value checks: a zero-length row preserves raw initial carry, emits +0
padding, and skips inactive loads/arithmetic.

## Status

Complete — authority only, accepted by MAIN on 2026-10-24 after Hume's final overall independent
Class C APPROVE, zero open P1/P2. Source implementation has not started.

## Change class

Class C — semantic/[API](../../../../glossary.md#term-application-programming-interface-api)/[AD](../../../../glossary.md#autograd)/[runtime](../../../../glossary.md#term-runtime)/wire boundary decision. This execution is documentation only.

## Goal and [scope](../../../../glossary.md#term-scope)

Enact exactly the user-selected whole-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) length-aware design in root/owning authority,
focused explanation and significant [ADR](../../../../glossary.md#term-architecture-decision-record-adr); [prepare](../../../../glossary.md#prepare) only the immediate bounded [Model](../../../../glossary.md#term-model) foundation.
No source, tests, build, [Javadoc](../../../../glossary.md#term-javadoc), native, [commit](../../../../glossary.md#term-commit)/push or successor launch. No GSD.

## Contracts

- [Root authority/scope index](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>,
  [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>,
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>,
  [Documentation authority and architecture updates](../../../../../ARCHITECTURE.md#documentation-authority-and-architecture-updates),
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- [Recurrent-scan: General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [General differentiable scan glossary](../../../../glossary.md#general-differentiable-scan))</span>,
  every subheading including the replaced final lengths gate and new Length-aware whole-batch scan;
  this remains the sole complete semantic owner.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [owner glossary](../../../../glossary.md#term-owner))</span>
- [Compiler](../../../../glossary.md#term-compiler)/[autograd](../../../../glossary.md#autograd) [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([automatic differentiation glossary](../../../../glossary.md#autograd))</span>,
  [Compile lifecycle / Bounded general-scan cold reconstruction](../../../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [cold reconstruction glossary](../../../../glossary.md#term-cold-reconstruction))</span>.
- Foundational [modules/model](../../../../architecture/contracts/foundational-modules.md#modulesmodel)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module); [model glossary](../../../../glossary.md#term-model))</span>,
  [modules/planning](../../../../architecture/contracts/foundational-modules.md#modulesplanning)<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>;
  Runtime <span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>[Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>;
  Backend <span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend))</span>[CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>.
- Extensions [Owned tensor Models](../../../../architecture/contracts/extensions-training.md#owned-tensor-models)<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor))</span>,
  [Canonical graph-model version 4](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-4)<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form); [graph glossary](../../../../glossary.md#graph))</span>,
  new [Canonical graph-model version 5](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-5).

## Dependencies and integration

- Depends on: Model0036 proposal delivery Complete; accepted Model0035, Planning0008,
  Compiler0011/0012, Prepare0010, CPU0030–0037, CP0012/0013, NN0031 and recorded joins Complete.
- Conflicts with: every scan API/semantic/AD/cold/runtime/wire writer; NN0032 fluent authority;
  shared root/contracts/planning/explanation. None active; sole writer, read-only review allowed.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [AD glossary](../../../../glossary.md#autograd); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus
  297 accepted dirty paths / 2425 Git-listed files, **uncommitted**, not merged predecessors.
- Integration order: Model0037 independent review → MAIN acceptance → promote Model0038 Ready
  after verifying base/dependencies/conflicts again → separate clean source launch by MAIN.
- Integration [validation](../../../../glossary.md#term-validation): authority-document [checkpoint](../../../../glossary.md#term-checkpoint) below; later Model0038 compatibility gate.
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN; this context has exclusive authority/[planning](../../../../glossary.md#term-planning) delegation.

[Ready](../../../../glossary.md#term-ready)/frontier verification before drafting, 2026-10-24: user “ok, pust se do prace” approves the
independently reviewed selected design; current request authorizes authority and first foundation
only. The current roadmap/master-plan and MAIN's supplied accepted evidence verify all predecessors
above Complete. HEAD/origin/base/index match, conflicts inactive and no parallel writer. Model0037
was the sole newly authorized **Ready authority frontier**; subsequent Hume approval and MAIN
acceptance complete this authority task. Model0036 remains historical proposal-only completion,
unchanged. MAIN has verified Model0038 Ready as the sole [Model](../../../../glossary.md#term-model) source frontier; launch is MAIN-owned.
Backup `/private/tmp/synaptik-length-scan-implementation.Dma3c9/{baseline.json,accepted.tar,accepted/}`
contains the accepted base; MAIN verified all [hashes](../../../../glossary.md#term-hash). Preserve all nonallowlisted work.

## Files and acceptance

Write allowlist: ARCHITECTURE.md; relevant headings in the six incorporated contracts; existing
docs/architecture/general-[differentiable](../../../../glossary.md#derivative-and-chain-rule)-[scan](../../../../glossary.md#term-cumulative-scan).md; new ADR0049-length-aware-general-scan.md;
[Model](../../../../glossary.md#term-model) master-plan and new 0037/0038 briefs; docs/[planning](../../../../glossary.md#term-planning)/{general-differentiable-scan,
length-aware-general-scan,roadmap}.md. No [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) tracker update is needed: explicit held-gate links suffice.

- Exact selected header/overload, local/actual count distinction, bounded row/terminal rules,
  original AD restriction including all unused/non-gradient sequences, history roles/pruning,
  activity-before-load/math, raw identity/+0, active batch reduction then one shared ADD/time,
  no B×P bank and full pre-write lengths validation are consistent in the semantic owner.<span data-glossary-navigation="true"> ([AD glossary](../../../../glossary.md#autograd); [gradient glossary](../../../../glossary.md#term-gradient); [identity glossary](../../../../glossary.md#term-identity-meanings); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [validation glossary](../../../../glossary.md#term-validation); [owner glossary](../../../../glossary.md#term-owner))</span>
- Other contracts only own boundaries/links. Selective v5/SYNGRF05/version5/opcode33 is accepted
  future wire; old v1–v4 bytes/default calls/identity streams stay unchanged. Record shape/hash/text
  compatibility is explicitly not implied by constructor-call compatibility.<span data-glossary-navigation="true"> ([hash glossary](../../../../glossary.md#term-hash))</span>
- Distinguish accepted future behavior from current all-valid implementation everywhere touched.
  Preserve all approved exclusions; no `.rnn` migration or higher-order waiver.
- Source brief is ≤200 lines/15KB, has exact scoped files/guards/tests/Javadoc/validation, and
  cannot launch before independent authority approval and MAIN acceptance.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## [Validation](../../../../glossary.md#term-validation) and documentation review

Documentation [checkpoint](../../../../glossary.md#term-checkpoint) only: local links/fragments, [EOF](../../../../glossary.md#term-eof) newlines, trailing whitespace,
`git diff --check`, review delta against accepted backup, [hash](../../../../glossary.md#term-hash) every nonallowlisted Git-listed
file, compare new path set, HEAD/origin/empty index and existing native [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) hash.
No [Gradle](../../../../glossary.md#term-gradle), Java, native build, generated-code or performance [run](../../../../glossary.md#term-run) in this phase.
Apply General, Architecture, Decision-record and [Planning](../../../../glossary.md#term-planning) profiles plus Example format.
Targeted glossary search: [General differentiable scan](../../../../glossary.md#general-differentiable-scan), [Scan formal and primal binding](../../../../glossary.md#scan-formal-and-primal-binding),
[Scan residual port and history demand](../../../../glossary.md#scan-residual-port-and-history-demand), [cotangent](../../../../glossary.md#cotangent), valid lengths. Existing meanings stay accurate
for the all-valid baseline; row notation is defined locally, not a new public type. No glossary
write in this allowlist; later source documentation review must assess its [API](../../../../glossary.md#term-application-programming-interface-api) impact.
[Javadoc](../../../../glossary.md#term-javadoc) impact: no Java/Javadoc edited now; required in Model0038 for changed contracts.
Independent Class C review is mandatory; Hume's read-only review owns no writes and reuses these
checks. MAIN alone accepts authority and advances status. No [dependency](../../../../glossary.md#term-dependency) rule changed, hence
architecture executable validation is deferred to the named source compatibility gate, not run now.

## Result

Drafted root/six-[owner](../../../../glossary.md#term-owner) cutover, focused explanation, ADR0049, synchronized trackers and bounded
Model0038 source/[IR](../../../../glossary.md#term-intermediate-representation-ir) packet. Numbers 0037/0038 and ADR0049 were free; [codec](../../../../glossary.md#term-codec) currently ends at
opcode32, so outer opcode33 is selected for future v5. Completed Model0036 is untouched.
[Validation](../../../../glossary.md#term-validation): shared `check-docs.mjs authority-paths.json` passed (15 files, 555 links,
214 fragments, zero errors; [EOF](../../../../glossary.md#term-eof)/whitespace included); `audit-tree.mjs authority-paths.json`
checked all 2425 baseline [hashes](../../../../glossary.md#term-hash): 12 allowed modifications, 3 allowed additions, no missing or
outside-allowlist changes, HEAD/origin unchanged, index empty, native [hash](../../../../glossary.md#term-hash) unchanged. Current
Git-listed count 2428 / dirty count 300. `git diff --check` passed. Reviewed boundary/root delta
against accepted/; the source brief remains within 200 lines / 15KB. Hume's draft corrections are included:
[singleton](../../../../glossary.md#term-singleton) shared-row broadcast, explicit V3/V4 [fixture](../../../../glossary.md#term-fixture) selectors and [Tensor](../../../../glossary.md#tensor)-ID-allocation-free
cold row checking (ordinary local [heap](../../../../glossary.md#term-heap) maps/arrays are allowed). No code/build claim.

Hume's final overall independent Class C verdict is APPROVE, zero open P1/P2; MAIN accepted
Model0037 on 2026-10-24. Hume also explicitly approved the factual correction requiring one
MAIN native-enabled repository test/[Javadoc](../../../../glossary.md#term-javadoc) [checkpoint](../../../../glossary.md#term-checkpoint) for Model0038's multi-[module](../../../../glossary.md#term-module) guards,
with the existing matching library, both performance flags false and no rebuild/campaign.
MAIN granted status/gate-only closure within the same 15 paths, verified all predecessors/base/
conflicts and promoted Model0038 [Ready](../../../../glossary.md#term-ready). All seven integration fields remain recorded there.
No semantics, Java, tests, build or Javadoc changed in closure; source/[AD](../../../../glossary.md#autograd)/[runtime](../../../../glossary.md#term-runtime)/v5 remain unimplemented.
No unresolved authority issue or required Model0037 follow-up. MAIN separately launches Model0038;
no other successor is Ready. Final closure uses the same documentation/preservation checkers.

Status: Complete
