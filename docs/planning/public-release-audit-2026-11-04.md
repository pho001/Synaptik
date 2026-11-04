# Public-[release](../glossary.md#term-release) readiness audit

## Purpose and status

Status: Complete — source audit, preserving repairs, independent review, joined checks and
isolated installed/source workflows accepted. Public [publication](../glossary.md#term-publication) remains Blocked by the [owner](../glossary.md#term-owner).
This audit supersedes private-[distribution](../glossary.md#term-distribution) assumptions for readiness assessment, but does not
select a licence, authorize public uploads or change numerical/[ownership](../glossary.md#term-ownership-meanings) contracts.

Common base revision: `564a9988349c259f6082b6905d98de24f2e216ff`.
The working repository is `/Users/phujka/IdeaProjects/Synaptik`; the old `/tmp` [snapshot](../glossary.md#term-snapshot) is not
a Git checkout. Findings refer to current working bytes, not earlier [sealed](../glossary.md#term-sealed) [release](../glossary.md#term-release) [artifacts](../glossary.md#term-artifact).

## Contracts and [admission](../glossary.md#term-admission)

Read `AGENTS.md`, the root authority index, Core invariants and the scoped headings for the
assigned workstream below. The six incorporated scoped contracts remain the only detailed
authority. A contradiction requiring architecture change is an explicit decision, not permission
to widen capability, invent another execution path or weaken [validation](../glossary.md#term-validation).

The coordinator verified the explicit user authorization, stable baseline, independent read
scopes and satisfied source/contract prerequisites. Each investigation is an [authorized frontier](../glossary.md#term-authorized-frontier).
Depends on: None. Conflicts with: None for read-only investigation. Parallel group:
`PublicReleaseAuditReadOnly`. Integration order: Any. Shared-document integration [owner](../glossary.md#term-owner):
coordinator. Classification: Class A investigation; any resulting behavioral repair is classified
separately, with isolated worktree implementation and independent review for Classes B/C.

## Workstreams and [scope](../glossary.md#term-scope)

| ID | Source scope | Exact scoped-contract headings | Audit focus |
|---|---|---|---|
| A1 | [Model](../glossary.md#term-model) and [Compiler](../glossary.md#term-compiler) | foundational-[modules](../glossary.md#term-module): modules/model, [Profile-free numerical semantics](../glossary.md#profile-free-numerical-semantics); compiler-[autograd](../glossary.md#autograd): modules/compiler, [Training graph](../glossary.md#training-graph) model, [Compile](../glossary.md#compile) [lifecycle](../glossary.md#lifecycle); recurrent-scan: [General differentiable scan](../glossary.md#general-differentiable-scan) | Public expressions, [shape](../glossary.md#shape)/[dtype](../glossary.md#data-type--datatype)/[layout](../glossary.md#layout) [inference](../glossary.md#term-inference-meanings), [constants](../glossary.md#term-constant), [gradients](../glossary.md#term-gradient), higher-order limits, Scan activity/[binding](../glossary.md#term-binding) and numerical boundaries. |
| A2 | [CPU](../glossary.md#term-cpu) and [OpenBLAS](../glossary.md#term-openblas) | [backend](../glossary.md#backend)-execution: CPU backend routes, Qualified [oneDNN](../glossary.md#term-onednn) CPU realization, [General Scan](../glossary.md#general-differentiable-scan) ordinary-unit realization, OpenBLAS provider | Generated [kernels](../glossary.md#kernel), preparation selection, low [precision](../glossary.md#term-precision), native safety, stale [identities](../glossary.md#term-identity-meanings), ordinary/Scan reuse and failure diagnostics. |
| A3 | [Metal](../glossary.md#term-metal) Java/native/build | backend-execution: Metal backend, [Concrete backend](../glossary.md#concrete-backend) modules | Supported route [matrix](../glossary.md#term-matrix-meanings), resource/[ABI](../glossary.md#term-application-binary-interface-abi)/code-image [validation](../glossary.md#term-validation), numerical boundaries, real-[device](../glossary.md#term-device)/native package prerequisites. |
| A4 | [Engine](../glossary.md#term-engine), [Prepare](../glossary.md#prepare), [Runtime](../glossary.md#term-runtime) | runtime-prepare-engine: modules/runtime, modules/prepare, modules/engine, Prepare lifecycle, Runtime lifecycle | [Admission](../glossary.md#term-admission), close/[leases](../glossary.md#term-lease), [rollback](../glossary.md#term-rollback), concurrent runs, storage [borrowing](../glossary.md#term-borrowing), mixed-[owner](../glossary.md#term-owner) transfer, [publication](../glossary.md#term-publication) and exception ordering. |
| A5 | [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) and Training | extensions-training: extensions/nn, Owned [tensor](../glossary.md#tensor) Models, modules/training, [Optimizer](../glossary.md#term-optimizer)/training lifecycle | Compute/repeated train/save/load integration, state [ownership](../glossary.md#term-ownership-meanings), [transactional update](../glossary.md#resource-transaction-and-rollback)/restore, numerical finiteness and optimizer invariants. |
| A6 | [Checkpoint](../glossary.md#term-checkpoint) and NN codecs | extensions-training: extensions/checkpoint, Retained NN checkpoint APIs, [Canonical](../glossary.md#term-canonical-form) [graph](../glossary.md#graph)-model versions | Untrusted [artifact](../glossary.md#term-artifact) bounds, corruption/[truncation](../glossary.md#term-truncation-toward-zero), [alias](../glossary.md#aliasing-and-arena)/state [integrity](../glossary.md#term-integrity), graph reconstruction, [atomic publication](../glossary.md#term-atomic-publication) and compatibility. Read-only overlap with A5 is intentional. |
| A7 | Data, Text, Config, [Planning](../glossary.md#term-planning), Tuning, other tools and architecture boundaries | extensions-training: extensions/data, extensions/text, extensions/onnx; foundational-modules: modules/config, modules/planning; backend-execution: Performance evidence and optimization tooling, Other tool scopes | Batching/[tokenizer](../glossary.md#frozen-whitespace-tokenizer) boundaries, [cache](../glossary.md#term-cache)/session [identity](../glossary.md#term-identity-meanings), capability truthfulness, dependencies and marker-only exclusions. |
| A8 | [Trace](../glossary.md#trace) and [distribution](../glossary.md#term-distribution) (coordinator) | foundational-modules: modules/trace; root: Testing requirements | Caller-observable diagnostics, disabled-[observer](../glossary.md#term-observer) behavior, exception/route correlation, legal/publication prerequisites, artifact content and clean consumer install. |

Investigators inspect actual production sources and focused tests, classify concrete consumer-visible
findings by severity and confidence, and distinguish intentional exclusions from defects.
Report exact path/[symbol](../glossary.md#term-symbol)/scenario, evidence, smallest preserving repair, affected contracts/docs,
and existing public [smoke](../glossary.md#term-smoke-test) commands. Do not [run](../glossary.md#term-run) builds/tests/linters/formatters during the
parallel investigation. Do not infer numerical correctness from catalog execution counts.

## [Dependency](../glossary.md#term-dependency) flow and repair rules

`A1..A8 -> confirmed findings -> isolated preserving repairs -> independent review -> joined verification`.
The coordinator retains integration [ownership](../glossary.md#term-ownership-meanings) and performs actual reproduction/[smoke](../glossary.md#term-smoke-test) evidence.
Parallel repairs need disjoint semantic/write ownership and their own [Ready](../glossary.md#term-ready) entries here before
launch. Existing wording/source-text/[opcode](../glossary.md#term-opcode)/default/count/wiring tests are removed rather than
re-pinned if incidental; preserve genuine numerical, transition, boundary and ownership coverage.
No [runtime](../glossary.md#term-runtime) fallback, value-based route selection, hidden [precision](../glossary.md#term-precision) conversion or new interpreter.
No unrelated [features](../glossary.md#term-feature), new [backend](../glossary.md#backend) or unfinished marker project is promoted into this [release](../glossary.md#term-release).

## Acceptance and verification

- Account for every implemented product and unsupported marker project, supported public workflow,
  generated/native boundary and legal/distribution gate.<span data-glossary-navigation="true"> ([distribution glossary](../glossary.md#term-distribution))</span>
- Reproduce confirmed release-critical failures through actual execution where possible; repair
  causes, retain meaningful regression coverage and observe the changed public path afterward.<span data-glossary-navigation="true"> ([release glossary](../glossary.md#term-release); [regression glossary](../glossary.md#term-regression))</span>
- Add typed producer-owned trace only for concrete diagnostic gaps; keep Trace passive/JDK-only,
  no graph/storage leaks, numerical selector, required observer or disabled-path allocation.<span data-glossary-navigation="true"> ([producer glossary](../glossary.md#term-producer); [trace glossary](../glossary.md#trace); [JDK glossary](../glossary.md#term-java-development-kit-jdk); [graph glossary](../glossary.md#graph); [observer glossary](../glossary.md#term-observer))</span>
- Run a final joined repository/native/Javadoc gate after ordered integration; record commands,
  skips and limits without relabeling historical failed commands.<span data-glossary-navigation="true"> ([Run glossary](../glossary.md#term-run); [Javadoc glossary](../glossary.md#term-javadoc))</span>
- Build matched versioned artifacts, inspect checksums/provenance/dependency closure, and execute
  fresh public CPU and real-device Metal consumers outside the checkout.<span data-glossary-navigation="true"> ([artifacts glossary](../glossary.md#term-artifact); [provenance glossary](../glossary.md#provenance); [dependency glossary](../glossary.md#term-dependency); [CPU glossary](../glossary.md#term-cpu); [device glossary](../glossary.md#term-device); [Metal glossary](../glossary.md#term-metal))</span>
- Publish an evidence-backed finding/disposition record, precise supported scope and remaining
  owner-controlled public-release gates. No public licence or trusted publisher identity is
  invented by the audit.<span data-glossary-navigation="true"> ([scope glossary](../glossary.md#term-scope); [owner glossary](../glossary.md#term-owner); [identity glossary](../glossary.md#term-identity-meanings))</span>

## Results

Investigation and execution receipts are recorded here as work completes. Historical source and
[artifact](../glossary.md#term-artifact) receipts in the production support [matrix](../glossary.md#term-review-matrix) remain historical and are not overwritten.

## Confirmed-finding repair frontiers

[Owner](../glossary.md#term-owner) decision: retain no public redistribution grant for now. Public [publication](../glossary.md#term-publication) remains
Blocked; completion of the technical audit and local [qualification](../glossary.md#term-qualification) does not grant publication rights.

R1–R14 repairs and R15/R16 reference migrations are integrated and locally accepted. The [Ready](../glossary.md#term-ready)
packets below record their original execution baseline, not an open implementation frontier:
`564a9988349c259f6082b6905d98de24f2e216ff`; source investigation and contract dependencies were satisfied.
Parallel group: `PublicReleaseRepairs`. Conflicts: only the shared generated-[identity](../glossary.md#term-identity-meanings) seam
assigned to R5 below; other write and semantic scopes are disjoint. Integration order: Any,
with R6 before R5 so the combined numerical identity join follows both implementations.
Shared-document integration owner: coordinator. Workers do not [run](../glossary.md#term-run) builds/tests/linters/formatters;
the coordinator exercises repros, integrates [commits](../glossary.md#commit-release-distribution-log-and-seed-meanings) and performs focused/final verification once.

| ID | Class | Exclusive write [scope](../glossary.md#term-scope) and acceptance |
|---|---|---|
| R1 | B | TrainingParameter restore and Training tests/[Javadoc](../glossary.md#term-javadoc): reject inconsistent full-[precision](../glossary.md#term-precision) logical/master bytes transactionally; retain low masters. |
| R2 | C | GraphSource/GraphArtifactCodec name [validation](../glossary.md#term-validation) and [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) OwnedModelSource/ModelArtifactCodec: eliminate quadratic namespace amplification without a new name/depth restriction; preserve wire/order/[aliases](../glossary.md#aliasing-and-arena); contextual invalidation and malformed-state errors. Own necessary [Module](../glossary.md#term-module) child-validation optimization only if required by the same import path. |
| R3 | B | tools/tuning WorkloadTuning and tests/Javadoc: session-only tuning performs no persistent-[cache](../glossary.md#term-cache) access; compatible configurations are authenticated against every exact [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) before result/cache publication. |
| R4 | C | [Engine](../glossary.md#term-engine)/[Prepare](../glossary.md#prepare)/[Runtime](../glossary.md#term-runtime) [lifecycle](../glossary.md#lifecycle) files and tests: [slice](../glossary.md#slice) sufficient-capacity caller storage to its established referenced prefix; close every acquired owner on bookkeeping failure; avoid allocating terminal shutdown snapshots; preserve [reentrant](../glossary.md#memory-management-atomic-counters-and-complexity)/concurrent failure replay. |
| R5 | C | [CPU](../glossary.md#term-cpu) statistical/[normalization](../glossary.md#term-normalization) working sites and references/tests: [F32](../glossary.md#floating-point-precision-and-rounding) uses F32 sites; retain overflowing square [infinity](../glossary.md#term-infinity) for F32/[F64](../glossary.md#floating-point-precision-and-rounding)/lows. Own selective generated compatibility updates for R5 and R6, including ordinary helpers in Scan; do not bump unchanged raw families. |
| R6 | C | CPU Pool2d/[Pool3d](../glossary.md#pool3d--ncdhw-pooling)/Attention exact integer divisors and references/tests: reuse existing exact-count divide emitter; preserve final low [narrowing](../glossary.md#term-narrowing), guards and saved/publication boundaries. Send changed identity-family facts to R5; do not edit its identity files. |
| R7 | C | [Compiler](../glossary.md#term-compiler) LossGradientRules and focused tests/[integration test](../glossary.md#term-integration-test): use categorical-[loss](../glossary.md#loss) shifted probability/[log](../glossary.md#term-log)-probability and exact [target](../glossary.md#term-target)/ignore guards, not ordinary [softmax](../glossary.md#softmax--log-softmax)'s distinct infinity formula. No [backend](../glossary.md#backend) backward path or changed [Model](../glossary.md#term-model) mathematics. |
| R8 | C | Model ScanBody plus Compiler ScanBodyInference/ScanGradientRules/SourceGraphReconstruction/BatchBindingPlan and focused tests: iterative deep-body traversal with unchanged encounter order; restore explicitly retained cold header exclusion only, never ordinary execution exclusion; preserve scoped [binding](../glossary.md#term-binding) diagnostic subject and actual B. |
| R9 | C | CPU integration/composition/native selector and native execute/wait paths, backend-owned typed diagnostic [DTO](../glossary.md#data-transfer-object--dto)/[observer](../glossary.md#term-observer) and tests: optional cold route/rejection/finalization diagnostics using existing [Trace](../glossary.md#trace) envelope; preserve no-observer path, numerical decisions and primary native execute failure with wait suppressed. No Engine/Runtime/Trace-module edits. |

R5 and R6 share no implementation edits: R5 is the sole generated-identity integration owner.
R2 owns NN source invalidation diagnostics and corrupt-values context; R1 changes only Training.
R4 owns Engine/Prepare/Runtime, while R9 owns only CPU. No worker edits shared architecture,
changelog, README, [release](../glossary.md#term-release) inventory, [planning](../glossary.md#term-planning) or user/[API](../glossary.md#term-application-programming-interface-api) guides. Each repair updates meaningful
Javadoc and deterministic behavioral [regression](../glossary.md#term-regression) coverage in its owned source/test scope.

### [Metal](../glossary.md#term-metal) repair join

R10 and R11 are additionally [Ready](../glossary.md#term-ready) at the same baseline, group and [validation](../glossary.md#term-validation) [checkpoint](../glossary.md#term-checkpoint).
Their contracts and write scopes are disjoint; integration order is R10 then R11. Coordinator
exclusively owns combined [shader](../glossary.md#term-shader) [authentication](../glossary.md#term-authentication) [metadata](../glossary.md#term-metadata) and coordinated compatibility [identities](../glossary.md#term-identity-meanings).

- **R10, Class C:** correct Conv2d/Conv3d contraction-then-bias in the shared ordinary/Scan/fused
  convolution shader and behavioral cancellation tests. No other shader or identity edits.<span data-glossary-navigation="true"> ([Conv2d glossary](../glossary.md#convolution--conv2d); [Conv3d glossary](../glossary.md#conv3d--ncdhw); [contraction glossary](../glossary.md#contraction-dimension); [bias glossary](../glossary.md#term-bias); [fused glossary](../glossary.md#specialization-and-fusion); [convolution glossary](../glossary.md#term-channels-first-convolution-layers); [cancellation glossary](../glossary.md#term-cancellation); [identity glossary](../glossary.md#term-identity-meanings))</span>
- **R11, Class C:** source-only active BOOL Scan validation, dense typed-control provider parity,
  FP8 rejection at storage/source/publication boundaries and empty-context lifetime checks.
  Preserve public indexing exception classes and native cause with a bounded invocation-owned
  Scan failure record (category, time, unit, operand/ordinal); no body replay/interpreter.
  Use existing run-private workspace transport, no new native export, global mutable error,
  payload capture or per-element diagnostic allocation. Existing ABI export signatures and
  Model numerical semantics are unchanged. Any extra workspace/layout meaning receives the
  coordinated append-only compatibility change owned by the coordinator.<span data-glossary-navigation="true"> ([BOOL glossary](../glossary.md#data-type--datatype); [FP8 glossary](../glossary.md#term-fp8); [publication glossary](../glossary.md#term-publication); [lifetime glossary](../glossary.md#term-lifetime); [run glossary](../glossary.md#term-run); [mutable glossary](../glossary.md#term-mutable); [payload glossary](../glossary.md#artifact-envelope-and-atomic-publication); [capture glossary](../glossary.md#term-capture-meanings); [element glossary](../glossary.md#term-element); [ABI glossary](../glossary.md#term-application-binary-interface-abi); [signatures glossary](../glossary.md#term-signature); [Model glossary](../glossary.md#term-model); [layout glossary](../glossary.md#layout))</span>

Mixed-low WHERE/comparison source-versus-authority discrepancies remain a separate explicit
coordinator disposition; these repairs do not broaden the declared Metal numerical domain.

### Remaining diagnostic and documentation joins

R12 is [Ready](../glossary.md#term-ready) after R4 integration (`769f19bdd`), using the current main integration base.
Class B/C: add an explicit opt-in [observer](../glossary.md#term-observer) overload to ordinary `Engine.prepareTuned`, reporting
typed [PREPARE](../glossary.md#prepare) fallback/rejection diagnostics (phase and closed reason category) at the existing
[producer](../glossary.md#term-producer). Preserve evidence-empty fallback, original failures and cleanup, no [cache](../glossary.md#term-cache)/[payload](../glossary.md#artifact-envelope-and-atomic-publication) leaks,
no string-based policy decisions and no no-observer diagnostic allocation. [Engine](../glossary.md#term-engine)-local passive
DTOs reuse the existing [Trace](../glossary.md#trace) envelope; no Trace/Config/[Runtime](../glossary.md#term-runtime) change or generic logger.

Documentation frontiers D1/D2 are Ready at the original baseline and may reconcile current
static source claims while code repairs proceed; they must not claim unrun repair acceptance.
D1 exclusively owns README, getting-started, public/training [API](../glossary.md#term-application-programming-interface-api) guidance and stale fixed-[RNN](../glossary.md#term-recurrent-neural-network-rnn)
[NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) public Javadocs. D2 exclusively owns production-[release](../glossary.md#term-release)-goal, roadmap and release-process.
Coordinator owns [backend](../glossary.md#backend) contracts/guides, debugging-trace, final audit findings/receipts,
changelog, source inventory and actual [artifact](../glossary.md#term-artifact) [qualification](../glossary.md#term-qualification). No public licence/upload is granted.

### Targeted boundary completion

R13/R14 are [Ready](../glossary.md#term-ready) at the current integrated repair base (`3c1b72792`), with source and contract
dependencies satisfied and independent write/semantic scopes. Parallel group:
`PublicReleaseBoundaryCompletion`. Integration order: R11 before R13; R14 independent.
Coordinator owns shared docs, compatibility [authentication](../glossary.md#term-authentication), verification and integration.

- **R13, Class C:** enforce the already-declared Metal BFLOAT16/FLOAT16 mixed-low exclusion
  for direct WHERE/comparison occurrences at provider and independent encoded/native admission.
  Preserve the fourteen qualified floating WHERE pairs, explicit casts and represented-value
  semantics. This corrects accidental over-admission; it grants no new numerical domain.
  R11 owns Scan/FP8 boundaries; do not change those semantics or shared identity constants.<span data-glossary-navigation="true"> ([Metal glossary](../glossary.md#term-metal); [BFLOAT16 glossary](../glossary.md#term-bfloat16); [FLOAT16 glossary](../glossary.md#term-float16); [admission glossary](../glossary.md#term-admission); [FP8 glossary](../glossary.md#term-fp8); [identity glossary](../glossary.md#term-identity-meanings); [constants glossary](../glossary.md#term-constant))</span>
- **R14, Class B:** bound and sanitize the displayed owned-source invalidation path without
  changing canonical names, legal depth, persisted bytes or source validation semantics.
  Preserve the captured failure category and actionable path prefix; add a behavioral invalidation
  regression for control characters and a very deep valid namespace.<span data-glossary-navigation="true"> ([canonical glossary](../glossary.md#term-canonical-form); [validation glossary](../glossary.md#term-validation); [regression glossary](../glossary.md#term-regression))</span>

Workers use isolated worktrees and [run](../glossary.md#term-run) no checks; the coordinator reproduces and verifies the
joined public paths and requests independent targeted source review.

### Joined-[checkpoint](../glossary.md#term-checkpoint) reference migrations

R15/R16 are [Ready](../glossary.md#term-ready) after the first joined gate, based at `0e9b12a39`; dependencies are the
integrated statistical working-site correction and observed reference mismatches. Group:
`PublicAuditReferenceMigration`. Disjoint test-only [owners](../glossary.md#term-owner); integration order Any; coordinator
owns source/contracts and the single affected/final rerun. Workers [run](../glossary.md#term-run) no checks.

- R15 owns trailing-normalization numerical evidence and the prepared RMS overflow regression.
  Preserve Model's binary32 named sites, exact point identities and mandatory square overflow;
  delete obsolete implementation/workspace assertions rather than force old stable/hypot behavior.<span data-glossary-navigation="true"> ([identities glossary](../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([normalization glossary](../glossary.md#term-normalization); [overflow glossary](../glossary.md#term-overflow); [regression glossary](../glossary.md#term-regression); [binary32 glossary](../glossary.md#floating-point-precision-and-rounding))</span>
- R16 owns the BatchNorm training numerical/performance reference. Reconcile independent binary32
  named rounding and centered-square/count equations for all published statistics. Never change
  production numerics or relax tolerances to satisfy a checksum.<span data-glossary-navigation="true"> ([rounding glossary](../glossary.md#term-rounding); [checksum glossary](../glossary.md#term-checksum))</span>

The first gate also exposed a stale native Scan-v1 coarse [guard](../glossary.md#term-guard) and an attention mixed-input
[F64](../glossary.md#floating-point-precision-and-rounding) [specialization](../glossary.md#term-specialization) guard; the coordinator corrects those small integration joins. The attempted
R8 cold empty-header restriction is withdrawn: implemented [graph](../glossary.md#graph)-[model](../glossary.md#term-model) v4/v5 explicitly admit
[carry](../glossary.md#term-carry)-free/zero-ROW sources in `extensions-training`, superseding the historical positive-ROW
baseline. Existing consumer [round-trip](../glossary.md#schema-serialization-and-checkpoint) tests remain; no constructor domain is narrowed.

## Finding disposition and coverage

The audit covered all seven assigned source areas, plus shared build, [distribution](../glossary.md#term-distribution) and [Trace](../glossary.md#trace)
[ownership](../glossary.md#term-ownership-meanings). Source review is bounded evidence, not an exhaustive proof of every legal [Model](../glossary.md#term-model)
geometry or [floating-point](../glossary.md#term-floating-point) result. Existing numerical acceptance and current [artifact](../glossary.md#term-artifact) acceptance
are separate from [operation](../glossary.md#operation)-catalog execution counts.

| Area | Finding and disposition | Preserved boundary |
|---|---|---|
| Model / [Compiler](../glossary.md#term-compiler) | R7 fixes categorical-[loss](../glossary.md#loss) [AD](../glossary.md#autograd)'s shifted formulas and exact [target](../glossary.md#term-target) guards; R8 makes valid deep Scan traversal iterative and retains scoped [binding](../glossary.md#term-binding) subjects and actual B. | Ordinary [softmax](../glossary.md#softmax--log-softmax)'s distinct [infinity](../glossary.md#term-infinity) rules, contribution order, Model IDs and constructor domains remain unchanged. |
| [CPU](../glossary.md#term-cpu) arithmetic | R5 preserves declared [binary32](../glossary.md#floating-point-precision-and-rounding) working sites and overflowing squares; [L2](../glossary.md#term-l2) point domains retain exact absolute values. R6 divides by exact integer contributor counts. | [F64](../glossary.md#floating-point-precision-and-rounding) stays F64; low values narrow only at declared [publication](../glossary.md#term-publication) sites; no [runtime](../glossary.md#term-runtime) numerical repair or blanket compatibility bump. |
| CPU/native diagnostics | R9 adds opt-in correlated cold selection/finalization facts and retains enqueue failure as primary when wait also fails. | No per-[run](../glossary.md#term-run) [observer](../glossary.md#term-observer) instrumentation, no numerical/route policy in Trace, no diagnostic [payload](../glossary.md#artifact-envelope-and-atomic-publication) retention of live objects. |
| [Metal](../glossary.md#term-metal) | R10 orders products before [bias](../glossary.md#term-bias); R11 fixes active [BOOL](../glossary.md#data-type--datatype)/control, [FP8](../glossary.md#term-fp8)/storage and empty-context boundaries, plus bounded native Scan failure context. R13 closes direct and nested mixed-low WHERE [admission](../glossary.md#term-admission). | Native ABI17/schema22 remain; Scan attribute v2 and coordinated identity45 reject stale plans. Explicit casts and qualified floating pairs remain supported. |
| [Engine](../glossary.md#term-engine) / [Prepare](../glossary.md#prepare) / Runtime | R4 slices valid excess-capacity caller storage to its referenced prefix and closes acquired [identities](../glossary.md#term-identity-meanings) on bookkeeping failure, without terminal shutdown snapshots. R12 observes actual tuning fallback/rejection. | Caller ownership, [reentrant](../glossary.md#memory-management-atomic-counters-and-complexity)/concurrent close, exact primary/cause/suppression behavior and empty-evidence fallback remain. Allocation-failure [rollback](../glossary.md#term-rollback) has source/focused-test evidence, not a claim of induced process [OOM](../glossary.md#term-out-of-memory-oom). |
| Training / [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) | R1 rejects unequal [F32](../glossary.md#floating-point-precision-and-rounding)/F64 logical/master restore bytes before mutation. R2/R14 preserve source invalidation with bounded sanitized displayed paths. | Low-[precision](../glossary.md#term-precision) master state remains independent; rejected restore is transactional; [canonical](../glossary.md#term-canonical-form) names and saved bytes are not sanitized or truncated. |
| [Checkpoint](../glossary.md#term-checkpoint) | R2 removes quadratic namespace [validation](../glossary.md#term-validation)/amplification, preserves canonical order and adds corrupt-state context. | Wire selection, [aliases](../glossary.md#aliasing-and-arena), legal names/depth, bounded readers and [atomic publication](../glossary.md#term-atomic-publication) remain. Zero-[carry](../glossary.md#term-carry)/zero-ROW v4/v5 sources remain admitted. |
| Tuning / other extensions | R3 avoids all persistent-[cache](../glossary.md#term-cache) access for session-only compatibility and authenticates decisions against every exact [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters). Data/Text had no confirmed [release](../glossary.md#term-release)-critical source defect; [CUDA](../glossary.md#term-cuda)/[ONNX](../glossary.md#term-onnx)/[CLI](../glossary.md#term-cli) remain non-publication placeholders. | No new discovery, executor, persistent Metal cache, [tokenizer](../glossary.md#frozen-whitespace-tokenizer)/model [inference](../glossary.md#term-inference), runtime fallback or public support inferred from a [module](../glossary.md#term-module) name. |

Independent source reviews approved the [lifecycle](../glossary.md#lifecycle), restore/namespace, tuning/Scan, numerical/count,
diagnostic and Metal corrections. Review findings caused additional repairs: L2 point-domain
[identity](../glossary.md#term-identity-meanings), bounded source-path display, native observer suppression, genuine authenticated nested
WHERE [regression](../glossary.md#term-regression) and the Scan-v2 coarse [guard](../glossary.md#term-guard). The incorrect proposed cold-header exclusion was
withdrawn against the implemented owning v4/v5 contract; positive consumer regressions were kept.
R15/R16 migrate independent test references to the retained Model sites without relaxing checks.

### Verification policy and remaining [release](../glossary.md#term-release) conditions

The coordinator reproduces public failures, runs the joined native-enabled `test build` and
affected [Javadoc](../glossary.md#term-javadoc) gate, and separately compiles/runs consumers. A rebuilt local [Maven](../glossary.md#term-maven) repository,
source archive and ad-hoc-signed native package are then extracted into a fresh directory and
consumed without checkout classes or a manual [transitive dependency](../glossary.md#term-transitive-dependency) list. [Qualification](../glossary.md#term-qualification) uses
`0.1.0-public-audit.20261104`, preserving the earlier [sealed](../glossary.md#term-sealed) `0.1.0` receipts.

Public [publication](../glossary.md#term-publication) remains blocked by the [owner](../glossary.md#term-owner)'s licence/redistribution decision. No upload,
trusted Developer-ID [signature](../glossary.md#term-signature), [notarization](../glossary.md#term-notarization), x86 low-[precision](../glossary.md#term-precision) qualification, native [FP8](../glossary.md#term-fp8) result
on this Arm machine, universal numerical envelope or performance SLA is claimed.
The [didactic documentation redesign](documentation-redesign.md) is queued as the separately
requested next work, including the whole-framework explanation, package atlas and complete
commented executable examples; it is not reported as implemented by this audit.

## Result

The joined native-enabled root `test build` and affected [Javadoc](../glossary.md#term-javadoc) gate passed. Source, standard
[Maven](../glossary.md#term-maven) publications and the ad-hoc-signed native package were rebuilt into a distinct
`0.1.0-public-audit.20261104` candidate. A fresh consumer resolved only `nn` and its published
transitive closure. The same public programs then passed against Maven-only classes/native
[artifacts](../glossary.md#term-artifact) and independently rebuilt extracted Java/native source, without checkout [runtime](../glossary.md#term-runtime) paths.
Exact coverage and counts are in the
[current support receipt](../backend-guide/production-support.md#public-readiness-audit-current-source-and-installed-candidate).

The initial gates and recoveries are retained, not rewritten as successes. Besides production
repair joins, native [fixture](../glossary.md#term-fixture) corrections preserved [canonical](../glossary.md#term-canonical-form) formals/typed controls, bound actual
materialized [fusion](../glossary.md#term-fusion) slots and asserted the now-observable native failure status. No production
numerical semantics or capability domain was changed to make those fixtures pass.

Versioned evidence belongs under
`build/release/0.1.0-public-audit.20261104/verification.json` and `qualification/`: commands,
source/runtime fingerprints, current-[layout](../glossary.md#layout) [JUnit](../glossary.md#term-junit) receipts, public output, original recovery
history and the exercised consumer-source archive. The final documentation-only archive refresh
must preserve all frozen non-Markdown source and runtime artifact [hashes](../glossary.md#term-hash); the receipt records
that comparison separately from the earlier execution.

No known blocking source finding remains in the audited [scope](../glossary.md#term-scope). This is bounded local technical
acceptance, not a public licence, trusted publisher, exhaustive numerical proof, portable hardware
[qualification](../glossary.md#term-qualification) or universal performance claim. Existing [Metal](../glossary.md#term-metal) [DTO](../glossary.md#data-transfer-object--dto) Javadoc [parameter](../glossary.md#term-parameter) warnings
remain documentation work; they were not suppressed or mistaken for runtime failures.
The next requested work is the [documentation redesign](documentation-redesign.md), including
the whole-framework explanation, complete package/native atlas and commented runnable examples.
