# Public-release readiness audit

## Purpose and status

Status: Implementation and source review complete; joined runtime/artifact qualification in progress.
The user requested framework-wide verification, release-critical stabilization and actionable diagnostics.
This audit supersedes private-distribution assumptions for readiness assessment, but does not
select a licence, authorize public uploads or change numerical/ownership contracts.

Common base revision: `564a9988349c259f6082b6905d98de24f2e216ff`.
The working repository is `/Users/phujka/IdeaProjects/Synaptik`; the old `/tmp` snapshot is not
a Git checkout. Findings refer to current working bytes, not earlier sealed release artifacts.

## Contracts and admission

Read `AGENTS.md`, the root authority index, Core invariants and the scoped headings for the
assigned workstream below. The six incorporated scoped contracts remain the only detailed
authority. A contradiction requiring architecture change is an explicit decision, not permission
to widen capability, invent another execution path or weaken validation.

The coordinator verified the explicit user authorization, stable baseline, independent read
scopes and satisfied source/contract prerequisites. Each investigation is an authorized frontier.
Depends on: None. Conflicts with: None for read-only investigation. Parallel group:
`PublicReleaseAuditReadOnly`. Integration order: Any. Shared-document integration owner:
coordinator. Classification: Class A investigation; any resulting behavioral repair is classified
separately, with isolated worktree implementation and independent review for Classes B/C.

## Workstreams and scope

| ID | Source scope | Exact scoped-contract headings | Audit focus |
|---|---|---|---|
| A1 | Model and Compiler | foundational-modules: modules/model, Profile-free numerical semantics; compiler-autograd: modules/compiler, Training graph model, Compile lifecycle; recurrent-scan: General differentiable scan | Public expressions, shape/dtype/layout inference, constants, gradients, higher-order limits, Scan activity/binding and numerical boundaries. |
| A2 | CPU and OpenBLAS | backend-execution: CPU backend routes, Qualified oneDNN CPU realization, General Scan ordinary-unit realization, OpenBLAS provider | Generated kernels, preparation selection, low precision, native safety, stale identities, ordinary/Scan reuse and failure diagnostics. |
| A3 | Metal Java/native/build | backend-execution: Metal backend, Concrete backend modules | Supported route matrix, resource/ABI/code-image validation, numerical boundaries, real-device/native package prerequisites. |
| A4 | Engine, Prepare, Runtime | runtime-prepare-engine: modules/runtime, modules/prepare, modules/engine, Prepare lifecycle, Runtime lifecycle | Admission, close/leases, rollback, concurrent runs, storage borrowing, mixed-owner transfer, publication and exception ordering. |
| A5 | NN and Training | extensions-training: extensions/nn, Owned tensor Models, modules/training, Optimizer/training lifecycle | Compute/repeated train/save/load integration, state ownership, transactional update/restore, numerical finiteness and optimizer invariants. |
| A6 | Checkpoint and NN codecs | extensions-training: extensions/checkpoint, Retained NN checkpoint APIs, Canonical graph-model versions | Untrusted artifact bounds, corruption/truncation, alias/state integrity, graph reconstruction, atomic publication and compatibility. Read-only overlap with A5 is intentional. |
| A7 | Data, Text, Config, Planning, Tuning, other tools and architecture boundaries | extensions-training: extensions/data, extensions/text, extensions/onnx; foundational-modules: modules/config, modules/planning; backend-execution: Performance evidence and optimization tooling, Other tool scopes | Batching/tokenizer boundaries, cache/session identity, capability truthfulness, dependencies and marker-only exclusions. |
| A8 | Trace and distribution (coordinator) | foundational-modules: modules/trace; root: Testing requirements | Caller-observable diagnostics, disabled-observer behavior, exception/route correlation, legal/publication prerequisites, artifact content and clean consumer install. |

Investigators inspect actual production sources and focused tests, classify concrete consumer-visible
findings by severity and confidence, and distinguish intentional exclusions from defects.
Report exact path/symbol/scenario, evidence, smallest preserving repair, affected contracts/docs,
and existing public smoke commands. Do not run builds/tests/linters/formatters during the
parallel investigation. Do not infer numerical correctness from catalog execution counts.

## Dependency flow and repair rules

`A1..A8 -> confirmed findings -> isolated preserving repairs -> independent review -> joined verification`.
The coordinator retains integration ownership and performs actual reproduction/smoke evidence.
Parallel repairs need disjoint semantic/write ownership and their own Ready entries here before
launch. Existing wording/source-text/opcode/default/count/wiring tests are removed rather than
re-pinned if incidental; preserve genuine numerical, transition, boundary and ownership coverage.
No runtime fallback, value-based route selection, hidden precision conversion or new interpreter.
No unrelated features, new backend or unfinished marker project is promoted into this release.

## Acceptance and verification

- Account for every implemented product and unsupported marker project, supported public workflow,
  generated/native boundary and legal/distribution gate.
- Reproduce confirmed release-critical failures through actual execution where possible; repair
  causes, retain meaningful regression coverage and observe the changed public path afterward.
- Add typed producer-owned trace only for concrete diagnostic gaps; keep Trace passive/JDK-only,
  no graph/storage leaks, numerical selector, required observer or disabled-path allocation.
- Run a final joined repository/native/Javadoc gate after ordered integration; record commands,
  skips and limits without relabeling historical failed commands.
- Build matched versioned artifacts, inspect checksums/provenance/dependency closure, and execute
  fresh public CPU and real-device Metal consumers outside the checkout.
- Publish an evidence-backed finding/disposition record, precise supported scope and remaining
  owner-controlled public-release gates. No public licence or trusted publisher identity is
  invented by the audit.

## Results

Investigation and execution receipts are recorded here as work completes. Historical source and
artifact receipts in the production support matrix remain historical and are not overwritten.

## Confirmed-finding repair frontiers

Owner decision: retain no public redistribution grant for now. Public publication remains
Blocked; technical audit, stabilization and local candidate qualification continue.

The coordinator source-confirmed the following findings. Status: Ready for each listed repair;
baseline remains `564a9988349c259f6082b6905d98de24f2e216ff`. Dependencies: completed source
investigation and stable existing contracts; parent records failing-before runtime receipts.
Parallel group: `PublicReleaseRepairs`. Conflicts: only the shared generated-identity seam
assigned to R5 below; other write and semantic scopes are disjoint. Integration order: Any,
with R6 before R5 so the combined numerical identity join follows both implementations.
Shared-document integration owner: coordinator. Workers do not run builds/tests/linters/formatters;
the coordinator exercises repros, integrates commits and performs focused/final verification once.

| ID | Class | Exclusive write scope and acceptance |
|---|---|---|
| R1 | B | TrainingParameter restore and Training tests/Javadoc: reject inconsistent full-precision logical/master bytes transactionally; retain low masters. |
| R2 | C | GraphSource/GraphArtifactCodec name validation and NN OwnedModelSource/ModelArtifactCodec: eliminate quadratic namespace amplification without a new name/depth restriction; preserve wire/order/aliases; contextual invalidation and malformed-state errors. Own necessary Module child-validation optimization only if required by the same import path. |
| R3 | B | tools/tuning WorkloadTuning and tests/Javadoc: session-only tuning performs no persistent-cache access; compatible configurations are authenticated against every exact batch before result/cache publication. |
| R4 | C | Engine/Prepare/Runtime lifecycle files and tests: slice sufficient-capacity caller storage to its established referenced prefix; close every acquired owner on bookkeeping failure; avoid allocating terminal shutdown snapshots; preserve reentrant/concurrent failure replay. |
| R5 | C | CPU statistical/normalization working sites and references/tests: F32 uses F32 sites; retain overflowing square infinity for F32/F64/lows. Own selective generated compatibility updates for R5 and R6, including ordinary helpers in Scan; do not bump unchanged raw families. |
| R6 | C | CPU Pool2d/Pool3d/Attention exact integer divisors and references/tests: reuse existing exact-count divide emitter; preserve final low narrowing, guards and saved/publication boundaries. Send changed identity-family facts to R5; do not edit its identity files. |
| R7 | C | Compiler LossGradientRules and focused tests/integration test: use categorical-loss shifted probability/log-probability and exact target/ignore guards, not ordinary softmax's distinct infinity formula. No backend backward path or changed Model mathematics. |
| R8 | C | Model ScanBody plus Compiler ScanBodyInference/ScanGradientRules/SourceGraphReconstruction/BatchBindingPlan and focused tests: iterative deep-body traversal with unchanged encounter order; restore explicitly retained cold header exclusion only, never ordinary execution exclusion; preserve scoped binding diagnostic subject and actual B. |
| R9 | C | CPU integration/composition/native selector and native execute/wait paths, backend-owned typed diagnostic DTO/observer and tests: optional cold route/rejection/finalization diagnostics using existing Trace envelope; preserve no-observer path, numerical decisions and primary native execute failure with wait suppressed. No Engine/Runtime/Trace-module edits. |

R5 and R6 share no implementation edits: R5 is the sole generated-identity integration owner.
R2 owns NN source invalidation diagnostics and corrupt-values context; R1 changes only Training.
R4 owns Engine/Prepare/Runtime, while R9 owns only CPU. No worker edits shared architecture,
changelog, README, release inventory, planning or user/API guides. Each repair updates meaningful
Javadoc and deterministic behavioral regression coverage in its owned source/test scope.

### Metal repair join

R10 and R11 are additionally Ready at the same baseline, group and validation checkpoint.
Their contracts and write scopes are disjoint; integration order is R10 then R11. Coordinator
exclusively owns combined shader authentication metadata and coordinated compatibility identities.

- **R10, Class C:** correct Conv2d/Conv3d contraction-then-bias in the shared ordinary/Scan/fused
  convolution shader and behavioral cancellation tests. No other shader or identity edits.
- **R11, Class C:** source-only active BOOL Scan validation, dense typed-control provider parity,
  FP8 rejection at storage/source/publication boundaries and empty-context lifetime checks.
  Preserve public indexing exception classes and native cause with a bounded invocation-owned
  Scan failure record (category, time, unit, operand/ordinal); no body replay/interpreter.
  Use existing run-private workspace transport, no new native export, global mutable error,
  payload capture or per-element diagnostic allocation. Existing ABI export signatures and
  Model numerical semantics are unchanged. Any extra workspace/layout meaning receives the
  coordinated append-only compatibility change owned by the coordinator.

Mixed-low WHERE/comparison source-versus-authority discrepancies remain a separate explicit
coordinator disposition; these repairs do not broaden the declared Metal numerical domain.

### Remaining diagnostic and documentation joins

R12 is Ready after R4 integration (`769f19bdd`), using the current main integration base.
Class B/C: add an explicit opt-in observer overload to ordinary `Engine.prepareTuned`, reporting
typed PREPARE fallback/rejection diagnostics (phase and closed reason category) at the existing
producer. Preserve evidence-empty fallback, original failures and cleanup, no cache/payload leaks,
no string-based policy decisions and no no-observer diagnostic allocation. Engine-local passive
DTOs reuse the existing Trace envelope; no Trace/Config/Runtime change or generic logger.

Documentation frontiers D1/D2 are Ready at the original baseline and may reconcile current
static source claims while code repairs proceed; they must not claim unrun repair acceptance.
D1 exclusively owns README, getting-started, public/training API guidance and stale fixed-RNN
NN public Javadocs. D2 exclusively owns production-release-goal, roadmap and release-process.
Coordinator owns backend contracts/guides, debugging-trace, final audit findings/receipts,
changelog, source inventory and actual artifact qualification. No public licence/upload is granted.

### Targeted boundary completion

R13/R14 are Ready at the current integrated repair base (`3c1b72792`), with source and contract
dependencies satisfied and independent write/semantic scopes. Parallel group:
`PublicReleaseBoundaryCompletion`. Integration order: R11 before R13; R14 independent.
Coordinator owns shared docs, compatibility authentication, verification and integration.

- **R13, Class C:** enforce the already-declared Metal BFLOAT16/FLOAT16 mixed-low exclusion
  for direct WHERE/comparison occurrences at provider and independent encoded/native admission.
  Preserve the fourteen qualified floating WHERE pairs, explicit casts and represented-value
  semantics. This corrects accidental over-admission; it grants no new numerical domain.
  R11 owns Scan/FP8 boundaries; do not change those semantics or shared identity constants.
- **R14, Class B:** bound and sanitize the displayed owned-source invalidation path without
  changing canonical names, legal depth, persisted bytes or source validation semantics.
  Preserve the captured failure category and actionable path prefix; add a behavioral invalidation
  regression for control characters and a very deep valid namespace.

Workers use isolated worktrees and run no checks; the coordinator reproduces and verifies the
joined public paths and requests independent targeted source review.

### Joined-checkpoint reference migrations

R15/R16 are Ready after the first joined gate, based at `0e9b12a39`; dependencies are the
integrated statistical working-site correction and observed reference mismatches. Group:
`PublicAuditReferenceMigration`. Disjoint test-only owners; integration order Any; coordinator
owns source/contracts and the single affected/final rerun. Workers run no checks.

- R15 owns trailing-normalization numerical evidence and the prepared RMS overflow regression.
  Preserve Model's binary32 named sites, exact point identities and mandatory square overflow;
  delete obsolete implementation/workspace assertions rather than force old stable/hypot behavior.
- R16 owns the BatchNorm training numerical/performance reference. Reconcile independent binary32
  named rounding and centered-square/count equations for all published statistics. Never change
  production numerics or relax tolerances to satisfy a checksum.

The first gate also exposed a stale native Scan-v1 coarse guard and an attention mixed-input
F64 specialization guard; the coordinator corrects those small integration joins. The attempted
R8 cold empty-header restriction is withdrawn: implemented graph-model v4/v5 explicitly admit
carry-free/zero-ROW sources in `extensions-training`, superseding the historical positive-ROW
baseline. Existing consumer round-trip tests remain; no constructor domain is narrowed.
