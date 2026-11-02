# Task 0018: Static length-aware Scan cold reconstruction

## Status

Complete — MAIN accepted implementation, independent code/docs APPROVE (0 P1/P2), public smoke
and final Compiler/Checkpoint/NN test/Javadoc validation on 2026-11-01.

## Change class

Class C — durable cold source/consumer boundary; clean implementation and independent targeted
code/documentation review required after admission.

## Goal and scope

Rebuild fully static homogeneous FLOAT32/FLOAT64 length-aware general Scans through the existing
Compiler cold boundary and public family constructors, without executing or time-unrolling them.
Cold reconstruction means recovering ordinary Tensor expressions, not prepared/runtime state.
Select the intersection of the existing body constructor inventory and existing length row rules:
ROW(0), rank-two ROW(1), SHARED sequences/invariants and terminal SHARED carry updates. ROW denotes
an explicitly identified batch axis, not an extent equal to B; preserve every ordered role.
This is the first module-owned prerequisite for later Checkpoint v5 and the practical RNN workflow.

## Non-goals

No new production type/package/API, executor, registry, project, runtime, orchestration or wire.
No symbolic cold reconstruction, Model0045 per-value dtype/cross-cast admission, empty carry,
higher-order, SIMD, performance work or CPU-provider-based cold gating. Do not change all-valid
cold breadth, fixed scans, ordinary unrolled `.rnn`, GRU/LSTM or low fixed execution paths.

## Contracts

- [ARCHITECTURE.md](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts):
  `Authority, incorporation, and precedence`, `Core invariants`, `Scope-indexed normative contracts`,
  `Module-ownership routing`, `Dependency rules`, `Explicit non-goals`, `Testing requirements`.
- [compiler-autograd.md](../../../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction):
  `Compile lifecycle`, `Bounded general-scan cold reconstruction`; cold inventory is not inference,
  backend capability or wire eligibility. Current length rejection needs explicit bounded cutover.
- [recurrent-scan.md](../../../../architecture/contracts/recurrent-scan.md#header-bindings-and-source-admission):
  `Header, bindings and source admission`, `Accepted per-value typed length-aware source`,
  `Bounded row-role analysis`, `Runtime preflight and implementation gates`; preserve Model0045's
  independent consumer gates, not its broader source dtype admission.
- [extensions-training.md](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-5):
  `Generic graph source and values`, `Canonical graph-model version 5` — accepted future codec,
  not implemented wire or an implicit Checkpoint/NN workflow grant.

## Ready blockers and decisions

MAIN selects the static homogeneous cold intersection, identity CAST only, complete declaration/
interior checks and substitution-alias rejection before formal/producer IDs. Homogeneity and alias
roles are per occurrence, not global across independent scans. Inference success is not cold admission.
The coordinated root/Compiler/Scan/extension contracts authorize the inherited Checkpoint
copy/rebind and NN graph/definition acquisition/explicit cold-forward effects; qualify those cold
consumers without a new flag, provider predicate or executor. V4 encoding remains independently
closed to length mode; v5 and owned numerical training/save/load remain separate successors.
Independent review found two P2 authority gaps; MAIN corrected inherited NN disposition and the
v4/all-valid inventory distinction. Correction-only final APPROVE has zero open findings.
Exact endpoint slicing and local method factoring are implementation details, not new architecture.

## Dependencies and integration

- Depends on: Compiler0012/0013/0015 and Model0038/0044 Complete, reverified from current
  Status/Result by MAIN on 2026-11-01. Compiler0017/Model0045 widening is not required.
- Conflicts with: Compiler0017, cold constructor/row inference changes, Checkpoint consumer gates,
  and shared authority/documentation writers; serialized under MAIN.
- Parallel group: None; one isolated executable implementation context after Ready.
- Common base revision: HEAD `07a281edd3086efde2853211b71b48288db027e1` plus the current
  2613-file main source snapshot; manifests under
  `/var/folders/z2/kdt1_wq1213732bs4wkhfrh40000gq/T/synaptik-production-run.gloed_6d/`.
- Integration order: accepted bounded authority/consumer disposition → Compiler0018 review/gates
  → later Checkpoint v5 → later F32/F64 RNN-length NN workflow; later stages remain coarse.
- Integration validation: MAIN's named cold/Checkpoint compatibility checkpoint below, once.
- Shared-document integration owner: MAIN; worker owns Compiler code/Javadoc and named consumer
  regression tests below, not shared authority, planning or explanatory documents.

MAIN Ready verification: dependencies above are Complete; current source is preserved in the
isolated `compiler0018-worktree` copied from MAIN, with executable hashes unchanged from the
initial manifest. No competing executable writer is active. MAIN alone integrates shared contracts,
plans and consumer effects; the clean worker must skip build/lint/tests/formatters mid-flight.

## Files and symbols

Existing package `io.github.pho001.synaptik.compiler`; added/moved production types: None.
Under `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/`:
- `SourceGraphReconstruction.java`: `preflight`, `validateLeaves`, `reconstructScan`.
- `GraphCompilationPort.java`: `captureSource`, `reconstructSource`, `SourceCapture` Javadoc.
- `GraphCapture.java`: cold-boundary support-status Javadoc only; retain one capture.
Read/reuse unchanged `CapturedGraphInference.inferAndValidate`, `ScanBodyInference.infer`/
`validateRows`/`validateResiduals`, and Model `Scan.apply`/`ScanBody` constructor seams.
Under the matching Compiler test directory, extend `SourceGraphReconstructionTest.java` and
`SourceGraphCaptureTest.java`; reuse length-inference fixtures without new production carriers.
Also qualify consumers in Checkpoint `GraphCheckpointGeneralScanTest`, `GraphArtifactV4Test`
and NN `OwnedModelSourceTest`, adding a focused consumer class if needed. Split obsolete blanket
rejection tests; retain independent symbolic/typed and direct v4-encoder negatives. Worker may
update GraphCheckpoint Javadoc, but no Checkpoint/NN/Model executable code, dependency or build
change is in scope. MAIN owns compile-api.md and graph-checkpoint-api.md integration.

## Acceptance criteria

- Independently bound cold eligibility checks all declarations/values, including unused mixed/low
  interiors hidden behind F32 casts; only one homogeneous F32/F64 body and identity CAST pass.
  Existing all-valid low/integral/BOOL inventory and body-only versus flat separation are unchanged.
- Complete late/dead/T=0/B=0 preflight rejects malformed headers, phases, deferred obligations,
  feature overflow, roles, demand, residual ordering and exact descriptors before any formal or
  producer ID; no storage read/copy/import, descriptor closure or provider query is used.
- Actuals are exactly carry/sequence/invariant/lengths (`c+s+i+1`), formals exactly `c+s+i`.
  Slice invariants at `c+s+i`, bind the final non-gradient INT64[B] separately, preserve rowCarryCount,
  every optional sequence axis and traversal through the existing length overload. Never read L.
- Full descriptor-compatible opaque/repeated substitutions remain supported only for compatible
  tied outer roles. Reject newly aliased ROW/SHARED or differing-row positions before IDs, not
  after constructing a body; same-descriptor replacement equality alone is insufficient.
- Round-trip both dtypes/directions, T=0/nonzero and B=0/1/nonzero, mixed sequence roles and terminal
  accumulators; preserve raw layouts/flags, scalar attributes, aliases and all canonical/latent
  ports. Empty demand and unique first-encounter role outputs remain mandatory.
- Fresh distinct unused formals and separate scope tables survive reused templates/arbitrary IDs;
  one body/node/Scan construction per occurrence, O(body) identities independent of T. Re-capture
  matches raw descriptors/header/roles; no bootstrap fixed producer or forced layout copy.
- Symbolic cold, newly typed length sources and old-wire length encoding still reject directly;
  original/rebuilt admitted RNN algebra retains ordinary first-order AD, not a higher-order grant.
- Checkpoint copy/rebind and inherited NN graph/definition/explicit cold-forward paths admit
  the selected static occurrences without an Engine. Preserve private named state/aliases and
  raw bits; external lengths prototypes are unread and storage-free. Named lengths state follows
  ordinary Checkpoint copying rules, not a blanket prohibition on reading named control bytes.
- Checkpoint rebind includes replacement-to-retained-state alias conflicts; valid repeated/
  opaque substitutions still pass. Checkpoint save independently rejects length mode before
  missing state storage or target-file effects, even after cold admission opens.

## Validation

No build/lint/tests/formatters mid-flight. After executable sources and independent review
stabilize, MAIN runs the focused Compiler classes plus Checkpoint GeneralScan/V4/Codec and
the new NN cold-consumer tests, then the affected Compiler/Checkpoint/NN test/Javadoc gate once.
MAIN's existing throwaway `LengthColdSmoke.java` records eight failing-before public cold cases
(F32/F64, both directions, T=0/3); run it without `before` after integration and compare all roles.
The later NN consumer smoke proves numerical train/save/load separately. No worker root suite,
native rebuild or timing campaign; MAIN owns the final production integration checkpoint.

## Documentation and review impact

Update cold-boundary Javadoc/API explanation after approved cutover; independent Class C review
checks inventory, failure ordering, exact descriptors and consumer effects. No new glossary term.

## Result

Status: Complete

- Implemented static homogeneous F32/F64 length cold admission and exact existing-overload
  reconstruction, with per-occurrence substitution-role checks before IDs. Reused existing
  raw inference, row/residual validation and family constructors; no new type/API/dependency.
- Ten isolated implementation/test/Javadoc files integrated after frozen-baseline hash checks.
  MAIN also updated Model Scan Javadoc, compile/graph-checkpoint API, owning contracts/root,
  explanation and changelog. Shared docs and the original user changes remain MAIN-owned.
- Independent authority/correction and implementation/docs reviews APPROVE, zero open P1/P2;
  explicit maintainability review confirmed no duplicate math or execution/bypass path.
- Initial focused command FAILED in three new test fixtures (zero-sized control span and
  nonscalar implicit seed). Exact-span/scalar-objective corrections preserved all domains;
  the three-method recovery passed. Original failed evidence remains `artifact://46`.
- Final command: `./gradlew :modules:compiler:test :modules:compiler:javadoc :extensions:checkpoint:test :extensions:checkpoint:javadoc :extensions:nn:test :extensions:nn:javadoc`
  PASS (`artifact://51`): Compiler 413, Checkpoint 68, NN 544 tests; zero failures/errors/skips.
- Public failing-before/passing-after cold smoke: eight F32/F64, T=0/3, both-direction cases;
  exact graph/header/role roundtrip after reconstruction. Logs/manifests are in the recorded
  production-run evidence directory. No numerical inference/training completion is inferred.
- No unresolved issue within this bounded task. Checkpoint0014 v5 and NN0033 numerical workflow
  are the separately admitted successors; symbolic/wider typed cold remains unsupported.
