# Task 0018: Static length-aware [Scan](../../../../glossary.md#general-differentiable-scan) [cold reconstruction](../../../../glossary.md#term-cold-reconstruction)

> **Editorial reading key — not part of the historical record.**
>
> The cold caller supplies a raw graph and descriptor-compatible positional substitutions.
> Compiler validates every declaration and role before rebuilding each selected length-aware Scan
> with fresh local formals, retaining its lengths actual without reading lengths values. A leaf
> means no producing expression, not available storage; the borrowed sidecar owns no payload. The
> homogeneous static intersection below is historical admission, separate from broader source
> typing, later wire versions and numerical train/save/load acceptance.
>
> Current reading: [raw source capture and positional reconstruction](../../../../reference/compiler/cold-source.md#current-cold-source-capture-and-reconstruction); [scoped Scan capture, roles and first-order differentiation](../../../../reference/compiler/verification-scan.md#current-scoped-scan-capture-and-first-order-ad).

## Status

Complete — MAIN accepted implementation, independent code/docs APPROVE (0 P1/P2), public [smoke](../../../../glossary.md#term-smoke-test)
and final [Compiler](../../../../glossary.md#term-compiler)/[Checkpoint](../../../../glossary.md#term-checkpoint)/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) test/[Javadoc](../../../../glossary.md#term-javadoc) [validation](../../../../glossary.md#term-validation) on 2026-11-01.

## Change class

Class C — durable cold source/consumer boundary; clean implementation and independent targeted
code/documentation review required after [admission](../../../../glossary.md#term-admission).

## Goal and [scope](../../../../glossary.md#term-scope)

Rebuild fully static [homogeneous](../../../../glossary.md#term-homogeneous) [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) length-aware general Scans through the existing
[Compiler](../../../../glossary.md#term-compiler) cold boundary and public family constructors, without executing or time-[unrolling](../../../../glossary.md#term-unrolling) them.
[Cold reconstruction](../../../../glossary.md#term-cold-reconstruction) means recovering ordinary [Tensor](../../../../glossary.md#tensor) expressions, not prepared/[runtime](../../../../glossary.md#term-runtime) state.
Select the intersection of the existing body constructor inventory and existing length row rules:
ROW(0), [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) ROW(1), SHARED sequences/invariants and terminal SHARED [carry](../../../../glossary.md#term-carry) updates. ROW denotes
an explicitly identified [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [axis](../../../../glossary.md#term-axis), not an [extent](../../../../glossary.md#term-extent) equal to B; preserve every ordered role.
This is the first [module](../../../../glossary.md#term-module)-owned prerequisite for later [Checkpoint](../../../../glossary.md#term-checkpoint) v5 and the practical [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn) workflow.

## Non-goals

No new production type/package/[API](../../../../glossary.md#term-application-programming-interface-api), executor, [registry](../../../../glossary.md#term-registry), project, [runtime](../../../../glossary.md#term-runtime), orchestration or wire.
No symbolic [cold reconstruction](../../../../glossary.md#term-cold-reconstruction), Model0045 per-value [dtype](../../../../glossary.md#data-type--datatype)/cross-cast [admission](../../../../glossary.md#term-admission), empty [carry](../../../../glossary.md#term-carry),
higher-order, [SIMD](../../../../glossary.md#simd), performance work or [CPU](../../../../glossary.md#term-cpu)-provider-based cold gating. Do not change all-valid
cold breadth, fixed scans, ordinary [unrolled](../../../../glossary.md#backpropagation-through-time-and-recurrent-state) `.rnn`, [GRU](../../../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) or low fixed execution paths.

## Contracts

- [ARCHITECTURE.md](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts):
  `Authority, incorporation, and precedence`, `Core invariants`, `Scope-indexed normative contracts`,
  `Module-ownership routing`, `Dependency rules`, `Explicit non-goals`, `Testing requirements`.
- [compiler-autograd.md](../../../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction)<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd))</span>:
  `Compile lifecycle`, `Bounded general-scan cold reconstruction`; cold inventory is not inference,
  backend capability or wire eligibility. Current length rejection needs explicit bounded cutover.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [backend capability glossary](../../../../glossary.md#backend-capability))</span>
- [recurrent-scan.md](../../../../architecture/contracts/recurrent-scan.md#header-bindings-and-source-admission)<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#general-differentiable-scan))</span>:
  `Header, bindings and source admission`, `Accepted per-value typed length-aware source`,
  `Bounded row-role analysis`, `Runtime preflight and implementation gates`; preserve Model0045's
  independent consumer gates, not its broader source dtype admission.<span data-glossary-navigation="true"> ([dtype glossary](../../../../glossary.md#data-type--datatype); [admission glossary](../../../../glossary.md#term-admission))</span>
- [extensions-training.md](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-5):
  `Generic graph source and values`, `Canonical graph-model version 5` — accepted future codec,
  not implemented wire or an implicit Checkpoint/NN workflow grant.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec); [Checkpoint glossary](../../../../glossary.md#term-checkpoint); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>

## [Ready](../../../../glossary.md#term-ready) blockers and decisions

MAIN selects the static [homogeneous](../../../../glossary.md#term-homogeneous) cold intersection, [identity](../../../../glossary.md#term-identity-meanings) CAST only, complete declaration/
interior checks and substitution-[alias](../../../../glossary.md#aliasing-and-arena) rejection before formal/[producer](../../../../glossary.md#term-producer) IDs. Homogeneity and alias
roles are per [occurrence](../../../../glossary.md#term-occurrence), not global across independent scans. [Inference](../../../../glossary.md#term-inference) success is not cold [admission](../../../../glossary.md#term-admission).
The coordinated root/[Compiler](../../../../glossary.md#term-compiler)/[Scan](../../../../glossary.md#general-differentiable-scan)/extension contracts authorize the inherited [Checkpoint](../../../../glossary.md#term-checkpoint)
copy/[rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction) and [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [graph](../../../../glossary.md#graph)/definition acquisition/explicit cold-forward effects; qualify those cold
consumers without a new flag, provider predicate or executor. V4 encoding remains independently
closed to length mode; v5 and owned numerical training/save/load remain separate successors.
Independent review found two P2 authority gaps; MAIN corrected inherited NN disposition and the
v4/all-valid inventory distinction. Correction-only final APPROVE has zero open findings.
Exact endpoint slicing and local method factoring are implementation details, not new architecture.

## Dependencies and integration

- Depends on: Compiler0012/0013/0015 and Model0038/0044 Complete, reverified from current
  Status/Result by MAIN on 2026-11-01. Compiler0017/Model0045 widening is not required.<span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening))</span>
- Conflicts with: Compiler0017, cold constructor/row inference changes, Checkpoint consumer gates,
  and shared authority/documentation writers; serialized under MAIN.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference))</span>
- Parallel group: None; one isolated executable implementation context after [Ready](../../../../glossary.md#term-ready).
- Common base revision: HEAD `07a281edd3086efde2853211b71b48288db027e1` plus the current
  2613-file main source snapshot; manifests under<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>
  `/var/folders/z2/kdt1_wq1213732bs4wkhfrh40000gq/T/synaptik-production-run.gloed_6d/`.
- Integration order: accepted bounded authority/consumer disposition → Compiler0018 review/gates
  → later Checkpoint v5 → later F32/F64 RNN-length NN workflow; later stages remain coarse.<span data-glossary-navigation="true"> ([F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Integration [validation](../../../../glossary.md#term-validation): MAIN's named cold/[Checkpoint](../../../../glossary.md#term-checkpoint) compatibility checkpoint below, once.
- Shared-document integration owner: MAIN; worker owns Compiler code/Javadoc and named consumer
  regression tests below, not shared authority, planning or explanatory documents.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [Compiler glossary](../../../../glossary.md#term-compiler); [Javadoc glossary](../../../../glossary.md#term-javadoc); [regression glossary](../../../../glossary.md#term-regression); [planning glossary](../../../../glossary.md#term-planning))</span>

MAIN Ready verification: dependencies above are Complete; current source is preserved in the
isolated `compiler0018-worktree` copied from MAIN, with executable [hashes](../../../../glossary.md#term-hash) unchanged from the
initial [manifest](../../../../glossary.md#term-manifest). No competing executable writer is active. MAIN alone integrates shared contracts,
plans and consumer effects; the clean worker must skip build/lint/tests/formatters mid-flight.

## Files and symbols

Existing package `io.github.pho001.synaptik.compiler`; added/moved production types: None.
Under `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/`:
- `SourceGraphReconstruction.java`: [`preflight`](../../../../glossary.md#term-preflight), `validateLeaves`, `reconstructScan`.
- `GraphCompilationPort.java`: `captureSource`, `reconstructSource`, `SourceCapture` [Javadoc](../../../../glossary.md#term-javadoc).
- `GraphCapture.java`: cold-boundary support-status Javadoc only; retain one [capture](../../../../glossary.md#term-capture-meanings).
Read/reuse unchanged `CapturedGraphInference.inferAndValidate`, `ScanBodyInference.infer`/
`validateRows`/`validateResiduals`, and [Model](../../../../glossary.md#term-model) `Scan.apply`/`ScanBody` constructor seams.
Under the matching [Compiler](../../../../glossary.md#term-compiler) test directory, extend `SourceGraphReconstructionTest.java` and
`SourceGraphCaptureTest.java`; reuse length-[inference](../../../../glossary.md#term-inference) fixtures without new production carriers.
Also qualify consumers in [Checkpoint](../../../../glossary.md#term-checkpoint) `GraphCheckpointGeneralScanTest`, `GraphArtifactV4Test`
and [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) `OwnedModelSourceTest`, adding a focused consumer class if needed. Split obsolete blanket
rejection tests; retain independent symbolic/typed and direct v4-encoder negatives. Worker may
update GraphCheckpoint Javadoc, but no Checkpoint/NN/Model executable code, [dependency](../../../../glossary.md#term-dependency) or build
change is in [scope](../../../../glossary.md#term-scope). MAIN owns [compile](../../../../glossary.md#compile)-api.md and [graph](../../../../glossary.md#graph)-checkpoint-api.md integration.

## Acceptance criteria

- Independently bound cold eligibility checks all declarations/values, including unused mixed/low
  interiors hidden behind F32 casts; only one homogeneous F32/F64 body and identity CAST pass.
  Existing all-valid low/integral/BOOL inventory and body-only versus flat separation are unchanged.<span data-glossary-navigation="true"> ([F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [homogeneous glossary](../../../../glossary.md#term-homogeneous); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [identity glossary](../../../../glossary.md#term-identity-meanings); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Complete late/dead/T=0/B=0 preflight rejects malformed headers, phases, deferred obligations,
  feature overflow, roles, demand, residual ordering and exact descriptors before any formal or
  producer ID; no storage read/copy/import, descriptor closure or provider query is used.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [feature glossary](../../../../glossary.md#term-feature); [overflow glossary](../../../../glossary.md#term-overflow); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [descriptors glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer))</span>
- Actuals are exactly [carry](../../../../glossary.md#term-carry)/sequence/[invariant](../../../../glossary.md#term-invariant)/lengths (`c+s+i+1`), formals exactly `c+s+i`.
  Slice invariants at <span data-glossary-navigation="true"> ([Slice glossary](../../../../glossary.md#slice))</span>`c+s+i`, bind the final non-gradient INT64[B] separately, preserve rowCarryCount,
  every optional sequence axis and traversal through the existing length overload. Never read L.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [INT64 glossary](../../../../glossary.md#data-type--datatype); [axis glossary](../../../../glossary.md#term-axis))</span>
- Full descriptor-compatible opaque/repeated substitutions remain supported only for compatible
  tied outer roles. Reject newly aliased ROW/SHARED or differing-row positions before IDs, not
  after constructing a body; same-descriptor replacement equality alone is insufficient.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque))</span>
- Round-trip both dtypes/directions, T=0/nonzero and B=0/1/nonzero, mixed sequence roles and terminal
  accumulators; preserve raw layouts/flags, scalar attributes, aliases and all canonical/latent
  ports. Empty demand and unique first-encounter role outputs remain mandatory.<span data-glossary-navigation="true"> ([Round-trip glossary](../../../../glossary.md#schema-serialization-and-checkpoint); [dtypes glossary](../../../../glossary.md#data-type--datatype); [layouts glossary](../../../../glossary.md#layout); [scalar glossary](../../../../glossary.md#term-scalar); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Fresh distinct unused formals and separate scope tables survive reused templates/arbitrary IDs;
  one body/node/Scan construction per occurrence, O(body) identities independent of T. Re-capture
  matches raw descriptors/header/roles; no bootstrap fixed producer or forced layout copy.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [node glossary](../../../../glossary.md#node); [Scan glossary](../../../../glossary.md#general-differentiable-scan); [occurrence glossary](../../../../glossary.md#term-occurrence); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- Symbolic cold, newly typed length sources and old-wire length encoding still reject directly;
  original/rebuilt admitted RNN algebra retains ordinary first-order AD, not a higher-order grant.<span data-glossary-navigation="true"> ([RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [AD glossary](../../../../glossary.md#autograd))</span>
- Checkpoint copy/rebind and inherited NN graph/definition/explicit cold-forward paths admit
  the selected static occurrences without an Engine. Preserve private named state/aliases and
  raw bits; external lengths prototypes are unread and storage-free. Named lengths state follows
  ordinary Checkpoint copying rules, not a blanket prohibition on reading named control bytes.<span data-glossary-navigation="true"> ([Checkpoint glossary](../../../../glossary.md#term-checkpoint); [rebind glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [graph glossary](../../../../glossary.md#graph); [Engine glossary](../../../../glossary.md#term-engine); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [prototypes glossary](../../../../glossary.md#term-prototype))</span>
- Checkpoint rebind includes replacement-to-retained-state alias conflicts; valid repeated/
  opaque substitutions still pass. Checkpoint save independently rejects length mode before
  missing state storage or target-file effects, even after cold admission opens.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [admission glossary](../../../../glossary.md#term-admission))</span>

## [Validation](../../../../glossary.md#term-validation)

No build/lint/tests/formatters mid-flight. After executable sources and independent review
stabilize, MAIN runs the focused [Compiler](../../../../glossary.md#term-compiler) classes plus [Checkpoint](../../../../glossary.md#term-checkpoint) GeneralScan/V4/[Codec](../../../../glossary.md#term-codec) and
the new [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) cold-consumer tests, then the affected Compiler/Checkpoint/NN test/[Javadoc](../../../../glossary.md#term-javadoc) gate once.
MAIN's existing throwaway `LengthColdSmoke.java` records eight failing-before public cold cases
([F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding), both directions, T=0/3); [run](../../../../glossary.md#term-run) it without `before` after integration and compare all roles.
The later NN consumer [smoke](../../../../glossary.md#term-smoke-test) proves numerical train/save/load separately. No worker root suite,
native rebuild or timing campaign; MAIN owns the final production integration checkpoint.

## Documentation and review impact

Update cold-boundary [Javadoc](../../../../glossary.md#term-javadoc)/[API](../../../../glossary.md#term-application-programming-interface-api) explanation after approved cutover; independent Class C review
checks inventory, failure ordering, exact [descriptors](../../../../glossary.md#term-descriptor) and consumer effects. No new glossary term.

## Result

Status: Complete

- Implemented static homogeneous F32/F64 length cold admission and exact existing-overload
  reconstruction, with per-occurrence substitution-role checks before IDs. Reused existing
  raw inference, row/residual validation and family constructors; no new type/API/dependency.<span data-glossary-navigation="true"> ([homogeneous glossary](../../../../glossary.md#term-homogeneous); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [admission glossary](../../../../glossary.md#term-admission); [occurrence glossary](../../../../glossary.md#term-occurrence); [inference glossary](../../../../glossary.md#term-inference); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [validation glossary](../../../../glossary.md#term-validation); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Ten isolated implementation/test/Javadoc files integrated after frozen-baseline hash checks.
  MAIN also updated Model Scan Javadoc, compile/graph-checkpoint API, owning contracts/root,
  explanation and changelog. Shared docs and the original user changes remain MAIN-owned.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [hash glossary](../../../../glossary.md#term-hash); [Model glossary](../../../../glossary.md#term-model); [Scan glossary](../../../../glossary.md#general-differentiable-scan); [compile glossary](../../../../glossary.md#compile); [graph glossary](../../../../glossary.md#graph))</span>
- Independent authority/correction and implementation/docs reviews APPROVE, zero open P1/P2;
  explicit maintainability review confirmed no duplicate math or execution/bypass path.
- Initial focused command FAILED in three new test fixtures (zero-sized control span and
  nonscalar implicit seed). Exact-span/scalar-objective corrections preserved all domains;
  the three-method recovery passed. Original failed evidence remains <span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [scalar glossary](../../../../glossary.md#term-scalar); [objective glossary](../../../../glossary.md#term-objective))</span>`artifact://46`.
- Final command: `./gradlew :modules:compiler:test :modules:compiler:javadoc :extensions:checkpoint:test :extensions:checkpoint:javadoc :extensions:nn:test :extensions:nn:javadoc`
  PASS (`artifact://51`): [Compiler](../../../../glossary.md#term-compiler) 413, [Checkpoint](../../../../glossary.md#term-checkpoint) 68, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 544 tests; zero failures/errors/skips.
- Public failing-before/passing-after cold smoke: eight F32/F64, T=0/3, both-direction cases;
  exact graph/header/role roundtrip after reconstruction. Logs/manifests are in the recorded
  production-run evidence directory. No numerical inference/training completion is inferred.<span data-glossary-navigation="true"> ([smoke glossary](../../../../glossary.md#term-smoke-test); [Logs glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [run glossary](../../../../glossary.md#term-run))</span>
- No unresolved issue within this bounded task. Checkpoint0014 v5 and NN0033 numerical workflow
  are the separately admitted successors; symbolic/wider typed cold remains unsupported.
