# Task CPU0030: [FLOAT32](../../../../glossary.md#term-float32) [pointwise](../../../../glossary.md#term-pointwise) general-[scan](../../../../glossary.md#term-cumulative-scan) forward and automatic reverse execution

## Status

Complete — bounded [FLOAT32](../../../../glossary.md#term-float32) [pointwise](../../../../glossary.md#term-pointwise) forward/actual automatic-reverse milestone accepted by main
2026-10-22 after final independent review and the source-matched focused [checkpoint](../../../../glossary.md#term-checkpoint) PASS.
Launch verified [Ready](../../../../glossary.md#term-ready)/sole frontier, contracts, seven fields, Complete predecessors and inactive conflicts.
Authorization covers CPU0030 only; no successor/[commit](../../../../glossary.md#term-commit)/push.

## Change class

Class C — generated hot loop, exact saved-state boundaries, resource declaration and [run](../../../../glossary.md#term-run) isolation.
Clean implementation and independent targeted code/documentation review completed.

## Goal and [scope](../../../../glossary.md#term-scope)

Deliver [canonical](../../../../glossary.md#term-canonical-form)/static/all-valid [FLOAT32](../../../../glossary.md#term-float32) [scan](../../../../glossary.md#term-cumulative-scan) and its [Compiler](../../../../glossary.md#term-compiler)-generated first-order reverse
through the real provider and public [Engine](../../../../glossary.md#term-engine), not a [kernel](../../../../glossary.md#kernel) [prototype](../../../../glossary.md#term-prototype).
Example: `next=emit=tanh(h+xStep)`; also same-[shape](../../../../glossary.md#shape) shared [parameters](../../../../glossary.md#term-parameter) and two interacting carries.
[Compile](../../../../glossary.md#compile) constructs the reverse; [CPU](../../../../glossary.md#term-cpu) prepares each body once and emits one outer time loop.

- Candidate source domain: common fully static step Shape S (rank 0..16, including zero extents),
  T >= 0, at least one carry/sequence; every carry, step, invariant and emission has S and FLOAT32.
  Sequence/history/emission boundaries have <span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [carry glossary](../../../../glossary.md#term-carry); [invariant glossary](../../../../glossary.md#term-invariant))</span>`[T]+S`; source inputs/produced boundaries are canonical
  zero-offset, with the necessary generated-read execution closure explicitly qualified below.
  T, N=elements(S), spans/products use checked long arithmetic. Initial limits: <=128 body nodes,
  <=256 local values/outer ports each, <=64 formals, input positions + demanded ports <=248 (safe
  even without tied-read deduplication). Reject excess before resource allocation/class generation.
  Automatic-reverse promise is the ACTUAL joint pair: BOTH forward and Compiler-generated reverse
  bodies must satisfy these independent budgets and complete binder/closure predicates. A true forward
  query alone grants no arbitrary backward guarantee. Qualify existing Compiler-emitted queries before
  physical resources/class generation; do not add CPU AD or change Compiler to estimate future closure.
  Rank-16 S yields stacked rank 17: qualify CPU-only forward/reverse explicitly, not Metal's transfer
  rank limit. All rank/shape predicates still apply to the actual pair.<span data-glossary-navigation="true"> ([elements glossary](../../../../glossary.md#term-element); [nodes glossary](../../../../glossary.md#node); [AD glossary](../../../../glossary.md#autograd); [Metal glossary](../../../../glossary.md#term-metal))</span>
- Source algebra: tensor ADD/SUB/MUL/DIV with equal S operands; existing typed scalar ADD/SUB/MUL/DIV;
  NEG/TANH/SIGMOID. No general broadcasting, reduction, MATMUL, transpose or cross-type operation.
  Identity bodies, unused formals, repeated actual roles and repeated result bindings remain valid.<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor); [scalar glossary](../../../../glossary.md#term-scalar); [TANH glossary](../../../../glossary.md#term-tanh); [SIGMOID glossary](../../../../glossary.md#term-sigmoid); [broadcasting glossary](../../../../glossary.md#broadcasting); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [transpose glossary](../../../../glossary.md#term-transpose); [operation glossary](../../../../glossary.md#operation); [Identity glossary](../../../../glossary.md#term-identity-meanings); [bindings glossary](../../../../glossary.md#term-binding))</span>
- Execution closure additionally admits scalar FLOAT32 non-gradient invariant formals and EXPAND
  from scalar to S (or identity S to S), with target attrs/derived zero-stride local layout validated;
  CONTIGUOUS only for those expansions or identity canonical S. No hidden constants or free leaves.
  Outer read closure also admits resolved zero-offset SCALAR_ALL_ZERO layouts of S or <span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [target glossary](../../../../glossary.md#term-target); [stride glossary](../../../../glossary.md#term-stride); [layout glossary](../../../../glossary.md#layout); [CONTIGUOUS glossary](../../../../glossary.md#term-contiguous); [constants glossary](../../../../glossary.md#term-constant))</span>`[T]+S`:
  Compiler zeroLike initializers/missing seeds are scalar EXPAND views, NOT dense-canonical actuals.
  Bind their exact scalar span without materializing N/T*N copies. Public promised inputs and all
  output/formal descriptors stay canonical; arbitrary strided/offset/broadcast reads still reject.
  These are structural query predicates, NOT a trusted “generated” tag or provenance-based privilege.
  Source AD promise above excludes scalar-broadcast parameters: such a forward execution can qualify,
  but a resulting SUM/sumToShape reverse body must reject until the later broadcast-closure slice.<span data-glossary-navigation="true"> ([seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [views glossary](../../../../glossary.md#term-view-meanings); [descriptors glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [slice glossary](../../../../glossary.md#slice))</span>
- Whole generated closure within the promised joint-pair budgets: seed/fan-out/tied-root ADDs, NEG, MUL/DIV/SUB,
  saved activation outputs, scalar-immediate arithmetic, scalar zero/one invariant EXPAND, disconnected
  zero EXPANDs, and per-invariant same-S carry accumulators. Equal-S normalization generates no
  sumToShape or cast. Accept identity CONTIGUOUS inserted by ordinary layout closure; neither rewrite
  Compiler formulas/flags nor assume optimizations erase EXPAND. Validate optimized AND disabled paths.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [fan-out glossary](../../../../glossary.md#term-fan-out); [activation glossary](../../../../glossary.md#term-activation); [normalization glossary](../../../../glossary.md#term-normalization))</span>

## Non-goals

No [CPU](../../../../glossary.md#term-cpu) [AD](../../../../glossary.md#autograd), second semantic body/[graph](../../../../glossary.md#graph), generic control-flow framework, executor/session/tape, T-fold
graphs/code, [FP16](../../../../glossary.md#term-fp16)/[BF16](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding), [vector](../../../../glossary.md#term-vector)/parallel/time-parallel, [padding](../../../../glossary.md#padding)/lengths, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), [persistence](../../../../glossary.md#term-persistence) [feature](../../../../glossary.md#term-feature),
native route, tuning/default change or broad performance campaign. Preserve fixed/static [scan](../../../../glossary.md#term-cumulative-scan) routes.
Remaining declared algebra, true broadcast/reduction/[view](../../../../glossary.md#term-view-meanings) closure and F64 are a later mandatory gate,
not abandoned or claimed complete by this milestone; low types retain separate [qualification](../../../../glossary.md#term-qualification).

## Contracts

- [Root](../../../../../ARCHITECTURE.md): `Core lifecycle`, `Core invariants`,
  `Scope-indexed normative contracts`, `Dependency rules`, `Testing requirements`.
- [Backend execution](../../../../architecture/contracts/backend-execution.md)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend))</span>:
  `Concrete backend modules`, `CPU backend routes`, `Performance evidence and optimization tooling`.
- [Recurrent scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan):
  [`General differentiable scan`](../../../../glossary.md#general-differentiable-scan) and ALL its subheadings — sole complete [scan](../../../../glossary.md#term-cumulative-scan) semantics.
- [Runtime/Prepare](../../../../architecture/contracts/runtime-prepare-engine.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare))</span>: `modules/runtime`,
  `modules/prepare`, `Prepare lifecycle`, `Run lifecycle`, `Public reusable prepared execution`.
- [Foundational](../../../../architecture/contracts/foundational-modules.md#profile-free-capability-queries):
  `Profile-free capability queries` — whole body AND binder, not child support alone.

No new authority decision is proposed. Stop on contract ambiguity, wider geometry or missing closure;
do not invent architecture or silently turn the first [slice](../../../../glossary.md#slice) into whole-domain completion.

## Dependencies and integration

- Depends on: Model0034/0035, Planning0008, Compiler0011, Prepare0010; CPU0022/0027/0028/0029;
  existing Engine0021/Runtime run ownership and CPU generated-route baseline, all accepted Complete.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [run glossary](../../../../glossary.md#term-run); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Conflicts with: CPU admission/lowering/IR/ABI/cache/schema/memory, scan semantic/AD/query/Prepare
  contracts, v2 ledger, shared scan/CPU documentation and planning writers; serialize all these scopes.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [lowering glossary](../../../../glossary.md#lowering); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [cache glossary](../../../../glossary.md#term-cache); [schema glossary](../../../../glossary.md#term-schema); [scan glossary](../../../../glossary.md#term-cumulative-scan); [AD glossary](../../../../glossary.md#autograd); [Prepare glossary](../../../../glossary.md#prepare); [planning glossary](../../../../glossary.md#term-planning))</span>
- Parallel group: None; no concurrent write task or successor launch.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus accepted
  uncommitted authority/Model0034/0035, Planning0008, Compiler0011, Prepare0010. Launch verified empty
  index/184 protected dirty paths; completion retains these plus CPU0030. Nine-path metadata closure only.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- Integration order: accepted planning/launch admission → clean CPU implementation → independent
  Class C code/docs review → main focused checkpoint/acceptance, all complete. No successor launch.
- Integration [validation](../../../../glossary.md#term-validation): named CPU0030 [checkpoint](../../../../glossary.md#term-checkpoint) below, once after ordered integration; root/native
  `test javadoc` deferred to the accepted low-level program join, not this worker/checkpoint.
- Shared-document integration owner: main; worker Javadoc/capability drafts, independent final docs,
  main shared identity/ledger/planning integration completed.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [Javadoc glossary](../../../../glossary.md#term-javadoc); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

## Current source seams and package map

Paths are under `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`;
read focused tests beside these classes.

- `CpuCapabilityProvider.supports`: add ScanKind dispatch BEFORE ordinary single-output guards.
  Share whole-body admission with lowering, with NO physical allocation/class generation (cold Java
  metadata allocation is allowed); preserve attachment, descriptors, flags,
  roles, canonical residual inventory and demanded indices; validate unsupported/dead interiors at T=0 too.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [admission glossary](../../../../glossary.md#term-admission); [lowering glossary](../../../../glossary.md#lowering); [metadata glossary](../../../../glossary.md#term-metadata); [descriptors glossary](../../../../glossary.md#term-descriptor); [canonical glossary](../../../../glossary.md#term-canonical-form); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand))</span>
- `internal.prepare.CpuPartitionPreparer`/`CpuPartitionPreparationPlan`: `directPartition`, `analyzeUnit`,
  `analyzeTopology`, `ExecutionUnitPlan`/`UnitRuntimeFacts` must carry the new generated family and exact
  resource facts, never masquerade as direct fixed recurrence. Shared Prepare projection is already done.<span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry); [Prepare glossary](../../../../glossary.md#prepare))</span>
- `internal.lowering.CpuPartitionLowering`/`CpuPartitionDagDecomposer`/`CpuUnitTopology`: one indivisible
  scan unit, attachment-bearing query (current `assertOccurrence` drops it), ordinary neighbors and
  dependency edges preserved; exclude scan from pointwise contraction/recognition/representation search.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [contraction glossary](../../../../glossary.md#contraction-dimension))</span>
- `internal.ir.CpuKernelIr`/`CpuAccessPlan`: reuse typed arithmetic and dense/scalar accesses.
  Captured CompiledScanBody solely supplies graph/roles/demand; local IDs are never outer slots.
  Prove local shape/type/attrs under ordinary semantics; derive CPU virtual accesses without rewriting
  unresolved captured layouts or rejecting them via ordinary provider physical-layout guards.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [CPU glossary](../../../../glossary.md#term-cpu); [layouts glossary](../../../../glossary.md#layout))</span>
- `internal.codegen.emit.CpuClassFileKernelGenerator`, `CpuLoopEmitter`, `CpuScalarEmitter`,
  `CpuCarrierEmitter`: reuse generation-time typed emission, not existing entry invocation per step.
- `internal.cache.CpuKernelSpecialization`, `CpuGeneratorSchema`, `CpuGeneratedKernelArtifactStore`,
  `internal.route.portable.CpuPortableRoutePlan`: matching typed scan compatibility, generation/verification
  and current weak-intern/trusted-envelope reuse. Keep equality checks; no fingerprint-mismatch workaround.<span data-glossary-navigation="true"> ([fingerprint glossary](../../../../glossary.md#term-fingerprint))</span>
- `internal.prepare.CpuPartitionFinalizer`/`CpuPreparedScheduleAssembler`,
  `internal.executable.CpuPreparedPartitionExecutable`, `internal.memory.CpuBufferRepresentation`,
  `CpuBufferArgument`, `CpuContiguousWorkspace`, `CpuRepresentationRecipes`: existing slot/representation
  contributions, typed cold binding, complete overlap checks, cleanup and exactly one RunState.<span data-glossary-navigation="true"> ([RunState glossary](../../../../glossary.md#run-state--runstate))</span>

New production types (all unsupported internal, no moves/[dependency](../../../../glossary.md#term-dependency)/public [facade](../../../../glossary.md#term-facade)):

| Exact package suffix under `io.github.pho001.synaptik.backend.cpu` | Type | Concrete responsibility |
|---|---|---|
| `.internal.lowering` | `CpuGeneralScanLowering` (nested [immutable](../../../../glossary.md#term-immutable) `Facts`, `Geometry`) | Whole-query/binder proof and exact cold resource/access facts from captured body. |
| `.internal.ir` | `CpuGeneralScanIr` | Family-specific derived code-shaping facts: existing CpuKernelIr arithmetic plus typed positional/ordinal [bindings](../../../../glossary.md#term-binding); not a copied [graph](../../../../glossary.md#graph)/new semantic body. |
| `.internal.codegen.emit` | `CpuGeneralScanEmitter` | Generate outer time loop and inline existing typed [scalar](../../../../glossary.md#term-scalar)/[carrier](../../../../glossary.md#term-storage-carrier) body emission with explicit bindings. |
| `.internal.executable` | `CpuGeneralScanPreparedExecutable` (private `Invocation`) | Cold-bind assigned representations to one generated call; no graph retained in [runtime](../../../../glossary.md#term-runtime) recipe. |

Extend the [sealed](../../../../glossary.md#term-sealed) portable [IR](../../../../glossary.md#term-intermediate-representation-ir)/route encoding and [specialization](../../../../glossary.md#term-specialization) with typed optional [scan](../../../../glossary.md#term-cumulative-scan) facts,
analogous to existing typed [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) facts; all four types solve this family only, not a region abstraction.

## [Binding](../../../../glossary.md#term-binding), storage and [identity](../../../../glossary.md#term-identity-meanings) decisions

1. Analyze BEFORE slots: retain captured body as cold source; lower arithmetic once to CpuKernelIr.
   CpuGeneralScanIr records formal→arithmetic ordinals, next/emission/residual ordinals, positional
   actual→unique read map, demanded outer→write map, carry-bank assignments, expand aliases and direction.
   Its encoded CpuKernelIr is a boundary/identity envelope, not another graph; structuralKey must equal
   that envelope's key. Typed scan facts supply emission, not parsed strings or a body registry.<span data-glossary-navigation="true"> ([residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [carry glossary](../../../../glossary.md#term-carry); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [graph glossary](../../../../glossary.md#graph); [scan glossary](../../../../glossary.md#term-cumulative-scan); [registry glossary](../../../../glossary.md#term-registry))</span>
2. Entry ABI: B distinct read/demanded-write MemorySegment parameters, optional scratch MemorySegment,
   then run-owned long[] geometry, long start, long end (one serial complete invocation, <span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [MemorySegment glossary](../../../../glossary.md#term-memorysegment); [parameters glossary](../../../../glossary.md#term-parameter); [run glossary](../../../../glossary.md#term-run))</span>`[0,N)`).
   Select segment access at prepare time for heap/native/mixed inputs through checked segmentArgument;
   no staging copy/carrier discovery at execution. Exact parameter slots = B + scratchPresent(0/1)<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [heap glossary](../../../../glossary.md#term-heap); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
   + geometry(1) + start/end(4) = B+5+scratchPresent, at most 254 for the current JDK26 findStatic
   MethodHandle path (despite JVM static descriptors allowing 255). With scratch, B<=248; without,
   B<=249 by slots, but the conservative positional budget above remains 248. Reject before finalization.
   Keep last long range parameter so firstGeneratedLocal remains valid; reserve CpuCarrierEmitter's
   six-layout block FIRST, once. Load layouts/T/N/bases once; guard empty time/space before body reads,
   then hoist scalar invariant loads before time. Cold-bind one typed MethodHandle spreader over a
   run-owned MemorySegment[] (no Object[]); exactly one invokeExact per scan invocation, never per step.<span data-glossary-navigation="true"> ([MethodHandle glossary](../../../../glossary.md#native-numerical-library-and-instruction-architecture); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [descriptors glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout); [guard glossary](../../../../glossary.md#term-guard); [scalar glossary](../../../../glossary.md#term-scalar); [invariant glossary](../../../../glossary.md#term-invariant))</span>
3. CpuLoopEmitter currently assumes geometry/start/end at B/B+1/B+3 and entry carriers at fixed slots.
   Add an explicit generation-local dense-loop binding overload (range/base/carrier/local slots), with
   existing entry overload unchanged; NEVER overwrite ABI parameter slots or allocate layout blocks per
   step. CpuCarrierEmitter.load/store accept carrier/local address slots; CpuScalarEmitter consumes
   preallocated ordinal→FLOAT locals. Bind EXPAND aliases to the scalar local, not a materialized array.
   For each t, dense addresses are base+i or base+t*N+i; scalar-all-zero reads retain base unchanged.
   Convert element offsets to byte offsets once as needed; key this dense/scalar binding distinction.
   Update primitive time bases once per step; no geometry-array reconstruction or MethodHandle per step.
   Select initial-versus-bank carrier references outside the coordinate loop, not per element.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
4. Loop order is time outside coordinate. Load ALL old carry/needed input locals for i, save demanded
   prior-carry residuals BEFORE overwriting, emit instructions in preserved order, save demanded exact
   produced sites, then write next carries/emissions. Reverse t descends, but all stacked stores use
   original t. Intermediate FLOAT locals preserve producing-site rounding; no activation replay/FMA
   across observable saved/state boundaries. No per-op lookup/callback or per-step allocation.<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [activation glossary](../../../../glossary.md#term-activation); [FMA glossary](../../../../glossary.md#term-fused-multiply-add-fma); [lookup glossary](../../../../glossary.md#term-lookup); [callback glossary](../../../../glossary.md#term-callback))</span>
5. First step reads initial input directly; no initial-copy staging/ping-pong. Prefer an ALREADY
   demanded emission/produced-value residual frame as carry backing when it saves the EXACT same
   represented next value/producing site, unchanged through its last carry read. Allocate/demand NO new
   history for reuse; a prior-carry residual is NOT automatically a saved next value. With traversal
   d=+1 forward/-1 reverse, step t after the first reads original-index frame t-d. At exit, copy the
   final saved frame into EACH distinct demanded final-carry output ONCE; T=0 raw-initial rule unchanged.
   Otherwise use a justified independent bank per evolving role: demanded final output or N-cell scratch.
   Other trivial stable elisions: unchanged self-carry backed by immutable initial input, invariant,
   or immutable sequence slice with its previous-step offset preserved through last read. Prove temporal
   lifetime, not just local load order: <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [slice glossary](../../../../glossary.md#slice); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>`nextCarry=(h+xStep,h)` must preserve old h independently before
   overwriting the first bank. Loading all locals protects NOW, not the next step. Share tied next
   bindings only for the SAME time-version through last read; otherwise retain separate storage.
   No generic optimizer/proof framework. Redundant bank stores require explicit clean-Java/dataflow
   rationale AND evidence; eliminated stores are avoidable work, not a guaranteed speedup.
   Body temporaries are locals, not arrays; no proved pointwise/expand step materialization.
   Preserve raw identity carries; scratch is zero for T<=1/N=0 when no bank is needed. No dead histories.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [dataflow glossary](../../../../glossary.md#term-dataflow); [pointwise glossary](../../../../glossary.md#term-pointwise); [materialization glossary](../../../../glossary.md#materialization))</span>
6. Declare checked exact bytes before slot assignment: demanded final outputs 4*N each; demanded
   emissions/residuals 4*T*N each (or their validated local [shape](../../../../glossary.md#shape) count); scratch 4*N*K for K necessary
   non-output carry banks, zero otherwise; staging/body-array temporaries zero. Include read spans,
   alignment and checked sums/offsets. Distinct demanded wrappers remain distinct outputs even for tied
   roles; no aliasing away publications. Internal uses alone allocate no history; reverse dead residuals
   also have zero bytes AND stores. Requested emissions still cost O(T), and live histories remain exact.
   Finalize only AFTER assigned slots, with no new requirement/choice. Mutable arrays/addresses/carries/
   changing geometry exist only in each run's typed Invocation under its ONE RunState; prepared arrays
   are immutable snapshots, never mutated or passed as writable cursor state. Runtime sees no graph/query.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment); [aliasing glossary](../../../../glossary.md#term-aliasing); [Mutable glossary](../../../../glossary.md#term-mutable); [RunState glossary](../../../../glossary.md#run-state--runstate); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
7. Advance global CpuGeneratorSchema CURRENT_VERSION 70→71, append family classIdentitySchema 71;
   preserve existing family projections but prefix them with 71 as current code does. Key every baked
   arithmetic opcode/order/immediate/type, alias/role multiplicity, direction, expand/access/rank form,
   carry-bank/frame-source/write mappings, demanded saved ports/site, ABI/scratch form and carrier choice.
   Declare frame/bank choices before slots; key every baked frame/source mapping.
   Concrete T, N/extents, target-shape sizes used only as geometry, slots/representation positions,
   base offsets/addresses, run arrays, seeds/invariant VALUES and source IDs stay outside structural
   identity. Any baked shape guard must instead enter identity. Reuse across COMPATIBLE T/extents/
   slot IDs only while baked bank/ABI/scratch/access forms remain unchanged; otherwise a keyed miss.
   Mutate each baked fact with miss/changed bytes. Reject schema70/stale/corrupt envelopes; retain
   authentication/verification, no new persistent store, tuning identity or broad cache infrastructure.<span data-glossary-navigation="true"> ([opcode glossary](../../../../glossary.md#term-opcode); [rank glossary](../../../../glossary.md#term-rank); [target glossary](../../../../glossary.md#term-target); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [authentication glossary](../../../../glossary.md#term-authentication); [cache glossary](../../../../glossary.md#term-cache))</span>

## Acceptance criteria and [validation](../../../../glossary.md#term-validation)

- New [CPU](../../../../glossary.md#term-cpu) tests: `CpuGeneralScanCapabilityTest`, `CpuGeneralScanLoweringTest`,
  `CpuGeneralScanPreparationTest`, `CpuGeneralScanExecutionTest`, `CpuGeneralScanGeneratedKernelTest`,
  `CpuGeneralScanTimingTest`; test-only `CpuGeneralScanCleanJavaOracle` in `internal.codegen.emit`.
  Reuse current `CpuPartitionDagResourceTest`, [cache](../../../../glossary.md#term-cache)/[specialization](../../../../glossary.md#term-specialization) tests and fixed-[scan](../../../../glossary.md#term-cumulative-scan) regressions.
- Real provider accepts promised forward AND actual Compiler-generated reverse queries; reject missing
  body, malformed roles/demand, dynamic/unqualified-layout/mixed/F64/low descriptors, supported child with
  unsupported binder, RELU/other unsupported interior and generated SUM/cast occurrence even at T=0/N=0.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [layout glossary](../../../../glossary.md#layout); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [descriptors glossary](../../../../glossary.md#term-descriptor); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Budget regressions: `h+xStep` followed by 64 TANHs qualifies forward but its actual generated reverse
  exceeds node/formal limits; full backward compile must fail closed at backend qualification BEFORE
  physical allocation/class generation. Prove 254-total-slot validity (B=248 with scratch) and reject
  255 slots before finalization, not by catching a late findStatic failure; test optional-scratch accounting.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [compile glossary](../../../../glossary.md#compile); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [backend glossary](../../../../glossary.md#backend); [qualification glossary](../../../../glossary.md#term-qualification))</span>
- Add `CpuGeneralScanConformanceTest` and `EngineCpuGeneralScanIntegrationTest` in existing testing
  conformance/integration packages: public Engine forward/backward and compile seeds against ordinary
  unrolled Tensor autograd AND specialized optimal clean Java, both directions, T=0/1/3/large, empty S,
  two carries, tied actual/result roles, both/missing seed classes, shared S invariants, unused/disconnected
  inputs, reused templates and cross-occurrence sums. Preserve h0=false with trainable sequence/weights;
  outer/formal flags unchanged. Verify actual saved values/raw special sites, not recomputed activations.
  Explicitly test rank-16 S/stacked rank 17 in CPU-only forward/backward; test <span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [unrolled glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [Tensor glossary](../../../../glossary.md#tensor); [autograd glossary](../../../../glossary.md#autograd); [seed glossary](../../../../glossary.md#term-seed); [weights glossary](../../../../glossary.md#term-weight); [activations glossary](../../../../glossary.md#term-activation); [rank glossary](../../../../glossary.md#term-rank))</span>`nextCarry=(h+xStep,h)`
  for >=3 steps in both directions AND its generated reverse, against unrolled and direct oracles.
  Frame reuse: forward/reverse T=0/1/>=3, `next=emit=tanh(h+xStep)` and demanded produced-value
  residual cases, final carry requested/unrequested; assert exact values, actual scratch bytes and
  store counts (no duplicate per-step bank stores, one final copy per demanded wrapper, no new history).<span data-glossary-navigation="true"> ([residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [carry glossary](../../../../glossary.md#term-carry))</span>
- Qualify NaN/Inf, signed zero, divisor zero, overflow/underflow under existing family rules. Raw T=0
  final/seed copies exact, empty emissions/history and typed-zero parameter sums; no empty bypass of
  admission. Test checked size/ABI overflow before finalization, sliced heap/native/mixed read aliases,
  write/scratch overlap including capacity tails, fail-before-write, close/rollback/retry and concurrent
  runs with fresh inputs while another result remains open. Count actual bytes/stores, not metadata only.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [signed zero glossary](../../../../glossary.md#term-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [parameter glossary](../../../../glossary.md#term-parameter); [admission glossary](../../../../glossary.md#term-admission); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [heap glossary](../../../../glossary.md#term-heap); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [rollback glossary](../../../../glossary.md#term-rollback); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Update `LowPrecisionCapabilityLedgerTest` + `low-precision-capability-ledger-v2.tsv`: new body-bearing
  scan witnesses query ACTUAL CPU/Metal providers and low dtypes independently. Fix its current exact
  254-row historical-projection assumption: separately preserve/compare the old baseline subset, then
  validate additional current witnesses. Never rewrite v1 or relax old evidence to admit arbitrary drift.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [dtypes glossary](../../../../glossary.md#data-type--datatype))</span>
- Vary T (0/1/3/257) and N: source/captured/prepared object and class-byte sizes O(body), not O(T);
  exact demanded histories O(T), inference dead bytes/stores zero and reverse dead bytes/stores zero.
  Inspect Class-File/decompile for time/inner loops, layout/invariant hoists, saved-site order and no
  hidden Synaptik arithmetic helpers, allocation, boxing, graph dispatch, copies or per-step invocation.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [invariant glossary](../../../../glossary.md#term-invariant); [graph glossary](../../../../glossary.md#graph); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Short opt-in timing: add CPU test-property forwarding for `synaptik.cpu.generalScan.performance`
  and `synaptik.cpu.generalScan.evidenceRoot`. One [JVM](../../../../glossary.md#term-java-virtual-machine-jvm), <=60s measured workload, two matched cases
  `(T,N)=(8,32),(64,256)`, forward and forward+reverse, native and heap/native inputs, alternating short
  warmed samples. Oracle uses the actual MemorySegment-only generated-route carrier, including heap
  segments, with matched FLOAT32 sites/layout/work/validation; no primitive-array fast-path claim.
  Report generation/class-load/first-JIT cold, entry/preflight, prebound kernel and per-step costs; consume outputs;
  no timing assertions/five forks/parity claim. Fix diagnosed avoidable overhead or justify with data.<span data-glossary-navigation="true"> ([forks glossary](../../../../glossary.md#term-fork-meanings))</span><span data-glossary-navigation="true"> ([MemorySegment glossary](../../../../glossary.md#term-memorysegment); [carrier glossary](../../../../glossary.md#term-storage-carrier); [FLOAT32 glossary](../../../../glossary.md#term-float32); [JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit); [preflight glossary](../../../../glossary.md#term-preflight); [kernel glossary](../../../../glossary.md#kernel))</span>

Recorded worker commands (reused evidence; outcomes in Result):

```bash
./gradlew :backends:cpu:test --tests '*CpuGeneralScan*' --tests '*CpuGeneratedKernelArtifactStoreTest' --tests '*CpuKernelSpecializationTest' --console=plain
./gradlew :backends:cpu:test -Dsynaptik.cpu.generalScan.performance=true -Dsynaptik.cpu.generalScan.evidenceRoot=/private/tmp/synaptik-cpu0030 --tests '*CpuGeneralScanTimingTest' --rerun-tasks --console=plain
./gradlew :backends:cpu:test :backends:cpu:javadoc --console=plain
```

Timing/Class-File [artifacts](../../../../glossary.md#term-artifact) and summary use the explicit evidence root, recording [oracle](../../../../glossary.md#term-test-oracle), cases,
[warmup](../../../../glossary.md#term-warmup), [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), environment and limitations.
Main [checkpoint](../../../../glossary.md#term-checkpoint) commands, passed after final code/docs review (outcomes in Result):

```bash
./gradlew :testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest' --tests '*LowPrecisionCapabilityLedgerTest' --tests '*CpuRecurrentScanConformanceTest' --console=plain
./gradlew :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest' --tests '*EngineCpuRecurrentScanIntegrationTest' --console=plain
./gradlew :testing:architecture-tests:test --tests '*CpuDependencyContractTest' --tests '*RuntimeDependencyAndHotPathContractTest' --tests '*BackendLayoutContractTest' --console=plain
git diff --check
```

## Documentation and review impact

Completed [Javadoc](../../../../glossary.md#term-javadoc), [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend)-guide, targeted [scan](../../../../glossary.md#term-cumulative-scan)/[residual](../../../../glossary.md#scan-residual-port-and-history-demand) glossary and examples under general +
backend-guide/[API](../../../../glossary.md#term-application-programming-interface-api)-Javadoc profiles. Independent review covered diff, clean Java/[dataflow](../../../../glossary.md#term-dataflow)/Class-File,
resources, [identity](../../../../glossary.md#term-identity-meanings) and [Engine](../../../../glossary.md#term-engine)/ledger evidence; final Javadoc PASS. Main integrated [schema](../../../../glossary.md#term-schema)/status
references. [Metadata](../../../../glossary.md#term-metadata) closure reuses executable evidence and changes no Java or authority.

## Result

Bounded [F32](../../../../glossary.md#floating-point-precision-and-rounding) forward/actual reverse delivered; [CPU](../../../../glossary.md#term-cpu) files/tests, v2 ledger and docs finalized.
Reused exact commands above; no reruns:

- CPU: 216 suites/1132 tests, 1100 passed/32 skipped, 0 failures/errors.
- GeneralScan 22 PASS/1 opt-in skip; later [residual](../../../../glossary.md#scan-residual-port-and-history-demand) [guard](../../../../glossary.md#term-guard) 1 PASS, integration [fixture](../../../../glossary.md#term-fixture) 5 PASS.
- [Javadoc](../../../../glossary.md#term-javadoc) PASS, 94 baseline warnings (92 constructors + 2 incubator).
- Main fresh XML: [conformance](../../../../glossary.md#term-conformance) 11, [Engine](../../../../glossary.md#term-engine) 10 (5+5), architecture 5; 0 failures/errors/skips.
- Final independent code/docs/fix APPROVE, 0 P1/P2; workers/reviewers CLOSED.
- Docs: 12 files/11 Java; 53 [token](../../../../glossary.md#term-token) streams unchanged; 524 links/430 anchors, 0 broken.
- Global/family 71; 41601 [owners](../../../../glossary.md#term-owner) (41336 generated/265 reject) unchanged; only cols 1/19/20 delta.
- Timing: 8 cases/1 [JVM](../../../../glossary.md#term-java-virtual-machine-jvm)/6.121s diagnostic; [JIT](../../../../glossary.md#term-just-in-time-compilation-jit)/[warmup](../../../../glossary.md#term-warmup), mixed-small ~7x gap; no parity.
- Main 75-path freeze PASS; closure 9 plans only, other dirty paths unchanged.
- Links/status/size/diff PASS; no acceptance issue or successor launch.
- Mandatory later Draft: declared algebra/closure + [F64](../../../../glossary.md#floating-point-precision-and-rounding); root join/low/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[padding](../../../../glossary.md#padding)/[persistence](../../../../glossary.md#term-persistence) gates retained.

Status: Complete

Size exception ([target](../../../../glossary.md#term-target) 200 lines/15 KB): this one [atomic](../../../../glossary.md#resource-transaction-and-rollback) capability keeps whole reverse closure,
entry/step [ABI](../../../../glossary.md#term-application-binary-interface-abi), no-copy [carry](../../../../glossary.md#term-carry) proof, exact resources, [cache](../../../../glossary.md#term-cache) [identity](../../../../glossary.md#term-identity-meanings) and Engine/ledger gates together.
Splitting them would authorize an incomplete route; this packet stays below the 25 KB ceiling.
