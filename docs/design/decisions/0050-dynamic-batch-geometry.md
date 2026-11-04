# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0050: Dynamic [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) geometry through numeric recipes


## Editorial reading context

The application changes an admitted batch extent while reusing one prepared computation. Compiler validates symbolic relationships; CPU recipes calculate numeric sizes; Runtime allocates this run's resources; Engine publishes actual descriptors. Reusing a recipe is not reusing previous input values or buffers. Zero batch remains operation-dependent rather than a blanket promise.

Use the [runtime binding and reusable preparations](../../user-guide/running-models.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted — authority enacted and MAIN-accepted 2026-10-24. User “ano” approves the reviewed contract and architecture
first, then ordinary [CPU](../../glossary.md#term-cpu) forward/actual first-order backward and bounded final A/B. Prepare0012A
completed the coordinated cutover; Pasteur definitive APPROVE, zero open P1/P2, docs/glossary
review complete. [Prepare0013](../../planning/modules/prepare/tasks/0013-ordinary-dynamic-batch-cpu-vertical.md#result)
now implements bounded ordinary [homogeneous](../../glossary.md#term-homogeneous) [FLOAT32](../../glossary.md#term-float32) CPU forward and actual first-order backward;
it is Complete, MAIN-accepted 2026-10-24 after final independent reviews and the native-enabled
integration gate passed. No automatic successor is authorized. ADR0050 was free when created.
Bounded [qualification](../../glossary.md#term-qualification) amendment — 2026-10-26: the user's “pokracuj” authorizes Prepare0015's
complete combined vertical, beginning with pre-Java support qualification. Its one-B/fixed-T/[features](../../glossary.md#term-feature)
[F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding) length-[Scan](../../glossary.md#term-cumulative-scan) and bounded outer/control [target](../../glossary.md#term-target) passed independent pre-Java review and MAIN acceptance.
Implementation amendment — 2026-10-26: Prepare0015 implements this bounded capability through
existing numeric recipes, including typed 34-[limb](../../glossary.md#exact-superaccumulator-and-limbs)/280-byte ordinary F64 reductions and explicit
B for zero-feature Scan. Owning [validation](../../glossary.md#term-validation) and independent code/structural review passed; root validation and
the sole bounded probe completed;
[Prepare0015 Result](../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result)
records evidence and limits; final MAIN acceptance remains pending.
The nine-type inventory and numeric resource/creator/[lease](../../glossary.md#term-lease) protocol are unchanged; no tenth type,
public [API](../../glossary.md#term-application-programming-interface-api), [dependency](../../glossary.md#term-dependency) or [memo](../../glossary.md#cache-and-memoization) is selected.

[`ARCHITECTURE.md`](../../../ARCHITECTURE.md) is the sole authority index; this [ADR](../../glossary.md#term-architecture-decision-record-adr) records rationale,
not another contract. The accepted [design packet](../../planning/modules/prepare/dynamic-batch-geometry-contract-proposal.md)
retains the exact reviewed [signatures](../../glossary.md#term-signature), nine-type map and historical design evidence.

## Context

At decision time, [Model](../../glossary.md#term-model) already represented symbolic [dimensions](../../glossary.md#dimension), but capability checks, byte declarations,
zero-[argument](../../glossary.md#term-argument) resource creators and output [descriptors](../../glossary.md#term-descriptor) required static geometry. [Runtime](../../glossary.md#term-runtime) already
allocated fresh storage per [run](../../glossary.md#term-run): sizing and [admission](../../glossary.md#term-admission), not freshness, prevented one [immutable](../../glossary.md#term-immutable)
[compilation](../../glossary.md#compiler-engine-and-execution-stage-names)/preparation from serving actual B=64→40→1→64. Changing the binder alone would leave
resource sizing, actual [derivative](../../glossary.md#term-derivative) contractions and output [metadata](../../glossary.md#term-metadata) incorrect.

## Decision drivers

- Allocate actual [run](../../glossary.md#term-run) bytes without mandatory public B_max or per-B full-preparation [cache](../../glossary.md#term-cache).
- Preserve fixed code/routes/slots, semantic [ownership](../../glossary.md#term-ownership-meanings), per-run isolation and static [lifetime](../../glossary.md#term-lifetime) rules.
- Keep [Model](../../glossary.md#term-model)/[Compiler](../../glossary.md#term-compiler) [metadata](../../glossary.md#term-metadata) out of [Runtime](../../glossary.md#term-runtime) and geometry interpretation out of shared [Prepare](../../glossary.md#prepare).
- Qualify actual automatic-differentiation ([AD](../../glossary.md#autograd)) closure and generated hot loops, not forward alone.

## Options considered

1. Keep fixed-only execution: preserves current implementation but cannot meet selected reuse.
2. Maximum-capacity banks or per-B fullprepare caches: impose a different allocation/lifecycle and
   public-bound or repeated-preparation requirement; not the selected product semantics.<span data-glossary-navigation="true"> ([lifecycle glossary](../../glossary.md#lifecycle))</span>
3. Immutable prepared numeric recipes with per-run binding: chosen; separates semantic admission,
   exact physical sizes and execution while retaining existing owners.<span data-glossary-navigation="true"> ([owners glossary](../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([Immutable glossary](../../glossary.md#term-immutable); [run glossary](../../glossary.md#term-run); [binding glossary](../../glossary.md#term-binding); [admission glossary](../../glossary.md#term-admission))</span>

## Decision

Keep eight CompileArtifacts components. [Compiler](../../glossary.md#term-compiler) derives BatchBindingPlan from final [metadata](../../glossary.md#term-metadata) and
private typed obligations. [Prepare](../../glossary.md#prepare) fixes complete declarations/slots and explicit [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-required
mode without evaluating [backend](../../glossary.md#backend) geometry. [CPU](../../glossary.md#term-cpu) owns numeric recipes. [Runtime](../../glossary.md#term-runtime) resolves authenticated
same-plan geometry, passes exact byte/[alignment](../../glossary.md#term-alignment) inputs to creators and owns one isolated [run](../../glossary.md#term-run);
[Engine](../../glossary.md#term-engine) admits actual [descriptors](../../glossary.md#term-descriptor) and publishes actual [occurrence](../../glossary.md#term-occurrence) metadata. Permit B=0 when
[operation](../../glossary.md#operation) predicates allow it, preserving empty [MEAN](../../glossary.md#term-mean)/typed-zero [contraction](../../glossary.md#contraction-dimension) and static semantics.

The exact new inventory is nine types, no moves/removals, new [module](../../glossary.md#term-module) or [dependency](../../glossary.md#term-dependency). Under
`io.github.pho001.synaptik`:

| [Owner](../../glossary.md#term-owner) | New types (including nested types) |
|---|---|
| Compiler | `compiler.BatchBindingPlan`, `compiler.BatchBindingPlan.Binding` |
| Runtime | `runtime.memory.PreparedMemoryPlan.ByteSize`, `runtime.memory.PreparedMemoryPlan.FixedByteSize`, `runtime.run.RunMemoryGeometry` |
| CPU | `backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize`, `backend.cpu.internal.ir.CpuAccessPlan.BatchBinding`, `backend.cpu.internal.lowering.CpuMatmulLowering.BatchGeometry`, `backend.cpu.internal.lowering.CpuAggregateLowering.BatchGeometry` |

Read the table as the exact implemented nine-type inventory and [ownership](../../glossary.md#term-ownership-meanings) map for Prepare0013.
[Signatures](../../glossary.md#term-signature) and invariants belong to
the linked scoped owners, not to this [ADR](../../glossary.md#term-architecture-decision-record-adr) or the [planning](../../glossary.md#term-planning) packet.

## Rationale

Conditional aggregate scratch is not universally affine. Execution mode is independent of byte
independence: `X[B,0]` can require B even when every byte count is [constant](../../glossary.md#term-constant). Explicit mode avoids
silently admitting missing B. Numeric recipes need no second [graph](../../glossary.md#graph), execution [owner](../../glossary.md#term-owner) or proof [runtime](../../glossary.md#term-runtime).
The admitted int-limited [FLOAT32](../../glossary.md#term-float32) SUM bound fits five limbs for all N, including 0/1; retaining
optimal constant-five-[limb](../../glossary.md#exact-superaccumulator-and-limbs) emission avoids unnecessary dynamic-limb generality.
For the accepted bounded [F64](../../glossary.md#floating-point-precision-and-rounding) outer SUM/[MEAN](../../glossary.md#term-mean) [target](../../glossary.md#term-target), the existing typed exact algorithm's
`2099+ceil(log2(N))` signed-bit bound fits 34 limbs at int-bounded N. Fixed 280-byte single-range
scratch (zero when no output cells, retaining the declaration) avoids dynamic limb selection;
static F64 domain-sized state and [F32](../../glossary.md#floating-point-precision-and-rounding) five-limb bytes remain unchanged. The
[CPU owner](../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract)<span data-glossary-navigation="true"> ([CPU glossary](../../glossary.md#term-cpu))</span>
bounds the added [occurrence](../../glossary.md#term-occurrence) predicates and `ordinary-batch-typed-v1` [identity](../../glossary.md#term-identity-meanings); this does not qualify
general F64 [MATMUL](../../glossary.md#matrix-multiplication--matmul) or change [Scan](../../glossary.md#term-cumulative-scan)'s sequential SUM. Real typed [rounding](../../glossary.md#term-rounding)/empty/compatibility evidence
remains required, not inferred from static emitter support.

## Consequences

Inward records and creator single-abstract-method (SAM) consumers migrate explicitly; preserve static constructors,
accessors, routes and native consumers without claiming record/[binary](../../glossary.md#term-binary-descriptor-and-metadata) [ABI](../../glossary.md#term-application-binary-interface-abi) compatibility. Actual
sizes, packed/access arrays, storage and [publication](../../glossary.md#term-publication) [descriptors](../../glossary.md#term-descriptor) belong to each [run](../../glossary.md#term-run); retain
borrows/results/prepared [leases](../../glossary.md#term-lease), close precedence, reverse cleanup, [rollback](../../glossary.md#term-rollback)/retry and [concurrency](../../glossary.md#term-concurrency).
The cost is new checked cold [admission](../../glossary.md#term-admission)/[binding](../../glossary.md#term-binding) and cross-[owner](../../glossary.md#term-owner) migration, not hidden hot-loop work.

Ordinary [homogeneous](../../glossary.md#term-homogeneous) [FLOAT32](../../glossary.md#term-float32) [CPU](../../glossary.md#term-cpu) forward **and actual first-order backward** precede [general Scan](../../glossary.md#general-differentiable-scan).
[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training one-state reuse, [Compiler](../../glossary.md#term-compiler)/[Checkpoint](../../glossary.md#term-checkpoint) cold acquisition, wire, builder, native/mixed and
tuning qualify separately; static length skipping is independent. The implemented first proof uses grad-eligible
X `[B,4]`, W `[4,3]`, Y=X@W, L=[mean](../../glossary.md#term-mean)(Y), all-one data and 64→40→1→64: Y=L=4,
dX=1/B, dW=1/3 under existing [rounding](../../glossary.md#term-rounding), with actual descriptors/resources. This is an illustrative
tested scenario, not runnable code; the [running guide](../../user-guide/running-models.md)
provides the public seeded [scalar](../../glossary.md#term-scalar)-[loss](../../glossary.md#loss) example. Fixed-[feature](../../glossary.md#term-feature) mismatch rejects.

Prepare0013 [validation](../../glossary.md#term-validation) covers actual [AD](../../glossary.md#autograd) closure/count/[contraction](../../glossary.md#contraction-dimension), B=0/1, [constant](../../glossary.md#term-constant)-byte mode, conditional
scratch, [overflow](../../glossary.md#term-overflow)/admission/storage, failure/cleanup/retry/concurrency and static/native retention.
Use optimal specialized clean-Java algorithm/hot-loop/[dataflow](../../glossary.md#term-dataflow) and Class-File inspection, then
bounded warmed matched-work generated/direct A/B at invocation B. Separate cold generation/
classload/[JIT](../../glossary.md#term-just-in-time-compilation-jit), admission and [kernel](../../glossary.md#kernel) costs; diagnose overhead without identical-JIT/zero-penalty claims.
The original authority-only cutover changed no executable/[dependency](../../glossary.md#term-dependency) and ran no measurement.
Prepare0013's bounded timing evidence discloses [variance](../../glossary.md#term-variance), unmatched medium [Engine](../../glossary.md#term-engine) routes and a
remaining exact-MEAN probe gap; it establishes neither negligible overhead nor a general speedup.
Final code/[Javadoc](../../glossary.md#term-javadoc)/performance and shared-document reviews approve with zero open P1/P2;
MAIN's native-enabled root gate passed and MAIN accepted the capability on 2026-10-24.

## Related documentation

- [Root lifecycle and authority](../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../glossary.md#lifecycle))</span>
- [Runtime sizing/creation](../../architecture/contracts/runtime-prepare-engine.md#accepted-symbolic-batch-resource-contract)<span data-glossary-navigation="true"> ([Runtime glossary](../../glossary.md#term-runtime))</span>
  and [Engine admission/publication](../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution)<span data-glossary-navigation="true"> ([Engine glossary](../../glossary.md#term-engine); [admission glossary](../../glossary.md#term-admission); [publication glossary](../../glossary.md#term-publication))</span>
- [Compiler binding](../../architecture/contracts/compiler-autograd.md#accepted-symbolic-batch-metadata-binding)<span data-glossary-navigation="true"> ([Compiler glossary](../../glossary.md#term-compiler); [binding glossary](../../glossary.md#term-binding))</span>
  and [CPU recipes/qualification](../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract)<span data-glossary-navigation="true"> ([CPU glossary](../../glossary.md#term-cpu); [qualification glossary](../../glossary.md#term-qualification))</span>
- [Scan gates](../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([Scan glossary](../../glossary.md#term-cumulative-scan))</span>
  and [owned-Model gates](../../architecture/contracts/extensions-training.md#owned-tensor-models)<span data-glossary-navigation="true"> ([Model glossary](../../glossary.md#term-model))</span>
- [Boundary explanation](../../architecture/runtime-prepare-backend-boundary.md#accepted-symbolic-batch-handoff)
  and [Prepare0012A](../../planning/modules/prepare/tasks/0012a-dynamic-batch-authority-cutover.md)
