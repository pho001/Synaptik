# ADR 0050: Dynamic batch geometry through numeric recipes

## Status

Accepted — authority enacted and MAIN-accepted 2026-10-24. User “ano” approves the reviewed contract and architecture
first, then ordinary CPU forward/actual first-order backward and bounded final A/B. Prepare0012A
completed the coordinated cutover; Pasteur definitive APPROVE, zero open P1/P2, docs/glossary
review complete. [Prepare0013](../../planning/modules/prepare/tasks/0013-ordinary-dynamic-batch-cpu-vertical.md#result)
now implements bounded ordinary homogeneous FLOAT32 CPU forward and actual first-order backward;
it is Complete, MAIN-accepted 2026-10-24 after final independent reviews and the native-enabled
integration gate passed. No automatic successor is authorized. ADR0050 was free when created.
Bounded qualification amendment — 2026-10-26: the user's “pokracuj” authorizes Prepare0015's
complete combined vertical, beginning with pre-Java support qualification. Its one-B/fixed-T/features
F32/F64 length-Scan and bounded outer/control target passed independent pre-Java review and MAIN acceptance.
Implementation amendment — 2026-10-26: Prepare0015 implements this bounded capability through
existing numeric recipes, including typed 34-limb/280-byte ordinary F64 reductions and explicit
B for zero-feature Scan. Owning validation and independent code/structural review passed; root validation and
the sole bounded probe completed;
[Prepare0015 Result](../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result)
records evidence and limits; final MAIN acceptance remains pending.
The nine-type inventory and numeric resource/creator/lease protocol are unchanged; no tenth type,
public API, dependency or memo is selected.

[`ARCHITECTURE.md`](../../../ARCHITECTURE.md) is the sole authority index; this ADR records rationale,
not another contract. The accepted [design packet](../../planning/modules/prepare/dynamic-batch-geometry-contract-proposal.md)
retains the exact reviewed signatures, nine-type map and historical design evidence.

## Context

At decision time, Model already represented symbolic dimensions, but capability checks, byte declarations,
zero-argument resource creators and output descriptors required static geometry. Runtime already
allocated fresh storage per run: sizing and admission, not freshness, prevented one immutable
compilation/preparation from serving actual B=64→40→1→64. Changing the binder alone would leave
resource sizing, actual derivative contractions and output metadata incorrect.

## Decision drivers

- Allocate actual run bytes without mandatory public B_max or per-B full-preparation cache.
- Preserve fixed code/routes/slots, semantic ownership, per-run isolation and static lifetime rules.
- Keep Model/Compiler metadata out of Runtime and geometry interpretation out of shared Prepare.
- Qualify actual automatic-differentiation (AD) closure and generated hot loops, not forward alone.

## Options considered

1. Keep fixed-only execution: preserves current implementation but cannot meet selected reuse.
2. Maximum-capacity banks or per-B fullprepare caches: impose a different allocation/lifecycle and
   public-bound or repeated-preparation requirement; not the selected product semantics.
3. Immutable prepared numeric recipes with per-run binding: chosen; separates semantic admission,
   exact physical sizes and execution while retaining existing owners.

## Decision

Keep eight CompileArtifacts components. Compiler derives BatchBindingPlan from final metadata and
private typed obligations. Prepare fixes complete declarations/slots and explicit batch-required
mode without evaluating backend geometry. CPU owns numeric recipes. Runtime resolves authenticated
same-plan geometry, passes exact byte/alignment inputs to creators and owns one isolated run;
Engine admits actual descriptors and publishes actual occurrence metadata. Permit B=0 when
operation predicates allow it, preserving empty MEAN/typed-zero contraction and static semantics.

The exact new inventory is nine types, no moves/removals, new module or dependency. Under
`io.github.pho001.synaptik`:

| Owner | New types (including nested types) |
|---|---|
| Compiler | `compiler.BatchBindingPlan`, `compiler.BatchBindingPlan.Binding` |
| Runtime | `runtime.memory.PreparedMemoryPlan.ByteSize`, `runtime.memory.PreparedMemoryPlan.FixedByteSize`, `runtime.run.RunMemoryGeometry` |
| CPU | `backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize`, `backend.cpu.internal.ir.CpuAccessPlan.BatchBinding`, `backend.cpu.internal.lowering.CpuMatmulLowering.BatchGeometry`, `backend.cpu.internal.lowering.CpuAggregateLowering.BatchGeometry` |

Read the table as the exact implemented nine-type inventory and ownership map for Prepare0013.
Signatures and invariants belong to
the linked scoped owners, not to this ADR or the planning packet.

## Rationale

Conditional aggregate scratch is not universally affine. Execution mode is independent of byte
independence: `X[B,0]` can require B even when every byte count is constant. Explicit mode avoids
silently admitting missing B. Numeric recipes need no second graph, execution owner or proof runtime.
The admitted int-limited FLOAT32 SUM bound fits five limbs for all N, including 0/1; retaining
optimal constant-five-limb emission avoids unnecessary dynamic-limb generality.
For the accepted bounded F64 outer SUM/MEAN target, the existing typed exact algorithm's
`2099+ceil(log2(N))` signed-bit bound fits 34 limbs at int-bounded N. Fixed 280-byte single-range
scratch (zero when no output cells, retaining the declaration) avoids dynamic limb selection;
static F64 domain-sized state and F32 five-limb bytes remain unchanged. The
[CPU owner](../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract)
bounds the added occurrence predicates and `ordinary-batch-typed-v1` identity; this does not qualify
general F64 MATMUL or change Scan's sequential SUM. Real typed rounding/empty/compatibility evidence
remains required, not inferred from static emitter support.

## Consequences

Inward records and creator single-abstract-method (SAM) consumers migrate explicitly; preserve static constructors,
accessors, routes and native consumers without claiming record/binary ABI compatibility. Actual
sizes, packed/access arrays, storage and publication descriptors belong to each run; retain
borrows/results/prepared leases, close precedence, reverse cleanup, rollback/retry and concurrency.
The cost is new checked cold admission/binding and cross-owner migration, not hidden hot-loop work.

Ordinary homogeneous FLOAT32 CPU forward **and actual first-order backward** precede general Scan.
NN/Training one-state reuse, Compiler/Checkpoint cold acquisition, wire, builder, native/mixed and
tuning qualify separately; static length skipping is independent. The implemented first proof uses grad-eligible
X `[B,4]`, W `[4,3]`, Y=X@W, L=mean(Y), all-one data and 64→40→1→64: Y=L=4,
dX=1/B, dW=1/3 under existing rounding, with actual descriptors/resources. This is an illustrative
tested scenario, not runnable code; the [running guide](../../user-guide/running-models.md)
provides the public seeded scalar-loss example. Fixed-feature mismatch rejects.

Prepare0013 validation covers actual AD closure/count/contraction, B=0/1, constant-byte mode, conditional
scratch, overflow/admission/storage, failure/cleanup/retry/concurrency and static/native retention.
Use optimal specialized clean-Java algorithm/hot-loop/dataflow and Class-File inspection, then
bounded warmed matched-work generated/direct A/B at invocation B. Separate cold generation/
classload/JIT, admission and kernel costs; diagnose overhead without identical-JIT/zero-penalty claims.
The original authority-only cutover changed no executable/dependency and ran no measurement.
Prepare0013's bounded timing evidence discloses variance, unmatched medium Engine routes and a
remaining exact-MEAN probe gap; it establishes neither negligible overhead nor a general speedup.
Final code/Javadoc/performance and shared-document reviews approve with zero open P1/P2;
MAIN's native-enabled root gate passed and MAIN accepted the capability on 2026-10-24.

## Related documentation

- [Root lifecycle and authority](../../../ARCHITECTURE.md#core-lifecycle)
- [Runtime sizing/creation](../../architecture/contracts/runtime-prepare-engine.md#accepted-symbolic-batch-resource-contract)
  and [Engine admission/publication](../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution)
- [Compiler binding](../../architecture/contracts/compiler-autograd.md#accepted-symbolic-batch-metadata-binding)
  and [CPU recipes/qualification](../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract)
- [Scan gates](../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)
  and [owned-Model gates](../../architecture/contracts/extensions-training.md#owned-tensor-models)
- [Boundary explanation](../../architecture/runtime-prepare-backend-boundary.md#accepted-symbolic-batch-handoff)
  and [Prepare0012A](../../planning/modules/prepare/tasks/0012a-dynamic-batch-authority-cutover.md)
