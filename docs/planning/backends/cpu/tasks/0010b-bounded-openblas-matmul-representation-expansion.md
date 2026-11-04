# Task 0010B: Bounded [OpenBLAS](../../../../glossary.md#term-openblas) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) Representation Expansion

## Reader context — historical integration boundary

The representation expansion changes where values live around one matrix call, not what MATMUL
means. CPU analysis selects input copies and/or output copy-out; shared Prepare assigns the declared
workspace slots; Runtime creates distinct physical workspaces for each run. The prepared recipe then
copies left and right inputs as selected, calls OpenBLAS once, and optionally copies to the logical
output. Input storage remains borrowed. A reusable copy recipe does not preserve copied matrix
values between runs, and an output workspace is not an input materialization. The eight masks and
costs below are historical selection facts, not measured speed guarantees.

Current reading: [Current OpenBLAS route](../../../../backend-guide/cpu-backend.md#current-narrow-openblas-matmul-route).

## Status

Complete

## Goal

Expand the existing [CPU](../../../../glossary.md#term-cpu)-owned [OpenBLAS](../../../../glossary.md#term-openblas) route from direct [matrices](../../../../glossary.md#term-matrix-meanings) or one copied input to the
complete bounded representation set for one positive, [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), same-type [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) bare
[`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul). CPU preparation may canonicalize either or both inputs before the provider call and may
compute into one [canonical](../../../../glossary.md#term-canonical-form) output workspace before copying to the logical result. The provider
continues to receive exactly one dense [row-major](../../../../glossary.md#term-row-major), non-transposed [SGEMM](../../../../glossary.md#general-matrix-multiplication--gemm) or [DGEMM](../../../../glossary.md#general-matrix-multiplication--gemm) call.

The selected plan must account for the complete per-[run](../../../../glossary.md#term-run) transition: workspace allocation and
[binding](../../../../glossary.md#term-binding), every input copy, the [GEMM](../../../../glossary.md#general-matrix-multiplication--gemm) call, optional output copy, workspace bytes, and expected run
count. Portable execution remains a valid preparation candidate, and every incomplete,
ineligible, tied, overflowing, or insufficient-benefit native candidate fails closed to that
portable plan during analysis. A selected native recipe never falls back later.

## [Scope](../../../../glossary.md#term-scope)

- Preserve the exact one-node, one-unit, bare rank-two MATMUL semantic boundary from CPU 0010:
  positive static <span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [CPU glossary](../../../../glossary.md#term-cpu))</span>`m`, `n`, and `k`; same-type FLOAT32 or same-type FLOAT64 inputs/result; exact
  default numerical mode; no rank promotion, batch prefix, broadcast batch, or epilogue.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [rank glossary](../../../../glossary.md#term-rank); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [epilogue glossary](../../../../glossary.md#term-epilogue))</span>
- Retain direct [canonical](../../../../glossary.md#term-canonical-form) native [matrices](../../../../glossary.md#term-matrix-meanings) where their current storage and [layout](../../../../glossary.md#layout) facts qualify.
- Reuse the existing typed access bindings and generated affine-copy machinery to canonicalize
  zero, one, or both inputs into distinct aligned native run-owned workspaces.<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding); [run glossary](../../../../glossary.md#term-run))</span>
- Add the smallest OpenBLAS-route-owned output-copy plan needed to compute into one canonical
  aligned native workspace and copy the represented result into an admitted logical output
  binding. Do not redefine the general external-read <span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>`CpuMaterializationPlan` as an output plan.
- Represent and validate all eight bounded copy masks over left input, right input, and output:
  direct; left; right; left+right; output; left+output; right+output; and left+right+output. Only
  masks required by the actual boundary facts are candidates.<span data-glossary-navigation="true"> ([masks glossary](../../../../glossary.md#term-boolean-mask))</span>
- Extend OpenBLAS finalization and its immutable prepared recipe so cold binding validates every
  buffer, workspace, generated copy, and provider precondition before the first write. Hot
  execution orders all selected input copies, exactly one GEMM, and the optional output copy.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [buffer glossary](../../../../glossary.md#term-buffer); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>
- Extend the CPU-private OpenBLAS cost snapshot with explicit non-negative representation terms
  for per-workspace allocation/binding, workspace bytes, copy-in invocation/elements, and
  copy-out invocation/elements. Keep provider/GEMM route terms separate.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [elements glossary](../../../../glossary.md#term-element))</span>
- Preserve stable shared workspace declarations in the immutable prepared plan. Physical
  workspaces remain run-owned and separately allocated for each [`RunState`](../../../../glossary.md#run-state--runstate); copied values are not
  persistent prepared resources and are rebuilt on every run.
- Add proportional native-free unit and backend-conformance coverage for the expanded
  representation boundary and extend the explicit real-native checkpoint for the supplied
  OpenBLAS 0.3.34 binary.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- Finalize affected Javadoc, package documentation, the CPU backend guide, glossary impact, this
  task, CPU master plan, and roadmap through a distinct clean documentation-focused context after
  executable Java stabilizes.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Any OpenBLAS-provider API, ABI, symbol, implementation, test, package, build, or documentation
  change. Provider calls remain row-major, non-transposed SGEMM/DGEMM with <span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [symbol glossary](../../../../glossary.md#term-symbol); [row-major glossary](../../../../glossary.md#term-row-major); [SGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [DGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>`alpha = 1` and
  `beta = 0`.
- Rank-one, rank greater than two, batch prefixes, batch broadcasting, repeated GEMM scheduling,
  strided-batched calls, dynamic Shapes, unresolved layouts, sparse, quantized, or complex
  MATMUL.<span data-glossary-navigation="true"> ([Rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [rank glossary](../../../../glossary.md#term-rank); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [broadcasting glossary](../../../../glossary.md#broadcasting); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [Shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- [`BFLOAT16`](../../../../glossary.md#term-bfloat16), mixed floating types, integral types, promotion, result conversion, or any operation
  other than bare MATMUL.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- [Bias](../../../../glossary.md#term-bias), [activation](../../../../glossary.md#term-activation), clamp, [`linear`](../../../../glossary.md#term-linear-meanings), attention, convolution, fused epilogues, provider transpose
  flags, provider packing APIs, panel packing, persistent packed weights, or workspace pooling.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [fused glossary](../../../../glossary.md#specialization-and-fusion); [transpose glossary](../../../../glossary.md#term-transpose); [weights glossary](../../../../glossary.md#term-weight); [pooling glossary](../../../../glossary.md#term-pooling))</span>
- Empty output or [contraction](../../../../glossary.md#contraction-dimension) geometry. `m == 0`, `n == 0`, or `k == 0` remains on the portable
  plan; this task adds no provider call or copy-only native route for those cases.
- Changing [CPU](../../../../glossary.md#term-cpu) 0010A discovery/loading, treating `LOADED` as `QUALIFIED`, binary qualification or
  fingerprinting, public Config/Engine composition, or provider lifetime ownership.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [qualification glossary](../../../../glossary.md#term-qualification); [Engine glossary](../../../../glossary.md#term-engine); [lifetime glossary](../../../../glossary.md#term-lifetime); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Changing the current externally coordinated `SINGLE_THREAD` provider rule, adding thread
  candidates, setting/restoring thread count in prepared execution, or introducing a CPU-wide
  concurrency coordinator. CPU 0010C owns those decisions.<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Model autotuning, tuning-cache lookup or mutation, measurement during preparation, general
  autotuning, persistent representation evidence, benchmarking, or performance claims. CPU 0010E
  and <span data-glossary-navigation="true"> ([Model autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [cache glossary](../../../../glossary.md#term-cache); [lookup glossary](../../../../glossary.md#term-lookup); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [benchmarking glossary](../../../../glossary.md#term-benchmarking))</span>`tools/tuning` own those later capabilities.
- Public Tensor, Config, Compile, Engine, Runtime, Prepare, Training, or CPU APIs; capability
  reporting; another <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare))</span>[`BackendId`](../../../../glossary.md#backend-identity--backendid); Planning ownership/cost logic; architecture or dependency
  changes; generated MATMUL implementations; or unrelated generator work.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Generalizing ordinary CPU 0008E representation selection, its `CO_CONSUMED_PAIR` rejection, or
  its maximum two external-read copies. The expanded copy masks are private to the selected
  OpenBLAS MATMUL route.<span data-glossary-navigation="true"> ([masks glossary](../../../../glossary.md#term-boolean-mask))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Planning ownership, staged
  preparation, Runtime resource ownership, CPU routes, the OpenBLAS provider, and performance
  evidence boundaries<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership); [Runtime glossary](../../../../glossary.md#term-runtime); [CPU glossary](../../../../glossary.md#term-cpu); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Runtime, Prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Performance evidence and tuning](../../../../architecture/performance-evidence-and-tuning.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [CPU 0005B access plans](0005b-universal-access-plans-and-right-aligned-broadcasting.md)
- [CPU 0005D materialization evidence](0005d-materialization-specialization-and-persistence-evidence-gate.md)<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- [CPU 0006 affine views and boundary materialization](0006-portable-static-affine-views-and-boundary-materialization.md)<span data-glossary-navigation="true"> ([views glossary](../../../../glossary.md#term-view-meanings))</span>
- [CPU 0008E bounded multi-input materialization](0008e-bounded-multi-input-materialization-and-representation-reuse.md)
- [CPU 0008E1 shared partition-DAG adoption](0008e1-shared-partition-dag-adoption-and-reconstruction-removal.md)<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag))</span>
- [CPU 0008F portable MATMUL](0008f-portable-matmul-execution-and-bounded-linear-epilogues.md)<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- [CPU 0009F1 MATMUL route decision](0009f1-matmul-and-convolution-route-decisions.md)
- [CPU 0010 narrow OpenBLAS route](0010-narrow-openblas-blas-compatible-native-route.md)
- [CPU 0010A discovery and composition foundation](0010a-automatic-openblas-discovery-and-internal-composition-foundation.md)
- [OpenBLAS provider master plan](../../openblas-provider/master-plan.md)
- [Provider 0001 loading and binding](../../openblas-provider/tasks/0001-library-loading-and-required-symbol-binding.md)<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- [Provider 0002 row-major GEMM](../../openblas-provider/tasks/0002-float32-float64-row-major-gemm-invocation.md)<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>
- [Provider 0003 thread control and checkpoint](../../openblas-provider/tasks/0003-thread-control-and-native-provider-checkpoint.md)<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

## Architecture constraints

- [Planning](../../../../glossary.md#term-planning) continues to select only `BackendId("cpu")`. CPU analysis alone constructs, filters,
  costs, and selects the OpenBLAS route and physical representation plan.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- Common CPU MATMUL lowering, exact rank-two geometry, typed access bindings, numerical mode,
  portable realization, and retained portable plan remain authoritative. The OpenBLAS leaf must
  consume those facts without graph reconstruction or semantic reinterpretation.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [lowering glossary](../../../../glossary.md#lowering); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [bindings glossary](../../../../glossary.md#term-binding); [graph glossary](../../../../glossary.md#graph))</span>
- Backend analysis selects the complete route, copy mask, costs, and exact shared resource
  requirements before slot assignment. Finalization may load existing affine-copy artifacts and
  build the immutable recipe, but it cannot change the selection or add a resource.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [mask glossary](../../../../glossary.md#term-boolean-mask); [artifacts glossary](../../../../glossary.md#term-artifact); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Runtime allocates and owns each run's declared workspaces and executes only the already-prepared
  sequence. It must not inspect layouts, select copies, select a route, discover a provider, or
  retry with portable execution.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [run glossary](../../../../glossary.md#term-run); [layouts glossary](../../../../glossary.md#layout))</span>
- `backends/openblas-provider` remains a low-level [leaf](../../../../glossary.md#term-leaf). The existing [dependency direction](../../../../glossary.md#java-platform-and-build-tools)
  `backends/cpu -> backends/openblas-provider` and the current provider surface remain unchanged.
- CPU 0010A discovery remains cold, bounded, and separate from qualification. Loading still does
  not authorize native selection.<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification))</span>
- Portable plans remain complete semantic alternatives during analysis. Native ineligibility
  retains portable; provider closure, thread drift, representation mismatch, copy failure, or
  provider failure after native selection is an error.
- Generated affine copies reuse the existing Java 26 Class-File machinery. If implementation
  discovers that an emitter, generated schema, or generated computation loop must change, stop
  and return the task to planning; do not introduce unrelated generator work under this task.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- Any need for a shared Prepare/Runtime contract, provider API, public Config/Engine API, module
  edge, architecture rule, or more general materialization contract is a stop condition.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Engine glossary](../../../../glossary.md#term-engine); [module glossary](../../../../glossary.md#term-module); [materialization glossary](../../../../glossary.md#materialization))</span>

## Current evidence and baseline

- CPU 0008F supplies complete static portable MATMUL geometry, access bindings, reference
  semantics, and the retained portable plan for every form in this task.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [bindings glossary](../../../../glossary.md#term-binding))</span>
- `CpuMatmulLowering` already accepts resolved non-negative rank-two layouts, computes exact
  offsets/strides/spans, and rejects non-injective output layouts. <span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [layouts glossary](../../../../glossary.md#layout); [strides glossary](../../../../glossary.md#term-stride); [injective glossary](../../../../glossary.md#term-injective))</span>`CpuAccessPlan.Binding` records
  the complete logical range and accessed element interval.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>
- `CpuRepresentationPlanner.openBlasInputMaterialization(...)` already builds one exact generated
  affine input copy into a canonical native workspace, but its fixed workspace identity and CPU
  0010 selector currently permit only one input copy.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `CpuMaterializationPlan` is specifically an external-read source-to-workspace plan with consumer
  coordinates. It cannot honestly represent a workspace-to-logical-output copy.<span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- `CpuPreparedPartitionExecutable` currently schedules zero through two general representation
  copies before all computation children. It has no post-computation copy phase. An OpenBLAS-
  specific sequence or a narrowly generalized phase contract is therefore required; the
  implementation must not disguise copy-out as an input materialization.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [materialization glossary](../../../../glossary.md#materialization))</span>
- `CpuOpenBlasRouteSelector` currently admits only canonical zero-offset output, requires the
  other input to be direct before considering one copied input, and ranks direct/copy-left/copy-
  right by complete checked cost.
- `CpuOpenBlasPreparedExecutable` currently consumes zero or one workspace, substitutes only one
  copied input, and performs one non-transposed provider call. Its provider/lifecycle/type/span
  validation is the baseline to preserve.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [validation glossary](../../../../glossary.md#term-validation))</span>
- `CpuContiguousWorkspace` already provides aligned native run-owned storage suitable for both
  canonical inputs and canonical output. Shared Prepare already assigns stable workspace slots
  from exact backend declarations; no shared resource contract change is evidenced.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- CPU 0010A adds only discovery metadata and an owned provider session. Neither source nor its
  tests participate in representation selection and they remain unchanged.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>

## Exact semantic and representation contract

### Logical geometry and empty cases

The only native semantic form remains:

```text
A[m,k] MATMUL B[k,n] -> C[m,n]
```

`m`, `n`, and `k` are static, positive, and no greater than `Integer.MAX_VALUE`; all [matrix](../../../../glossary.md#term-matrix-meanings)
[element](../../../../glossary.md#term-element) counts, byte counts, accessed spans, workspace totals, and cost arithmetic must succeed
exactly. `A`, `B`, and `C` have one exact common [FLOAT32](../../../../glossary.md#term-float32) or [FLOAT64](../../../../glossary.md#term-float64) type. The [occurrence](../../../../glossary.md#term-occurrence) is bare,
[rank](../../../../glossary.md#term-rank) two, exact/default, and has one output.

If any [dimension](../../../../glossary.md#dimension) is zero, if a dimension exceeds the provider integer range, or if checked
geometry overflows, no [OpenBLAS](../../../../glossary.md#term-openblas) candidate exists. Portable execution owns `m == 0`, `n == 0`, and
`k == 0`, including the positive-zero empty-[contraction](../../../../glossary.md#contraction-dimension) result. No workspace is declared for a
rejected native candidate.

### Accepted [affine access](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) geometry

For each boundary, the logical extents must equal its [matrix](../../../../glossary.md#term-matrix-meanings) extents, the [binding](../../../../glossary.md#term-binding) must cover the
complete [half-open](../../../../glossary.md#term-half-open-interval) logical range `[0, elementCount)`, its non-negative base offset and effective
[strides](../../../../glossary.md#term-stride) must address only the declared [referenced element span](../../../../glossary.md#referenced-element-span), and its access kind must be
`READ` for `A`/`B` or `WRITE` for `C`. The output binding must retain the current [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) lowerer's
[injectivity](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) proof.

In this task, a **[canonical](../../../../glossary.md#term-canonical-form) matrix** has zero base offset, [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) extents `[rows, columns]`,
effective [element](../../../../glossary.md#term-element) strides `[columns, 1]`, dense-[linear](../../../../glossary.md#term-linear-meanings) access, and an exact [referenced span](../../../../glossary.md#referenced-element-span) of
`rows * columns`. It is direct only when the expected boundary fact and cold-bound
representation also prove a type-width-aligned native segment, with a writable segment for `C`.

An **affine matrix [view](../../../../glossary.md#term-view-meanings)** is any other current rank-two binding with the exact complete range and
checked non-negative offset/strides/span above. Zero strides are permitted for read-only inputs
because they repeat a represented input element; they are not permitted for a multi-element
output because the existing injectivity proof rejects repeated writes. A **[transpose](../../../../glossary.md#term-transpose) view** is
not a provider flag or new semantic attribute: it is the affine special case whose logical [axes](../../../../glossary.md#term-axis)
address a dense underlying matrix in swapped order, such as logical extents `[rows, columns]`
with strides `[1, rows]`. [CPU](../../../../glossary.md#term-cpu) materializes its logical [row-major](../../../../glossary.md#term-row-major) values before the unchanged
provider call.

Any direct-boundary failure caused by [heap](../../../../glossary.md#term-heap) storage, nonzero offset, noncanonical strides, a larger
referenced span, or missing native/[alignment](../../../../glossary.md#term-alignment) proof may be resolved only by the corresponding
copy. A copied matrix's logical element count must fit the existing affine-copy address-table
limit and its exact bytes must fit the complete representation byte ceiling. Unresolved,
negative-stride, out-of-span, partial-range, non-[injective](../../../../glossary.md#term-injective)-output, or otherwise unproved geometry
is ineligible rather than guessed.

### Copy [masks](../../../../glossary.md#term-boolean-mask), workspaces, and execution order

The selector enumerates exactly these stable masks:

| Stable [rank](../../../../glossary.md#term-rank) | Representation | [Prepared execution](../../../../glossary.md#prepared-execution--preparedexecution) |
|---|---|---|
| 0 | direct | [GEMM](../../../../glossary.md#general-matrix-multiplication--gemm) |
| 1 | copy left | copy `A`, GEMM |
| 2 | copy right | copy `B`, GEMM |
| 3 | copy output | GEMM to workspace, copy to `C` |
| 4 | copy left + right | copy `A`, copy `B`, GEMM |
| 5 | copy left + output | copy `A`, GEMM to workspace, copy to `C` |
| 6 | copy right + output | copy `B`, GEMM to workspace, copy to `C` |
| 7 | copy left + right + output | copy `A`, copy `B`, GEMM to workspace, copy to `C` |

Each selected copy owns one distinct exact-size, type-width-aligned native workspace declaration.
Input workspaces contain [canonical](../../../../glossary.md#term-canonical-form) [row-major](../../../../glossary.md#term-row-major) logical values. The optional output workspace is a
canonical row-major `[m,n]` provider destination and is the source of one generated affine copy
whose destination [binding](../../../../glossary.md#term-binding) is the original logical `C` binding.

All input copies execute in left-then-right order, followed by exactly one [SGEMM](../../../../glossary.md#general-matrix-multiplication--gemm)/[DGEMM](../../../../glossary.md#general-matrix-multiplication--gemm) call, then
the optional output copy. Every generated copy executes exactly once per [run](../../../../glossary.md#term-run). Finalization loads
or generates only selected affine-copy [artifacts](../../../../glossary.md#term-artifact). Preparation declares no workspace for an
unselected or rejected candidate.

### Cost and deterministic selection

Let `O = m * n`, `W = m * n * k`, `IA = m * k` when left is copied, `IB = k * n` when right is
copied, `OC = m * n` when output is copied, `S` be the number of selected workspaces, `B` their
total bytes, and `R` the positive expected-[run](../../../../glossary.md#term-run) count. Extend the existing [immutable](../../../../glossary.md#term-immutable) route facts
with complete non-negative representation terms and compare checked costs:

```text
portable = R * (portableFixed + portablePerOutput * O + portablePerMac * W)

openblas = R * (openblasFixed + openblasPerOutput * O + openblasPerMac * W
                + workspaceAllocationAndBindingFixed * S
                + workspaceCostPerByte * B
                + copyInFixed * inputCopyCount
                + copyInPerElement * (IA + IB)
                + copyOutFixed * outputCopyCount
                + copyOutPerElement * OC)
```

`openblasFixed` continues to include the native transition/provider call and current thread
configuration cost, but it must not double-count a representation term. Every physical workspace
and every copy is per-run, so the entire parenthesized cost is multiplied by `R`; this task adds
no persistent copied representation amortized across runs. The stable prepared declarations and
copy [artifacts](../../../../glossary.md#term-artifact) are reusable recipes, not reusable copied data.

The candidate must be strictly cheaper than portable and meet the configured absolute and
relative basis-point thresholds. [Overflow](../../../../glossary.md#term-overflow), a missing term, inconsistent expected-run fact, zero
relative denominator when a nonzero threshold applies, equality, or insufficient benefit rejects
the candidate. Candidate comparison uses, in order: lower complete [OpenBLAS](../../../../glossary.md#term-openblas) cost, fewer total
copies, fewer workspace bytes, then the stable [mask](../../../../glossary.md#term-boolean-mask) [rank](../../../../glossary.md#term-rank) above. No fixed OpenBLAS priority,
measurement, benchmark [lookup](../../../../glossary.md#term-lookup), or tuning-[cache](../../../../glossary.md#term-cache) access is permitted.

### [Aliasing](../../../../glossary.md#term-aliasing), failure, [lifecycle](../../../../glossary.md#lifecycle), and [concurrency](../../../../glossary.md#term-concurrency)

- `A` and `B` may overlap each other, including when one or both are copied.
- The logical output [buffer](../../../../glossary.md#term-buffer) `C` must not overlap either logical input buffer, even when the input
  is copied before GEMM. This conservative rule preserves the current functional boundary and
  avoids representation-dependent alias semantics.<span data-glossary-navigation="true"> ([GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Every selected workspace must be distinct and pairwise disjoint from all selected buffers and
  other workspaces. The output workspace is never reused as an input workspace.
- Cold binding validates provider open/thread state, exact carrier/type/layout roles, segment
  accessibility, alignment, spans, output writability, generated artifact compatibility, and all
  overlap rules before any copy or GEMM writes.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [carrier glossary](../../../../glossary.md#term-storage-carrier); [layout glossary](../../../../glossary.md#layout); [alignment glossary](../../../../glossary.md#term-alignment); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- A selected plan never falls back. Input-copy, provider, or output-copy failure propagates. A
  provider failure may have partially written a direct `C`; with an output workspace, `C` is not
  touched until GEMM succeeds, although an output-copy failure need not be transactional.
- The prepared recipe, route plan, copy plans, artifacts, selections, and workspace-slot
  identities are immutable and reusable. Each run has a distinct <span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [run glossary](../../../../glossary.md#term-run))</span>[`RunState`](../../../../glossary.md#run-state--runstate) and distinct physical
  workspaces. Concurrent runs may reuse the same prepared recipe only while the borrowed provider
  remains open and the existing CPU 0010 external single-thread-state exclusion contract is
  satisfied; this task adds no stronger cross-run or cross-owner coordination guarantee.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [CPU glossary](../../../../glossary.md#term-cpu); [owner glossary](../../../../glossary.md#term-owner))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.ir`
- `io.github.pho001.synaptik.backend.cpu.internal.lowering`
- `io.github.pho001.synaptik.backend.cpu.internal.prepare`
- `io.github.pho001.synaptik.backend.cpu.internal.executable`
- `io.github.pho001.synaptik.backend.cpu.internal.memory`
- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas`

Packages added or changed:

- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas` — extends only the
  existing route's representation facts and prepared sequencing.
- Existing lowering, preparation, or executable packages may receive the minimum reusable seam
  needed to build and bind existing affine-copy artifacts; no new package is added.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [artifacts glossary](../../../../glossary.md#term-artifact))</span>

Type placement:

- `CpuOpenBlasRoutePlan` — owns the exact selected copy mask, ordered input-copy plans, optional
  output-copy plan, workspace declarations, complete costs, and provider geometry.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask))</span>
- `CpuOpenBlasOutputCopyPlan` — proposed route-local immutable workspace-to-logical-output affine
  copy contract because the general <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>`CpuMaterializationPlan` specifically owns external reads and
  consumer workspaces. If implementation finds a smaller equally typed route-local name, update
  this task before creating it.
- `CpuOpenBlasRouteSelector` — continues to own exact eligibility, bounded enumeration, checked
  whole-plan costing, and deterministic selection.
- `CpuOpenBlasPreparedExecutable` — owns the immutable copy-in/GEMM/copy-out recipe or the native
  GEMM child within one narrowly typed OpenBLAS sequence.<span data-glossary-navigation="true"> ([GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- `CpuRepresentationPlanner` — may expose focused constructors for existing input-copy and
  output-copy affine plans; it must not select the OpenBLAS route or generalize ordinary 0008E
  policy.

## Affected files

Expected [CPU](../../../../glossary.md#term-cpu) production paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionAnalysisInputs.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuRepresentationPlanner.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuMaterializationPlan.java`, only if an [invariant](../../../../glossary.md#term-invariant)/[Javadoc](../../../../glossary.md#term-javadoc) clarification is required without changing its external-read [ownership](../../../../glossary.md#term-ownership-meanings)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedPartitionExecutable.java`, only if a narrow reusable before/after-copy [binding](../../../../glossary.md#term-binding) seam is smaller than route-local sequencing
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasRoutePlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasRouteSelector.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/package-info.java`
- new `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasOutputCopyPlan.java`, only if the output plan is not a nested [immutable](../../../../glossary.md#term-immutable) value in an existing route type

Expected CPU test/[checkpoint](../../../../glossary.md#term-checkpoint) paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`, only if a top-level production type is added
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuRepresentationPlannerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedPartitionExecutableTest.java`, only if its production [owner](../../../../glossary.md#term-owner) changes
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasRouteSelectorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasNativeCheckpoint.java`

[Backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) path:

- `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/CpuOpenBlasRouteConformanceTest.java`

Documentation and [planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`, only if the current OpenBLAS route, CPU native peer route, materialization,
  or MATMUL entries cannot accurately express the finalized distinction<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [CPU native peer route glossary](../../../../glossary.md#cpu-native-peer-route); [materialization glossary](../../../../glossary.md#materialization); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Review-only paths include `CpuMatmulLowering`, `CpuAccessPlan`, `CpuAffineCopyIr`, the affine-copy
emitter/generator/tests, [buffer](../../../../glossary.md#term-buffer)/workspace implementations, discovery source/tests, provider
source/tests/plans, public APIs, architecture/ADRs, [Gradle](../../../../glossary.md#term-gradle) files, other backend-conformance
sources, integration tests, and other [modules](../../../../glossary.md#term-module). They remain byte-for-byte unchanged unless the
task is returned to planning first.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 25 paths:

- at most 11 CPU production paths, including at most one new route-local output-copy type and at
  most one of the two conditional existing paths above;
- at most eight [CPU](../../../../glossary.md#term-cpu) test/[checkpoint](../../../../glossary.md#term-checkpoint) paths;
- the one existing [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) test path;
- at most two explanatory documentation paths; and
- the three [planning](../../../../glossary.md#term-planning) paths.

The final implementation should be smaller when conditional files are unnecessary. A need for an
emitter/generator/[schema](../../../../glossary.md#term-schema) path, provider path, [Gradle](../../../../glossary.md#term-gradle) file, shared [module](../../../../glossary.md#term-module), architecture/conformance
test beyond the listed existing path, [integration test](../../../../glossary.md#term-integration-test), more than one new production type, or more
than 25 total paths is a planning stop rather than implied scope.

## Acceptance criteria

1. Exact filtering retains the CPU 0010 semantic/type/provider/thread boundary and admits only the
   resolved positive rank-two access geometry defined here. Empty, zero-K, batched, broadcast-
   batched, promoted, epilogue, negative-stride, unresolved, partial-range, overflow, and
   unproved-span cases remain portable with no native-only resource.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [epilogue glossary](../../../../glossary.md#term-epilogue); [stride glossary](../../../../glossary.md#term-stride); [overflow glossary](../../../../glossary.md#term-overflow))</span>
2. Direct boundaries require exact canonical zero-offset row-major native/alignment facts. Every
   admitted transpose or other affine view is represented only through an existing generated
   affine copy; no provider transpose flag or semantic transpose inference exists.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [row-major glossary](../../../../glossary.md#term-row-major); [alignment glossary](../../../../glossary.md#term-alignment); [transpose glossary](../../../../glossary.md#term-transpose); [view glossary](../../../../glossary.md#term-view-meanings); [inference glossary](../../../../glossary.md#term-inference))</span>
3. The selector constructs only the eight bounded masks, with zero through two input copies and
   zero or one output copy. Each mask is internally complete, declares exactly one distinct
   workspace per copy, and fits the combined byte ceiling. Tests prove every mask and every
   missing-required-copy rejection.<span data-glossary-navigation="true"> ([masks glossary](../../../../glossary.md#term-boolean-mask))</span>
4. Input materializations reuse `CpuMaterializationPlan` and existing affine-copy identity. Output
   copy-out has an honest typed route-local direction/ownership contract and does not pretend to
   be an external-read consumer materialization or alter ordinary 0008E selection.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [materialization glossary](../../../../glossary.md#materialization))</span>
5. The exact cost equation, expected runs, thresholds, and tie-break order are implemented with
   checked arithmetic. Table-driven tests cover every term independently, all masks, byte and
   copied-element totals, threshold boundaries, ties, missing terms, overflow, and deterministic
   fail-closed selection.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
6. Analysis retains the complete portable plan, selects one immutable OpenBLAS plan at most, and
   declares all selected workspace requirements before assignment. Constructor invariants reject
   copy-mask, binding, resource-ID, byte/alignment, cost, and route disagreements. Finalization
   adds no resource and cannot change selection.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [binding glossary](../../../../glossary.md#term-binding))</span>
7. Finalization loads only selected input/output affine-copy artifacts and builds one immutable
   copy-in/GEMM/copy-out recipe. No OpenBLAS representation change alters the portable MATMUL
   generator, generated schema, or ordinary representation candidates.<span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [schema glossary](../../../../glossary.md#term-schema))</span>
8. Cold binding validates every resource and overlap before the first write. It permits A/B
   overlap, rejects C overlap with either input regardless of copy mask, and rejects every buffer-
   workspace or workspace-workspace overlap. Tests cover heap/native/mixed carriers, offsets,
   transposed/strided/zero-stride input reads, noncanonical injective output writes, alignment,
   read-only output, closed resources, and provider thread drift.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [heap glossary](../../../../glossary.md#term-heap); [injective glossary](../../../../glossary.md#term-injective))</span>
9. Hot execution performs selected input copies left then right, one typed provider call with the
   exact existing dimensions/scalars, and optional output copy. Native-free fakes prove order,
   call count, segment identity, represented results, repeated-run recipe reuse, isolated
   concurrent-run workspaces, and failure propagation with no route lookup or late fallback.<span data-glossary-navigation="true"> ([dimensions glossary](../../../../glossary.md#dimension); [scalars glossary](../../../../glossary.md#term-scalar); [run glossary](../../../../glossary.md#term-run); [lookup glossary](../../../../glossary.md#term-lookup))</span>
10. The existing backend-conformance route test adds at least one FLOAT32 and one FLOAT64 expanded
    representation case through staged analysis, assignment, finalization, cold binding, and
    execution. Together they cover both input copies and output copy without native lookup,
    provider dependency, Engine, environment activation, or skips; the original direct-route
    coverage remains.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [dependency glossary](../../../../glossary.md#term-dependency); [Engine glossary](../../../../glossary.md#term-engine); [activation glossary](../../../../glossary.md#term-activation))</span>
11. The explicit native checkpoint uses the caller-supplied exact OpenBLAS 0.3.34 binary and
    exercises FLOAT32/FLOAT64 direct, left-transpose, right-affine, both-input-copy, output-copy,
    and both-input-plus-output-copy cases against the existing higher-precision numerical oracle.
    It remains an explicit non-JUnit gate, restores thread count in <span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [precision glossary](../../../../glossary.md#term-precision); [numerical oracle glossary](../../../../glossary.md#adr-ci-and-qualification); [JUnit glossary](../../../../glossary.md#term-junit))</span>`finally`, and never silently
    skips, discovers a substitute binary, or makes a performance claim.
12. `CpuCapabilityProvider`, `BackendId("cpu")`, capability reporting, 0010A discovery/loading,
    provider public shape/source/tests, public/shared APIs, dependencies, Gradle, generated schema,
    and other modules remain unchanged. Package inventory changes only if one exact new route-
    local type is created.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>
13. A distinct clean documentation-focused agent reviews the final diff and implementation
    evidence, finalizes affected Javadoc/package documentation under the General and API/Javadoc
    profiles, updates the CPU guide under the Backend Guide profile, resolves glossary impact,
    and synchronizes task/master/roadmap under the Planning profile. It records reasoned no-change
    conclusions for public Tensor/Compile/Training/Config/Engine APIs, Model semantics, provider,
    shared Prepare/Runtime, architecture/ADRs, Gradle/dependencies, generated code/schema,
    discovery, other backend-conformance sources, integration, and other modules.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Planning glossary](../../../../glossary.md#term-planning); [Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Model glossary](../../../../glossary.md#term-model); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
14. The required Java, native, Javadoc, generated-page, Markdown, exact-scope/status/order, and
    whitespace validation passes before this task becomes <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [validation glossary](../../../../glossary.md#term-validation))</span>`Complete`.

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, [run](../../../../glossary.md#term-run) focused tests as needed. After executable Java stabilizes, run one
final affected-project validation:

```bash
./gradlew :backends:cpu:test :testing:backend-conformance:test
```

Ordinary tests must require no installed [OpenBLAS](../../../../glossary.md#term-openblas) or native-access permission. They must cover the
selector/config/plan invariants, exact geometry, all eight [masks](../../../../glossary.md#term-boolean-mask), resource declarations and
assignment, finalization, sequencing, [binding](../../../../glossary.md#term-binding)/failure-before-write, fake provider invocation,
portable fallback, and the [conformance](../../../../glossary.md#term-conformance) cases above.

[Compile](../../../../glossary.md#compile) and run the expanded explicit [checkpoint](../../../../glossary.md#term-checkpoint) with the caller-supplied exact compatible
OpenBLAS 0.3.34 [binary](../../../../glossary.md#term-binary-descriptor-and-metadata):

```bash
./gradlew :backends:cpu:testClasses
java --enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector \
  -cp backends/cpu/build/classes/java/test:backends/cpu/build/classes/java/main:backends/openblas-provider/build/classes/java/main:modules/model/build/classes/java/main:modules/config/build/classes/java/main:modules/planning/build/classes/java/main:modules/runtime/build/classes/java/main:modules/prepare/build/classes/java/main:modules/backend-contract/build/classes/java/main:modules/trace/build/classes/java/main \
  io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasNativeCheckpoint \
  <ABSOLUTE_OPENBLAS_0_3_34_LIBRARY>
```

The implementation context then runs one repository-wide gate because behavior and tests span the
[CPU](../../../../glossary.md#term-cpu) and [backend](../../../../glossary.md#backend)-conformance projects:

```bash
./gradlew test
```

The repository-wide result may serve as the final affected-project evidence if it is run after
all executable/test changes; do not repeat the same stable suites merely to duplicate evidence.
The 0010A-0010E capability checkpoint and [CI](../../../../glossary.md#term-continuous-integration-ci) remain later independent gates.

No Class-File [decompilation](../../../../glossary.md#term-decompilation), generated structural inspection, generated/direct performance gate,
or benchmark is required because this task reuses the existing affine-copy generator and changes
no generated-code implementation or [schema](../../../../glossary.md#term-schema). If generated code must change, stop and replan; any
later authorized generated implementation must use optimal clean Java as its structural and
performance [oracle](../../../../glossary.md#term-test-oracle) and add proportionate semantic, Class-File, forbidden-call/allocation, and
performance evidence.

The documentation-focused context receives and reuses the successful Java/native evidence unless
it changes executable behavior. After final [Javadoc](../../../../glossary.md#term-javadoc)/documentation edits it runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git status --short -uall
```

It renders and inspects every changed generated Javadoc page; checks local Markdown links and
anchors, heading order/uniqueness, balanced fences, LF/final newlines, terminology, examples, and
trailing whitespace; validates the exact 25-path ceiling and package inventory; confirms 0010B is
`Complete` only after all evidence, 0010C-0010E remain `Draft`, 0011 remains `Blocked`, and no
detailed 0010C/0010D/0010E file exists. It must not rerun successful Java suites unless it changes
executable behavior or records a concrete stale-evidence risk.

## Dependencies

- Complete [CPU 0010](0010-narrow-openblas-blas-compatible-native-route.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>, which supplies the
  qualified one-call route, provider-free invocation seam, checked whole-plan selection, staged
  finalization, native-free conformance case, and native checkpoint.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Complete [CPU 0010A](0010a-automatic-openblas-discovery-and-internal-composition-foundation.md),
  which supplies bounded discovery and explicit provider-session ownership without changing
  qualification.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [qualification glossary](../../../../glossary.md#term-qualification))</span>
- Complete [CPU 0008E](0008e-bounded-multi-input-materialization-and-representation-reuse.md),
  plus 0008E1 and 0008F, which supply bounded input-copy facts, shared partition-DAG adoption,
  and complete portable MATMUL/access geometry.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- Complete OpenBLAS provider tasks 0001-0003 and the current compatible OpenBLAS 0.3.34 checkpoint
  environment.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- Current shared Prepare workspace assignment and Runtime per-run ownership contracts, which are
  sufficient and remain unchanged.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [run glossary](../../../../glossary.md#term-run))</span>

All dependencies are complete and the current source demonstrates no shared-contract or
architecture gap.

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0010C remains `Draft` and owns coordinated provider thread-count candidates, explicit
  composition-owned state exclusion/restoration, and the shared CPU concurrency budget.<span data-glossary-navigation="true"> ([CPU concurrency budget glossary](../../../../glossary.md#cpu-concurrency-budget--cpuconcurrencybudget))</span>
- CPU 0010D remains `Draft` and owns installed-binary qualification and stable target/binary
  fingerprinting. This task continues to require explicit qualification.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [qualification glossary](../../../../glossary.md#term-qualification); [target glossary](../../../../glossary.md#term-target))</span>
- CPU 0010E remains `Draft` and owns typed OpenBLAS tuning candidates, compatibility signatures,
  and consumption of explicit compatible selection evidence. It must include this task's exact
  copy mask and cost-schema meaning rather than redefining it.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [signatures glossary](../../../../glossary.md#term-signature); [mask glossary](../../../../glossary.md#term-boolean-mask); [schema glossary](../../../../glossary.md#term-schema))</span>
- CPU 0011 remains `Blocked` on the ordered OpenBLAS sequence plus its independent Intel use-case
  and supported oneMKL ABI evidence.<span data-glossary-navigation="true"> ([oneMKL glossary](../../../../glossary.md#term-onemkl); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- Batch, broadcast, repeated GEMM, epilogue, packing, and persistent packed-weight work has no
  authorized detailed task and is not an implied follow-up.<span data-glossary-navigation="true"> ([Batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [epilogue glossary](../../../../glossary.md#term-epilogue); [weight glossary](../../../../glossary.md#term-weight))</span>

## Architecture impact

Expected impact: None. This is a concrete-[CPU](../../../../glossary.md#term-cpu) representation and prepared-recipe expansion under
the existing staged [backend](../../../../glossary.md#backend) boundary. It retains one CPU [backend identity](../../../../glossary.md#backend-identity--backendid), the current provider
[leaf](../../../../glossary.md#term-leaf) and [dependency direction](../../../../glossary.md#java-platform-and-build-tools), shared [Prepare](../../../../glossary.md#prepare) workspace declarations/assignment, [Runtime](../../../../glossary.md#term-runtime) per-[run](../../../../glossary.md#term-run)
resource [ownership](../../../../glossary.md#term-ownership), and [immutable](../../../../glossary.md#term-immutable) [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution).

If implementation requires a provider or shared/public [API](../../../../glossary.md#term-application-programming-interface-api) change, architecture update, [module](../../../../glossary.md#term-module) or
[Gradle](../../../../glossary.md#term-gradle) edge, general [materialization](../../../../glossary.md#materialization) redesign, Runtime route logic, or another file category
outside the exact ceiling, stop and report the conflicting rule and needed decision.

## Implementation prompt

Use this prompt in a separate agentic task/thread with a clean context:

```text
You are the clean implementation agent for Synaptik CPU task 0010B. Work in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD, commit, or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the CPU master plan, and
docs/planning/backends/cpu/tasks/0010b-bounded-openblas-matmul-representation-expansion.md in full,
plus every architecture, prerequisite task, source, test, and documentation contract directly
referenced by that task. Implement it exactly within its 25-path ceiling. Stop for any
architecture, provider, shared-contract, generated-code, public-API, or scope conflict instead of
inventing a boundary.

After executable code and Java/native validation stabilize, hand the exact diff and recorded
evidence to a distinct clean documentation-focused agent in the same overall change. That pass
must follow docs/developer-guide/documentation-rules.md, independently inspect the implementation
and tests, finalize affected Javadocs/package documentation, CPU guide, glossary impact, and
planning evidence, and avoid repeating successful Java suites unless executable behavior changes
or a concrete stale-evidence risk is recorded. Do not mark 0010B Complete until that pass and all
specified validation succeed.
```

## Local decisions

- Keep transpose handling in CPU representation planning. A transpose is a resolved affine
  access pattern copied into canonical order, not a provider flag or new MATMUL semantic.<span data-glossary-navigation="true"> ([transpose glossary](../../../../glossary.md#term-transpose); [CPU glossary](../../../../glossary.md#term-cpu); [planning glossary](../../../../glossary.md#term-planning); [affine access glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [canonical glossary](../../../../glossary.md#term-canonical-form); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- Permit current proved zero-stride input reads as affine copies because they repeat immutable
  represented values and the existing address plan handles them exactly. Continue to require an
  injective output.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [immutable glossary](../../../../glossary.md#term-immutable); [injective glossary](../../../../glossary.md#term-injective))</span>
- Use one workspace per copied boundary. Do not alias or recycle the output workspace with an
  input workspace; the bounded extra allocation makes ownership and overlap validation explicit.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [validation glossary](../../../../glossary.md#term-validation))</span>
- Preserve the conservative C-versus-A/B non-overlap rule even if an input is copied first. Alias
  legality therefore does not change with the selected representation.
- Keep general external-read materialization and route-local output copy as separate typed
  contracts. This avoids weakening <span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>`CpuMaterializationPlan` consumer meaning or ordinary 0008E
  invariants.
- Price every run-owned workspace and copy on every expected run. This task introduces no
  persistent materialized data or amortized packing.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- Require one real OpenBLAS 0.3.34 checkpoint because the expanded path changes CPU-side data
  representation surrounding the same provider call. The checkpoint is semantic evidence, not
  qualification automation or a performance claim.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [qualification glossary](../../../../glossary.md#term-qualification))</span>

## Known limitations

- Only positive rank-two bare same-type FLOAT32/FLOAT64 MATMUL is expanded. Empty/zero-K,
  promoted, batched, broadcast, and fused forms remain portable.<span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [fused glossary](../../../../glossary.md#specialization-and-fusion))</span>
- Every non-direct boundary is fully copied on every run. There is no packing, partial copy,
  persistent weight representation, workspace reuse within a run, or cross-run copied-data reuse.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [weight glossary](../../../../glossary.md#term-weight))</span>
- The existing affine-copy implementation eagerly retains address pairs and therefore bounds a
  copied matrix to its current representable element-count limit even when the provider dimensions
  themselves fit 32-bit <span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [element glossary](../../../../glossary.md#term-element); [dimensions glossary](../../../../glossary.md#dimension))</span>`blasint`.
- Provider qualification, single-thread installation/exclusion, and lifetime remain explicit
  composition responsibilities. Discovery success alone is insufficient.<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- The task supplies no measured profitability calibration. Selection uses explicit dimensionless
  complete-cost facts; CPU 0010E later owns compatible measured candidate evidence.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation context `01a08fee-b64a-7dc1-ade2-2500c16b1f3e` completed the executable work on
baseline `63aed444`. Its final affected-project command
`./gradlew :backends:cpu:test :testing:backend-conformance:test` passed: [CPU](../../../../glossary.md#term-cpu) reported 961 tests,
zero failures/errors, and 28 pre-existing skips; [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance) reported 7 tests with zero
failures, errors, or skips. `./gradlew :backends:cpu:testClasses` also passed. The explicit native
[checkpoint](../../../../glossary.md#term-checkpoint) used `/opt/homebrew/opt/openblas/lib/libopenblas.dylib`, whose stable link resolved to
[OpenBLAS](../../../../glossary.md#term-openblas) 0.3.34. [FLOAT32](../../../../glossary.md#term-float32) and [FLOAT64](../../../../glossary.md#term-float64) direct, left-[transpose](../../../../glossary.md#term-transpose), right-gapped-affine, both-input-copy,
output-copy, and both-input-plus-output-copy cases passed, and the process restored thread count
16. The final `./gradlew test` repository gate passed with all 63 actionable tasks current. The
implementation context also passed exact executable-[scope](../../../../glossary.md#term-scope) and `git diff --check` gates.

Clean documentation context `01a09003-ea2c-7ec3-8a06-fc93954f1b90` independently inspected the
architecture, [planning](../../../../glossary.md#term-planning) contracts, complete implementation diff, changed Java sources/tests,
directly related CPU/provider contracts, and the recorded reports rather than assuming the
[handoff](../../../../glossary.md#term-handoff) summary was sufficient. It made no executable behavior change and therefore reused the
fresh Java/native/root evidence above. After finalizing [Javadoc](../../../../glossary.md#term-javadoc) and documentation it passed:

- `./gradlew :backends:cpu:javadoc`; the build succeeded and emitted no warning for a 0010B
  surface (94 pre-existing warnings remain in unchanged legacy constructor documentation);
- rendered inspection of every changed generated Javadoc type and package page;
- local Markdown link/anchor, heading order/uniqueness, balanced-fence, example, terminology,
  LF/final-newline, and trailing-whitespace checks;
- exact changed-path and allowlist checks with 19 paths, below the 25-path ceiling;
- CPU internal package inventory, task-status/order, absent-future-file, `git diff --check`, and
  `git status --short -uall` checks.

The new production type is exactly
`io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasOutputCopyPlan`;
no package was added. The glossary required one update because its CPU native peer entry still
described the superseded one-input-only boundary. No new public term was introduced.

The final 19 changed paths are exactly:

```text
backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuRepresentationPlanner.java
backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionAnalysisInputs.java
backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java
backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java
backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java
backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasOutputCopyPlan.java
backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasPreparedExecutable.java
backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasRoutePlan.java
backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasRouteSelector.java
backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/package-info.java
backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java
backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasNativeCheckpoint.java
backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasRouteSelectorTest.java
docs/backend-guide/cpu-backend.md
docs/glossary.md
docs/planning/backends/cpu/master-plan.md
docs/planning/backends/cpu/tasks/0010b-bounded-openblas-matmul-representation-expansion.md
docs/planning/roadmap.md
testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/CpuOpenBlasRouteConformanceTest.java
```

## Implementation notes

`CpuOpenBlasRoutePlan.Representation` is the closed eight-[mask](../../../../glossary.md#term-boolean-mask) [identity](../../../../glossary.md#term-identity-meanings). Input copies remain
ordered `CpuMaterializationPlan` values; the new route-local `CpuOpenBlasOutputCopyPlan` honestly
records [canonical](../../../../glossary.md#term-canonical-form)-workspace-to-logical-output direction. Selected copies receive distinct
analysis-local workspace IDs 8 through 10 in left, right, output order. All declarations enter
the analysis before assignment. `CpuOpenBlasPreparedExecutable` owns the final left-copy,
right-copy, one-[GEMM](../../../../glossary.md#general-matrix-multiplication--gemm), output-copy sequence and validates the complete bound resource [graph](../../../../glossary.md#graph) before
returning the invocation.

`OpenBlasRouteConfig` now carries six complete representation coefficients separately from the
portable and [OpenBLAS](../../../../glossary.md#term-openblas) GEMM terms. Selection records expected runs, total workspace bytes, input
and output copied [elements](../../../../glossary.md#term-element), both checked totals, absolute benefit, and relative basis points.
Incomplete or overflowing facts [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation). No implementation decision differed from the
planned route-local output-copy design.

Review concluded that no change is required to public [Tensor](../../../../glossary.md#tensor), [Compile](../../../../glossary.md#compile), Training, Config, or
[Engine](../../../../glossary.md#term-engine) APIs; [Model](../../../../glossary.md#term-model) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) semantics or capability reporting; the OpenBLAS provider source, tests,
[API](../../../../glossary.md#term-application-programming-interface-api), [ABI](../../../../glossary.md#term-application-binary-interface-abi), or documentation; shared [Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime) contracts; `ARCHITECTURE.md`, explanatory
architecture documents, or ADRs; [Gradle](../../../../glossary.md#term-gradle)/dependencies; affine-copy emitter, generated [schema](../../../../glossary.md#term-schema), or
other generated families; [CPU](../../../../glossary.md#term-cpu) 0010A discovery; other [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) sources; integration
tests; or other [modules](../../../../glossary.md#term-module). The existing CPU single-thread external coordination, provider [lifetime](../../../../glossary.md#term-lifetime),
and [qualification](../../../../glossary.md#term-qualification) boundaries remain exact. No performance or tuning claim is made.

## Completion summary

Completed the bounded [OpenBLAS](../../../../glossary.md#term-openblas) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) representation expansion across 19 authorized paths. Nine
[CPU](../../../../glossary.md#term-cpu) production paths implement all eight [masks](../../../../glossary.md#term-boolean-mask), honest copy-out [ownership](../../../../glossary.md#term-ownership-meanings), exact resource/cost
facts, declaration-before-assignment, finalization, and pre-write validated execution. Three CPU
test/[checkpoint](../../../../glossary.md#term-checkpoint) paths and one [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) path cover selection, costs, [binding](../../../../glossary.md#term-binding),
sequencing, failures, reuse/[concurrency](../../../../glossary.md#term-concurrency), and real OpenBLAS 0.3.34 execution. The clean
documentation pass finalized affected [Javadoc](../../../../glossary.md#term-javadoc)/package documentation, the CPU guide, the necessary
glossary correction, and synchronized [planning](../../../../glossary.md#term-planning) evidence.

All required executable, native, repository, Javadoc, rendered-page, Markdown, package, [scope](../../../../glossary.md#term-scope),
status/order, future-file, and whitespace [validation](../../../../glossary.md#term-validation) passed. There are no unresolved 0010B issues
and no required corrective follow-up. CPU 0010C is the next `Draft` planning frontier; 0010D and
0010E remain `Draft`, and CPU 0011 remains `Blocked`.

Status: Complete
