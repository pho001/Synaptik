# Task 0005A: [Atomic](../../../../glossary.md#resource-transaction-and-rollback) [Partition](../../../../glossary.md#partition)-[Kernel](../../../../glossary.md#kernel) Architecture Reset

## Reader context

Atomic here meant a whole implementation cutover, not a Runtime resource transaction. The reset
changed what CPU generated, not what the application’s graph meant. CPU analysis consumed a
whole planned partition, grouped eligible work into a computation unit, and lowered that unit to
CPU intermediate representation (IR). In the ADD–GELU–MUL example, the two intermediate values
still existed semantically, but the generated loop kept them as local values instead of
allocating separate buffers or making separate node calls. Only external reads and the final
stored result crossed the unit’s memory boundary.

“Leaf” in the package discussion means an endpoint package such as `route.portable`, not an
application input Tensor. The scalar reference was a checking implementation, not Runtime’s
graph interpreter. Read the original eight-node/unit and topology limits as this checkpoint’s
bounds; the current lowering explanation describes the later partition decomposition.

Current reading alongside this historical record:

- [CPU partition and unit lowering](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternallowering)
  — explains virtual intermediates, boundary stores, and family-specific mappings.
- [CPU typed intermediate representation](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalir)
  — distinguishes code-shaping structure from concrete geometry and live values.

## Status

Complete

## Goal

Replace the provisional one-[graph](../../../../glossary.md#graph)-[node](../../../../glossary.md#node)-to-one-[lowering](../../../../glossary.md#lowering)/emitter/invocation/[kernel](../../../../glossary.md#kernel) [CPU](../../../../glossary.md#term-cpu) pipeline in
one [atomic](../../../../glossary.md#resource-transaction-and-rollback) change before further [operation](../../../../glossary.md#operation)-family expansion. CPU analysis lowers one complete
CPU-owned `PlannedPartition`, forms computation-oriented execution units, performs legal [fusion](../../../../glossary.md#term-fusion)
before declaring resources, builds one route-independent [canonical](../../../../glossary.md#term-canonical-form) `CpuKernelIr` per unit, selects
the final route and execution strategy, and finalizes one [partition](../../../../glossary.md#partition)-level [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable).

The portable CPU route is the [bytecode](../../../../glossary.md#term-bytecode)-first Java 26 [Class-File API](../../../../glossary.md#term-class-file-api) plus [Vector API](../../../../glossary.md#term-vector-api) baseline and
the always-available semantic fallback for every [occurrence](../../../../glossary.md#term-occurrence) it supports. CPU 0005A realizes only
its [scalar](../../../../glossary.md#term-scalar) strategy; later portable tasks add the Vector API strategies without changing the route
boundary. A [metadata](../../../../glossary.md#term-metadata)-only unit may emit no class. [OpenBLAS](../../../../glossary.md#term-openblas) is a narrow [BLAS](../../../../glossary.md#term-blas)-compatible native
fallback, while vendor/platform-specialized libraries are peer routes selected by exact capability
and whole-plan cost. The scalar reference realization is for [conformance](../../../../glossary.md#term-conformance) and [fail-closed](../../../../glossary.md#term-fail-closed) checking,
not an `Operation` or [IR](../../../../glossary.md#term-intermediate-representation-ir) interpreter in the [Runtime](../../../../glossary.md#term-runtime) [hot path](../../../../glossary.md#term-hot-path).

The proving [slice](../../../../glossary.md#slice) is one fully static, canonical-dense [FLOAT64](../../../../glossary.md#term-float64) partition:

```text
a ----\
       ADD -> sum -> GELU -> activated ----\
b ----/                                     MUL -> output
c -----------------------------------------/

one PlannedPartition
  -> one CPU execution unit
  -> one canonical CpuKernelIr
  -> one generated class/artifact
  -> one partition-level BoundInvocation
```

`sum` and `activated` remain graph and canonical-IR values, but receive no [buffer](../../../../glossary.md#term-buffer) declaration,
slot, representation, or invocation [argument](../../../../glossary.md#term-argument). The generated loop accepts primitive `start` and
`end` bounds. The exact same generated class bytes and loaded compatibility [identity](../../../../glossary.md#term-identity-meanings) must serve at
least two compatible [element](../../../../glossary.md#term-element) extents for the same ADD -> [GELU](../../../../glossary.md#gaussian-error-linear-unit--gelu) -> MUL [topology](../../../../glossary.md#term-topology).

This task is an atomic working-tree replacement, not deletion of Git history. CPU tasks 0001
through 0005 remain historical `Complete` records during implementation. They become
`Superseded` only in the final synchronized status edit after implementation, focused and final
[validation](../../../../glossary.md#term-validation), the mandatory independent documentation pass, and the reset [checkpoint](../../../../glossary.md#term-checkpoint) all pass.

## [Scope](../../../../glossary.md#term-scope)

- Adopt the structured [CPU](../../../../glossary.md#term-cpu)-internal package layout defined below. `CpuCapabilityProvider` remains
  the sole supported public CPU API; technically public cross-package contracts live only below
  `io.github.pho001.synaptik.backend.cpu.internal` and are documented as unsupported internal [API](../../../../glossary.md#term-application-programming-interface-api).
- Establish one route-neutral selected-plan seam plus one `internal.route.portable` realization
  contract for the proving slice. Do not add native/vendor adapters or placeholder packages.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice))</span>
- Replace every per-node portable candidate, pointwise-ADD lowerer, finalizer, executable, and
  obsolete parallel foundation with one coherent partition pipeline. Leave no old/new parallel
  pipelines, aliases, bridges, adapters, registries, service locators, or dead production types.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [pointwise glossary](../../../../glossary.md#term-pointwise); [partition glossary](../../../../glossary.md#partition); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Lower the complete CPU-owned partition, form computation-oriented units, fuse before resource
  declaration, and emit exact declarations only for materialized unit and partition boundaries.
- Prove exactly one legal ADD -> exact [GELU](../../../../glossary.md#gaussian-error-linear-unit--gelu) -> MUL [fusion](../../../../glossary.md#term-fusion), one `CpuKernelIr`, one generated class,
  one partition-level [`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation), and no physical resources for its two internal values.
- Admit only parameterless [FLOAT64](../../../../glossary.md#term-float64) `ADD`, exact `GELU`, and parameterless FLOAT64 `MUL` for the
  proving slice. Shapes are fully static, compatible, scalar or zero-element, and layouts are
  resolved canonical-dense, zero-offset native <span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout))</span><span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [element glossary](../../../../glossary.md#term-element); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>[`MemorySegment`](../../../../glossary.md#term-memorysegment) layouts. Other occurrences fail
  closed as complete occurrences before artifact access.<span data-glossary-navigation="true"> ([fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Generate universal primitive `start`/`end` loops whose class bytes do not bake a concrete
  element count or compatible extents. Bind checked element count and extents once on the cold
  path. Lock same-byte and same-loaded-compatibility-identity reuse for at least two non-zero
  compatible extents, plus scalar and zero-element behavior.<span data-glossary-navigation="true"> ([cold path glossary](../../../../glossary.md#term-cold-path); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Preserve exact GELU semantics as the [Model](../../../../glossary.md#term-model) [target](../../../../glossary.md#term-target)
  `0.5 * x * (1 + erf(x / sqrt(2)))`. Preserve operation order and prohibit reassociation,
  contraction, fast math, or substitution of <span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [reassociation glossary](../../../../glossary.md#term-reassociation); [contraction glossary](../../../../glossary.md#contraction-dimension))</span>`GELU_TANH_APPROXIMATION`. Use a CPU-private [scalar](../../../../glossary.md#term-scalar)
  [`erf`](../../../../glossary.md#term-erf) implementation shared by generated and reference realizations and retain the existing
  planned oracle/error requirements.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Keep the final 0005A execution strategy scalar and single-threaded while representing execution
  strategy as the orthogonal compute/orchestration axes defined below. Do not retain unused worker
  or vector production foundations merely for later tasks.<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [vector glossary](../../../../glossary.md#term-vector))</span>
- Make generated-class persistence an optional cold-path policy. Without a configured trusted
  artifact root, deterministically emit, verify, and define in memory. With a root, a compatible
  persistent hit may avoid emission but must still be verified and defined. Persistence is never
  correctness-critical, Runtime state, or preservation of JIT machine code or profiling.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence); [Runtime glossary](../../../../glossary.md#term-runtime); [JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit); [machine code glossary](../../../../glossary.md#term-machine-code))</span>
- Preserve deterministic structural keys, process-local compatible loaded-artifact reuse, and
  strong ownership by the prepared execution. The default persistent policy remains disabled and
  carries no performance claim until a later evidence gate. Add no migration reader.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>
- Harden shared [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) narrowly: each logical value crossing two or more planned
  partitions must be declared by its producer partition, when present, and every distinct external
  consumer partition. Do not infer CPU fusion, virtuality, or materialization in shared Prepare.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [materialization glossary](../../../../glossary.md#materialization); [Prepare glossary](../../../../glossary.md#prepare))</span>
- Independently finalize affected Javadocs, the CPU guide, glossary, task/master/roadmap records,
  and status synchronization in the mandatory clean documentation context after Java stabilizes.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Implementing general broadcast, offset/strided access, heap or mixed carriers, vector execution,
  parallel execution, CPU-internal materialization, reductions, scans, matrix multiplication,
  convolution, semantic specialized kernels, vendor routes, tuning, or benchmarks in 0005A.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [vector glossary](../../../../glossary.md#term-vector); [CPU glossary](../../../../glossary.md#term-cpu); [materialization glossary](../../../../glossary.md#materialization); [matrix multiplication glossary](../../../../glossary.md#matrix-multiplication--matmul); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [kernels glossary](../../../../glossary.md#kernel))</span>
- Creating `route.nativeblas`, `route.nativeops`, or provider-specific packages before their
  concrete Draft implementation tasks; adding an OpenBLAS, Accelerate, oneMKL, oneDNN, AOCL, or
  ZenDNN adapter in this reset.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [oneMKL glossary](../../../../glossary.md#term-onemkl); [oneDNN glossary](../../../../glossary.md#term-onednn); [AOCL glossary](../../../../glossary.md#term-aocl); [ZenDNN glossary](../../../../glossary.md#term-zendnn))</span>
- Fixed-shape or unrolled generated variants. They are permitted only in a later evidence-selected
  specialization with an explicit specialization budget.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [unrolled glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Interpreting [`Operation`](../../../../glossary.md#operation), `CompiledNode`, or `CpuKernelIr` in Runtime; class-per-operation
  generation; hot cursor objects; per-element division/modulo; hot lookup, allocation, route
  dispatch, storage classification, reflection, or registry access.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [element glossary](../../../../glossary.md#term-element); [lookup glossary](../../../../glossary.md#term-lookup); [dispatch glossary](../../../../glossary.md#term-dispatch); [registry glossary](../../../../glossary.md#term-registry))</span>
- Changing Model, Compiler, Planning, or Runtime semantics; putting gradient policy into CPU
  broadcasting; altering <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [gradient glossary](../../../../glossary.md#term-gradient); [broadcasting glossary](../../../../glossary.md#broadcasting))</span>`LogicalMemoryPlan`; or changing shared Prepare beyond the narrow
  cross-partition declaration-completeness check.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [partition glossary](../../../../glossary.md#partition))</span>
- Editing `ARCHITECTURE.md`, ADRs, architecture tests, Gradle, module dependencies, public API,
  backend-conformance, integration behavior, or another backend.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Creating a detailed specification for any later CPU task. Later work is represented only by the
  ordered Draft rows in the CPU master plan and roadmap.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Runtime, Prepare, and Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [Performance evidence and tuning](../../../../architecture/performance-evidence-and-tuning.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [CPU master plan](../master-plan.md)

## Architecture constraints

- Planning selects ownership and maximal same-owner partitions. CPU analysis owns whole-partition
  lowering, unit formation, fusion, route and execution-strategy selection, and exact resource
  declarations before shared assignment.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [owner glossary](../../../../glossary.md#term-owner); [CPU glossary](../../../../glossary.md#term-cpu); [partition glossary](../../../../glossary.md#partition); [lowering glossary](../../../../glossary.md#lowering); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- [Planning](../../../../glossary.md#term-planning) selects the single `BackendId("cpu")`; it never selects OpenBLAS or a vendor library as
  another backend. CPU Prepare compares the portable route, eligible native fallbacks, and
  eligible vendor/platform routes as peer candidates with no fixed vendor priority.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [backend glossary](../../../../glossary.md#backend); [Prepare glossary](../../../../glossary.md#prepare))</span>
- Partition lowering, fusion legality and profitability, canonical IR, access planning,
  materialization accounting, numerical/determinism checks, and representation planning are
  shared and route-independent. Provider adapters may consume the selected route plan but may not
  reinterpret the graph or duplicate those responsibilities or resource lifetime ownership.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [materialization glossary](../../../../glossary.md#materialization); [graph glossary](../../../../glossary.md#graph); [resource lifetime glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- CPU 0005A remains an exact/default ADD -> exact GELU -> MUL proving slice. Common analysis owns
  numerical eligibility and the selected numerical mode; emitters consume that decision, and no
  route may infer relaxed permission from hardware, provider availability, workload size,
  objectives, tuning, or benchmarks. The numerical mode participates in specialization/cache
  compatibility and the cold lowering manifest, with no hot-path policy lookup.<span data-glossary-navigation="true"> ([GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [slice glossary](../../../../glossary.md#slice); [specialization glossary](../../../../glossary.md#term-specialization); [cache glossary](../../../../glossary.md#term-cache); [manifest glossary](../../../../glossary.md#term-manifest); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- The same exact/default policy applies to forward and compiler-generated gradient operations.
  This task creates no gradient-specific numerical policy and no <span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient))</span>`POW` strength reduction; those
  remain concise later master-plan rows.
- `LogicalMemoryPlan` remains complete for graph values. A physical Runtime slot exists only when
  backend analysis emits a <span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>`PreparationResourceRequirement.Buffer`.
- Analysis fixes fusion, access regime, materialization boundary, route, execution strategy, and
  declarations. Finalization after assignment may reuse or emit the selected artifact but may not
  change those decisions.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
- Runtime receives one partition executable and a cold-bound invocation. It receives no graph
  semantics, operation objects, canonical IR, cache policy, or materialization decision.<span data-glossary-navigation="true"> ([bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation); [operation glossary](../../../../glossary.md#operation))</span>
- CPU consumes current Model shape and layout contracts. It does not replace symbolic dimension
  identity, singleton expansion, zero-extent rules, checked long spans, base offsets, or long
  strides with legacy CPU rules.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [dimension glossary](../../../../glossary.md#dimension); [identity glossary](../../../../glossary.md#term-identity-meanings); [singleton glossary](../../../../glossary.md#term-singleton); [extent glossary](../../../../glossary.md#term-extent); [strides glossary](../../../../glossary.md#term-stride))</span>
- Any need to alter an authoritative architecture rule, dependency direction, or module boundary
  is a stop condition rather than an implied part of this task.<span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [module glossary](../../../../glossary.md#term-module))</span>

## Permanent [CPU](../../../../glossary.md#term-cpu) [lowering](../../../../glossary.md#lowering) [model](../../../../glossary.md#term-model)

The reset establishes the boundary and data model; 0005A implements only the proving [slice](../../../../glossary.md#slice).
Eventually, CPU analysis must support all of the following through that same boundary:

- [pointwise](../../../../glossary.md#term-pointwise) and broadcast [fusion](../../../../glossary.md#term-fusion);
- folding of [view](../../../../glossary.md#term-view-meanings) and [metadata](../../../../glossary.md#term-metadata) operations when their semantics permit it;
- reduction and [scan](../../../../glossary.md#term-cumulative-scan) units, including numerically stable multi-pass units;
- [matrix](../../../../glossary.md#term-matrix-meanings)-multiplication and [convolution](../../../../glossary.md#term-channels-first-convolution-layers) epilogues; and
- explicit semantic specialized [kernels](../../../../glossary.md#kernel).

Legal fusion and profitable fusion are separate decisions. Legality preserves [node](../../../../glossary.md#node) order,
numerical semantics, [aliases](../../../../glossary.md#aliasing-and-arena) and overlap, state, random-number semantics, [publication](../../../../glossary.md#term-publication), [fan-out](../../../../glossary.md#term-fan-out),
and cross-[partition](../../../../glossary.md#partition) boundaries. Profitability may reject legal fusion because of code size,
register pressure, reuse, [materialization](../../../../glossary.md#materialization), vendor eligibility, or measured cost. A decomposed
[softmax](../../../../glossary.md#softmax--log-softmax) sequence must not silently become a stable `SOFTMAX` unit unless an explicit semantic and
numerical equivalence policy authorizes that recognition.

The reset does not create one class per Model [operation](../../../../glossary.md#operation). One generated class realizes one selected
computation unit and may contain multiple ordered operations. Metadata-only units may need no
class. The portable generated route is the baseline and supported semantic fallback. [OpenBLAS](../../../../glossary.md#term-openblas) is
only a narrow [BLAS](../../../../glossary.md#term-blas)-compatible native fallback, not a universal or preferred route. Accelerate,
[oneMKL](../../../../glossary.md#term-onemkl), [oneDNN](../../../../glossary.md#term-onednn), [AOCL](../../../../glossary.md#term-aocl), and optional [ZenDNN](../../../../glossary.md#term-zendnn) are later peer routes, not fallback [dispatch](../../../../glossary.md#term-dispatch) inside a
generated loop.

All routes consume common CPU analysis. Native provider adapters own only [ABI](../../../../glossary.md#term-application-binary-interface-abi) calls and their own
provider-specific compatibility/[lifetime](../../../../glossary.md#term-lifetime) mechanics. They do not own [graph](../../../../glossary.md#graph) interpretation,
broadcast [planning](../../../../glossary.md#term-planning), fusion, [canonical](../../../../glossary.md#term-canonical-form) [IR](../../../../glossary.md#term-intermediate-representation-ir) construction, materialization decisions,
numerical/determinism policy, representation planning, or [Runtime](../../../../glossary.md#term-runtime) [resource lifetime](../../../../glossary.md#ownership-borrowing-and-lease).

Route families remain distinct: Apple CPU may select Accelerate BLAS, vDSP, or vForce; Intel may
select oneMKL BLAS/[VML](../../../../glossary.md#term-vml) or the separate oneDNN family; [AMD](../../../../glossary.md#term-amd) may select AOCL-BLAS/AOCL-LibM or the
separate optional ZenDNN family. ARM is capability-first, not tied to one library: Apple Silicon
may use Accelerate, while other ARM [targets](../../../../glossary.md#term-target) use portable [code generation](../../../../glossary.md#term-code-generation) unless a later task adds
an explicitly verified provider.

## [Canonical](../../../../glossary.md#term-canonical-form) [IR](../../../../glossary.md#term-intermediate-representation-ir), access plans, and [identity](../../../../glossary.md#term-identity-meanings)

There is one shared route-independent `CpuKernelIr`. It records:

- typed boundary values and typed virtual values;
- ordered computation [topology](../../../../glossary.md#term-topology) and exact [operation](../../../../glossary.md#operation) semantics;
- the structural form of each normalized input access plan;
- a universal primitive `start`/`end` loop [model](../../../../glossary.md#term-model); and
- ordered output stores.

It does not record a selected route, thread count, [vector species](../../../../glossary.md#term-vector-species), trusted [artifact](../../../../glossary.md#term-artifact) root, [Runtime](../../../../glossary.md#term-runtime)
slot, segment instance or address, [graph](../../../../glossary.md#graph) identity, concrete invocation [binding](../../../../glossary.md#term-binding), generator version,
or [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) state. The default canonical identity does not include concrete compatible extents or the
[element](../../../../glossary.md#term-element) count.

The eventual normalized `CpuAccessPlan` is the single access system for [elementwise](../../../../glossary.md#term-elementwise), `WHERE`, and
[fused](../../../../glossary.md#specialization-and-fusion) operands. Each input has aligned [rank](../../../../glossary.md#term-rank), cold-bound concrete extents, base offset, effective
long [strides](../../../../glossary.md#term-stride), and zero strides on broadcast [axes](../../../../glossary.md#term-axis). It must cover [scalars](../../../../glossary.md#term-scalar), rank expansion, [singleton](../../../../glossary.md#term-singleton)
and multi-axis broadcast, zero extents, multi-input [masks](../../../../glossary.md#term-boolean-mask) and [parameters](../../../../glossary.md#term-parameter), different fused-operand
patterns, offset-[contiguous](../../../../glossary.md#term-contiguous) and positive-strided [layouts](../../../../glossary.md#layout), zero-stride [views](../../../../glossary.md#term-view-meanings), [heap](../../../../glossary.md#term-heap) arrays,
[`MemorySegment`](../../../../glossary.md#term-memorysegment), mixed carriers, and dynamic or symbolic [dimensions](../../../../glossary.md#dimension) once [Prepare](../../../../glossary.md#prepare) has bound them
exactly. The IR contains the access-plan form; the invocation binding supplies instance extents,
offsets, carriers, and addresses.

The [specialization](../../../../glossary.md#term-specialization) and artifact identity compose canonical IR with the structural access regime,
[fusion](../../../../glossary.md#term-fusion) form, data types, and separately selected route, execution strategy, route configuration,
and [CPU](../../../../glossary.md#term-cpu)/[JDK](../../../../glossary.md#term-java-development-kit-jdk)/class-file/generator compatibility fingerprints. Instance bindings remain excluded.
Fixed-[shape](../../../../glossary.md#shape) or [unrolled](../../../../glossary.md#backpropagation-through-time-and-recurrent-state) variants may add exact extents only as explicit evidence-selected
specializations, subject to a bounded specialization budget that prevents class explosion.

## [Broadcasting](../../../../glossary.md#broadcasting) and [layout](../../../../glossary.md#layout) requirements for follow-up work

Full right-aligned broadcasting must meet or exceed the selected legacy capability while using
current `ShapeBroadcast` and `LayoutDescriptor` semantics. Current [Model](../../../../glossary.md#term-model) contracts are stronger
than the legacy implementation: [`Dimension`](../../../../glossary.md#dimension) equality is symbolic/[canonical](../../../../glossary.md#term-canonical-form), [singleton](../../../../glossary.md#term-singleton) expansion
is explicit, zero extents are valid, offsets and [strides](../../../../glossary.md#term-stride) are `long`, zero-stride [views](../../../../glossary.md#term-view-meanings) are
representable, and spans use checked arithmetic. [CPU](../../../../glossary.md#term-cpu) must consume those contracts.

CPU must not copy the legacy `int` arithmetic, `Math.max` zero-dimension bug, duplicated
[elementwise](../../../../glossary.md#term-elementwise)/`WHERE`/[fusion](../../../../glossary.md#term-fusion) planners, per-[operation](../../../../glossary.md#operation) storage-method [matrices](../../../../glossary.md#term-matrix-meanings), [Runtime](../../../../glossary.md#term-runtime) fallback,
hot cursor objects, or incomplete native/[vector](../../../../glossary.md#term-vector) broadcast eligibility. Broadcast [gradients](../../../../glossary.md#term-gradient) remain
the [Compiler](../../../../glossary.md#term-compiler)/Model `SUM_TO_SHAPE` responsibility and later CPU reduction coverage; CPU access
[planning](../../../../glossary.md#term-planning) contains no gradient policy.

The planned performance tiers, from simplest to most general, are:

1. dense [linear](../../../../glossary.md#term-linear-meanings) access;
2. [scalar](../../../../glossary.md#term-scalar) or all-zero-stride broadcast;
3. last-[axis](../../../../glossary.md#term-axis) [bias](../../../../glossary.md#term-bias) broadcast;
4. block/outer broadcast with a [contiguous](../../../../glossary.md#term-contiguous) inner loop;
5. general positive-strided odometer initialized once from `start`;
6. scalar fallback;
7. cost-gated vector [gather](../../../../glossary.md#gather); and
8. optional contiguous [materialization](../../../../glossary.md#materialization).

Generated [bytecode](../../../../glossary.md#term-bytecode) performs offset and [carry](../../../../glossary.md#term-carry) arithmetic directly. It creates no hot cursor object,
makes no per-[element](../../../../glossary.md#term-element) virtual call, and uses no per-element division or modulo. General gather is
optional where scalar execution or materialization wins. Complete semantic coverage does not imply
that every layout is vectorized.

## [Materialization](../../../../glossary.md#materialization) analysis and shared assignment

Before shared assignment, [CPU](../../../../glossary.md#term-cpu) analysis eventually compares direct strided/broadcast execution with
CPU-internal [contiguous](../../../../glossary.md#term-contiguous) materialization. The comparison includes copy cost, [kernel](../../../../glossary.md#kernel) benefit,
reuse/[fan-out](../../../../glossary.md#term-fan-out), vendor eligibility, memory cost, and expected repeated runs. If materialization is
selected, CPU analysis declares the exact internal resource before assignment and lowers a copy
unit plus consumer unit without changing the [Model](../../../../glossary.md#term-model) [graph](../../../../glossary.md#graph). Shared [Prepare](../../../../glossary.md#prepare) remains CPU-blind.

For 0005A, apply these exact rules:

1. Retain every [graph value](../../../../glossary.md#graph-value) and its logical requirement in `LogicalMemoryPlan`.
2. Declare [buffers](../../../../glossary.md#term-buffer) for `a`, `b`, `c`, and `output` in deterministic encounter order.
3. Represent `sum` and `activated` as virtual `CpuKernelIr` values with no declaration or slot.
4. Reject publication, extra fan-out, cross-partition use, alias ambiguity, or other facts that
   would require either intermediate to be materialized.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
5. Use checked `long` [element](../../../../glossary.md#term-element) and byte arithmetic, including [scalar](../../../../glossary.md#term-scalar) and zero-element cases.

Shared Prepare checks only [partition](../../../../glossary.md#partition) boundaries. For a logical value produced in `P` and consumed
in distinct partitions `Q` and `R`, analyses for `P`, `Q`, and `R` must each declare the exact
[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot). A value confined to `P` may remain undeclared. Existing bindable-input, [constant](../../../../glossary.md#term-constant),
initialized-representation, [publication](../../../../glossary.md#term-publication), and geometry checks remain unchanged.

## Execution strategies

The final portable strategy vocabulary is exactly the [Cartesian product](../../../../glossary.md#term-cartesian-product) of two [axes](../../../../glossary.md#term-axis):

| Compute axis | Orchestration axis | Strategy |
|---|---|---|
| [scalar](../../../../glossary.md#term-scalar) | single-thread | scalar |
| [vector](../../../../glossary.md#term-vector) | single-thread | vector |
| scalar | parallel | parallel-scalar |
| vector | parallel | parallel-vector |

Generated [kernels](../../../../glossary.md#kernel) always accept primitive `start` and `end`. Parallel workers [dispatch](../../../../glossary.md#term-dispatch) chunks
outside the generated inner loop. Vector chunks and scalar or masked tails must be correct for
arbitrary legal bounds. [CPU](../../../../glossary.md#term-cpu) 0005A implements only scalar/single-thread; ordered Draft follow-ups
deliver vector, parallel-scalar, and parallel-vector without changing this vocabulary.

## Optional generated-class [persistence](../../../../glossary.md#term-persistence)

Deterministic structural [identity](../../../../glossary.md#term-identity-meanings) and class-byte verification are mandatory; filesystem
persistence is not. The [cold path](../../../../glossary.md#term-cold-path) has two policies:

- no trusted artifact root: deterministically emit, structurally verify, define, and process-locally
  intern compatible generated class bytes in memory;
- configured trusted artifact root: attempt compatible lookup, then verify and define a hit, or
  deterministically emit, verify, define, and optionally publish a miss.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span>

A persistent hit can avoid byte emission only. It cannot preserve [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) [machine code](../../../../glossary.md#term-machine-code), profile,
or hidden-class state. Corrupt, incompatible, absent, or unusable storage must safely fall back to
deterministic in-memory emission. Persistence is disabled and makes no default performance claim
until the ordered evidence-gate task justifies a policy change.

The reset defines one current [schema](../../../../glossary.md#term-schema) and accepts only that schema. It has no reader, converter,
[alias](../../../../glossary.md#aliasing-and-arena), or migration path for earlier entries. `CpuPreparedExecutable` strongly owns the selected
loaded [artifact](../../../../glossary.md#term-artifact) and direct handle for its [lifetime](../../../../glossary.md#term-lifetime); persistent storage and process-local interning
do not weaken that [ownership](../../../../glossary.md#term-ownership-meanings).

## Structured internal packages and visibility

The reset adopts this package structure immediately:

```text
io.github.pho001.synaptik.backend.cpu
  CpuCapabilityProvider                 sole supported public CPU API

io.github.pho001.synaptik.backend.cpu.internal
  memory                                representations and cold binding
  prepare                               analysis/finalization lifecycle
  lowering                              partition-to-unit lowering and fusion
  ir                                    canonical IR and normalized access plans
  codegen.emit                           portable Class-File generation and loop emission
  route.portable                         portable selection/realization plan
  cache                                 structural identity and optional persistence
  executable                            prepared partition execution
  reference                             conformance/fail-closed scalar reference
```

[CPU](../../../../glossary.md#term-cpu) 0005A creates only the packages above. The permanent [target](../../../../glossary.md#term-target) map reserves these later locations
without creating them now:

```text
internal.route.nativeblas
  openblas                              narrow BLAS-compatible native fallback
  accelerate                            Apple Accelerate BLAS
  mkl                                   Intel oneMKL BLAS
  aocl                                  AMD AOCL-BLAS

internal.route.nativeops
  accelerate                            Apple vDSP/vForce
  mkl                                   Intel oneMKL VML
  onednn                                Intel oneDNN partition operations
  aocl                                  AMD AOCL-LibM
  zendnn                                optional AMD ZenDNN partition operations
```

Those are route-family/provider leaves, not separate [backends](../../../../glossary.md#backend). A concrete later task may refine a
[leaf](../../../../glossary.md#term-leaf) when verified provider boundaries require it, but may not move common [lowering](../../../../glossary.md#lowering), access,
[fusion](../../../../glossary.md#term-fusion), [materialization](../../../../glossary.md#materialization), numerical, representation, or [lifecycle](../../../../glossary.md#lifecycle) [ownership](../../../../glossary.md#term-ownership-meanings) into that leaf.

Java subpackages are not friends. The exact technically public, unsupported cross-package types are
`CpuBorrowedBuffer`, `CpuBufferArgument`, `CpuBufferRepresentation`, `CpuNativeBuffer`,
`CpuPartitionAnalysisInputs`, `CpuPartitionPreparationPlan`, `CpuPartitionPreparer`,
`CpuPartitionFinalizer`, `CpuPartitionLowering`, `CpuKernelIr`, `CpuAccessPlan`,
`CpuClassFileKernelGenerator`, `CpuGeneratedKernel`, `CpuPortableRoutePlan`,
`CpuGeneratedKernelArtifactStore`,
`CpuGeneratorSchema`, `CpuKernelSpecialization`, `CpuLoweringFingerprint`,
`CpuPreparedExecutable`, and `CpuScalarReferenceKernel`. Each is required by a named adjacent
package or by the sole supported provider and remains unsupported internal [API](../../../../glossary.md#term-application-programming-interface-api) by namespace and
[Javadoc](../../../../glossary.md#term-javadoc). `CpuCarrierEmitter`, `CpuLoopEmitter`, and `CpuScalarEmitter` remain package-private to
`codegen.emit`; implementation-only nested types are no more visible than their exact consumer
requires. Add no [JPMS](../../../../glossary.md#term-java-platform-module-system-jpms) export, [service locator](../../../../glossary.md#term-service-locator), [registry](../../../../glossary.md#term-registry), broad [facade](../../../../glossary.md#term-facade), or public compatibility
bridge. Focused inventory tests enforce the sole supported public surface, this visibility list,
and the exact internal package/type inventory.

## Exact production path and type disposition

Every current production path is listed. Destination shorthands such as
`internal/memory/CpuBorrowedBuffer.java` expand beneath
`backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`. `Move/rewrite` deletes the old
path and creates only the named destination. `Remove` has no replacement [alias](../../../../glossary.md#aliasing-and-arena). Nested types
follow their enclosing type.

| Current path | Action and exact destination |
|---|---|
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java` | Rewrite in place; sole supported public entry and complete-[occurrence](../../../../glossary.md#term-occurrence) [fail-closed](../../../../glossary.md#term-fail-closed) reporting |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java` | Rewrite in place; supported surface and internal-package boundary |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuBorrowedBuffer.java` | Move/rewrite to `internal/memory/CpuBorrowedBuffer.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuBufferArgument.java` | Move/rewrite to `internal/memory/CpuBufferArgument.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuBufferRepresentation.java` | Move/rewrite to `internal/memory/CpuBufferRepresentation.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuNativeBuffer.java` | Move/rewrite to `internal/memory/CpuNativeBuffer.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuCarrierEmitter.java` | Move/rewrite to `internal/codegen/emit/CpuCarrierEmitter.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuClassFileKernelGenerator.java` | Move/rewrite to `internal/codegen/emit/CpuClassFileKernelGenerator.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuGeneratedKernel.java` | Move/rewrite to `internal/codegen/emit/CpuGeneratedKernel.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuLoopEmitter.java` | Move/rewrite to `internal/codegen/emit/CpuLoopEmitter.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuScalarEmitter.java` | Move/rewrite to `internal/codegen/emit/CpuScalarEmitter.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuGeneratedKernelArtifactStore.java` | Move/rewrite to `internal/cache/CpuGeneratedKernelArtifactStore.java`; optional trusted-root policy |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuGeneratorSchema.java` | Move/rewrite to `internal/cache/CpuGeneratorSchema.java`; one current [schema](../../../../glossary.md#term-schema), no migration reader |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuKernelSpecialization.java` | Move/rewrite to `internal/cache/CpuKernelSpecialization.java`; structural [specialization](../../../../glossary.md#term-specialization) without instance extents by default |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuLoweringFingerprint.java` | Move/rewrite to `internal/cache/CpuLoweringFingerprint.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPreparedExecutable.java` | Move/rewrite to `internal/executable/CpuPreparedExecutable.java`; [partition](../../../../glossary.md#partition)-level executable and strong [artifact](../../../../glossary.md#term-artifact) [owner](../../../../glossary.md#term-owner) |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuFamilyKernelEmitter.java` | Remove; [canonical](../../../../glossary.md#term-canonical-form)-[IR](../../../../glossary.md#term-intermediate-representation-ir) emission replaces family [callback](../../../../glossary.md#term-callback) [shape](../../../../glossary.md#shape) |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuNativeWorkspace.java` | Remove; no 0005A workspace consumer |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuParallelExecutionException.java` | Remove; parallel strategy returns in ordered follow-up |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPointwiseAddCandidateSource.java` | Remove; replaced by whole-partition [lowering](../../../../glossary.md#lowering) |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPointwiseAddInvocation.java` | Remove; replaced by partition-level cold [binding](../../../../glossary.md#term-binding) |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPointwiseAddKernelEmitter.java` | Remove; replaced by canonical-IR [scalar](../../../../glossary.md#term-scalar) emission |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPointwiseAddLowering.java` | Remove; replaced by `internal/lowering/CpuPartitionLowering.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortableAnalysisInputs.java` | Remove; replaced by `internal/prepare/CpuPartitionAnalysisInputs.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortableCandidateSource.java` | Remove; no candidate [registry](../../../../glossary.md#term-registry)/source layer |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortableExecutionMode.java` | Remove; strategy is explicit plan data, not emitter [dispatch](../../../../glossary.md#term-dispatch) |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortableInvocationBinder.java` | Remove; binding belongs to the selected preparation plan/executable |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortableKernelCandidate.java` | Remove; replaced by canonical IR plus selected plan facts |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePartitionCandidate.java` | Remove; replaced by `internal/prepare/CpuPartitionPreparationPlan.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePartitionFinalizer.java` | Remove; replaced by `internal/prepare/CpuPartitionFinalizer.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePartitionPreparer.java` | Remove; replaced by `internal/prepare/CpuPartitionPreparer.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePreparationPlan.java` | Remove; replaced by `internal/prepare/CpuPartitionPreparationPlan.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePreparedExecutable.java` | Remove; replaced by `internal/executable/CpuPreparedExecutable.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPreparedParallelConfiguration.java` | Remove; no 0005A parallel strategy implementation |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuRangeBody.java` | Remove; generated primitive bounds replace hot range callbacks |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuReductionEmitter.java` | Remove; no current consumer and reductions return in a later unit task |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuVectorEmitter.java` | Remove; [vector](../../../../glossary.md#term-vector) emission returns with implemented vector strategy |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuWorkerGroup.java` | Remove; parallel orchestration returns in ordered follow-up |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/package-info.java` | Remove; replaced by exact structured package documentation below |

The new top-level production types are exactly:

| New path | Type and role |
|---|---|
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionAnalysisInputs.java` | Checked complete-partition analysis inputs |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java` | [Immutable](../../../../glossary.md#term-immutable) units, selected route/strategy, bindings, and declarations; exact nested types `ExecutionUnitPlan`, `Route`, and `ExecutionStrategy` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java` | Whole-partition analysis entry |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java` | Post-assignment verification, artifact realization, and executable finalization |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java` | Unit formation, exact [fusion](../../../../glossary.md#term-fusion) legality, and canonical lowering |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIr.java` | Route-independent typed computation and universal loop IR; exact nested value types `Value`, `Instruction`, `Loop`, and `Store` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuAccessPlan.java` | One normalized per-input access-plan form and cold binding contract; exact nested value types `Regime` and `Binding` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java` | Immutable portable Class-File realization facts beneath the route-neutral partition plan; no [graph](../../../../glossary.md#graph) interpretation, scalar-reference [ownership](../../../../glossary.md#term-ownership-meanings), or native-provider policy |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java` | [Conformance](../../../../glossary.md#term-conformance)/fail-closed scalar reference, never a [Runtime](../../../../glossary.md#term-runtime) IR interpreter |

Add exactly these ten package documentation paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/memory/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/package-info.java`

They state unsupported-internal status, ownership, hot/cold boundaries, and permitted
collaboration.

## Exact test path disposition

| Current path | Action and exact destination |
|---|---|
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/.gitkeep` | Remove; real root-package tests remain |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderPublicShapeTest.java` | Rewrite in place for sole supported public type and internal namespace exclusion |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java` | Rewrite in place for exact complete-[occurrence](../../../../glossary.md#term-occurrence) truth |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuBufferBindingTest.java` | Move/rewrite to `internal/memory/CpuBufferBindingTest.java` |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuClassFileKernelGeneratorTest.java` | Move/rewrite to `internal/codegen/emit/CpuClassFileKernelGeneratorTest.java` |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuGeneratedKernelArtifactStoreTest.java` | Move/rewrite to `internal/cache/CpuGeneratedKernelArtifactStoreTest.java` |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuGeneratedKernelShapeTest.java` | Move/rewrite to `internal/codegen/emit/CpuGeneratedKernelShapeTest.java` |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuKernelSpecializationTest.java` | Move/rewrite to `internal/cache/CpuKernelSpecializationTest.java` |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuNativeRepresentationTest.java` | Move/rewrite to `internal/memory/CpuNativeRepresentationTest.java` |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPreparedExecutableTest.java` | Move/rewrite to `internal/executable/CpuPreparedExecutableTest.java` |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPointwiseAddCandidateSourceTest.java` | Remove; obsolete candidate architecture |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPointwiseAddExecutionTest.java` | Remove; replaced by [fused](../../../../glossary.md#specialization-and-fusion) generated/reference coverage |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePartitionFinalizerTest.java` | Remove; replaced by [partition](../../../../glossary.md#partition) finalizer coverage |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePartitionPreparerTest.java` | Remove; replaced by partition preparer coverage |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePreparedExecutableTest.java` | Remove; replaced by partition executable coverage |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuWorkerGroupTest.java` | Remove; parallel coverage returns with parallel strategies |

Add exactly these seven new test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuFusedGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuShapePolymorphicArtifactTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

## [Atomic](../../../../glossary.md#resource-transaction-and-rollback) implementation path ceiling

The implementation task may touch at most 114 paths:

| Category | Maximum | Arithmetic |
|---|---:|---|
| [CPU](../../../../glossary.md#term-cpu) production | 72 | 39 current paths plus 14 moved destinations, 9 new type paths, and 10 new package-info paths; the two in-place paths are already among the 39 |
| CPU tests | 30 | 16 current paths, including `.gitkeep`, plus 7 moved destinations and 7 new test paths |
| Shared [Prepare](../../../../glossary.md#prepare) | 2 | `GraphPreparation.java` and `GraphPreparationTest.java` only |
| Explanatory documentation | 2 | CPU [backend](../../../../glossary.md#backend) guide and glossary |
| [Planning](../../../../glossary.md#term-planning)/status | 8 | this task, CPU master plan, roadmap, and the five historical CPU task status files |
| **Total** | **114** | **72 + 30 + 2 + 2 + 8** |

All source destination paths are enumerated above. No Java source may remain under the old
`backend.cpu.execution` package. No test may remain there. A necessary path outside this ceiling is
a planning stop and correction, not implied authorization.

## Acceptance criteria

- The old per-[node](../../../../glossary.md#node) pipeline and flat `execution` package are absent; every production and test
  path has the exact disposition above, and no dead or compatibility production code remains.
- Only `route.portable` exists after 0005A. No `route.nativeblas`, `route.nativeops`, OpenBLAS, or
  vendor leaf package is created before its concrete Draft task.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [leaf glossary](../../../../glossary.md#term-leaf))</span>
- `CpuCapabilityProvider` is the sole supported public CPU API. Minimal cross-subpackage contracts
  are technically public only below <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>`.internal`, documented unsupported, and enforced by focused
  public-shape and inventory tests.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
- One complete eligible partition lowers to one execution unit, one canonical IR, one generated
  class, one partition executable, and one <span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [canonical glossary](../../../../glossary.md#term-canonical-form); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>[`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation); its two internal values have no
  declaration or physical slot.
- The same generated class bytes and loaded compatibility identity execute at least two compatible
  non-zero extents for the identical ADD -> GELU -> MUL topology. Concrete extents and element
  count are cold-bound and excluded from default class/cache identity.<span data-glossary-navigation="true"> ([compatibility identity glossary](../../../../glossary.md#identity-meanings-and-algebraic-identities))</span><span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [topology glossary](../../../../glossary.md#term-topology); [element glossary](../../../../glossary.md#term-element); [cache glossary](../../../../glossary.md#term-cache))</span>
- Generated loops use primitive `start` and `end`; scalar/single-thread execution handles arbitrary
  bounds, scalar shape, zero elements, checked long arithmetic, and correct overlap/alias rejection.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Canonical IR is route-independent and excludes every selected-route/configuration, Runtime,
  graph-identity, generator-version, and instance-binding fact named above.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [graph glossary](../../../../glossary.md#graph); [binding glossary](../../../../glossary.md#term-binding))</span>
- Portable Class-File generation is the supported semantic baseline/fallback; route-neutral
  preparation admits later peer routes without a bridge layer, provider-owned lowering, another<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering))</span>
  [`BackendId`](../../../../glossary.md#backend-identity--backendid), or a fixed vendor priority.
- No-root artifact realization works entirely in memory. Configured-root hits are fully verified
  and defined; corrupt/missing storage safely emits in memory. No migration reader exists and no
  test claims persistence preserves JIT code or is faster.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [persistence glossary](../../../../glossary.md#term-persistence); [JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit))</span>
- Capability is truthful for complete occurrences and fails closed for unsupported type, shape,
  layout, parameter, alias, state, fan-out, publication, or partition facts.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [parameter glossary](../../../../glossary.md#term-parameter); [fan-out glossary](../../../../glossary.md#term-fan-out); [publication glossary](../../../../glossary.md#term-publication))</span>
- Exact GELU, NaN/infinity/signed-zero classifications, fixed operation order, generated/reference
  differential results, vector-tail invariants at the IR contract boundary, determinism, and
  compatibility fingerprints have focused tests. Every finite result has absolute error at most<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [operation glossary](../../../../glossary.md#operation); [finite glossary](../../../../glossary.md#term-finite); [absolute error glossary](../../../../glossary.md#term-absolute-error))</span>
  `2e-7 * max(1, abs(idealGelu))` against precomputed high-[precision](../../../../glossary.md#term-precision) [oracle](../../../../glossary.md#term-test-oracle) [vectors](../../../../glossary.md#term-vector) spanning tails,
  `[-12, 12]`, threshold neighborhoods, [subnormals](../../../../glossary.md#term-subnormal), and ordinary values.
- The proving slice uses exact/default eligibility only; its selected numerical mode is present in
  specialization/cache compatibility and the cold manifest, and neither emitters nor the hot path
  can introduce or look up an independent relaxed policy.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [specialization glossary](../../../../glossary.md#term-specialization); [manifest glossary](../../../../glossary.md#term-manifest); [hot path glossary](../../../../glossary.md#term-hot-path))</span>
- Shared Prepare rejects missing cross-partition producer/consumer declarations while permitting an
  undeclared same-partition value. It learns no CPU virtuality or materialization policy.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [producer glossary](../../../../glossary.md#term-producer); [materialization glossary](../../../../glossary.md#materialization))</span>
- Trace/debug support can emit a cold lowering manifest containing unit boundaries, fusion reasons,
  access regime, selected route/strategy, structural key, and resource declarations without
  exposing segments, addresses, or hot-path lookup.<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace); [fusion glossary](../../../../glossary.md#term-fusion); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Prepared execution strongly owns artifacts and bound resources for the documented lifetime and
  performs no hot lookup, allocation, reflection, storage classification, or semantic dispatch.<span data-glossary-navigation="true"> ([Prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [lifetime glossary](../../../../glossary.md#term-lifetime); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Focused CPU, Model/Planning/Prepare/Runtime contract checks, CPU Javadoc, module validation, and
  the reset checkpoint pass before statuses change. Repository-wide validation is required because
  shared Prepare and the complete CPU package surface change.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Planning glossary](../../../../glossary.md#term-planning); [Javadoc glossary](../../../../glossary.md#term-javadoc); [module glossary](../../../../glossary.md#term-module); [validation glossary](../../../../glossary.md#term-validation); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

## [Validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) the implementation-time commands required by the [planning](../../../../glossary.md#term-planning) guide and record exact suites,
counts, and results. At minimum:

```bash
./gradlew :backends:cpu:test
./gradlew :modules:prepare:test
./gradlew :backends:cpu:javadoc
./gradlew test
git diff --check
```

Also validate:

- same topology/two extents -> identical deterministic class bytes, one compatibility identity,
  different cold-bound start/end and correct results;<span data-glossary-navigation="true"> ([compatibility identity glossary](../../../../glossary.md#identity-meanings-and-algebraic-identities))</span><span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- exactly four [buffer](../../../../glossary.md#term-buffer) declarations and no slots for the [fused](../../../../glossary.md#specialization-and-fusion) intermediate values;
- no-root, compatible-root hit, corrupt-root fallback, and no-migration behavior;
- exact public and internal package/type inventory, only the implemented `route.portable` leaf,
  and no <span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf))</span>`backend.cpu.execution` paths/imports or native/vendor placeholder packages;
- no [`Operation`](../../../../glossary.md#operation), `CompiledNode`, IR interpretation, cursor object, per-element division/modulo,
  lookup, allocation, reflection, registry, or service locator in the generated/hot path;<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [element glossary](../../../../glossary.md#term-element); [lookup glossary](../../../../glossary.md#term-lookup); [registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator); [hot path glossary](../../../../glossary.md#term-hot-path))</span>
- checked long arithmetic, zero/scalar behavior, alias rejection, numerical/determinism rules,
  vector chunk/tail contract, fingerprints, and generated/reference differential vectors;<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [alias glossary](../../../../glossary.md#aliasing-and-arena); [vector glossary](../../../../glossary.md#term-vector))</span>
- local Markdown links, anchors, fences, final newlines, and trailing whitespace;
- 0005A is the sole detailed [Ready](../../../../glossary.md#term-ready) [CPU](../../../../glossary.md#term-cpu) task and every immediate follow-up is Draft with no spec;
- CPU 0001–0005 remain Complete until every final gate passes, then change atomically to
  Superseded while 0005A changes to Complete.

## Dependencies

- [CPU](../../../../glossary.md#term-cpu) tasks 0001 through 0005 are completed historical evidence.
- Current [Model](../../../../glossary.md#term-model) [`Shape`](../../../../glossary.md#shape), [`Dimension`](../../../../glossary.md#dimension), `ShapeBroadcast`, and `LayoutDescriptor` contracts.
- Current [Planning](../../../../glossary.md#term-planning) `LogicalMemoryPlan` and `PlannedPartition` contracts.
- Current [Prepare](../../../../glossary.md#prepare) staged analysis/assignment/finalization and [Runtime](../../../../glossary.md#term-runtime) [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) /
  [`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation) contracts.

## Ordered follow-up gates

The [CPU](../../../../glossary.md#term-cpu) master plan contains the only authorized follow-up records. In order they deliver:

1. universal right-aligned access plans and complete broadcast/[layout](../../../../glossary.md#layout) semantics;
2. [vector](../../../../glossary.md#term-vector), parallel-[scalar](../../../../glossary.md#term-scalar), and parallel-vector execution strategies;
3. CPU-internal [materialization](../../../../glossary.md#materialization) analysis and the optional [persistence](../../../../glossary.md#term-persistence)/[specialization](../../../../glossary.md#term-specialization) evidence gate;
4. broader portable pointwise types, carriers, and semantic families; then reduction/scan,
   matrix/convolution, vendor, and tuning milestones.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [scan glossary](../../../../glossary.md#term-cumulative-scan); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>

No follow-up has a detailed task specification while 0005A is [Ready](../../../../glossary.md#term-ready).

## Implementation prompt

Implement this task in a clean coding context. Read the required architecture, [planning](../../../../glossary.md#term-planning),
documentation, completed [CPU](../../../../glossary.md#term-cpu)-task, current CPU/[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime)/[Model](../../../../glossary.md#term-model)-contract, and selected legacy
evidence files. Treat legacy as behavioral evidence only. Respect the exact path map and 114-path
ceiling. Do not retain both pipelines, invent architecture, create future detailed specs, [commit](../../../../glossary.md#term-commit),
or push. After Java and tests stabilize, use a distinct clean documentation context to finalize
Javadocs and the authorized Markdown paths. Mark historical tasks Superseded and this task Complete
only after every implementation, [validation](../../../../glossary.md#term-validation), documentation, and [checkpoint](../../../../glossary.md#term-checkpoint) gate passes.

## Local decisions

- The current route set contains exactly `internal.route.portable`; no native or vendor leaf is
  present before its ordered concrete task.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf))</span>
- CPU 0005A selects only exact/default numerical mode and scalar/single-thread execution. The
  selected mode and strategy participate in compatibility identity rather than hot-path policy.<span data-glossary-navigation="true"> ([compatibility identity glossary](../../../../glossary.md#identity-meanings-and-algebraic-identities))</span><span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [scalar glossary](../../../../glossary.md#term-scalar); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Optional persistence accepts one current schema only. Absence, incompatibility, corruption, or
  publication failure returns to verified in-memory generation and does not affect correctness.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence); [schema glossary](../../../../glossary.md#term-schema); [publication glossary](../../../../glossary.md#term-publication))</span>
- Shared Prepare changed only its backend-neutral cross-partition declaration-completeness check.
  It gained no knowledge of CPU units, fusion, virtuality, or materialization.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [partition glossary](../../../../glossary.md#partition); [fusion glossary](../../../../glossary.md#term-fusion); [materialization glossary](../../../../glossary.md#materialization))</span>

## Known limitations

- The implemented semantic [slice](../../../../glossary.md#slice) is exactly parameterless [canonical](../../../../glossary.md#term-canonical-form)-dense [FLOAT64](../../../../glossary.md#term-float64)
  `ADD -> GELU -> MUL` with fully static equal [shapes](../../../../glossary.md#shape) and exact [GELU](../../../../glossary.md#gaussian-error-linear-unit--gelu) semantics.
- Only native [`MemorySegment`](../../../../glossary.md#term-memorysegment) buffers, canonical-dense zero-offset access, and the portable scalar
  strategy are executable. Broadcasting, views, heap or mixed carriers, vector and parallel
  execution, native routes, CPU-internal materialization, and broader semantic families remain
  ordered Draft work.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [scalar glossary](../../../../glossary.md#term-scalar); [Broadcasting glossary](../../../../glossary.md#broadcasting); [views glossary](../../../../glossary.md#term-view-meanings); [heap glossary](../../../../glossary.md#term-heap); [vector glossary](../../../../glossary.md#term-vector); [CPU glossary](../../../../glossary.md#term-cpu); [materialization glossary](../../../../glossary.md#materialization))</span>
- The task makes no backend-conformance, integration, performance, relaxed-math, tuning, or
  persistence-speed claim.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [persistence glossary](../../../../glossary.md#term-persistence))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation context completed the executable validation before the independent documentation
pass:

- `./gradlew :backends:cpu:test` passed 16 suites and 18 tests with zero failures, errors, or
  skips. The final replacement run recompiled CPU production because concurrent Javadoc edits were
  visible and still passed.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- `./gradlew :modules:prepare:test` passed 10 suites and 36 tests with zero failures, errors, or
  skips.
- `./gradlew test` passed with 56 actionable tasks: 1 executed and 55 up-to-date.
- `git diff --check` passed before the documentation pass.

The clean documentation-focused context independently reviewed the final [CPU](../../../../glossary.md#term-cpu) and shared [Prepare](../../../../glossary.md#prepare)
source and tests against the [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), [backend](../../../../glossary.md#backend)-guide, [planning](../../../../glossary.md#term-planning), example, and general profiles.
It finalized every affected CPU contract Javadoc, all ten structured internal package summaries,
the CPU guide, glossary, and synchronized planning/status records. It reused the successful Java
test evidence because it changed no executable behavior. Its final evidence is:

- `./gradlew :backends:cpu:javadoc` passed after the final Javadoc edits.
- Targeted local Markdown target/anchor, fence, final-newline, and trailing-whitespace checks
  passed for the authorized documentation paths.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- Exact production/test inventory, route-leaf, removed-package/import, later-task, affected-path,
  and 114-path-ceiling checks passed; the two pre-existing compiler/config planning edits were
  identified separately and preserved unchanged.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- `git diff --check` passed after all documentation and status edits.

## Implementation notes

The flat `backend.cpu.execution` production and test trees were removed atomically. The replacement
contains 35 [CPU](../../../../glossary.md#term-cpu) production Java paths and 16 CPU test Java paths, with one `route.portable` [leaf](../../../../glossary.md#term-leaf)
and no native placeholders. One eligible [partition](../../../../glossary.md#partition) lowers to one [fused](../../../../glossary.md#specialization-and-fusion) unit, one [canonical](../../../../glossary.md#term-canonical-form) [IR](../../../../glossary.md#term-intermediate-representation-ir),
one [artifact](../../../../glossary.md#term-artifact), one executable, and one [bound invocation](../../../../glossary.md#bound-invocation--boundinvocation). Its four declarations cover `a`, `b`,
`c`, and output; `sum` and `activated` remain virtual with no physical slot. Focused tests lock
universal primitive bounds, two compatible extents sharing deterministic bytes and loaded class
[identity](../../../../glossary.md#term-identity-meanings), exact generated/reference results, [persistence](../../../../glossary.md#term-persistence) fallback, preserved [graph](../../../../glossary.md#graph)/logical-memory
facts, and narrow cross-partition declaration hardening.

No [architecture contract](../../../../glossary.md#architecture-contract), focused architecture explanation, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), [architecture test](../../../../glossary.md#term-architecture-test), [backend](../../../../glossary.md#backend)
[conformance](../../../../glossary.md#term-conformance) test, [integration test](../../../../glossary.md#term-integration-test), [Gradle](../../../../glossary.md#term-gradle) configuration, [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler),
[Planning](../../../../glossary.md#term-planning), [Runtime](../../../../glossary.md#term-runtime), Config, [Trace](../../../../glossary.md#trace), or other backend change was required. CPU-private [ownership](../../../../glossary.md#term-ownership-meanings) and
the existing staged [Prepare](../../../../glossary.md#prepare)/Runtime boundary remain unchanged; only the authorized shared Prepare
[validation](../../../../glossary.md#term-validation) was hardened.

## Completion summary

Implemented and documented the [atomic](../../../../glossary.md#resource-transaction-and-rollback) [CPU](../../../../glossary.md#term-cpu) [partition](../../../../glossary.md#partition)-[kernel](../../../../glossary.md#kernel) reset. Historical CPU tasks 0001
through 0005 are preserved as Superseded records, CPU 0005A is Complete, and every later CPU task
remains Draft without a detailed task specification.

Status: Complete
