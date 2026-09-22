# Metal Backend Master Plan

## Goal

Implement Metal capability, MPSGraph and custom-kernel preparation, storage, native integration, and execution.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- Metal capability provider
- MPSGraph and custom-kernel lowering
- Metal executables, storage, workspace, and materialization
- native bridge and typed tracing
- typed, version-controlled, tested MPSGraph/custom-route candidate generators and compatible
  workload-cache lookup during prepare

## Out of scope

- global autograd
- public Tensor ownership
- engine dependency
- training-owned Metal optimizer bridge

## Module invariants

- Metal prepare owns lowering and route selection.
- Planning must select Metal ownership before MPSGraph or a custom Metal kernel can be considered.
- MPSGraph and custom Metal kernels are exclusive to this backend. CPU never calls them as an
  internal Apple optimization route.
- Metal-specific optimizer execution remains in backend prepare or kernels.
- Metal backend never depends on engine.
- Metal candidate generators return complete valid route-specific configurations and remain
  opaque to shared tuning orchestration.
- Safe Metal heuristics remain correct without a compatible tuning result.
- Metal mixed-precision routes prioritize FLOAT16. BFLOAT16 remains a distinct logical type and is
  eligible only through an exact capability-, device-, and operation-gated route.
- Model task 0026 must define FLOAT16 and every affected input, accumulation/intermediate, and
  output contract before Metal advertises FLOAT16. FLOAT32 accumulation is the expected default
  for numerically sensitive 16-bit work unless Model explicitly specifies an exception.
- A logical data type or 16-bit storage representation alone never advertises or selects a Metal
  route. Exact target/operation/numerical filtering precedes route/workload benchmarking, safe
  heuristics, or compatible tuning evidence.

## Allowed dependencies

- modules/model
- modules/config
- modules/planning
- modules/runtime
- modules/prepare
- modules/backend-contract
- modules/trace

Task 0002 additionally permits `modules/compiler` only as a `testImplementation` dependency so
the backend-local `GraphPreparation` integration test can construct typed `CompileArtifacts`.
This does not add Compiler to Metal production dependencies. Backend-conformance may add a
test-only dependency on Metal for the public capability/partition-closure check.

## Forbidden dependencies

- modules/engine
- extensions/training implementation dependencies

## Package structure

```text
io.github.pho001.synaptik.backend.metal
  Deliberate public backend surface. Task 0002 changes only the existing capability provider's
  exact NEG answer; later supported integration types require their own task.

io.github.pho001.synaptik.backend.metal.internal
  Package-private native ABI, device/queue context, physical buffer/workspace representations,
  checked host byte access, and resource lifetime. Task 0002 keeps its first small vertical
  MPSGraph route, staged preparation, persistent resource, binding, and schedule assembly here so
  it does not widen cross-package implementation visibility before a second route proves a seam.

io.github.pho001.synaptik.backend.metal.internal.prepare
  Deferred package boundary for later preparation families after more than one route establishes
  a stable split.

io.github.pho001.synaptik.backend.metal.internal.route.mpsgraph
  Deferred package boundary for broader MPSGraph lowering after the task-0002 proof route. No
  custom-kernel or CPU route belongs here.
```

The module root is not the default destination for implementation types. The first two tasks keep
their coupled non-public foundation and proof route in one internal package. A later task must
update this map before extracting a route or preparation subpackage.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Metal capability, storage, and native foundation](tasks/0001-metal-capability-storage-and-native-foundation.md) | Complete | Complete shared planning, runtime, prepare, backend-contract, and trace contracts; no Model 0026 dependency while fail-closed | Established a fail-closed provider plus native device/queue and run-owned storage lifecycle without executable operation or CPU-owned offload. |
| 0002 | [MPSGraph prepared execution route](tasks/0002-mpsgraph-prepared-execution-route.md) | Complete | 0001; Runtime 0016; Prepare 0006; Engine 0010; Compiler 0006B7 | Added whole-maximal-partition positive-shape contiguous FLOAT32 NEG lowering with per-run splat initialization, ABI-v2 supplied MTLBuffer destinations, explicit run-owned binding storage, and reusable prepared ownership. |
| 0003 | Custom Metal kernel routes | Draft | 0001–0002 | Add validated custom-kernel lowering and execution for eligible Metal-owned work without making custom kernels a CPU route. |
| 0004 | Typed Metal route candidate generators and cache compatibility | Draft | 0002–0003, opaque prepare/tuning boundary and artifact versioning | Add colocated typed complete-candidate generation and canonical workload compatibility without exposing Metal knobs to planning or shared parameter bags. |


## Milestones

- Capability, native bridge, and storage foundation: task 0001 is the foundation checkpoint and
  closes only when its real macOS arm64 ABI/resource validation and documentation pass succeed.
- MPSGraph preparation and first executable operation path: task 0002.
- Custom routes, broader storage/materialization, and backend conformance: tasks 0003–0004 and the
  final Metal checkpoint.

## Current status

Task 0001 is `Complete`. The implementation and independent documentation pass found the current
Planning capability, Runtime representation/lifetime, Prepare staging, Backend Contract identity,
and JDK 26 FFM contracts sufficient for a bounded fail-closed foundation. It requires no
architecture, dependency, shared-module, or Engine change.

Task 0001 advertises no supported operation. Its native device/queue context and run-owned
storage prove only backend-private resource mechanics. Model 0026 is not a blocker for that work;
it remains mandatory before any later FLOAT16 semantic or capability claim. BFLOAT16 likewise
gains no capability from a two-byte representation.

Detailed [task 0002](tasks/0002-mpsgraph-prepared-execution-route.md) is `Complete`. The public
provider now reports exactly positive fully static rank-`1..16`, resolved dense-contiguous,
non-view, zero-offset `FLOAT32` unary `NEG` occurrences with equal input/output shapes and gradient
flags. Metal preparation accepts every resulting non-empty maximal partition as one executable,
including chains, independent nodes, fan-out, repeated consumption, internal values, multiple
boundaries, and differing valid shapes.

Analysis declares boundary buffers and one run-owned address workspace before shared slot
assignment. Finalization compiles one persistent shape-specialized MPSGraph executable and hands
its resource transactionally to `PreparedExecution`. Each run borrows caller buffers, creates
fresh initialized splat and output buffers, and fills its own address workspace during cold
binding. Hot execution makes one synchronous native downcall into supplied destinations. ABI
version `2` preserves the original seven symbols and statuses `0..7`, adds exactly three
executable symbols and statuses `8..11`, and links Foundation, Metal, and
MetalPerformanceShadersGraph.

Evidence is deliberately composed. Public `GraphCompilationPort`, shared `GraphPreparation`,
Runtime, and real MPSGraph prove the caller-input route. Backend-local typed `PrepareContext`
tests prove positive-rank splats through analysis, finalization, schedule assembly, and Runtime;
existing shared Prepare tests prove `ConstantSource -> PrepareContext.constants() ->
InitializedBuffer`. The public port cannot express explicit positive-rank forward constants, so
no combined public-port splat test is claimed. Deterministic fake/native-seam coverage is the
accepted status/error matrix.

The final ordinary Metal suite passed 35 total tests with three expected opt-in skips; the focused
prepared-execution suite passed 20 total tests with two expected opt-in skips. The native build,
exact ten-export and three-framework inspections, both exact real-device tests outside the
sandbox, focused two-test backend conformance, one-test dependency architecture check, source/
`javap` hot-path inspection, independent documentation review, Metal Javadoc, Markdown, exact
28-path scope, consistency, staged-file, and whitespace gates passed.

The public Metal surface remains only `MetalCapabilityProvider`. Current standard Engine
composition remains CPU-only; task 0002 adds no public Metal adapter, mixed-owner contract,
fallback, or generic integration surface. `FLOAT16` remains gated by Model 0026, and no BFLOAT16
support is inferred. Metal 0003 is the next ordered Draft row, but progressive planning leaves it
without a detailed specification.

The non-authoritative [Metal backend strategy note](../../../design/notes/metal-backend-strategy.md)
records the intended residency and synchronization direction for later task planning. It selects
`MTLBuffer` as the primary resident representation, preparation-time shape-specialized
`MPSGraphExecutable` construction for a maximal supported Metal region, direct input and supplied
output buffer binding, boundary-only transfers, and one synchronous wait at the region boundary or
before CPU/host consumption. These are planning constraints, not current capability, performance,
platform, or application binary interface promises.

Compiling the graph on every run, hiding it in a backend-global integration object, treating it as
per-run workspace, or relying on garbage-collection cleanup would violate cold-prepare, hot-run,
ownership, or deterministic-cleanup rules. Task 0002 evolved the same library from foundation ABI
version `1` to version `2`, preserved the original seven functions, and added exactly three opaque
executable functions. Neither task authorizes a generic backend registry, mixed-owner Engine
composition contract, or ABI workaround.

## Open questions

- Exact route-specific configuration records, target fingerprints, and candidate-schema versions
  wait for implemented Metal routes and the shared opaque orchestration consumer.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- Task 0001 uses the current non-executing foundation pattern: stable `metal` identity,
  unconditional fail-closed operation capability, opaque native resource ownership, and no
  Prepare or Engine integration.
- The macOS arm64 native foundation uses a versioned Objective-C C ABI reached through JDK 26 FFM
  from an explicitly supplied library path. Task 0001 builds a local test dylib but does not
  package, discover, extract, sign, notarize, or publish it.
- Foundation ABI version `1` and its exact seven symbols, signatures, statuses, opaque-handle
  ownership, output-cell rules, shared-storage/zero-byte behavior, FFM carriers, and eager
  resolution order are fixed by task 0001 rather than delegated to implementation.
- One native context owns the default Metal device and one command queue; run-owned buffer and
  workspace representations retain child leases so context shutdown cannot release their native
  prerequisite early. They implement Runtime's nominal representation roles, but task 0001 has
  no prepared Runtime integration and submits no commands.
- MPSGraph execution, the first truthful operation capability, and prepared-route integration are
  an atomic task-0002 boundary. Native availability or storage allocation alone never makes a
  capability answer true.
- Task 0002 must not move MPSGraph compilation onto the hot run path or place its compiled native
  executable in hidden backend-global ownership. Its detailed specification follows the completed
  fresh audit after Runtime 0016, Prepare 0006, and Engine 0010.
- Task 0002 must preserve Metal residency across adjacent Metal work, bind assigned input and
  output `MTLBuffer` instances through `MPSGraphTensorData`, supply the preallocated Synaptik
  destinations, and perform no explicit post-execution copy or requested hidden output
  materialization. This makes no claim about MPSGraph internal temporary storage. Eligible
  compile-time splats allocate and upload one fresh run-owned initialized Metal buffer during
  cold run-state creation for each run; no persistent/shared prepared constant buffer or generic
  constant materializer is introduced. No wait belongs
  inside a compiled region; general asynchronous execution, custom kernels, and whole-plan
  autotuning remain later work.
- Task 0002 retains the exact `MetalDeviceContext` identity in its analysis plan and requires the
  finalizer and schedule assembler to reject any different context. It declares one run-owned
  binding workspace so cold binding can pre-marshal native address arrays with explicit lifetime;
  bound execution allocates and marshals nothing in Java and performs one native downcall.
- Operation family selects the appropriate Metal candidate generator but is not a universal cache
  key. Model tuning may compare complete plans while Metal retains route and lowering ownership.
- Apple CPU acceleration through Accelerate belongs to the CPU backend. MPSGraph and custom Metal
  kernels remain here and are available only after Planning selects `owner = Metal`; CPU does not
  fall through or offload to this backend internally.
- FLOAT16 is the prioritized Metal mixed-precision type, but only after Model owns its true
  IEEE-754 binary16 semantics. BFLOAT16 remains supported only when exact device and operation
  capabilities admit it; neither route follows from storage width alone.

## Risks

- Moving Metal lowering into shared prepare or training.
- Exposing private Metal candidate fields through generic maps, reflection, or shared string
  dispatch.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
