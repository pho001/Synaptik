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

## Forbidden dependencies

- modules/engine
- extensions/training implementation dependencies

## Package structure

```text
io.github.pho001.synaptik.backend.metal
  Deliberate public backend surface. Task 0001 adds only the fail-closed capability provider;
  later supported integration types require their own task.

io.github.pho001.synaptik.backend.metal.internal
  Task-0001 package-private native ABI, device/queue context, physical buffer/workspace
  representations, checked host byte access, and resource lifetime.

io.github.pho001.synaptik.backend.metal.internal.prepare
  Planned task-0002 partition analysis, exact resource declaration, finalization, and schedule
  integration after at least one complete executable route exists.

io.github.pho001.synaptik.backend.metal.internal.route.mpsgraph
  Planned task-0002 MPSGraph-only lowering and executable state. No custom-kernel or CPU route
  belongs here.
```

The module root is not the default destination for implementation types. Task 0001 keeps its
coupled non-public resource foundation in one internal package; task 0002 must update this map if
evidence requires a different package boundary before creating it.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Metal capability, storage, and native foundation](tasks/0001-metal-capability-storage-and-native-foundation.md) | Complete | Complete shared planning, runtime, prepare, backend-contract, and trace contracts; no Model 0026 dependency while fail-closed | Established a fail-closed provider plus native device/queue and run-owned storage lifecycle without executable operation or CPU-owned offload. |
| 0002 | MPSGraph prepared execution route | Draft | 0001 | Add backend-owned MPSGraph lowering, executable creation, storage/materialization integration, and execution only for Metal-owned partitions. |
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

Task 0002 remains Draft and has no detailed specification. It begins only after 0001 completes
and owns the first complete MPSGraph prepared execution route: exact advertised operations,
Metal-owned lowering, analysis and resource declarations, finalization, representation and
materialization/transfer integration, schedule execution, and conformance evidence. Neither 0001
nor 0002 authorizes a generic backend registry or mixed-owner Engine composition contract.

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
