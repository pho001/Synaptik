# [CUDA](../../../glossary.md#term-cuda) [Backend](../../../glossary.md#backend) Master Plan

Current source: `modules/backend/cuda`; [Gradle](../../../glossary.md#term-gradle) [identity](../../../glossary.md#term-identity-meanings) remains `:backends:cuda`. The
[merged README](../../../../modules/backend/cuda/README.md) retains the native scaffold only;
no executable CUDA capability or platform tree is added. Historical tasks remain here.

## Reader context

CUDA is NVIDIA's GPU programming platform. In this draft, the Java process is the **host** and
the GPU would be the **device**. A capability provider would answer whether a particular graph
operation is supported; a CUDA partition preparer would turn an accepted group into concrete
executable work. Runtime would invoke that prepared work, not choose kernels or discover a
device on demand. These are proposed component responsibilities, not implemented calls.

“Backend-owned” describes responsibility for CUDA lowering and resources, not ownership of
the application's Tensor objects. A Tensor describes a value or expression; that description
must not be repurposed as a mutable GPU-residency record. Device allocation, transfer completion
and cleanup need their own future contracts before any borrowed host storage could be released.
The [planned CUDA guide](../../../backend-guide/cuda-backend.md#planned-scope) explains that
separation, and its [native/resource prerequisites](../../../backend-guide/cuda-backend.md#native-and-resource-prerequisites)
identify the unresolved driver, context, stream and lifetime requirements. The Draft statuses
below remain Draft; this reading key adds neither a runnable CUDA route nor support evidence.

## Goal

Implement [CUDA](../../../glossary.md#term-cuda) capability, [backend-owned lowering](../../../glossary.md#backend-owned-lowering), [kernels](../../../glossary.md#kernel), storage, native integration, and [prepared execution](../../../glossary.md#prepared-execution--preparedexecution).

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../glossary.md#architecture-contract))</span>
- [Module boundaries](../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../glossary.md#term-module))</span>
- [Dependency rules](../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../glossary.md#term-dependency))</span>

## [Scope](../../../glossary.md#term-scope)

- [CUDA](../../../glossary.md#term-cuda) capability provider
- [partition](../../../glossary.md#partition) [lowering](../../../glossary.md#lowering), [specialization](../../../glossary.md#term-specialization), [fusion](../../../glossary.md#term-fusion), and route selection
- CUDA executables, storage, workspace, and transfers
- [native bridge](../../../glossary.md#term-native-bridge) and typed tracing
- typed, version-controlled, tested CUDA route candidate generators and compatible workload-cache
  lookup during prepare<span data-glossary-navigation="true"> ([cache glossary](../../../glossary.md#term-cache); [lookup glossary](../../../glossary.md#term-lookup); [prepare glossary](../../../glossary.md#prepare))</span>

## Out of [scope](../../../glossary.md#term-scope)

- global [compiler](../../../glossary.md#term-compiler) logic
- public [Tensor](../../../glossary.md#tensor) [ownership](../../../glossary.md#term-ownership-meanings)
- [engine](../../../glossary.md#term-engine) [dependency](../../../glossary.md#term-dependency)
- training-owned [CUDA](../../../glossary.md#term-cuda) [optimizer](../../../glossary.md#term-optimizer) bridge

## [Module](../../../glossary.md#term-module) invariants

- [CUDA](../../../glossary.md#term-cuda) [prepare](../../../glossary.md#prepare) owns concrete implementation selection.
- CUDA [backend](../../../glossary.md#backend) never depends on [engine](../../../glossary.md#term-engine).
- [Runtime](../../../glossary.md#term-runtime) receives only [prepared executable](../../../glossary.md#prepared-executable--preparedexecutable) contracts.
- CUDA candidate generators return complete valid route-specific configurations and remain
  opaque to shared tuning orchestration.<span data-glossary-navigation="true"> ([opaque glossary](../../../glossary.md#term-opaque))</span>
- Safe CUDA heuristics remain correct without a compatible tuning result.

## Allowed dependencies

- [modules](../../../glossary.md#term-module)/[model](../../../glossary.md#term-model)
- modules/config
- modules/[planning](../../../glossary.md#term-planning)
- modules/[runtime](../../../glossary.md#term-runtime)
- modules/[prepare](../../../glossary.md#prepare)
- modules/[backend](../../../glossary.md#backend)-contract
- modules/[trace](../../../glossary.md#trace)

## Forbidden dependencies

- [modules](../../../glossary.md#term-module)/[engine](../../../glossary.md#term-engine)
- extensions/training implementation dependencies

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [CUDA](../../../glossary.md#term-cuda) capability and native foundation | Draft | Stable shared [planning](../../../glossary.md#term-planning), [runtime](../../../glossary.md#term-runtime), [prepare](../../../glossary.md#prepare), [backend](../../../glossary.md#backend)-contract, and [trace](../../../glossary.md#trace) contracts | Establish truthful capability and native [lifecycle](../../../glossary.md#lifecycle) contracts. |
| 0002 | CUDA [prepared execution](../../../glossary.md#prepared-execution--preparedexecution) and routes | Draft | 0001 | Add validated [lowering](../../../glossary.md#lowering), storage, transfers, [kernels](../../../glossary.md#kernel), and prepared execution. |
| 0003 | Typed CUDA route candidate generators and [cache](../../../glossary.md#term-cache) compatibility | Draft | 0002, [opaque](../../../glossary.md#term-opaque) prepare/tuning boundary and [artifact](../../../glossary.md#term-artifact) versioning | Add colocated typed complete-candidate generation and [canonical](../../../glossary.md#term-canonical-form) workload compatibility without exposing CUDA knobs to planning or shared [parameter](../../../glossary.md#term-parameter) bags. |


## Milestones

- Capability and [native bridge](../../../glossary.md#term-native-bridge)
- [Prepared execution](../../../glossary.md#prepared-execution--preparedexecution) and storage
- [Kernel](../../../glossary.md#kernel) coverage and [conformance](../../../glossary.md#term-conformance)

## Current status

Draft.

This [backend](../../../glossary.md#backend) is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- Exact route-specific configuration records, target fingerprints, and candidate-schema versions
  wait for implemented CUDA routes and the shared opaque orchestration consumer.<span data-glossary-navigation="true"> ([target glossary](../../../glossary.md#term-target); [schema glossary](../../../glossary.md#term-schema); [CUDA glossary](../../../glossary.md#term-cuda); [opaque glossary](../../../glossary.md#term-opaque))</span>

## Decisions made

- The implementation must follow the current [architecture contract](../../../glossary.md#architecture-contract).
- Legacy code is capability evidence only; new implementation is written from scratch.
- Operation family selects the appropriate CUDA candidate generator but is not a universal cache
  key. Model tuning may compare complete plans while CUDA retains route and lowering ownership.<span data-glossary-navigation="true"> ([Operation glossary](../../../glossary.md#operation); [CUDA glossary](../../../glossary.md#term-cuda); [candidate generator glossary](../../../glossary.md#candidate-generator); [cache key glossary](../../../glossary.md#term-cache-key); [Model glossary](../../../glossary.md#term-model); [lowering glossary](../../../glossary.md#lowering); [ownership glossary](../../../glossary.md#term-ownership-meanings))</span>

## Risks

- Leaking concrete [CUDA](../../../glossary.md#term-cuda) details into [planning](../../../glossary.md#term-planning) or [runtime](../../../glossary.md#term-runtime).
- Exposing private CUDA candidate fields through generic maps, reflection, or shared string
  dispatch.<span data-glossary-navigation="true"> ([dispatch glossary](../../../glossary.md#term-dispatch))</span>

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md)<span data-glossary-navigation="true"> ([planning glossary](../../../glossary.md#term-planning))</span>.
