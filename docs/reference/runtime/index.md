<a id="runtime-api"></a>
# Runtime purpose and mental model

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java)

Think of Runtime as following an already prepared work list, not interpreting a Tensor graph.
A **schedule** gives the order of actions, **slots** name planned storage roles, and
**representations** supply the backend's concrete resources for those roles. One **run state**
tracks a single invocation's resources and which buffer copies currently hold usable values.

<a id="purpose-and-implementation-status"></a>
## Purpose and implementation status

This reference explains the implemented Runtime foundations for prepared-memory geometry,
reusable representation creation descriptions, one run's physical-representation lifecycle and
validity, checked binding of backend recipes, ordered scheduling, prepared publication, the
completed-result lease, the immutable
prepared-execution root, and the current Prepare-owned orchestration that constructs geometry,
finalizes backend recipes, validates the complete schedule, and returns that root. The current
public surface contains:

- `runtime.memory`: `BufferSlot`, `WorkspaceSlot`, and `PreparedMemoryPlan` with its nested
  `BufferEntry` and `WorkspaceEntry` records;
- `runtime.resource`: the nominal `BufferRepresentation` and `WorkspaceRepresentation` lifecycle
  roles implemented by concrete backends, the persistent `PreparedResource` cleanup role, plus
  `PreparedRepresentationPlan` and its nested preparation and creator contracts;
- `runtime.run`: `RunResourceOwnership`, `BufferRepresentationBinding`, `RunState`,
  `PreparedPublication`, `BoundPublication`, `RunResult`, and stateless
  `PreparedExecutionRunner`;
- `runtime.execution`: `PreparedExecution`, `PreparedExecutable`, its nested `BufferSelection`,
  `WorkspaceSelection`, and `BufferAccess` contracts, `BoundInvocation`,
  `PreparedBufferTransfer`, and `BoundBufferTransfer`; and
- `runtime.schedule`: `PreparedSchedule`, its sealed nested `Step` contract,
  `RepresentationCreationStep`, `ExecutionStep`, `BufferTransferStep`, and `PublicationStep`.

The geometry, reusable creation description, package-private all-or-cleaned setup, structural
residency, explicit per-buffer-copy validity, cold-bound invocation and transfer contracts,
creation, execution, transfer, and dense publication-suffix schedule recipes, the whole-state
result lease, lifecycle-bearing prepared-execution root, Prepare-owned resource assignments,
typed backend finalization, `PreparedPartition`, and complete graph preparation are current.
Ordinary compile, shared single-owner or mixed-owner prepare, logical host-input binding,
synchronous run, typed publication metadata, and explicit bounded detached host materialization
are current through `Engine`. `Engine.standard()` supplies fixed CPU ownership;
`Engine.builder()` can explicitly compose CPU and Metal. Their exact integrations implement
physical allocation, storage access, finalization and contribution, execution, transfer endpoints,
ingress, and materialization for their supported domains.

<a id="mental-model"></a>
## Mental model

```text
compile facts             prepare handoff               Runtime state
ValueId / requirement -> source-to-slot assignment -> PreparedMemoryPlan
current                  current                      current
                                                              |
                         PreparedRepresentationPlan ----------+
                         current reusable creation description |
                                                              v
                         cold setup -> RunState + validity
                         current internal  current per run
     PreparedMemoryPlan + PreparedSchedule + persistent resources -> PreparedExecution
     current              current            current contract       current reusable owner

        PreparedExecutionRunner -> lease -> create/bind all -> ordered direct traversal
        current public runner      current  cold setup          current synchronous run

        BufferTransferStep -> cold bind -> BoundBufferTransfer -> transfer + validity
        current recipe        current       current per run      current contract

        PublicationStep -> cold bind -> BoundPublication -> publish -> RunResult
        current recipe      current       current per run     current    current lease
```

Read the flow from logical source facts toward invocation state. Prepare exposes exact buffer and
workspace requirements through `BackendPartitionAnalysis`, then its current package-internal
handoff retains source-to-slot associations and constructs the Runtime geometry. Public
`GraphPreparation.prepare(...)` coordinates that handoff and validates one explicitly supplied
schedule recipe before returning `PreparedExecution`. Current `PreparedRepresentationPlan`
describes borrowed inputs and concrete-backend creators in that geometry's encounter order.
Package-private cold setup validates all caller inputs, creates fresh run-owned representation
objects and workspaces, and constructs one `RunState`. A creator may return a fresh read-only
binding whose child lease refers to immutable prepared physical storage; object freshness and
run-owned cleanup remain unchanged. A current `PreparedExecutable` selects those resident
representations by dense position, checks backend compatibility during cold binding, and creates a
per-run `BoundInvocation`. A current `PreparedBufferTransfer` selects two distinct already-created
representations of one buffer, checks their concrete compatibility during cold binding, and creates
a per-run `BoundBufferTransfer` that orchestrates the explicit validity transition around
backend-owned physical transfer work. The shared contracts still implement no concrete allocation
or storage access.
The current schedule can retain one first-only creation prefix
followed by executable or transfer occurrences and then a dense publication-only suffix. It does
not invoke or execute any step. Current publication names an already-created valid copy and
leases the complete state to a result. While that exact lease remains open, the inward Runtime
result can return the borrowed nominal representation selected for one result occurrence; this
is not output-value or storage access. The runner composes these contracts without backend
discovery or graph interpretation.


---

[Next topic](engine-boundary.md)

## Topic navigation

- [Runtime public contract relationships](contracts.md)
- [Prepare declarations, assignment and finalization](prepare.md)
- [Public Engine binding and results](engine-boundary.md)
- [Advanced Engine and prepared execution](prepared-execution.md)
- [Slots and prepared-memory geometry](memory.md)
- [Per-run representations and validity](representations.md)
- [Representation creation, run-state closure and failures](creation.md)
- [Prepared executables and cold invocation binding](executables.md)
- [Prepared buffer transfers](transfers.md)
- [Ordered prepared schedules](schedules.md)
- [Prepared publication and result leases](publication.md)
- [Prepare finalization and run orchestration](orchestration.md)
