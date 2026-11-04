# Prepared executables and cold invocation [binding](../../glossary.md#term-binding)

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java)

A [prepared executable](../../glossary.md#prepared-executable--preparedexecutable) is a reusable recipe; a [bound invocation](../../glossary.md#bound-invocation--boundinvocation) is that recipe connected to one
[run](../../glossary.md#term-run)'s concrete resources. **Cold binding** performs compatibility checks before the repeated
execution path, so execution can use direct typed references. Here cold means setup rather than
numerical work; it is not the separate cold [graph](../../glossary.md#graph)-reconstruction [API](../../glossary.md#term-application-programming-interface-api).

<a id="current-prepared-executable-and-bound-invocation"></a>
## Current [prepared executable](../../glossary.md#prepared-executable--preparedexecutable) and [bound invocation](../../glossary.md#bound-invocation--boundinvocation)

[`PreparedExecutable`](../../glossary.md#prepared-executable--preparedexecutable) is an [immutable](../../glossary.md#term-immutable) reusable recipe for one prepared computation region. Its
constructor retains the exact [`PreparedMemoryPlan`](../../glossary.md#prepared-memory-plan--preparedmemoryplan) reference and snapshots ordered selections
into private arrays:

- `BufferSelection(bufferIndex, representationIndex)` addresses a dense position in
  `memoryPlan().buffers()` and then a dense representation position in that [buffer](../../glossary.md#term-buffer)'s [`RunState`](../../glossary.md#run-state--runstate)
  bindings; and
- `WorkspaceSelection(workspaceIndex)` addresses a dense position in
  `memoryPlan().workspaces()`.

These indices are list positions, not `BufferSlot.value()` or `WorkspaceSlot.value()` values.
Selections may be empty or repeated. Repetition represents repeated operand roles and does not
duplicate resource [ownership](../../glossary.md#term-ownership) in the state.

`bind(runState)` is the cold boundary. It first requires an open state whose `memoryPlan()` is the
exact same object as the executable's plan; an equal plan constructed separately is rejected. It
resolves buffers in selection order, then workspaces, and calls the [backend](../../glossary.md#backend)'s checked
compatibility hook exactly once for each resolved selection. Only after all checks pass does it
call `bindCompatible` with fresh [nominal](../../glossary.md#term-nominal) arrays in the same order.

The backend uses explicit checked tests such as `instanceof`, then constructs a
[`BoundInvocation`](../../glossary.md#bound-invocation--boundinvocation) with direct concrete typed fields. The returned invocation must retain the exact
supplied `RunState`. [Binding](../../glossary.md#term-binding) may allocate the temporary [Java Virtual Machine](../../glossary.md#term-java-virtual-machine-jvm) ([JVM](../../glossary.md#java-platform-and-build-tools)) arrays and
invocation object, but it acquires no auxiliary closeable or native binding resource, changes no
[ownership](../../glossary.md#term-ownership-meanings), and performs no cleanup on failure.

`BoundInvocation.execute()` checks that its retained state is still open, then calls the
backend's `executeBound()` method. It performs no slot [lookup](../../glossary.md#term-lookup), compatibility cast, [graph](../../glossary.md#graph) work,
backend discovery, route/configuration search, allocation, transfer, [residency](../../glossary.md#residency) decision,
[publication](../../glossary.md#term-publication), tuning, or tracing. Sequential calls while open are permitted. One invocation is not
[thread-safe](../../glossary.md#term-thread-safe) and must not [race](../../glossary.md#term-race) execution with state closure; after closure it fails with
`IllegalStateException("run state is closed")` before backend work.

One immutable executable may bind concurrently to distinct [run](../../glossary.md#term-run) states. Each resulting invocation
belongs to exactly one state and owns neither that state nor its buffer/workspace representations.
Concrete executable subclasses must therefore be immutable and thread-safe, while invocation
subclasses keep their per-run direct references isolated.

<a id="focused-cold-binding-example"></a>
## Focused cold-[binding](../../glossary.md#term-binding) example

<a id="cold-binding-goal-and-inputs"></a>
### Cold-[binding](../../glossary.md#term-binding) goal and inputs

Bind one [backend](../../glossary.md#backend)-specific [buffer](../../glossary.md#term-buffer) and workspace from a matching open state, execute once through
direct typed fields, then demonstrate the post-close [guard](../../glossary.md#term-guard). This local fake backend illustrates
the extension contract; it is not a production backend or a [Prepare](../../glossary.md#prepare) finalizer.

```java
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.util.List;

final class ExampleBuffer implements BufferRepresentation {
    int calls;
    @Override public void close() {}
}

final class ExampleWorkspace implements WorkspaceRepresentation {
    int calls;
    @Override public void close() {}
}

final class ExampleInvocation extends BoundInvocation {
    private final ExampleBuffer buffer;
    private final ExampleWorkspace workspace;

    ExampleInvocation(RunState state, ExampleBuffer buffer, ExampleWorkspace workspace) {
        super(state);
        this.buffer = buffer;
        this.workspace = workspace;
    }

    @Override protected void executeBound() {
        buffer.calls++;
        workspace.calls++;
    }
}

final class ExampleExecutable extends PreparedExecutable {
    ExampleExecutable(PreparedMemoryPlan plan) {
        super(
                plan,
                List.of(new BufferSelection(0, 0)),
                List.of(new WorkspaceSelection(0)));
    }

    @Override protected boolean acceptsBufferRepresentation(
            int index, BufferRepresentation representation) {
        return index == 0 && representation instanceof ExampleBuffer;
    }

    @Override protected boolean acceptsWorkspaceRepresentation(
            int index, WorkspaceRepresentation representation) {
        return index == 0 && representation instanceof ExampleWorkspace;
    }

    @Override protected BoundInvocation bindCompatible(
            RunState state,
            BufferRepresentation[] buffers,
            WorkspaceRepresentation[] workspaces) {
        return new ExampleInvocation(
                state, (ExampleBuffer) buffers[0], (ExampleWorkspace) workspaces[0]);
    }
}
```

Given the one-buffer/one-workspace `plan` from the earlier geometry example:

```java
ExampleBuffer buffer = new ExampleBuffer();
ExampleWorkspace workspace = new ExampleWorkspace();
RunState state =
        new RunState(
                plan,
                List.of(
                        List.of(
                                new BufferRepresentationBinding(
                                        buffer, RunResourceOwnership.RUN_OWNED))),
                List.of(workspace));

BoundInvocation invocation = new ExampleExecutable(plan).bind(state);
invocation.execute();
state.close();
```

<a id="cold-binding-result-and-interpretation"></a>
### Cold-[binding](../../glossary.md#term-binding) result and interpretation

Both `calls` fields are `1`. Compatibility was checked during `bind`, and execution used the two
direct concrete fields. Calling `invocation.execute()` after `state.close()` fails before either
counter changes. This proves the current binding and [lifecycle](../../glossary.md#lifecycle) [guard](../../glossary.md#term-guard) only; it does not allocate
storage, finalize a [backend](../../glossary.md#backend) analysis, schedule work, transfer values, or publish a result.


---

[Previous topic](creation.md) · [Next topic](transfers.md)
