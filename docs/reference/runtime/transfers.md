# Prepared buffer transfers

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java)

<a id="current-prepared-buffer-transfer"></a>
## Current prepared buffer transfer

`PreparedBufferTransfer` is an immutable reusable recipe for copying one logical buffer value
between two distinct, already-created representation positions of the same prepared buffer.
Materialization is this same explicit transfer when it produces the equivalent value in an
already-created representation required by later work. It is not another recipe, allocation
operation, route search, or coherence policy.

`bind(runState)` is the cold boundary. It requires the exact open state, validates both positions,
and lets the concrete backend check source compatibility before destination compatibility. The
backend then constructs a `BoundBufferTransfer` subclass with direct typed source and destination
fields. Binding performs no transfer and changes no ownership or validity.

`BoundBufferTransfer.execute()` first makes an already-valid destination a no-op, even if the
selected source is invalid. Otherwise it requires a valid source, calls the backend hook exactly
once, and marks only the destination valid after successful return. A backend
`RuntimeException` or `Error` propagates unchanged and leaves every Runtime validity bit
unchanged. Physical destination contents may be partial after failure, but Runtime still
classifies that copy as invalid.

<a id="focused-direct-reference-transfer-example"></a>
### Focused direct-reference transfer example

<a id="transfer-goal-and-inputs"></a>
#### Transfer goal and inputs

This current extension pattern binds two concrete buffers directly. It demonstrates success, the
resulting destination-valid no-op, and backend failure. A later runner will bind schedule
occurrences conceptually; this example calls the current recipe and action directly because no
runner exists yet.

```java
import io.github.pho001.synaptik.runtime.execution.BoundBufferTransfer;
import io.github.pho001.synaptik.runtime.execution.PreparedBufferTransfer;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;

final class TransferBuffer implements BufferRepresentation {
    int reads;
    int writes;
    @Override public void close() {}
}

final class TransferAction extends BoundBufferTransfer {
    private final TransferBuffer source;
    private final TransferBuffer destination;
    private final boolean fail;

    TransferAction(
            RunState state,
            TransferBuffer source,
            TransferBuffer destination,
            boolean fail) {
        super(state, 0, 0, 1);
        this.source = source;
        this.destination = destination;
        this.fail = fail;
    }

    @Override protected void executeTransfer() {
        source.reads++;
        destination.writes++;
        if (fail) {
            throw new IllegalStateException("physical copy failed");
        }
    }
}

final class TransferRecipe extends PreparedBufferTransfer {
    private final boolean fail;

    TransferRecipe(PreparedMemoryPlan plan, boolean fail) {
        super(plan, 0, 0, 1);
        this.fail = fail;
    }

    @Override protected boolean acceptsSourceBufferRepresentation(
            BufferRepresentation representation) {
        return representation instanceof TransferBuffer;
    }

    @Override protected boolean acceptsDestinationBufferRepresentation(
            BufferRepresentation representation) {
        return representation instanceof TransferBuffer;
    }

    @Override protected BoundBufferTransfer bindCompatible(
            RunState state,
            BufferRepresentation source,
            BufferRepresentation destination) {
        return new TransferAction(
                state, (TransferBuffer) source, (TransferBuffer) destination, fail);
    }
}
```

Given the earlier one-buffer/one-workspace `plan`, create two isolated states. Each starts with a
valid borrowed source and invalid run-owned destination and receives its own workspace because
the plan declares one workspace position.

```java
TransferBuffer source = new TransferBuffer();
TransferBuffer destination = new TransferBuffer();
RunState transferState =
        new RunState(
                plan,
                List.of(
                        List.of(
                                new BufferRepresentationBinding(
                                        source, RunResourceOwnership.BORROWED),
                                new BufferRepresentationBinding(
                                        destination, RunResourceOwnership.RUN_OWNED))),
                List.of(new ExampleWorkspace()));

BoundBufferTransfer transfer = new TransferRecipe(plan, false).bind(transferState);
transfer.execute();
transfer.execute();

TransferBuffer failingSource = new TransferBuffer();
TransferBuffer failingDestination = new TransferBuffer();
RunState failingState =
        new RunState(
                plan,
                List.of(
                        List.of(
                                new BufferRepresentationBinding(
                                        failingSource, RunResourceOwnership.BORROWED),
                                new BufferRepresentationBinding(
                                        failingDestination, RunResourceOwnership.RUN_OWNED))),
                List.of(new ExampleWorkspace()));
BoundBufferTransfer failing = new TransferRecipe(plan, true).bind(failingState);
try {
    failing.execute();
} catch (IllegalStateException expected) {
    // The exact backend failure propagates; Runtime leaves destination validity false.
}
```

<a id="transfer-result-and-interpretation"></a>
#### Transfer result and interpretation

After the two successful calls, `source.reads` and `destination.writes` are both `1` and both
copies in `transferState` are valid. The second call observed the valid destination and skipped
the backend hook. In `failingState`, the source remains valid and the destination remains invalid;
both physical counters are `1` because one backend attempt occurred. The action uses only its two
direct fields for physical work. This proves the current transfer and validity contract, not
byte-copy correctness, schedule traversal, executable-output invalidation, publication, result
ownership, or a public run lifecycle.


---

[Previous topic](executables.md) · [Next topic](schedules.md)
