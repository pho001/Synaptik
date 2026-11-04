# [Prepared publication](../../glossary.md#prepared-publication--preparedpublication) and result [leases](../../glossary.md#term-lease)

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java)

**[Publication](../../glossary.md#term-publication)** makes one already valid [buffer](../../glossary.md#term-buffer) copy available at an ordered result position;
it does not compute the value or copy it to the [host](../../glossary.md#term-host). A result **[occurrence](../../glossary.md#term-occurrence)** is one position,
so two positions may intentionally refer to the same representation. The result lease keeps the
complete [run state](../../glossary.md#run-state--runstate) alive until close; [detached](../../glossary.md#detached-model-results) host [materialization](../../glossary.md#materialization) is a separate [Engine](../../glossary.md#term-engine) [operation](../../glossary.md#operation).

## Who receives the lease?

Runtime's runner calls `publish()` on each bound publication and constructs a Runtime result
only after all required occurrences succeed. Engine receives that inward result and wraps it
for the application. Neither publication call copies tensor values: it selects a valid resource
already kept alive by the state. An internal Engine materializer may borrow that representation
while the lease is open, but must not close it independently or retain it after result closure.
The application instead receives [detached host bytes](engine-boundary.md#current-ordinary-engine-boundary)
from its Engine result. The two result classes serve different callers and should not be
interchanged.

<a id="current-prepared-publication-and-result-lease"></a>
## Current [prepared publication](../../glossary.md#prepared-publication--preparedpublication) and result [lease](../../glossary.md#term-lease)

[`PreparedPublication`](../../glossary.md#prepared-publication--preparedpublication) is an [immutable](../../glossary.md#term-immutable), reusable recipe for one ordered result position. It uses
only dense [Runtime](../../glossary.md#term-runtime) coordinates: a [buffer](../../glossary.md#term-buffer) position in one exact [`PreparedMemoryPlan`](../../glossary.md#prepared-memory-plan--preparedmemoryplan), a
representation position within that buffer in a matching [`RunState`](../../glossary.md#run-state--runstate), and a result position. These
coordinates are not [compiler](../../glossary.md#term-compiler) [`TensorId`](../../glossary.md#tensorid) or [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot) [identities](../../glossary.md#term-identity-meanings). During preparation, the current
schedule assembler translates compiler logical [publication](../../glossary.md#term-publication) roles and assigned resources into
these Runtime coordinates.

`bind(runState)` is the sole representation [lookup](../../glossary.md#term-lookup). It requires an open state associated with the
same exact plan reference, validates the representation position, and returns a new
[`BoundPublication`](../../glossary.md#bound-publication--boundpublication) retaining the selected physical representation directly. [Binding](../../glossary.md#term-binding) changes no
validity or [ownership](../../glossary.md#term-ownership-meanings) and invokes no [backend](../../glossary.md#backend) work.

`BoundPublication.publish()` requires the state to remain open and the selected copy to be valid
at that exact moment. It then changes only the bound [occurrence](../../glossary.md#term-occurrence)'s one-shot flag. A repeated
publication, a closed state, or an invalid selected copy fails. Publication never searches for a
different valid copy, performs an implicit transfer or [materialization](../../glossary.md#materialization), calls a backend, or
changes `RunState` validity. When another representation is needed, an explicit [prepared buffer
transfer](../../glossary.md#prepared-buffer-transfer--preparedbuffertransfer) must occur before the publication suffix.

[`RunResult`](../../glossary.md#run-result--runresult) accepts a complete dense result-ordered list of successfully published occurrences
for one exact open state. It privately snapshots their direct representation references,
including intentional [aliases](../../glossary.md#aliasing-and-arena), and semantically takes responsibility for closing the complete
`RunState`. `publicationRepresentation(resultIndex)` returns the exact retained
`BufferRepresentation` for one dense occurrence while that exact result lease remains open.
Repeated access returns the same reference, and aliased occurrences return the same exact
reference without merging their positions. The accessor performs no copy, wrapper construction,
cast, validity query, transfer, mutation, or backend [operation](../../glossary.md#operation).

The returned representation is [borrowed](../../glossary.md#ownership-borrowing-and-lease) from the result. A caller must not close it, transfer
its ownership, retain or use it after result closure, or infer a longer [lifetime](../../glossary.md#term-lifetime) from alias
[identity](../../glossary.md#term-identity-meanings). The result remains non-[thread-safe](../../glossary.md#term-thread-safe), so callers must externally prevent this access from
racing any result or retained-state activity or closure. Closure is checked before index bounds;
an open-result out-of-range index fails with `IndexOutOfBoundsException`, while any access after
closure fails with `IllegalStateException("run state is closed")`. `resultCount()` and
`isClosed()` retain their post-close [metadata](../../glossary.md#term-metadata) behavior.

This is an inward Runtime service-provider interface ([SPI](../../glossary.md#term-service-provider-interface-spi)) boundary. `BufferRepresentation` has
no shared physical-access methods and is not [host storage](../../glossary.md#host-storage), a [Tensor](../../glossary.md#tensor) value, or a promise of [host](../../glossary.md#term-host)
accessibility. Ordinary [Engine](../../glossary.md#term-engine) `RunResult` exposes no representation; its separate explicit
materialization method composes this lease with the current [CPU](../../glossary.md#term-cpu) copy and returns only a [detached](../../glossary.md#detached-model-results)
host value. An empty Runtime publication list is valid. Constructor failure and partial
publication transfer no cleanup responsibility, so the runner remains responsible for closing
the state. Borrowed inputs remain caller-owned throughout the result lease, while state-owned
resources retain the existing deterministic cleanup behavior.

`BoundPublication` and Runtime `RunResult` are not thread-safe. They must not [race](../../glossary.md#term-race) publication,
validity mutation, execution, transfer, result construction, access, or closure. Immutable
prepared publication recipes may bind concurrently to distinct states, producing isolated flags
and leases.

<a id="focused-publication-example"></a>
### Focused [publication](../../glossary.md#term-publication) example

<a id="publication-goal-and-inputs"></a>
#### [Publication](../../glossary.md#term-publication) goal and inputs

Publish two ordered results that intentionally [alias](../../glossary.md#aliasing-and-arena) one already-valid [borrowed](../../glossary.md#ownership-borrowing-and-lease) representation,
then close the complete state through the result. This current example calls the publication
objects directly to isolate publication and result leasing from runner traversal.

```java
// Two result positions may share one representation; the result owns state cleanup, not borrowed storage.
import io.github.pho001.synaptik.runtime.run.BoundPublication;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.PreparedPublication;
import io.github.pho001.synaptik.runtime.run.RunResult;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

BufferRepresentation published = new CreationBuffer(new AtomicInteger());
RunState publicationState =
        new RunState(
                plan,
                List.of(
                        List.of(
                                new BufferRepresentationBinding(
                                        published, RunResourceOwnership.BORROWED))),
                List.of(new CreationWorkspace(new AtomicInteger())));

PreparedPublication first = new PreparedPublication(plan, 0, 0, 0);
PreparedPublication alias = new PreparedPublication(plan, 0, 0, 1);
PreparedSchedule schedule =
        new PreparedSchedule(
                plan,
                List.of(
                        new PreparedSchedule.PublicationStep(first),
                        new PreparedSchedule.PublicationStep(alias)));

BoundPublication boundFirst = first.bind(publicationState);
BoundPublication boundAlias = alias.bind(publicationState);
boundFirst.publish();
boundAlias.publish();

try (RunResult result = new RunResult(
        publicationState, List.of(boundFirst, boundAlias))) {
    assert schedule.publicationCount() == 2;
    assert result.resultCount() == 2;
    // This inward borrowed reference must not escape the open result lease.
    BufferRepresentation firstRepresentation = result.publicationRepresentation(0);
    assert firstRepresentation == published;
    assert result.publicationRepresentation(1) == firstRepresentation;
}
```

<a id="publication-result-and-interpretation"></a>
#### [Publication](../../glossary.md#term-publication) result and interpretation

Both result positions return the same exact [borrowed](../../glossary.md#ownership-borrowing-and-lease) representation, but the result count is two
because [aliases](../../glossary.md#aliasing-and-arena) preserve ordered result multiplicity. Access is valid only inside the open-result
[scope](../../glossary.md#term-scope) shown above; retaining or using either reference after the closing brace would violate the
[lease](../../glossary.md#term-lease). Neither publication nor access reads storage, performs a copy, establishes [host](../../glossary.md#term-host) semantics,
or transfers [ownership](../../glossary.md#term-ownership-meanings). Closing the result closes the leased state; because this example's
selected representation is borrowed, its physical cleanup remains the caller's responsibility.

If the selected copy were invalid, `publish()` would fail with
`IllegalStateException("published buffer representation is invalid")`. It would not choose
another copy or create a partial result. The [owner](../../glossary.md#term-owner) of the still-open state would then close it.


---

[Previous topic](schedules.md) · [Next topic](orchestration.md)
