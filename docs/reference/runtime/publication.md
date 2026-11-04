# Prepared publication and result leases

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java)

<a id="current-prepared-publication-and-result-lease"></a>
## Current prepared publication and result lease

`PreparedPublication` is an immutable, reusable recipe for one ordered result position. It uses
only dense Runtime coordinates: a buffer position in one exact `PreparedMemoryPlan`, a
representation position within that buffer in a matching `RunState`, and a result position. These
coordinates are not compiler `TensorId` or `ValueId` identities. Prepare will later translate
compiler logical publication roles into these Runtime coordinates.

`bind(runState)` is the sole representation lookup. It requires an open state associated with the
same exact plan reference, validates the representation position, and returns a new
`BoundPublication` retaining the selected physical representation directly. Binding changes no
validity or ownership and invokes no backend work.

`BoundPublication.publish()` requires the state to remain open and the selected copy to be valid
at that exact moment. It then changes only the bound occurrence's one-shot flag. A repeated
publication, a closed state, or an invalid selected copy fails. Publication never searches for a
different valid copy, performs an implicit transfer or materialization, calls a backend, or
changes `RunState` validity. When another representation is needed, an explicit prepared buffer
transfer must occur before the publication suffix.

`RunResult` accepts a complete dense result-ordered list of successfully published occurrences
for one exact open state. It privately snapshots their direct representation references,
including intentional aliases, and semantically takes responsibility for closing the complete
`RunState`. `publicationRepresentation(resultIndex)` returns the exact retained
`BufferRepresentation` for one dense occurrence while that exact result lease remains open.
Repeated access returns the same reference, and aliased occurrences return the same exact
reference without merging their positions. The accessor performs no copy, wrapper construction,
cast, validity query, transfer, mutation, or backend operation.

The returned representation is borrowed from the result. A caller must not close it, transfer
its ownership, retain or use it after result closure, or infer a longer lifetime from alias
identity. The result remains non-thread-safe, so callers must externally prevent this access from
racing any result or retained-state activity or closure. Closure is checked before index bounds;
an open-result out-of-range index fails with `IndexOutOfBoundsException`, while any access after
closure fails with `IllegalStateException("run state is closed")`. `resultCount()` and
`isClosed()` retain their post-close metadata behavior.

This is an inward Runtime service-provider interface (SPI) boundary. `BufferRepresentation` has
no shared physical-access methods and is not host storage, a Tensor value, or a promise of host
accessibility. Ordinary Engine `RunResult` exposes no representation; its separate explicit
materialization method composes this lease with the current CPU copy and returns only a detached
host value. An empty Runtime publication list is valid. Constructor failure and partial
publication transfer no cleanup responsibility, so the runner remains responsible for closing
the state. Borrowed inputs remain caller-owned throughout the result lease, while state-owned
resources retain the existing deterministic cleanup behavior.

`BoundPublication` and Runtime `RunResult` are not thread-safe. They must not race publication,
validity mutation, execution, transfer, result construction, access, or closure. Immutable
prepared publication recipes may bind concurrently to distinct states, producing isolated flags
and leases.

<a id="focused-publication-example"></a>
### Focused publication example

<a id="publication-goal-and-inputs"></a>
#### Publication goal and inputs

Publish two ordered results that intentionally alias one already-valid borrowed representation,
then close the complete state through the result. This current example calls the publication
objects directly to isolate publication and result leasing from runner traversal.

```java
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
    BufferRepresentation firstRepresentation = result.publicationRepresentation(0);
    assert firstRepresentation == published;
    assert result.publicationRepresentation(1) == firstRepresentation;
}
```

<a id="publication-result-and-interpretation"></a>
#### Publication result and interpretation

Both result positions return the same exact borrowed representation, but the result count is two
because aliases preserve ordered result multiplicity. Access is valid only inside the open-result
scope shown above; retaining or using either reference after the closing brace would violate the
lease. Neither publication nor access reads storage, performs a copy, establishes host semantics,
or transfers ownership. Closing the result closes the leased state; because this example's
selected representation is borrowed, its physical cleanup remains the caller's responsibility.

If the selected copy were invalid, `publish()` would fail with
`IllegalStateException("published buffer representation is invalid")`. It would not choose
another copy or create a partial result. The owner of the still-open state would then close it.


---

[Previous topic](schedules.md) · [Next topic](orchestration.md)
