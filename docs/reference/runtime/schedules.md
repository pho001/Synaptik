# Ordered prepared schedules

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java)

<a id="current-prepared-schedule"></a>
## Current prepared schedule

`PreparedSchedule` is an immutable reusable recipe that associates an ordered list of work
occurrences with one exact `PreparedMemoryPlan`. Its sealed nested `Step` interface exposes only
that plan association. `RepresentationCreationStep(PreparedRepresentationPlan)` retains the sole
optional cold-setup prefix, `ExecutionStep(PreparedExecutable)` retains one executable occurrence,
`BufferTransferStep(PreparedBufferTransfer)` retains one explicit transfer/materialization
occurrence, and `PublicationStep(PreparedPublication)` retains one final ordered result
occurrence. Each derives `memoryPlan()` from its exact component.

Construction validates each occurrence in supplied order and requires reference identity with
the schedule plan; a structurally equal but separately constructed plan is not the same prepared
association. Only after the complete scan succeeds does the schedule use `List.copyOf` to retain
an immutable ordered snapshot. It keeps exact step and recipe references. Empty schedules,
repeated executable or transfer step references, and repeated recipe references are valid because
each list position is one explicit occurrence and does not create another ownership relationship. A
creation step may occur zero or one time and, when present, must be index zero. A later Prepare
validator may require it for a runnable result; compatibility schedules may still be empty or
executable-only. Publication steps, when present, form the final suffix. Their result indices must
be `0..N-1` in encounter order, and `publicationCount()` reports `N`. Distinct result positions
may intentionally name the same buffer and representation coordinate.

<a id="focused-schedule-example"></a>
### Focused schedule example

<a id="schedule-goal-and-inputs"></a>
#### Schedule goal and inputs

Retain the representation plan from the creation example as the first occurrence, followed by one
transfer occurrence and two occurrences of the current `ExampleExecutable` recipe from the cold-
binding example. All occurrences use the same exact plan; constructing the schedule invokes no
callback and needs no run state or physical representation.

```java
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.util.ArrayList;
import java.util.List;

ExampleExecutable executable = new ExampleExecutable(plan);
PreparedSchedule.ExecutionStep occurrence =
        new PreparedSchedule.ExecutionStep(executable);
PreparedSchedule.RepresentationCreationStep creation =
        new PreparedSchedule.RepresentationCreationStep(representationPlan);
PreparedSchedule.BufferTransferStep transferOccurrence =
        new PreparedSchedule.BufferTransferStep(new TransferRecipe(plan, false));
ArrayList<PreparedSchedule.Step> supplied =
        new ArrayList<>(List.of(creation, transferOccurrence, occurrence, occurrence));

PreparedSchedule schedule = new PreparedSchedule(plan, supplied);
supplied.clear();
```

<a id="schedule-result-and-interpretation"></a>
#### Schedule result and interpretation

`schedule.memoryPlan()` is the exact `plan` object. `schedule.steps().getFirst()` is the exact
`creation` reference, `schedule.steps().get(1)` is the exact `transferOccurrence`, and the list
still contains the same `occurrence` reference twice after the mutable source list is cleared. The
returned list is immutable. This proves creation-plan reachability, deterministic recipe order,
and snapshot isolation only. Construction does not invoke a creator, bind or execute either
recipe, allocate or close a resource, create a `RunState`, perform a transfer, or select
publication behavior.

A null executable fails with `NullPointerException("executable")`. A null schedule plan, list,
or element reports `memoryPlan`, `steps`, or `steps[index]`, respectively. The first step whose
plan is not the exact schedule plan fails with
`IllegalArgumentException("steps[index] memory plan does not match schedule memory plan")`.
A creation step after index zero fails with
`IllegalArgumentException("steps[index] representation creation must be the first schedule occurrence")`.
Current transfer and materialization use `BufferTransferStep`; no second materialization variant
is hidden in the schedule. A publication step with a reordered or gapped result index fails, as
does any non-publication occurrence after publication begins. Schedule construction itself does
not bind, publish, inspect validity, or transfer ownership.


---

[Previous topic](transfers.md) · [Next topic](publication.md)
