# Ordered prepared schedules

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java)

A **schedule** is an ordered recipe for actions, not the actions already performed. Each list
position is an **[occurrence](../../glossary.md#term-occurrence)**: retaining the same executable twice requests two executions, not
two [owners](../../glossary.md#term-owner) of its resources. A **suffix** is the final portion of the list; [publication](../../glossary.md#term-publication) belongs
there so outputs are exposed only after the preceding execution and transfer actions.

## Who writes the schedule, and who follows it?

Engine's preparation composition supplies an assembler to shared Prepare. The assembler describes
creation, backend executions, necessary transfers and final publications; Prepare validates the
recipe before returning an owner. Only a later `PreparedExecutionRunner.run` binds resources and
follows those occurrences. A copied list of repeated recipe references is not copied native
storage and does not duplicate cleanup responsibility. See [Prepare assembly](orchestration.md#current-prepare-orchestration-assignment-and-backend-finalization)
for validation and [Runtime execution binding](executables.md) for the later direct calls.

<a id="current-prepared-schedule"></a>
## Current [prepared schedule](../../glossary.md#prepared-schedule--preparedschedule)

[`PreparedSchedule`](../../glossary.md#prepared-schedule--preparedschedule) is an [immutable](../../glossary.md#term-immutable) reusable recipe that associates an ordered list of work
occurrences with one exact [`PreparedMemoryPlan`](../../glossary.md#prepared-memory-plan--preparedmemoryplan). Its [sealed](../../glossary.md#term-sealed) nested `Step` interface exposes only
that plan association. `RepresentationCreationStep(PreparedRepresentationPlan)` retains the sole
optional cold-setup prefix, `ExecutionStep(PreparedExecutable)` retains one executable [occurrence](../../glossary.md#term-occurrence),
`BufferTransferStep(PreparedBufferTransfer)` retains one explicit transfer/[materialization](../../glossary.md#materialization)
occurrence, and `PublicationStep(PreparedPublication)` retains one final ordered result
occurrence. Each derives `memoryPlan()` from its exact component.

Construction validates each occurrence in supplied order and requires [reference identity](../../glossary.md#term-reference-identity) with
the schedule plan; a structurally equal but separately constructed plan is not the same prepared
association. Only after the complete [scan](../../glossary.md#term-cumulative-scan) succeeds does the schedule use `List.copyOf` to retain
an immutable ordered [snapshot](../../glossary.md#term-snapshot). It keeps exact step and recipe references. Empty schedules,
repeated executable or transfer step references, and repeated recipe references are valid because
each list position is one explicit occurrence and does not create another [ownership](../../glossary.md#term-ownership-meanings) relationship. A
creation step may occur zero or one time and, when present, must be index zero. A later [Prepare](../../glossary.md#prepare)
validator may require it for a runnable result; compatibility schedules may still be empty or
executable-only. [Publication](../../glossary.md#term-publication) steps, when present, form the final suffix. Their result indices must
be `0..N-1` in encounter order, and `publicationCount()` reports `N`. Distinct result positions
may intentionally name the same [buffer](../../glossary.md#term-buffer) and representation [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix).

<a id="focused-schedule-example"></a>
### Focused schedule example

<a id="schedule-goal-and-inputs"></a>
#### Schedule goal and inputs

Retain the representation plan from the creation example as the first [occurrence](../../glossary.md#term-occurrence), followed by one
transfer occurrence and two occurrences of the current `ExampleExecutable` recipe from the cold-
[binding](../../glossary.md#term-binding) example. All occurrences use the same exact plan; constructing the schedule invokes no
[callback](../../glossary.md#term-callback) and needs no [run state](../../glossary.md#run-state--runstate) or physical representation.

```java
// Repeating an executable reference requests two occurrences, not duplicate ownership of its resources.
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
still contains the same [`occurrence`](../../glossary.md#term-occurrence) reference twice after the [mutable](../../glossary.md#term-mutable) source list is cleared. The
returned list is [immutable](../../glossary.md#term-immutable). This proves creation-plan reachability, deterministic recipe order,
and [snapshot](../../glossary.md#term-snapshot) isolation only. Construction does not invoke a creator, bind or execute either
recipe, allocate or close a resource, create a [`RunState`](../../glossary.md#run-state--runstate), perform a transfer, or select
[publication](../../glossary.md#term-publication) behavior.

A null executable fails with `NullPointerException("executable")`. A null schedule plan, list,
or [element](../../glossary.md#term-element) reports `memoryPlan`, `steps`, or `steps[index]`, respectively. The first step whose
plan is not the exact schedule plan fails with
`IllegalArgumentException("steps[index] memory plan does not match schedule memory plan")`.
A creation step after index zero fails with
`IllegalArgumentException("steps[index] representation creation must be the first schedule occurrence")`.
Current transfer and [materialization](../../glossary.md#materialization) use `BufferTransferStep`; no second materialization variant
is hidden in the schedule. A publication step with a reordered or gapped result index fails, as
does any non-[publication occurrence](../../glossary.md#publication-occurrence) after publication begins. Schedule construction itself does
not bind, publish, inspect validity, or transfer [ownership](../../glossary.md#term-ownership-meanings).


---

[Previous topic](transfers.md) · [Next topic](publication.md)
