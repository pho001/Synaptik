# Runtime API

## Purpose and implementation status

This reference explains the current runtime-owned buffer-slot identity and separates it from
planned preparation and execution contracts. The only implemented runtime production type is
`io.github.pho001.synaptik.runtime.memory.BufferSlot`; the prepare lifecycle and runnable runtime
API remain conceptual.

## Mental model

```text
compile                         prepare                          run
logical ValueId --planned--> plan-local BufferSlot --planned--> bound storage
immutable graph identity       reusable slot identity           per-run state
```

`ValueId` names logical data in one compiled graph. `BufferSlot` names a position only within one
owning prepared-memory-plan context. Later preparation may associate the two, and later run state
may bind a slot to storage, but `BufferSlot` itself performs neither step.

## Current buffer-slot contract

`BufferSlot` is a public, deeply immutable record with one `long value` component. It accepts
every value from zero through `Long.MAX_VALUE`; no sentinel is reserved. A negative value fails
with `IllegalArgumentException` and message `value must be non-negative`.

The component is opaque outside its owning plan context. The record stores no owner reference, so
ordinary record equality and hashing compare only the numeric component. Two plans may reuse the
same number without referring to the same conceptual slot. Diagnostic record text is not a
serialization format.

Creating a slot does not allocate, acquire, retain, release, or identify physical storage. A slot
is not a `ValueId`, address, storage handle, allocation, device, residency fact, workspace, or
resource.

## Focused example

### Goal and inputs

Create the first slot identity in one future prepared-memory-plan context and inspect its exact
stored value.

```java
import io.github.pho001.synaptik.runtime.memory.BufferSlot;

BufferSlot firstSlot = new BufferSlot(0L);
long identity = firstSlot.value();
```

### Result and interpretation

`identity` is `0`. The result proves only that `firstSlot` retains a valid plan-local numeric
identity. It does not allocate slot zero, bind storage, create a prepared memory plan, or make the
slot globally unique.

As a useful failure boundary, `new BufferSlot(-1L)` throws
`IllegalArgumentException("value must be non-negative")` before construction completes.

## Planned prepared contracts

`PreparedExecution` will contain or reference prepared partitions, executable units, a physical memory plan, and a schedule. Preparation creates it once for a selected set of explicitly registered backends; multiple runs may reuse it.

`PreparedExecutable` will compute only its prepared region. Its hot-path contract will not receive `Operation` or `CompiledNode`, and runtime will not ask it to rediscover a backend or select a kernel.

## Planned run contract

```java
// Conceptual API; not currently runnable.
RunResult result = execution.run(inputs, RunOptions.defaults());
```

- `inputs` will bind invocation values to prepared input bindings.
- `RunOptions` will hold declarative run and publication choices, not live services.
- `RunState` will own per-run mutable slots, resources, and residency facts.
- `RunResult` will expose results published by the prepared publication plan and run policy.

Exact collection types, nullability, concurrency guarantees, ownership of returned values, exception types, and resource-lifetime methods remain open until focused runtime tasks define and test them.

## Boundary and failure model

Run may fail because an input binding is missing or incompatible, a prepared resource cannot be used, transfer or execution fails, or publication fails. It must not recover by discovering another backend and lowering again. Unsupported work must be resolved during compile ownership or fail during prepare.

## Related contracts

- [Runtime, prepare, and backend boundary](../architecture/runtime-prepare-backend-boundary.md)
- [Preparing execution](../user-guide/preparing-execution.md)
- [Running models](../user-guide/running-models.md)
- [Glossary](../glossary.md)
