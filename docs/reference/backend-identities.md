# [Backend](../glossary.md#backend) [identities](../glossary.md#term-identity-meanings) and declarative requirements

[Reference goals](index.md) · [Javadoc and exact signatures](javadoc.md)

A [backend identity](../glossary.md#backend-identity--backendid) names the implementation responsible for computation; a [device](../glossary.md#term-device) [identity](../glossary.md#term-identity-meanings) names
one reported device within that implementation. Availability says that a device was reported at
a particular moment, while capability says whether an exact [operation](../glossary.md#operation) can be supported. A
requirement filters acceptable choices; none of these data values opens a device or runs work.


The implemented `modules:backend-contract` surface contains:

- [`BackendId`](../glossary.md#backend-identity--backendid), an [immutable](../glossary.md#term-immutable) open-string identity for a [backend ownership](../glossary.md#backend-ownership) domain; and
- [`BackendDeviceId`](../glossary.md#backend-device-identity--backenddeviceid), an immutable composite identity for one opaque device token scoped by its
  owning `BackendId`; and
- [`DeviceClass`](../glossary.md#device-class--deviceclass), a coarse declarative category with exactly [`CPU`](../glossary.md#term-cpu) and [`ACCELERATOR`](../glossary.md#term-accelerator), in that
  declaration order; and
- [`BackendAvailabilitySnapshot`](../glossary.md#backend-availability-snapshot--backendavailabilitysnapshot), an immutable point-in-time association from one backend's
  currently reported available device identities to their classes; and
- the [sealed](../glossary.md#term-sealed), method-free [`BackendRequirement`](../glossary.md#backend-requirement--backendrequirement) family, with one-component
  `BackendIdRequirement`, `BackendDeviceIdRequirement`, and `DeviceClassRequirement` records.

The current concepts have separate roles:

```text
BackendId       = backend ownership domain
BackendDeviceId = one exact device identity inside that domain
DeviceClass     = coarse CPU or accelerator category
BackendAvailabilitySnapshot
                = one backend's supplied device-to-class availability fact
BackendRequirement
                = one hard target; no matching or preference behavior
```

The two identity records reject null components and blank string values. Every other identity
component is retained by the exact caller-supplied reference, and string content keeps its case
and surrounding whitespace; the records do not trim, normalize, intern, or resolve [aliases](../glossary.md#aliasing-and-arena).
Ordinary record equality therefore compares the exact stored content, and the backend component
prevents equal device [tokens](../glossary.md#term-token) from different backends from colliding.
`DeviceClass` is not stored in `BackendDeviceId`; the [snapshot](../glossary.md#term-snapshot) supplies the association without
changing either identity or category. The enum declaration order supports stable identity and
diagnostics, not preference, score, priority, capability, or fallback policy.

For example, the following inputs report one accelerator device for the `"cuda"` backend:

```java
// These illustrative availability records report facts; they do not open or register CUDA.
import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import java.util.Map;

BackendId cuda = new BackendId("cuda");
BackendDeviceId cudaZero = new BackendDeviceId(cuda, "0");
BackendAvailabilitySnapshot availability =
        new BackendAvailabilitySnapshot(
                cuda,
                Map.of(cudaZero, DeviceClass.ACCELERATOR));
```

The snapshot requires every device identity to have a `BackendId` equal to `cuda`. It retains the
exact backend, device, and class references and uses an immutable structural copy of the map, so
later changes to a [mutable](../glossary.md#term-mutable) source map cannot change the snapshot. Map iteration order is
unspecified. An empty map means that the supplying context reports no currently available device
for that backend; the backend identity remains part of the snapshot.

The snapshot does not discover devices, register a backend, monitor [liveness](../glossary.md#term-liveness), refresh itself,
evaluate capability, choose [ownership](../glossary.md#term-ownership-meanings), or guarantee preparation or execution.

The requirement family records exactly one hard [target](../glossary.md#term-target). These inputs construct all three current
variants:

```java
// Requirements filter later eligibility; constructing them neither chooses nor opens a device.
import io.github.pho001.synaptik.backend.contract.BackendDeviceIdRequirement;
import io.github.pho001.synaptik.backend.contract.BackendIdRequirement;
import io.github.pho001.synaptik.backend.contract.BackendRequirement;
import io.github.pho001.synaptik.backend.contract.DeviceClassRequirement;

BackendRequirement exactBackend = new BackendIdRequirement(cuda);
BackendRequirement exactDevice = new BackendDeviceIdRequirement(cudaZero);
BackendRequirement acceleratorClass =
        new DeviceClassRequirement(DeviceClass.ACCELERATOR);
```

`exactBackend` targets later ownership by a `BackendId` equal to `cuda`. `exactDevice` targets a
`BackendDeviceId` equal to `cudaZero`, which also fixes the owning backend. `acceleratorClass`
allows any later eligible device whose class is `ACCELERATOR`; it does not identify a particular
backend or device. Each record rejects a null component with a `NullPointerException` whose
message is that component's name and returns the exact supplied reference from its accessor.

These values neither inspect `availability` nor prove that a target is registered, available,
capable, or preparable. They contain no sentinel for absence, preference, fallback, combination,
matcher, or score. Configuration owns whether a requirement is present. Current internal [planning](../glossary.md#term-planning)
evaluates it with availability and capability facts; an empty hard-eligible result then fails in
the internal selector with
`IllegalStateException("no hard-eligible backend is available for ownership selection")`.
That internal message is not the ordinary public [compile](../glossary.md#compile) failure contract. `Engine.standard()`
supplies one CPU provider; `Engine.builder()` compiles against the frozen providers of explicitly
registered CPU and/or [Metal](../glossary.md#term-metal) integrations. Generic [plugin](../glossary.md#term-plugin) registration, other executable backends,
and a public [graph](../glossary.md#graph)-wide Planning surface remain planned.


---

[Previous topic](model-trace.md) · [Next topic](compiler/configuration-values.md)
## Who supplies availability, and who consumes it?

An opened backend integration supplies Engine with an identity, capability provider and availability
snapshot. Engine freezes that inventory when built; Compiler later calls Planning with those facts
for each operation occurrence. The example below constructs illustrative CUDA **data**, not an
opened CUDA integration or an execution promise. The records copy map/list membership where
specified and retain immutable identity references; they own no device resource to close.
Follow [explicit CPU/Metal composition](engine/composition.md) for resource transfer and
[per-occurrence Planning](compiler/planning.md#current-operation-capability-contracts) for the
separate question of semantic support.

