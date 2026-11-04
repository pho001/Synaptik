# Backend identities and declarative requirements

[Reference goals](index.md) · [Javadoc and exact signatures](javadoc.md)

The implemented `modules:backend-contract` surface contains:

- `BackendId`, an immutable open-string identity for a backend ownership domain; and
- `BackendDeviceId`, an immutable composite identity for one opaque device token scoped by its
  owning `BackendId`; and
- `DeviceClass`, a coarse declarative category with exactly `CPU` and `ACCELERATOR`, in that
  declaration order; and
- `BackendAvailabilitySnapshot`, an immutable point-in-time association from one backend's
  currently reported available device identities to their classes; and
- the sealed, method-free `BackendRequirement` family, with one-component
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
and surrounding whitespace; the records do not trim, normalize, intern, or resolve aliases.
Ordinary record equality therefore compares the exact stored content, and the backend component
prevents equal device tokens from different backends from colliding.
`DeviceClass` is not stored in `BackendDeviceId`; the snapshot supplies the association without
changing either identity or category. The enum declaration order supports stable identity and
diagnostics, not preference, score, priority, capability, or fallback policy.

For example, the following inputs report one accelerator device for the `"cuda"` backend:

```java
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
later changes to a mutable source map cannot change the snapshot. Map iteration order is
unspecified. An empty map means that the supplying context reports no currently available device
for that backend; the backend identity remains part of the snapshot.

The snapshot does not discover devices, register a backend, monitor liveness, refresh itself,
evaluate capability, choose ownership, or guarantee preparation or execution.

The requirement family records exactly one hard target. These inputs construct all three current
variants:

```java
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
matcher, or score. Configuration owns whether a requirement is present. Current internal planning
evaluates it with availability and capability facts; an empty hard-eligible result then fails in
the internal selector with
`IllegalStateException("no hard-eligible backend is available for ownership selection")`.
That internal message is not the ordinary public compile failure contract. `Engine.standard()`
supplies one CPU provider; `Engine.builder()` compiles against the frozen providers of explicitly
registered CPU and/or Metal integrations. Generic plugin registration, other executable backends,
and a public graph-wide Planning surface remain planned.


---

[Previous topic](model-trace.md) · [Next topic](compiler/configuration-values.md)
