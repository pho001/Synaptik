# Report backend capabilities (planned contract)

## Outcome and status

This guide explains how a future concrete backend will report which planned graph work it can
accept. The shared `BackendId` and `BackendDeviceId` identity values and the coarse `DeviceClass`
category are current. Capability, availability, planning, prepare, registration, and concrete
backend contracts remain planned, so the capability sample is conceptual.

A capability is a declarative answer to “can this backend own this work?” It is not a live
executable, a kernel registry, or a route selection.

## Current shared identity and classification vocabulary

The current Java API can name an ownership domain, name a device within that domain, and express
the independent CPU-versus-accelerator category vocabulary:

```java
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;

BackendId cuda = new BackendId("cuda");
BackendDeviceId cudaZero = new BackendDeviceId(cuda, "0");
BackendDeviceId metalZero = new BackendDeviceId(new BackendId("metal"), "0");
DeviceClass accelerator = DeviceClass.ACCELERATOR;
```

The concrete inputs are backend names `"cuda"` and `"metal"` plus the opaque device token
`"0"`. The two device identities are unequal because the backend component scopes the token.
Both types retain their exact caller-supplied component references. String case and surrounding
whitespace remain significant, and no predefined backend vocabulary or device-number
interpretation exists. `accelerator` is only a coarse category value; it is not stored in either
device identity and does not distinguish a graphics processing unit from another non-CPU compute
device. A later availability fact may associate `cudaZero` or `metalZero` with that category, but
no such association type is current. The enum order does not express preference, score,
capability, or fallback. These values do not show that either backend or device is registered,
present, available, capable, or accessible.

## Lifecycle position

```text
operation + data type + shape + layout + availability
  -> capability provider
  -> valid ownership candidates
  -> backend-neutral scoring
```

A concrete backend will implement capability evaluation. Planning will call the shared contract
and compare supported backend identities. Compile-time plans will retain the current `BackendId`
value, never the provider object.

## Conceptual example

Conceptual example: assume a CPU capability provider receives a `FLOAT32` matrix multiplication
with shapes `[2, 3]` and `[3, 4]`. The output shape is `[2, 4]`, containing `2 × 4 = 8` values.
The provider may report CPU ownership as supported based on semantic facts. It must not select
OpenBLAS or a scalar loop; CPU prepare makes that route decision later.

A rejection should carry typed or structured diagnostic evidence explaining the unsupported fact,
such as data type or layout. Exact data-transfer objects (DTOs) remain to be defined.

## Failures and diagnostics

- Invalid graph semantics belong to compiler validation, not capability fallback.
- Backend unavailability should remove or reject that ownership candidate before prepare.
- If no candidate supports required work, compilation must fail rather than defer discovery to
  runtime.
- Capability evaluation must be deterministic for the supplied immutable compile-time facts.

## Validation expectations

Future implementations require unit tests for supported and rejected combinations, architecture
tests for dependency direction, and backend-conformance tests comparing declared support with
actual preparation.

See [Partition scoring](../architecture/partition-scoring.md), [backend
selection](../user-guide/backend-selection.md), and the [backend guide
style](../developer-guide/documentation/backend-guide-style.md).
