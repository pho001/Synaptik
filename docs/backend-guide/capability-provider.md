# Report backend capabilities

## Outcome and status

This guide explains the current backend-neutral contract through which a concrete backend can
report whether it can semantically own one operation occurrence. `OperationCapabilityQuery` and
`BackendCapabilityProvider` are current public planning contracts. The shared backend identities,
supplied availability snapshot, hard-requirement vocabulary, and `BackendIntent` optionality are
also current. The repository does not yet ship a provider implementation or a compiler/planning
consumer. Capability matrices, requirement evaluation, ownership, concrete backend preparation,
registration, and execution remain planned.

A capability is a declarative answer to “can this backend own this work?” It is not a live
executable, a kernel registry, or a route selection.

## Prerequisites and current contracts

A provider implementation depends inward on the current planning, model, and backend-contract
APIs. It needs no runtime, prepare, engine, registry, or discovery service. Its two obligations are:

- `backendId()` always returns one stable non-null backend ownership identity; and
- `supports(query)` rejects null with `NullPointerException("query")` and returns a deterministic
  boolean for an immutable query and unchanged immutable provider configuration.

The query snapshots ordered input and output list membership while retaining the exact immutable
`Operation` and `TensorDescriptor` references. It validates occurrence counts, not descriptor
compatibility or eventual executability.

## Current shared identity, availability, and requirement vocabulary

The current Java API can name an ownership domain, name a device within that domain, and express
the independent CPU-versus-accelerator category vocabulary. It can also associate one backend's
currently reported available devices with those categories:

```java
import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import java.util.Map;

BackendId cuda = new BackendId("cuda");
BackendDeviceId cudaZero = new BackendDeviceId(cuda, "0");
BackendDeviceId metalZero = new BackendDeviceId(new BackendId("metal"), "0");
DeviceClass accelerator = DeviceClass.ACCELERATOR;
BackendAvailabilitySnapshot availability =
        new BackendAvailabilitySnapshot(cuda, Map.of(cudaZero, accelerator));
```

The concrete inputs are backend names `"cuda"` and `"metal"` plus the opaque device token
`"0"`. The two device identities are unequal because the backend component scopes the token.
Both types retain their exact caller-supplied component references. String case and surrounding
whitespace remain significant, and no predefined backend vocabulary or device-number
interpretation exists. `accelerator` is only a coarse category value; it is not stored in either
device identity and does not distinguish a graphics processing unit from another non-CPU compute
device. `availability` reports `cudaZero` as a currently available accelerator for `cuda`. It
cannot contain `metalZero` because every device key must have a backend identity equal to the
snapshot's backend identity. The constructor makes an immutable structural copy of its map while
retaining the exact backend, device, and class references; the copied map has no specified
iteration order. An empty map reports no currently available device for the named backend.

The enum order does not express preference, score, capability, or fallback. The snapshot contains
only supplied facts: it neither discovers devices nor proves registration, liveness, capability,
resource access, preparation success, or executability. A producer decides when to create or
replace a snapshot; the snapshot has no refresh, timestamp, status, or reason field.

The current hard-target vocabulary is separate from both availability and capability:

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

The three inputs respectively target later ownership by an equal `BackendId`, an equal
`BackendDeviceId` and therefore its owning backend, or any later eligible device in the requested
class. The records retain their exact non-null component references. They do not query
`availability`, ask a capability provider, or prove that a target can own particular graph work.
There is no absence sentinel, preference, fallback, combination, evaluator, or score in the
family. Current `BackendIntent` owns optionality without adding any of those meanings:

```java
import io.github.pho001.synaptik.config.compile.BackendIntent;

BackendIntent unconstrained = BackendIntent.unconstrained();
BackendIntent requireCuda = BackendIntent.requiring(exactBackend);
```

`unconstrained` contains no hard target; it does not promise discovery, fallback, availability,
capability, or successful ownership. `requireCuda` retains `exactBackend` by exact reference for
later planning. Neither value queries `availability` or a provider. Later planning owns the
intersection of the optional hard target, supplied availability, and reported capability.

## Lifecycle position

```text
operation + data type + shape + layout
  -> current immutable operation-capability query
  -> current explicitly supplied provider
  -> current boolean semantic-capability answer
  -> planned capability matrix
optional hard target + supplied availability + capable candidates
  -> later planning eligibility
  -> valid ownership candidates
  -> backend-neutral scoring
```

A concrete backend may implement the current provider interface. Later planning will call
explicitly supplied instances for work-specific support, then combine capable candidates with
supplied availability and any hard target. If no eligible candidate remains, the later owning
layer will fail instead of weakening the target; its exception details are not yet defined.
Compile-time plans will retain the selected current `BackendId` value, never the provider object.

## Illustrative current provider

The repository does not ship this class; it is an illustrative implementation of the current
interfaces. It reports support for one FLOAT32 binary-ADD occurrence without selecting a CPU
route:

```java
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

final class IllustrativeCpuCapabilities implements BackendCapabilityProvider {
    private final BackendId backendId = new BackendId("cpu");

    @Override
    public BackendId backendId() {
        return backendId;
    }

    @Override
    public boolean supports(OperationCapabilityQuery query) {
        Objects.requireNonNull(query, "query");
        return query.operation().kind() == BinaryArithmeticKind.ADD
                && query.inputs().stream()
                        .allMatch(input -> input.dataType() == DataType.FLOAT32)
                && query.outputs().stream()
                        .allMatch(output -> output.dataType() == DataType.FLOAT32);
    }
}

TensorDescriptor matrix =
        new TensorDescriptor(DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), false);
Operation add = new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE);
OperationCapabilityQuery query =
        new OperationCapabilityQuery(add, List.of(matrix, matrix), List.of(matrix));

boolean supported = new IllustrativeCpuCapabilities().supports(query);
```

The two exact input descriptors and one output descriptor all describe FLOAT32 Shape `[2, 3]`, so
`supported` is `true`. The result means only that the illustrative provider accepts semantic CPU
ownership of this occurrence. It does not prove CPU availability, evaluate a hard requirement,
select scalar, Vector API, or OpenBLAS execution, prepare the occurrence, or calculate matrix
values. CPU prepare will own any route decision after ownership planning exists.

A current `false` answer carries no rejection reason. A later separate result or trace contract may
provide typed evidence only after its consumers and diagnostic vocabulary are defined.

## Failures and diagnostics

- Invalid graph semantics belong to compiler validation, not capability fallback.
- Backend unavailability should remove or reject that ownership candidate before prepare.
- When the compiler consumer exists, no capable eligible candidate must fail compilation rather
  than defer discovery to runtime.
- Capability evaluation must be deterministic for the supplied immutable compile-time facts.
- A null query must fail with `NullPointerException("query")`; a provider must not reinterpret it
  as unsupported work.

## Validation expectations

Current provider implementations require unit tests for supported and rejected combinations and
architecture tests for dependency direction. Backend-conformance tests comparing declared support
with actual preparation remain necessary once concrete preparation exists; task 0001 adds no
concrete backend behavior to test.

See [Partition scoring](../architecture/partition-scoring.md), [backend
selection](../user-guide/backend-selection.md), and the [backend guide
style](../developer-guide/documentation/backend-guide-style.md).
