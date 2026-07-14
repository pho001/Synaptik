# Report backend capabilities

## Outcome and status

This guide explains the current backend-neutral contract through which a concrete backend can
report whether it can semantically own one operation occurrence. `OperationCapabilityQuery` and
`BackendCapabilityProvider` are current public planning contracts. The shared backend identities,
supplied availability snapshot, hard-requirement vocabulary, and `BackendIntent` optionality are
also current. The repository does not yet ship a provider implementation, compiler consumer, or
public planning consumer. Planning now contains one internal per-query hard-eligibility consumer,
but it is not a public integration API: it combines explicitly supplied providers, matching
availability snapshots, and `BackendIntent` into backend identities only. One further internal
step selects an owner from those identities by optional preferred-class match and provider order.
Reusable/public
capability matrices, public planning orchestration or owner selection, cost scoring, concrete
backend preparation, registration, and execution remain planned.

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
planning. Neither value queries `availability` or a provider. Current internal planning owns the
intersection of the optional hard target, supplied availability, and reported capability.

## Lifecycle position

```text
operation + data type + shape + layout
  -> current immutable operation-capability query
  -> current explicitly supplied provider
  -> current boolean semantic-capability answer
optional hard target + supplied availability + current boolean answers
  -> current internal per-query hard eligibility
  -> immutable provider-ordered BackendId list
optional preferred DeviceClass + that complete candidate list + associated snapshots
  -> current internal preferred-class/provider-order baseline
  -> one BackendId owner
  -> planned public orchestration, partitioning, and cost scoring
```

A concrete backend may implement the current provider interface. Current internal planning first
validates that every supplied provider has exactly one equal-`BackendId` snapshot and vice versa.
It then skips empty snapshots and exact hard-target mismatches before calling each remaining
provider once in provider order. A true answer retains that provider's exact `BackendId`
reference. Snapshot list order does not reorder calls or results.

An exact-device or device-class target consults a snapshot only to prove that a matching device is
currently reported. The provider answer remains backend-level: this step does not claim support
for a particular device, choose one, or retain one. A valid no-match produces an immutable empty
list.

The current package-private selector consumes that list directly. It validates the complete
snapshot input, associates equal backend IDs, permits extra unique snapshots, and treats an empty
matching snapshot as a preference nonmatch. With no preference it returns the first eligible
identity. With a preference it returns the first provider-order match, or falls back to the first
eligible identity when none matches. An empty eligible list fails internally before snapshot
elements are read, and the exact eligibility identity reference is returned. The selector never
re-evaluates capability or hard eligibility and selects no device, route, or kernel. Later public
orchestration may translate the internal failure but must not weaken the hard target. Compile-time
plans will retain `BackendId` values, never provider objects.

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
- Complete provider/snapshot composition errors fail before any `supports` call.
- Empty availability or an exact hard-target mismatch skips the provider before capability is
  queried.
- Empty internal hard eligibility fails before baseline selection can choose a fallback; when a
  compiler consumer exists, it must translate that condition rather than defer discovery to
  runtime.
- Capability evaluation must be deterministic for the supplied immutable compile-time facts.
- A null query must fail with `NullPointerException("query")`; a provider must not reinterpret it
  as unsupported work.

## Validation expectations

Provider implementations require unit tests for supported and rejected combinations and
architecture tests for dependency direction once concrete implementations exist. Backend-
conformance tests comparing declared support with actual preparation remain necessary once
concrete preparation exists. The current internal eligibility and baseline-selection steps change
no concrete backend behavior and therefore add no backend-conformance or integration test
requirement.

See [Partition scoring](../architecture/partition-scoring.md), [backend
selection](../user-guide/backend-selection.md), and the [backend guide
style](../developer-guide/documentation/backend-guide-style.md).
