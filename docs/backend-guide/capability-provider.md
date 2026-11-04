# Report [backend](../glossary.md#backend) capabilities

## Outcome and status

This guide explains the current [backend](../glossary.md#backend)-neutral contract through which a [concrete backend](../glossary.md#concrete-backend) can
report whether it can semantically own one [operation](../glossary.md#operation) [occurrence](../glossary.md#term-occurrence). [`OperationCapabilityQuery`](../glossary.md#operation-capability-query--operationcapabilityquery) and
[`BackendCapabilityProvider`](../glossary.md#backend-capability-provider--backendcapabilityprovider) are current public [planning](../glossary.md#term-planning) contracts. The shared backend [identities](../glossary.md#term-identity-meanings),
supplied availability [snapshot](../glossary.md#term-snapshot), hard-requirement vocabulary, and [`BackendIntent`](../glossary.md#backend-intent--backendintent) optionality are
also current. [CPU](../glossary.md#term-cpu) and [Metal](../glossary.md#term-metal) ship public capability providers. `Engine.standard()` supplies the CPU
provider, while `Engine.builder()` freezes the providers and availability snapshots from
explicitly registered CPU and/or Metal integrations. Ordinary [Engine](../glossary.md#term-engine) [compilation](../glossary.md#compiler-engine-and-execution-stage-names) passes that
inventory to the public cross-[module](../glossary.md#term-module) `GraphCompilationPort` service-provider interface ([SPI](../glossary.md#term-service-provider-interface-spi)).
The port delegates to package-private `GraphCompiler`, which creates one query for every final
[graph](../glossary.md#graph) [node](../glossary.md#node) and calls public `BackendOwnerPlanning.selectOwner(...)`. That Planning collaboration
composes internal per-query hard eligibility and baseline [owner](../glossary.md#term-owner) comparison without exposing their
intermediate. Reusable/public capability [matrices](../glossary.md#term-review-matrix), a public graph-wide Planning workflow, numeric
cost scoring, and generic [plugin](../glossary.md#term-plugin) registration remain planned. Ordinary Engine can execute the
mixed CPU/Metal [ownership](../glossary.md#term-ownership-meanings) selected by the current inventory when every transfer is supported.

A capability is a declarative answer to “can this backend own this work?” It is not a live
executable, a [kernel](../glossary.md#kernel) [registry](../glossary.md#term-registry), or a route selection.

An **occurrence** is one concrete use of an operation, including its ordered inputs and outputs.
A **[descriptor](../glossary.md#term-descriptor)** describes a value's numerical type, [shape](../glossary.md#shape), [layout](../glossary.md#layout) and [gradient](../glossary.md#term-gradient) eligibility;
it is not the stored data itself. Consequently, supporting one [FLOAT32](../glossary.md#term-float32) [matrix multiplication](../glossary.md#matrix-multiplication--matmul)
does not imply supporting every [matrix](../glossary.md#term-matrix-meanings) size, storage mapping, low-[precision](../glossary.md#term-precision) type or [derivative](../glossary.md#term-derivative).

## Prerequisites and current contracts

A provider implementation depends inward on the current [planning](../glossary.md#term-planning), [model](../glossary.md#term-model), and [backend](../glossary.md#backend)-contract
APIs. It needs no [runtime](../glossary.md#term-runtime), [prepare](../glossary.md#prepare), [engine](../glossary.md#term-engine), [registry](../glossary.md#term-registry), or discovery service. Its two obligations are:

- `backendId()` always returns one stable non-null [backend ownership](../glossary.md#backend-ownership) [identity](../glossary.md#term-identity-meanings); and
- `supports(query)` rejects null with `NullPointerException("query")` and returns a deterministic
  boolean for an immutable query and unchanged immutable provider configuration.<span data-glossary-navigation="true"> ([boolean glossary](../glossary.md#data-type--datatype))</span>

The query snapshots ordered input and output list membership while retaining the exact [immutable](../glossary.md#term-immutable)
[`Operation`](../glossary.md#operation) and `TensorDescriptor` references. It validates [occurrence](../glossary.md#term-occurrence) counts, not [descriptor](../glossary.md#term-descriptor)
compatibility or eventual executability.

## Profile-free low-[precision](../glossary.md#term-precision) capability evidence

“Profile-free” means there is no caller-selected alternate numerical contract. **Low precision**
names a stored floating format with fewer bits, not a licence to use arbitrary approximations.
The ledger is a table of actual representative provider answers; it is neither an exhaustive
list of possible inputs nor evidence that every listed computation executed on hardware.

The checked-in
[`low-precision-capability-ledger-v2.tsv`](../../testing/backend-conformance/src/test/resources/low-precision-capability-ledger-v2.tsv)
is the [canonical](../glossary.md#term-canonical-form) profile-free representative [snapshot](../glossary.md#term-snapshot) of actual [CPU](../glossary.md#term-cpu) and [Metal](../glossary.md#term-metal) provider answers.
It records supported and explicit unsupported [FLOAT32](../glossary.md#term-float32) occurrences, plus independently queried
[BFLOAT16](../glossary.md#term-bfloat16) and [FLOAT16](../glossary.md#term-float16) counterparts. The current Metal FLOAT32, BFLOAT16, and FLOAT16 [`EXP`](../glossary.md#term-expx) rows
are independently true for their bounded canonical no-[gradient](../glossary.md#term-gradient) occurrences. The bounded canonical
no-gradient FLOAT32, BFLOAT16, and FLOAT16 [`SIGMOID`](../glossary.md#term-sigmoid) rows are independently true. The 508-row
[`v1 ledger`](../../testing/backend-conformance/src/test/resources/low-precision-capability-ledger-v1.tsv)
is historical two-profile evidence; it is not a current provider query or numerical [certificate](../glossary.md#formal-proof-certificate-and-trusted-assumptions).
`LowPrecisionCapabilityLedgerTest` reconstructs the basis, calls the providers, serializes [UTF-8](../glossary.md#term-utf-8)
TSV with LF endings, enforces stable ordering and unique keys, and reports the first missing,
added, or byte-changed row.

Each [F32](../glossary.md#floating-point-precision-and-rounding) provider portion ends at the `supported` [boolean](../glossary.md#data-type--datatype). Separate
`bf16_current_supported` and `fp16_current_supported` fields are the actual answers from the same
provider for the corresponding mapped [occurrence](../glossary.md#term-occurrence): every F32 [descriptor](../glossary.md#term-descriptor) and F32-typed [scalar](../glossary.md#term-scalar) or
cast [target](../glossary.md#term-target) is replaced by the selected low-precision type while every other field is retained.
Neither answer is inferred from F32 or from the other mapped type. Historical v1 target and
exclusion columns were architecture mapping, not provider output; v2 omits them. No row asserts
route, [runtime](../glossary.md#term-runtime), [device](../glossary.md#term-device), or generated-backward [ownership](../glossary.md#term-ownership-meanings).
The basis is representative rather than an enumeration of every legal [shape](../glossary.md#shape), [layout](../glossary.md#layout), attribute,
or gradient combination; the provider's focused tests remain authoritative for its complete
current predicate.

This mechanism does not redesign [`BackendCapabilityProvider`](../glossary.md#backend-capability-provider--backendcapabilityprovider). The separate [identity](../glossary.md#term-identity-meanings)-allocation
ledger records historical allocations: [Model](../glossary.md#term-model) `DataType.FLOAT16` ordinal 6, CPU generator [schema](../glossary.md#term-schema)
68, Metal type wire 7, program schema 19, [backend](../glossary.md#backend) identities 29, and native [ABI](../glossary.md#term-application-binary-interface-abi) 7. Current CPU
generator schema is 70; current Metal program schema is 20 with a 124-byte header and tuning
identity 39 (30 was the profile-free cutover; 31/32 added FLOAT32/low `EXP`; 33/34 added
FLOAT32/low `SIGMOID`; 35 repaired native/exact-low boundaries; 36 corrected standalone scalar
[admission](../glossary.md#term-admission); 37 added dense-[view](../glossary.md#term-view-meanings) [MATMUL](../glossary.md#matrix-multiplication--matmul); 38 added materializing RESHAPE). Current 39 qualifies
FLOAT32 image-fold overlap for [Conv3d](../glossary.md#conv3d--ncdhw) training; no low-type admission follows from that addition.
Native ABI remains 7. The ledger is not a current-schema certificate.

## Current shared [identity](../glossary.md#term-identity-meanings), availability, and requirement vocabulary

Keep four questions separate: **identity** names an [owner](../glossary.md#term-owner) or [device](../glossary.md#term-device), **availability** reports
what is present, **capability** answers whether an [occurrence](../glossary.md#term-occurrence) is supported, and a **requirement**
constrains which owner may be selected. A preference can influence choice among eligible owners;
it cannot make an ineligible one eligible.

The current Java [API](../glossary.md#term-application-programming-interface-api) can name an [ownership](../glossary.md#term-ownership-meanings) domain, name a device within that domain, and express
the independent [CPU](../glossary.md#term-cpu)-versus-[accelerator](../glossary.md#term-accelerator) category vocabulary. It can also associate one [backend](../glossary.md#backend)'s
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

The concrete inputs are backend names `"cuda"` and `"metal"` plus the [opaque](../glossary.md#term-opaque) device [token](../glossary.md#term-token)
`"0"`. The two device identities are unequal because the backend component scopes the token.
Both types retain their exact caller-supplied component references. String case and surrounding
whitespace remain significant, and no predefined backend vocabulary or device-number
interpretation exists. `accelerator` is only a coarse category value; it is not stored in either
device identity and does not distinguish a [graphics processing unit](../glossary.md#cpu-gpu-host-and-device) from another non-CPU compute
device. `availability` reports `cudaZero` as a currently available accelerator for `cuda`. It
cannot contain `metalZero` because every device key must have a [backend identity](../glossary.md#backend-identity--backendid) equal to the
[snapshot](../glossary.md#term-snapshot)'s backend identity. The constructor makes an [immutable](../glossary.md#term-immutable) structural copy of its map while
retaining the exact backend, device, and class references; the copied map has no specified
iteration order. An empty map reports no currently available device for the named backend.

The enum order does not express preference, score, capability, or fallback. The snapshot contains
only supplied facts: it neither discovers devices nor proves registration, [liveness](../glossary.md#term-liveness), capability,
resource access, preparation success, or executability. A [producer](../glossary.md#term-producer) decides when to create or
replace a snapshot; the snapshot has no refresh, timestamp, status, or reason field.

The current hard-[target](../glossary.md#term-target) vocabulary is separate from both availability and capability:

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

The three inputs respectively target later ownership by an equal [`BackendId`](../glossary.md#backend-identity--backendid), an equal
[`BackendDeviceId`](../glossary.md#backend-device-identity--backenddeviceid) and therefore its owning backend, or any later eligible device in the requested
class. The records retain their exact non-null component references. They do not query
`availability`, ask a capability provider, or prove that a target can own particular [graph](../glossary.md#graph) work.
There is no absence sentinel, preference, fallback, combination, evaluator, or score in the
family. Current [`BackendIntent`](../glossary.md#backend-intent--backendintent) owns optionality without adding any of those meanings:

```java
import io.github.pho001.synaptik.config.compile.BackendIntent;

BackendIntent unconstrained = BackendIntent.unconstrained();
BackendIntent requireCuda = BackendIntent.requiring(exactBackend);
```

`unconstrained` contains no hard target; it does not promise discovery, fallback, availability,
capability, or successful ownership. `requireCuda` retains `exactBackend` by exact reference for
[planning](../glossary.md#term-planning). Neither value queries `availability` or a provider. Current internal planning owns the
intersection of the optional hard target, supplied availability, and reported capability.

## [Lifecycle](../glossary.md#lifecycle) position

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
complete per-occurrence owners + closed graph
  -> Compiler-owned complete owner map
  -> current public stateless maximal same-owner partitioning
  -> current immutable partition and logical-memory recipes
  -> current immutable CompileArtifacts
  -> current owner-bound ordinary or advanced Engine compile handle
```

The ordinary and advanced [Engine](../glossary.md#term-engine) [compile](../glossary.md#compile) workflows enter that complete [Compiler](../glossary.md#term-compiler) pipeline through
the public cross-[module](../glossary.md#term-module) `GraphCompilationPort` integration [SPI](../glossary.md#term-service-provider-interface-spi); the port is an entry boundary,
not a stage after [artifact](../glossary.md#term-artifact) construction. The [CPU](../glossary.md#term-cpu) [backend](../glossary.md#backend) implements the current provider
interface. Current internal [planning](../glossary.md#term-planning) first
validates that every supplied provider has exactly one equal-[`BackendId`](../glossary.md#backend-identity--backendid) [snapshot](../glossary.md#term-snapshot) and vice versa.
It then skips empty snapshots and exact hard-[target](../glossary.md#term-target) mismatches before calling each remaining
provider once in provider order. A true answer retains that provider's exact `BackendId`
reference. Snapshot list order does not reorder calls or results.

An exact-[device](../glossary.md#term-device) or device-class target consults a snapshot only to prove that a matching device is
currently reported. The provider answer remains backend-level: this step does not claim support
for a particular device, choose one, or retain one. A valid no-match produces an [immutable](../glossary.md#term-immutable) empty
list.

The package-private selector consumes that list directly. It validates the complete
snapshot input, associates equal backend IDs, permits extra unique snapshots, and treats an empty
matching snapshot as a preference nonmatch. With no preference it returns the first eligible
[identity](../glossary.md#term-identity-meanings). With a preference it returns the first provider-order match, or falls back to the first
eligible identity when none matches. An empty eligible list fails internally before snapshot
[elements](../glossary.md#term-element) are read, and the exact eligibility identity reference is returned. The selector never
re-evaluates capability or hard eligibility and selects no device, route, or [kernel](../glossary.md#kernel).

Public `BackendOwnerPlanning.selectOwner(...)` validates the query, intent, providers, snapshots,
and scoring config in declaration order, then calls eligibility and selection once each. Its
result is the exact selected eligibility identity. Package-private `GraphCompiler` invokes it once
per final [graph](../glossary.md#graph) [node](../glossary.md#node), stops at the first failure, and retains only `BackendId` [ownership](../glossary.md#term-ownership-meanings) in [compile
artifacts](../glossary.md#compile-artifacts). Public `GraphCompilationPort` exposes that complete Compiler pipeline only as a
cross-module SPI, while the ordinary `Engine.compile(...)` overloads and
`AdvancedEngine.compile(...)` are the current public lifecycle facades that consume it. There is
no standalone ordinary compiler [facade](../glossary.md#term-facade) or generic provider-registration workflow. The terminal
no-hard-eligible failure receives node context from Compiler; other composition failures and
provider-thrown [runtime](../glossary.md#term-runtime) exceptions propagate unchanged. Neither providers nor snapshots are
retained in the artifacts.

## Illustrative provider

The repository ships the substantially broader `CpuCapabilityProvider`, whose current [operation](../glossary.md#operation)
coverage is documented in the [CPU backend guide](cpu-backend.md)<span data-glossary-navigation="true"> ([backend glossary](../glossary.md#backend))</span>. The smaller class below is
deliberately illustrative rather than a copy of that production provider. It demonstrates the
shared interfaces by reporting support for one [FLOAT32](../glossary.md#term-float32) [binary](../glossary.md#term-binary-descriptor-and-metadata)-ADD [occurrence](../glossary.md#term-occurrence) without selecting a
[CPU](../glossary.md#term-cpu) route:

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

The two exact input [descriptors](../glossary.md#term-descriptor) and one output descriptor all describe FLOAT32 [Shape](../glossary.md#shape) `[2, 3]`, so
`supported` is `true`. The result means only that the illustrative provider accepts semantic CPU
[ownership](../glossary.md#term-ownership-meanings) of this occurrence. It does not prove CPU availability, evaluate a hard requirement,
select [scalar](../glossary.md#term-scalar), [Vector API](../glossary.md#term-vector-api), or [OpenBLAS](../glossary.md#term-openblas) execution, [prepare](../glossary.md#prepare) the occurrence, or calculate [matrix](../glossary.md#term-matrix-meanings)
values. CPU prepare will own any route decision during later preparation.

A current `false` answer carries no rejection reason. A later separate result or [trace](../glossary.md#trace) contract may
provide typed evidence only after its consumers and diagnostic vocabulary are defined.

## Failures and diagnostics

- Invalid [graph](../glossary.md#graph) semantics belong to [compiler](../glossary.md#term-compiler) [validation](../glossary.md#term-validation), not capability fallback.
- Complete provider/[snapshot](../glossary.md#term-snapshot) composition errors fail before any `supports` call.
- Empty availability or an exact hard-target mismatch skips the provider before capability is
  queried.<span data-glossary-navigation="true"> ([target glossary](../glossary.md#term-target))</span>
- Empty internal hard eligibility fails before baseline selection can choose a fallback; current
  Compiler adds node occurrence context and does not defer discovery to runtime.<span data-glossary-navigation="true"> ([node glossary](../glossary.md#node); [occurrence glossary](../glossary.md#term-occurrence); [runtime glossary](../glossary.md#term-runtime))</span>
- Capability [evaluation](../glossary.md#term-evaluation) must be deterministic for the supplied [immutable](../glossary.md#term-immutable) [compile](../glossary.md#compile)-time facts.
- A null query must fail with `NullPointerException("query")`; a provider must not reinterpret it
  as unsupported work.

## [Validation](../glossary.md#term-validation) expectations

[Conformance](../glossary.md#term-conformance) means agreement between an implementation and the contract it claims to satisfy.
For example, checking that a provider says “yes” is different from preparing that [occurrence](../glossary.md#term-occurrence),
running it on current inputs, and comparing the result with an independently calculated answer.

Provider implementations require unit tests for supported and rejected combinations and
architecture tests for [dependency direction](../glossary.md#java-platform-and-build-tools). The production [CPU](../glossary.md#term-cpu) provider has focused public-
[shape](../glossary.md#shape) and support-[matrix](../glossary.md#term-review-matrix) tests, while [Engine](../glossary.md#term-engine) [lifecycle](../glossary.md#lifecycle) tests cover explicit [compile](../glossary.md#compile)-time
composition. [Backend](../glossary.md#backend)-conformance tests must compare declared support with actual preparation for
the concrete routes they exercise. This documentation correction changes no provider,
preparation, or execution behavior and therefore adds no new backend-conformance or [integration
test](../glossary.md#term-integration-test) requirement.

See [Partition scoring](../architecture/partition-scoring.md)<span data-glossary-navigation="true"> ([Partition glossary](../glossary.md#partition))</span>, [backend
selection](../user-guide/backend-selection.md), and the [backend guide
style](../developer-guide/documentation/backend-guide-style.md).

## Profile-free capability

[`OperationCapabilityQuery`](../glossary.md#operation-capability-query--operationcapabilityquery) carries complete [occurrence](../glossary.md#term-occurrence) facts but no numerical selector. [Model](../glossary.md#term-model)
defines one family/[dtype](../glossary.md#data-type--datatype) semantic contract. A provider answers the concrete occurrence and must
not infer support from semantic reachability or [`DeviceClass`](../glossary.md#device-class--deviceclass); return `false` for unsupported
[descriptors](../glossary.md#term-descriptor) or attributes. The v1 [`ACCELERATOR`](../glossary.md#term-accelerator) answers remain historical cutover evidence, not a
permanent ceiling on the current provider or its v2 ledger.

The current [Metal](../glossary.md#term-metal) provider admits the exact common [unary](../glossary.md#term-unary), affine, [canonicalization](../glossary.md#occurrence-identity-and-canonical-form), indexing,
classification, [BOOL](../glossary.md#data-type--datatype), replacement, movement, non-overlapping fold/window, ordering/top-K/numeric
arg-[extrema](../glossary.md#saturation-pole-and-extremum), maximum-pool, initial-state, and promoted integral [MATMUL](../glossary.md#matrix-multiplication--matmul) rows.
All 49 ordered casts, [FLOAT64](../glossary.md#term-float64)/[FLOAT32](../glossary.md#term-float32)/[BFLOAT16](../glossary.md#term-bfloat16)/[FLOAT16](../glossary.md#term-float16) classification, [scalar](../glossary.md#term-scalar) and right-aligned
BOOL logic, the 14 floating `WHERE` [signatures](../glossary.md#term-signature) other than direct BFLOAT16/FLOAT16 mixing, exact
seven-[carrier](../glossary.md#term-storage-carrier) movement, [INT32](../glossary.md#data-type--datatype)/[INT64](../glossary.md#data-type--datatype) index parity, `FOLD_AXIS` over all six numeric carriers, and
2D/3D unfold/fold over all four floating carriers are admitted within their exact predicates.
Replacement scatter remains `NONE`. FOLD_AXIS and FLOAT64/BFLOAT16/FLOAT16 image folds remain
non-overlapping; FLOAT32 FOLD2D/FOLD3D additionally accumulate overlap with one output writer.
Every admitted [Shape](../glossary.md#shape)/[layout](../glossary.md#layout) is fully static and checked. This closes qualified static FLOAT32
[Conv3d](../glossary.md#conv3d--ncdhw) training through [Compiler](../glossary.md#term-compiler)'s shared grouped window/MATMUL/reduction algebra, not [backend](../glossary.md#backend)
[autograd](../glossary.md#autograd) or Metal [Scan](../glossary.md#term-cumulative-scan). See the [production matrix](production-support.md#metal-production-target)<span data-glossary-navigation="true"> ([matrix glossary](../glossary.md#term-matrix-meanings))</span>.
`SORT`, `ARGSORT`, and positive-K `TOP_K` admit all seven carriers; `ARG_MAX` and `ARG_MIN` admit
the six numeric carriers over [canonical](../glossary.md#term-canonical-form) dense ranks `1..16` with unsigned-32-bit-bounded geometry.

Metal also admits the documented FLOAT32 arithmetic, extrema, scalar, reduction,
scan, [MSE](../glossary.md#mean-squared-error--mse), general MATMUL, average-[pooling](../glossary.md#term-pooling), [convolution](../glossary.md#term-channels-first-convolution-layers), explicit-state [dropout](../glossary.md#dropout--inverted-dropout), and exact
[rank-one](../glossary.md#axis-rank-scalar-vector-and-matrix) no-[gradient](../glossary.md#term-gradient) FLOAT32 `L1_NORM` and `SCATTER_ADD`, plus [singleton](../glossary.md#term-singleton) [`VARIANCE`](../glossary.md#term-variance). Every
supported frozen-baseline [homogeneous](../glossary.md#term-homogeneous) FLOAT32 occurrence retains its independently queried
BFLOAT16 and FLOAT16 counterparts where declared; a new FLOAT32 answer does not grant either low
answer. Current post-cutover [`EXP`](../glossary.md#term-expx) support admits separate homogeneous FLOAT32, BFLOAT16, and
FLOAT16 answers for one canonical, positive-static-[rank](../glossary.md#term-rank) `1..16`, no-gradient input and
same-type/equal-Shape canonical output with no attributes and checked dtype-specific byte geometry.
FLOAT32 uses direct [MPSGraph](../glossary.md#term-mpsgraph); each low type uses its own typed custom step. FLOAT64,
gradient-bearing, scalar-rank, noncanonical, mixed-low, and over-limit Metal `EXP` remain false.
Direct BFLOAT16/FLOAT16 mixed-low execution remains unsupported. ScatterAdd uses
[axis](../glossary.md#term-axis) zero,
canonical base/index/update/output roles, and a materialized INT32/INT64 index feed; the compiler
explicitly canonicalizes its generated zero base, and the complete range scan precedes [dispatch](../glossary.md#term-dispatch)
and mutation. Variance requires no-gradient input `[1]`, axis `[0]`, correction zero, and canonical
scalar or retained `[1]` output. Metal admits no-gradient INT32/INT64 MATMUL pairs with
INT64-dominant promotion and modular result arithmetic. Every [partition](../glossary.md#partition) containing BFLOAT16 or
FLOAT16 values and the three exact rank-one FLOAT32 special occurrences use one fixed
`CUSTOM_PROGRAM` route. A low feed explicitly cast to FLOAT32 may then use this FLOAT32 `EXP` as
an internal MPSGraph boundary step; the whole partition remains custom. Low-valued `EXP` instead
uses a qualified typed custom step without an implicit cast. Low-[precision](../glossary.md#term-precision) partitions expose no
MPSGraph, [classic-MPS](../glossary.md#mps-mpsgraph-and-openblas), [MPP](../glossary.md#term-mpp), [CPU](../glossary.md#term-cpu), retry,
or fallback whole-partition route.
Capability distinguishes logical affine layouts from their independently authenticated physical
spans and rejects dynamic/empty geometry, zero or negative external [strides](../glossary.md#term-stride), physical [storage overlap](../glossary.md#aliasing-and-arena), every other
additive scatter, reduction-dependent adjoints, and every other unlisted occurrence before route
selection.

[ABI](../glossary.md#term-application-binary-interface-abi) 7 exposes thirteen functions and consumes one bounded [schema](../glossary.md#term-schema)-20 route-bearing program image.
[Operation](../glossary.md#operation) wires `1..115`, attribute wires `0..41`, route wires `1..3`, and type wires `1..7` are
structural vocabulary only. The custom-program extension authenticates compact materialized slots,
deterministic generated [pointwise](../glossary.md#term-pointwise) units, and qualified MATMUL/[Conv2d](../glossary.md#convolution--conv2d) anchor epilogues
without [widening](../glossary.md#term-widening) capability. Version-thirty-two workload, policy, candidate, compatibility,
route, and [codec](../glossary.md#term-codec) [identities](../glossary.md#term-identity-meanings) authenticate that meaning; prior identities 30 and 31 and other stale values
[fail closed](../glossary.md#admission-preflight-and-fail-closed-validation). Current production capability is `87/28` and structural execution is `102/13` over
operation kinds; the frozen cutover counts were `86/29` and `101/14`.
