<a id="model-checkpoint-api"></a>
# Whole-[model](../../glossary.md#term-model) [checkpoint](../../glossary.md#term-checkpoint) save and load

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphCheckpoint.java)

A **whole-model** file includes a supported layer definition as well as state values. A
**values-only** file supplies state for a compatible model the caller has already recreated.
The state [schema](../../glossary.md#term-schema) describes paths, roles, types and [shapes](../../glossary.md#shape); it is not the [parameter](../../glossary.md#term-parameter) values
themselves. Neither workflow stores the [optimizer](../../glossary.md#term-optimizer)'s memory of earlier updates.

<a id="purpose-and-mental-model"></a>
## Purpose and [mental model](../../glossary.md#term-mental-model)

`ModelCheckpoint` supports two recovery workflows: save a supported whole [model](../../glossary.md#term-model) and automatically
load a `DeclarativeModel`, or [capture](../../glossary.md#term-capture-meanings) values and restore them into a caller-recreated fully bound
neural-network ([NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)) [module](../../glossary.md#term-module). An envelope is an [immutable](../../glossary.md#term-immutable) in-memory collection
of named byte payloads; [canonical](../../glossary.md#term-canonical-form) bytes encode logical [elements](../../glossary.md#term-element) in [row-major](../../glossary.md#term-row-major) order, [big-endian](../../glossary.md#text-encoding-and-file-boundaries).
The [architecture contract](../../architecture/contracts/extensions-training.md#retained-nn-checkpoint-apis)<span data-glossary-navigation="true"> ([architecture contract glossary](../../glossary.md#architecture-contract))</span>
owns this [NN model](../../glossary.md#model-module-buffer-and-scan-distinctions)-only boundary. Import the three public types from
`io.github.pho001.synaptik.nn.checkpoint` and declare an explicit `:extensions:nn` [dependency](../../glossary.md#term-dependency);
[Checkpoint](../../glossary.md#term-checkpoint) alone no longer exports them. [Artifact](../../glossary.md#term-artifact) imports and the `:extensions:checkpoint`
project [identity](../../glossary.md#term-identity-meanings) are unchanged. This [API](../../glossary.md#term-application-programming-interface-api) retains `model.v1`/`whole-model.v1` readers and flexible
[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)/[rank](../../glossary.md#term-rank) semantics; the separate [generic graph API](graph.md)<span data-glossary-navigation="true"> ([graph glossary](../../glossary.md#graph))</span> has static [signatures](../../glossary.md#term-signature).
It does not persist a [Java module](../../glossary.md#model-module-buffer-and-scan-distinctions) class, [optimizer](../../glossary.md#term-optimizer), or training
continuation.

| Workflow | Stored kind | What load requires and returns |
|---|---|---|
| Whole model | `whole-model.v1` / [FP8](../../glossary.md#term-fp8) `whole-model.v2` | No [target](../../glossary.md#term-target) or configuration [digest](../../glossary.md#term-digest); returns an independent, value-loaded [evaluation](../../glossary.md#term-evaluation) `DeclarativeModel`. |
| Values only | `model.v1` / FP8 `model.v2` | Caller-recreated fully bound target and expected digests; installs values into that target. |

Choose whole-model recovery for the closed layer [matrix](../../glossary.md#term-matrix-meanings) below; values-only recovery remains the
advanced boundary for other NN state. The [opaque](../../glossary.md#term-opaque) artifact API interprets neither kind.

<a id="whole-model-save-and-automatic-load"></a>
## Whole-[model](../../glossary.md#term-model) save and automatic load

The following ten static overloads use existing `ModelCheckpoint`; file operations declare
`IOException`, and envelope operations declare its subtype `ArtifactFormatException`:

```text
void save(Path, Module, Engine)
void save(Path, Module, Engine, ArtifactLimits)
void save(Path, Module, Engine, InputArtifactFingerprint, ArtifactLimits)
DeclarativeModel load(Path)
DeclarativeModel load(Path, ArtifactLimits)
DeclarativeModel load(Path, InputArtifactFingerprint, ArtifactLimits)
ArtifactEnvelope captureWholeModel(Module, Engine, ArtifactLimits)
ArtifactEnvelope captureWholeModel(Module, Engine, InputArtifactFingerprint, ArtifactLimits)
DeclarativeModel loadWholeModel(ArtifactEnvelope, ArtifactLimits)
DeclarativeModel loadWholeModel(ArtifactEnvelope, InputArtifactFingerprint, ArtifactLimits)
```

Although the source [parameter](../../glossary.md#term-parameter) is [`Module`](../../glossary.md#term-module), only ordinary supported `Sequential` and final
`DeclarativeModel` roots are recognized. Custom roots/children and unbound state fail before
[Engine](../../glossary.md#term-engine) or filesystem work. The supported NN0026 [matrix](../../glossary.md#term-matrix-meanings) is nested sequences, [Linear](../../glossary.md#term-linear-meanings), affine
LayerNorm, [Embedding](../../glossary.md#embedding), Conv1d/2d/3d, and RELU/[TANH](../../glossary.md#term-tanh)/[SIGMOID](../../glossary.md#term-sigmoid). Parameters support [FLOAT64](../../glossary.md#term-float64), [FLOAT32](../../glossary.md#term-float32),
[BFLOAT16](../../glossary.md#term-bfloat16), [FLOAT16](../../glossary.md#term-float16), FLOAT8_E4M3 and FLOAT8_E5M2. Each layer retains its complete configuration, guards and input semantics:
Embedding accepts [INT32](../../glossary.md#data-type--datatype)/[INT64](../../glossary.md#data-type--datatype) indices; compatible leading [Dimensions](../../glossary.md#dimension) are not frozen to an example
[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters). See the [layer compatibility guide](../../user-guide/declarative-models.md#compatibility-sharing-and-failures).
Recurrence, BatchNorm, [Dropout](../../glossary.md#dropout--inverted-dropout), branches, custom I/O and arbitrary `Model.define` callbacks are not
captured. No [topology](../../glossary.md#term-topology) is guessed from values or [Shapes](../../glossary.md#shape).

[FP8](../../glossary.md#term-fp8) selects new formats conditionally, not for every save. Values-only state containing either
FP8 type uses `model.v2`, eight [ASCII](../../glossary.md#term-ascii) [schema](../../glossary.md#term-schema) [magic bytes](../../glossary.md#term-magic-bytes) `SYNMOD02`, and [big-endian](../../glossary.md#text-encoding-and-file-boundaries) i32
version 2. Definition [metadata](../../glossary.md#term-metadata) requiring FP8 uses `SYNDEF02` and i32 version 2, and its complete
composition is `whole-model.v2` with matching `model.v2` values under `model/`.
Each v2 [dtype](../../glossary.md#data-type--datatype) domain appends 8=FLOAT8_E4M3 and 9=FLOAT8_E5M2; old IDs 1..7 retain their
meanings. Raw FP8 payloads use one byte per [element](../../glossary.md#term-element), preserving every bit without scales.
Non-FP8 sources retain the original v1 format selection and byte-identical schemas/payloads:
`model.v1` / `SYNMOD01`, `SYNDEF01`, and `whole-model.v1`. Matching kind/header/version
[validation](../../glossary.md#term-validation) rejects incompatible domains rather than reinterpreting them. These [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) formats are
separate from the generic [graph](../../glossary.md#graph) v7 format.

`whole-model.v1` contains [canonical](../../glossary.md#term-canonical-form) `SYNDEF01` version-one definition bytes at `definition/schema`
and unchanged values-only entries under `model/`. Configuration [SHA-256](../../glossary.md#term-sha-256) is automatically computed
from those exact definition bytes. Complete ordered [descriptors](../../glossary.md#term-descriptor) and raw bytes within every [alias](../../glossary.md#aliasing-and-arena)
group are independently checked before [publication](../../glossary.md#term-publication). Aliases mean the same exact parameter [Tensor](../../glossary.md#tensor),
not equal values or overlapping storage; re-saving observes current joins and splits. Neither [hashes](../../glossary.md#term-hash)
nor a prior inspection authenticate the [artifact](../../glossary.md#term-artifact) or authorize later validation. Existing values-only
inspection/byte-access methods accept `model.v1` and `model.v2`, not an outer whole-model kind.

<a id="expected-input-identity"></a>
### Expected input [identity](../../glossary.md#term-identity-meanings)

The retained public type, `nn.checkpoint.InputArtifactFingerprint`, represents an input
[artifact](../../glossary.md#term-artifact)'s [SHA-256](../../glossary.md#term-sha-256) identity already verified by the application. Its constructor requires exactly
32 non-null bytes; construction and `bytes()` defensively copy, and equality/[hash code](../../glossary.md#term-hash-code) use all bytes.
[Checkpoint](../../glossary.md#term-checkpoint) neither reads nor verifies that external artifact.

Overloads without this value save absent input identity and require absence on load. Stored
presence is **not** silently accepted by default load. Explicit overloads require a non-null value;
load requires both stored presence and exact bytes. Absence differs from every present [digest](../../glossary.md#term-digest),
including an all-zero digest. A mismatch raises `ArtifactFormatException`; supply the application-verified expected
identity through the typed overload rather than treating stored bytes as verified input.

<a id="bounds-lifetime-and-failures"></a>
### Bounds, [lifetime](../../glossary.md#term-lifetime) and failures

Short file calls use exactly
`ArtifactLimits(268435456L, 4194304, 16384, 64, 67108864, 268435456L)`: 256 MiB complete file,
4 MiB [manifest](../../glossary.md#term-manifest), 16384 entries, 64-byte [ASCII](../../glossary.md#term-ascii) kind/key, 64 MiB single [payload](../../glossary.md#artifact-envelope-and-atomic-publication), and 256 MiB aggregate
payload. One MiB is 1048576 bytes. The definition ceiling remains 1048576 bytes even with advanced
limits; [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) bounds remain 4096 occurrences and depth 64, counting the root at depth one. These
limits bound encoded bytes, not defensive copies, workspace or peak [heap](../../glossary.md#term-heap); no [backend](../../glossary.md#backend) [rank](../../glossary.md#term-rank)-16
restriction is added. [Capture](../../glossary.md#term-capture-meanings) checks outer bounds before [Engine](../../glossary.md#term-engine) [admission](../../glossary.md#term-admission) or bulk value copies.

Keep source [bindings](../../glossary.md#term-binding) and storage live, accessible and [quiescent](../../glossary.md#term-quiescent) for the entire save/capture.
Description and public values capture may each read state; there is no combined [snapshot](../../glossary.md#term-snapshot), lock or
[transaction](../../glossary.md#term-transaction). Empty state makes no Engine call, but still requires a non-null Engine. Semantic
empty/[identity](../../glossary.md#term-identity-meanings) models do not imply generic zero-[operation](../../glossary.md#operation) Engine admission.

Load needs no Engine or caller [Arena](../../glossary.md#term-arena). It returns only after complete [validation](../../glossary.md#term-validation) and installation
into a private NN candidate: independent heap-backed, [evaluation](../../glossary.md#term-evaluation)-mode state, with [gradient](../../glossary.md#term-gradient)-eligible
[parameters](../../glossary.md#term-parameter). It performs no [autograd](../../glossary.md#autograd), [compilation](../../glossary.md#compiler-engine-and-execution-stage-names), [RNG](../../glossary.md#graph-rng-state--graphrngstate) initialization or native allocation and retains
no source/file handle. The values survive source Arena/Engine closure and file deletion. To execute,
construct a forward expression and use a fresh caller Engine with an admitted route. No backend,
[device](../../glossary.md#term-device) state or compiled/prepared plan is saved. Evaluation mode is not a no-gradient guarantee.

Required nulls raise `NullPointerException`; unsupported source/configuration or capture bounds raise
`IllegalArgumentException`; unbound state/Engine contract failures raise `IllegalStateException`;
checked capture [overflow](../../glossary.md#term-overflow) raises `ArithmeticException`. Engine unchecked failures propagate.
Malformed, unsupported, incoherent or over-limit stored data and expected-input mismatches raise
`ArtifactFormatException`, with [owner](../../glossary.md#term-owner) validation causes retained. [Tensor](../../glossary.md#tensor) identifier exhaustion
remains `IllegalStateException`. File access/[publication](../../glossary.md#term-publication) failures remain `IOException`. Loading a
values-only file automatically reports a missing definition and directs callers to [target](../../glossary.md#term-target)-load.
Failures return no partial [model](../../glossary.md#term-model). Save retains the [artifact atomic-live-publication policy](artifacts.md#lifecycle-and-invariants)<span data-glossary-navigation="true"> ([artifact glossary](../../glossary.md#term-artifact); [atomic glossary](../../glossary.md#resource-transaction-and-rollback))</span>,
not portable power-[loss](../../glossary.md#loss) [durability](../../glossary.md#term-durability) or a cross-owner transaction.

For the concrete save → source closure → automatic load → fresh-Engine workflow, use the
[running-models example](../../user-guide/running-models.md#load-a-saved-whole-model).


---

[Next topic](values.md)
