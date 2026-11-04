<a id="model-checkpoint-api"></a>
# Whole-model checkpoint save and load

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphCheckpoint.java)

<a id="purpose-and-mental-model"></a>
## Purpose and mental model

`ModelCheckpoint` supports two recovery workflows: save a supported whole model and automatically
load a `DeclarativeModel`, or capture values and restore them into a caller-recreated fully bound
neural-network (NN) module. An envelope is an immutable in-memory collection
of named byte payloads; canonical bytes encode logical elements in row-major order, big-endian.
The [architecture contract](../../architecture/contracts/extensions-training.md#retained-nn-checkpoint-apis)
owns this NN model-only boundary. Import the three public types from
`io.github.pho001.synaptik.nn.checkpoint` and declare an explicit `:extensions:nn` dependency;
Checkpoint alone no longer exports them. Artifact imports and the `:extensions:checkpoint`
project identity are unchanged. This API retains `model.v1`/`whole-model.v1` readers and flexible
batch/rank semantics; the separate [generic graph API](graph.md) has static signatures.
It does not persist a Java module class, optimizer, or training
continuation.

| Workflow | Stored kind | What load requires and returns |
|---|---|---|
| Whole model | `whole-model.v1` / FP8 `whole-model.v2` | No target or configuration digest; returns an independent, value-loaded evaluation `DeclarativeModel`. |
| Values only | `model.v1` / FP8 `model.v2` | Caller-recreated fully bound target and expected digests; installs values into that target. |

Choose whole-model recovery for the closed layer matrix below; values-only recovery remains the
advanced boundary for other NN state. The opaque artifact API interprets neither kind.

<a id="whole-model-save-and-automatic-load"></a>
## Whole-model save and automatic load

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

Although the source parameter is `Module`, only ordinary supported `Sequential` and final
`DeclarativeModel` roots are recognized. Custom roots/children and unbound state fail before
Engine or filesystem work. The supported NN0026 matrix is nested sequences, Linear, affine
LayerNorm, Embedding, Conv1d/2d/3d, and RELU/TANH/SIGMOID. Parameters support FLOAT64, FLOAT32,
BFLOAT16, FLOAT16, FLOAT8_E4M3 and FLOAT8_E5M2. Each layer retains its complete configuration, guards and input semantics:
Embedding accepts INT32/INT64 indices; compatible leading Dimensions are not frozen to an example
batch. See the [layer compatibility guide](../../user-guide/declarative-models.md#compatibility-sharing-and-failures).
Recurrence, BatchNorm, Dropout, branches, custom I/O and arbitrary `Model.define` callbacks are not
captured. No topology is guessed from values or Shapes.

FP8 selects new formats conditionally, not for every save. Values-only state containing either
FP8 type uses `model.v2`, eight ASCII schema magic bytes `SYNMOD02`, and big-endian i32
version 2. Definition metadata requiring FP8 uses `SYNDEF02` and i32 version 2, and its complete
composition is `whole-model.v2` with matching `model.v2` values under `model/`.
Each v2 dtype domain appends 8=FLOAT8_E4M3 and 9=FLOAT8_E5M2; old IDs 1..7 retain their
meanings. Raw FP8 payloads use one byte per element, preserving every bit without scales.
Non-FP8 sources retain the original v1 format selection and byte-identical schemas/payloads:
`model.v1` / `SYNMOD01`, `SYNDEF01`, and `whole-model.v1`. Matching kind/header/version
validation rejects incompatible domains rather than reinterpreting them. These NN formats are
separate from the generic graph v7 format.

`whole-model.v1` contains canonical `SYNDEF01` version-one definition bytes at `definition/schema`
and unchanged values-only entries under `model/`. Configuration SHA-256 is automatically computed
from those exact definition bytes. Complete ordered descriptors and raw bytes within every alias
group are independently checked before publication. Aliases mean the same exact parameter Tensor,
not equal values or overlapping storage; re-saving observes current joins and splits. Neither hashes
nor a prior inspection authenticate the artifact or authorize later validation. Existing values-only
inspection/byte-access methods accept `model.v1` and `model.v2`, not an outer whole-model kind.

<a id="expected-input-identity"></a>
### Expected input identity

The retained public type, `nn.checkpoint.InputArtifactFingerprint`, represents an input
artifact's SHA-256 identity already verified by the application. Its constructor requires exactly
32 non-null bytes; construction and `bytes()` defensively copy, and equality/hash code use all bytes.
Checkpoint neither reads nor verifies that external artifact.

Overloads without this value save absent input identity and require absence on load. Stored
presence is **not** silently accepted by default load. Explicit overloads require a non-null value;
load requires both stored presence and exact bytes. Absence differs from every present digest,
including an all-zero digest. A mismatch raises `ArtifactFormatException`; supply the application-verified expected
identity through the typed overload rather than treating stored bytes as verified input.

<a id="bounds-lifetime-and-failures"></a>
### Bounds, lifetime and failures

Short file calls use exactly
`ArtifactLimits(268435456L, 4194304, 16384, 64, 67108864, 268435456L)`: 256 MiB complete file,
4 MiB manifest, 16384 entries, 64-byte ASCII kind/key, 64 MiB single payload, and 256 MiB aggregate
payload. One MiB is 1048576 bytes. The definition ceiling remains 1048576 bytes even with advanced
limits; NN bounds remain 4096 occurrences and depth 64, counting the root at depth one. These
limits bound encoded bytes, not defensive copies, workspace or peak heap; no backend rank-16
restriction is added. Capture checks outer bounds before Engine admission or bulk value copies.

Keep source bindings and storage live, accessible and quiescent for the entire save/capture.
Description and public values capture may each read state; there is no combined snapshot, lock or
transaction. Empty state makes no Engine call, but still requires a non-null Engine. Semantic
empty/identity models do not imply generic zero-operation Engine admission.

Load needs no Engine or caller Arena. It returns only after complete validation and installation
into a private NN candidate: independent heap-backed, evaluation-mode state, with gradient-eligible
parameters. It performs no autograd, compilation, RNG initialization or native allocation and retains
no source/file handle. The values survive source Arena/Engine closure and file deletion. To execute,
construct a forward expression and use a fresh caller Engine with an admitted route. No backend,
device state or compiled/prepared plan is saved. Evaluation mode is not a no-gradient guarantee.

Required nulls raise `NullPointerException`; unsupported source/configuration or capture bounds raise
`IllegalArgumentException`; unbound state/Engine contract failures raise `IllegalStateException`;
checked capture overflow raises `ArithmeticException`. Engine unchecked failures propagate.
Malformed, unsupported, incoherent or over-limit stored data and expected-input mismatches raise
`ArtifactFormatException`, with owner validation causes retained. Tensor identifier exhaustion
remains `IllegalStateException`. File access/publication failures remain `IOException`. Loading a
values-only file automatically reports a missing definition and directs callers to target-load.
Failures return no partial model. Save retains the [artifact atomic-live-publication policy](artifacts.md#lifecycle-and-invariants),
not portable power-loss durability or a cross-owner transaction.

For the concrete save → source closure → automatic load → fresh-Engine workflow, use the
[running-models example](../../user-guide/running-models.md#load-a-saved-whole-model).


---

[Next topic](values.md)
