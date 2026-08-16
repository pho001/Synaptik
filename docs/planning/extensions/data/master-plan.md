# Data Extension Master Plan

## Goal

Plan a narrow host-side numeric batching boundary. It stacks compatible equal-shape samples and
converts caller-owned variable-length numeric sequences into Tensor values plus explicit
valid-length metadata. The extension keeps batch construction outside an NN `Model` while letting
model code receive one self-describing batch value instead of manually maintained primitive length
arrays.

Mental model:

```text
variable-length numeric sequences
  -> explicit truncation, padding value, layout, and type policy
  -> right-padded Tensor + immutable valid sequence lengths
  -> typed Model input
```

This is future planning. `extensions/data` is not present in the authoritative architecture or
Gradle build. No row below authorizes implementation before the coordinated architecture decision.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)
- [Planning guide](../../planning-guide.md)
- [Implementation roadmap](../../roadmap.md)
- [NN master plan](../nn/master-plan.md)

## Scope

- immutable host-side valid-sequence-length metadata;
- explicit batch/time layout and valid-prefix semantics;
- focused numeric sequence batchers with explicit padding/truncation/type policies;
- focused equal-shape numeric sample batching that a downstream image extension may reuse without
  moving image semantics into Data;
- Tensor construction through existing Model APIs after the architecture boundary is accepted; and
- a reusable batch value that text-specific batching may consume without moving tokenization into
  NN.

## Out of scope

- tokenization, vocabularies, normalization, special-token policy, decoding, or text models;
- NN modules, parameters, recurrent cells, recurrent scan/control flow, or optimizers;
- image decoding, image metadata, orientation, color, alpha, resize, crop, normalization, or
  channel-layout policy; these belong to the proposed [Vision extension](../vision/master-plan.md);
- loading files, shuffling datasets, sampling, multiprocessing, prefetch, caching, epochs, or a
  catch-all `DataLoader` facade;
- compiler/runtime/backend execution, device transfer, pinned memory, or kernel skipping; and
- implementation before the architecture contract, module explanations, ADR, dependency tests,
  settings, and build ownership are updated together.

## Module invariants

- Data batching owns host input preparation, not neural-network computation.
- Valid sequence lengths record one valid prefix per batch row. They are the canonical metadata
  for ordinary right padding and are never inferred by comparing Tensor values with zero, a token
  identifier, NaN, or another sentinel.
- Padding length is derived as `timeExtent - validLength`; it is not stored because it depends on
  the other sequences selected for the batch rather than only on the logical sequence.
- A general dense mask is a different capability. It may represent holes or arbitrary validity,
  but it is derived or supplied only for a concrete consumer and must not be stored alongside
  canonical valid lengths as a second source of truth.
- Batch and time extents come from the supplied sequences. They are runtime batch facts, not model
  architecture parameters.
- Padding/truncation/layout/type choices are explicit batcher policy. A `Model` does not tokenize,
  pad raw collections, or mutate caller sequences.

## Proposed dependencies

Subject to the architecture task:

```text
modules/model
  -> extensions/data
```

`extensions/data` would depend on `modules/model` only. It would not depend on NN, Training,
Compiler, Runtime, Prepare, Engine, a concrete backend, or Text. `extensions/text` may depend on
Data, never the reverse. The proposed Vision extension may also depend on Data for focused numeric
sample batching and on Model for Tensor construction; Data must not import Vision or image codecs.

## Forbidden dependencies

- extensions/nn
- extensions/training
- extensions/text
- modules/compiler
- modules/runtime
- modules/prepare
- modules/engine
- concrete backends

These are proposed planning constraints and become enforceable only through the required
architecture/build task.

## Package structure

Proposed after architecture acceptance:

```text
io.github.pho001.synaptik.data/
  batching/  focused equal-shape numeric sample batching and shared host-batch ownership policy
  sequence/  public valid-length metadata, sequence layout, batch value, and focused batchers
```

Do not add generic `util`, `common`, `loader`, or `pipeline` packages. A broader dataset/input
system needs a concrete consumer and separate architecture decision.

## Valid-length representation decision

The conceptual public model-facing value carries values plus valid lengths, not a stored Boolean
mask and not padding lengths:

```text
RightPaddedSequenceBatch
  values       Tensor
  validLengths ValidSequenceLengths
  layout       TIME_MAJOR or BATCH_MAJOR

ValidSequenceLengths
  one valid prefix length for each original batch row
```

The exact Java type names remain Draft until task 0002 becomes the active frontier, but the public
meaning and staged representation decision are fixed:

- `validLengths()` conceptually returns an immutable validated host value with one length per
  original batch row; callers do not maintain or synchronize a naked primitive array;
- every length is in `[0, timeExtent]`, the number of lengths equals the batch extent, and numeric
  values in `values()` never determine validity;
- Data task 0002 must not claim that this host value is already a runtime graph input;
- a later input-binding and recurrent-scan program may materialize/bind the same information as a
  rank-one non-gradient `INT64` Tensor with Shape `[batch]`, but only after Model, Compiler,
  Prepare, Runtime, Engine, and backend ownership and reuse contracts exist; and
- a dense Boolean validity mask is derived on demand only for a concrete attention, loss, or
  similar consumer. The batch does not store both lengths and a mask.

The two lifecycle stages must not share a misleading Java type. The first Data batch is a host
preparation result whose `validLengths()` component is the immutable host value above. A future
typed Model input may instead expose `Tensor values()` and `Tensor validLengths()` after both have
been declared and bound as runtime inputs. The exact runtime-input record, materialization, and
binding API remain deliberately unnamed until their owning lifecycle task; Data 0002 implements
neither that record nor an implicit conversion.

For valid lengths `[5, 3, 1]` and `timeExtent = 5`, padding lengths are derived as `[0, 2, 4]`.
An operation that genuinely requires a dense validity mask may derive:

```text
true true  true  true  true
true true  true  false false
true false false false false
```

The canonical representation remains only `[5, 3, 1]`, validated and owned by the batch metadata.
Numeric zeros in `values` remain ordinary data.

## Computation boundary

The plan distinguishes four non-equivalent stages or execution models:

| Form | Length/validity input | Graph/expression reuse | Padded recurrent work |
|---|---|---|---|
| Data batch metadata | Immutable host `ValidSequenceLengths`-style value | By itself constructs no recurrent expression and promises no graph reuse. | By itself performs no recurrent work. |
| Current NN static containers | Direct snapshotted Java `long[]` compatibility API | A different length pattern may build and compile a different expression graph. | Current expression construction can omit padded logical rows, but no runtime kernel-skipping claim follows. |
| Derived dense mask | Boolean Tensor or dense validity expression created for a concrete consumer | A fixed-shape graph may accept different mask values only after the relevant runtime input lifecycle exists. | Recurrent cell expressions still cover the full dense batch/time rectangle; masking alone does not skip cell work. |
| Future recurrent scan/control flow | Runtime rank-one `INT64` valid lengths consumed by a fixed Model-owned recurrent body/node | Specific length values must not change Model topology or compiled graph structure; compatible prepared programs consume them as runtime data. | May genuinely skip inactive rows/steps only after Model, Compiler, Prepare, Runtime, Engine, and backend contracts implement it. |

The current repository has only the static NN construction technique, exposed through direct Java
length arrays in sequence methods. Those methods remain an explicit current compatibility/legacy
contract until a deliberate migration after the genuine recurrent scan exists. Data task 0002
will define only the batch metadata boundary; it will not add a host adapter that presents static
specialization as the target new API. A runtime-length recurrent overload is meaningful only after
the scan and runtime input-binding prerequisites exist. Arbitrary masks with holes remain a
separate future capability.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | Data/Text/Vision extension architecture and dependency boundary | Draft | Explicit architecture decision; current CPU frontier preserved | Update `ARCHITECTURE.md`, module/dependency explanations, a significant ADR, settings/build ownership, and architecture tests to authorize `extensions/data`, `extensions/text`, and `extensions/vision` with one-way dependencies; create no batching, tokenizer, or image API in the architecture task. |
| 0002 | Valid sequence lengths and right-padded batch value | Draft | 0001; stable Model Tensor construction | Add immutable layout-aware Tensor batch metadata with one validated valid-prefix length per row, staged host/runtime-Tensor boundaries, no stored Boolean mask, no value-based inference, and no batching algorithm beyond the value boundary. |
| 0003 | Focused numeric sample and right-padding batchers | Draft | 0002 | Add explicit equal-shape numeric sample batching plus integral and floating sequence batchers with configured padding value, truncation, layout, and data type; derive extents and canonical valid lengths from source sequences. |
| 0004 | Data batching capability checkpoint | Draft | 0003 | Validate sample/sequence batch geometry, valid-length semantics, documentation, architecture boundaries, and Model Tensor construction without adding a general dataset manager, image semantics, or recurrent execution claims. |

No detailed Data task exists. The current global implementation frontier remains CPU, and NN 0018
is the separately user-authorized future planning frontier. Promote Data 0001 only after the user
authorizes that architecture/module change for implementation.

## Milestones

- Accepted Data/Text extension architecture and one-way dependencies
- Compact valid-length metadata and layout-aware right-padded sequence batch value
- Focused numeric batch construction
- Cross-module sequence-batching checkpoint
- Reusable numeric sample batching boundary for the downstream Vision extension

## Current status

Draft planning only. No `extensions/data` Gradle project, package, production type, dependency, or
architecture permission exists. The user authorized planning this future program, not executing it
ahead of the active CPU frontier or silently changing architecture.

## Open questions

- Select exact public type names and whether layout is a batch component or valid-length axis
  contract when task 0002 becomes current.
- Define the future host-value-to-runtime-`INT64` binding only with the Model/Engine input
  lifecycle and recurrent-scan consumer; Data 0002 must not invent that lifecycle.
- Decide exact truncation policies only with the first concrete numeric/text consumer.
- Decide whether dense arbitrary masks have a sufficiently concrete non-recurrent consumer to
  justify a separate Data type before the runtime recurrent scan exists.
- Select the smallest equal-shape numeric sample input contract that Vision can reuse without
  introducing a generic dataset or loader abstraction.

## Decisions made

- Padding is public semantic metadata, not a value sentinel and not model-internal inference.
- Valid prefix lengths are the sole canonical metadata for ordinary right padding.
- Padding lengths and dense masks are derived and are not stored beside valid lengths.
- The model receives an already constructed batch; it does not pad raw sequences itself.
- Data owns generic numeric batching. Text owns tokenization and the adapter that supplies token
  sequences to Data.
- Vision, not Data, owns image decoding and image-specific transformation policy. It may delegate
  only the final numeric sample stacking step to Data.
- Current static compatibility, derived dense masking, and runtime recurrent scan remain
  explicitly distinct.

## Risks

- Calling valid lengths a mask could imply unsupported holes or bidirectional validity patterns.
- Converting compact lengths to a dense Tensor too early would allocate more metadata and still
  would not make current recurrent computation dynamic.
- Converting host lengths to an eager `INT64` Tensor without a runtime input-binding contract could
  freeze one batch's values into expression construction and falsely imply compiled-graph reuse.
- Adding a broad DataLoader could combine unrelated dataset, batching, sampling, concurrency, and
  I/O responsibilities.
- Putting image decoding in Data would force unrelated numeric/text users to inherit codec,
  metadata, resource-limit, and potentially `java.desktop` obligations.
- Implementing this module without the coordinated architecture update would silently change the
  repository layout and dependency contract.

## Notes

This master plan is non-authoritative future coordination. The architecture contract wins.
