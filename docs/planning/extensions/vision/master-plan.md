# Vision Extension Master Plan

## Goal

Plan explicit, resource-bounded image loading and preprocessing without importing codec, desktop,
color, or image-layout concerns into the generic Data extension.

Mental model:

```text
caller-selected image source + caller-selected decoder
  -> bounded decoded image with explicit metadata
  -> explicit orientation/color/alpha transform
  -> explicit resize/crop/normalize/channel layout
  -> Vision image batch
  -> Model Tensor
```

This is future planning only. `extensions/vision` is not authorized by `ARCHITECTURE.md`, included
in Gradle settings, or implemented.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)
- [Planning guide](../../planning-guide.md)
- [Implementation roadmap](../../roadmap.md)
- [Data master plan](../data/master-plan.md)
- [Model master plan](../../modules/model/master-plan.md)

## Scope

- an explicit image decoder contract and explicitly constructed decoder implementations;
- bounded Path/stream decoding with limits on encoded bytes, dimensions, decoded pixels, frames,
  channels, and output allocation before or during decode as the selected codec permits;
- explicit format, orientation, color-space, channel-order, sample-range, and alpha semantics;
- focused resize, crop, and normalization transforms with documented numerical policy;
- conversion to Model Tensor values with explicit NCHW or NHWC layout; and
- image-specific batching that rejects incompatible samples or applies only caller-selected
  transforms before delegating the final equal-shape numeric stacking boundary to Data.

## Out of scope

- a generic dataset, DataLoader, filesystem manager, crawler, cache, prefetch pool, augmentation
  registry, or implicit transform pipeline;
- image decoding or `java.desktop` dependencies inside `extensions/data`;
- automatic format/codec plugin discovery as the public selection mechanism;
- network download, URL fetching, archive traversal, recursive directory scanning, annotation
  formats, video/audio decoding, or multi-frame animation in the first capability;
- model topology, convolution layers, gradients, Training, Engine execution, backend kernels, or
  device transfer; and
- implementation before the coordinated Data/Text/Vision architecture/build decision.

## Module and dependency decision

The selected future boundary is:

```text
modules/model -> extensions/data
modules/model + extensions/data -> extensions/vision
```

Vision depends on Model for Tensor construction and on Data for the focused equal-shape numeric
sample batching/ownership contract. Data never imports Vision, an image type, or a codec. NN and
Training may consume Vision-created Tensors but Vision does not depend on them.

This keeps generic numeric and text batching usable without image codecs or desktop modules while
still avoiding a second generic batching framework inside Vision.

## Proposed package structure

After architecture acceptance:

```text
io.github.pho001.synaptik.vision/
  image/       decoded-image value, dimensions, color/alpha/orientation metadata
  decoding/    decoder contract, limits, source and failure vocabulary
  transform/   explicit resize, crop, color/alpha conversion, normalization, layout conversion
  batching/    image-specific compatibility checks and Data numeric-batch adaptation
```

A selected JDK or third-party decoder implementation remains explicitly named and constructed. If
its dependency footprint justifies a separate adapter project, the architecture task must name
that project and its one-way edge rather than hide it behind discovery.

Do not add `util`, `common`, `loader`, `pipeline`, `service`, `manager`, `registry`, or `plugin`
packages.

## Image semantics decision

Decoding must produce a value whose meaning does not depend on ambient desktop settings:

- width, height, channel count, sample representation, and decoded frame are explicit;
- metadata orientation is either applied under a declared policy or reported/rejected; it is not
  silently ignored while claiming normalized orientation;
- target color meaning is explicit, for example grayscale or sRGB-like RGB, with a documented
  conversion owner;
- alpha is explicitly preserved, dropped only under a defined rule, or composited against a
  caller-supplied background; and
- resize interpolation, crop coordinates, value scaling, mean/std normalization, data type, and
  NCHW/NHWC channel layout are explicit transform inputs.

An image batch does not infer resizing from the other samples. After caller-selected transforms,
all samples must have compatible dimensions/layout/type or batching fails before Tensor
construction. Batch size and spatial extents come from the supplied samples and transform policy,
not from the NN topology.

## JDK decoder implication

The repository targets Java 26. Standard JDK image APIs such as `javax.imageio.ImageIO` and
`java.awt.image.BufferedImage` live in the `java.desktop` module. Using them inside Data would
force that runtime-module footprint on numeric/text users. Generic `ImageIO.read` also delegates
reader selection through the Image I/O provider registry and does not by itself define Synaptik's
resource, orientation, color, alpha, or deterministic reader policy.

A future JDK adapter is therefore optional and explicit, for example a caller-constructed
`JdkImageDecoder` with an allow-listed format/reader policy and resource limits. Its task must
document any remaining JDK provider selection and metadata limitations. If those limitations
cannot meet the selected contract, choose an explicitly registered external decoder instead.
Neither choice is made implicitly by Vision or discovered by Model/Engine.

## Resource and failure boundary

Before allocating a full decoded image where the codec permits, the decoder inspects and validates
the encoded size, format, frame count, dimensions, channel/sample facts, and checked pixel/byte
products. Streams have bounded reads; Paths are ordinary caller-selected files, not traversal
roots. Decompression bombs, integer overflow, unsupported/corrupt metadata, truncated input, and
resource-limit violations fail without returning a partial image or Tensor.

The selected decoder contract must define source ownership: caller streams remain caller-owned
unless an exact overload says otherwise, while decoder-opened file channels close within the
operation. No hidden cache, shared mutable reader, or global registry is introduced.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | Data/Text/Vision architecture boundary participation | Draft | Data 0001; same coordinated overall architecture change; current CPU frontier preserved | Record and validate Vision downstream of Model+Data in the architecture change owned by Data 0001, including any explicit decoder-adapter project; do not create a separate competing architecture implementation or image API. |
| 0002 | Decoded-image and resource-limit contracts | Draft | 0001; concrete decoder consumer | Define immutable image/metadata values, bounded source ownership, checked dimension/pixel/byte limits, explicit failure taxonomy, and no Tensor or transform behavior. |
| 0003 | First explicitly selected decoder adapter | Draft | 0002; selected JDK ImageIO or external codec policy | Decode an allow-listed initial format set with explicit implementation selection, preallocation limits where supported, orientation/metadata behavior, cleanup, corruption tests, and no plugin discovery facade. |
| 0004 | Image transform and Tensor conversion | Draft | 0002–0003; stable Model Tensor import | Add explicit orientation, color, alpha, resize, crop, normalize, data-type, and NCHW/NHWC conversion semantics without hidden defaults or backend execution claims. |
| 0005 | Image batching | Draft | 0004; Data 0003 equal-shape numeric batching | Validate transformed sample compatibility, derive batch extent, delegate final numeric stacking to Data, and return an image-specific Tensor batch without a dataset/loader facade. |
| 0006 | Vision capability checkpoint | Draft | 0003–0005 | Validate representative/corrupt/adversarial inputs, resource bounds, metadata semantics, Tensor values/layout, batch geometry, documentation, optional decoder dependency, and architecture enforcement. |

No detailed Vision task exists. NN 0018 remains the sole new detailed `Ready` task in this
planning program. Promote Vision 0001 only as part of the explicitly authorized coordinated
architecture change.

## Planned image-batch flow

The following is conceptual and not current runnable Synaptik code:

```java
ImageDecoder decoder = JdkImageDecoder.builder()
        .allowFormats(ImageFormat.PNG, ImageFormat.JPEG)
        .limits(new ImageDecodeLimits(maxEncodedBytes, maxWidth, maxHeight, maxPixels))
        .build();

ImageTransform transform = ImageTransform.builder()
        .orientation(OrientationPolicy.APPLY)
        .color(TargetColor.RGB)
        .alpha(AlphaPolicy.composite(background))
        .resize(Resize.exact(224, 224, Interpolation.BILINEAR))
        .normalize(mean, standardDeviation)
        .layout(ImageTensorLayout.NCHW)
        .build();

ImageBatch batch = ImageBatcher.of(decoder, transform)
        .batch(List.of(firstPath, secondPath));

Tensor input = batch.values();
```

Interpretation: the caller selects the decoder and every semantic transform. Vision owns image
meaning and compatibility; Data owns only the final equal-shape numeric batch assembly; Model
receives the resulting Tensor. None of the shown APIs exists yet, and exact names remain Draft.

## Milestones

- Accepted Vision module/dependency and decoder-adapter architecture
- Bounded decoded-image/source contract
- First explicit decoder implementation
- Explicit image transformation and Tensor conversion
- Image batching over Data's focused numeric boundary
- Adversarial resource/metadata and architecture checkpoint

## Current status

Draft planning only. The legacy Neurotic `ImgDataSet` was commented out and did not implement
decoding, resource limits, metadata semantics, or batching, so it supplies no reusable design.

## Open questions

- Select the first image formats and whether the JDK Image I/O provider behavior is sufficiently
  explicit/testable or an external decoder adapter is required.
- Select the initial orientation metadata scope, color conversion standard, alpha defaults (if
  any), interpolation definitions, and exact numeric rounding.
- Decide whether decoded storage is a Vision-owned primitive array/value or another bounded host
  carrier before Model Tensor construction.
- Define maximum encoded bytes, dimensions, pixels, channels, frames, and decoded bytes for the
  first decoder configuration.

## Decisions made

- Image loading does not belong directly in generic Data.
- Vision depends downstream on Model+Data and owns every image-specific semantic and resource
  decision.
- Decoder implementations are explicitly selected and constructed; there is no implicit public
  plugin discovery or broad file-loader facade.
- Data may stack compatible numeric samples but does not decode, orient, recolor, resize, crop, or
  normalize images.
- The JDK `java.desktop`/ImageIO dependency, if selected, belongs only to an explicit Vision
  decoder adapter and must disclose provider/metadata limitations.

## Risks

- A convenience `ImageIO.read` wrapper could allocate too much memory, ignore orientation, select
  ambient providers, return ambiguous color/alpha values, or collapse corrupt and unsupported
  input into unclear behavior.
- Implicit resize/normalization/layout defaults would make trained model meaning depend on an
  invisible preprocessing pipeline.
- A broad DataLoader or transform registry could combine file traversal, decode, augmentation,
  concurrency, caching, batching, and Tensor creation into one god component.
- Keeping both interleaved and planar decoded copies unnecessarily could double peak memory before
  Tensor construction.
- Decoder or transform integer overflow could bypass allocation limits unless every dimension and
  byte product is checked.

## Notes

This master plan is non-authoritative future coordination. The architecture contract wins.
