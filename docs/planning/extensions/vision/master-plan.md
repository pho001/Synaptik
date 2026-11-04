# Vision Extension Master Plan

## Goal

Plan explicit, resource-bounded image loading and preprocessing without importing [codec](../../../glossary.md#term-codec), desktop,
color, or image-[layout](../../../glossary.md#layout) concerns into the generic Data extension.

[Mental model](../../../glossary.md#term-mental-model):

```text
caller-selected image source + caller-selected decoder
  -> bounded decoded image with explicit metadata
  -> explicit orientation/color/alpha transform
  -> explicit resize/crop/normalize/channel layout
  -> Vision image batch
  -> Model Tensor
```

This is future [planning](../../../glossary.md#term-planning) only. `extensions/vision` is not authorized by `ARCHITECTURE.md`, included
in [Gradle](../../../glossary.md#term-gradle) settings, or implemented.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../glossary.md#architecture-contract))</span>
- [Module boundaries](../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../glossary.md#term-module))</span>
- [Dependency rules](../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../glossary.md#term-dependency))</span>
- [Planning guide](../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../roadmap.md)
- [Data master plan](../data/master-plan.md)
- [Model master plan](../../modules/model/master-plan.md)<span data-glossary-navigation="true"> ([Model glossary](../../../glossary.md#term-model))</span>

## [Scope](../../../glossary.md#term-scope)

- an explicit image decoder contract and explicitly constructed decoder implementations;
- bounded Path/stream decoding with limits on encoded bytes, dimensions, decoded pixels, frames,
  channels, and output allocation before or during decode as the selected codec permits;<span data-glossary-navigation="true"> ([dimensions glossary](../../../glossary.md#dimension); [codec glossary](../../../glossary.md#term-codec))</span>
- explicit format, orientation, color-space, [channel](../../../glossary.md#term-channel)-order, [sample](../../../glossary.md#term-sample-machine-learning-and-hyperparameters)-range, and alpha semantics;
- focused resize, crop, and [normalization](../../../glossary.md#term-normalization) transforms with documented numerical policy;
- conversion to [Model](../../../glossary.md#term-model) [Tensor](../../../glossary.md#tensor) values with explicit [NCHW](../../../glossary.md#term-nchw) or [NHWC](../../../glossary.md#term-nhwc) [layout](../../../glossary.md#layout); and
- image-specific batching that rejects incompatible samples or applies only caller-selected
  transforms before delegating the final equal-shape numeric stacking boundary to Data.<span data-glossary-navigation="true"> ([shape glossary](../../../glossary.md#shape))</span>

## Out of [scope](../../../glossary.md#term-scope)

- a generic dataset, DataLoader, filesystem manager, crawler, cache, prefetch pool, augmentation
  registry, or implicit transform pipeline;<span data-glossary-navigation="true"> ([dataset glossary](../../../glossary.md#term-dataset); [cache glossary](../../../glossary.md#term-cache); [registry glossary](../../../glossary.md#term-registry))</span>
- image decoding or `java.desktop` dependencies inside `extensions/data`;
- automatic format/[codec](../../../glossary.md#term-codec) [plugin](../../../glossary.md#term-plugin) discovery as the public selection mechanism;
- network download, URL fetching, archive traversal, recursive directory scanning, annotation
  formats, video/audio decoding, or multi-frame animation in the first capability;
- model topology, convolution layers, gradients, Training, Engine execution, backend kernels, or
  device transfer; and<span data-glossary-navigation="true"> ([model glossary](../../../glossary.md#term-model); [topology glossary](../../../glossary.md#term-topology); [convolution glossary](../../../glossary.md#term-channels-first-convolution-layers); [gradients glossary](../../../glossary.md#term-gradient); [Engine glossary](../../../glossary.md#term-engine); [backend glossary](../../../glossary.md#backend); [kernels glossary](../../../glossary.md#kernel); [device glossary](../../../glossary.md#term-device))</span>
- implementation before Vision's separate explicit architecture/build decision.

## [Module](../../../glossary.md#term-module) and [dependency](../../../glossary.md#term-dependency) decision

The proposed boundary needs separate Vision architecture approval; Data0001 grants no Vision edge:

```text
modules/model -> extensions/data
modules/model + extensions/data -> extensions/vision
```

Vision depends on [Model](../../../glossary.md#term-model) for [Tensor](../../../glossary.md#tensor) construction and on Data for the focused equal-[shape](../../../glossary.md#shape) numeric
[sample](../../../glossary.md#term-sample-machine-learning-and-hyperparameters) batching/[ownership](../../../glossary.md#term-ownership-meanings) contract. Data never imports Vision, an image type, or a [codec](../../../glossary.md#term-codec). [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) and
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

A selected [JDK](../../../glossary.md#term-java-development-kit-jdk) or third-party decoder implementation remains explicitly named and constructed. If
its [dependency](../../../glossary.md#term-dependency) footprint justifies a separate adapter project, the architecture task must name
that project and its one-way edge rather than hide it behind discovery.

Do not add `util`, `common`, `loader`, `pipeline`, `service`, `manager`, [`registry`](../../../glossary.md#term-registry), or [`plugin`](../../../glossary.md#term-plugin)
packages.

## Image semantics decision

Decoding must produce a value whose meaning does not depend on ambient desktop settings:

- width, height, [channel](../../../glossary.md#term-channel) count, [sample](../../../glossary.md#term-sample-machine-learning-and-hyperparameters) representation, and decoded frame are explicit;
- metadata orientation is either applied under a declared policy or reported/rejected; it is not
  silently ignored while claiming normalized orientation;<span data-glossary-navigation="true"> ([metadata glossary](../../../glossary.md#term-metadata))</span>
- target color meaning is explicit, for example grayscale or sRGB-like RGB, with a documented
  conversion owner;<span data-glossary-navigation="true"> ([target glossary](../../../glossary.md#term-target); [grayscale glossary](../../../glossary.md#term-grayscale); [sRGB glossary](../../../glossary.md#term-srgb); [RGB glossary](../../../glossary.md#term-rgb); [owner glossary](../../../glossary.md#term-owner))</span>
- alpha is explicitly preserved, dropped only under a defined rule, or composited against a
  caller-supplied background; and
- resize interpolation, crop coordinates, value scaling, mean/std normalization, data type, and
  NCHW/NHWC channel layout are explicit transform inputs.<span data-glossary-navigation="true"> ([resize interpolation glossary](../../../glossary.md#text-formats-image-terms-and-decoding-limits); [coordinates glossary](../../../glossary.md#axis-rank-scalar-vector-and-matrix); [mean glossary](../../../glossary.md#term-mean); [normalization glossary](../../../glossary.md#term-normalization); [data type glossary](../../../glossary.md#data-type--datatype); [NCHW glossary](../../../glossary.md#term-nchw); [NHWC glossary](../../../glossary.md#term-nhwc))</span>

An image [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters) does not infer resizing from the other samples. After caller-selected transforms,
all samples must have compatible [dimensions](../../../glossary.md#dimension)/[layout](../../../glossary.md#layout)/type or batching fails before [Tensor](../../../glossary.md#tensor)
construction. Batch size and [spatial](../../../glossary.md#term-spatial-axes) extents come from the supplied samples and transform policy,
not from the [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [topology](../../../glossary.md#term-topology).

## [JDK](../../../glossary.md#term-java-development-kit-jdk) decoder implication

The repository [targets](../../../glossary.md#term-target) Java 26. Standard JDK image APIs such as `javax.imageio.ImageIO` and
`java.awt.image.BufferedImage` live in the `java.desktop` [module](../../../glossary.md#term-module). Using them inside Data would
force that [runtime](../../../glossary.md#term-runtime)-module footprint on numeric/text users. Generic `ImageIO.read` also delegates
reader selection through the Image I/O provider [registry](../../../glossary.md#term-registry) and does not by itself define Synaptik's
resource, orientation, color, alpha, or deterministic reader policy.

A future JDK adapter is therefore optional and explicit, for example a caller-constructed
`JdkImageDecoder` with an allow-listed format/reader policy and resource limits. Its task must
document any remaining JDK provider selection and [metadata](../../../glossary.md#term-metadata) limitations. If those limitations
cannot meet the selected contract, choose an explicitly registered external decoder instead.
Neither choice is made implicitly by Vision or discovered by [Model](../../../glossary.md#term-model)/[Engine](../../../glossary.md#term-engine).

## Resource and failure boundary

Before allocating a full decoded image where the [codec](../../../glossary.md#term-codec) permits, the decoder inspects and validates
the encoded size, format, frame count, [dimensions](../../../glossary.md#dimension), [channel](../../../glossary.md#term-channel)/[sample](../../../glossary.md#term-sample-machine-learning-and-hyperparameters) facts, and checked pixel/byte
products. Streams have bounded reads; Paths are ordinary caller-selected files, not traversal
roots. Decompression bombs, integer [overflow](../../../glossary.md#term-overflow), unsupported/corrupt [metadata](../../../glossary.md#term-metadata), truncated input, and
resource-limit violations fail without returning a partial image or [Tensor](../../../glossary.md#tensor).

The selected decoder contract must define source [ownership](../../../glossary.md#term-ownership-meanings): caller streams remain caller-owned
unless an exact overload says otherwise, while decoder-opened file channels close within the
[operation](../../../glossary.md#operation). No hidden [cache](../../../glossary.md#term-cache), shared [mutable](../../../glossary.md#term-mutable) reader, or global [registry](../../../glossary.md#term-registry) is introduced.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | Vision architecture and [dependency](../../../glossary.md#term-dependency) boundary | Draft | Separate explicit Vision architecture/[module](../../../glossary.md#term-module) decision; approved Data boundary before any Data edge | Decide and validate Vision downstream of [Model](../../../glossary.md#term-model)+Data and any concrete decoder-adapter project in its own future coordinated architecture change; Data0001 grants no Vision module, edge or image [API](../../../glossary.md#term-application-programming-interface-api). |
| 0002 | Decoded-image and resource-limit contracts | Draft | 0001; concrete decoder consumer | Define [immutable](../../../glossary.md#term-immutable) image/[metadata](../../../glossary.md#term-metadata) values, bounded source [ownership](../../../glossary.md#term-ownership-meanings), checked [dimension](../../../glossary.md#dimension)/pixel/byte limits, explicit failure taxonomy, and no [Tensor](../../../glossary.md#tensor) or transform behavior. |
| 0003 | First explicitly selected decoder adapter | Draft | 0002; selected [JDK](../../../glossary.md#term-java-development-kit-jdk) ImageIO or external [codec](../../../glossary.md#term-codec) policy | Decode an allow-listed initial format set with explicit implementation selection, preallocation limits where supported, orientation/metadata behavior, cleanup, corruption tests, and no [plugin](../../../glossary.md#term-plugin) discovery [facade](../../../glossary.md#term-facade). |
| 0004 | Image transform and Tensor conversion | Draft | 0002–0003; stable Model Tensor import | Add explicit orientation, color, alpha, resize, crop, normalize, data-type, and [NCHW](../../../glossary.md#term-nchw)/[NHWC](../../../glossary.md#term-nhwc) conversion semantics without hidden defaults or [backend](../../../glossary.md#backend) execution claims. |
| 0005 | Image batching | Draft | 0004; deferred Data0006 equal-[shape](../../../glossary.md#shape) numeric batching | Validate transformed [sample](../../../glossary.md#term-sample-machine-learning-and-hyperparameters) compatibility, derive [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [extent](../../../glossary.md#term-extent), delegate final numeric stacking to Data, and return an image-specific Tensor batch without a [dataset](../../../glossary.md#term-dataset)/loader facade. |
| 0006 | Vision capability [checkpoint](../../../glossary.md#term-checkpoint) | Draft | 0003–0005 | Validate representative/corrupt/adversarial inputs, resource bounds, metadata semantics, Tensor values/[layout](../../../glossary.md#layout), batch geometry, documentation, optional decoder dependency, and architecture enforcement. |

No detailed Vision task exists and no Vision task is [Ready](../../../glossary.md#term-ready). Authorized Vision frontier: None.
Vision0001 needs its own explicit architecture decision; [Data0001](../data/tasks/0001-data-extension-boundary.md)
is Data-only. Model's numerical cutover is Complete and is not a pending Vision prerequisite.
Other frontiers belong to the roadmap.

## Planned image-[batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters) flow

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
meaning and compatibility; Data owns only the final equal-[shape](../../../glossary.md#shape) numeric batch assembly; [Model](../../../glossary.md#term-model)
receives the resulting [Tensor](../../../glossary.md#tensor). None of the shown APIs exists yet, and exact names remain Draft.

## Milestones

- Accepted Vision [module](../../../glossary.md#term-module)/[dependency](../../../glossary.md#term-dependency) and decoder-adapter architecture
- Bounded decoded-image/source contract
- First explicit decoder implementation
- Explicit image transformation and [Tensor](../../../glossary.md#tensor) conversion
- Image batching over Data's focused numeric boundary
- Adversarial resource/[metadata](../../../glossary.md#term-metadata) and architecture [checkpoint](../../../glossary.md#term-checkpoint)

## Current status

Draft [planning](../../../glossary.md#term-planning) only. The legacy Neurotic `ImgDataSet` was commented out and did not implement
decoding, resource limits, [metadata](../../../glossary.md#term-metadata) semantics, or batching, so it supplies no reusable design.

## Open questions

- Select the first image formats and whether the JDK Image I/O provider behavior is sufficiently
  explicit/testable or an external decoder adapter is required.<span data-glossary-navigation="true"> ([JDK glossary](../../../glossary.md#term-java-development-kit-jdk))</span>
- Select the initial orientation metadata scope, color conversion standard, alpha defaults (if
  any), interpolation definitions, and exact numeric rounding.<span data-glossary-navigation="true"> ([metadata glossary](../../../glossary.md#term-metadata); [scope glossary](../../../glossary.md#term-scope); [rounding glossary](../../../glossary.md#term-rounding))</span>
- Decide whether decoded storage is a Vision-owned primitive array/value or another bounded host
  carrier before Model Tensor construction.<span data-glossary-navigation="true"> ([host glossary](../../../glossary.md#term-host); [carrier glossary](../../../glossary.md#term-storage-carrier); [Model glossary](../../../glossary.md#term-model); [Tensor glossary](../../../glossary.md#tensor))</span>
- Define maximum encoded bytes, dimensions, pixels, channels, frames, and decoded bytes for the
  first decoder configuration.<span data-glossary-navigation="true"> ([dimensions glossary](../../../glossary.md#dimension))</span>

## Decisions made

- Image loading does not belong directly in generic Data.
- Vision depends downstream on Model+Data and owns every image-specific semantic and resource
  decision.<span data-glossary-navigation="true"> ([Model glossary](../../../glossary.md#term-model))</span>
- Decoder implementations are explicitly selected and constructed; there is no implicit public
  plugin discovery or broad file-loader facade.<span data-glossary-navigation="true"> ([plugin glossary](../../../glossary.md#term-plugin); [facade glossary](../../../glossary.md#term-facade))</span>
- Data may stack compatible numeric samples but does not decode, orient, recolor, resize, crop, or
  normalize images.<span data-glossary-navigation="true"> ([samples glossary](../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- The [JDK](../../../glossary.md#term-java-development-kit-jdk) `java.desktop`/ImageIO dependency, if selected, belongs only to an explicit Vision
  decoder adapter and must disclose provider/metadata limitations.<span data-glossary-navigation="true"> ([dependency glossary](../../../glossary.md#term-dependency); [metadata glossary](../../../glossary.md#term-metadata))</span>

## Risks

- A convenience `ImageIO.read` wrapper could allocate too much memory, ignore orientation, select
  ambient providers, return ambiguous color/alpha values, or collapse corrupt and unsupported
  input into unclear behavior.
- Implicit resize/normalization/layout defaults would make trained model meaning depend on an
  invisible preprocessing pipeline.<span data-glossary-navigation="true"> ([normalization glossary](../../../glossary.md#term-normalization); [layout glossary](../../../glossary.md#layout); [model glossary](../../../glossary.md#term-model))</span>
- A broad DataLoader or transform registry could combine file traversal, decode, augmentation,
  concurrency, caching, batching, and Tensor creation into one god component.<span data-glossary-navigation="true"> ([registry glossary](../../../glossary.md#term-registry); [concurrency glossary](../../../glossary.md#term-concurrency); [Tensor glossary](../../../glossary.md#tensor))</span>
- Keeping both interleaved and planar decoded copies unnecessarily could double peak memory before
  Tensor construction.
- Decoder or transform integer overflow could bypass allocation limits unless every dimension and
  byte product is checked.<span data-glossary-navigation="true"> ([overflow glossary](../../../glossary.md#term-overflow); [dimension glossary](../../../glossary.md#dimension))</span>

## Notes

This master plan is non-authoritative future coordination. The [architecture contract](../../../glossary.md#architecture-contract) wins.
