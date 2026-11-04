# First terms

You do not need to learn the whole dictionary before running a [model](glossary.md#term-model). Start with the question you
have now. Each definition explains its prerequisites and distinguishes similar-looking words.
The [complete glossary](glossary.md) is the single dictionary; this page is a reading order, not a
second set of definitions. Links beside specialist words elsewhere in the documentation lead to
the same homes, including specific definitions inside a longer entry.

| Your question | Read these definitions |
|---|---|
| What does “learning” mean? | [Machine learning, targets and hyperparameters](glossary.md#machine-learning-and-hyperparameters), [sample, feature and batch](glossary.md#learning-data-vocabulary) |
| What do the brackets and numbers mean? | [Axis, rank, scalar, vector and matrix](glossary.md#axis-rank-scalar-vector-and-matrix), [Shape](glossary.md#shape) |
| Why are shape and memory [layout](glossary.md#layout) different? | [Layout](glossary.md#layout), [strides, offsets and contiguous storage](glossary.md#storage-mapping-contiguity-and-injectivity); then [worked geometry](explanation/tensors-and-gradients.md) |
| What is a [Tensor](glossary.md#tensor), rather than an executing calculation? | [Tensor](glossary.md#tensor), [leaf versus stored values](glossary.md#leaf-prototype-and-constant), [descriptor and metadata](glossary.md#descriptor-and-metadata) |
| Which numbers can learning change? | [Trainable parameter](glossary.md#trainable-parameter), [ordinary parameters and arguments](glossary.md#parameter-argument-and-target-meanings), [bias and activation](glossary.md#bias-and-activation) |
| How is a [prediction](glossary.md#term-prediction) judged? | [Loss and training targets](glossary.md#loss), [mean-squared error](glossary.md#mean-squared-error--mse), [reduction](glossary.md#aggregate-reduction) |
| How does the update direction arise? | [Derivative and chain rule](glossary.md#derivative-and-chain-rule), [gradient and learning rate](glossary.md#gradient-and-learning-rate), [optimizer and SGD](glossary.md#optimizer-and-stochastic-gradient-descent) |
| Why is train/[eval mode](glossary.md#inference-and-evaluation) not the same as training a parameter? | [Inference and evaluation](glossary.md#inference-and-evaluation), [training session](glossary.md#training-session--trainingsession) |
| What are [Compiler](glossary.md#term-compiler), [Prepare](glossary.md#prepare), [Runtime](glossary.md#term-runtime) and [Engine](glossary.md#term-engine)? | [Names and responsibilities](glossary.md#compiler-engine-and-execution-stage-names), [compile versus prepare versus run](glossary.md#compile-versus-prepare-versus-run) |
| Which object or memory must remain alive? | [Ownership, borrowing and leases](glossary.md#ownership-borrowing-and-lease), [aliasing and arena](glossary.md#aliasing-and-arena), [detached Model results](glossary.md#detached-model-results) |
| Why do numerical formats and special values matter? | [Precision and rounding](glossary.md#floating-point-precision-and-rounding), [NaN, infinity and signed zero](glossary.md#nan-infinity-and-signed-zero), [ULP and tolerance](glossary.md#ulp-and-numerical-tolerance) |
| What can a saved file restore? | [Schema, serialization and checkpoint](glossary.md#schema-serialization-and-checkpoint), [exact training continuation](glossary.md#exact-training-continuation) |
| What does a reused execution or native route mean? | [Cache and memoization](glossary.md#cache-and-memoization), [backend route](glossary.md#backend-route), [CPU/GPU/host/device](glossary.md#cpu-gpu-host-and-device) |
| What do [API](glossary.md#term-application-programming-interface-api), [ABI](glossary.md#term-application-binary-interface-abi), [JDK](glossary.md#term-java-development-kit-jdk) and other contributor abbreviations mean? | [API/SPI/DTO](glossary.md#api-spi-and-dto), [native ABI](glossary.md#application-binary-interface-and-native-bridge), [Java and build tools](glossary.md#java-platform-and-build-tools), [ADR/CI/qualification](glossary.md#adr-ci-and-qualification) |

A familiar word can have a different technical meaning: [Model/module/buffer/Scan](glossary.md#model-module-buffer-and-scan-distinctions),
[stride/normalization/planning/publication](glossary.md#words-with-several-technical-meanings),
[binding/signature/token](glossary.md#binding-signature-and-token-meanings), and
[commit/release/log/seed](glossary.md#commit-release-distribution-log-and-seed-meanings) have explicit
contrasts. Use the qualifier in the surrounding sentence rather than guessing from the spelling.

Follow the [tutorial](tutorials/index.md) for one complete program. Use the
[reference](reference/index.md) when you know the [operation](glossary.md#operation) or API boundary you need; its local
explanations and glossary links remain available when you arrive directly at a section.

## Reading preserved evidence

The [Gate-1 proof record](planning/backends/metal/evidence/0052/README.md) and
[Gate-3 timing record](planning/backends/metal/evidence/0052-gate3/README.md) are preserved
byte-for-byte rather than being rewritten as current guidance. Their vocabulary is explained
in the same dictionary:

- **[Coordinates](glossary.md#axis-rank-scalar-vector-and-matrix) and bounds:** [axes/rank](glossary.md#axis-rank-scalar-vector-and-matrix),
  [Cartesian products, bijections and mixed-radix indexing](glossary.md#index-mappings-and-cost-dominance),
  [strides and injective layouts](glossary.md#storage-mapping-contiguity-and-injectivity).
- **Numerical evidence:** [binary formats and rounding](glossary.md#floating-point-precision-and-rounding),
  [NaN/signed zero](glossary.md#nan-infinity-and-signed-zero),
  [aggregate floors](glossary.md#rounding-sites-fma-and-aggregate-floor),
  [proof assumptions versus observed tests](glossary.md#formal-proof-certificate-and-trusted-assumptions).
- **Execution and measurements:** [dispatch and synchronization](glossary.md#device-submission-synchronization-and-execution-context),
  [structural cost dominance](glossary.md#index-mappings-and-cost-dominance),
  [warmup, medians and measurement samples](glossary.md#benchmark-measurement-vocabulary).

Reading or linking those records does not rerun their experiments or qualify a new implementation.

