# [Tensor](../glossary.md#tensor) [API](../glossary.md#term-application-programming-interface-api) — migrated reference

This reference now lives in topic-sized [canonical](../glossary.md#term-canonical-form) pages. Start at [the reference goals](../reference/index.md) or [the area entry](../reference/tensor/index.md). Detailed contracts and examples are maintained there, not duplicated here.

## Previous sections

<a id="tensor-api"></a>
- [Tensor API](../reference/tensor/index.md#tensor-api)<span data-glossary-navigation="true"> ([Tensor glossary](../glossary.md#tensor); [API glossary](../glossary.md#term-application-programming-interface-api))</span>

<a id="purpose-and-mental-model"></a>
- [Purpose and mental model](../history/reference/tensor/qualification-overview.md#purpose-and-mental-model)<span data-glossary-navigation="true"> ([mental model glossary](../glossary.md#term-mental-model))</span>

<a id="data-types"></a>
- [Data types](../reference/tensor/data-types.md#data-types)

<a id="numeric-promotion"></a>
- [Numeric promotion](../reference/tensor/data-types.md#numeric-promotion)<span data-glossary-navigation="true"> ([Numeric promotion glossary](../glossary.md#numeric-promotion))</span>

<a id="sixteen-bit-floating-representations"></a>
- [Sixteen-bit floating representations](../reference/tensor/data-types.md#sixteen-bit-floating-representations)

<a id="exact-typed-scalar-values"></a>
- [Exact typed scalar values](../reference/tensor/data-types.md#exact-typed-scalar-values)<span data-glossary-navigation="true"> ([scalar glossary](../glossary.md#term-scalar))</span>

<a id="shapes-and-dimensions"></a>
- [Shapes and dimensions](../reference/tensor/shapes.md#shapes-and-dimensions)<span data-glossary-navigation="true"> ([Shapes glossary](../glossary.md#shape); [dimensions glossary](../glossary.md#dimension))</span>

<a id="symbolic-extent-expressions"></a>
- [Symbolic extent expressions](../reference/tensor/shapes.md#symbolic-extent-expressions)<span data-glossary-navigation="true"> ([extent glossary](../glossary.md#term-extent))</span>

<a id="complete-symbolic-extent-example"></a>
- [Complete symbolic-extent example](../reference/tensor/shapes.md#complete-symbolic-extent-example)

<a id="goal-and-inputs"></a>
- [Goal and inputs](../reference/tensor/shapes.md#goal-and-inputs)

<a id="result-and-interpretation"></a>
- [Result and interpretation](../reference/tensor/shapes.md#result-and-interpretation)

<a id="failures-and-useful-variations"></a>
- [Failures and useful variations](../reference/tensor/shapes.md#failures-and-useful-variations)

<a id="scalars-and-empty-tensors"></a>
- [Scalars and empty tensors](../reference/tensor/shapes.md#scalars-and-empty-tensors)

<a id="static-and-dynamic-element-counts"></a>
- [Static and dynamic element counts](../reference/tensor/shapes.md#static-and-dynamic-element-counts)<span data-glossary-navigation="true"> ([element glossary](../glossary.md#term-element))</span>

<a id="broadcasting"></a>
- [Broadcasting](../reference/tensor/shapes.md#broadcasting)<span data-glossary-navigation="true"> ([Broadcasting glossary](../glossary.md#broadcasting))</span>

<a id="shape-example"></a>
- [Shape example](../reference/tensor/shapes.md#shape-example)

<a id="resolved-layouts"></a>
- [Resolved layouts](../reference/tensor/layouts.md#resolved-layouts)<span data-glossary-navigation="true"> ([layouts glossary](../glossary.md#layout))</span>

<a id="layout-example"></a>
- [Layout example](../reference/tensor/layouts.md#layout-example)

<a id="tensor-descriptors"></a>
- [Tensor descriptors](../reference/tensor/descriptors.md#tensor-descriptors)<span data-glossary-navigation="true"> ([descriptors glossary](../glossary.md#term-descriptor))</span>

<a id="complete-descriptor-example"></a>
- [Complete descriptor example](../reference/tensor/descriptors.md#complete-descriptor-example)

<a id="goal-and-inputs-1"></a>
- [Goal and inputs](../reference/tensor/descriptors.md#goal-and-inputs-1)

<a id="meaningful-lines"></a>
- [Meaningful lines](../reference/tensor/descriptors.md#meaningful-lines)

<a id="result-and-interpretation-1"></a>
- [Result and interpretation](../reference/tensor/descriptors.md#result-and-interpretation-1)

<a id="failures-and-useful-variations-1"></a>
- [Failures and useful variations](../reference/tensor/descriptors.md#failures-and-useful-variations-1)

<a id="public-tensor-state"></a>
- [Public Tensor state](../reference/tensor/identity.md#public-tensor-state)

<a id="expression-producers-and-indexed-provenance"></a>
- [Expression producers and indexed provenance](../reference/tensor/identity.md#expression-producers-and-indexed-provenance)<span data-glossary-navigation="true"> ([provenance glossary](../glossary.md#provenance))</span>

<a id="single-output-example"></a>
- [Single-output example](../reference/tensor/identity.md#single-output-example)

<a id="goal-and-inputs-2"></a>
- [Goal and inputs](../reference/tensor/identity.md#goal-and-inputs-2)

<a id="result-and-interpretation-2"></a>
- [Result and interpretation](../reference/tensor/identity.md#result-and-interpretation-2)

<a id="useful-variation"></a>
- [Useful variation](../reference/tensor/identity.md#useful-variation)

<a id="multi-output-and-hidden-output-identity"></a>
- [Multi-output and hidden-output identity](../reference/tensor/identity.md#multi-output-and-hidden-output-identity)<span data-glossary-navigation="true"> ([identity glossary](../glossary.md#term-identity-meanings))</span>

<a id="complete-constant-creation-example"></a>
- [Complete constant-creation example](../reference/tensor/constants.md#complete-constant-creation-example)<span data-glossary-navigation="true"> ([constant glossary](../glossary.md#term-constant))</span>

<a id="goal-and-inputs-3"></a>
- [Goal and inputs](../reference/tensor/constants.md#goal-and-inputs-3)

<a id="meaningful-lines-1"></a>
- [Meaningful lines](../reference/tensor/constants.md#meaningful-lines-1)

<a id="result-and-interpretation-3"></a>
- [Result and interpretation](../reference/tensor/constants.md#result-and-interpretation-3)

<a id="failures-and-useful-variations-2"></a>
- [Failures and useful variations](../reference/tensor/constants.md#failures-and-useful-variations-2)

<a id="complete-full-value-and-rectangular-identity-example"></a>
- [Complete full-value and rectangular-identity example](../reference/tensor/constants.md#complete-full-value-and-rectangular-identity-example)

<a id="goal-and-inputs-4"></a>
- [Goal and inputs](../reference/tensor/constants.md#goal-and-inputs-4)

<a id="meaningful-lines-and-intermediate-values"></a>
- [Meaningful lines and intermediate values](../reference/tensor/constants.md#meaningful-lines-and-intermediate-values)

<a id="result-and-interpretation-4"></a>
- [Result and interpretation](../reference/tensor/constants.md#result-and-interpretation-4)

<a id="failures-and-useful-variations-3"></a>
- [Failures and useful variations](../reference/tensor/constants.md#failures-and-useful-variations-3)

<a id="complete-integer-range-example"></a>
- [Complete integer-range example](../reference/tensor/random-initialization.md#complete-integer-range-example)

<a id="goal-and-inputs-5"></a>
- [Goal and inputs](../reference/tensor/random-initialization.md#goal-and-inputs-5)

<a id="meaningful-lines-2"></a>
- [Meaningful lines](../reference/tensor/random-initialization.md#meaningful-lines-2)

<a id="result-and-interpretation-5"></a>
- [Result and interpretation](../reference/tensor/random-initialization.md#result-and-interpretation-5)

<a id="failures-and-useful-variations-4"></a>
- [Failures and useful variations](../reference/tensor/random-initialization.md#failures-and-useful-variations-4)

<a id="complete-normal-random-example"></a>
- [Complete normal-random example](../reference/tensor/random-initialization.md#complete-normal-random-example)

<a id="goal-and-inputs-6"></a>
- [Goal and inputs](../reference/tensor/random-initialization.md#goal-and-inputs-6)

<a id="meaningful-lines-3"></a>
- [Meaningful lines](../reference/tensor/random-initialization.md#meaningful-lines-3)

<a id="result-and-interpretation-6"></a>
- [Result and interpretation](../reference/tensor/random-initialization.md#result-and-interpretation-6)

<a id="failures-and-useful-variations-5"></a>
- [Failures and useful variations](../reference/tensor/random-initialization.md#failures-and-useful-variations-5)

<a id="complete-continuous-uniform-example"></a>
- [Complete continuous-uniform example](../reference/tensor/random-initialization.md#complete-continuous-uniform-example)

<a id="goal-and-inputs-7"></a>
- [Goal and inputs](../reference/tensor/random-initialization.md#goal-and-inputs-7)

<a id="meaningful-lines-4"></a>
- [Meaningful lines](../reference/tensor/random-initialization.md#meaningful-lines-4)

<a id="result-and-interpretation-7"></a>
- [Result and interpretation](../reference/tensor/random-initialization.md#result-and-interpretation-7)

<a id="failures-and-useful-variations-6"></a>
- [Failures and useful variations](../reference/tensor/random-initialization.md#failures-and-useful-variations-6)

<a id="complete-bounded-integral-example"></a>
- [Complete bounded-integral example](../reference/tensor/random-initialization.md#complete-bounded-integral-example)

<a id="goal-and-inputs-8"></a>
- [Goal and inputs](../reference/tensor/random-initialization.md#goal-and-inputs-8)

<a id="meaningful-lines-5"></a>
- [Meaningful lines](../reference/tensor/random-initialization.md#meaningful-lines-5)

<a id="result-and-interpretation-8"></a>
- [Result and interpretation](../reference/tensor/random-initialization.md#result-and-interpretation-8)

<a id="failures-and-useful-variations-7"></a>
- [Failures and useful variations](../reference/tensor/random-initialization.md#failures-and-useful-variations-7)

<a id="complete-bernoulli-random-example"></a>
- [Complete Bernoulli-random example](../reference/tensor/random-initialization.md#complete-bernoulli-random-example)<span data-glossary-navigation="true"> ([Bernoulli glossary](../glossary.md#term-bernoulli-distribution))</span>

<a id="goal-and-inputs-9"></a>
- [Goal and inputs](../reference/tensor/random-initialization.md#goal-and-inputs-9)

<a id="meaningful-lines-6"></a>
- [Meaningful lines](../reference/tensor/random-initialization.md#meaningful-lines-6)

<a id="result-and-interpretation-9"></a>
- [Result and interpretation](../reference/tensor/random-initialization.md#result-and-interpretation-9)

<a id="failures-and-useful-variations-8"></a>
- [Failures and useful variations](../reference/tensor/random-initialization.md#failures-and-useful-variations-8)

<a id="explicit-graph-rng-state"></a>
- [Explicit graph RNG state](../reference/tensor/graph-randomness.md#explicit-graph-rng-state)<span data-glossary-navigation="true"> ([graph RNG state glossary](../glossary.md#graph-rng-state--graphrngstate))</span>

<a id="complete-current-state-example"></a>
- [Complete current-state example](../reference/tensor/graph-randomness.md#complete-current-state-example)

<a id="goal-and-inputs-10"></a>
- [Goal and inputs](../reference/tensor/graph-randomness.md#goal-and-inputs-10)

<a id="meaningful-lines-7"></a>
- [Meaningful lines](../reference/tensor/graph-randomness.md#meaningful-lines-7)

<a id="result-and-interpretation-10"></a>
- [Result and interpretation](../reference/tensor/graph-randomness.md#result-and-interpretation-10)

<a id="explicit-state-dropout-construction"></a>
- [Explicit-state dropout construction](../reference/tensor/graph-randomness.md#explicit-state-dropout-construction)<span data-glossary-navigation="true"> ([dropout glossary](../glossary.md#dropout--inverted-dropout))</span>

<a id="complete-dropout-construction-example"></a>
- [Complete dropout-construction example](../reference/tensor/graph-randomness.md#complete-dropout-construction-example)

<a id="goal-and-inputs-11"></a>
- [Goal and inputs](../reference/tensor/graph-randomness.md#goal-and-inputs-11)

<a id="meaningful-lines-8"></a>
- [Meaningful lines](../reference/tensor/graph-randomness.md#meaningful-lines-8)

<a id="result-and-interpretation-11"></a>
- [Result and interpretation](../reference/tensor/graph-randomness.md#result-and-interpretation-11)

<a id="state-advancement-branching-and-edge-cases"></a>
- [State advancement, branching, and edge cases](../reference/tensor/graph-randomness.md#state-advancement-branching-and-edge-cases)

<a id="complete-flat-import-example"></a>
- [Complete flat-import example](../reference/tensor/import.md#complete-flat-import-example)

<a id="goal-and-inputs-12"></a>
- [Goal and inputs](../reference/tensor/import.md#goal-and-inputs-12)

<a id="meaningful-lines-9"></a>
- [Meaningful lines](../reference/tensor/import.md#meaningful-lines-9)

<a id="result-and-interpretation-12"></a>
- [Result and interpretation](../reference/tensor/import.md#result-and-interpretation-12)

<a id="failures-and-useful-variations-9"></a>
- [Failures and useful variations](../reference/tensor/import.md#failures-and-useful-variations-9)

<a id="complete-nested-import-example"></a>
- [Complete nested-import example](../reference/tensor/import.md#complete-nested-import-example)

<a id="goal-and-inputs-13"></a>
- [Goal and inputs](../reference/tensor/import.md#goal-and-inputs-13)

<a id="meaningful-lines-10"></a>
- [Meaningful lines](../reference/tensor/import.md#meaningful-lines-10)

<a id="result-and-interpretation-13"></a>
- [Result and interpretation](../reference/tensor/import.md#result-and-interpretation-13)

<a id="failures-and-useful-variations-10"></a>
- [Failures and useful variations](../reference/tensor/import.md#failures-and-useful-variations-10)

<a id="binary-arithmetic-expressions"></a>
- [Binary arithmetic expressions](../reference/tensor/arithmetic.md#binary-arithmetic-expressions)<span data-glossary-navigation="true"> ([Binary glossary](../glossary.md#term-binary-descriptor-and-metadata))</span>

<a id="complete-binary-expression-example"></a>
- [Complete binary-expression example](../reference/tensor/arithmetic.md#complete-binary-expression-example)

<a id="goal-and-inputs-14"></a>
- [Goal and inputs](../reference/tensor/arithmetic.md#goal-and-inputs-14)

<a id="meaningful-lines-and-intermediate-results"></a>
- [Meaningful lines and intermediate results](../reference/tensor/arithmetic.md#meaningful-lines-and-intermediate-results)

<a id="result-and-interpretation-14"></a>
- [Result and interpretation](../reference/tensor/arithmetic.md#result-and-interpretation-14)

<a id="failures-and-useful-variations-11"></a>
- [Failures and useful variations](../reference/tensor/arithmetic.md#failures-and-useful-variations-11)

<a id="binary-comparison-expressions"></a>
- [Binary comparison expressions](../reference/tensor/arithmetic.md#binary-comparison-expressions)

<a id="complete-comparison-expression-example"></a>
- [Complete comparison-expression example](../reference/tensor/arithmetic.md#complete-comparison-expression-example)

<a id="goal-and-inputs-15"></a>
- [Goal and inputs](../reference/tensor/arithmetic.md#goal-and-inputs-15)

<a id="meaningful-lines-and-intermediate-results-1"></a>
- [Meaningful lines and intermediate results](../reference/tensor/arithmetic.md#meaningful-lines-and-intermediate-results-1)

<a id="result-and-interpretation-15"></a>
- [Result and interpretation](../reference/tensor/arithmetic.md#result-and-interpretation-15)

<a id="failures-and-useful-variations-12"></a>
- [Failures and useful variations](../reference/tensor/arithmetic.md#failures-and-useful-variations-12)

<a id="boolean-logical-expressions"></a>
- [Boolean logical expressions](../reference/tensor/arithmetic.md#boolean-logical-expressions)<span data-glossary-navigation="true"> ([Boolean glossary](../glossary.md#data-type--datatype))</span>

<a id="failures-and-useful-variations-13"></a>
- [Failures and useful variations](../reference/tensor/arithmetic.md#failures-and-useful-variations-13)

<a id="conditional-selection-expression"></a>
- [Conditional-selection expression](../reference/tensor/selection-casts.md#conditional-selection-expression)

<a id="complete-conditional-selection-example"></a>
- [Complete conditional-selection example](../reference/tensor/selection-casts.md#complete-conditional-selection-example)

<a id="goal-and-inputs-16"></a>
- [Goal and inputs](../reference/tensor/selection-casts.md#goal-and-inputs-16)

<a id="meaningful-lines-and-intermediate-results-2"></a>
- [Meaningful lines and intermediate results](../reference/tensor/selection-casts.md#meaningful-lines-and-intermediate-results-2)

<a id="result-and-interpretation-16"></a>
- [Result and interpretation](../reference/tensor/selection-casts.md#result-and-interpretation-16)

<a id="failures-and-useful-variations-14"></a>
- [Failures and useful variations](../reference/tensor/selection-casts.md#failures-and-useful-variations-14)

<a id="cast-expressions"></a>
- [Cast expressions](../reference/tensor/selection-casts.md#cast-expressions)

<a id="complete-cast-expression-example"></a>
- [Complete cast-expression example](../reference/tensor/selection-casts.md#complete-cast-expression-example)

<a id="goal-and-inputs-17"></a>
- [Goal and inputs](../reference/tensor/selection-casts.md#goal-and-inputs-17)

<a id="meaningful-lines-and-intermediate-results-3"></a>
- [Meaningful lines and intermediate results](../reference/tensor/selection-casts.md#meaningful-lines-and-intermediate-results-3)

<a id="result-and-interpretation-17"></a>
- [Result and interpretation](../reference/tensor/selection-casts.md#result-and-interpretation-17)

<a id="failures-and-useful-variations-15"></a>
- [Failures and useful variations](../reference/tensor/selection-casts.md#failures-and-useful-variations-15)

<a id="unary-numeric-transforms-and-floating-classifications"></a>
- [Unary numeric transforms and floating classifications](../reference/tensor/elementwise.md#unary-numeric-transforms-and-floating-classifications)<span data-glossary-navigation="true"> ([Unary glossary](../glossary.md#term-unary))</span>

<a id="complete-transform-and-classification-example"></a>
- [Complete transform-and-classification example](../reference/tensor/elementwise.md#complete-transform-and-classification-example)

<a id="goal-and-inputs-18"></a>
- [Goal and inputs](../reference/tensor/elementwise.md#goal-and-inputs-18)

<a id="meaningful-lines-and-intermediate-results-4"></a>
- [Meaningful lines and intermediate results](../reference/tensor/elementwise.md#meaningful-lines-and-intermediate-results-4)

<a id="result-and-interpretation-18"></a>
- [Result and interpretation](../reference/tensor/elementwise.md#result-and-interpretation-18)

<a id="failures-and-useful-variations-16"></a>
- [Failures and useful variations](../reference/tensor/elementwise.md#failures-and-useful-variations-16)

<a id="scalar-arithmetic-and-clamp-expressions"></a>
- [Scalar arithmetic and clamp expressions](../reference/tensor/elementwise.md#scalar-arithmetic-and-clamp-expressions)

<a id="complete-scalar-expression-example"></a>
- [Complete scalar-expression example](../reference/tensor/elementwise.md#complete-scalar-expression-example)

<a id="goal-and-inputs-19"></a>
- [Goal and inputs](../reference/tensor/elementwise.md#goal-and-inputs-19)

<a id="meaningful-lines-and-intermediate-results-5"></a>
- [Meaningful lines and intermediate results](../reference/tensor/elementwise.md#meaningful-lines-and-intermediate-results-5)

<a id="result-and-interpretation-19"></a>
- [Result and interpretation](../reference/tensor/elementwise.md#result-and-interpretation-19)

<a id="failures-and-useful-variations-17"></a>
- [Failures and useful variations](../reference/tensor/elementwise.md#failures-and-useful-variations-17)

<a id="numeric-aggregate-expressions"></a>
- [Numeric aggregate expressions](../reference/tensor/reductions.md#numeric-aggregate-expressions)

<a id="complete-numeric-aggregate-example"></a>
- [Complete numeric-aggregate example](../reference/tensor/reductions.md#complete-numeric-aggregate-example)

<a id="goal-and-inputs-20"></a>
- [Goal and inputs](../reference/tensor/reductions.md#goal-and-inputs-20)

<a id="meaningful-lines-and-intermediate-results-6"></a>
- [Meaningful lines and intermediate results](../reference/tensor/reductions.md#meaningful-lines-and-intermediate-results-6)

<a id="result-and-interpretation-20"></a>
- [Result and interpretation](../reference/tensor/reductions.md#result-and-interpretation-20)

<a id="failures-and-useful-variations-18"></a>
- [Failures and useful variations](../reference/tensor/reductions.md#failures-and-useful-variations-18)

<a id="binding-aware-sum-to-shape-expressions"></a>
- [Binding-aware sum-to-Shape expressions](../reference/tensor/reductions.md#binding-aware-sum-to-shape-expressions)<span data-glossary-navigation="true"> ([Binding glossary](../glossary.md#term-binding); [sum-to-Shape glossary](../glossary.md#sum-to-shape))</span>

<a id="complete-sum-to-shape-metadata-example"></a>
- [Complete sum-to-Shape metadata example](../reference/tensor/reductions.md#complete-sum-to-shape-metadata-example)<span data-glossary-navigation="true"> ([metadata glossary](../glossary.md#term-metadata))</span>

<a id="goal-and-inputs-21"></a>
- [Goal and inputs](../reference/tensor/reductions.md#goal-and-inputs-21)

<a id="result-and-interpretation-21"></a>
- [Result and interpretation](../reference/tensor/reductions.md#result-and-interpretation-21)

<a id="multi-axis-and-statistical-reduction-expressions"></a>
- [Multi-axis and statistical reduction expressions](../reference/tensor/reductions.md#multi-axis-and-statistical-reduction-expressions)<span data-glossary-navigation="true"> ([axis glossary](../glossary.md#term-axis))</span>

<a id="complete-multi-axis-metadata-example"></a>
- [Complete multi-axis metadata example](../reference/tensor/reductions.md#complete-multi-axis-metadata-example)

<a id="masked-sum-and-mean-expressions"></a>
- [Masked sum and mean expressions](../reference/tensor/masked-index-reductions.md#masked-sum-and-mean-expressions)<span data-glossary-navigation="true"> ([mean glossary](../glossary.md#term-mean))</span>

<a id="complete-masked-reduction-example"></a>
- [Complete masked-reduction example](../reference/tensor/masked-index-reductions.md#complete-masked-reduction-example)

<a id="goal-and-inputs-22"></a>
- [Goal and inputs](../reference/tensor/masked-index-reductions.md#goal-and-inputs-22)

<a id="meaningful-lines-and-intermediate-results-7"></a>
- [Meaningful lines and intermediate results](../reference/tensor/masked-index-reductions.md#meaningful-lines-and-intermediate-results-7)

<a id="result-and-interpretation-22"></a>
- [Result and interpretation](../reference/tensor/masked-index-reductions.md#result-and-interpretation-22)

<a id="failures-and-useful-variations-19"></a>
- [Failures and useful variations](../reference/tensor/masked-index-reductions.md#failures-and-useful-variations-19)

<a id="boolean-aggregate-expressions"></a>
- [Boolean aggregate expressions](../reference/tensor/masked-index-reductions.md#boolean-aggregate-expressions)

<a id="arg-extrema-expressions"></a>
- [Arg-extrema expressions](../reference/tensor/masked-index-reductions.md#arg-extrema-expressions)<span data-glossary-navigation="true"> ([extrema glossary](../glossary.md#saturation-pole-and-extremum))</span>

<a id="cumulative-scan-expressions"></a>
- [Cumulative-scan expressions](../reference/tensor/fixed-scans.md#cumulative-scan-expressions)<span data-glossary-navigation="true"> ([scan glossary](../glossary.md#term-cumulative-scan))</span>

<a id="fixed-recurrent-scan-expressions"></a>
- [Fixed recurrent-scan expressions](../reference/tensor/fixed-scans.md#fixed-recurrent-scan-expressions)

<a id="softmax-expressions"></a>
- [Softmax expressions](../reference/tensor/normalization.md#softmax-expressions)<span data-glossary-navigation="true"> ([Softmax glossary](../glossary.md#softmax--log-softmax))</span>

<a id="complete-expression-construction-example"></a>
- [Complete expression-construction example](../reference/tensor/normalization.md#complete-expression-construction-example)

<a id="goal-and-inputs-23"></a>
- [Goal and inputs](../reference/tensor/normalization.md#goal-and-inputs-23)

<a id="meaningful-lines-and-intermediate-results-8"></a>
- [Meaningful lines and intermediate results](../reference/tensor/normalization.md#meaningful-lines-and-intermediate-results-8)

<a id="result-and-interpretation-23"></a>
- [Result and interpretation](../reference/tensor/normalization.md#result-and-interpretation-23)

<a id="failures-and-useful-variations-20"></a>
- [Failures and useful variations](../reference/tensor/normalization.md#failures-and-useful-variations-20)

<a id="layer-normalization-expressions"></a>
- [Layer-normalization expressions](../reference/tensor/normalization.md#layer-normalization-expressions)<span data-glossary-navigation="true"> ([normalization glossary](../glossary.md#term-normalization))</span>

<a id="finite-numerical-example"></a>
- [Finite numerical example](../reference/tensor/normalization.md#finite-numerical-example)<span data-glossary-navigation="true"> ([Finite glossary](../glossary.md#term-finite))</span>

<a id="static-checks-and-symbolic-deferral"></a>
- [Static checks and symbolic deferral](../reference/tensor/normalization.md#static-checks-and-symbolic-deferral)

<a id="empty-and-special-value-policy"></a>
- [Empty and special-value policy](../reference/tensor/normalization.md#empty-and-special-value-policy)

<a id="result-identity-and-lifecycle-boundary"></a>
- [Result identity and lifecycle boundary](../reference/tensor/normalization.md#result-identity-and-lifecycle-boundary)<span data-glossary-navigation="true"> ([lifecycle glossary](../glossary.md#lifecycle))</span>

<a id="rms-normalization-expressions"></a>
- [RMS-normalization expressions](../reference/tensor/normalization.md#rms-normalization-expressions)

<a id="batch-normalization-inference-expressions"></a>
- [Batch-normalization inference expressions](../reference/tensor/normalization.md#batch-normalization-inference-expressions)<span data-glossary-navigation="true"> ([Batch glossary](../glossary.md#term-batch-machine-learning-and-hyperparameters); [inference glossary](../glossary.md#term-inference))</span>

<a id="finite-numerical-example-1"></a>
- [Finite numerical example](../reference/tensor/normalization.md#finite-numerical-example-1)

<a id="static-checks-symbolic-deferral-and-empty-results"></a>
- [Static checks, symbolic deferral, and empty results](../reference/tensor/normalization.md#static-checks-symbolic-deferral-and-empty-results)

<a id="negative-variance-and-special-values"></a>
- [Negative variance and special values](../reference/tensor/normalization.md#negative-variance-and-special-values)<span data-glossary-navigation="true"> ([variance glossary](../glossary.md#term-variance))</span>

<a id="metadata-provenance-and-lifecycle-boundary"></a>
- [Metadata, provenance, and lifecycle boundary](../reference/tensor/normalization.md#metadata-provenance-and-lifecycle-boundary)

<a id="general-scan-source-construction"></a>
- [General-scan source construction](../reference/tensor/general-scan.md#general-scan-source-construction)

<a id="formals-actual-positions-and-source-closure"></a>
- [Formals, actual positions and source closure](../reference/tensor/general-scan.md#formals-actual-positions-and-source-closure)

<a id="descriptor-only-ordinary-checks"></a>
- [Descriptor-only ordinary checks](../reference/tensor/general-scan.md#descriptor-only-ordinary-checks)

<a id="computed-logical-layout-completion"></a>
- [Computed logical-layout completion](../reference/tensor/general-scan.md#computed-logical-layout-completion)

<a id="per-output-ordinary-coordinate-facts"></a>
- [Per-output ordinary coordinate facts](../reference/tensor/general-scan.md#per-output-ordinary-coordinate-facts)<span data-glossary-navigation="true"> ([coordinate glossary](../glossary.md#axis-rank-scalar-vector-and-matrix))</span>

<a id="static-and-symbolic-construction-through-the-same-api"></a>
- [Static and symbolic construction through the same API](../reference/tensor/general-scan.md#static-and-symbolic-construction-through-the-same-api)

<a id="canonical-results-residual-metadata-and-eligibility"></a>
- [Canonical results, residual metadata and eligibility](../reference/tensor/general-scan.md#canonical-results-residual-metadata-and-eligibility)<span data-glossary-navigation="true"> ([Canonical glossary](../glossary.md#term-canonical-form); [residual glossary](../glossary.md#scan-residual-port-and-history-demand))</span>

<a id="length-aware-source-only-construction"></a>
- [length-aware-source-only-construction](../reference/tensor/general-scan.md#length-aware-source-only-construction)

<a id="length-aware-source-construction-and-compiler-support"></a>
- [length-aware-source-construction-and-compiler-support](../reference/tensor/general-scan.md#length-aware-source-construction-and-compiler-support)<span data-glossary-navigation="true"> ([compiler glossary](../glossary.md#term-compiler))</span>

<a id="length-aware-source-construction-and-qualified-execution"></a>
- [Length-aware source construction and qualified execution](../reference/tensor/general-scan.md#length-aware-source-construction-and-qualified-execution)

<a id="complete-metadata-construction-example"></a>
- [Complete metadata-construction example](../reference/tensor/general-scan.md#complete-metadata-construction-example)

<a id="batch-normalization-training-and-statistic-transition-expressions"></a>
- [Batch-normalization training and statistic-transition expressions](../reference/tensor/statistics-losses.md#batch-normalization-training-and-statistic-transition-expressions)<span data-glossary-navigation="true"> ([statistic glossary](../glossary.md#term-statistic))</span>

<a id="numerical-example"></a>
- [Numerical example](../reference/tensor/statistics-losses.md#numerical-example)

<a id="static-checks-deferred-obligations-and-empty-channels"></a>
- [Static checks, deferred obligations, and empty channels](../reference/tensor/statistics-losses.md#static-checks-deferred-obligations-and-empty-channels)

<a id="special-values-identity-effects-and-lifecycle-boundary"></a>
- [Special values, identity effects, and lifecycle boundary](../reference/tensor/statistics-losses.md#special-values-identity-effects-and-lifecycle-boundary)

<a id="mean-squared-error-loss-expressions"></a>
- [Mean-squared-error loss expressions](../reference/tensor/statistics-losses.md#mean-squared-error-loss-expressions)<span data-glossary-navigation="true"> ([loss glossary](../glossary.md#loss))</span>

<a id="categorical-cross-entropy-with-logits-expressions"></a>
- [Categorical-cross-entropy-with-logits expressions](../reference/tensor/statistics-losses.md#categorical-cross-entropy-with-logits-expressions)

<a id="contiguous-expressions"></a>
- [Contiguous expressions](../reference/tensor/reshape-expand.md#contiguous-expressions)<span data-glossary-navigation="true"> ([Contiguous glossary](../glossary.md#term-contiguous))</span>

<a id="complete-expression-construction-example-1"></a>
- [Complete expression-construction example](../reference/tensor/reshape-expand.md#complete-expression-construction-example-1)

<a id="goal-and-inputs-24"></a>
- [Goal and inputs](../reference/tensor/reshape-expand.md#goal-and-inputs-24)

<a id="meaningful-lines-and-intermediate-results-9"></a>
- [Meaningful lines and intermediate results](../reference/tensor/reshape-expand.md#meaningful-lines-and-intermediate-results-9)

<a id="result-and-interpretation-24"></a>
- [Result and interpretation](../reference/tensor/reshape-expand.md#result-and-interpretation-24)

<a id="failures-and-useful-variations-21"></a>
- [Failures and useful variations](../reference/tensor/reshape-expand.md#failures-and-useful-variations-21)

<a id="reshape-expressions"></a>
- [Reshape expressions](../reference/tensor/reshape-expand.md#reshape-expressions)

<a id="complete-expression-construction-example-2"></a>
- [Complete expression-construction example](../reference/tensor/reshape-expand.md#complete-expression-construction-example-2)

<a id="goal-and-inputs-25"></a>
- [Goal and inputs](../reference/tensor/reshape-expand.md#goal-and-inputs-25)

<a id="meaningful-lines-and-intermediate-results-10"></a>
- [Meaningful lines and intermediate results](../reference/tensor/reshape-expand.md#meaningful-lines-and-intermediate-results-10)

<a id="result-and-interpretation-25"></a>
- [Result and interpretation](../reference/tensor/reshape-expand.md#result-and-interpretation-25)

<a id="failures-and-useful-variations-22"></a>
- [Failures and useful variations](../reference/tensor/reshape-expand.md#failures-and-useful-variations-22)

<a id="expand-expressions"></a>
- [Expand expressions](../reference/tensor/reshape-expand.md#expand-expressions)

<a id="complete-expression-construction-example-3"></a>
- [Complete expression-construction example](../reference/tensor/reshape-expand.md#complete-expression-construction-example-3)

<a id="goal-and-inputs-26"></a>
- [Goal and inputs](../reference/tensor/reshape-expand.md#goal-and-inputs-26)

<a id="meaningful-lines-and-intermediate-results-11"></a>
- [Meaningful lines and intermediate results](../reference/tensor/reshape-expand.md#meaningful-lines-and-intermediate-results-11)

<a id="result-and-interpretation-26"></a>
- [Result and interpretation](../reference/tensor/reshape-expand.md#result-and-interpretation-26)

<a id="failures-and-useful-variations-23"></a>
- [Failures and useful variations](../reference/tensor/reshape-expand.md#failures-and-useful-variations-23)

<a id="permute-and-transpose-expressions"></a>
- [Permute and transpose expressions](../reference/tensor/axis-transforms.md#permute-and-transpose-expressions)<span data-glossary-navigation="true"> ([transpose glossary](../glossary.md#term-transpose))</span>

<a id="complete-expression-construction-example-4"></a>
- [Complete expression-construction example](../reference/tensor/axis-transforms.md#complete-expression-construction-example-4)

<a id="goal-and-inputs-27"></a>
- [Goal and inputs](../reference/tensor/axis-transforms.md#goal-and-inputs-27)

<a id="meaningful-lines-and-intermediate-results-12"></a>
- [Meaningful lines and intermediate results](../reference/tensor/axis-transforms.md#meaningful-lines-and-intermediate-results-12)

<a id="result-and-interpretation-27"></a>
- [Result and interpretation](../reference/tensor/axis-transforms.md#result-and-interpretation-27)

<a id="failures-and-useful-variations-24"></a>
- [Failures and useful variations](../reference/tensor/axis-transforms.md#failures-and-useful-variations-24)

<a id="expand-dimensions-and-squeeze-expressions"></a>
- [Expand-dimensions and squeeze expressions](../reference/tensor/axis-transforms.md#expand-dimensions-and-squeeze-expressions)

<a id="complete-rank-editing-example"></a>
- [Complete rank-editing example](../reference/tensor/axis-transforms.md#complete-rank-editing-example)<span data-glossary-navigation="true"> ([rank glossary](../glossary.md#term-rank))</span>

<a id="goal-and-inputs-28"></a>
- [Goal and inputs](../reference/tensor/axis-transforms.md#goal-and-inputs-28)

<a id="meaningful-lines-and-intermediate-results-13"></a>
- [Meaningful lines and intermediate results](../reference/tensor/axis-transforms.md#meaningful-lines-and-intermediate-results-13)

<a id="result-and-interpretation-28"></a>
- [Result and interpretation](../reference/tensor/axis-transforms.md#result-and-interpretation-28)

<a id="failures-and-useful-variations-25"></a>
- [Failures and useful variations](../reference/tensor/axis-transforms.md#failures-and-useful-variations-25)

<a id="slice-expressions"></a>
- [Slice expressions](../reference/tensor/slicing.md#slice-expressions)<span data-glossary-navigation="true"> ([Slice glossary](../glossary.md#slice))</span>

<a id="length-defined-extraction"></a>
- [Length-defined extraction](../reference/tensor/slicing.md#length-defined-extraction)

<a id="complete-slice-expression-example"></a>
- [Complete slice-expression example](../reference/tensor/slicing.md#complete-slice-expression-example)

<a id="goal-and-inputs-29"></a>
- [Goal and inputs](../reference/tensor/slicing.md#goal-and-inputs-29)

<a id="meaningful-lines-and-intermediate-results-14"></a>
- [Meaningful lines and intermediate results](../reference/tensor/slicing.md#meaningful-lines-and-intermediate-results-14)

<a id="result-and-interpretation-29"></a>
- [Result and interpretation](../reference/tensor/slicing.md#result-and-interpretation-29)

<a id="failures-and-useful-variations-26"></a>
- [Failures and useful variations](../reference/tensor/slicing.md#failures-and-useful-variations-26)

<a id="slice-update-and-target-relative-crop-expressions"></a>
- [Slice update and target-relative crop expressions](../reference/tensor/slicing.md#slice-update-and-target-relative-crop-expressions)<span data-glossary-navigation="true"> ([Slice update glossary](../glossary.md#slice-update); [target-relative crop glossary](../glossary.md#target-relative-crop))</span>

<a id="complete-placement-metadata-example"></a>
- [Complete placement-metadata example](../reference/tensor/slicing.md#complete-placement-metadata-example)

<a id="goal-and-inputs-30"></a>
- [Goal and inputs](../reference/tensor/slicing.md#goal-and-inputs-30)

<a id="meaningful-lines-and-intermediate-results-15"></a>
- [Meaningful lines and intermediate results](../reference/tensor/slicing.md#meaningful-lines-and-intermediate-results-15)

<a id="result-and-interpretation-30"></a>
- [Result and interpretation](../reference/tensor/slicing.md#result-and-interpretation-30)

<a id="failures-and-useful-variations-27"></a>
- [Failures and useful variations](../reference/tensor/slicing.md#failures-and-useful-variations-27)

<a id="complete-length-defined-and-symbolic-placement-metadata-example"></a>
- [Complete length-defined and symbolic-placement metadata example](../reference/tensor/slicing.md#complete-length-defined-and-symbolic-placement-metadata-example)

<a id="goal-and-inputs-31"></a>
- [Goal and inputs](../reference/tensor/slicing.md#goal-and-inputs-31)

<a id="meaningful-lines-and-intermediate-results-16"></a>
- [Meaningful lines and intermediate results](../reference/tensor/slicing.md#meaningful-lines-and-intermediate-results-16)

<a id="result-and-interpretation-31"></a>
- [Result and interpretation](../reference/tensor/slicing.md#result-and-interpretation-31)

<a id="failures-and-useful-variations-28"></a>
- [Failures and useful variations](../reference/tensor/slicing.md#failures-and-useful-variations-28)

<a id="scalar-select-expressions"></a>
- [Scalar select expressions](../reference/tensor/gather.md#scalar-select-expressions)<span data-glossary-navigation="true"> ([Scalar select glossary](../glossary.md#scalar-select))</span>

<a id="complete-scalar-select-expression-example"></a>
- [Complete scalar-select expression example](../reference/tensor/gather.md#complete-scalar-select-expression-example)

<a id="goal-and-inputs-32"></a>
- [Goal and inputs](../reference/tensor/gather.md#goal-and-inputs-32)

<a id="meaningful-lines-and-intermediate-results-17"></a>
- [Meaningful lines and intermediate results](../reference/tensor/gather.md#meaningful-lines-and-intermediate-results-17)

<a id="result-and-interpretation-32"></a>
- [Result and interpretation](../reference/tensor/gather.md#result-and-interpretation-32)

<a id="failures-and-useful-variations-29"></a>
- [Failures and useful variations](../reference/tensor/gather.md#failures-and-useful-variations-29)

<a id="axis-gather-expressions"></a>
- [Axis-gather expressions](../reference/tensor/gather.md#axis-gather-expressions)<span data-glossary-navigation="true"> ([gather glossary](../glossary.md#gather))</span>

<a id="embedding-convenience"></a>
- [Embedding convenience](../reference/tensor/gather.md#embedding-convenience)<span data-glossary-navigation="true"> ([Embedding glossary](../glossary.md#embedding))</span>

<a id="complete-embedding-metadata-example"></a>
- [Complete embedding metadata example](../reference/tensor/gather.md#complete-embedding-metadata-example)

<a id="complete-axis-gather-expression-example"></a>
- [Complete axis-gather expression example](../reference/tensor/gather.md#complete-axis-gather-expression-example)

<a id="goal-and-inputs-33"></a>
- [Goal and inputs](../reference/tensor/gather.md#goal-and-inputs-33)

<a id="meaningful-lines-and-intermediate-results-18"></a>
- [Meaningful lines and intermediate results](../reference/tensor/gather.md#meaningful-lines-and-intermediate-results-18)

<a id="result-and-interpretation-33"></a>
- [Result and interpretation](../reference/tensor/gather.md#result-and-interpretation-33)

<a id="failures-and-useful-variations-30"></a>
- [Failures and useful variations](../reference/tensor/gather.md#failures-and-useful-variations-30)

<a id="one-hot-encoding-expressions"></a>
- [One-hot encoding expressions](../reference/tensor/gather.md#one-hot-encoding-expressions)<span data-glossary-navigation="true"> ([One-hot encoding glossary](../glossary.md#one-hot-encoding))</span>

<a id="conceptual-vector-and-scalar-examples"></a>
- [Conceptual vector and scalar examples](../reference/tensor/gather.md#conceptual-vector-and-scalar-examples)<span data-glossary-navigation="true"> ([vector glossary](../glossary.md#term-vector))</span>

<a id="failures-and-execution-boundary"></a>
- [Failures and execution boundary](../reference/tensor/gather.md#failures-and-execution-boundary)

<a id="gather-nd-expressions"></a>
- [Gather-ND expressions](../reference/tensor/scatter.md#gather-nd-expressions)<span data-glossary-navigation="true"> ([Gather-ND glossary](../glossary.md#gather-nd))</span>

<a id="complete-gather-nd-expression-example"></a>
- [Complete Gather-ND expression example](../reference/tensor/scatter.md#complete-gather-nd-expression-example)

<a id="goal-and-inputs-34"></a>
- [Goal and inputs](../reference/tensor/scatter.md#goal-and-inputs-34)

<a id="meaningful-lines-and-intermediate-results-19"></a>
- [Meaningful lines and intermediate results](../reference/tensor/scatter.md#meaningful-lines-and-intermediate-results-19)

<a id="result-and-interpretation-34"></a>
- [Result and interpretation](../reference/tensor/scatter.md#result-and-interpretation-34)

<a id="failures-and-useful-variations-31"></a>
- [Failures and useful variations](../reference/tensor/scatter.md#failures-and-useful-variations-31)

<a id="gather-compatible-scatter-add-expressions"></a>
- [Gather-compatible Scatter Add expressions](../reference/tensor/scatter.md#gather-compatible-scatter-add-expressions)<span data-glossary-navigation="true"> ([Scatter Add glossary](../glossary.md#scatter-add))</span>

<a id="complete-scatter-add-metadata-example"></a>
- [Complete Scatter Add metadata example](../reference/tensor/scatter.md#complete-scatter-add-metadata-example)

<a id="scatter-elements-expressions"></a>
- [Scatter Elements expressions](../reference/tensor/scatter.md#scatter-elements-expressions)<span data-glossary-navigation="true"> ([Scatter Elements glossary](../glossary.md#scatter-elements))</span>

<a id="complete-scatter-elements-expression-example"></a>
- [Complete Scatter Elements expression example](../reference/tensor/scatter.md#complete-scatter-elements-expression-example)

<a id="goal-and-inputs-35"></a>
- [Goal and inputs](../reference/tensor/scatter.md#goal-and-inputs-35)

<a id="result-and-interpretation-35"></a>
- [Result and interpretation](../reference/tensor/scatter.md#result-and-interpretation-35)

<a id="failures-and-useful-variations-32"></a>
- [Failures and useful variations](../reference/tensor/scatter.md#failures-and-useful-variations-32)

<a id="scatter-nd-expressions"></a>
- [Scatter-ND expressions](../reference/tensor/scatter.md#scatter-nd-expressions)<span data-glossary-navigation="true"> ([Scatter-ND glossary](../glossary.md#scatter-nd))</span>

<a id="complete-scatter-nd-expression-example"></a>
- [Complete Scatter-ND expression example](../reference/tensor/scatter.md#complete-scatter-nd-expression-example)

<a id="goal-and-inputs-36"></a>
- [Goal and inputs](../reference/tensor/scatter.md#goal-and-inputs-36)

<a id="meaningful-lines-and-intermediate-results-20"></a>
- [Meaningful lines and intermediate results](../reference/tensor/scatter.md#meaningful-lines-and-intermediate-results-20)

<a id="result-and-interpretation-36"></a>
- [Result and interpretation](../reference/tensor/scatter.md#result-and-interpretation-36)

<a id="failures-and-useful-variations-33"></a>
- [Failures and useful variations](../reference/tensor/scatter.md#failures-and-useful-variations-33)

<a id="pad-and-tile-expressions"></a>
- [Pad and tile expressions](../reference/tensor/composition.md#pad-and-tile-expressions)

<a id="symbolic-shape-examples"></a>
- [Symbolic Shape examples](../reference/tensor/composition.md#symbolic-shape-examples)

<a id="goal-and-inputs-37"></a>
- [Goal and inputs](../reference/tensor/composition.md#goal-and-inputs-37)

<a id="result-and-interpretation-37"></a>
- [Result and interpretation](../reference/tensor/composition.md#result-and-interpretation-37)

<a id="complete-pad-and-tile-expression-example"></a>
- [Complete pad-and-tile expression example](../reference/tensor/composition.md#complete-pad-and-tile-expression-example)

<a id="goal-and-inputs-38"></a>
- [Goal and inputs](../reference/tensor/composition.md#goal-and-inputs-38)

<a id="meaningful-lines-and-intermediate-results-21"></a>
- [Meaningful lines and intermediate results](../reference/tensor/composition.md#meaningful-lines-and-intermediate-results-21)

<a id="result-and-interpretation-38"></a>
- [Result and interpretation](../reference/tensor/composition.md#result-and-interpretation-38)

<a id="failures-and-useful-variations-34"></a>
- [Failures and useful variations](../reference/tensor/composition.md#failures-and-useful-variations-34)

<a id="tensor-composition-expressions"></a>
- [Tensor composition expressions](../reference/tensor/composition.md#tensor-composition-expressions)<span data-glossary-navigation="true"> ([Tensor composition glossary](../glossary.md#tensor-composition))</span>

<a id="complete-tensor-composition-expression-example"></a>
- [Complete tensor-composition expression example](../reference/tensor/composition.md#complete-tensor-composition-expression-example)

<a id="goal-and-inputs-39"></a>
- [Goal and inputs](../reference/tensor/composition.md#goal-and-inputs-39)

<a id="result-and-interpretation-39"></a>
- [Result and interpretation](../reference/tensor/composition.md#result-and-interpretation-39)

<a id="failures-and-useful-variations-35"></a>
- [Failures and useful variations](../reference/tensor/composition.md#failures-and-useful-variations-35)

<a id="window-transform-expressions"></a>
- [Window-transform expressions](../reference/tensor/windows.md#window-transform-expressions)

<a id="complete-window-transform-expression-example"></a>
- [Complete window-transform expression example](../reference/tensor/windows.md#complete-window-transform-expression-example)

<a id="matrix-multiplication-expressions"></a>
- [Matrix-multiplication expressions](../reference/tensor/matmul-linear.md#matrix-multiplication-expressions)<span data-glossary-navigation="true"> ([Matrix glossary](../glossary.md#term-matrix-meanings))</span>

<a id="concrete-matrix-example"></a>
- [Concrete matrix example](../reference/tensor/matmul-linear.md#concrete-matrix-example)

<a id="dynamic-dimensions-and-local-failures"></a>
- [Dynamic dimensions and local failures](../reference/tensor/matmul-linear.md#dynamic-dimensions-and-local-failures)

<a id="data-type-numerical-meaning-and-result-metadata"></a>
- [Data type, numerical meaning, and result metadata](../reference/tensor/matmul-linear.md#data-type-numerical-meaning-and-result-metadata)<span data-glossary-navigation="true"> ([Data type glossary](../glossary.md#data-type--datatype))</span>

<a id="linear-projection-convenience"></a>
- [Linear-projection convenience](../reference/tensor/matmul-linear.md#linear-projection-convenience)<span data-glossary-navigation="true"> ([Linear glossary](../glossary.md#term-linear-meanings))</span>

<a id="complete-producer-chain-example"></a>
- [Complete producer-chain example](../reference/tensor/matmul-linear.md#complete-producer-chain-example)<span data-glossary-navigation="true"> ([producer glossary](../glossary.md#term-producer))</span>

<a id="scaled-dot-product-attention-expressions"></a>
- [Scaled dot-product attention expressions](../reference/tensor/attention.md#scaled-dot-product-attention-expressions)<span data-glossary-navigation="true"> ([Scaled dot-product attention glossary](../glossary.md#scaled-dot-product-attention))</span>

<a id="output-slots-and-shared-occurrence"></a>
- [Output slots and shared occurrence](../reference/tensor/attention.md#output-slots-and-shared-occurrence)<span data-glossary-navigation="true"> ([occurrence glossary](../glossary.md#term-occurrence))</span>

<a id="shape-type-and-deferred-checks"></a>
- [Shape, type, and deferred checks](../reference/tensor/attention.md#shape-type-and-deferred-checks)

<a id="eligibility-and-special-values"></a>
- [Eligibility and special values](../reference/tensor/attention.md#eligibility-and-special-values)

<a id="runnable-output-and-weights-metadata-example"></a>
- [Runnable output-and-weights metadata example](../reference/tensor/attention.md#runnable-output-and-weights-metadata-example)<span data-glossary-navigation="true"> ([weights glossary](../glossary.md#term-weight))</span>

<a id="grouped-ncw-conv1d-composition"></a>
- [Grouped NCW Conv1d composition](../reference/tensor/convolution.md#grouped-ncw-conv1d-composition)<span data-glossary-navigation="true"> ([NCW glossary](../glossary.md#term-ncw); [Conv1d composition glossary](../glossary.md#conv1d-composition))</span>

<a id="biased-grouped-conv1d-metadata-example"></a>
- [Biased grouped Conv1d metadata example](../reference/tensor/convolution.md#biased-grouped-conv1d-metadata-example)

<a id="grouped-nchw-conv2d-expressions"></a>
- [Grouped NCHW Conv2d expressions](../reference/tensor/convolution.md#grouped-nchw-conv2d-expressions)<span data-glossary-navigation="true"> ([NCHW glossary](../glossary.md#term-nchw); [Conv2d glossary](../glossary.md#convolution--conv2d))</span>

<a id="grouped-shape-and-provenance-example"></a>
- [Grouped Shape and provenance example](../reference/tensor/convolution.md#grouped-shape-and-provenance-example)

<a id="dynamic-height-example"></a>
- [Dynamic-height example](../reference/tensor/convolution.md#dynamic-height-example)

<a id="numerical-and-metadata-policy"></a>
- [Numerical and metadata policy](../reference/tensor/convolution.md#numerical-and-metadata-policy)

<a id="grouped-ncdhw-conv3d-expressions"></a>
- [Grouped NCDHW Conv3d expressions](../reference/tensor/convolution.md#grouped-ncdhw-conv3d-expressions)<span data-glossary-navigation="true"> ([NCDHW glossary](../glossary.md#conv3d--ncdhw); [Conv3d glossary](../glossary.md#conv3d--ncdhw))</span>

<a id="biased-grouped-conv3d-metadata-example"></a>
- [Biased grouped Conv3d metadata example](../reference/tensor/convolution.md#biased-grouped-conv3d-metadata-example)

<a id="numerical-and-provenance-policy"></a>
- [Numerical and provenance policy](../reference/tensor/convolution.md#numerical-and-provenance-policy)

<a id="ncw-pool1d-composition"></a>
- [NCW Pool1d composition](../reference/tensor/pooling.md#ncw-pool1d-composition)<span data-glossary-navigation="true"> ([Pool1d composition glossary](../glossary.md#pool1d-composition))</span>

<a id="pool1d-shape-and-provenance-example"></a>
- [Pool1d Shape and provenance example](../reference/tensor/pooling.md#pool1d-shape-and-provenance-example)

<a id="nchw-maximum-pooling-expressions"></a>
- [NCHW maximum-pooling expressions](../reference/tensor/pooling.md#nchw-maximum-pooling-expressions)<span data-glossary-navigation="true"> ([pooling glossary](../glossary.md#term-pooling))</span>

<a id="static-floor-and-literal-ceil-example"></a>
- [Static floor and literal-ceil example](../reference/tensor/pooling.md#static-floor-and-literal-ceil-example)

<a id="symbolic-floor-and-ceil-example"></a>
- [Symbolic floor and ceil example](../reference/tensor/pooling.md#symbolic-floor-and-ceil-example)

<a id="extrema-and-padding-policy"></a>
- [Extrema and padding policy](../reference/tensor/pooling.md#extrema-and-padding-policy)<span data-glossary-navigation="true"> ([padding glossary](../glossary.md#padding))</span>

<a id="nchw-average-pooling-expressions"></a>
- [NCHW average-pooling expressions](../reference/tensor/pooling.md#nchw-average-pooling-expressions)

<a id="static-floor-literal-ceil-and-symbolic-examples"></a>
- [Static floor, literal-ceil, and symbolic examples](../reference/tensor/pooling.md#static-floor-literal-ceil-and-symbolic-examples)

<a id="count-padding-numerical-example"></a>
- [Count-padding numerical example](../reference/tensor/pooling.md#count-padding-numerical-example)

<a id="numerical-empty-and-metadata-policy"></a>
- [Numerical, empty, and metadata policy](../reference/tensor/pooling.md#numerical-empty-and-metadata-policy)

<a id="ncdhw-pool3d-expressions"></a>
- [NCDHW Pool3d expressions](../reference/tensor/pooling.md#ncdhw-pool3d-expressions)<span data-glossary-navigation="true"> ([Pool3d glossary](../glossary.md#pool3d--ncdhw-pooling))</span>

<a id="pool3d-metadata-example"></a>
- [Pool3d metadata example](../reference/tensor/pooling.md#pool3d-metadata-example)

<a id="matrix-multiplication-semantic-kind"></a>
- [Matrix-multiplication semantic kind](../reference/tensor/sort-topk.md#matrix-multiplication-semantic-kind)

<a id="stable-sort-and-argsort-expressions"></a>
- [Stable sort and argsort expressions](../reference/tensor/sort-topk.md#stable-sort-and-argsort-expressions)

<a id="complete-metadata-example"></a>
- [Complete metadata example](../reference/tensor/sort-topk.md#complete-metadata-example)

<a id="top-k-values-and-indices"></a>
- [Top-K values and indices](../reference/tensor/sort-topk.md#top-k-values-and-indices)

<a id="complete-construction-example"></a>
- [Complete construction example](../reference/tensor/sort-topk.md#complete-construction-example)

<a id="host-visible-storage"></a>
- [Host-visible storage](../reference/tensor/host-storage.md#host-visible-storage)<span data-glossary-navigation="true"> ([Host glossary](../glossary.md#term-host))</span>

<a id="borrowed-lifetime-example"></a>
- [Borrowed-lifetime example](../reference/tensor/host-storage.md#borrowed-lifetime-example)<span data-glossary-navigation="true"> ([Borrowed glossary](../glossary.md#ownership-borrowing-and-lease); [lifetime glossary](../glossary.md#term-lifetime))</span>

<a id="goal-and-inputs-40"></a>
- [Goal and inputs](../reference/tensor/host-storage.md#goal-and-inputs-40)

<a id="meaningful-lines-11"></a>
- [Meaningful lines](../reference/tensor/host-storage.md#meaningful-lines-11)

<a id="result-and-interpretation-40"></a>
- [Result and interpretation](../reference/tensor/host-storage.md#result-and-interpretation-40)

<a id="failures-and-useful-variations-36"></a>
- [Failures and useful variations](../reference/tensor/host-storage.md#failures-and-useful-variations-36)

<a id="typed-identifiers"></a>
- [Typed identifiers](../reference/tensor/identifiers.md#typed-identifiers)

<a id="operation-semantic-foundation"></a>
- [Operation semantic foundation](../reference/tensor/operation-foundation.md#operation-semantic-foundation)<span data-glossary-navigation="true"> ([Operation glossary](../glossary.md#operation))</span>

<a id="binary-arithmetic-semantic-kinds"></a>
- [Binary arithmetic semantic kinds](../reference/tensor/operation-arithmetic.md#binary-arithmetic-semantic-kinds)

<a id="binary-comparison-semantic-kinds"></a>
- [Binary comparison semantic kinds](../reference/tensor/operation-arithmetic.md#binary-comparison-semantic-kinds)

<a id="boolean-logical-semantic-kinds"></a>
- [Boolean logical semantic kinds](../reference/tensor/operation-arithmetic.md#boolean-logical-semantic-kinds)

<a id="where-selection-semantic-kind"></a>
- [Where selection semantic kind](../reference/tensor/operation-arithmetic.md#where-selection-semantic-kind)

<a id="cast-semantic-kind-and-attributes"></a>
- [Cast semantic kind and attributes](../reference/tensor/operation-arithmetic.md#cast-semantic-kind-and-attributes)

<a id="aggregate-reduction-semantic-kinds-and-attributes"></a>
- [Aggregate reduction semantic kinds and attributes](../reference/tensor/operation-reductions.md#aggregate-reduction-semantic-kinds-and-attributes)<span data-glossary-navigation="true"> ([Aggregate reduction glossary](../glossary.md#aggregate-reduction))</span>

<a id="cumulative-scan-semantic-kinds-and-attributes"></a>
- [Cumulative-scan semantic kinds and attributes](../reference/tensor/operation-reductions.md#cumulative-scan-semantic-kinds-and-attributes)

<a id="softmax-semantic-kinds-and-attributes"></a>
- [Softmax semantic kinds and attributes](../reference/tensor/operation-reductions.md#softmax-semantic-kinds-and-attributes)

<a id="loss-semantic-kind-reduction-and-attributes"></a>
- [Loss semantic kind, reduction, and attributes](../reference/tensor/operation-reductions.md#loss-semantic-kind-reduction-and-attributes)

<a id="contiguous-semantic-kind"></a>
- [Contiguous semantic kind](../reference/tensor/operation-geometry.md#contiguous-semantic-kind)

<a id="reshape-and-expand-semantic-kinds-and-target-shape-attributes"></a>
- [Reshape and expand semantic kinds and target-shape attributes](../reference/tensor/operation-geometry.md#reshape-and-expand-semantic-kinds-and-target-shape-attributes)<span data-glossary-navigation="true"> ([target glossary](../glossary.md#term-target))</span>

<a id="axis-transform-semantic-kinds-and-attributes"></a>
- [Axis-transform semantic kinds and attributes](../reference/tensor/operation-geometry.md#axis-transform-semantic-kinds-and-attributes)

<a id="output-to-input-order-example"></a>
- [Output-to-input order example](../reference/tensor/operation-geometry.md#output-to-input-order-example)

<a id="goal-and-inputs-41"></a>
- [Goal and inputs](../reference/tensor/operation-geometry.md#goal-and-inputs-41)

<a id="result-and-interpretation-41"></a>
- [Result and interpretation](../reference/tensor/operation-geometry.md#result-and-interpretation-41)

<a id="failure-variation"></a>
- [Failure variation](../reference/tensor/operation-geometry.md#failure-variation)

<a id="slice-semantic-kind-and-normalized-attributes"></a>
- [Slice semantic kind and normalized attributes](../reference/tensor/operation-geometry.md#slice-semantic-kind-and-normalized-attributes)

<a id="parallel-half-open-example"></a>
- [Parallel half-open example](../reference/tensor/operation-geometry.md#parallel-half-open-example)<span data-glossary-navigation="true"> ([half-open glossary](../glossary.md#term-half-open-interval))</span>

<a id="goal-and-inputs-42"></a>
- [Goal and inputs](../reference/tensor/operation-geometry.md#goal-and-inputs-42)

<a id="meaningful-entries-and-result"></a>
- [Meaningful entries and result](../reference/tensor/operation-geometry.md#meaningful-entries-and-result)

<a id="ownership-validation-and-useful-variations"></a>
- [Ownership, validation, and useful variations](../reference/tensor/operation-geometry.md#ownership-validation-and-useful-variations)<span data-glossary-navigation="true"> ([Ownership glossary](../glossary.md#term-ownership-meanings); [validation glossary](../glossary.md#term-validation))</span>

<a id="pad-and-tile-semantic-kinds-and-normalized-attributes"></a>
- [Pad and tile semantic kinds and normalized attributes](../reference/tensor/operation-indexing.md#pad-and-tile-semantic-kinds-and-normalized-attributes)

<a id="constant-padding-example"></a>
- [Constant-padding example](../reference/tensor/operation-indexing.md#constant-padding-example)

<a id="goal-and-inputs-43"></a>
- [Goal and inputs](../reference/tensor/operation-indexing.md#goal-and-inputs-43)

<a id="result-and-interpretation-42"></a>
- [Result and interpretation](../reference/tensor/operation-indexing.md#result-and-interpretation-42)

<a id="complete-pattern-tiling-example"></a>
- [Complete-pattern tiling example](../reference/tensor/operation-indexing.md#complete-pattern-tiling-example)<span data-glossary-navigation="true"> ([tiling glossary](../glossary.md#tiling))</span>

<a id="goal-and-inputs-44"></a>
- [Goal and inputs](../reference/tensor/operation-indexing.md#goal-and-inputs-44)

<a id="result-and-interpretation-43"></a>
- [Result and interpretation](../reference/tensor/operation-indexing.md#result-and-interpretation-43)

<a id="ownership-validation-and-useful-variations-1"></a>
- [Ownership, validation, and useful variations](../reference/tensor/operation-indexing.md#ownership-validation-and-useful-variations-1)

<a id="tensor-composition-semantic-kinds-and-attributes"></a>
- [Tensor composition semantic kinds and attributes](../reference/tensor/operation-indexing.md#tensor-composition-semantic-kinds-and-attributes)

<a id="scalar-select-semantic-kind-and-attributes"></a>
- [Scalar select semantic kind and attributes](../reference/tensor/operation-indexing.md#scalar-select-semantic-kind-and-attributes)

<a id="scalar-axis-removal-example"></a>
- [Scalar-axis removal example](../reference/tensor/operation-indexing.md#scalar-axis-removal-example)

<a id="goal-and-inputs-45"></a>
- [Goal and inputs](../reference/tensor/operation-indexing.md#goal-and-inputs-45)

<a id="result-and-interpretation-44"></a>
- [Result and interpretation](../reference/tensor/operation-indexing.md#result-and-interpretation-44)

<a id="validation-and-boundaries"></a>
- [Validation and boundaries](../reference/tensor/operation-indexing.md#validation-and-boundaries)

<a id="gather-and-gather-elements-semantic-kinds-and-attributes"></a>
- [Gather and Gather Elements semantic kinds and attributes](../reference/tensor/operation-indexing.md#gather-and-gather-elements-semantic-kinds-and-attributes)<span data-glossary-navigation="true"> ([Gather Elements glossary](../glossary.md#gather-elements))</span>

<a id="one-hot-semantic-kind-and-attributes"></a>
- [One-hot semantic kind and attributes](../reference/tensor/operation-indexing.md#one-hot-semantic-kind-and-attributes)

<a id="gather-nd-semantic-kind-and-attributes"></a>
- [Gather-ND semantic kind and attributes](../reference/tensor/operation-indexing.md#gather-nd-semantic-kind-and-attributes)

<a id="tuple-index-shape-examples"></a>
- [Tuple-index Shape examples](../reference/tensor/operation-indexing.md#tuple-index-shape-examples)

<a id="goal-and-inputs-46"></a>
- [Goal and inputs](../reference/tensor/operation-indexing.md#goal-and-inputs-46)

<a id="results-and-interpretation"></a>
- [Results and interpretation](../reference/tensor/operation-indexing.md#results-and-interpretation)

<a id="validation-and-boundaries-1"></a>
- [Validation and boundaries](../reference/tensor/operation-indexing.md#validation-and-boundaries-1)

<a id="axis-scatter-semantic-kinds-reduction-and-attributes"></a>
- [Axis-scatter semantic kinds, reduction, and attributes](../reference/tensor/operation-indexing.md#axis-scatter-semantic-kinds-reduction-and-attributes)

<a id="scatter-nd-semantic-kind-and-attributes"></a>
- [Scatter-ND semantic kind and attributes](../reference/tensor/operation-indexing.md#scatter-nd-semantic-kind-and-attributes)

<a id="tuple-index-update-shape-examples"></a>
- [Tuple-index update Shape examples](../reference/tensor/operation-indexing.md#tuple-index-update-shape-examples)

<a id="window-transform-semantic-kinds-and-attributes"></a>
- [Window-transform semantic kinds and attributes](../reference/tensor/operation-windows.md#window-transform-semantic-kinds-and-attributes)

<a id="general-axis-unfold-and-fold"></a>
- [General-axis unfold and fold](../reference/tensor/operation-windows.md#general-axis-unfold-and-fold)<span data-glossary-navigation="true"> ([unfold and fold glossary](../glossary.md#window-transform))</span>

<a id="goal-and-inputs-47"></a>
- [Goal and inputs](../reference/tensor/operation-windows.md#goal-and-inputs-47)

<a id="result-and-interpretation-45"></a>
- [Result and interpretation](../reference/tensor/operation-windows.md#result-and-interpretation-45)

<a id="validation-and-ownership"></a>
- [Validation and ownership](../reference/tensor/operation-windows.md#validation-and-ownership)

<a id="nchw-im2col-and-overlap-summing-col2im"></a>
- [NCHW im2col and overlap-summing col2im](../reference/tensor/operation-windows.md#nchw-im2col-and-overlap-summing-col2im)<span data-glossary-navigation="true"> ([im2col glossary](../glossary.md#term-im2col); [col2im glossary](../glossary.md#term-col2im))</span>

<a id="goal-and-inputs-48"></a>
- [Goal and inputs](../reference/tensor/operation-windows.md#goal-and-inputs-48)

<a id="geometry-and-result"></a>
- [Geometry and result](../reference/tensor/operation-windows.md#geometry-and-result)

<a id="structural-boundary"></a>
- [Structural boundary](../reference/tensor/operation-windows.md#structural-boundary)

<a id="unary-elementwise-semantic-kinds"></a>
- [Unary elementwise semantic kinds](../reference/tensor/operation-elementwise.md#unary-elementwise-semantic-kinds)<span data-glossary-navigation="true"> ([elementwise glossary](../glossary.md#term-elementwise))</span>

<a id="floating-classification-semantic-kinds"></a>
- [Floating-classification semantic kinds](../reference/tensor/operation-elementwise.md#floating-classification-semantic-kinds)

<a id="scalar-arithmetic-and-clamp-semantic-kinds"></a>
- [Scalar arithmetic and clamp semantic kinds](../reference/tensor/operation-elementwise.md#scalar-arithmetic-and-clamp-semantic-kinds)

<a id="test-local-conceptual-example"></a>
- [Test-local conceptual example](../reference/tensor/operation-elementwise.md#test-local-conceptual-example)

<a id="graph-values-and-compiled-nodes"></a>
- [Graph values and compiled nodes](../reference/tensor/compiled-graph.md#graph-values-and-compiled-nodes)<span data-glossary-navigation="true"> ([Graph glossary](../glossary.md#graph); [nodes glossary](../glossary.md#node))</span>

<a id="complete-test-local-graph-model-example"></a>
- [Complete test-local graph-model example](../reference/tensor/compiled-graph.md#complete-test-local-graph-model-example)<span data-glossary-navigation="true"> ([model glossary](../glossary.md#term-model))</span>

<a id="goal-and-inputs-49"></a>
- [Goal and inputs](../reference/tensor/compiled-graph.md#goal-and-inputs-49)

<a id="meaningful-lines-12"></a>
- [Meaningful lines](../reference/tensor/compiled-graph.md#meaningful-lines-12)

<a id="result-and-interpretation-46"></a>
- [Result and interpretation](../reference/tensor/compiled-graph.md#result-and-interpretation-46)

<a id="failures-and-useful-variations-37"></a>
- [Failures and useful variations](../reference/tensor/compiled-graph.md#failures-and-useful-variations-37)

<a id="planned-contracts"></a>
- [Planned contracts](../history/reference/tensor/historical-status.md#planned-contracts)

<a id="failures-and-ownership-summary"></a>
- [Failures and ownership summary](../reference/tensor/failures.md#failures-and-ownership-summary)
