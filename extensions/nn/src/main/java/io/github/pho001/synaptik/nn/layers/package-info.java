/**
 * Provides stateful neural-network layers that own named parameters and compose Model Tensor
 * expressions.
 *
 * <p>A layer in this package is a {@link io.github.pho001.synaptik.nn.module.Module}: it owns the
 * parameter wrappers used by later forward construction, while generic Tensor operation meaning
 * remains in {@code modules/model}. Calling a forward method builds storage-free expression
 * metadata. It does not evaluate values, compile a graph, select a backend, or execute work.</p>
 *
 * <p>{@link io.github.pho001.synaptik.nn.layers.Linear} uses the conventional
 * {@code [outFeatures, inFeatures]} weight orientation and delegates each forward call to
 * {@link io.github.pho001.synaptik.model.tensor.Tensor#linear(
 * io.github.pho001.synaptik.model.tensor.Tensor)} or its biased overload. Its behavior is
 * identical in training and evaluation mode.</p>
 *
 * <p>{@link io.github.pho001.synaptik.nn.layers.LayerNorm} owns mandatory equal-Shape scale and
 * bias parameters over one positive-rank fully static normalized Shape, plus one exact
 * parameter-typed positive epsilon. It delegates to the affine
 * {@link io.github.pho001.synaptik.model.tensor.Tensor#layerNorm(
 * io.github.pho001.synaptik.model.shape.Shape,
 * io.github.pho001.synaptik.model.tensor.Tensor,
 * io.github.pho001.synaptik.model.tensor.Tensor,
 * io.github.pho001.synaptik.model.datatype.ScalarValue)} expression and is also mode-insensitive.
 * It exposes no default epsilon, partial affine state, or second configuration-introspection
 * surface. Compatible parameter replacement affects later forward calls; already constructed
 * Tensor expressions keep their earlier exact inputs.</p>
 *
 * <p>{@link io.github.pho001.synaptik.nn.layers.Embedding} owns one caller-supplied positive
 * rank-two floating table in {@code [vocabularySize, embeddingSize]} orientation. It delegates
 * each forward call to the current table's
 * {@link io.github.pho001.synaptik.model.tensor.Tensor#embedding(
 * io.github.pho001.synaptik.model.tensor.Tensor)} convenience and is mode-insensitive. Model owns
 * accepted index types, result metadata, ordinary Gather failures, and provenance. The layer adds
 * no table initializer, padding-row policy, numerical lookup, or execution behavior. Compatible
 * table replacement affects later calls, while already constructed expressions retain their
 * earlier exact table.</p>
 *
 * <p>{@link io.github.pho001.synaptik.nn.layers.BatchNorm} owns mandatory rank-one
 * {@code scale}, {@code bias}, {@code runningMean}, and {@code runningVariance} state for one
 * explicit logical channel axis. Its immutable
 * {@link io.github.pho001.synaptik.nn.module.ForwardContext} selects exact Model inference or
 * training composition. Evaluation preserves both buffers; successful training installs the
 * pure producer's next mean and then next variance expressions into their stable wrappers. This
 * symbolic NN binding transition performs no eager value mutation, compiler/runtime publication,
 * backend work, or numerical execution.</p>
 *
 * <p>{@link io.github.pho001.synaptik.nn.layers.Dropout} declares no module state and receives an
 * explicit {@link io.github.pho001.synaptik.model.tensor.GraphRngState} on every forward call. Its
 * immutable {@link io.github.pho001.synaptik.nn.module.ForwardContext} selects one Model training
 * dropout occurrence or an evaluation bypass. The bypass returns the exact input and incoming
 * state references without allocating a Tensor or advancing state; training returns the exact
 * public Model output and next-state references through a fresh NN-owned
 * {@link io.github.pho001.synaptik.nn.layers.DropoutForwardResult}. The layer owns no hidden
 * generator, seed, counter, parameter, buffer, execution, or backend behavior.</p>
 */
package io.github.pho001.synaptik.nn.layers;
