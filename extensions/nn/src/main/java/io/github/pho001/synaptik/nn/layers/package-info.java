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
 * surface. For both current layers, compatible parameter replacement affects later forward calls;
 * already constructed Tensor expressions keep their earlier exact inputs.</p>
 */
package io.github.pho001.synaptik.nn.layers;
