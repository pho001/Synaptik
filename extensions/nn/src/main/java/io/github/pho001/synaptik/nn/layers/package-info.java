/**
 * Provides stateful neural-network layers that own named parameters and compose Model Tensor
 * expressions.
 *
 * <p>A layer in this package is a {@link io.github.pho001.synaptik.nn.module.Module}: it owns the
 * parameter wrappers used by later forward construction, while generic Tensor operation meaning
 * remains in {@code modules/model}. Calling a forward method builds storage-free expression
 * metadata. It does not evaluate values, compile a graph, select a backend, or execute work.</p>
 *
 * <p>The current {@link io.github.pho001.synaptik.nn.layers.Linear} layer uses the conventional
 * {@code [outFeatures, inFeatures]} weight orientation and delegates each forward call to
 * {@link io.github.pho001.synaptik.model.tensor.Tensor#linear(
 * io.github.pho001.synaptik.model.tensor.Tensor)} or its biased overload. Its behavior is
 * identical in training and evaluation mode. Parameter replacement affects later forward calls;
 * already constructed Tensor expressions keep their earlier exact inputs.</p>
 */
package io.github.pho001.synaptik.nn.layers;
