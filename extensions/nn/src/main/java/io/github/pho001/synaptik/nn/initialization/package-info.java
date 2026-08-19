/**
 * Provides stateless eager creation policies for neural-network parameter tensors.
 *
 * <p>An initializer in this package consumes any requested random samples immediately and returns
 * a fresh, unlabeled, provenance-free Model Tensor leaf backed by dense host storage, with
 * {@code requiresGrad == true}. It does not declare or bind a
 * {@link io.github.pho001.synaptik.nn.module.Parameter}; a module remains responsible for giving
 * the returned Tensor a state name and owning its current binding. {@code Parameter} retains a
 * caller-supplied Tensor and has no initialization policy.</p>
 *
 * <p>The low-level random initializer and dispatcher overloads receive a caller-owned
 * {@link java.util.random.RandomGenerator} for each call. Their caller selects, configures, seeds,
 * owns, advances, and coordinates access to that exact source; this package neither chooses a
 * default nor retains, substitutes, synchronizes, resets, splits, serializes, or closes it. A
 * high-level recurrent layer may separately create the documented standard
 * {@code L64X128MixRandom} source from its explicit seed and pass that source transiently to this
 * package. This eager boundary is distinct from deferred graph random-number-generator state: it
 * creates no random Tensor expression and neither accepts nor creates
 * {@link io.github.pho001.synaptik.model.tensor.GraphRngState}.</p>
 *
 * <p>{@link io.github.pho001.synaptik.nn.initialization.ParameterInitialization} is the closed
 * immutable selection used by automatic dense and recurrent layers. A layer retains only that
 * policy and its own separate seed or caller-supplied generator factory, then dispatches to one
 * existing eager initializer when its complete parameter Shape becomes known. The policy owns no
 * Shape, fan value, data type, Tensor, parameter, random source, seed, layer order, or bias rule.
 * Its zero/one route creates and consumes no random generator. The policy introduces no general
 * initializer object model, registry, callback, or default source.</p>
 */
package io.github.pho001.synaptik.nn.initialization;
