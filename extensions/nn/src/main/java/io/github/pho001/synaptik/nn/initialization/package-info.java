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
 * <p>Random initialization receives a caller-owned {@link java.util.random.RandomGenerator} for
 * each call. The caller selects, configures, seeds, owns, advances, and coordinates access to that
 * exact source; this package neither chooses a default nor retains, substitutes, synchronizes,
 * resets, splits, serializes, or closes it. This eager boundary is distinct from deferred graph
 * random-number-generator state: it creates no random Tensor expression and neither accepts nor
 * creates {@link io.github.pho001.synaptik.model.tensor.GraphRngState}.</p>
 *
 * <p>{@link io.github.pho001.synaptik.nn.initialization.LinearWeightInitialization} is the closed
 * selection used by the input-width-inferring Linear constructor. The layer retains an explicit
 * deterministic random-generator factory and seed, then dispatches to one existing eager
 * initializer during its first compatible forward. The enum introduces no general initializer
 * object model, registry, callback, or default source.</p>
 */
package io.github.pho001.synaptik.nn.initialization;
