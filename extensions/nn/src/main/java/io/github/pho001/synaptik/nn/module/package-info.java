/**
 * Module ownership, named state, deterministic state snapshots, and forward-mode contracts.
 *
 * <p>A {@link io.github.pho001.synaptik.nn.module.Module} owns stable parameter and buffer
 * wrappers plus exclusively attached children. A
 * {@link io.github.pho001.synaptik.nn.module.StateDictionary} captures the exact current Tensor
 * references under deterministic qualified paths: each module contributes its parameters, then
 * buffers, then child subtrees. Strict loading identifies entries by path, validates the complete
 * target and candidate before sequential installation through stable wrappers, and accepts a
 * different candidate-list order. This package owns only shallow in-memory bindings; persistent
 * checkpoint formats, optimizer and session state, graph random state, execution, and backend
 * storage remain outside it.</p>
 *
 * <p>A concrete module may privately reserve future parameter names whose input-dependent Shapes
 * are not known at construction. No incomplete Parameter is exposed: complete parameter
 * discovery and state export fail until all reservations of every visited module are published.
 * Strict loading may validate and publish a complete reserved set from candidate Tensors without
 * invoking a layer initializer. Publication has a narrow release/acquire completion boundary for
 * the direct group; traversal, replacement, loading, mode changes, and arbitrary forward bodies
 * otherwise retain their caller-coordinated threading contract.</p>
 *
 * <p>{@link io.github.pho001.synaptik.nn.module.UnaryTensorModule} combines that ownership with
 * one type-safe {@code Tensor forward(Tensor)} signature. An immutable
 * {@link io.github.pho001.synaptik.nn.module.Sequential} owns such modules under numeric child
 * names and passes exact Tensor-expression references through them from left to right. Empty
 * composition is exact-reference identity. Context-sensitive and explicit-state modules retain
 * their own signatures outside this narrow contract.</p>
 *
 * <p>{@link io.github.pho001.synaptik.nn.module.Model} adds a typed Java input/output boundary
 * above ordinary modules. Its functional factory collects descriptive child names through a
 * short-lived {@link io.github.pho001.synaptik.nn.module.Topology}, validates the complete
 * definition before installing ownership, and then seals the structure. Model topology is the
 * owned module tree and its stable state paths; it is not Tensor graph topology. Tokenizer or
 * batch preparation, checkpoint persistence, backward construction,
 * compilation, training orchestration, and execution remain outside this package contract.
 * Defined models inherit Module's mutable state/mode lifecycle and caller-coordinated threading
 * rules; only their named structure is sealed.</p>
 */
package io.github.pho001.synaptik.nn.module;
