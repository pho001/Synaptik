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
 */
package io.github.pho001.synaptik.nn.module;
