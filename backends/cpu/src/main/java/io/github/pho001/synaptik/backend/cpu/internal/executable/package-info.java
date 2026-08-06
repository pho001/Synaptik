/**
 * Owns unsupported immutable prepared-partition execution and direct cold binding.
 *
 * <p>The partition executable strongly owns its verified generated artifact and exact prepared
 * buffer selections. During cold binding it validates CPU representations, exact carrier pattern,
 * geometry, mutability, liveness, size, alignment, and input/output accessed-span overlap, then
 * captures direct array or segment references and primitive range/address state in one invocation.
 * Runtime invokes that object without graph semantics, route selection, reflection, allocation,
 * storage classification, or resource lookup in the generated hot loop.
 *
 * <p>Runtime retains run-level lifecycle ownership; CPU memory representations retain physical
 * allocation and release ownership.
 */
package io.github.pho001.synaptik.backend.cpu.internal.executable;
