/**
 * Owns unsupported immutable prepared-partition execution and direct cold binding.
 *
 * <p>The partition executable strongly owns its verified generated artifact and exact prepared
 * buffer selections. During cold binding it validates CPU representations, exact carrier pattern,
 * geometry, mutability, liveness, size, alignment, and input/output accessed-span overlap, then
 * captures direct array or segment references and primitive range/address state in one invocation.
 * For a selected materialization it validates the original source and exact contiguous workspace,
 * retains the original carrier for the copy, substitutes the workspace segment only in the
 * generated consumer pattern, and completes one canonical-order copy before any consumer chunk.
 * A parallel plan additionally borrows one caller-owned fixed platform-worker group. Cold binding
 * partitions a requested non-empty range into deterministic contiguous non-overlapping chunks;
 * zero chunks submit nothing, one chunk runs inline, and two or more chunks join synchronously.
 * Worker failure, interruption, racing close, nested submission, and group shutdown have explicit
 * deterministic contracts. Runtime invokes the bound object without graph semantics, route
 * selection, reflection, allocation, storage classification, or resource lookup in the generated
 * hot loop.
 *
 * <p>Runtime retains run-level lifecycle ownership; CPU memory representations retain physical
 * allocation and release ownership. Composition owns and closes the worker group; finalizers,
 * prepared executables, and bound invocations only borrow it.
 */
package io.github.pho001.synaptik.backend.cpu.internal.executable;
