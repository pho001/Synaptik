/**
 * Owns unsupported immutable prepared-partition execution and direct cold binding.
 *
 * <p>The partition executable strongly owns its verified generated artifact and exact prepared
 * buffer selections. During cold binding it validates CPU representations, exact carrier pattern,
 * geometry, mutability, liveness, size, alignment, canonical BOOL bytes, and input/output
 * accessed-span overlap, then
 * captures direct array or segment references and primitive range/address state in one invocation.
 * Static movement recipes additionally retain compact immutable output/input layout geometry and
 * family mapping facts. Cold binding packs only the selected invocation range's coordinate and
 * address state; it retains no per-output-element selector or address table.
 * Indexing recipes retain similarly compact geometry plus one direct typed validator. Each run
 * validates the complete logical index domain on the invoking thread before every generated call
 * or worker submission. Invalid input therefore mutates no physical output bytes. Empty index
 * domains validate without a load; zero output still validates any non-empty index domain and,
 * after success, invokes no generated entry and submits no worker work.
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
