/**
 * Owns unsupported CPU buffer/workspace representations and their checked cold-binding contracts.
 *
 * <p>Runtime owns run-level resource orchestration, while these types own CPU-native allocation,
 * segment access, compatibility checks, and physical release. Prepare-time declarations and slot
 * assignments identify resources without allocating them. Cold binding converts assigned runtime
 * representations into typed direct references before execution; generated kernels receive those
 * references and perform no storage classification or lookup in their hot loop.
 * {@code CpuContiguousWorkspace} is one run-owned aligned shared-arena byte workspace. A selected
 * FLOAT64 materialization uses it as a canonical contiguous copy that the invoking thread writes
 * before selected workers read. A floating-product scatter instead partitions it into exact
 * non-overlapping per-range scratch slices that generated scalar calls reset and reuse. These uses
 * are mutually exclusive in the current preparation plan. Runtime still owns run-level workspace
 * lifecycle and cleanup orchestration.
 */
package io.github.pho001.synaptik.backend.cpu.internal.memory;
