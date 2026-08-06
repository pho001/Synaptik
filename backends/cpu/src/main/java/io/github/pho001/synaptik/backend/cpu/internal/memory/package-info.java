/**
 * Owns unsupported CPU buffer/workspace representations and their checked cold-binding contracts.
 *
 * <p>Runtime owns run-level resource orchestration, while these types own CPU-native allocation,
 * segment access, compatibility checks, and physical release. Prepare-time declarations and slot
 * assignments identify resources without allocating them. Cold binding converts assigned runtime
 * representations into typed direct references before execution; generated kernels receive those
 * references and perform no storage classification or lookup in their hot loop.
 * A selected materialization uses one run-owned aligned shared-arena FLOAT64 workspace. The
 * invoking thread writes it exactly once before selected workers read it; Runtime still owns the
 * run-level workspace lifecycle and cleanup orchestration.
 */
package io.github.pho001.synaptik.backend.cpu.internal.memory;
