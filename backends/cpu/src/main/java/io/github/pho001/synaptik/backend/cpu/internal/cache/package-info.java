/**
 * Owns unsupported structural identity, compatibility interning, and optional cold persistence.
 *
 * <p>Keys combine canonical lowering with the selected numerical mode, execution strategy, and
 * current generator schema while excluding instance extents and runtime resources. With no trusted
 * root, realization remains entirely in memory. With a root, persisted class bytes are treated as
 * an optional cold-path optimization and must be verified before definition; missing or corrupt
 * entries fall back to deterministic emission.
 *
 * <p>This package collaborates with canonical IR and Class-File generation. It owns neither graph
 * lowering nor Runtime lookup, JIT machine code, profiling state, or run-resource lifetime.
 */
package io.github.pho001.synaptik.backend.cpu.internal.cache;
