/**
 * Owns unsupported structural identity, compatibility interning, and optional cold persistence.
 *
 * <p>Keys combine canonical lowering with the selected numerical mode, generated scalar/vector
 * compute form, exact preferred FLOAT64 species when applicable, ordered carrier pattern, and
 * current generator schema. Instance extents, chunk configuration, worker identity, and Runtime
 * resources remain outside artifact identity. With no trusted root, realization remains entirely
 * in memory. With a root, persisted class bytes are an optional cold-path optimization and must be
 * verified before definition; missing or corrupt entries fall back to deterministic emission.
 *
 * <p>This package collaborates with canonical IR and Class-File generation. It owns neither graph
 * lowering nor Runtime lookup, JIT machine code, profiling state, or run-resource lifetime.
 */
package io.github.pho001.synaptik.backend.cpu.internal.cache;
