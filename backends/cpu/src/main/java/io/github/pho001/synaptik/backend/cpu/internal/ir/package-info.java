/**
 * Defines the unsupported, route-independent canonical CPU kernel intermediate representation.
 *
 * <p>The representation records typed boundary and virtual values, ordered exact computation,
 * structural access forms, primitive half-open loop bounds, and ordered stores. It deliberately
 * excludes routes, slots, graph identities, concrete extents, segment instances, generator
 * versions, and invocation bindings so compatible shapes can share one structural artifact.
 *
 * <p>Lowering creates this cold immutable model. Portable code generation and the scalar reference
 * realization may consume its established semantics, but Runtime never interprets it.
 */
package io.github.pho001.synaptik.backend.cpu.internal.ir;
