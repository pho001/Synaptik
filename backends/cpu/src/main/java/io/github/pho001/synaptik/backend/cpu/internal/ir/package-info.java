/**
 * Defines the unsupported, route-independent canonical CPU kernel intermediate representation.
 *
 * <p>The representation records typed boundary and virtual values, one twenty-two-opcode
 * family-oriented pointwise vocabulary, exact typed scalar-immediate bits, the selected
 * scalar-power realization where applicable, ordered computation,
 * normalized access forms, primitive half-open loop bounds, and ordered stores. Each access form
 * selects one of five scalar state-machine regimes while its binding retains cold extents, base
 * offset, effective strides, start coordinates/address, and exact half-open accessed span. It deliberately
 * excludes routes, slots, graph identities, concrete extents, segment instances, generator
 * versions, and invocation bindings so structurally compatible shapes can share one artifact.
 * A selected contiguous copy produces an adjusted canonical consumer access structure for the
 * generated artifact, while its concrete extent, source binding, workspace, and cost evidence
 * remain instance facts outside canonical identity.
 *
 * <p>Lowering creates this cold immutable model. Portable code generation and the scalar reference
 * realization may consume its established semantics, but Runtime never receives or interprets it.
 */
package io.github.pho001.synaptik.backend.cpu.internal.ir;
