/**
 * Implements Metal's package-private native, preparation, storage, and execution boundary.
 *
 * <p>One context owns the default Metal device, command queue, and native-library lifetime.
 * Backend analysis validates and lowers one complete maximal partition of supported unary
 * {@code NEG} occurrences, generates a complete typed route-candidate batch, and selects one of
 * two private routes before declaring exact resources and shared slot assignment. An absent
 * decision keeps the safe singleton heuristic. A present backend-local decision is untrusted:
 * fresh analysis authenticates its schema, canonical workload, exact context session, and
 * candidate membership before applying it. An exact singleton in the custom 32-bit index domain
 * can select either route; custom has only feed and target buffers, while MPSGraph also has the
 * address workspace.
 * Backend finalization compiles the selected custom pipeline or MPSGraph executable and returns
 * its typed persistent owner. The choice is a deterministic implementation-domain boundary, not
 * tuning, capability narrowing, fallback, retry, or repartitioning.</p>
 *
 * <p>Each run borrows caller input buffers, creates fresh initialized constant buffers and output
 * buffers. MPSGraph runs also own native-address workspace storage through their
 * {@code RunState}; custom runs do not. Cold binding creates a route-specific invocation with
 * direct references. The bound hot invocation makes one synchronous native downcall into the
 * selected reusable resource, which writes the assigned output destination directly without an
 * explicit host-staging or intermediate-copy step.</p>
 *
 * <p>The bounded session codec is not a persistent workload cache and performs no file I/O,
 * measurement, or tools integration. The Metal application binary interface (ABI) version alone
 * is not a stable device fingerprint, so compatibility includes one fresh private nonce per live
 * context and deliberately rejects cross-session reuse.</p>
 *
 * <p>All types remain package-private. Shared Runtime sees only its nominal resource,
 * representation, executable, and schedule contracts; no native handle, Objective-C value,
 * MPSGraph type, operation, graph node, backend lookup, or route choice crosses the hot-path
 * boundary.</p>
 */
package io.github.pho001.synaptik.backend.metal.internal;
