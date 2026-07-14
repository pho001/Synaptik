/**
 * Defines trace-local identifiers used to correlate diagnostic facts without retaining or
 * exposing producer-domain objects.
 *
 * <p>A producer translates its node, logical-value, or public-Tensor identity into the matching
 * trace-owned identifier. The producer owns allocation, uniqueness, lifetime, and any mapping
 * between the two domains. A trace identifier is meaningful only within the producer-defined
 * trace stream or correlation domain; its numeric value need not equal the producer identifier's
 * numeric value and carries no process-wide or cross-stream guarantee.</p>
 *
 * <p>The types in this package are immutable correlation values only. They do not allocate
 * identifiers, inspect producer state, perform translation, or define serialization.</p>
 */
package io.github.pho001.synaptik.trace.id;
