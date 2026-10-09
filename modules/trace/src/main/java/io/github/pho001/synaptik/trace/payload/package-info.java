/**
 * Defines closed immutable diagnostic payloads and value vocabularies for backend preparation and
 * invocation outcomes.
 *
 * <p>Types in this package express producer facts only in trace-owned terms. They retain no
 * producer object, mutable runtime state, arbitrary string field, generic attribute map, native
 * handle, tensor value, exception, emitter, observer, or serialization behavior. Producers remain
 * responsible for translating their local identities, routes, cache facts, and native
 * statuses into these data-transfer objects.</p>
 */
package io.github.pho001.synaptik.trace.payload;
