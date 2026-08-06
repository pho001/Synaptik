/**
 * Contains the CPU backend's unsupported implementation contracts.
 *
 * <p>Types below this package are technically public only where Java package boundaries require
 * collaboration between CPU-owned lowering, preparation, code generation, caching, memory, and
 * execution code. They are not supported application APIs. Shared Planning selects CPU ownership;
 * these packages then lower and finalize that complete partition without exposing CPU policy to
 * shared Prepare or Runtime.
 *
 * <p>Analysis, route selection, artifact realization, verification, and checked invocation binding
 * are cold-path work. Runtime receives only a prepared partition executable and a bound invocation;
 * the generated loop performs no graph inspection, route selection, or resource lookup.
 */
package io.github.pho001.synaptik.backend.cpu.internal;
