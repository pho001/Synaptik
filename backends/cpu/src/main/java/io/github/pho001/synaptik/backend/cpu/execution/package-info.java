/**
 * Backend-private CPU memory, cold-binding, generated-kernel, direct invocation, and bounded
 * worker foundation.
 *
 * <p>These declarations expose no public storage or executor API and implement no Model operation
 * or production route. Cold binding resolves each exact selected segment independently to either
 * an observable primitive-array carrier plus carrier-relative offset or an exact-segment argument.
 * The latter does not imply native provenance: it also preserves heap segments whose matching
 * carrier is not observable. The generated-kernel foundation consumes an already-complete typed
 * specialization and a family emitter, supports the exact scalar/vector and single/parallel mode
 * matrix over primitive arrays and {@link java.lang.foreign.MemorySegment} arguments, verifies
 * deterministic Java 26 class bytes, and retains each fresh hidden class and exact method handle.
 * It implements only synthetic test probes today and performs no Model-operation lowering,
 * preparation, route selection, caching, or scheduling. Runtime continues to own logical per-run
 * state and cleanup orchestration.</p>
 */
package io.github.pho001.synaptik.backend.cpu.execution;
