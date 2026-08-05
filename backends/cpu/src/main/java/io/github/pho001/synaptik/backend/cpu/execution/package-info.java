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
 * deterministic Java 26 class bytes, and retains each hidden class and exact method handle.
 * Cold CPU finalization may durably reuse exact compatible class bytes through an explicitly
 * rooted trusted-local artifact store. Complete metadata and class shape are revalidated before
 * hidden definition; forced temporary-file publication uses atomic replacement and never a
 * non-atomic fallback. Equal in-flight requests share one process-local attempt, while loaded
 * artifacts are interned only weakly and remain live through their callers. Checksums detect
 * accidental corruption but do not authenticate a root writable by an attacker. This executable-
 * artifact store is distinct from persistent workload tuning evidence and is never consulted by
 * Runtime binding or execution. Typed portable analysis now selects only injected complete
 * candidates and declares exact resources before shared assignment. Backend-declared byte
 * geometry remains opaque to this analysis; shared Prepare validates it against assigned plan
 * geometry. CPU finalization then loads the artifact and constructs an immutable recipe with a
 * family-owned direct binder. The recipe consumes {@code CpuBorrowedBuffer} only through
 * Runtime's {@code BufferRepresentation} boundary and borrows its worker group. The package still
 * implements only synthetic test probes today and performs no Model-operation lowering, truthful
 * capability coverage, public composition, or scheduling. Runtime continues to own logical
 * per-run state, representation lifetime, and cleanup orchestration.</p>
 */
package io.github.pho001.synaptik.backend.cpu.execution;
