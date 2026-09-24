/**
 * Supplies explicit capability, configuration, and lifecycle integration for the Metal backend.
 *
 * <p>{@link io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider} reports two
 * profile-qualified domains. {@code STRICT_IEEE} admits parameterless {@code NEG} and {@code ABS},
 * {@code RESHAPE}, {@code EXPAND}, {@code PERMUTE}, {@code EXPAND_DIMS}, {@code SQUEEZE}, and the
 * explicit {@code CONTIGUOUS} canonicalization barrier. {@code ACCELERATOR} admits exact
 * {@code ABS}, tensor {@code ADD}, {@code SUB}, {@code MUL}, and {@code DIV}, and canonical
 * {@code FLOAT32} {@code SUM}, {@code MEAN}, and binding-resolved {@code SUM_TO_SHAPE}.
 * Reductions support full, normalized single-axis, ordered normalized multi-axis including empty,
 * and exact keep-dimensions forms. Feeds and ordinary outputs are fully static positive-rank
 * {@code 1..16}; a locally produced reduction target may be rank zero and materializes as exactly
 * four canonical bytes without widening caller ingress or CPU/Metal transfer. Accelerator binary
 * inputs and outputs are canonical dense non-views and use exact right-aligned broadcasting.
 * Every {@code ABS} operand/output and strict graph feed or {@code NEG} operand is canonical;
 * strict affine inputs may also be exact resolved zero-offset views produced earlier in the same
 * maximal partition. Strict affine outputs retain their exact Model view geometry, while
 * {@code CONTIGUOUS} produces canonical geometry. Metal lowers one complete
 * profile-homogeneous partition as a typed whole-partition program during preparation.</p>
 *
 * <p>{@link io.github.pho001.synaptik.backend.metal.MetalBackendConfiguration} names one explicit
 * native bridge. {@link io.github.pho001.synaptik.backend.metal.MetalBackendIntegration} opens and
 * owns one default-device native context and supplies availability, partition preparation,
 * backend-local schedule assembly, a shared physical schedule contribution, checked native
 * upload/download binders, caller-host ingress, and bounded canonical host materialization to
 * Engine's explicit builder. Exact prepared affine-view targets may materialize from their
 * authenticated backend-private dense represented-order buffers without rewriting logical view
 * descriptors or widening the canonical non-view CPU/Metal transfer predicate. Engine uses those
 * contributions and binders for the bounded bidirectional CPU/Metal mixed-owner schedule. Native
 * internals remain package-private. There is no library discovery, CPU fallback, or backend-global
 * integration.</p>
 *
 * <p>The selected numerical profile participates in partition-plan, route, tuning,
 * decision-codec, and workload identity. Java rejects profile/schema mismatches before native
 * entry. ABI version four remains stable; node schema version seven appends {@code SUM=13} and
 * {@code MEAN=14} with typed reduction attributes.</p>
 */
package io.github.pho001.synaptik.backend.metal;
