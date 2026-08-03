/**
 * Backend-private CPU memory, cold-binding, invocation, and bounded-worker foundation.
 *
 * <p>These declarations expose no public storage or executor API and implement no operation or
 * route. Cold binding resolves each exact selected segment independently to either an observable
 * primitive-array carrier plus carrier-relative offset or an exact-segment argument. The latter
 * does not imply native provenance: it also preserves heap segments whose matching carrier is not
 * observable. Runtime continues to own logical per-run state and cleanup orchestration.</p>
 */
package io.github.pho001.synaptik.backend.cpu.execution;
