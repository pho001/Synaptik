# Task 0046: Empty-carry Scan and explicit length-mode source

## Status

Complete for bounded source only — exact reviewed source/caller/guard and sole-formatter cutover
is integrated. Affected3,998passes/60existing CPU skips, actual96source/reject cases and28old CPU
artifact checks passed. Independent final core and wire/evidence/docs reviews APPROVE0.
Typed execution/AD, all-PURE composition and persistence are not expanded.

## Change class

Class C — advanced public scalar-header migration and independent cross-module consumer guards.
Clean implementation, independent source/Javadoc review and MAIN source/compatibility smoke/gates.

## Goal and source evidence

Represent an actual empty carry, and length-aware reverse scans with zero ROW carries, in the
existing Scan. Before this cutover, ScanBody forbade empty carry and ScanAttrs required carryCount>=1
and used rowCarryCount=0 as all-valid. A legal first-order derivative with only discrete original
carries can need no floating reverse carry, or only SHARED floating accumulators. Do not introduce dummy
floating state, fake discrete gradients, replay, unrolling, a map namespace or a second executor.

The [reviewed proposal](../../../../design/decisions/0059-empty-carry-scan-and-explicit-length-mode.md)
uses established fixed-shape/pytree practice, not copied framework code. This task delivers source
representation and truthful independent guard preservation, not typed inference/AD/CPU enablement.

## Exact source cutover

- Existing ScanBody permits empty carry and nextCarry together. Step sequences stay nonempty;
  carry+emission roots stay nonempty. Own nonempty carry shape/dtype equality is unchanged.
- Existing ScanAttrs canonical components become `(timeCount, carryCount, sequenceCount,
  invariantCount, emissionCount, residualCount, reverse, lengthAware, rowCarryCount,
  sequenceRowAxes)`. Preserve the meaningful seven-argument all-valid convenience constructor;
  remove/migrate the old nine-argument sentinel form everywhere, with no shim/alias.
- Checked counts: T>=0, c>=0, s>=1, i/e/r>=0 and c+e>=1. All-valid means explicit false, rowCarryCount=0,
  empty axes; length-aware true allows rowCarryCount in [0,c] and one existing optional role per
  sequence. inputCount uses lengthAware, never the row count; lengths stays last and has no formal.
- Existing Scan.apply overloads accept the matching empty actual list. Infer static/named B from
  validLengths INT64[B] metadata before checking every explicit ROW declaration. No first-carry
  assumption, length read, extent-based row guess or altered role/algebra rule. Existing positive-
  ROW source meanings, axes, aliases, canonical ports, flags and storage-free validation stay.
- ScanKind minimum inputs becomes one for carry-free all-valid with one sequence. CompiledScanBody
  structural admission permits empty nextCarry but requires at least one carry/emission root.
  ScanResult already permits empty groups; preserve it without another result type.

Migrate all header-derived sentinel checks and constructors. Do not mechanically rewrite CPU IR's
separate qualified internal fields or broaden a downstream gate just because a source flag exists.
No added/moved production type/package/dependency/callback/operation kind or runtime protocol.

## Independent consumer and frozen-wire boundaries

Compiler inference/AD, direct Prepare and direct CPU must reject both newly representable forms:
`carryCount==0` OR `lengthAware && rowCarryCount==0`, before first-carry assumptions, derivative IDs,
analyzer calls, resources or generated realization. Keep ordinary homogeneous nonempty accepted
cases and independently authenticate source/header/body cardinalities. Structural CompiledScanBody
validity is not inference/execution permission. Later consumers explicitly qualify these forms.

Cold SourceGraphReconstruction and raw GraphArtifactCodec retain the old intersection independently.
Opcode23/v4 still requires nonempty carry; opcode33/v5 requires nonempty carry and positive ROW
carry. Writer rejects before sizing/storage, reader before body/role allocations where possible.
Mode is opcode-derived; old seven-field tuples, roles, opcodes, v1-v5 selection and accepted bytes
stay unchanged. No boolean wire, v6, invented opcode or silent empty-carry serialization. This
includes raw codecs after relaxed ScanKind/ScanAttrs structural checks; early Compiler failure
alone is not codec evidence. Checkpoint copy/rebind/NN acquisition stay in their accepted bounds.

Compiler0017's obsolete no-floating-carry rejection has been removed from its Draft proposal and
acceptance/dependencies now require this representation. This task does not implement that reverse;
ordinary floating target/cast/disconnected/unsupported-operation and higher-order gates stay.

## Contracts and authority prerequisites

- [Root](../../../../../ARCHITECTURE.md#core-lifecycle): lifecycle, source-only status and exact
  per-owner consumer gates. Before Ready MAIN coordinates explicit header/source permission.
- [Scan owner](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan):
  source/header/bindings, canonical outputs, traversal/empties, accepted per-value typing and row
  semantics. Source counts/mode and independent downstream gates must change together.
- [Compiler](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
  and its bounded cold reconstruction; [Prepare](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare)
  and [CPU](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes): retain negative admission.
- [Checkpoint](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-4)
  retains old v4/v5 domain/bytes. The source flag is not a wire allocation or CPU identity bump.

## Files and current owners

Model: operation.scan.ScanAttrs/ScanKind; tensor.Scan/ScanBody/ScanResult Javadocs; graph.CompiledScanBody;
focused constructor/attachment/body/semantics tests. No new production type or helper protocol.

All old canonical constructor and header-derived mode callers/tests across Compiler, Planning,
Prepare, CPU, Checkpoint, NN and test/tool fixtures must migrate. Preserve current ABI/IR bytes of
qualified CPU paths; only source header access changes. SourceGraphReconstruction, GraphArtifactCodec,
ScanBodyInference/ScanGradientRules preflight, PrepareContext.validateScan and CpuGeneralScanLowering
own independent new-topology rejection. Existing private helpers stay in current owners.
MAIN owns root/scoped/ADR/API guides/plans/changelog; implementer owns affected source/Javadocs/tests.

## Dependencies and integration

- Depends on: Model0045 and practical Compiler0018/Checkpoint0014/NN0033 Complete, CPU0078 integration,
  approved ADR0059 design, coordinated source/count/consumer authority, exact Ready verification.
- Conflicts with: Scan header/source/compiled body, typed consumers/guards, CPU GeneralScanLowering,
  raw codec/cold copies and affected tests/docs. No concurrent all-PURE/typed writer.
- Parallel group: None; source-typing cutover serial after CPU0078. Engine0022 cleanup is unrelated.
- Common base revision: exact accepted dirty source/authority manifest after CPU0078, captured at Ready.
- Integration order: authority/Ready -> isolated complete source/callers/guards -> independent review ->
  MAIN source constructor/negative-consumer and old-wire/CPU compatibility gates -> typed consumer source.
- Integration validation: Model/Javadocs, affected Compiler/Planning/Prepare/CPU/Checkpoint/NN tests,
  direct raw guards and actual source-mode CLI; no native/timing campaign; exact selectors at Ready.
- Shared-document integration owner: MAIN.

## Acceptance

Both directions, T/B=0/1 and named B: emission-only all-valid/length source and length-mode zero-ROW
SHARED accumulators have exact counts, last-length actual, canonical ports, shapes/flags/aliases and
no fake state. Invalid empty semantic roots, counts, roles, changed carry type/shape or contradictory
bindings fail before output identities. Full input/output arithmetic remains checked independently
of empty axes; no first-carry access or guessed batch. No storage read/runtime state is introduced.

Retain old seven/nine-form semantics through migrated seven/ten forms, including homogeneous cold
v4/v5 byte equality and canonical capture/reconstruction. Independently reject new topologies at
raw Compiler/Prepare/CPU/cold/wire seams, including crafted attrs/body and zero-domain cases. Remove
incidental reflection/member-name/diagnostic pins rather than repinning new shapes/counts.

The actual CLI constructs source and observes explicit mode/count/port metadata plus truthful current
consumer rejection; this is source-surface proof, not execution completion. Permanent tests target
real cardinality, alias, metadata or admission defects. No builds/tests/lint/formatters mid-flight;
MAIN runs the final source-matched gate after independent source/Javadoc review.

## Result

MAIN integrated the exact33reviewed worker paths, the independently approved removal of an obsolete
Compiler reflection/copy/count test, and the sole current-ledger formatter in one35-file source
cutover. No nine-argument compatibility alias or dummy carry remains.

Actual Model1210, Compiler418, Planning73, Prepare99, CPU1474, Checkpoint80, NN566 and current
conformance78 passed;60existing CPU cases skipped. Model/Compiler/Checkpoint Javadocs and required
jars passed. The first full gate remains FAILED at the obsolete copied Scan minimum-input tuple;
the owning test was deleted rather than repinned. Its affected full Compiler after-gate passed.

The public CLI passed96F32/F64 emission-only/zero-ROW Shared cases across both directions,
T/B=0/1/named B, explicit mode, aliases, canonical ports and no source storage; all96current
Engine consumers rejected these source-only topologies. The unchanged compatibility driver
captured14qualified families/28actual forward/reverse CPU kernels before and after: all85SHA
records, including canonical IR fingerprints, both identity streams and complete class bytes,
match. Actual current-ledger rendering remains the exact306rows; no new capability is inferred.

model0046-integrated-files.json, first-failed-gate, compiler-final, actual-cpu-compatibility and
current-ledger-authentication retain exact sources, commands/XML and limits. Independent final
evidence/docs reviews returned APPROVE0. Required all-PURE composition and typed reverse remain open.

## Exact Ready launch

MAIN froze HEAD 07a281edd3086efde2853211b71b48288db027e1 plus all 2,647 dirty source-file hashes
in model0046-base.json and created model0046-worktree. The implementation owns the complete
serial source/callers/guards cutover; no concurrent typed/all-PURE writer. The independent Metal
lifetime slice is read-only and source-disjoint. All docs/resources and the shared
LowPrecisionCapabilityLedgerTest source remain MAIN-only, including its constructor/formatter
migration during the separately owned current Metal basis refresh. No child executable checks.
