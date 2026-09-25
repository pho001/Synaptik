# Task 0051: ACCELERATOR FLOAT32 EXP/SIGMOID Recursive-Floor Realization

## Status

Ready

## Change class

Class C — this is the first executable Metal consumer of the completed recursive numerical-profile
contract. It changes profile-qualified capability, typed native schema/lowering, workload and tuning
identity, prepared execution, generated-gradient reachability, and public Engine evidence without
changing Model semantics, public API, Runtime policy, or native ABI exports.

## Goal

Add exactly canonical `FLOAT32` `UnaryElementwiseKind.EXP` and `SIGMOID` under
`NumericalProfile.ACCELERATOR`. `STRICT_IEEE` remains false for both. Select the production
realization only after one fresh real-device recursive-floor oracle certifies every staged candidate
and one bounded, predeclared route adjudication records hot-run time, route dispatch count, and
temporary memory.

The expected production route is direct MPSGraph only if the evidence and decision rule select it.
Disposable fused custom candidates and the stable composed sigmoid candidate exist solely for the
cold decision and must be removed unless selected.

## Authorized capability domain

An occurrence is supported only when all of the following hold:

- the graph profile is exactly `ACCELERATOR`;
- kind is exactly parameterless `EXP` or `SIGMOID` with `NoOperationAttrs.INSTANCE`;
- there is exactly one input and one output;
- both descriptors are exactly `FLOAT32`, fully static positive rank `1..16`, and every extent is
  positive with checked element and byte geometry;
- input and output Shapes and `requiresGrad` flags are equal; and
- input and output layouts are canonical dense-contiguous, zero-offset non-views.

Both operations consume and produce canonical state. They may compose inside the existing complete
maximal accelerator partition with current canonical binary, reduction, MATMUL, exact/common, and
layout operations only where the existing state rules already permit it. No affine-view operand is
newly accepted. Every other unary kind and every strict `EXP`/`SIGMOID` occurrence remains false.

## Model result-set gate

The candidate gate implements the completed Model accelerator result set, without assigning a
whole-output elementary envelope to composite sigmoid. For each represented EXP input, it evaluates
the exact represented binary32 value with high precision, rounds once to the correctly rounded
binary32 reference `r`, and requires:

- NaN input produces NaN; payload, sign, and quieting remain free;
- `EXP(-infinity)=+0`, `EXP(+infinity)=+infinity`, and either zero maps to `+1`;
- a subnormal input may remain represented or be read as same-signed zero (DAZ);
- an ordinary finite non-subnormal `r` admits only finite results whose inclusive
  ordered-binary32 distance from `r` is at most five;
- a nonzero subnormal `r` admits exactly `r`, positive zero, or negative zero (FTZ); and
- zero, infinity, overflow, underflow, sign, and class checks remain independent of distance.

For each SIGMOID input, the gate instead enumerates the finite union of Model's recursive stable
formula results. It starts from the represented input plus its same-signed-zero DAZ alternative
when subnormal, applies the exact sign guard, and evaluates the selected negative
`EXP(x) -> ADD(1) -> DIV(exp,denominator)` or nonnegative
`NEG -> EXP -> ADD(1) -> DIV(1,denominator)` branch. At EXP it enumerates exactly the elementary
set above. At every explicit arithmetic site it enumerates same-signed-zero operand DAZ, one
binary32 round-to-nearest-even operation, and exact-subnormal-or-either-signed-zero result FTZ.
A direct or custom sigmoid candidate passes only when its represented output belongs to this union,
with independent NaN/infinity/zero-sign/range checks. It is never checked against a final
whole-sigmoid distance ceiling.

Ordered distance uses the Model key: unsigned `~bits` for a sign-set word and unsigned
`bits ^ 0x80000000` otherwise, followed by mathematical absolute key difference.

## Candidate staging and exactly one fresh oracle

Before any capability can become true, stage these implementations with production capability
still false:

1. direct MPSGraph `exponentWithTensor` for `EXP`;
2. one disposable custom Metal `float` EXP kernel;
3. direct MPSGraph sigmoid for `SIGMOID`;
4. one disposable custom stable-branch sigmoid kernel; and
5. one MPSGraph stable sigmoid composition using explicit negation/exponential/addition/division
   sites and graph constants.

The disposable oracle is outside the repository and exports no ABI. It must compile all five
candidates first, then perform exactly one real-device process invocation. That one invocation uses
one default-device context, direct supplied targets, optimization/fast-math state recorded in the
raw report, input-preservation and target-canary checks, and the same frozen corpus for every
candidate. There is no retry, second context, optimization-level matrix, route fallback, or repeated
oracle after a failure.

The frozen corpus is the ordered de-duplicated union of:

```text
00000000 80000000 00000001 80000001
007fffff 807fffff 00800000 80800000 00800001 80800001
3f000000 bf000000 3f800000 bf800000 40000000 c0000000
7f7fffff ff7fffff 7f800000 ff800000
7fc12345 ffc54321 7fa12345 ffa54321
```

and each listed base word plus its immediately adjacent raw words when finite:

```text
c2d00000 c2cff1b5 c2cfcccd c2ce0000 c2b40000 c2b00000 c2af0000
c1a00000 c0e80000 bf900000 bf800000 bf000000
3f000000 3f800000 40e80000 41a00000 42a00000 42b00000 42b17218
42b20000 42d00000
```

Those words cover signed zeros, signed minimum/maximum subnormals, the normal boundary, ordinary
small/large values, retained historical `-88`, `-7.25`, and `-1.125` observations, EXP underflow and
overflow neighborhoods, maximum finite values, both infinities, and multiple quiet/signaling NaN
signs/payloads. The reference evaluates the exact represented binary32 value with at least 160
significant decimal digits, performs stable-branch sigmoid, and resolves final binary32
round-to-nearest-even by comparing adjacent exact binary32 values rather than by double rounding.

The raw oracle report must name every candidate, corpus SHA-256, output SHA-256, maximum ordered
distance and its input, FTZ count, class/sign/range verdicts, canary/input/Shape verdicts, and final
`PASS`/`FAIL`. Only `PASS` candidates proceed to timing. The task Result records the unedited raw
verdict text and its SHA-256; probe source/binary/output are then removed.

## One bounded route adjudication

Run exactly one post-oracle adjudication workload on the same real device:

```text
profile = ACCELERATOR
input = canonical caller FLOAT32 [1_048_576]
outputs = EXP(input), SIGMOID(input)
optimization = production MPSGraphCompilationDescriptor default
reducedPrecisionFastMath = None with readback
corpus = frozen edge prefix, then Float.floatToRawIntBits((((i * 17) % 257) - 128) / 16.0f)
```

Every surviving route is prepared before timing and exact-checked against the recursive-floor gate.
Use four alternating-order warmup rounds and eight alternating-order retained rounds in one process;
each retained batch runs to a 25 ms floor with a 1,000,000-execution ceiling. Discard and retry
nothing. The hot boundary is synchronous route execution plus result closure only; preparation,
upload, download, oracle checks, and reporting are excluded.

Record every raw batch duration, iteration count, normalized duration, and the descriptive median.
After uncaptured timing, use `MTLCaptureManager` queue capture for exactly one hot logical execution
of each passing route in the same workload. Read each resulting GPU trace with supported installed
tooling and record its SHA-256, actual compute-dispatch records, and peak transient/temporary Metal
resource bytes above the route's steady prepared input/output baseline. The custom route's explicit
resource ledger is a cross-check, not a substitute for the trace. Framework-internal work must not
be relabeled as a single backend call or reported only as opaque. If a trace cannot be emitted or
the actual dispatch and peak temporary facts cannot be read for every passing route, adjudication
is incomplete: Task 0051 becomes `Blocked` with capability still false.

The predeclared selection rule is:

1. remove every numerical-gate failure;
2. direct MPSGraph is the maintenance-safe baseline for each operation;
3. another route replaces it only if its retained median is at least 10% lower, its actual compute
   dispatch count is no greater, and its peak temporary bytes are no greater;
4. if more than one alternative qualifies, choose the lowest median, then fewer actual compute
   dispatches, then fewer peak temporary bytes, then direct/composed MPSGraph before custom code;
   and
5. exact ties select direct MPSGraph; missing dispatch or temporary-memory facts block the task.

The decision is fixed in production; it is not a runtime threshold, tuning matrix, retry, fallback,
or input-dependent choice. Remove every unselected disposable kernel/composition/probe.

## Typed schema and identity cutover

- Keep native ABI version `4`, the fixed 160-byte record, statuses, signatures, complete-plan
  wrapper schema `1`, and exactly thirteen exports unchanged.
- Advance node schema `11 -> 12`; retain wires `1..19`; append no-attribute canonical unary
  `EXP=20` and `SIGMOID=21`; require `UINT32_MAX` second-input/axis sentinels, zero auxiliary,
  zero attribute count/payload, equal input/output Shape, and FLOAT32 canonical state.
- Java and native preflight independently reject strict profile use, stale schema 11, unknown wire
  `22+`, wrong attrs/sentinel/padding/type/state/topology/Shape/rank, and malformed or overflowed
  geometry before resource creation.
- Advance workload signature, exact default policy, candidate schema, compatibility schema,
  route-policy, and tuning codec together `12 -> 13`. Hash profile, schema, ordered unary topology,
  descriptors/states, feeds/targets, splats, ABI, and route facts. Version-12 bytes safely miss or
  reject; no alias, dual decoder, migration shim, or fallback remains.

If adjudication selects direct MPSGraph for both operations, the existing whole-partition
`MPSGRAPH` candidate is the only production route for them and the existing NEG-only custom
candidate domain remains unchanged. Do not retain candidate enum values for discarded probes.

## Production lifecycle and gradients

Preserve current whole-partition analysis, stable node/value/feed/target order, complete candidate
batches, authenticated optional decisions, declarations-before-assignment, transactional
finalization, context child leases, immutable splats, direct assigned targets, run-owned outputs and
address workspace, synchronous invocation, reuse, concurrency isolation, rollback, close
rejection, publication, transfer, tracing, and session-scoped tuning behavior. Add no hot-path
numerical check, retry, fallback, secondary route, per-node invocation, host staging, or hidden
materialization.

Compiler generation is unchanged. Prove a seeded `EXP` gradient is all Metal under
`ACCELERATOR`: forward `EXP` and generated `seed * output` use only the new EXP wire and existing
canonical MUL. Inspect and execute the generated topology, not a handwritten substitute.

Attempt the generated seeded `SIGMOID` gradient
`seed * output * (one - output)` without widening binary capability. If the current compiler emits
an affine/view or other unsupported boundary, keep explicit mixed ownership and document exact
partition owners/transfers; do not broaden affine-to-binary or scalar capability. If the existing
canonical/splat domain already closes it, prove the all-Metal topology instead.

## Non-goals

No strict EXP/SIGMOID; other unary kind; scalar operation; new dtype; rank zero; zero/dynamic Shape;
view ingress; binary/affine capability widening; Model/Compiler gradient change; Runtime/Trace
profile state; async work; device selection; pooling; package/discovery change; ABI/export/status;
persistent tuning format; runtime numerical check; retry; fallback; benchmark matrix; generalized
custom-unary framework; retained unselected kernel; or final repository-wide build.

## Dependencies and integration

- Depends on: completed and independently approved Model 0030/0031 at remediation
  `97cb9d116bd85ee6a0dfbf2e9b70d32604633c85`; Metal 0050 packaged baseline; retained Metal 0016
  device evidence; Compiler unary gradients; Config 0006; Engine 0018.
- Common base revision: `621555a500355d4f0a4b32a54bf7d828331e12f0`.
- Conflicts with: every concurrent Metal capability/schema/native-preflight/candidate/codec/tuning/
  lifecycle/public Engine edit and shared numerical-profile documentation.
- Parallel group: None.
- Integration order: this planning commit; capability-false candidate staging; exactly one oracle;
  exactly one adjudication; delete losers; production schema/lifecycle/capability; focused proof;
  documentation; implementation commit; independent Class C review.
- Shared-document integration owner: task implementer.

## Files and symbols

Expected production owners:

- `MetalCapabilityProvider`, `MetalMpsGraphProgram`, `MetalNativeApi`,
  `MetalNegPartitionPreparer`, and affected package Javadocs;
- `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, `MetalNegTuningCodec`, and current
  decision/complete-plan compatibility tests;
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m` and its README;
- focused Metal capability/schema/native/prepared/candidate/codec tests, backend conformance, and
  public Engine integration;
- Metal/backend/API/glossary explanatory docs, this task, Metal master, and roadmap.

No probe, oracle output, route report, custom kernel, benchmark artifact, or cache remains unless the
selection rule chooses it as production.

## Acceptance criteria

1. Exactly one fresh real-device oracle invocation certifies EXP candidates against the elementary
   result set and SIGMOID candidates by recursive stable-formula membership while capability is
   false; raw verdict and hashes are recorded and artifacts removed.
2. Exactly one bounded adjudication workload times only passing routes and records raw hot-run
   evidence plus captured actual compute dispatches and peak temporary bytes; production matches the
   predeclared rule or the task blocks with capability false.
3. Capability admits exactly accelerator canonical FLOAT32 EXP/SIGMOID and rejects strict plus all
   malformed/excluded rows before native creation.
4. Schema 12 wires 20/21 and every version-13 identity fail closed against stale/malformed data;
   ABI 4, thirteen exports, 160-byte records, and complete-plan wrapper 1 remain exact.
5. Production direct/custom/composed lowering is only the selected route, preserves the complete
   existing lifecycle, and performs no runtime checking, retry, fallback, or matrix selection.
6. Focused tests prove native numerical results, composed partitions, stale identity rejection,
   seeded EXP all-Metal execution, honest SIGMOID gradient ownership, and a CPU-free public Engine
   forward scenario through the real dylib with no skip.
7. Javadocs/docs/Markdown/diff pass; no final full repository build runs. Status becomes
   `Review needed` for independent Class C review.

## Validation

Build and inspect the actual native library once after production stabilization:

```bash
./native/metal-macos-arm64/build.sh
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
otool -L native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
```

Run focused suites only, with the freshly built dylib supplied explicitly:

```bash
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" \
  ./gradlew :backends:metal:test --tests '*Metal*'
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" \
  ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
./gradlew :backends:metal:javadoc :testing:architecture-tests:test
```

Inspect the actual test report for zero failures/errors/skips in the mandatory native scenario.
Validate changed Markdown links, anchors, fences, final newlines, exact evidence text, path scope,
artifact removal, and `git diff --check`. Do not run a full repository build.

## Documentation and review impact

Synchronize current Metal capability, ABI/schema/identity, lifecycle, route decision, gradient,
public Engine, native guide, package Javadocs, master-plan, and roadmap wording. Preserve historical
blocked task evidence. Independent Class C review must inspect raw oracle/adjudication evidence,
production selection, strict false gates, schema/stale rejection, lifecycle, gradients, real no-skip
Engine proof, removed artifacts, and absence of runtime checks/fallback.

## Result

Pending implementation.
