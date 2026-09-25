# Task 0051: ACCELERATOR FLOAT32 EXP/SIGMOID Recursive-Floor Realization

## Status

Blocked

The completed 83-word invocation is regression/rejection evidence only, not complete-domain proof.
Gate 1B now leaves every previously sampled route ineligible: opaque direct EXP and any composition
inheriting it are `DOMAIN-BLOCKED`, while direct/custom SIGMOID and the original custom EXP retain
their numerical failures. This remediation is planning/evidence only. Capability, schema, identity,
routes, and production remain unchanged, and the consumed one-shot oracle must not be rerun.

## Change class

Class C — an eventual implementation would be the first executable Metal consumer of the completed
recursive numerical-profile contract. This planning/evidence remediation changes no
profile-qualified capability, typed native schema/lowering, workload or tuning identity, prepared
execution, generated-gradient reachability, public Engine behavior, Model semantics, public API,
Runtime policy, or native ABI export.

## Goal

Define the gates for eventually adding exactly canonical `FLOAT32` `UnaryElementwiseKind.EXP` and
`SIGMOID` under `NumericalProfile.ACCELERATOR`; `STRICT_IEEE` remains false for both.

No current candidate is eligible for timing or production. Direct MPSGraph EXP passed the bounded
sample but lacks an authoritative all-binary32/rank-`1..16` contract and is `DOMAIN-BLOCKED`; the
sampled MPSGraph stable SIGMOID composition inherits that blocker. Direct/custom SIGMOID and the
original custom EXP also retain their recorded numerical failures. A future successor may stage the
auditable integer custom EXP and stable SIGMOID designs below only after their complete-domain
proof artifacts pass independent review, then run a newly authorized one-shot oracle under that
successor's own contract.

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

## Gate 1B: complete-domain proof prerequisite

Gate 1B is independent of the bounded numerical oracle. A numerical pass can reject an incorrect
proof, but cannot prove that a candidate's outputs are a subset of the Model set for all
`2^32` input words, every positive canonical rank `1..16` Shape, and every checked extent. An opaque
route requires an authoritative, deployment-version-applicable contract covering raw-word special
classes, DAZ, correctly rounded reference/distance, underflow/overflow, shape-dependent algorithms,
and exact output publication. Selector names, SDK declarations, samples, and final-output agreement
are insufficient.

| Candidate from the consumed invocation | Gate 1A/API | Gate 1B | Bounded numerical verdict | Consequence |
|---|---|---|---|---|
| direct MPSGraph EXP | `PASS` | `DOMAIN-BLOCKED` — no authoritative all-binary32/rank-`1..16` selector contract | `PASS` on 83 words | regression evidence only; not eligible for timing or production |
| original custom EXP | `PASS` | `DOMAIN-BLOCKED` — no reviewed complete source/constant/error proof | `FAIL` | rejected; remains removed |
| direct MPSGraph SIGMOID | `PASS` | `DOMAIN-BLOCKED` — opaque direct formula/algorithm | `FAIL` | rejected; remains removed |
| original custom stable SIGMOID | `PASS` | `DOMAIN-BLOCKED` — no complete primitive/formula proof | `FAIL` | rejected; remains removed |
| MPSGraph stable SIGMOID composition | `PASS` | `DOMAIN-BLOCKED` — inherits direct MPSGraph EXP's blocker | `PASS` on 83 words | regression evidence only; not eligible for timing or production |

No candidate from the consumed invocation is `DOMAIN-PASS`. Consequently the prior timing stop is
not only an instrumentation stop: even with trace tooling installed, Gate 3 cannot begin until a
new candidate passes Gate 1B and a separately authorized fresh numerical gate.

### Successor-only auditable custom candidate design

`CUSTOM_EXP_CR_U32_V1` is a fixed typed pointwise kernel over raw `uint32` words, not a generalized
transcendental framework. It calls no MPSGraph selector, Metal transcendental, native `float`
operation, or opaque library routine:

1. Decode sign/exponent/fraction bits and handle every NaN, both infinities, and both zeros by
   explicit raw-word branches. Emit one fixed quiet NaN for every NaN, `+0` for negative infinity,
   `+infinity` for positive infinity, and `+1` for either zero.
2. Choose the represented finite input, never the optional DAZ alternative. This is one allowed
   Model input path, including represented subnormal inputs.
3. Convert that exact binary rational into a frozen signed multi-limb fixed-point representation.
   Use only explicitly implemented `uint32` add/subtract/multiply/shift/compare operations with
   checked shift counts, proved carry bounds, and no signed overflow or implementation-defined
   behavior.
4. Select `k` with certified outward rational bounds for `ln(2)/256`, then enclose
   `r = x - k * ln(2)/256` without treating the irrational constant as exactly represented. Frozen
   outward-rounded bounds, a 256-entry `2^(j/256)` interval table, and a fixed polynomial for
   `exp(r)` produce a directed lower/upper result interval. The proof generator selects and freezes
   limb width, polynomial degree, coefficients, evaluation order, and every truncation direction;
   the kernel may not substitute compiler floating arithmetic.
5. If the interval identifies one correctly rounded binary32 result, pack that raw word directly.
   Every unresolved rounding-boundary input uses a frozen sorted exception table keyed by the exact
   input word. Table generation must prove exhaustiveness and uniqueness; an unbounded or
   unexplained exception set rejects the candidate.
6. Handle overflow and gradual underflow through proved exact raw thresholds and integer packing.
   Emit the correctly rounded normal/subnormal/zero/infinity word directly, so the chosen route uses
   no output FTZ. Correct rounding is ordered distance `0`, a strict subset of the Model's inclusive
   five-step ordinary-result allowance; exact subnormal `r` is one allowed subnormal result.

`CUSTOM_SIGMOID_STABLE_U32_V1` composes only reviewed raw-word primitives. It handles NaN explicitly,
uses the raw sign guard, and evaluates the negative branch
`EXP(x) -> ADD_RNE(1,e) -> DIV_RNE(e,d)` or nonnegative branch
`NEG_BITS(x) -> EXP -> ADD_RNE(1,e) -> DIV_RNE(1,d)`. `EXP` is exactly
`CUSTOM_EXP_CR_U32_V1`; `ADD_RNE` and `DIV_RNE` are fixed total software-binary32 integer
algorithms with explicit NaN/infinity/zero rules, normalization, guard/round/sticky bits, and
ties-to-even packing. They choose represented operands and exact non-FTZ results, one legal
DAZ/RNE/FTZ path at each Model site. Every intermediate is materialized as a raw binary32 word in a
guarded buffer; the proof never relies on opaque fusion or excess precision.

The reviewable proof package for these two candidates has mandatory, independently checked
obligations:

1. freeze source, generator, coefficient/table/exception bytes, proof checker, compiler options,
   resource ledger, and SHA-256 identities before any device process;
2. prove raw classification and all special outputs, including both zeros, both infinities, every
   NaN word, overflow, underflow, and exact subnormal publication;
3. prove every multi-limb primitive, carry bound, shift precondition, table index, and buffer offset;
   for element count $N=\prod_{i=0}^{rank-1}d_i$, prove checked positive multiplication and the
   launch map $\ell=gid_x+W(gid_y+Hgid_z)$ covers every integer $0\le\ell<N$ exactly once and no
   other thread writes, for every resource-feasible positive rank-`1..16` Shape;
4. prove range-reduction uniqueness and coverage, outward interval validity, approximation and
   truncation error, exception-table exhaustiveness, and final round-to-nearest-even packing;
5. independently enumerate all `2^32` raw inputs with a separate at-least-256-bit reference/checker,
   recording class counts, maximum ordered distance and witness, unresolved interval count,
   exception count, and a certificate digest; EXP must have zero failures and maximum distance
   `<=5`;
6. prove `ADD_RNE` and `DIV_RNE` against their complete raw-word truth tables or equivalent checked
   partitions, then prove stable SIGMOID site-by-site recursive membership for all `2^32` inputs,
   including exact branch selection and materialized intermediate words;
7. establish that the Metal source uses only the proved unsigned operations, fixed constants, and
   safe compile settings, with no optimizer-enabled floating reassociation or hidden intrinsic;
8. independently review the source-to-certificate correspondence and rerun the proof checker from a
   clean environment before assigning `DOMAIN-PASS`; and
9. keep capability false and stage both candidates outside the repository until all obligations
   pass.

This document can statically prove the current direct/inherited routes are domain-blocked, preserve
the recorded numerical failures, and define a sufficient auditable proof shape. It cannot claim the
new designs are `DOMAIN-PASS` until concrete source/constants/certificates exist and are reviewed.
That work requires a new explicitly authorized successor task. Because Task 0051's sole device
invocation is consumed, the successor must own any fresh compiled-device regression oracle; it must
not rerun, amend, or relabel the Task-0051 invocation.

## Consumed candidate staging and exactly one oracle

The completed capability-false invocation staged these historical implementations:

1. direct MPSGraph `exponentWithTensor` for `EXP`;
2. one disposable custom Metal `float` EXP kernel;
3. direct MPSGraph sigmoid for `SIGMOID`;
4. one disposable custom stable-branch sigmoid kernel; and
5. one MPSGraph stable sigmoid composition using explicit negation/exponential/addition/division
   sites and graph constants.

The disposable oracle was outside the repository and exported no ABI. It compiled all five
candidates first, then performed exactly one real-device process invocation with one default-device
context, direct supplied targets, recorded optimization/fast-math state, input-preservation and
target-canary checks, and the same frozen corpus for every candidate. There was no retry, second
context, optimization-level matrix, route fallback, or repeated oracle. That invocation is now
consumed permanently.

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

The raw oracle report names every historical candidate, corpus SHA-256, output SHA-256, maximum
ordered distance and its input, FTZ count, class/sign/range verdicts, canary/input/Shape verdicts,
and bounded `PASS`/`FAIL`. A candidate could proceed only with both `DOMAIN-PASS` and numerical
`PASS`; none did. The task Result retains the unedited raw verdict text and SHA-256; all disposable
artifacts were removed.

## One bounded route adjudication

**Not authorized:** no current candidate is `DOMAIN-PASS`, the Task-0051 oracle is consumed, and
supported trace tooling is also unavailable. This workload did not run and must not run under Task
0051.

The unexecuted historical adjudication specification was:

```text
profile = ACCELERATOR
input = canonical caller FLOAT32 [1_048_576]
outputs = EXP(input), SIGMOID(input)
optimization = production MPSGraphCompilationDescriptor default
reducedPrecisionFastMath = None with readback
corpus = frozen edge prefix, then Float.floatToRawIntBits((((i * 17) % 257) - 128) / 16.0f)
```

It would have prepared every surviving route before timing and exact-checked it against the
recursive-floor gate.
It would have used four alternating-order warmup rounds and eight alternating-order retained rounds
in one process; each retained batch would run to a 25 ms floor with a 1,000,000-execution ceiling.
Nothing would be discarded or retried. The hot boundary would be synchronous route execution plus
result closure only, excluding preparation, upload, download, oracle checks, and reporting.

It would have recorded every raw batch duration, iteration count, normalized duration, and
descriptive median. After uncaptured timing, `MTLCaptureManager` queue capture would cover exactly
one hot logical execution of each passing route in the same workload. Supported installed tooling
would have to read each trace's SHA-256, actual compute-dispatch records, and peak
transient/temporary Metal resource bytes above steady prepared inputs/outputs. A custom resource
ledger is only a cross-check; framework-internal work cannot be relabeled as a single backend call
or reported only as opaque. Missing actual dispatch or peak temporary facts blocks capability.

The corrected Task-0051 selection result is empty:

1. remove every candidate without both Gate-1B `DOMAIN-PASS` and numerical `PASS`;
2. all five historical candidates are removed at step 1, so Task 0051 has no baseline, alternative,
   timing population, tie, route winner, or production decision; and
3. a new explicitly authorized successor must predeclare its own complete-domain candidates,
   one-shot numerical gate, all-survivor cost rule, and deterministic tie-break before measurement.

Any future decision is fixed in production, never a runtime threshold, tuning matrix, retry,
fallback, or input-dependent choice, and removes every unselected disposable candidate artifact.

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
- Original execution base revision: `621555a500355d4f0a4b32a54bf7d828331e12f0`; Gate-1B
  remediation base: `6efe3b859ebcb84ffdc05e8925d7bbb9da52782d`.
- Conflicts with: every concurrent Metal capability/schema/native-preflight/candidate/codec/tuning/
  lifecycle/public Engine edit and shared numerical-profile documentation.
- Parallel group: None.
- Integration order reached: original planning; capability-false historical staging; exactly one
  consumed oracle; artifact removal; Gate-1B evidence correction. A new explicitly authorized
  successor must own concrete custom source/certificates, independent complete-domain review, its
  own one-shot oracle, instrumentation-enabled adjudication, route freeze, production work, and
  independent Class C review.
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

1. The consumed 83-word oracle remains exact regression/rejection evidence with unchanged raw
   hashes/verdicts; it is not complete-domain certification and is never rerun.
2. Every historical candidate has a separate Gate-1B verdict. Direct EXP and its MPSGraph stable
   SIGMOID composition are `DOMAIN-BLOCKED`; the original custom EXP and both failed SIGMOID
   candidates retain their numerical failures.
3. The successor-only custom EXP and stable SIGMOID designs name all source, fixed-point,
   special-value, approximation, exception, arithmetic-site, full-raw-domain, rank/launch,
   certificate, and independent-review obligations without claiming those unbuilt candidates pass.
4. No timing, trace capture, route decision, production staging, build, native invocation, or test
   runs in this remediation.
5. Capability, schema 11, wires `1..19`, version-twelve identities, ABI 4, thirteen exports,
   production code, and every strict/excluded row remain unchanged.
6. Task/master/roadmap wording, Markdown, path scope, artifact absence, and `git diff --check` pass;
   the remediation is committed for independent planning/evidence review.

## Validation

Validate only the planning/evidence remediation:

```bash
git diff --check
```

Also validate the three changed Markdown files for links, anchors, fences, final newlines, trailing
whitespace, status consistency, exact retained oracle hashes, no Task-0052 task-file edit, and zero
production/native/test/generated artifacts. Do not run a build, test, native process, timing
workload, trace capture, or device oracle.

## Documentation and review impact

Synchronize only this task's Gate-1B correction and the Task-0051 summaries in the Metal master plan
and roadmap. Preserve the original raw oracle evidence, every historical blocked record, and
Task-0052 content under independent review. A future successor's independent Class C review must
inspect concrete custom source/constants/certificates, complete-domain proofs, its newly authorized
one-shot numerical evidence, instrumentation-backed cost evidence, fixed route, false exclusions,
stale-data rejection, lifecycle, and removed artifacts.

## Result

Blocked before complete-domain authorization, route timing, or capability. Planning completed at
`3a81874528b9a3cbcca4a5643c121bc164c2f75a`; this Gate-1B remediation corrects the interpretation of
its evidence. Production capability remained false throughout, and no Java, native, schema,
identity, test, guide, Javadoc, ABI, export, package, or generated artifact changed.

Exactly one fresh Apple M3 Max numerical-oracle process invocation ran with the frozen 83-word
corpus, MPSGraph's production-default optimization level `1`,
`MPSGraphReducedPrecisionFastMathNone` (`0`), custom `MTLMathModeSafe` (`0`), direct supplied
targets, input preservation, and target canaries. Oracle source SHA-256 was
`b25723c322d9a3078272ea0e79d090d296312186b0e1563b8021f6b7c67e865a`; executable SHA-256 was
`35432d9124478d01a1729ab01d4d2a10778fbf6a8bc87a79b02b0c8b6b881e8f`; raw device-output SHA-256
was `3b762b96b3a1a57aecb97384362069ac2cc9187d271103bc333e841be4274e01`. The 200-digit exact
reference/recursive-membership validator SHA-256 was
`e851ad5c9feabec8282052d458cd40b1efeacddada7723f1292dd2025060794f`; its raw verdict SHA-256 was
`6ac29b44e9a1a693f0e4822ddf6165ef690151f16b63b7f1110434b007be17db`.

The raw recursive verdict was:

```text
REFERENCE_PRECISION_DIGITS=200
CORPUS_COUNT=83
CORPUS_SHA256=06e9137c3b6a9ba147f0a0b73c7face1f5b8cf15ef150964040b64f9be2cad25
RAW_OUTPUT_SHA256=3b762b96b3a1a57aecb97384362069ac2cc9187d271103bc333e841be4274e01
CANDIDATE=direct_exp VERDICT=PASS MAX_ORDERED_DISTANCE=1 MAX_INPUT=3f800000 FTZ_COUNT=16 CLASS_SIGN_RANGE=PASS FAILURES=0
CANDIDATE=custom_exp VERDICT=FAIL MAX_ORDERED_DISTANCE=53 MAX_INPUT=42b17217 FTZ_COUNT=16 CLASS_SIGN_RANGE=PASS FAILURES=9
FAIL name=custom_exp input=c19fffff actual=310da44b allowed={822977600, 822977601, 822977602, 822977603, 822977604, 822977605, 822977606, 822977607, 822977608, 822977609, 822977599}
FAIL name=custom_exp input=419fffff actual=4de7581e allowed={1307007010, 1307007011, 1307007012, 1307007013, 1307007014, 1307007015, 1307007016, 1307007017, 1307007018, 1307007019, 1307007020}
FAIL name=custom_exp input=41a00001 actual=4de7585a allowed={1307007072, 1307007073, 1307007074, 1307007075, 1307007076, 1307007077, 1307007078, 1307007068, 1307007069, 1307007070, 1307007071}
FAIL name=custom_exp input=429fffff actual=792abb5b allowed={2032843635, 2032843636, 2032843637, 2032843638, 2032843639, 2032843640, 2032843641, 2032843642, 2032843643, 2032843644, 2032843645}
FAIL name=custom_exp input=42a00001 actual=792abc0c allowed={2032843808, 2032843809, 2032843810, 2032843811, 2032843812, 2032843813, 2032843814, 2032843815, 2032843816, 2032843806, 2032843807}
FAIL name=custom_exp input=42afffff actual=7ef88243 allowed={2130215488, 2130215478, 2130215479, 2130215480, 2130215481, 2130215482, 2130215483, 2130215484, 2130215485, 2130215486, 2130215487}
FAIL name=custom_exp input=42b00000 actual=7ef88299 allowed={2130215602, 2130215603, 2130215604, 2130215605, 2130215606, 2130215607, 2130215608, 2130215609, 2130215610, 2130215611, 2130215612}
FAIL name=custom_exp input=42b00001 actual=7ef88346 allowed={2130215726, 2130215727, 2130215728, 2130215729, 2130215730, 2130215731, 2130215732, 2130215733, 2130215734, 2130215735, 2130215736}
CANDIDATE=direct_sigmoid VERDICT=FAIL RECURSIVE_MEMBERSHIP=FAIL MAX_SET_SIZE=11 MAX_SET_INPUT=c0000000 FTZ_COUNT=16 CLASS_SIGN_RANGE=PASS FAILURES=2
FAIL name=direct_sigmoid input=bf900000 actual=3e7af78f allowed=3e7af78b,3e7af78c,3e7af78e,3e7af790,3e7af791,3e7af793,3e7af794,3e7af796
FAIL name=direct_sigmoid input=bf900001 actual=3e7af78f allowed=3e7af789,3e7af78b,3e7af78c,3e7af78e,3e7af790,3e7af791,3e7af793,3e7af794,3e7af796
CANDIDATE=custom_sigmoid VERDICT=FAIL RECURSIVE_MEMBERSHIP=FAIL MAX_SET_SIZE=11 MAX_SET_INPUT=c0000000 FTZ_COUNT=16 CLASS_SIGN_RANGE=PASS FAILURES=1
FAIL name=custom_sigmoid input=c19fffff actual=310da44b allowed=310da43f,310da440,310da441,310da442,310da443,310da444,310da445,310da446,310da447,310da448,310da449
CANDIDATE=composed_sigmoid VERDICT=PASS RECURSIVE_MEMBERSHIP=PASS MAX_SET_SIZE=11 MAX_SET_INPUT=c0000000 FTZ_COUNT=16 CLASS_SIGN_RANGE=PASS FAILURES=0
PASSING_CANDIDATES=direct_exp,composed_sigmoid
ORACLE_VERDICT=PASS
```

The composed SIGMOID numerical validator selected captured device results from the negative or
nonnegative stable branch with the exact raw-bit sign guard, then checked membership in the sampled
recursive site-by-site union. Direct SIGMOID was not given a final elementary distance envelope.
These are bounded numerical facts only: direct EXP and composed SIGMOID passed 83 words, while the
original custom EXP/custom SIGMOID/direct SIGMOID failed. Direct EXP remains `DOMAIN-BLOCKED`, so
the composition inheriting it is also `DOMAIN-BLOCKED`; neither sampled pass is certified for all
binary32 inputs, ranks `1..16`, or positive checked extents.

The required actual GPU dispatch and peak transient-memory instrumentation was unavailable before
timing. The installed command-line tools produced these raw results:

```text
$ xcrun --sdk macosx xctrace list templates
xcrun: error: unable to find utility \"xctrace\", not a developer tool or in PATH
exit=72

$ /usr/bin/xctrace list templates
xcode-select: error: tool 'xctrace' requires Xcode, but active developer directory '/Library/Developer/CommandLineTools' is a command line tools instance
exit=1
```

`MTLCaptureManager` can describe a GPU-trace destination, but this workstation has no supported
installed trace reader capable of extracting actual framework-internal compute dispatches and peak
transient resource bytes. A production-equivalent Level-1/fast-math-none adjudication harness
compiled successfully (source SHA-256
`0ad83c650d00c24c74e225c7d7502fee2c293f521d68747f538ceecd859d3b12`, executable SHA-256
`4fb094ec5ea9472af53adca8272bb5aafcdbe6d08e59f32c52f4f8768503a762`) but was deliberately never
run, so there is no timing, inferred backend-call count, opaque-memory substitute, route winner, or
capability authorization. All disposable oracle, validator, raw-output, verdict, adjudicator source,
and executable artifacts were removed after this evidence was recorded.

The static remediation adds no candidate source and consumes no device action. It proves only the
current Gate-1B blocker/failure classification and freezes the proof obligations for
`CUSTOM_EXP_CR_U32_V1` and `CUSTOM_SIGMOID_STABLE_U32_V1`. Concrete generation, exhaustive
certificate production, independent proof review, and any compiled-device regression require a new
explicitly authorized successor task with its own one-shot rule. Even after such a successor obtains
`DOMAIN-PASS` and numerical `PASS`, actual GPU dispatch and peak transient-memory facts remain
mandatory before timing can select a route or capability can change.
