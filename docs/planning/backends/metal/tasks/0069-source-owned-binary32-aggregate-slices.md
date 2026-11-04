# Task 0069: Source-Owned [Binary32](../../../../glossary.md#floating-point-precision-and-rounding) Aggregate-Floor Slices

## Reader context

A leaf here is one aggregate-tree contributor, not an application Tensor leaf. L1 uses raw ABS contributors; ScatterAdd starts from a base word and includes each matching update occurrence; singleton variance keeps literal arithmetic sites. An invented zero leaf can change signed-zero/rounded behavior despite a real-number identity.

Model owns legal site results. Lean proofs, the source certificate and compiled Apple Intermediate Representation (AIR) inspection connect meaning to specific shader bytes; rebinding maintains evidence, not a runtime certificate. Profile-era slices and later source renewals are distinct checkpoints, not today's entire numerical domain.

- [Current contributors and arithmetic sites](../../../../backend-guide/metal-backend.md#numerical-semantics-and-capability) — Explains aggregate leaves and membership.
- [Current proof provenance and source qualification](../../../../backend-guide/metal-backend.md#evidence-composition) — Separates retained proofs from approval of changed implementations.

## Status

Complete. The shared proof substrate and Slices 1–3 are independently approved. The post-cutover
source/[AIR](../../../../glossary.md#term-air) evidence was rebound to current [schema](../../../../glossary.md#term-schema)-20 source and passed final independent Class C
rereview without changing production capability or numerical behavior.

### Post-cutover review remediation (closed)

This amendment supersedes the profile-era evidence instructions below only where they refer to a
current [`ACCELERATOR`](../../../../glossary.md#term-accelerator) selector, [schema](../../../../glossary.md#term-schema) 16, [identity](../../../../glossary.md#term-identity-meanings) 25, or a source [hash](../../../../glossary.md#term-hash) predating [Model](../../../../glossary.md#term-model) 0032.
Historical checkpoint records remain historical; current authority is `ARCHITECTURE.md` Core
invariants, `foundational-modules.md` [Profile-free numerical semantics](../../../../glossary.md#profile-free-numerical-semantics), and
`backend-execution.md` [Metal](../../../../glossary.md#term-metal) [backend](../../../../glossary.md#backend) / Profile-free [backend capability](../../../../glossary.md#backend-capability) and identity.

- Goal: rebind the retained Task-0069 source/compiler-site and compiled-MSL review evidence to
  the exact current native foundation and fixed kernel assembly. Audit the changes since the
  Slice-3 checkpoint in 0069 preflight, route selection, index validation, dispatch, buffer
  ownership, and source assembly before changing any hash. Preserve the unchanged proof and
  literal DIV–SUB–MUL–DIV kernel; do not turn the retained evidence into a runtime certificate.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [MSL glossary](../../../../glossary.md#term-metal-shading-language-msl); [kernel glossary](../../../../glossary.md#kernel); [Slice glossary](../../../../glossary.md#slice); [preflight glossary](../../../../glossary.md#term-preflight); [dispatch glossary](../../../../glossary.md#term-dispatch); [buffer glossary](../../../../glossary.md#term-buffer); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [runtime glossary](../../../../glossary.md#term-runtime); [certificate glossary](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions))</span>
- Files: `backends/metal/evidence/0069/` manifests, extraction/verification scripts as necessary;
  read-only review of `native/metal-macos-arm64/src/synaptik_metal_foundation.m` and
  `synaptik_task0069_aggregate_kernels.h`. No production-source edit is authorized here.
- Acceptance: certificate checks current source and profile-free wording; the exact current fixed
  source is compiled with pinned safe options; AIR inspection still proves the three bounded
  0069 kernels' source/site/store invariants; the package source authentication and existing
  schema-20/profile-free conformance remain intact. If an invariant fails, stop and report a
  separate behavior remediation instead of weakening the verifier or updating a hash blindly.<span data-glossary-navigation="true"> ([AIR glossary](../../../../glossary.md#term-air); [kernels glossary](../../../../glossary.md#kernel); [authentication glossary](../../../../glossary.md#term-authentication); [conformance glossary](../../../../glossary.md#term-conformance); [invariant glossary](../../../../glossary.md#term-invariant))</span>
- [Validation](../../../../glossary.md#term-validation): [run](../../../../glossary.md#term-run) `backends/metal/evidence/0069/run-proof.sh`, check current source/package
  authentication and `git diff --check`; reuse Model-0032's successful full build and focused
  device evidence because no executable source changes. Select the installed pinned compiler with<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device))</span>
  `TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.921.5`; default `xcrun` does not currently select
  that component. Independent targeted Class C rereview then decides final Slice-3 acceptance.
- Frontier verification: Model 0032 is `Complete`; Task 0069's original Slice-3 implementation is
  present; the independent review P1 is the identified evidence gate. The pinned toolchain is
  installed and its <span data-glossary-navigation="true"> ([toolchain glossary](../../../../glossary.md#term-toolchain))</span>`metal`/`metallib` versions were verified with explicit selection. No
  conflicting Metal writer is active. Parallel group: None. Common base revision: `1cab0731b75c035f589a60f8b9922a767211e6c6`.
  Integration order: evidence remediation → independent review → 0069 closure → 0074 readiness.
  Integration validation: focused proof/source/AIR runner plus documentation/diff checks.
  Shared-document integration owner: main coordination context.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>

## Change class

Class C — the eventual work changes numerical capability, generated-[gradient](../../../../glossary.md#term-gradient) reachability, source-owned [GPU](../../../../glossary.md#term-gpu) code, private preparation [identity](../../../../glossary.md#term-identity-meanings), and [validation](../../../../glossary.md#term-validation)-before-mutation behavior.

## Goal

Build the smallest task-local [binary32](../../../../glossary.md#floating-point-precision-and-rounding) and contributor-tree proof substrate, then consider three independently
shippable [`ACCELERATOR`](../../../../glossary.md#term-accelerator)/[`FLOAT32`](../../../../glossary.md#term-float32) custom-route slices in strict serial order: `L1_NORM`, `SCATTER_ADD`, then
[`VARIANCE`](../../../../glossary.md#term-variance). A failed proof, source audit, or [runtime](../../../../glossary.md#term-runtime) gate leaves that row false and promises no later [slice](../../../../glossary.md#slice).

## Baseline audit and decisions

- Common base is clean `69c07e30e2ec90c91de604c396896ee258e851e5`. Task 0067's current 115-wire ledger
  is structural `101/14`, capability `83/32`, [MPSGraph](../../../../glossary.md#term-mpsgraph) `75/35/5`, custom `70/45/0`, [schema](../../../../glossary.md#term-schema) 15, [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5, thirteen exports, route wires `1..3`, [identity](../../../../glossary.md#term-identity-meanings) 22.
- Model owns exact mapping, contributor membership, empty/point/divisor rules, binary32 RNE
  primitive sites with set-valued DAZ/FTZ, and any all-terms-once binary aggregate tree. L1 is<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [L1 glossary](../../../../glossary.md#term-l1))</span>
  `ABS` then sum; scatter-add includes base and every matching update occurrence; variance is mean,
  one subtraction and square per contributor, aggregate sum, then <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [mean glossary](../../../../glossary.md#term-mean))</span>`N-correction` division.
- Compiler keeps the existing formulas: L1's cotangent is sign-selected upstream with zero at
  signed zero/NaN; variance is <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [cotangent glossary](../../../../glossary.md#cotangent); [signed zero glossary](../../../../glossary.md#term-signed-zero); [NaN glossary](../../../../glossary.md#term-nan))</span>`g * 2 * (x-mean) / (N-correction)`; scatter-add returns upstream for
  base, no index cotangent, and `gather(upstream, indices, axis)` for updates. This task does not
  alter those formulas or first-order classifications.
- Task 0060's MPSGraph recipes prove only structural execution. Current direct `SCATTER_ADD` and
  [`VARIANCE`](../../../../glossary.md#term-variance), and composed `L1_NORM`, are opaque and cannot authorize production. Every selected
  occurrence therefore uses one fixed source-owned <span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque))</span>`CUSTOM_PROGRAM`; the retained MPSGraph states
  remain descriptive and are never fallback candidates.
- Task 0053 remains `CORROBORATED_NOT_CERTIFIED`; its EXP partitions, finite checks, and source
  correspondence do not transfer. Only its useful proof discipline is retained: total raw-word
  classification, unsigned bounds, integer ties-to-even lemmas, source inventory, pinned tools, and
  explicit proof-status manifests. No Task-0053 theorem is treated as aggregate authorization.<span data-glossary-navigation="true"> ([EXP glossary](../../../../glossary.md#term-expx); [finite glossary](../../../../glossary.md#term-finite); [ties-to-even glossary](../../../../glossary.md#term-ties-to-even))</span>
- Task 0061's accepted source/compiler-site certificate is the recipe for binding exact kernel text,
  safe compiler mode, site inventory, and artifacts. Task 0069 adds no general numerical framework.<span data-glossary-navigation="true"> ([certificate glossary](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions); [binding glossary](../../../../glossary.md#term-binding); [kernel glossary](../../../../glossary.md#kernel); [artifacts glossary](../../../../glossary.md#term-artifact))</span>

## Exact [occurrence](../../../../glossary.md#term-occurrence) domains

All slices require [`ACCELERATOR`](../../../../glossary.md#term-accelerator), [`FLOAT32`](../../../../glossary.md#term-float32) data/result, fully static positive [canonical](../../../../glossary.md#term-canonical-form)-dense
[layouts](../../../../glossary.md#layout), 32-bit-bounded [element](../../../../glossary.md#term-element)/byte/[dispatch](../../../../glossary.md#term-dispatch) arithmetic, distinct authenticated value-table
[buffers](../../../../glossary.md#term-buffer), and no [gradient](../../../../glossary.md#term-gradient)-bearing role on the selected [operation](../../../../glossary.md#operation). A generated [Gather](../../../../glossary.md#gather) may require
gradients, but its emitted Scatter-Add [node](../../../../glossary.md#node) itself must satisfy that no-gradient domain.
`STRICT_IEEE`, dynamic/empty/noncanonical/aliased occurrences and every unlisted [rank](../../../../glossary.md#term-rank), [axis](../../../../glossary.md#term-axis),
[carrier](../../../../glossary.md#term-storage-carrier), attribute, or gradient combination remain false.

### [Slice](../../../../glossary.md#slice) 1 — `L1_NORM` wire 114

- Input [rank](../../../../glossary.md#term-rank) is exactly one with [extent](../../../../glossary.md#term-extent) `N>0`; `N`, the four-byte input span, and dispatch count
  must each fit unsigned 32-bit arithmetic. Axes are exactly <span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [Axes glossary](../../../../glossary.md#term-axis))</span>`[0]`, never empty.
  `keepDimensions=false` produces [canonical](../../../../glossary.md#term-canonical-form) rank zero; `true` produces canonical `[1]`.
- One output thread clears each input sign bit for exact raw `ABS`. Its reduction tree has exactly
  the `N` ABS values as leaves and exactly `N-1` actual safe binary32 additions in increasing
  contributor order; it has no zero leaf. <span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>`N=1` is the direct ABS leaf with no add site. The result
  stores once, with no invented identity, pre-truncation, fusion, epsilon, or hidden reduction.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [truncation glossary](../../../../glossary.md#term-truncation-toward-zero); [fusion glossary](../../../../glossary.md#term-fusion); [epsilon glossary](../../../../glossary.md#epsilon))</span>
- Either input zero becomes `+0`; [NaN](../../../../glossary.md#term-nan) remains NaN; [infinity](../../../../glossary.md#term-infinity) yields `+infinity` unless a NaN is
  present; finite results are nonnegative.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite))</span>
- Gradient-bearing L1 remains false, so the existing sign/zero/NaN cotangent graph is not newly
  reachable.<span data-glossary-navigation="true"> ([Gradient glossary](../../../../glossary.md#term-gradient); [L1 glossary](../../../../glossary.md#term-l1); [cotangent glossary](../../../../glossary.md#cotangent); [graph glossary](../../../../glossary.md#graph))</span>

### [Slice](../../../../glossary.md#slice) 2 — `SCATTER_ADD` wire 70

- Data/result [rank](../../../../glossary.md#term-rank) is exactly one with length `D>0`; indices and updates are rank one with common
  length `U>0`; [axis](../../../../glossary.md#term-axis) is exactly zero. Indices are [`INT32`](../../../../glossary.md#data-type--datatype) or [`INT64`](../../../../glossary.md#data-type--datatype), are a materialized partition
  feed, and update Shape is exactly the rank-one Gather result. All data/update/result gradient
  flags are false.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [Shape glossary](../../../../glossary.md#shape))</span>
- A complete [CPU](../../../../glossary.md#term-cpu)-side scan validates every `0 <= index < D` before any command encoding, graph-node
  dispatch, or output mutation. Duplicate targets are legal and are never uniqueness-filtered.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [dispatch glossary](../../../../glossary.md#term-dispatch); [targets glossary](../../../../glossary.md#term-target))</span>
- One output thread owns each target: it raw-loads base exactly once, scans update ordinals in
  increasing order, applies one safe binary32 add for every matching occurrence, and stores once.
  An unaddressed cell is a raw base copy, preserving NaN payload/quietness, subnormal bits, and
  signed zero. Addressed NaN, infinity, subnormal, overflow, and signed-zero results come only from
  the proved set-valued site chain; no final canonicalization is allowed. There are no atomics or
  write races.<span data-glossary-navigation="true"> ([binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [occurrence glossary](../../../../glossary.md#term-occurrence); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [subnormal glossary](../../../../glossary.md#term-subnormal); [signed zero glossary](../../../../glossary.md#term-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [overflow glossary](../../../../glossary.md#term-overflow); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- The same admitted no-[gradient](../../../../glossary.md#term-gradient) [node](../../../../glossary.md#node) closes the existing first-order [`GATHER`](../../../../glossary.md#gather) data cotangent only
  for canonical rank-one <span data-glossary-navigation="true"> ([cotangent glossary](../../../../glossary.md#cotangent))</span>[`FLOAT32`](../../../../glossary.md#term-float32) data, [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) materialized [canonical](../../../../glossary.md#term-canonical-form) `INT32`/`INT64` indices,
  axis zero, positive static extents, and canonical cotangent. The compiler-generated zero base is
  explicitly represented by <span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler))</span>[`CONTIGUOUS`](../../../../glossary.md#term-contiguous) before the Scatter-Add node; upstream updates are captured
  and proved end to end. Gradient-bearing `SCATTER_ADD`, its base/update
  cotangents, Gather-Elements/Gather-ND, and higher order remain false; no origin discriminator is
  added. Slice 2 atomically replaces the backend contract's current additive-scatter fail-closed statement with only this exact domain and generated Gather closure; it cannot ship beforehand.<span data-glossary-navigation="true"> ([Elements glossary](../../../../glossary.md#term-element); [Gather-ND glossary](../../../../glossary.md#gather-nd); [backend glossary](../../../../glossary.md#backend); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>

#### [Slice](../../../../glossary.md#slice)-2 [scope](../../../../glossary.md#term-scope) amendment — representation-only [Gather](../../../../glossary.md#gather) closure

The existing Gather data-[cotangent](../../../../glossary.md#cotangent) formula is still
`scatter_add(zero_like(data), indices, upstream, axis)`, but `zero_like(data)` is physically a
zero-[stride](../../../../glossary.md#term-stride) `EXPAND`. The exact [occurrence](../../../../glossary.md#term-occurrence) domain above forbids that noncanonical base just as it
forbids every other noncanonical Scatter role. Slice 2 therefore explicitly authorizes one shared
[Compiler](../../../../glossary.md#term-compiler) representation [operation](../../../../glossary.md#operation):
`constants.zeroLike(data).contiguous().scatterAdd(indices, gradient, axis)`.
[`CONTIGUOUS`](../../../../glossary.md#term-contiguous) changes only the zero [tensor](../../../../glossary.md#tensor)'s physical representation; it does not change the
mathematical cotangent formula, first-order classification, [target](../../../../glossary.md#term-target)/update values, or any public
operation contract. [Backend](../../../../glossary.md#backend)-neutral compiler [graph](../../../../glossary.md#graph) coverage must prove the explicit
[canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), [CPU](../../../../glossary.md#term-cpu) execution must prove the unchanged cotangent, and [Metal](../../../../glossary.md#term-metal)-only public
execution must prove the now-[canonical](../../../../glossary.md#term-canonical-form) Scatter occurrence. CPU's generic no-specialized-geometry
range [guard](../../../../glossary.md#term-guard) uses the output [binding](../../../../glossary.md#term-binding)'s logical [element](../../../../glossary.md#term-element) count, rather than a broadcast [scalar](../../../../glossary.md#term-scalar) source
binding, so this explicit [materialization](../../../../glossary.md#materialization) remains executable without changing arithmetic. This
reviewed amendment is the sole exception to the Compiler non-goal below; it does not authorize any
other Compiler formula, representation, or CPU-[kernel](../../../../glossary.md#kernel) change.

### [Slice](../../../../glossary.md#slice) 3 — [`VARIANCE`](../../../../glossary.md#term-variance) wire 112

- Input [rank](../../../../glossary.md#term-rank) is exactly one with [extent](../../../../glossary.md#term-extent) `N=1`, [axes](../../../../glossary.md#term-axis) exactly `[0]`, correction exactly zero, and no
  gradient flags. Both keep forms target one canonical scalar cell; every other domain stays false.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [target glossary](../../../../glossary.md#term-target); [canonical glossary](../../../../glossary.md#term-canonical-form); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Exactly one dispatched output thread owns that cell and stores once. With no payload classifier
  or special publication, it evaluates only <span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>`mean=DIV(x,+1)`, `d=SUB(x,mean)`, `q=MUL(d,d)`, the
  singleton square-sum leaf <span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [leaf glossary](../../../../glossary.md#term-leaf))</span>`q`, then `DIV(q,+1)`. Each name is one safe binary32 site; there is no
  aggregate add, clamp, reciprocal substitution, FMA, stable replacement, or final tolerance.<span data-glossary-navigation="true"> ([binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [FMA glossary](../../../../glossary.md#term-fused-multiply-add-fma); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>
- The proof must derive [NaN](../../../../glossary.md#term-nan) for NaN/[infinity](../../../../glossary.md#term-infinity) and exact `+0` for every finite input, including both
  zeros and subnormals under every DAZ/FTZ choice, from those sites alone.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [subnormals glossary](../../../../glossary.md#term-subnormal); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz))</span>

## Proof and source gates

1. Add task-local `Task0069Binary32.lean` and `Task0069ReductionTree.lean`: total binary32 raw
   classes; exact typed-one facts; set-valued RNE/DAZ/FTZ primitive relations; leaf-labelled binary
   trees/multisets; and the theorem that each source left chain is a permitted all-terms-once tree.
   L1's leaves are exactly the ABS values, internal nodes correspond one-to-one with the <span data-glossary-navigation="true"> ([binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz); [leaf glossary](../../../../glossary.md#term-leaf); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [L1 glossary](../../../../glossary.md#term-l1); [nodes glossary](../../../../glossary.md#node))</span>`N-1`
   source additions, and its singleton tree is the direct ABS leaf. No <span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton))</span>`sorry`, axioms, native
   floating oracle, or general framework.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle))</span>
2. Add per-slice Lean domain theorems. L1 proves raw ABS, point, class, and contributor obligations.
   Scatter proves base-plus-filtered-occurrence membership per target, duplicates retained,
   unaddressed raw identity, and one writer. Singleton variance proves literal-site membership,
   special classes, one-thread ownership, and one store without preclassification.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [Lean glossary](../../../../glossary.md#proof-assistant-and-model-checking-tools); [occurrence glossary](../../../../glossary.md#term-occurrence); [target glossary](../../../../glossary.md#term-target); [identity glossary](../../../../glossary.md#term-identity-meanings); [variance glossary](../../../../glossary.md#term-variance); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
3. Bind the exact header, native dispatcher/preflight, compiler flags, function names, loop/site
   inventory, generated library, and proof inputs in a hash manifest plus source/compiler-site
   certificate. Compile with <span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [compiler glossary](../../../../glossary.md#term-compiler); [hash glossary](../../../../glossary.md#term-hash); [manifest glossary](../../../../glossary.md#term-manifest); [certificate glossary](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions); [Compile glossary](../../../../glossary.md#compile))</span>`MTLMathModeSafe`; reject fast math, reassociation outside the written
   tree, contraction, hidden reduction, and unbound artifacts. Device regressions corroborate but
   never authorize an uncovered domain.<span data-glossary-navigation="true"> ([reassociation glossary](../../../../glossary.md#term-reassociation); [artifacts glossary](../../../../glossary.md#term-artifact); [Device glossary](../../../../glossary.md#term-device))</span>
4. Java and native independently authenticate profile, kind, attributes, carrier, rank, Shape,
   canonical layout, gradient flags, exact counts/bytes, buffer distinctness, dispatch/writer
   geometry, and correction/index constraints before allocation or encoding. Whole-program scatter
   payload validation precedes every dispatch and mutation; invalid execution preserves sentinels.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [rank glossary](../../../../glossary.md#term-rank); [Shape glossary](../../../../glossary.md#shape); [canonical glossary](../../../../glossary.md#term-canonical-form); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [buffer glossary](../../../../glossary.md#term-buffer); [dispatch glossary](../../../../glossary.md#term-dispatch); [validation glossary](../../../../glossary.md#term-validation))</span>
5. Each slice lands as a separate identity/capability/source/proof/test/docs checkpoint and receives
   independent code, evidence, and security review. Failure leaves its row and every later serial
   slice false until a reviewed amendment; already shipped earlier slices remain valid.

## [Identity](../../../../glossary.md#term-identity-meanings) and count ledger

| Accepted checkpoint | Structural | Capability | [MPSGraph](../../../../glossary.md#term-mpsgraph) | Custom | [Schema](../../../../glossary.md#term-schema) / [ABI](../../../../glossary.md#term-application-binary-interface-abi) / exports | Identity |
|---|---:|---:|---:|---:|---|---:|
| Baseline | `101/14` | `83/32` | `75/35/5` | `70/45/0` | `15 / 5 / 13` | 22 |
| [Slice](../../../../glossary.md#slice) 1 only | unchanged | `84/31` | unchanged | `71/44/0` | `16 / 5 / 13` | 23 |
| Slices 1–2 | unchanged | `85/30` | unchanged | `72/43/0` | `16 / 5 / 13` | 24 |
| Slices 1–3 | unchanged | `86/29` | unchanged | `73/42/0` | `16 / 5 / 13` | 25 |

Each checkpoint advances all six [backend](../../../../glossary.md#backend)-local workload/policy/candidate/compatibility/route/[codec](../../../../glossary.md#term-codec)
identities together and rejects the prior identity and all older values. Slice 1 cleanly advances
the image from schema 15 to schema 16 so the first formerly reserved header word carries the exact
existing numerical-profile wire; schema 15 and every other schema reject. Identity remains 23
because its compatibility and codec already bind the profile and its workload [digest](../../../../glossary.md#term-digest) already binds
the schema version and complete encoded image, so no schema-15 decision can match a schema-16
workload. [Operation](../../../../glossary.md#operation), attribute, type, and route wires do not change. At each checkpoint every other
false row remains false.

## Non-goals

No [Model](../../../../glossary.md#term-model)/[API](../../../../glossary.md#term-application-programming-interface-api)/[Compiler](../../../../glossary.md#term-compiler)-formula change, new [operation](../../../../glossary.md#operation)/attribute/type/route wire, [ABI](../../../../glossary.md#term-application-binary-interface-abi)/export change,
broad aggregate helper, strict-profile capability, noncanonical or empty geometry, positive
[variance](../../../../glossary.md#term-variance) correction, [L2 norm](../../../../glossary.md#norms-elementary-functions-and-probability-distributions), [mean](../../../../glossary.md#term-mean) expansion, additive [Scatter-ND](../../../../glossary.md#scatter-nd)/[Elements](../../../../glossary.md#term-element), timing, [autotuning](../../../../glossary.md#autotuning--model-autotuning),
fallback, retry, atomics, [host](../../../../glossary.md#term-host) repair, [opaque](../../../../glossary.md#term-opaque)-selector inference, source-unbound corpus proof,
approximate acceptance, or final output [epsilon](../../../../glossary.md#epsilon). The [schema](../../../../glossary.md#term-schema)-16 profile cell is the sole image
grammar revision.

## Contracts

- [`ARCHITECTURE.md` — Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence).
- [Foundational module contract — Numerical profiles / Aggregate floor / Composite inheritance](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [Aggregate floor glossary](../../../../glossary.md#term-aggregate-floor))</span>.
- [Backend execution contract — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Metal glossary](../../../../glossary.md#term-metal))</span>.
- [Backend execution contract — Numerical-profile backend identity](../../../../architecture/contracts/backend-execution.md#numerical-profile-backend-identity)<span data-glossary-navigation="true"> ([backend identity glossary](../../../../glossary.md#backend-identity--backendid))</span>.

If current source or an applicable contract disagrees with this brief, stop, keep the [occurrence](../../../../glossary.md#term-occurrence)
false, and submit a reviewed plan amendment rather than inventing architecture.

## Dependencies and integration

- Depends on: Task 0068 Complete at approved `1775081a`; Task 0067 current ledger; current Model and
  Compiler contracts; retained Tasks 0053/0060/0061 evidence only as bounded above.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Conflicts with: every concurrent Metal capability/native/custom-source/catalog/route/identity/
  package/shared-document edit, Compiler indexing/reduction/autograd edit, and resumed Task 0053
  production work.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [identity glossary](../../../../glossary.md#term-identity-meanings); [autograd glossary](../../../../glossary.md#autograd))</span>
- Parallel group: None.
- Common base revision: `69c07e30e2ec90c91de604c396896ee258e851e5`.
- Integration order: proof substrate + [Slice](../../../../glossary.md#slice) 1, then Slice 2, then Slice 3; no batched cutover.
- Integration validation: the slice's proof/certificate gates, native package, complete Metal,
  focused Compiler, Metal conformance/public Engine, architecture/Javadoc, then repository test/build.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [certificate glossary](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): Task 0069 executor, serially.

## Files and symbols

- `backends/metal/evidence/0069/` — pinned task-local proof, manifests, certificates, and checkers.
- `backends/metal/.../MetalCapabilityProvider.java`, `MetalOperationRouteCatalog.java`, preparation
  identity owners and focused tests — exact occurrence admission, fixed custom route, identities.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [occurrence glossary](../../../../glossary.md#term-occurrence); [admission glossary](../../../../glossary.md#term-admission))</span>
- `native/metal-macos-arm64/src/synaptik_task0069_aggregate_kernels.h` and
  `synaptik_metal_foundation.m` — source-owned [kernels](../../../../glossary.md#kernel), independent [preflight](../../../../glossary.md#term-preflight), fixed [dispatch](../../../../glossary.md#term-dispatch).
- `modules/compiler/.../IndexingGradientRules.java`,
  `backends/cpu/.../CpuPreparedExecutable.java`, and focused Compiler/CPU/Metal integration tests —
  the sole representation-only <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal))</span>[`CONTIGUOUS`](../../../../glossary.md#term-contiguous) amendment for generated Gather data cotangents and its
  output-domain range guard, with unchanged mathematical results.<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather); [guard glossary](../../../../glossary.md#term-guard))</span>
- Metal/conformance tests and active architecture/package documentation — exact slice behavior,
  generated Gather closure, false boundaries, counts, identity rejection, and the Slice-2 atomic
  update to <span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [slice glossary](../../../../glossary.md#slice); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>`backend-execution.md` additive-scatter and generated-[gradient](../../../../glossary.md#term-gradient) text.

## Acceptance and [validation](../../../../glossary.md#term-validation)

For each independently approved [slice](../../../../glossary.md#slice): its [Lean](../../../../glossary.md#proof-assistant-and-model-checking-tools) proof and source [certificate](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions) pass; the exact route
executes through the public [Engine](../../../../glossary.md#term-engine) on the packaged [dylib](../../../../glossary.md#term-dylib); special classes, duplicate contributors,
raw-copy cells, correction/count bounds, one-writer/store-once, both keep forms where applicable,
generated [Gather](../../../../glossary.md#gather) closure, false neighboring domains, [alias](../../../../glossary.md#aliasing-and-arena)/error-before-mutation, [identity](../../../../glossary.md#term-identity-meanings)
rejection, and complete count partitions pass. Native builds use warnings as errors. Permanent
tests defend these observable boundaries, not source text or incidental plumbing.

Worker validation is the existing Task-0068 native/package [matrix](../../../../glossary.md#term-matrix-meanings) plus the new `0069` proof runner,
complete `:backends:metal:test`, focused [Compiler](../../../../glossary.md#term-compiler) indexing/reduction/[autograd](../../../../glossary.md#autograd) tests, [Metal](../../../../glossary.md#term-metal)
[conformance](../../../../glossary.md#term-conformance) and public Engine integration, architecture/[Javadoc](../../../../glossary.md#term-javadoc), `./gradlew test`, `./gradlew build`,
and `git diff --check`. No benchmark or timing entry point runs.

## Documentation and review impact

The implementation must update [Metal](../../../../glossary.md#term-metal) package/[Javadoc](../../../../glossary.md#term-javadoc), [backend](../../../../glossary.md#backend) execution [identity](../../../../glossary.md#term-identity-meanings)/count/domain text,
this task, master plan, roadmap, and any affected native package evidence. Class C requires separate
code, numerical-evidence, and security/determinism review at every [slice](../../../../glossary.md#slice) checkpoint.

## Result
### [Slice](../../../../glossary.md#slice) 1 checkpoint

The authorized shared substrate and Slice 1 are implemented. [Metal](../../../../glossary.md#term-metal) now admits only [accelerator](../../../../glossary.md#term-accelerator)
[FLOAT32](../../../../glossary.md#term-float32) no-[gradient](../../../../glossary.md#term-gradient) `L1_NORM` for one [canonical](../../../../glossary.md#term-canonical-form) positive-static [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) input, ordered multi-[axis](../../../../glossary.md#term-axis)
`[0]`, and canonical [scalar](../../../../glossary.md#term-scalar) or retained `[1]` output. Wire 114 always selects the fixed
source-owned `CUSTOM_PROGRAM` under `CA_0069`; Java and native creation reject the [opaque](../../../../glossary.md#term-opaque) direct
[MPSGraph](../../../../glossary.md#term-mpsgraph) route. Java/native [validation](../../../../glossary.md#term-validation) independently enforces the exact domain, unsigned-32-bit
[element](../../../../glossary.md#term-element)/four-byte-span/[dispatch](../../../../glossary.md#term-dispatch) bounds, and distinct [buffers](../../../../glossary.md#term-buffer). One
thread raw-clears every contributor sign bit, initializes from ordinal zero, performs exactly
`N-1` safe [binary32](../../../../glossary.md#floating-point-precision-and-rounding) additions in increasing ordinal order, and publishes one logical result;
`N=1` performs no addition. SCATTER_ADD and [VARIANCE](../../../../glossary.md#term-variance) remain production-false.

The two [Lean](../../../../glossary.md#proof-assistant-and-model-checking-tools) files prove total raw classes; exact binary32 `+1`; exact ABS; complete set-valued
[DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz) alternatives; an explicit [finite](../../../../glossary.md#term-finite) nonnegative nearest-even relation in `2^-149` units with
[overflow](../../../../glossary.md#term-overflow)-midpoint and even-[significand](../../../../glossary.md#term-significand) tie rules; [NaN](../../../../glossary.md#term-nan), positive-[infinity](../../../../glossary.md#term-infinity), and nonnegative non-NaN
class behavior; the aggregate [L1](../../../../glossary.md#term-l1) nonnegative-or-NaN [invariant](../../../../glossary.md#term-invariant); reachable-source NaN absorption with
signed NaN allowed and every non-NaN operand sign-clear; positive-infinity [persistence](../../../../glossary.md#term-persistence) without NaN;
labelled contributor multiplicity; the source left tree's `N-1` add [nodes](../../../../glossary.md#node); direct [singleton](../../../../glossary.md#term-singleton) ABS;
and source-fold membership constrained by `Binary32RneContract` rather than an arbitrary [binary](../../../../glossary.md#term-binary-descriptor-and-metadata)
function. Both [compile](../../../../glossary.md#compile) without `sorry` or axioms. The source [certificate](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions) pins
the exact [kernel](../../../../glossary.md#kernel)/foundation/proof inputs. The required compiled-[MSL](../../../../glossary.md#term-metal-shading-language-msl) audit extracts
the exact assembled [runtime](../../../../glossary.md#term-runtime) source and compiles it with `DEVELOPER_DIR` fixed to signed Xcode 27.0
build 27A266a, Metal 32023.921, macOS [SDK](../../../../glossary.md#term-sdk) 27.0, `metal3.2`, no-fast-math, warnings as errors, and
the explicit SDK isysroot. Pinned source/[AIR](../../../../glossary.md#term-air)/metallib [SHA-256](../../../../glossary.md#term-sha-256) values are
`070c910ff3be6853720274d670fb5a573c490ceec60d680eb323d9001c965c10`,
`fc6258b8627ec607434f02fe63eb77dd05957f9b5821e95b157e1149f2fe3189`, and
`ffa3c751fbab39acff50c06f6fda720e4e831a3d06657ee459921f50d11ba8f3`. AIR inspection finds
exactly one unflagged `fadd` in `l1_norm_f32_0069`, raw sign [masks](../../../../glossary.md#term-boolean-mask) and the ordinal loop, no other
floating arithmetic, and only the final raw [serialization](../../../../glossary.md#term-serialization) path.

Native warnings-as-errors build, fixed-identifier signing, canonical package verification, [Gradle](../../../../glossary.md#term-gradle)
package ingestion/ZIP creation, fresh extraction, extracted-package verification, and byte
comparison passed. The packaged [dylib](../../../../glossary.md#term-dylib) is `484944` bytes with SHA-256
`c036c78259b0d30db14e96ddf7ae205ae92f99901d7afc0be192f46b59a0e529` and ad-hoc CDHash
`6b64ae9f178b4b12b4d0f53dfd525b37f9758a0e`; the local ZIP is `141328` bytes with SHA-256
`4bb2822ac35f3a70ac76dbca72bb1a628efc3f9eefbadb1e05080f9c883cc235`.

The initial implementation checkpoint passed complete Metal, [Compiler](../../../../glossary.md#term-compiler), [conformance](../../../../glossary.md#term-conformance), public [Engine](../../../../glossary.md#term-engine)
Metal integration, architecture, [Javadoc](../../../../glossary.md#term-javadoc), repository test, and serialized full build, totaling
`3718` tests with zero failure/error. Initial independent security review approved with zero
P0/P1/P2. Initial code and numerical-evidence findings required the explicit nearest-even/class
contract, raw native malformed-image coverage, direct [identity](../../../../glossary.md#term-identity-meanings)-22 rejection, and current ledger
documentation; cumulative review also identified and closed the forceable direct-MPSGraph L1 seam.
External design review then required the clean [schema](../../../../glossary.md#term-schema)-16 profile cutover and narrowed the RNE NaN
premises to signed NaN or sign-clear non-NaN reachable operands; the generic source fold carries
the same invariant.

After that remediation, the proof/source-certificate/compiled-MSL runner, focused raw native schema
and L1 tests, identity tests, affine schema [fixture](../../../../glossary.md#term-fixture), and direct public L1 Engine scenario all pass.
The rebuilt signed package, Gradle ingestion/ZIP, fresh extraction, extracted-package verifier, and
byte comparison pass. The final unfiltered serialized `./gradlew test --rerun-tasks
--max-workers=1` passes all 76 actionable tasks in 3m02s. The subsequent unfiltered serialized
`./gradlew build --rerun-tasks --max-workers=1` passes all 87 actionable tasks in 2m57s, including
the unchanged `CpuPartitionDagGeneratedEvidenceTest`; no timing policy, exclusion, suppression, or
waiver is used. At the Slice-1 checkpoint the ledgers were capability `84/31`, structural
`101/14`, MPSGraph `75/35/5`, custom `71/44/0`, and identity 23; schema 16 bound the exact supported
profile wire, while [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5, thirteen exports, and all other false rows remained unchanged. Slice 2
was outside that checkpoint.

### [Slice](../../../../glossary.md#slice) 2 checkpoint

The authorized ScatterAdd slice is implemented at wire 70. [Metal](../../../../glossary.md#term-metal) admits only [accelerator](../../../../glossary.md#term-accelerator),
no-[gradient](../../../../glossary.md#term-gradient), [canonical](../../../../glossary.md#term-canonical-form) [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [FLOAT32](../../../../glossary.md#term-float32) base/update/output with [axis](../../../../glossary.md#term-axis) zero, positive `D`/`U`
extents bounded to unsigned 32-bit bytes/[dispatch](../../../../glossary.md#term-dispatch), and a materialized canonical rank-one [INT32](../../../../glossary.md#data-type--datatype) or
[INT64](../../../../glossary.md#data-type--datatype) index [partition](../../../../glossary.md#partition) feed. The [compiler](../../../../glossary.md#term-compiler) places explicit [`CONTIGUOUS`](../../../../glossary.md#term-contiguous) between its generated
zero-base `EXPAND` and ScatterAdd. Java and native creation independently reject unsupported
profile/route, axis attributes, carriers, ranks, [layouts](../../../../glossary.md#layout), [aliases](../../../../glossary.md#aliasing-and-arena), feeds, gradients, and geometry.
The direct raw-native [matrix](../../../../glossary.md#term-matrix-meanings) now covers strict profile; [MPSGraph](../../../../glossary.md#term-mpsgraph) route; nonzero axis; missing,
wrong-kind, wrong-count, and invalid-value axis attributes; absent index feed; each independently
gradient-marked data, indices, updates, and output role; all six pairwise role-value-ID aliases;
wrong type, [rank](../../../../glossary.md#term-rank), and layout for every role; unequal index/update and base/output extents; zero
`D`; zero `U`; FLOAT32 `D` byte [overflow](../../../../glossary.md#term-overflow); INT32 `U` count and byte overflow; and INT64 `U` byte
overflow.

Native execution completes the entire [CPU](../../../../glossary.md#term-cpu) index scan into `[0,D)` before creating a [command buffer](../../../../glossary.md#term-command-buffer),
encoding any step, or mutating output. One [target](../../../../glossary.md#term-target)-[linear](../../../../glossary.md#term-linear-meanings) thread raw-loads its base exactly once,
scans every update ordinal in increasing source order, applies the sole safe [binary32](../../../../glossary.md#floating-point-precision-and-rounding) addition for
each matching [occurrence](../../../../glossary.md#term-occurrence), and reaches one final store. Duplicate targets are retained without a
uniqueness pass; unaddressed targets select the original raw base word; no atomic, retry, [race](../../../../glossary.md#term-race), or
[host](../../../../glossary.md#term-host) repair exists. The same admitted occurrence closes the existing compiler-generated rank-one
[Gather](../../../../glossary.md#gather) data [cotangent](../../../../glossary.md#cotangent) end to end without admitting gradient-bearing ScatterAdd, its own
cotangents, higher order, Gather-[Elements](../../../../glossary.md#term-element), or [Gather-ND](../../../../glossary.md#gather-nd). The reviewed representation-only amendment
also corrects CPU's generic-unit range [guard](../../../../glossary.md#term-guard) to use the trailing output [binding](../../../../glossary.md#term-binding) rather than a
broadcast [scalar](../../../../glossary.md#term-scalar) input; it changes no CPU arithmetic.

`Task0069Binary32.lean` now defines the total signed `exactRneGeneral` relation and
`GeneralBinary32RneContract` for arbitrary [finite](../../../../glossary.md#term-finite), [infinity](../../../../glossary.md#term-infinity), [NaN](../../../../glossary.md#term-nan), [subnormal](../../../../glossary.md#term-subnormal), overflow, [cancellation](../../../../glossary.md#term-cancellation),
and [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) operand pairs. `Task0069ReductionTree.lean` constructs occurrence ordinals from
source-list positions and proves strictly increasing filtered order, exact duplicate target/word
multiplicity, General-contract-constrained stable occurrence membership, raw unaddressed [identity](../../../../glossary.md#term-identity-meanings),
and [injective](../../../../glossary.md#term-injective) one-writer [ownership](../../../../glossary.md#term-ownership-meanings) in addition to the Slice-1 [L1](../../../../glossary.md#term-l1) theorems. The
source/compiler-site [certificate](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions) pins both
[kernels](../../../../glossary.md#kernel) and the native [preflight](../../../../glossary.md#term-preflight). The Xcode-27 compiled-[MSL](../../../../glossary.md#term-metal-shading-language-msl) runner pins source/[AIR](../../../../glossary.md#term-air)/metallib
[SHA-256](../../../../glossary.md#term-sha-256) values `7f10907eba7d03118d85091e7bd4e48197fea51defb908801bdad0c1e9f58caa`,
`5b4855918e2cf5af710a87fa6c9e303440cabab821b0a1ff01883a4dca53b5e0`, and
`7e90fd70a47087fb2ee55f269edd8a5c7ca0cad6a887bd3eb6de2f5a642bd491`.
AIR inspection finds one unflagged `fadd` per Task-0069 [kernel](../../../../glossary.md#kernel), no other floating arithmetic,
ScatterAdd's ordinal-zero equality-filter loop and raw-base unaddressed select, one static
[serialization](../../../../glossary.md#term-serialization) loop, and no atomics.

The warnings-as-errors native build, fixed-identifier [ad-hoc signing](../../../../glossary.md#term-ad-hoc-signing), canonical package verifier,
[Gradle](../../../../glossary.md#term-gradle) package ingestion/ZIP creation, fresh permission-preserving extraction, extracted verifier,
and byte comparison pass. The packaged [dylib](../../../../glossary.md#term-dylib) is `484944` bytes with SHA-256
`b48cf79cc26190243280a8c9b9cfac1d2fd8fa1dd33b875905fa231acff0a91f` and CDHash
`25d195103e26088d6577bba6427515fe712bbcdb`; the local ZIP is `142634` bytes with SHA-256
`59bc207a2dde3b4f40b42fa177a79bfc760c2a03696e7beeac98894a01d189ac`.
Complete Metal [backend](../../../../glossary.md#backend) tests pass all 260 tests. Focused Compiler [graph](../../../../glossary.md#graph) and CPU cotangent
regressions, Metal capability/partition [conformance](../../../../glossary.md#term-conformance), and public [Engine](../../../../glossary.md#term-engine) integration pass, including
canonical INT32/INT64 indices, duplicate source order, bit-exact unaddressed signed-zero/subnormal/
NaN copies, addressed normal/subnormal/infinity/NaN classes, positive and negative
error-before-mutation, [runtime](../../../../glossary.md#term-runtime) handle-alias rejection, the complete raw-native malformed image
matrix, and direct plus explicitly canonicalized generated-Gather execution on the packaged dylib.
The final focused raw-native method passes after the [producer](../../../../glossary.md#term-producer)-backed missing-index-feed [fixture](../../../../glossary.md#term-fixture),
and the proof/source-certificate/compiled-MSL runner passes unchanged.
The unfiltered serialized `./gradlew test --rerun-tasks --no-daemon --max-workers=1` passes all
76 actionable tasks in 3m. The final unfiltered serialized
`./gradlew build --rerun-tasks --no-daemon --max-workers=1` passes all 87 actionable tasks in
2m59s, including the completed remediation, with no exclusion, suppression, waiver, fallback, or
timing evidence. Initial independent review found the two residual raw-matrix/documentation P2
gaps and approved their remediation. A subsequent independent rereview found one remaining
documentation-only P2 in the active candidate-generator, partition-preparer, and accepted
[device](../../../../glossary.md#term-device)-contract [ADR](../../../../glossary.md#term-architecture-decision-record-adr) passages. Those passages and the remaining active Metal master/roadmap status
and identity claims are reconciled to the current [schema](../../../../glossary.md#term-schema)-16/version-24/custom-program contract;
historical checkpoint and rejection references remain intact. Final rereview approves with no
remaining P0/P1/P2 blockers. No production behavior changed in either documentation remediation.
Current ledgers are capability `85/30`, structural `101/14`, MPSGraph `75/35/5`, custom
`72/43/0`, schema 16, [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5, thirteen exports, and identity 24; identity 23 and every older value
[fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation). [`VARIANCE`](../../../../glossary.md#term-variance) remains false and Slice 3 is not authorized.

### [Slice](../../../../glossary.md#slice) 3 checkpoint

Fresh checkpoint authorization admitted only [accelerator](../../../../glossary.md#term-accelerator), no-[gradient](../../../../glossary.md#term-gradient) [FLOAT32](../../../../glossary.md#term-float32) [singleton](../../../../glossary.md#term-singleton)
[`VARIANCE`](../../../../glossary.md#term-variance): input `[1]`, [axis](../../../../glossary.md#term-axis) `[0]`, correction zero, and [canonical](../../../../glossary.md#term-canonical-form) [scalar](../../../../glossary.md#term-scalar) or retained `[1]`
output. Wire 112 selects the fixed `CUSTOM_PROGRAM` only for that exact [occurrence](../../../../glossary.md#term-occurrence). Java and native
creation reject its direct route, while the pre-existing non-domain [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)/correction-one
[MPSGraph](../../../../glossary.md#term-mpsgraph) structural [fixture](../../../../glossary.md#term-fixture) remains direct. No fallback, retry, timing, selector inference, [host](../../../../glossary.md#term-host)
repair, or additional gradient closure was added.

`Task0069Binary32.lean` now defines typed [binary32](../../../../glossary.md#floating-point-precision-and-rounding) DIV, SUB, and MUL primitive-result relations over
explicit signed exact values, one RNE [publication](../../../../glossary.md#term-publication), and per-site [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz) alternatives. Both
divisions carry the exact typed `+1` divisor; signed [infinity](../../../../glossary.md#term-infinity) survives the first division before
self-subtraction produces [NaN](../../../../glossary.md#term-nan); NaN propagates; every [finite](../../../../glossary.md#term-finite) singleton, including [signed zero](../../../../glossary.md#term-signed-zero) and
all [subnormal](../../../../glossary.md#term-subnormal) presentation choices, reaches [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero). The [trace](../../../../glossary.md#trace) instantiates the exact
literal DIV-SUB-MUL-DIV source rather than an asserted aggregate result. The reduction proof fixes
one dispatched writer, the shared canonical storage cell for both output forms, and exactly one
store. The source/[compiler](../../../../glossary.md#term-compiler) [certificate](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions) and compiled-[MSL](../../../../glossary.md#term-metal-shading-language-msl) audit pin the [kernel](../../../../glossary.md#kernel), native [preflight](../../../../glossary.md#term-preflight),
proofs, two unflagged divisions, one subtraction, one multiplication, no [FMA](../../../../glossary.md#term-fused-multiply-add-fma)/addition, and the sole
store.

The raw-native boundary [matrix](../../../../glossary.md#term-matrix-meanings) covers strict and exact direct-route rejection; wrong statistical
attribute kind, count, axis count/value, keep word, and correction; zero, wrong-[rank](../../../../glossary.md#term-rank), and
wrong-[extent](../../../../glossary.md#term-extent) inputs; scalar/retained output mismatch; wrong input/output [carrier](../../../../glossary.md#term-storage-carrier); both gradient
roles; input and output noncanonical [layout](../../../../glossary.md#layout)/offset/[stride](../../../../glossary.md#term-stride) cases; and input/output value-role [alias](../../../../glossary.md#aliasing-and-arena).
Native execution tests cover finite normal, both zeros, positive/negative subnormals, NaN, both
[infinities](../../../../glossary.md#nan-infinity-and-signed-zero), both output forms, and alias rejection before mutation. The public [Engine](../../../../glossary.md#term-engine) scenario
executes through the packaged [artifact](../../../../glossary.md#term-artifact). The current ledgers are capability `86/29`, structural
`101/14`, MPSGraph `75/35/5`, custom `73/42/0`, [schema](../../../../glossary.md#term-schema) 16, [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5, thirteen exports, and [identity](../../../../glossary.md#term-identity-meanings) 25;
identity 24 and every older value [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation).

The stabilized native source was rebuilt with warnings as errors, signed with the fixed ad-hoc
identifier, published through `package-local.sh`, independently verified, ingested by [Gradle](../../../../glossary.md#term-gradle), and
archived. A fresh permission-preserving ZIP extraction passed the same verifier and byte-compared
equal to the canonical package. The packaged [dylib](../../../../glossary.md#term-dylib) is `484992` bytes with [SHA-256](../../../../glossary.md#term-sha-256)
`a281ece41681845476f0eeed24a9da958d1a5beafae258e946a6e2b3944b4e63` and CDHash
`3467e8f6e751b0af47c88990772647561fed82b0`; the local ZIP is `144293` bytes with SHA-256
`edd68eebb8d4b66c8b5f13fa44021f070cbdea798e3faccf324867db63d1155e`. Focused native raw-ABI,
VARIANCE, preserved direct structural VARIANCE, and public Engine scenarios all ran against only
that fresh extracted dylib.

Final [validation](../../../../glossary.md#term-validation) passed the Task-0069 [Lean](../../../../glossary.md#proof-assistant-and-model-checking-tools)/source-certificate/Xcode [AIR](../../../../glossary.md#term-air) runner; the focused
`MetalVarianceNativeTest`, raw malformed-boundary, preserved direct structural VARIANCE, and public
Engine cases; the complete [Metal](../../../../glossary.md#term-metal) suite; focused compiler indexing/reduction/gradient suites; Metal
[backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance); architecture tests; and Metal [Javadoc](../../../../glossary.md#term-javadoc). Serialized repository-wide
`./gradlew test --rerun-tasks --no-daemon --max-workers=1` completed all 76 actionable tasks, then
serialized `./gradlew build --rerun-tasks --no-daemon --max-workers=1` completed all 87 actionable
tasks against the fresh ZIP-extracted dylib.

### Post-cutover review and remediation attempt

Independent current-source review found no separate [Slice](../../../../glossary.md#slice)-3 behavior defect in the inspected
provider, Java/native [validation](../../../../glossary.md#term-validation), fixed route, [kernel](../../../../glossary.md#kernel), [alias](../../../../glossary.md#aliasing-and-arena) rejection, or public [Engine](../../../../glossary.md#term-engine) paths, but
the Task-0069 [certificate](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions)'s foundation [hash](../../../../glossary.md#term-hash) no longer matches [schema](../../../../glossary.md#term-schema)-20 source. The focused
remediation attempt changed no files: [Lean](../../../../glossary.md#proof-assistant-and-model-checking-tools) proofs and current native package [authentication](../../../../glossary.md#term-authentication)
passed; `run-proof.sh` stopped on that stale hash. Compiling the current 84,603-byte fixed source
could not proceed because Xcode 27.0 build 27A266a reports `missing Metal Toolchain` and suggests
`xcodebuild -downloadComponent MetalToolchain`. The existing source/[AIR](../../../../glossary.md#term-air) hashes were not replaced
without compiled-code inspection. Task-0032's full build and real-[device](../../../../glossary.md#term-device) evidence were reused,
not rerun; `git diff --check` passed. Installing the Xcode component changed external machine
state and was separately approved. The component reports `installed` and [compiler](../../../../glossary.md#term-compiler) version
`32023.921`; explicit `TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.921.5` selection executes
it. The source/AIR audit resumed afterward and reached the accepted checkpoint below.

### Post-cutover evidence remediation checkpoint

The clean evidence worker audited current route, [preflight](../../../../glossary.md#term-preflight), index [validation](../../../../glossary.md#term-validation), [dispatch](../../../../glossary.md#term-dispatch), [buffer](../../../../glossary.md#term-buffer)
[ownership](../../../../glossary.md#term-ownership-meanings), source assembly, and safe math mode against the original [Slice](../../../../glossary.md#slice)-3 checkpoint before
rebinding source and compiled [hashes](../../../../glossary.md#term-hash). It changed only five files in `backends/metal/evidence/0069/`:
the source [certificate](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions), compiled-[MSL](../../../../glossary.md#term-metal-shading-language-msl) audit, runner, source verifier, and [AIR](../../../../glossary.md#term-air) verifier. The current
84,603-byte fixed source passed source [authentication](../../../../glossary.md#term-authentication); AIR checks retained the [L1](../../../../glossary.md#term-l1) and ScatterAdd
add sites and the [singleton](../../../../glossary.md#term-singleton) [VARIANCE](../../../../glossary.md#term-variance) DIV–SUB–MUL–DIV [dependency](../../../../glossary.md#term-dependency), safe flags, and one store.
`backends/metal/evidence/0069/run-proof.sh`, the [canonical](../../../../glossary.md#term-canonical-form) native `verify-package.sh`, and
`git diff --check` passed. The current packaged [dylib](../../../../glossary.md#term-dylib) remains byte-identical to its verified
build. No production source or Java/[Javadoc](../../../../glossary.md#term-javadoc) changed, so [Model](../../../../glossary.md#term-model)-0032 full/[device](../../../../glossary.md#term-device) results were reused
without a redundant build. Independent review found one closure defect: the verifier required
`review-pending`, which would reject the final `accepted` manifests. The worker changed both
[manifest](../../../../glossary.md#term-manifest) statuses and their bound [digest](../../../../glossary.md#term-digest), made the verifier require exact `accepted`, and reran
`run-proof.sh` and `git diff --check`; both passed. Targeted independent rereview returned
`APPROVE` with zero findings, independently reran the source-certificate verifier and diff check,
and reused the passing proof/package/Task-0032 full-device evidence. No production, Java, or
Javadoc changed; the five evidence files are the only non-[planning](../../../../glossary.md#term-planning) edits. There are no unresolved
Slice-3 findings.

Status: Complete
