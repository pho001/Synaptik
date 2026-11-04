# Task 0050: Final packaged [Metal](../../../../glossary.md#term-metal) repository verification

## Reader context

This is an artifact-chain receipt, not a new Metal implementation. Native build output becomes a local package, then a ZIP, then an extracted library supplied explicitly to tests. Matching checksums establish byte agreement between stages; the ad-hoc signature does not establish a trusted public publisher.

The extracted temporary directory was verification-owned storage and was removed, whereas input/output ownership belongs to execution. Hashes, command output, test totals and runner/release blockers below describe that exact revision. They are neither fresh verification of current source nor release approval.

- [Current evidence interpretation](../../../../backend-guide/metal-backend.md#evidence-composition) — Separates historical package/device receipts from current route qualification.

## Status

Complete

Evidence-only final program verification from exact clean Task 0049 completion revision
`2c6326a9663bea5aa99e407351d685919630bbcb`. No source, test, build logic, native source, [ABI](../../../../glossary.md#term-application-binary-interface-abi),
capability, route, [identity](../../../../glossary.md#term-identity-meanings), package contract, or [release](../../../../glossary.md#term-release) behavior changed.

## Change class

Verification record. This task executes the one reserved final full repository build and records
[artifact](../../../../glossary.md#term-artifact)/test evidence; it adds no implementation and requires no benchmark or new test.

## Goal

Prove end to end that the current native source builds into the existing Task 0045 package contract,
the Task 0046 archive preserves that package exactly, the extracted verified [dylib](../../../../glossary.md#term-dylib) loads through the
existing explicit test environment, and one complete repository build passes from the exact final-
approved Task 0049 revision.

## Dependencies and integration

- Depends on: Task 0045 Complete at approved remediation `26c6c911`; Task 0046 Complete at
  independently approved implementation `4aad1ab6`; Task 0048 Complete at independently approved
  implementation `89f9fbb9`; Task 0049 Complete at finalization `2c6326a9`
- Conflicts with: any concurrent source, test, Gradle, native, package, archive, Metal planning, or
  roadmap edit during verification<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [Metal glossary](../../../../glossary.md#term-metal); [planning glossary](../../../../glossary.md#term-planning))</span>
- Parallel group: None
- Exact verification revision: `2c6326a9663bea5aa99e407351d685919630bbcb`
- Integration order: after Task 0049 final cumulative review and completion
- Integration validation: current native build/sign/package/verifier; Task 0046 archive creation and
  extracted-package verification; exactly one full <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`./gradlew build --rerun-tasks`; JUnit XML and
  Markdown/status/diff/clean checks<span data-glossary-navigation="true"> ([JUnit glossary](../../../../glossary.md#term-junit))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): Task 0050 evidence finalizer

## Boundaries

This task must not:

- change Java, Objective-C, Gradle, shell, workflow, test, benchmark, ABI, schema, route, capability,
  trace, cache, or tuning code;<span data-glossary-navigation="true"> ([Objective-C glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary); [Gradle glossary](../../../../glossary.md#term-gradle); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [schema glossary](../../../../glossary.md#term-schema); [trace glossary](../../../../glossary.md#trace); [cache glossary](../../../../glossary.md#term-cache))</span>
- [run](../../../../glossary.md#term-run) a benchmark [matrix](../../../../glossary.md#term-review-matrix) or baseline;
- run a second full repository build after the successful reserved build;
- claim Developer ID, [notarization](../../../../glossary.md#term-notarization), [authentication](../../../../glossary.md#term-authentication), [publication](../../../../glossary.md#term-publication), [provenance](../../../../glossary.md#provenance), or [release](../../../../glossary.md#term-release); or
- push a [commit](../../../../glossary.md#term-commit).

The local package and ZIP remain ignored, unauthenticated development [artifacts](../../../../glossary.md#term-artifact). The extracted
temporary directory is caller-owned verification state and is removed after evidence is recorded.

## Native package and archive procedure

From clean revision `2c6326a9663bea5aa99e407351d685919630bbcb`:

1. `native/metal-macos-arm64/build.sh` built the current [dylib](../../../../glossary.md#term-dylib).
2. `/usr/bin/codesign --force --sign - --timestamp=none` replaced the incidental signature with
   fixed identifier `io.github.pho001.synaptik.metal.foundation`.
3. `package-local.sh` staged, independently verified, and published the [canonical](../../../../glossary.md#term-canonical-form)
   `build/package-v1/macos-arm64` package.
4. `verify-package.sh` independently reverified that published package.
5. `:backends:metal:metalNativeLocalZip --rerun-tasks` received the canonical package through exact
   absolute property `synaptikMetalNativePackage`; its verifier and ZIP tasks both executed.
6. `/usr/bin/ditto -x -k` extracted the ZIP permission-preservingly into fresh directory
   `/tmp/synaptik-metal-final.ixeG5R`.
7. The unchanged `verify-package.sh` passed on extracted `macos-arm64`.

Both package verifications enforce exactly three regular non-symlink files; canonical [schema](../../../../glossary.md#term-schema)-1
[manifest](../../../../glossary.md#term-manifest)/[checksum](../../../../glossary.md#term-checksum) content; [arm64](../../../../glossary.md#native-packaging-and-platform-vocabulary) [Mach-O](../../../../glossary.md#term-mach-o) 64-bit `DYLIB`; macOS minimum `26.0`; [install name](../../../../glossary.md#term-install-name)
`@rpath/libsynaptik_metal_foundation.dylib`; no `LC_RPATH` or legacy minimum-version command; only
normalized system dependencies including exactly Foundation, [Metal](../../../../glossary.md#term-metal), and
MetalPerformanceShadersGraph; the exact thirteen [ABI](../../../../glossary.md#term-application-binary-interface-abi)-v4 exports; strict ad-hoc [signature](../../../../glossary.md#term-signature) [validation](../../../../glossary.md#term-validation);
identifier `io.github.pho001.synaptik.metal.foundation`; and no Team ID. The manifest retains native
ABI 4 and [node](../../../../glossary.md#node) schema 11.

## [Artifact](../../../../glossary.md#term-artifact) [identities](../../../../glossary.md#term-identity-meanings)

[Canonical](../../../../glossary.md#term-canonical-form) package:

- [dylib](../../../../glossary.md#term-dylib) size: `154032` bytes
- dylib [SHA-256](../../../../glossary.md#term-sha-256): `20e44270123412fd929eb2bb71c961f4471aa7959640c33775fd3d91a74b4c46`
- [manifest](../../../../glossary.md#term-manifest) SHA-256: `9e4a1334044d6b0cb3275f87aaed972b13f3875dedb8fb3935e716b3cea65452`
- `SHA256SUMS` SHA-256:
  `ba99f9e2cb3b249df5e668ffd0cf4c52e84e7df5b8038d8dc9c8e0cf4ca89c92`

Local archive:

- path: `backends/metal/build/distributions/synaptik-metal-macos-arm64-local.zip`
- size: `37426` bytes
- SHA-256: `82e569f96772addeaf661b6d9d6fb2651926d727d7dd16ab3ae299d9f6dd5655`
- uncompressed bytes: `154725`
- compressed bytes: `36890`
- all timestamps: fixed `1980-02-01 00:00:00`

Exact ordered entries:

```text
0755  macos-arm64/
0755  macos-arm64/libsynaptik_metal_foundation.dylib  154032 bytes
0644  macos-arm64/manifest.json                          512 bytes
0644  macos-arm64/SHA256SUMS                             181 bytes
```

Every extracted member is byte-identical to its canonical package member. The displayed extracted
[signature](../../../../glossary.md#term-signature) is ad-hoc with the fixed identifier, no Team ID, and CDHash
`28222dc3e112879cff28280637a06edb68377a8a`.

## Sole final full build

The extracted absolute [dylib](../../../../glossary.md#term-dylib) path was:

```text
/tmp/synaptik-metal-final.ixeG5R/macos-arm64/libsynaptik_metal_foundation.dylib
```

Exactly one final full repository build ran with that value as
`SYNAPTIK_METAL_TEST_LIBRARY`:

```bash
SYNAPTIK_METAL_TEST_LIBRARY=/tmp/synaptik-metal-final.ixeG5R/macos-arm64/libsynaptik_metal_foundation.dylib \
  ./gradlew build --rerun-tasks
```

It completed `BUILD SUCCESSFUL` in 3 minutes 1 second with exactly 87 actionable tasks, all 87
executed. No failure required a focused rerun, no code changed, and no replacement full build ran.
Every repository test task was rerun; tests that honor the [Metal](../../../../glossary.md#term-metal) environment used the absolute
dylib extracted from the verified archive.

## [JUnit](../../../../glossary.md#term-junit) XML evidence

All `TEST-*.xml` reports produced by the final build were parsed rather than inferred from [Gradle](../../../../glossary.md#term-gradle)'s
exit code:

- repository total: 539 suites, 3,553 tests, 0 failures, 0 errors, 29 skips;
- [Metal](../../../../glossary.md#term-metal) [backend](../../../../glossary.md#backend): all 19 suites, 130 tests, 0 failures, 0 errors, 0 skips;
- explicit public [Engine](../../../../glossary.md#term-engine) Metal integration:
  `EngineExplicitCompositionMetalIntegrationTest`, 1 suite, 15 tests, 0 failures, 0 errors, 0 skips;
- real-Metal public `prepareTuned` coverage within that integration suite: 2 tests, 0 failures,
  0 errors, 0 skips:
  - `realMetalLifecycleMixedOwnerTransfersAndCpuTuning(Path)`;
  - `publicMetalRouteTuningPreparesAndRunsSingletonNeg(Path)`.

The 29 repository skips are outside the Metal backend and the explicit Metal integration suite:
28 are in `backends/cpu`, and one is in another integration suite. No relevant Metal test was
skipped.

## Result

The exact final-approved Task 0049 revision builds successfully end to end against the current
freshly built, fixed-identifier ad-hoc-signed, canonically packaged, archived, permission-preserving
extracted, and independently reverified [Metal](../../../../glossary.md#term-metal) [dylib](../../../../glossary.md#term-dylib). Package and archive [identities](../../../../glossary.md#term-identity-meanings), [signature](../../../../glossary.md#term-signature),
[ABI](../../../../glossary.md#term-application-binary-interface-abi)/export, deployment, install-name, no-[rpath](../../../../glossary.md#term-rpath), mode, and [checksum](../../../../glossary.md#term-checksum) contracts all pass unchanged.
All relevant real packaged/extracted Metal tests execute without skip or failure.

No benchmark or baseline ran. No implementation changed. No second full build ran. The source tree
was clean after build. No Metal task is [Ready](../../../../glossary.md#term-ready).

## Remaining external blockers

- Task 0047 remains Blocked until an eligible organization/enterprise plan, billing/spending, and
  actual access to the required real GPU-capable <span data-glossary-navigation="true"> ([GPU glossary](../../../../glossary.md#term-gpu))</span>`macos-26-xlarge` M2 [hosted runner](../../../../glossary.md#continuous-integration-and-hosted-runners) exist.
- Developer ID signing, secure keychain/credential ownership, notarization, authenticated/versioned
  distribution, provenance, publication coordinates, project-wide license/notice policy, rollback,
  and public release remain separately blocked on explicit release planning and external inputs.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [notarization glossary](../../../../glossary.md#term-notarization); [distribution glossary](../../../../glossary.md#term-distribution); [provenance glossary](../../../../glossary.md#provenance); [publication glossary](../../../../glossary.md#term-publication); [rollback glossary](../../../../glossary.md#term-rollback); [release glossary](../../../../glossary.md#term-release); [planning glossary](../../../../glossary.md#term-planning))</span>

All [operation](../../../../glossary.md#operation)-specific semantic/evidence blockers retain their recorded status; this verification
does not reopen or weaken them.
