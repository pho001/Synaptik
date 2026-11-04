# Task 0045: Verified Local macOS [arm64](../../../../glossary.md#native-packaging-and-platform-vocabulary) Native Package

## Status

Complete

## Change class

Class C — this task leaves the native [ABI](../../../../glossary.md#term-application-binary-interface-abi) and Java loader unchanged, but establishes a durable
[Mach-O](../../../../glossary.md#term-mach-o) deployment, code-[signature](../../../../glossary.md#term-signature), and package contract at a security-sensitive native boundary.
Use one clean implementation context and an independent targeted review before completion.

## Goal

Produce one ignored local package containing the exact macOS [arm64](../../../../glossary.md#native-packaging-and-platform-vocabulary) [Metal](../../../../glossary.md#term-metal) [dylib](../../../../glossary.md#term-dylib) after explicit
[ad-hoc signing](../../../../glossary.md#term-ad-hoc-signing), a [canonical](../../../../glossary.md#term-canonical-form) [schema](../../../../glossary.md#term-schema)-1 [manifest](../../../../glossary.md#term-manifest), and [SHA-256](../../../../glossary.md#term-sha-256) checksums. Independently verify the
final packaged bytes, [Mach-O](../../../../glossary.md#term-mach-o) contract, exports, system dependencies, and ad-hoc [signature](../../../../glossary.md#term-signature).

This is a verified local development package. It is not Developer-ID-signed, notarized,
authenticated, published, or a public [release](../../../../glossary.md#term-release).

## [Scope](../../../../glossary.md#term-scope)

- Make the native build [atomic](../../../../glossary.md#resource-transaction-and-rollback) and pin [arm64](../../../../glossary.md#native-packaging-and-platform-vocabulary), macOS `26.0`, the exact [install name](../../../../glossary.md#term-install-name), and no [rpath](../../../../glossary.md#term-rpath).
- Add an ad-hoc-only packager for an already explicitly signed [dylib](../../../../glossary.md#term-dylib).
- Add an independent [fail-closed](../../../../glossary.md#term-fail-closed) package verifier.
- Document the local workflow, [integrity](../../../../glossary.md#term-integrity) boundary, and [release](../../../../glossary.md#term-release)-only boundary.
- Preserve [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4, [node](../../../../glossary.md#node) [schema](../../../../glossary.md#term-schema) 11, and the exact thirteen exports.

## Non-goals

- Developer ID mode, [certificate](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions)/Team-ID placeholders, credentials, [notarization](../../../../glossary.md#term-notarization) or a no-op for it
- archive, [publication](../../../../glossary.md#term-publication), [provenance](../../../../glossary.md#provenance), versioning, [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), [Gradle](../../../../glossary.md#term-gradle) [distribution](../../../../glossary.md#term-distribution), or [CI](../../../../glossary.md#term-continuous-integration-ci) workflow
- Java loading, discovery, extraction, caching, configuration, [API](../../../../glossary.md#term-application-programming-interface-api), test, or [dependency](../../../../glossary.md#term-dependency) changes
- native [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) source, [ABI](../../../../glossary.md#term-application-binary-interface-abi), [schema](../../../../glossary.md#term-schema), capability, numerical, framework, or route changes
- Intel macOS, a lower deployment [target](../../../../glossary.md#term-target), or a reproducible cross-[toolchain](../../../../glossary.md#term-toolchain) [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) claim

## Contracts

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md) headings `Authority, incorporation, and
  precedence`, `Repository layout`, `Core invariants`, and `Testing requirements` — architecture
  authority, repository placement, and proportionate validation remain unchanged.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- [`docs/architecture/contracts/backend-execution.md`](../../../../architecture/contracts/backend-execution.md)
  headings `Concrete backend modules` and `Metal backend` — Metal owns its native bridge; Java
  continues to load one caller-selected absolute path and performs no discovery.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [native bridge glossary](../../../../glossary.md#term-native-bridge))</span>
- [`docs/developer-guide/release-process.md`](../../../../developer-guide/release-process.md) — the
  repository has no public release process; local verification must not be described as release,
  publication, provenance, compatibility, or publisher authentication.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [provenance glossary](../../../../glossary.md#provenance); [authentication glossary](../../../../glossary.md#term-authentication))</span>

If implementation needs another [ownership](../../../../glossary.md#term-ownership-meanings) rule, public workflow, [release](../../../../glossary.md#term-release) mode, or [architecture
contract](../../../../glossary.md#architecture-contract), stop and return this task to [planning](../../../../glossary.md#term-planning).

## Dependencies and integration

- Depends on: Metal 0025 Complete for current ABI v4/schema 11; Metal 0044 Complete as the current
  serialized frontier<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [schema glossary](../../../../glossary.md#term-schema))</span>
- Conflicts with: native Metal build/package/README edits; Metal master/roadmap edits; any Gradle
  distribution, native discovery, release-signing, or publication work<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [release glossary](../../../../glossary.md#term-release); [publication glossary](../../../../glossary.md#term-publication))</span>
- Parallel group: None
- Common base revision: `7dc4cfec2a051b5ca9c0734f91d0fdc96ad7405b`
- Integration order: before any native [distribution](../../../../glossary.md#term-distribution)-configuration task
- Integration [validation](../../../../glossary.md#term-validation): [Metal](../../../../glossary.md#term-metal) native package [checkpoint](../../../../glossary.md#term-checkpoint)
- Shared-document integration [owner](../../../../glossary.md#term-owner): Main planner

## Files and responsibilities

- `native/metal-macos-arm64/build.sh` — [atomic](../../../../glossary.md#resource-transaction-and-rollback) [arm64](../../../../glossary.md#native-packaging-and-platform-vocabulary)/macOS-26 [dylib](../../../../glossary.md#term-dylib) build and [install name](../../../../glossary.md#term-install-name)
- `native/metal-macos-arm64/package-local.sh` — verified ad-hoc local package staging/[publication](../../../../glossary.md#term-publication)
- `native/metal-macos-arm64/verify-package.sh` — independent exact package/[Mach-O](../../../../glossary.md#term-mach-o)/[signature](../../../../glossary.md#term-signature) checks
- `native/metal-macos-arm64/README.md` — build, signing, package, verification, and boundaries
- this task, `../master-plan.md`, and `../../../roadmap.md` — authorization and status

Exactly these seven repository paths may change. All generated files remain below the already
ignored `native/metal-macos-arm64/build/` directory.

## Package contract

The fixed package directory is `build/package-v1/macos-arm64/` and contains exactly three regular,
non-symlink files:

1. `libsynaptik_metal_foundation.dylib`, mode `0755`;
2. `manifest.json`, mode `0644`; and
3. `SHA256SUMS`, mode `0644`.

`manifest.json` is [canonical](../../../../glossary.md#term-canonical-form) [UTF-8](../../../../glossary.md#term-utf-8) [JSON](../../../../glossary.md#term-json) with LF, fixed key order, no insignificant whitespace, and
these exact [schema](../../../../glossary.md#term-schema)-1 facts: relative [artifact](../../../../glossary.md#term-artifact) filename, byte size and [SHA-256](../../../../glossary.md#term-sha-256); `macos`; [`arm64`](../../../../glossary.md#native-packaging-and-platform-vocabulary);
minimum macOS `26.0`; [install name](../../../../glossary.md#term-install-name) `@rpath/libsynaptik_metal_foundation.dylib`; empty rpaths; [ABI](../../../../glossary.md#term-application-binary-interface-abi)
`4`; [node](../../../../glossary.md#node) schema `11`; ordered Foundation, [Metal](../../../../glossary.md#term-metal), and MetalPerformanceShadersGraph frameworks; and
ad-hoc [signature](../../../../glossary.md#term-signature) identifier `io.github.pho001.synaptik.metal.foundation`. It contains no time,
[host](../../../../glossary.md#term-host), absolute path, [source revision](../../../../glossary.md#term-source-revision), product version, [SDK](../../../../glossary.md#term-sdk) version, Team ID, [notarization](../../../../glossary.md#term-notarization),
[provenance](../../../../glossary.md#provenance), or [release](../../../../glossary.md#term-release) field.

`SHA256SUMS` contains exactly the locale-C, lexically ordered final-[dylib](../../../../glossary.md#term-dylib) and [manifest](../../../../glossary.md#term-manifest) entries. The
manifest [hashes](../../../../glossary.md#term-hash) the final signed dylib; the [checksum](../../../../glossary.md#term-checksum) file hashes both without a circular self-[hash](../../../../glossary.md#term-hash).
For the same exact signed input, repeated packaging emits byte-identical manifest and checksum
files.

## Verification contract

The verifier derives facts from the final [dylib](../../../../glossary.md#term-dylib) rather than trusting the [manifest](../../../../glossary.md#term-manifest). It fails for a
symlink package root or member; any missing, extra, or non-regular member; wrong mode, manifest,
[checksum](../../../../glossary.md#term-checksum), architecture, file type, minimum OS, [install name](../../../../glossary.md#term-install-name), [rpath](../../../../glossary.md#term-rpath), [dependency](../../../../glossary.md#term-dependency), export, [signature](../../../../glossary.md#term-signature),
or identifier; and any tool/parsing ambiguity.

The final dylib must be one [arm64](../../../../glossary.md#native-packaging-and-platform-vocabulary) [Mach-O](../../../../glossary.md#term-mach-o) 64-bit `DYLIB`, minimum macOS `26.0`, with exact
`LC_ID_DYLIB` `@rpath/libsynaptik_metal_foundation.dylib` and no `LC_RPATH`. It directly links the
three required Apple frameworks. Every dependency is an Apple system framework or `/usr/lib`
path; no tokenized, workspace, Homebrew, or MacPorts dependency is allowed. Its globally defined
exports, after removing Mach-O's leading underscore, equal the documented thirteen-[symbol](../../../../glossary.md#term-symbol) [ABI](../../../../glossary.md#term-application-binary-interface-abi).

`codesign --verify --strict` must pass, and displayed [metadata](../../../../glossary.md#term-metadata) must report `Signature=adhoc`, the
fixed identifier, and `TeamIdentifier=not set`. Do not use `--deep` or claim Gatekeeper acceptance.
Self-contained checksums detect corruption but do not authenticate a hostile replacement. An
ad-hoc signature protects code [integrity](../../../../glossary.md#term-integrity) but supplies no publisher [identity](../../../../glossary.md#term-identity-meanings) or Apple trust.

## Acceptance criteria

- `build.sh` compiles to a temporary file and atomically replaces the output only after success.
- The exact Mach-O, dependency, export, min-OS, install-name, no-rpath, package, and signature
  contracts above pass without changing native source, ABI, schema, Java, Gradle, or loader code.<span data-glossary-navigation="true"> ([Mach-O glossary](../../../../glossary.md#term-mach-o); [dependency glossary](../../../../glossary.md#term-dependency); [rpath glossary](../../../../glossary.md#term-rpath); [signature glossary](../../../../glossary.md#term-signature); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [schema glossary](../../../../glossary.md#term-schema); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- `package-local.sh` accepts only the explicitly ad-hoc-signed input, copies exact bytes into a
  private staging directory, verifies the staged copy, generates metadata after signing, invokes
  the independent verifier, and atomically publishes the package without a stale partial result.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- Packaging the same signed input twice yields identical manifest and checksum bytes; a copied or
  moved package still verifies.<span data-glossary-navigation="true"> ([manifest glossary](../../../../glossary.md#term-manifest); [checksum glossary](../../../../glossary.md#term-checksum))</span>
- Targeted corrupt-member, extra-member, symlink-member, checksum/manifest, missing-signature, and
  wrong-signature-identifier cases fail nonzero without modifying the rejected package.
- Existing real-device Metal tests and explicit Engine composition integration pass against the
  packaged dylib path, not the pre-package build output.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [Metal glossary](../../../../glossary.md#term-metal); [Engine glossary](../../../../glossary.md#term-engine); [dylib glossary](../../../../glossary.md#term-dylib))</span>
- The README and planning consistently distinguish corruption detection, ad-hoc integrity,
  Developer ID identity, notarization, publication, and release.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [integrity glossary](../../../../glossary.md#term-integrity); [identity glossary](../../../../glossary.md#term-identity-meanings); [notarization glossary](../../../../glossary.md#term-notarization); [publication glossary](../../../../glossary.md#term-publication); [release glossary](../../../../glossary.md#term-release))</span>
- No archive, credential, Developer ID path, notarization placeholder, release claim, generated
  tracked file, or path outside the seven-file scope exists.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker validation:

```bash
./native/metal-macos-arm64/build.sh
/usr/bin/codesign --force --sign - --timestamp=none \
  --identifier io.github.pho001.synaptik.metal.foundation \
  native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/package-local.sh \
  native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/verify-package.sh \
  native/metal-macos-arm64/build/package-v1/macos-arm64
```

Also inspect with `file`, `xcrun lipo -archs`, `xcrun vtool -show-build`, `otool -hv/-D/-l/-L`,
`nm -gUj`, `codesign --verify --strict`, and `codesign --display --verbose=4`; verify
`SHA256SUMS`; prove repeat [metadata](../../../../glossary.md#term-metadata) [identity](../../../../glossary.md#term-identity-meanings) and copy/move success; and [run](../../../../glossary.md#term-run) the targeted negative
cases above.

Run the existing real-[device](../../../../glossary.md#term-device) commands with `SYNAPTIK_METAL_TEST_LIBRARY` set to the packaged [dylib](../../../../glossary.md#term-dylib):

```bash
./gradlew :backends:metal:test --tests '*Metal*' --rerun-tasks
./gradlew :testing:integration-tests:test \
  --tests '*EngineExplicitCompositionMetalIntegrationTest' --rerun-tasks
```

Do not run a full repository build. Validate changed Markdown, exact path/status [scope](../../../../glossary.md#term-scope), shell syntax,
and `git diff --check`.

## Follow-up

A separate [distribution](../../../../glossary.md#term-distribution)-configuration task may consume only this verified package boundary. It
owns reproducible archive [metadata](../../../../glossary.md#term-metadata), version/name/legal-notice policy, explicit [Gradle](../../../../glossary.md#term-gradle) input and
copying, and any install/extraction [lifecycle](../../../../glossary.md#lifecycle) while preserving the absolute-path loader contract.
Developer ID signing and [notarization](../../../../glossary.md#term-notarization) remain externally blocked on a real authorized [identity](../../../../glossary.md#term-identity-meanings),
secure keychain/[CI](../../../../glossary.md#term-continuous-integration-ci) secret handling, notary credentials, and an accepted [release](../../../../glossary.md#term-release) container/process.

## Documentation and review impact

The native README and [planning](../../../../glossary.md#term-planning) change a durable developer workflow and security terminology.
Independent Class C review must inspect quoting, staging/[atomicity](../../../../glossary.md#term-atomicity), [fail-closed](../../../../glossary.md#term-fail-closed) parsing,
path/symlink handling, [signature](../../../../glossary.md#term-signature) ordering, [checksum](../../../../glossary.md#term-checksum) coverage, [Mach-O](../../../../glossary.md#term-mach-o) assertions, and absence of
[identity](../../../../glossary.md#term-identity-meanings), [notarization](../../../../glossary.md#term-notarization), [distribution](../../../../glossary.md#term-distribution), or [release](../../../../glossary.md#term-release) claims.

## Result

Implemented at `07174014` from [planning](../../../../glossary.md#term-planning) [commit](../../../../glossary.md#term-commit) `86553466` and clean base
`7dc4cfec2a051b5ca9c0734f91d0fdc96ad7405b`. The first independent Class C review found three P1
fail-open/containment defects: a trailing-separator package-root symlink bypass, incomplete raw
`otool -L` parsing, and native build/package destinations that could traverse symlinks or accept
non-regular outputs.

Remediation rejects redundant path separators/components before the non-following package-root
symlink check; validates every complete raw [dependency](../../../../glossary.md#term-dependency) record, its version suffix, and the
[Mach-O](../../../../glossary.md#term-mach-o)-record count before extracting paths; and confines build, staging, backup, output, and
cleanup paths to real directories and regular non-symlink files. Native source, [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4, [schema](../../../../glossary.md#term-schema) 11,
Java, [Gradle](../../../../glossary.md#term-gradle), and loader behavior remain unchanged.

Shell syntax; a fresh build, explicit [ad-hoc signing](../../../../glossary.md#term-ad-hoc-signing), two byte-identical package publications, and
copy/move verification passed. The original eight corrupt/extra/root-or-member-symlink/[manifest](../../../../glossary.md#term-manifest)/
[checksum](../../../../glossary.md#term-checksum)/missing-[signature](../../../../glossary.md#term-signature)/wrong-identifier cases still fail. Exact symlink-root spellings with
plain, trailing `/`, terminal `/.`, and repeated `/` forms fail; a re-signed malformed Foundation
load command with an embedded newline and hostile `/usr/lib` record fails after regenerated
manifest/checksums. External build/staging symlinks, non-directory build paths, and output symlink,
directory, symlink-to-directory, and FIFO cases fail without an outside write; the prior valid
build tree remains byte- and mode-identical. The regenerated packaged [dylib](../../../../glossary.md#term-dylib) passed 126 [Metal](../../../../glossary.md#term-metal) tests
in 19 suites and 15 [Engine](../../../../glossary.md#term-engine) integration tests with zero skips, failures, or errors. Markdown,
seven-path [scope](../../../../glossary.md#term-scope), shell syntax, and diff checks passed; no full repository build ran.

The [artifact](../../../../glossary.md#term-artifact) remains local and unauthenticated; every [release](../../../../glossary.md#term-release)-only exclusion remains. The same
independent Class C reviewer approved remediation commit `26c6c911` with zero remaining findings.
Status: Complete.
