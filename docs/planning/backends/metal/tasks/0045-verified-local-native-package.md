# Task 0045: Verified Local macOS arm64 Native Package

## Status

Review needed

## Change class

Class C — this task leaves the native ABI and Java loader unchanged, but establishes a durable
Mach-O deployment, code-signature, and package contract at a security-sensitive native boundary.
Use one clean implementation context and an independent targeted review before completion.

## Goal

Produce one ignored local package containing the exact macOS arm64 Metal dylib after explicit
ad-hoc signing, a canonical schema-1 manifest, and SHA-256 checksums. Independently verify the
final packaged bytes, Mach-O contract, exports, system dependencies, and ad-hoc signature.

This is a verified local development package. It is not Developer-ID-signed, notarized,
authenticated, published, or a public release.

## Scope

- Make the native build atomic and pin arm64, macOS `26.0`, the exact install name, and no rpath.
- Add an ad-hoc-only packager for an already explicitly signed dylib.
- Add an independent fail-closed package verifier.
- Document the local workflow, integrity boundary, and release-only boundary.
- Preserve ABI v4, node schema 11, and the exact thirteen exports.

## Non-goals

- Developer ID mode, certificate/Team-ID placeholders, credentials, notarization or a no-op for it
- archive, publication, provenance, versioning, coordinates, Gradle distribution, or CI workflow
- Java loading, discovery, extraction, caching, configuration, API, test, or dependency changes
- native Objective-C source, ABI, schema, capability, numerical, framework, or route changes
- Intel macOS, a lower deployment target, or a reproducible cross-toolchain binary claim

## Contracts

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md) headings `Authority, incorporation, and
  precedence`, `Repository layout`, `Core invariants`, and `Testing requirements` — architecture
  authority, repository placement, and proportionate validation remain unchanged.
- [`docs/architecture/contracts/backend-execution.md`](../../../../architecture/contracts/backend-execution.md)
  headings `Concrete backend modules` and `Metal backend` — Metal owns its native bridge; Java
  continues to load one caller-selected absolute path and performs no discovery.
- [`docs/developer-guide/release-process.md`](../../../../developer-guide/release-process.md) — the
  repository has no public release process; local verification must not be described as release,
  publication, provenance, compatibility, or publisher authentication.

If implementation needs another ownership rule, public workflow, release mode, or architecture
contract, stop and return this task to planning.

## Dependencies and integration

- Depends on: Metal 0025 Complete for current ABI v4/schema 11; Metal 0044 Complete as the current
  serialized frontier
- Conflicts with: native Metal build/package/README edits; Metal master/roadmap edits; any Gradle
  distribution, native discovery, release-signing, or publication work
- Parallel group: None
- Common base revision: `7dc4cfec2a051b5ca9c0734f91d0fdc96ad7405b`
- Integration order: before any native distribution-configuration task
- Integration validation: Metal native package checkpoint
- Shared-document integration owner: Main planner

## Files and responsibilities

- `native/metal-macos-arm64/build.sh` — atomic arm64/macOS-26 dylib build and install name
- `native/metal-macos-arm64/package-local.sh` — verified ad-hoc local package staging/publication
- `native/metal-macos-arm64/verify-package.sh` — independent exact package/Mach-O/signature checks
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

`manifest.json` is canonical UTF-8 JSON with LF, fixed key order, no insignificant whitespace, and
these exact schema-1 facts: relative artifact filename, byte size and SHA-256; `macos`; `arm64`;
minimum macOS `26.0`; install name `@rpath/libsynaptik_metal_foundation.dylib`; empty rpaths; ABI
`4`; node schema `11`; ordered Foundation, Metal, and MetalPerformanceShadersGraph frameworks; and
ad-hoc signature identifier `io.github.pho001.synaptik.metal.foundation`. It contains no time,
host, absolute path, source revision, product version, SDK version, Team ID, notarization,
provenance, or release field.

`SHA256SUMS` contains exactly the locale-C, lexically ordered final-dylib and manifest entries. The
manifest hashes the final signed dylib; the checksum file hashes both without a circular self-hash.
For the same exact signed input, repeated packaging emits byte-identical manifest and checksum
files.

## Verification contract

The verifier derives facts from the final dylib rather than trusting the manifest. It fails for a
symlink package root or member; any missing, extra, or non-regular member; wrong mode, manifest,
checksum, architecture, file type, minimum OS, install name, rpath, dependency, export, signature,
or identifier; and any tool/parsing ambiguity.

The final dylib must be one arm64 Mach-O 64-bit `DYLIB`, minimum macOS `26.0`, with exact
`LC_ID_DYLIB` `@rpath/libsynaptik_metal_foundation.dylib` and no `LC_RPATH`. It directly links the
three required Apple frameworks. Every dependency is an Apple system framework or `/usr/lib`
path; no tokenized, workspace, Homebrew, or MacPorts dependency is allowed. Its globally defined
exports, after removing Mach-O's leading underscore, equal the documented thirteen-symbol ABI.

`codesign --verify --strict` must pass, and displayed metadata must report `Signature=adhoc`, the
fixed identifier, and `TeamIdentifier=not set`. Do not use `--deep` or claim Gatekeeper acceptance.
Self-contained checksums detect corruption but do not authenticate a hostile replacement. An
ad-hoc signature protects code integrity but supplies no publisher identity or Apple trust.

## Acceptance criteria

- `build.sh` compiles to a temporary file and atomically replaces the output only after success.
- The exact Mach-O, dependency, export, min-OS, install-name, no-rpath, package, and signature
  contracts above pass without changing native source, ABI, schema, Java, Gradle, or loader code.
- `package-local.sh` accepts only the explicitly ad-hoc-signed input, copies exact bytes into a
  private staging directory, verifies the staged copy, generates metadata after signing, invokes
  the independent verifier, and atomically publishes the package without a stale partial result.
- Packaging the same signed input twice yields identical manifest and checksum bytes; a copied or
  moved package still verifies.
- Targeted corrupt-member, extra-member, symlink-member, checksum/manifest, missing-signature, and
  wrong-signature-identifier cases fail nonzero without modifying the rejected package.
- Existing real-device Metal tests and explicit Engine composition integration pass against the
  packaged dylib path, not the pre-package build output.
- The README and planning consistently distinguish corruption detection, ad-hoc integrity,
  Developer ID identity, notarization, publication, and release.
- No archive, credential, Developer ID path, notarization placeholder, release claim, generated
  tracked file, or path outside the seven-file scope exists.

## Validation

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
`SHA256SUMS`; prove repeat metadata identity and copy/move success; and run the targeted negative
cases above.

Run the existing real-device commands with `SYNAPTIK_METAL_TEST_LIBRARY` set to the packaged dylib:

```bash
./gradlew :backends:metal:test --tests '*Metal*' --rerun-tasks
./gradlew :testing:integration-tests:test \
  --tests '*EngineExplicitCompositionMetalIntegrationTest' --rerun-tasks
```

Do not run a full repository build. Validate changed Markdown, exact path/status scope, shell syntax,
and `git diff --check`.

## Follow-up

A separate distribution-configuration task may consume only this verified package boundary. It
owns reproducible archive metadata, version/name/legal-notice policy, explicit Gradle input and
copying, and any install/extraction lifecycle while preserving the absolute-path loader contract.
Developer ID signing and notarization remain externally blocked on a real authorized identity,
secure keychain/CI secret handling, notary credentials, and an accepted release container/process.

## Documentation and review impact

The native README and planning change a durable developer workflow and security terminology.
Independent Class C review must inspect quoting, staging/atomicity, fail-closed parsing,
path/symlink handling, signature ordering, checksum coverage, Mach-O assertions, and absence of
identity, notarization, distribution, or release claims.

## Result

Implemented at `07174014` from planning commit `86553466` and clean base
`7dc4cfec2a051b5ca9c0734f91d0fdc96ad7405b`. The first independent Class C review found three P1
fail-open/containment defects: a trailing-separator package-root symlink bypass, incomplete raw
`otool -L` parsing, and native build/package destinations that could traverse symlinks or accept
non-regular outputs.

Remediation rejects redundant path separators/components before the non-following package-root
symlink check; validates every complete raw dependency record, its version suffix, and the
Mach-O-record count before extracting paths; and confines build, staging, backup, output, and
cleanup paths to real directories and regular non-symlink files. Native source, ABI v4, schema 11,
Java, Gradle, and loader behavior remain unchanged.

Shell syntax; a fresh build, explicit ad-hoc signing, two byte-identical package publications, and
copy/move verification passed. The original eight corrupt/extra/root-or-member-symlink/manifest/
checksum/missing-signature/wrong-identifier cases still fail. Exact symlink-root spellings with
plain, trailing `/`, terminal `/.`, and repeated `/` forms fail; a re-signed malformed Foundation
load command with an embedded newline and hostile `/usr/lib` record fails after regenerated
manifest/checksums. External build/staging symlinks, non-directory build paths, and output symlink,
directory, symlink-to-directory, and FIFO cases fail without an outside write; the prior valid
build tree remains byte- and mode-identical. The regenerated packaged dylib passed 126 Metal tests
in 19 suites and 15 Engine integration tests with zero skips, failures, or errors. Markdown,
seven-path scope, shell syntax, and diff checks passed; no full repository build ran.

The artifact remains local and unauthenticated; every release-only exclusion remains. Status:
Review needed for the same independent Class C reviewer after P1 remediation.
