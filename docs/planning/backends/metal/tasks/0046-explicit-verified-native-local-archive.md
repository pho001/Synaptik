# Task 0046: Explicit Verified Metal Native Local Archive

## Status

Review needed

## Change class

Class C — the runtime API and native ABI remain unchanged, but this task creates a durable Gradle
handoff for security-sensitive native executable bytes. Use one clean implementation context and
an independent targeted review before completion.

## Goal

Given one explicitly supplied absolute path to the final reviewed Task 0045 macOS-arm64 package,
rerun its fail-closed verifier and create one reproducible, unversioned local ZIP. Preserve the
caller-owned extraction lifecycle and the existing explicit absolute dylib path passed to
`MetalBackendConfiguration`.

## Scope

- Add two lazily registered, opt-in tasks to `backends/metal`: `verifyMetalNativePackage` and
  `metalNativeLocalZip`.
- Require `-PsynaptikMetalNativePackage=/absolute/path` only when either task is selected.
- Reuse the Task 0045 verifier unchanged and archive only its exact three verified members.
- Document manual extraction, reverification, explicit loading, and the local unauthenticated
  boundary.
- Preserve ordinary Gradle lifecycle behavior and every runtime/native contract.

## Non-goals

- native build, signing, repair, normalization, install, extraction, discovery, download, cache, or
  runtime cleanup
- Gradle distribution plugin, consumable configuration, outgoing artifact, publication,
  repository, coordinates, version, classifier contract, or lifecycle relationship
- Developer ID, Team ID, credentials, keychain/CI secrets, notarization, provenance, CI, or public
  release
- Java, Config, loader, native source, ABI, schema, capability, dependency, test-source, settings,
  root-build, or architecture-contract changes
- Intel macOS, another platform selector, or a reproducible cross-toolchain dylib claim

## Contracts

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md) headings `Authority, incorporation, and
  precedence`, `Repository layout`, `Core invariants`, and `Testing requirements` — repository
  placement and proportionate verification remain unchanged.
- [`docs/architecture/contracts/backend-execution.md`](../../../../architecture/contracts/backend-execution.md)
  headings `Concrete backend modules` and `Metal backend` — Metal owns its bridge while callers
  continue to select one explicit absolute library path; Engine performs no discovery.
- [`docs/developer-guide/release-process.md`](../../../../developer-guide/release-process.md) — the
  local archive must not invent release versioning, coordinates, publisher identity, provenance,
  redistribution rights, or publication.
- [Task 0045](0045-verified-local-native-package.md) headings `Package contract`, `Verification
  contract`, and `Follow-up` — this task consumes only the final reviewed exact package boundary
  and reuses its verifier.

If implementation needs another platform, package member, public workflow, install/cache owner,
release mode, or architecture contract, stop and return this task to planning.

## Dependencies and integration

- Depends on: Metal 0045 Complete at same-reviewer-approved remediation `26c6c911`
- Conflicts with: every concurrent Metal Gradle/native distribution, native README,
  release-process, master, or roadmap edit; native discovery, release-signing, notarization, or
  publication work
- Parallel group: None
- Common base revision: `26c6c911`
- Integration order: sole Metal frontier after 0045
- Integration validation: Metal verified local archive checkpoint
- Shared-document integration owner: Main planner

## Files and responsibilities

- `backends/metal/build.gradle.kts` — lazy explicit input, verifier task, and exact reproducible ZIP
- `native/metal-macos-arm64/README.md` — opt-in archive, manual extraction/reverification, and
  absolute loader workflow
- `docs/developer-guide/release-process.md` — local archive versus public release boundary
- this task, `../master-plan.md`, and `../../../roadmap.md` — authorization and status

Exactly these six repository paths may change. Generated archives and test reports remain in
ignored build directories; disposable negative fixtures remain outside the repository and are
removed.

## Gradle input and isolation contract

The sole input is `-PsynaptikMetalNativePackage=/absolute/path/to/macos-arm64`. It has no default,
environment fallback, repository lookup, working-directory resolution, or platform discovery.
Blank and relative values fail. The exact supplied string reaches the existing verifier: Gradle
must not trim, normalize, call `toRealPath()`, canonicalize, or otherwise hide a trailing separator,
terminal dot, repeated separator, or symlink package root.

`verifyMetalNativePackage` is an `Exec` task with no outputs. It directly runs
`native/metal-macos-arm64/verify-package.sh` on every selection and inherits its exact schema,
Mach-O, dependency, export, signature, mode, checksum, path, and member checks. It does not call
`build.sh`, `package-local.sh`, or signing commands.

`metalNativeLocalZip` is one dedicated `Zip` task depending on `verifyMetalNativePackage`. Neither
task is related to `build`, `assemble`, `check`, `jar`, `test`, `clean`, publication, or another
lifecycle task. Configuring or running ordinary Gradle work without the property must not evaluate
it, inspect a package, or invoke native tools.

## Archive contract

The only output is
`backends/metal/build/distributions/synaptik-metal-macos-arm64-local.zip`. The fixed name is
unversioned and local; it must not inherit the root `0.1.0-SNAPSHOT` version.

The archive contains exactly:

```text
macos-arm64/
  libsynaptik_metal_foundation.dylib  0755
  manifest.json                       0644
  SHA256SUMS                          0644
```

Directory mode is `0755`. File order is reproducible and filesystem timestamps are not preserved.
No source path, host, time, revision, product version, SDK version, Team ID, provenance, legal
placeholder, or extra file is added. Platform selection is fixed to `macos-arm64`; the reused
verifier, not a new Gradle selector, authenticates that package shape.

## Extraction, loader, and legal boundary

Gradle installs or extracts nothing. The caller chooses a directory, uses a permission-preserving
extractor, reruns the same verifier on the extracted `macos-arm64` directory, retains/removes that
directory, and passes the extracted dylib's absolute path to `MetalBackendConfiguration`.
`SYNAPTIK_METAL_TEST_LIBRARY` remains test-only.

The ZIP is a verified local development transport, not an authenticated or public release.
Self-contained hashes do not authenticate hostile replacement, and the ad-hoc signature supplies
no publisher identity or Apple trust. The repository has no project-wide license file;
`THIRD_PARTY_NOTICES.md` applies only to CPU ERF material absent from this archive and must not be
copied into it. Do not invent redistribution rights or legal metadata.

## Acceptance criteria

- Ordinary `:backends:metal:build` succeeds without the property and neither new task runs.
- Selecting either task without the property fails clearly; blank and relative inputs fail without
  creating a new archive.
- A symlink-root package and the Task 0045 redundant terminal path forms fail through the existing
  verifier; no path is canonicalized before that check.
- A valid absolute Task 0045 package is verified before archiving, and the task graph has no native
  build/sign or ordinary lifecycle edge.
- Two forced ZIP executions over identical package bytes are byte-identical.
- The ZIP has exactly the three paths, fixed modes/order/timestamps, fixed local filename, and no
  source path or inferred version.
- Permission-preserving extraction followed by the unchanged verifier passes; a focused existing
  real-device Metal test loads the extracted dylib through an absolute path.
- Documentation preserves manual caller ownership and distinguishes integrity from authentication,
  notarization, publication, and release.
- No generated file or repository path outside the six-path scope remains.

## Validation

Worker validation:

```bash
./gradlew :backends:metal:build
./gradlew :backends:metal:verifyMetalNativePackage \
  -PsynaptikMetalNativePackage=/absolute/path/to/macos-arm64
./gradlew :backends:metal:metalNativeLocalZip --rerun-tasks \
  -PsynaptikMetalNativePackage=/absolute/path/to/macos-arm64
```

Also run missing, blank, relative, symlink-root, trailing-`/`, terminal-`/.`, and repeated-`/`
negative invocations; inspect the task graph and exact ZIP entries/modes/timestamps; compare two
fresh archive hashes/bytes; extract with `/usr/bin/ditto -x -k`; rerun `verify-package.sh`; and run
one focused existing real-device Metal test with `SYNAPTIK_METAL_TEST_LIBRARY` set to the extracted
absolute dylib path. Run shell/Markdown/six-path/diff checks. Do not run a full repository build.

No permanent Gradle TestKit test is required: the repository has no such convention, and command
behavior plus the existing permanent package verifier directly exercise laziness, isolation, and
artifact semantics without a source-text wiring assertion.

## Follow-up

Developer ID signing, secure keychain/CI secret handling, notary credentials, an accepted
notarizable container/process, project license and notice policy, product versioning, coordinates,
repository, provenance, CI promotion, rollback, support, and public release remain separately
blocked.

## Documentation and review impact

The native README and release guide change a durable developer workflow and security terminology.
Independent Class C review must inspect Provider laziness, exact raw path forwarding, task graph
isolation, verifier ordering, archive members/modes/reproducibility, absence of path/version/legal
leaks, manual extraction/loading, and absence of authentication or release claims.

## Result

Implemented from planning commit `35b222be` and clean reviewed Task 0045 base `26c6c911`.
`backends/metal` now lazily registers only `verifyMetalNativePackage` and
`metalNativeLocalZip`. The exact raw absolute property reaches the existing verifier; the verifier
has no outputs and executes on every selection. The ZIP task depends only on it, copies the exact
three members under `macos-arm64`, and has no ordinary lifecycle, build, signing, install,
discovery, configuration, publication, coordinate, version, Java, or native-source effect.

Task listing and ordinary `:backends:metal:build` passed without the property. Missing-property
selection of both tasks and blank, relative, symlink-root, trailing-`/`, terminal-`/.`, and
repeated-`/` inputs failed while the prior archive stayed unchanged. The explicit ZIP dry-run
contained only the verifier and ZIP tasks. Valid verifier executions ran under both stored and
reused configuration cache entries.

Two forced archives were byte-identical at SHA-256
`82e569f96772addeaf661b6d9d6fb2651926d727d7dd16ab3ae299d9f6dd5655`.
The 37,426-byte ZIP has exactly the `macos-arm64/` directory plus the three package members, fixed
`0755` directory/dylib and `0644` metadata modes, one `1980-02-01 00:00:00` timestamp, and no
source path or `0.1.0-SNAPSHOT` text. Permission-preserving `ditto` extraction followed by the
unchanged verifier passed. The extracted absolute dylib path passed the existing real-device
`MetalMpsGraphMatmulNativeTest`: one test, zero skips, failures, or errors.

The native README and release guide document manual caller-owned extraction, reverification,
retention, and explicit absolute-path loading while retaining the unauthenticated ad-hoc local
boundary. Six-path scope, Markdown, and diff checks passed; no full repository build ran. Status:
Review needed for mandatory independent Class C review.
