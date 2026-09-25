# Release process status

## What you will learn

Synaptik does not currently define or automate a public release process. This page records the safe contributor boundary so a local build is not mistaken for a published release.

## Current verification

Before sharing a development change, run the checks required by its task. The broad repository checks are:

```bash
./gradlew test
./gradlew build
git diff --check
```

For a Java module with public API changes, also generate that module's Javadoc; for example:

```bash
./gradlew :modules:model:javadoc
```

These commands verify the checkout and documentation formatting. They do not assign a version, publish artifacts, create a tag, generate a changelog, sign output, or establish compatibility.

## Opt-in Metal local archive

`:backends:metal:metalNativeLocalZip` accepts only an explicitly supplied absolute path to the
reviewed macOS-arm64 local package, reruns its verifier, and writes the fixed unversioned
`synaptik-metal-macos-arm64-local.zip` below that module's ignored `build/distributions/`
directory. The task is not connected to ordinary Gradle lifecycle work and performs no native
build, signing, installation, extraction, discovery, publication, or upload.

This ZIP remains an unauthenticated local development transport. Its self-contained hashes detect
corruption but not hostile replacement, and its ad-hoc signature establishes no publisher identity
or Apple trust. Callers own permission-preserving extraction, reverification, retention, and the
explicit absolute dylib path supplied to Metal. The archive defines no release version,
coordinates, provenance, redistribution rights, or legal notice policy.

## Not yet defined

Versioning policy, compatibility guarantees, artifact coordinates, repositories, signing, provenance, release notes, CI promotion, rollback, and support policy require explicit future planning. Contributors must not infer them from Gradle project names or create ad hoc publication credentials and workflows.

## Typical mistakes

| Symptom | Cause | Correction |
|---|---|---|
| `build` success is called a release | Verification and publication were conflated. | Describe it as a verified development build. |
| A guide promises semantic versioning | No policy has been accepted. | Wait for a focused release plan and decision. |
| Credentials appear in repository files | Publication was improvised. | Stop and design a secure release workflow before adding secrets. |

## Related documentation

- [Public API status](../api/public-api.md)
- [Planning guide](../planning/planning-guide.md)
- [Documentation rules](documentation-rules.md)
