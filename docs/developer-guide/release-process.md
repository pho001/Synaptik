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
