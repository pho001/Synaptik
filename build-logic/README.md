# Build logic

No shared [Gradle](../docs/glossary.md#term-gradle) convention plugins are implemented here. Current build configuration lives in the
[root build script](../build.gradle.kts) and individual project build scripts.

A **[convention plugin](../docs/glossary.md#term-convention-plugin)** is a reusable bundle of build rules, such as a common Java [toolchain](../docs/glossary.md#term-toolchain) and
[Java compiler](../docs/glossary.md#term-compiler) settings. Introducing one would reduce repeated build configuration; this directory's
presence does not mean such a [plugin](../docs/glossary.md#term-plugin) is already available.
