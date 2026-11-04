# Executable learning examples

Build one small Java application and follow it from a first prediction through training,
persistence and repeated execution. The [learning journey](../tutorials/index.md) explains the
same project chapter by chapter. This is an external Maven consumer: it imports supported public
APIs, not repository internals, and has one direct Synaptik dependency, `nn`.

## Prerequisites and artifact boundary

- JDK **26** selected by both `java` and Maven, and Maven 3.9 or newer.
- Access to the caller-supplied **file Maven repository** containing the sealed private
  `0.1.0-public-audit.20261104` candidate and its transitive artifacts. Obtain its absolute `file:`
  URL from the person who supplied those artifacts. This project neither publishes artifacts nor
  assumes that candidate is on Maven Central. It grants no licensing or redistribution permission.
- A terminal initially at the repository root; alternatively copy the entire `learning-journey`
  directory and enter that copied directory instead of the first command below.
- An existing writable destination directory for the saved graph. The commands use the Maven-created
  `target` directory. `save-load` and `all` **replace** the destination file if it already exists.

No data download, GPU, native BLAS installation, test framework, preview flag, or Vector API flag
is needed. Enable native access: `Engine.standard()` can probe optional OpenBLAS through the
Foreign Function and Memory API even when the numerical route uses portable Java. This flag is
permission, not backend selection.

## Build the complete project

First set `SYNAPTIK_REPOSITORY_URL` in your shell to the absolute file URL you were supplied
(for example, the URL form starts with `file:///`, not a bare filesystem path). Then run:

```bash
cd docs/examples/learning-journey
: "${SYNAPTIK_REPOSITORY_URL:?Set this to the supplied absolute file Maven repository URL}"
case "$SYNAPTIK_REPOSITORY_URL" in
  file:///*) ;;
  *) printf '%s\n' 'Expected an absolute file:/// repository URL' >&2; exit 1 ;;
esac
java -version
mvn -version
mvn -Dsynaptik.repository.url="$SYNAPTIK_REPOSITORY_URL" \
  compile dependency:build-classpath -Dmdep.outputFile=target/runtime-classpath.txt
export LEARNING_CP="target/classes:$(cat target/runtime-classpath.txt)"
java --enable-native-access=ALL-UNNAMED -cp "$LEARNING_CP" \
  io.github.pho001.synaptik.examples.LearningJourney all target/learning-journey.graph
```

The commands target a POSIX shell on macOS/Linux. Windows requires its platform's classpath
separator and equivalent environment-variable syntax. Re-export `LEARNING_CP` after opening a new
terminal. The [POM](learning-journey/pom.xml) pins the candidate and compiler plugin; build output
and the graph stay under `target` when you use these commands. Maven plugins may require normal
plugin-repository access even though the Synaptik artifacts come from a file repository.

## Choose a chapter

Every invocation starts a new independent experiment. In particular `train` does not leave a
trained Java object for a later process's `reuse`: that mode performs its own training first.

| Mode | Concrete input and outcome | Tutorial |
|---|---|---|
| `first` | Weight `0`, input `[1,3]`: prediction `[0,0]`, checked after Model close | [First model](../tutorials/first-model.md) |
| `tensors` | Copy a Java array, compose storage-free expressions, compute `[0,0]` | [Tensors, shapes and composition](../tutorials/tensors-and-composition.md) |
| `gradients` | Sum squared error `40`, actual first weight derivative `-40`, no update | [Loss and gradients](../tutorials/loss-and-gradients.md) |
| `train` | One `Sgd(0.05)` update: private weight `2`, prediction `[2,6]` | [Train and evaluate](../tutorials/train-and-evaluate.md) |
| `save-load` | Train, save, close, load: fresh `[2,4]` predicts `[4,8]`; configure new training | [Save, close and load](../tutorials/save-close-load.md) |
| `reuse` | Fresh values through cached Model execution, then one explicit prepared handle | [Reusable execution](../tutorials/reusable-execution.md) |
| `invalid-shape` | Reject `[1,2]` where `[2,1]` is required, then accept a correct input | [Tensors, shapes and composition](../tutorials/tensors-and-composition.md) |
| `all` (default) | Run the modes above in table order, ending with the rejected-shape check | [Complete journey](../tutorials/index.md) |

Arguments are `[mode] [graph-file]`; defaults are `all` and `learning-journey.graph` in the current
working directory. Only `all` and `save-load` use the file argument. Unknown modes and more than
two arguments fail with `IllegalArgumentException`. See the documented public `main` in the
[complete canonical Java source](learning-journey/src/main/java/io/github/pho001/synaptik/examples/LearningJourney.java).
There are no separate chapter snippets to keep in sync.

## Expected application stdout

[expected-stdout.txt](learning-journey/expected-stdout.txt) is the complete expected stdout for
`all`. Chapter pages show their own portions. The complete stdout was checked byte-for-byte
against the installed private candidate using a fresh external Maven consumer and JDK26.
This is bounded execution evidence, not qualification of other backends, geometries or versions.
Maven logs/JVM stderr are not part of that transcript. Numerical and ownership checkpoints throw
`AssertionError` without requiring `-ea`; the program does not merely print success labels.

The gradient mode obtains the derivative from `Engine.backward`, not from a hardcoded formula.
The train mode obtains the pre-update objective and committed step number from `TrainingStep`.
The shape mode catches a real public compute rejection and fails if the bad input is accepted.

## Useful variation and failures

Run just one mode by replacing `all` with `gradients`, or save to another existing directory by
changing the second argument. To change numerical inputs, edit the canonical source and rebuild;
chapter variations explain which expected values/checks must change together.

| Symptom | Cause and action |
|---|---|
| Maven cannot resolve `io.github.pho001.synaptik:nn` at the candidate version | Confirm the supplied file URL points to the Maven repository root and includes the full candidate dependency set. Do not silently substitute another version. |
| `release version 26 not supported`, or class-file version failure | Select JDK 26 for Maven and `java`; inspect both version commands. |
| Restricted native-access warning | Use the JVM flag shown above on every run; it does not install native libraries. |
| Save raises `IOException` | Use an existing writable parent directory and a regular-file destination, not a directory. |
| `AssertionError` from a chapter | Its behavioral/numerical checkpoint did not match. Preserve the output and environment; do not delete the check to make the example pass. |

For supported shapes and backend domains beyond these tiny FLOAT32 matrices, use the
[API reference](../reference/index.md), [framework explanation](../explanation/framework.md) and
[backend/performance next steps](../tutorials/backend-and-performance.md).
