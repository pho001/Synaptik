# Executable learning examples

Build one small Java application and follow it from a first [prediction](../glossary.md#term-prediction) through training,
[persistence](../glossary.md#term-persistence) and repeated execution. The [learning journey](../tutorials/index.md) explains the
same project chapter by chapter. This is an external [Maven](../glossary.md#term-maven) consumer: it imports supported public
APIs, not repository internals, and has one direct Synaptik [dependency](../glossary.md#term-dependency), `nn`.

Here **prediction** means computing an answer from current [weights](../glossary.md#term-weight); **training** changes those
weights by comparing predictions with supplied answers. **Persistence** writes the supported
[model](../glossary.md#term-model) description and current values to a file. **Repeated execution** reuses prepared work with
new input values; it does not mean reusing an old answer. The
[experiment table](../tutorials/index.md#the-experiment-in-concrete-terms) fixes the actual
[FLOAT32](../glossary.md#term-float32) `[2,1]` data and the one [bias](../glossary.md#term-bias)-free weight used throughout the chapters.

## Prerequisites and [artifact](../glossary.md#term-artifact) boundary

- [JDK](../glossary.md#term-java-development-kit-jdk) **26** selected by both `java` and [Maven](../glossary.md#term-maven), and Maven 3.9 or newer.
- Access to the caller-supplied **file Maven repository** containing the [sealed](../glossary.md#term-sealed) private
  `0.1.0-public-audit.20261104` candidate and its transitive artifacts. Obtain its absolute `file:`
  URL from the person who supplied those artifacts. This project neither publishes artifacts nor
  assumes that candidate is on Maven Central. It grants no licensing or redistribution permission.
- A terminal initially at the repository root; alternatively copy the entire `learning-journey`
  directory and enter that copied directory instead of the first command below.
- An existing writable destination directory for the saved [graph](../glossary.md#graph). The commands use the Maven-created
  [`target`](../glossary.md#term-target) directory. `save-load` and `all` **replace** the destination file if it already exists.

No data download, [GPU](../glossary.md#term-gpu), native [BLAS](../glossary.md#term-blas) installation, test framework, preview flag, or [Vector API](../glossary.md#term-vector-api) flag
is needed. Enable native access: `Engine.standard()` can probe optional [OpenBLAS](../glossary.md#term-openblas) through the
[Foreign Function and Memory API](../glossary.md#application-binary-interface-and-native-bridge) even when the numerical route uses portable Java. This flag is
permission, not [backend](../glossary.md#backend) selection.

An **artifact** is a built package, here a Maven [JAR](../glossary.md#term-jar) and its accompanying [metadata](../glossary.md#term-metadata). **Transitive**
artifacts are dependencies required by that direct `nn` [dependency](../glossary.md#term-dependency). A [file repository](../glossary.md#term-file-repository) is a local
directory arranged for Maven [lookup](../glossary.md#term-lookup), not an instruction to download a different [release](../glossary.md#term-release).
**Native** libraries [run](../glossary.md#term-run) [machine code](../glossary.md#term-machine-code) outside Java; the [Foreign Function and Memory](../glossary.md#application-binary-interface-and-native-bridge) ([FFM](../glossary.md#term-ffm)) [API](../glossary.md#term-application-programming-interface-api)
lets Java call that code and access non-[heap](../glossary.md#term-heap) memory. [CPU](../glossary.md#term-cpu) means central processor, GPU means
graphics processor, and BLAS expands to [Basic Linear Algebra Subprograms](../glossary.md#mps-mpsgraph-and-openblas), a numerical-library
interface. These explanations do not add any installation requirement to this example.

## Build the complete project

First set `SYNAPTIK_REPOSITORY_URL` in your shell to the absolute file URL you were supplied
(for example, the URL form starts with `file:///`, not a bare filesystem path). Then [run](../glossary.md#term-run):

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

The commands [target](../glossary.md#term-target) a POSIX shell on macOS/Linux. Windows requires its platform's [classpath](../glossary.md#term-classpath)
separator and equivalent environment-variable syntax. Re-export `LEARNING_CP` after opening a new
terminal. The [POM](learning-journey/pom.xml)<span data-glossary-navigation="true"> ([POM glossary](../glossary.md#term-pom))</span> pins the candidate and [compiler](../glossary.md#term-compiler) [plugin](../glossary.md#term-plugin); build output
and the [graph](../glossary.md#graph) stay under `target` when you use these commands. [Maven](../glossary.md#term-maven) plugins may require normal
plugin-repository access even though the Synaptik [artifacts](../glossary.md#term-artifact) come from a [file repository](../glossary.md#term-file-repository).

## Choose a chapter

Every invocation starts a new independent experiment. In particular `train` does not leave a
trained Java object for a later process's `reuse`: that mode performs its own training first.

| Mode | Concrete input and outcome | Tutorial |
|---|---|---|
| `first` | [Weight](../glossary.md#term-weight) `0`, input `[1,3]`: [prediction](../glossary.md#term-prediction) `[0,0]`, checked after [Model](../glossary.md#term-model) close | [First model](../tutorials/first-model.md) |
| [`tensors`](../glossary.md#tensor) | Copy a Java array, compose storage-free expressions, compute `[0,0]` | [Tensors, shapes and composition](../tutorials/tensors-and-composition.md) |
| [`gradients`](../glossary.md#term-gradient) | Sum squared error `40`, actual first weight [derivative](../glossary.md#term-derivative) `-40`, no update | [Loss and gradients](../tutorials/loss-and-gradients.md)<span data-glossary-navigation="true"> ([Loss glossary](../glossary.md#loss))</span> |
| `train` | One `Sgd(0.05)` update: private weight `2`, prediction `[2,6]` | [Train and evaluate](../tutorials/train-and-evaluate.md) |
| `save-load` | Train, save, close, load: fresh `[2,4]` predicts `[4,8]`; configure new training | [Save, close and load](../tutorials/save-close-load.md) |
| `reuse` | Fresh values through cached Model execution, then one explicit prepared handle | [Reusable execution](../tutorials/reusable-execution.md) |
| `invalid-shape` | Reject `[1,2]` where `[2,1]` is required, then accept a correct input | [Tensors, shapes and composition](../tutorials/tensors-and-composition.md) |
| `all` (default) | [Run](../glossary.md#term-run) the modes above in table order, ending with the rejected-[shape](../glossary.md#shape) check | [Complete journey](../tutorials/index.md) |

[Arguments](../glossary.md#term-argument) are `[mode] [graph-file]`; defaults are `all` and `learning-journey.graph` in the current
working directory. Only `all` and `save-load` use the file argument. Unknown modes and more than
two arguments fail with `IllegalArgumentException`. See the documented public `main` in the
[complete canonical Java source](learning-journey/src/main/java/io/github/pho001/synaptik/examples/LearningJourney.java)<span data-glossary-navigation="true"> ([canonical glossary](../glossary.md#term-canonical-form))</span>.
There are no separate chapter snippets to keep in sync.

## Expected application stdout

[expected-stdout.txt](learning-journey/expected-stdout.txt) is the complete expected stdout for
`all`. Chapter pages show their own portions. The complete stdout was checked byte-for-byte
against the installed private candidate using a fresh external [Maven](../glossary.md#term-maven) consumer and JDK26.
This is bounded execution evidence, not [qualification](../glossary.md#term-qualification) of other [backends](../glossary.md#backend), geometries or versions.
Maven [logs](../glossary.md#commit-release-distribution-log-and-seed-meanings)/[JVM](../glossary.md#term-java-virtual-machine-jvm) stderr are not part of that transcript. Numerical and [ownership](../glossary.md#term-ownership-meanings) [checkpoints](../glossary.md#term-checkpoint) throw
`AssertionError` without requiring `-ea`; the program does not merely print success labels.

The [gradient](../glossary.md#term-gradient) mode obtains the [derivative](../glossary.md#term-derivative) from `Engine.backward`, not from a hardcoded formula.
The [train mode](../glossary.md#inference-and-evaluation) obtains the pre-update [objective](../glossary.md#term-objective) and committed step number from `TrainingStep`.
The [shape](../glossary.md#shape) mode catches a real public compute rejection and fails if the bad input is accepted.

## Useful variation and failures

[Run](../glossary.md#term-run) just one mode by replacing `all` with [`gradients`](../glossary.md#term-gradient), or save to another existing directory by
changing the second [argument](../glossary.md#term-argument). To change numerical inputs, edit the [canonical](../glossary.md#term-canonical-form) source and rebuild;
chapter variations explain which expected values/checks must change together.

| Symptom | Cause and action |
|---|---|
| [Maven](../glossary.md#term-maven) cannot resolve `io.github.pho001.synaptik:nn` at the candidate version | Confirm the supplied file URL points to the Maven repository root and includes the full candidate [dependency](../glossary.md#term-dependency) set. Do not silently substitute another version. |
| `release version 26 not supported`, or class-file version failure | Select [JDK](../glossary.md#term-java-development-kit-jdk) 26 for Maven and `java`; inspect both version commands. |
| Restricted native-access warning | Use the [JVM](../glossary.md#term-java-virtual-machine-jvm) flag shown above on every run; it does not install native libraries. |
| Save raises `IOException` | Use an existing writable parent directory and a regular-file destination, not a directory. |
| `AssertionError` from a chapter | Its behavioral/numerical [checkpoint](../glossary.md#term-checkpoint) did not match. Preserve the output and environment; do not delete the check to make the example pass. |

For supported [shapes](../glossary.md#shape) and [backend](../glossary.md#backend) domains beyond these tiny [FLOAT32](../glossary.md#term-float32) [matrices](../glossary.md#term-matrix-meanings), use the
[API reference](../reference/index.md)<span data-glossary-navigation="true"> ([API glossary](../glossary.md#term-application-programming-interface-api))</span>, [framework explanation](../explanation/framework.md) and
[backend/performance next steps](../tutorials/backend-and-performance.md).
