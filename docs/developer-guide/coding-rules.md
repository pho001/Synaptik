# Coding rules

The project compiles and tests against Java 26. Production code may use stable Java 26 language features and APIs when they preserve the module boundaries in [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md).

Preview features are not enabled globally. A task that requires a preview feature must justify it, limit compiler and runtime flags to the owning module, document portability and lifecycle risks, and add focused validation. Incubator modules follow the same task-scoped rule.

The Vector API remains an incubator module in Java 26. A future vectorization task may configure `jdk.incubator.vector` for the applicable backend module; the core project does not add that module until it is used. Incubator module activation uses module flags and must not be confused with the unrelated `--enable-preview` flag.

All changes must also follow [`../../AGENTS.md`](../../AGENTS.md), including testing, documentation review, complete Javadoc, package-structure planning, and task isolation.
