# Getting started

Synaptik requires JDK 26. Confirm the active JDK before importing or building the project:

```bash
java -version
```

Import the repository as a Gradle project in IntelliJ IDEA and select a Java 26 SDK. The project uses the checked-in Gradle wrapper, so a separate Gradle installation is not required.

Useful validation commands are:

```bash
./gradlew projects
./gradlew test
./gradlew build
```

Java preview features are disabled by default. Incubator or preview APIs are configured only by focused module tasks when the capability cannot be implemented through stable Java 26 APIs.

The public compile, prepare, and run workflow will be documented here when the corresponding modules are implemented.
