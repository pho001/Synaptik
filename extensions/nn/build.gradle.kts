import org.gradle.api.tasks.testing.Test

dependencies {
    api(project(":modules:model"))
    api(project(":extensions:training"))
    api(project(":extensions:checkpoint"))
    api(project(":modules:engine"))
}

// Live-adapter transaction tests exercise the existing public CPU Engine lifecycle.
tasks.withType<Test>().configureEach {
    jvmArgs("--add-modules", "jdk.incubator.vector")
}
