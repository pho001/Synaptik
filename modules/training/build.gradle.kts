import org.gradle.api.tasks.testing.Test

dependencies {
    api(project(":modules:model"))
    api(project(":modules:engine"))
}

tasks.withType<Test>().configureEach {
    jvmArgs("--add-modules", "jdk.incubator.vector")
}
