dependencies {
    api(project(":extensions:nn"))
    api(project(":modules:engine"))
    api(project(":modules:model"))
}

tasks.withType<Test>().configureEach {
    jvmArgs("--add-modules", "jdk.incubator.vector")
}
