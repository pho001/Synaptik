dependencies {
    api(project(":modules:model"))
    implementation(project(":modules:planning"))
    api(project(":modules:compiler"))
    api(project(":modules:runtime"))
    implementation(project(":modules:prepare"))
    api(project(":modules:config"))
    implementation(project(":modules:trace"))
    api(project(":backends:cpu"))
    api(project(":backends:metal"))
    implementation(project(":tools:tuning"))
}

tasks.withType<Test>().configureEach {
    systemProperty("synaptik.cpu.dynamicBatch.performance", System.getProperty("synaptik.cpu.dynamicBatch.performance", "false"))
}
