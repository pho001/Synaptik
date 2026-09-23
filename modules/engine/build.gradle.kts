dependencies {
    implementation(project(":modules:model"))
    implementation(project(":modules:planning"))
    implementation(project(":modules:compiler"))
    implementation(project(":modules:runtime"))
    implementation(project(":modules:prepare"))
    implementation(project(":modules:config"))
    implementation(project(":modules:trace"))
    api(project(":backends:cpu"))
    api(project(":backends:metal"))
    implementation(project(":backends:cuda"))
    implementation(project(":tools:tuning"))
}
