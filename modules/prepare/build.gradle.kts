dependencies {
    api(project(":modules:runtime"))
    api(project(":modules:planning"))
    api(project(":modules:compiler"))
    implementation(project(":modules:config"))
    api(project(":modules:backend-contract"))
    implementation(project(":modules:trace"))
}
