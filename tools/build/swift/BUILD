# load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

package(default_visibility = ["//visibility:public"])

# swift_library(
#     name = "audio_video_swift",
#     srcs = [
#         "//tensorflow_io/core:swift/audio.swift",
#         "//tensorflow_io/core:swift/video.swift",
#     ],
#     copts = [
#         "-target",
#         "x86_64-apple-macosx10.14",
#     ],
#     linkopts = [
#         "-L/Applications/Xcode.app/Contents/Developer//Toolchains/XcodeDefault.xctoolchain/usr/lib/swift",
#         "-Wl,-rpath,/Applications/Xcode.app/Contents/Developer//Toolchains/XcodeDefault.xctoolchain/usr/lib/swift",
#         "-target",
#         "x86_64-apple-macosx10.14",
#     ],
#     module_name = "audio_video",
#     alwayslink = True,
# )

cc_library(
    name = "audio_video_swift",
    srcs = [":sub/audio_video_swift"],
    # srcs = select({
    #     "@bazel_tools//src/conditions:windows": [
    #         "//tools/build/swift:libaudio_video_swift.dylib",
    #     ],
    #     "//conditions:default": [
    #         "//tools/build/swift:libaudio_video_swift.dylib",
    #     ],
    # }),
    #data = ["lib/libtensorflow_framework.so"],
    visibility = ["//visibility:public"],
    alwayslink = True,
)

cc_binary(
    name = "sub/audio_video_swift",
    srcs = [],
    linkopts = select({
        "@bazel_tools//src/conditions:windows": [],
        "@bazel_tools//src/conditions:darwin": [
            "-Wl,-install_name,@rpath/libaudio_video_swift.dylib",
        ],
        "//conditions:default": [
            "-Wl,--disable-new-dtags",
            "-Wl,-rpath,'$$ORIGIN/'",
            "-Wl,-soname,libaudio_video_swift.dylib",
        ],
    }),
    linkshared = 1,
    deps = [],
)

# cc_library(
#     name = "libaudio_video_swift.dylib",
#     srcs = ["//tools/build/swift:libaudio_video_swift.dylib"],
#     # copts = [
#     #     "-target",
#     #     "x86_64-apple-macosx10.14",
#     # ],
#     # linkopts = [
#     #     "-L/Applications/Xcode.app/Contents/Developer//Toolchains/XcodeDefault.xctoolchain/usr/lib/swift",
#     #     "-Wl,-rpath,/Applications/Xcode.app/Contents/Developer//Toolchains/XcodeDefault.xctoolchain/usr/lib/swift",
#     #     "-target",
#     #     "x86_64-apple-macosx10.14",
#     # ],
#     # srcs = ["//tools/build/swift:libaudio_video_swift.a", "//tools/build/swift:libaudio_video_swift.dylib"],
#     # hdrs = glob(["path/to/headers/*.h"]), # Include headers if needed
#     # module_name = "audio_video",
#     visibility = ["//visibility:public"],
#     # static_build_only = True,
#     alwayslink = True,
# )