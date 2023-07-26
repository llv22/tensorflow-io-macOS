***

# TensorFlow I/O

Currently we have to use python 3.8 to compile and copy .so file to miniconda library path.

[![GitHub CI](https://github.com/tensorflow/io/workflows/GitHub%20CI/badge.svg?branch=master)](https://github.com/tensorflow/io/actions?query=branch%3Amaster)
[![PyPI](https://badge.fury.io/py/tensorflow-io.svg)](https://pypi.org/project/tensorflow-io/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/tensorflow/io/blob/master/LICENSE)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/io)

TensorFlow I/O is a collection of file systems and file formats that are not
available in TensorFlow's built-in support. A full list of supported file systems
and file formats by TensorFlow I/O can be found [here](https://www.tensorflow.org/io/api_docs/python/tfio).

The use of tensorflow-io is straightforward with keras. Below is an example
to [Get Started with TensorFlow](https://www.tensorflow.org/tutorials/quickstart/beginner) with
the data processing aspect replaced by tensorflow-io:

```python
import tensorflow as tf
import tensorflow_io as tfio

# Read the MNIST data into the IODataset.
dataset_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
d_train = tfio.IODataset.from_mnist(
    dataset_url + "train-images-idx3-ubyte.gz",
    dataset_url + "train-labels-idx1-ubyte.gz",
)

# Shuffle the elements of the dataset.
d_train = d_train.shuffle(buffer_size=1024)

# By default image data is uint8, so convert to float32 using map().
d_train = d_train.map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), y))

# prepare batches the data just like any other tf.data.Dataset
d_train = d_train.batch(32)

# Build the model.
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ]
)

# Compile the model.
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Fit the model.
model.fit(d_train, epochs=5, steps_per_epoch=200)
```

In the above [MNIST](http://yann.lecun.com/exdb/mnist/) example, the URL's
to access the dataset files are passed directly to the `tfio.IODataset.from_mnist` API call.
This is due to the inherent support that `tensorflow-io` provides for `HTTP`/`HTTPS` file system,
thus eliminating the need for downloading and saving datasets on a local directory.

NOTE: Since `tensorflow-io` is able to detect and uncompress the MNIST dataset automatically if needed,
we can pass the URL's for the compressed files (gzip) to the API call as is.

Please check the official [documentation](https://www.tensorflow.org/io) for more
detailed and interesting usages of the package.

## Installation

### How to build

```bash
conda create -n tf_io python=3.8
# only python 3.8 is supported with partool.par, so create a new env
conda activate tf_io
pip install --no-deps /tmp/tensorflow_pkg/tensorflow-2.9.0-cp38-cp38-macosx_10_13_x86_64.whl
sh ./configure.sh
export TF_HEADER_DIR=/Users/llv23/opt/miniconda3/lib/python3.10/site-packages/tensorflow/include
export TF_SHARED_LIBRARY_DIR=/Users/llv23/opt/miniconda3/lib/python3.10/site-packages/tensorflow
export TF_SHARED_LIBRARY_NAME=tensorflow_framework
# cp /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/tensorflow-io-macOS/bazel-bin/tensorflow_io_gcs_filesystem/core/python/ops/libtensorflow_io_gcs_filesystem.so tensorflow_io_gcs_filesystem/core/python/ops
# https://stackoverflow.com/questions/40260242/how-to-set-c-standard-version-when-build-with-bazel
bazel build --verbose_failures --experimental_repo_remote_exec --cxxopt='-std=c++14' --macos_sdk_version=10.14 //tensorflow_io_gcs_filesystem/...
# comment out xcode version check in /private/var/tmp/_bazel_llv23/a82ad01ec0c5d2a91897f1531acdf67b/external/build_bazel_rules_swift/swift/internal/xcode_swift_toolchain.bzl
# comment out /private/var/tmp/_bazel_llv23/a82ad01ec0c5d2a91897f1531acdf67b/external/com_github_azure_azure_sdk_for_cpp/sdk/core/azure-core/inc/azure/core/http/policies/policy.hpp and /private/var/tmp/_bazel_llv23/a82ad01ec0c5d2a91897f1531acdf67b/external/com_github_azure_azure_sdk_for_cpp/sdk/storage/azure-storage-common/inc/azure/storage/common/internal/storage_per_retry_policy.hpp for std::make_unique
bazel build --verbose_failures --experimental_repo_remote_exec --cxxopt='-std=c++14' --macos_sdk_version=10.14 //tensorflow_io/...
# refer to https://stackoverflow.com/questions/73141963/cannot-build-tensorflow-io-linking-tensorflow-io-python-ops-libtensorflow-io-g
bazel build -s --verbose_failures $BAZEL_OPTIMIZATION --experimental_repo_remote_exec //tensorflow_io/... //tensorflow_io_gcs_filesystem/...
python setup.py bdist_wheel
python setup.py bdist_wheel --project tensorflow-io-gcs-filesystem
```

1, change /private/var/tmp/_bazel_llv23/a82ad01ec0c5d2a91897f1531acdf67b/external/build_bazel_rules_swift/swift/internal/xcode_swift_toolchain.bzl

```bash Line 702
    # # Xcode 11.0 implies Swift 5.1.
    # if _is_xcode_at_least_version(xcode_config, "11.0"):
    #     requested_features.append(SWIFT_FEATURE_SUPPORTS_LIBRARY_EVOLUTION)
    #     requested_features.append(SWIFT_FEATURE_SUPPORTS_PRIVATE_DEPS)

    # # Xcode 11.4 implies Swift 5.2.
    # if _is_xcode_at_least_version(xcode_config, "11.4"):
    #     requested_features.append(SWIFT_FEATURE_ENABLE_SKIP_FUNCTION_BODIES)

    # # Xcode 12.5 implies Swift 5.4.
    # if _is_xcode_at_least_version(xcode_config, "12.5"):
    #     requested_features.append(SWIFT_FEATURE_SUPPORTS_SYSTEM_MODULE_FLAG)
```

```bash Line 556
    # # Xcode 12.0 implies Swift 5.3.
    # if _is_xcode_at_least_version(xcode_config, "12.0"):
    #     tool_configs[swift_action_names.PRECOMPILE_C_MODULE] = (
    #         swift_toolchain_config.driver_tool_config(
    #             driver_mode = "swiftc",
    #             env = env,
    #             execution_requirements = execution_requirements,
    #             swift_executable = swift_executable,
    #             toolchain_root = toolchain_root,
    #             use_param_file = True,
    #             worker_mode = "wrap",
    #         )
    #     )
```

2, change /private/var/tmp/_bazel_llv23/a82ad01ec0c5d2a91897f1531acdf67b/external/com_github_azure_azure_sdk_for_cpp/sdk/core/azure-core/src/cryptography/md5.cpp
/private/var/tmp/_bazel_llv23/a82ad01ec0c5d2a91897f1531acdf67b/external/com_github_azure_azure_sdk_for_cpp/sdk/core/azure-core/inc/azure/core/http/policies/policy.hpp
/private/var/tmp/_bazel_llv23/a82ad01ec0c5d2a91897f1531acdf67b/external/com_github_azure_azure_sdk_for_cpp/sdk/core/azure-core/inc/azure/core/internal/http/pipeline.hpp

```bash
#include "absl/memory/memory.h" 
std::make_unique to absl::make_unique
```

3, issue

```bash
Use --sandbox_debug to see verbose messages from the sandbox and retain the sandbox build root for debugging
ld: illegal thread local variable reference to regular symbol __ZN9grpc_core7ExecCtx9exec_ctx_E for architecture x86_64
clang: error: linker command failed with exit code 1 (use -v to see invocation)
```

Solution: <https://github.com/grpc/grpc/issues/13856>
change /private/var/tmp/_bazel_llv23/a82ad01ec0c5d2a91897f1531acdf67b/external/com_github_grpc_grpc/include/grpc/impl/codegen/port_platform.h
just change MACRO declaration for GPR_GCC_TLS to GPR_PTHREAD_TLS (#define GPR_GCC_TLS 1) -> (#define GPR_PTHREAD_TLS )

4, issue

```bash
ERROR: /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/tensorflow-io-macOS/tools/build/swift/BUILD:5:14: Compiling Swift module //tools/build/swift:audio_video_swift failed: Exec failed due to IOException: xcrun failed with code 1.
This most likely indicates that SDK version [10.10] for platform [MacOSX] is unsupported for the target version of xcode.
Process exited with status 1
stdout: stderr: xcodebuild: error: SDK "macosx10.10" cannot be located.
xcodebuild: error: SDK "macosx10.10" cannot be located.
xcrun: error: unable to lookup item 'Path' in SDK 'macosx10.10'
```

Solution: <https://github.com/google/mediapipe/issues/130>

```bash
bazel build --verbose_failures --experimental_repo_remote_exec --cxxopt='-std=c++14' --macos_sdk_version=10.14 //tensorflow_io_gcs_filesystem/...
bazel build --verbose_failures --experimental_repo_remote_exec --cxxopt='-std=c++14' --macos_sdk_version=10.14 //tensorflow_io/...
```

check version

```bash
(tf_io) Orlando:tensorflow-io-macOS llv23$ xcodebuild -showsdks
iOS SDKs:
	iOS 12.1                      	-sdk iphoneos12.1

iOS Simulator SDKs:
	Simulator - iOS 12.1          	-sdk iphonesimulator12.1

macOS SDKs:
	macOS 10.14                   	-sdk macosx10.14
	macOS 10.14                   	-sdk macosx10.14

tvOS SDKs:
	tvOS 12.1                     	-sdk appletvos12.1

tvOS Simulator SDKs:
	Simulator - tvOS 12.1         	-sdk appletvsimulator12.1

watchOS SDKs:
	watchOS 5.1                   	-sdk watchos5.1

watchOS Simulator SDKs:
	Simulator - watchOS 5.1       	-sdk watchsimulator5.1
```

5, issue

```bash comment out Line 277-278
/private/var/tmp/_bazel_llv23/a82ad01ec0c5d2a91897f1531acdf67b/external/build_bazel_rules_swift/tools/worker/swift_runner.cc
```

comment out for library  name = "python/ops/libtensorflow_io.so" in Line 16

/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/tensorflow-io-macOS/tensorflow_io/BUILD

cp /private/var/tmp/_bazel_llv23/a82ad01ec0c5d2a91897f1531acdf67b//execroot/org_tensorflow_io/bazel-out/darwin-fastbuild/bin/tensorflow_io/python/ops/libtensorflow_io.so /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/dl-built-libraries/tensorflow-built/2.9.1-cuda10.1-py3.10/ios/tensorflow_io_gcs_filesystem/core/python/ops

cp /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/dl-built-libraries/tensorflow-built/2.9.1-cuda10.1-py3.10/ios/tensorflow_io_gcs_filesystem/core/python/ops/*.so /Users/llv23/opt/miniconda3/lib/python3.10/site-packages/tensorflow_io/python/ops/

6, now temporiarly disable audio and video module

```bash
ERROR: /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/tensorflow-io-macOS/tools/build/swift/BUILD:5:14: Compiling Swift module //tools/build/swift:audio_video_swift failed: (Exit 1): worker failed: error executing command 
  (cd /private/var/tmp/_bazel_llv23/a82ad01ec0c5d2a91897f1531acdf67b/execroot/org_tensorflow_io && \
  exec env - \
    APPLE_SDK_PLATFORM=MacOSX \
    APPLE_SDK_VERSION_OVERRIDE=10.14 \
  bazel-out/darwin-opt-exec-50AE0418/bin/external/build_bazel_rules_swift/tools/worker/worker swiftc @bazel-out/darwin-fastbuild/bin/tools/build/swift/audio_video.swiftmodule-0.params)
# Configuration: 8e768e62908f0dc2c00112f94b5a81081c8500d096777954be71a7758c743006
# Execution platform: @local_execution_config_platform//:platform
<unknown>:0: error: no such file or directory: 'bazel-out/darwin-fastbuild/bin/tools/build/swift/audio_video.swiftmodule'
swift_worker: Could not copy bazel-out/darwin-fastbuild/bin/_swift_incremental/tools/build/swift/audio_video_swift_objs/swift/audio.swift.o to bazel-out/darwin-fastbuild/bin/tools/build/swift/audio_video_swift_objs/swift/audio.swift.o (errno 2)
```

you need to check tensorflow_io/core/BUILD and comments out

```bash
cc_library(
    name = "audio_video_ops",
    srcs = [
        "kernels/audio_kernels.cc",
        "kernels/audio_kernels.h",
        "kernels/audio_video_flac_kernels.cc",
        "kernels/audio_video_mp3_kernels.cc",
        "kernels/audio_video_mp4_kernels.cc",
        "kernels/audio_video_ogg_kernels.cc",
        "kernels/audio_video_wav_kernels.cc",
        "kernels/video_kernels.cc",
        "kernels/video_kernels.h",
        "ops/audio_ops.cc",
        "ops/video_ops.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@flac",
        "@minimp3",
        "@speexdsp",
        "@minimp4",
        "@vorbis",
        "//tensorflow_io/core:dataset_ops",
    ] + select({
        #
        # "@bazel_tools//src/conditions:darwin": [
        #     "//tools/build/swift:audio_video_swift",
        # ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)
```

Copy output of all files in /private/var/tmp/_bazel_llv23/a82ad01ec0c5d2a91897f1531acdf67b//execroot/org_tensorflow_io/bazel-out/darwin-fastbuild/bin/tensorflow_io/python/ops/ to /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/dl-built-libraries/tensorflow-built/2.9.1-cuda10.1-py3.10/ios/tensorflow_io_gcs_filesystem/core/python/ops/

then install for 

cp -rf /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/dl-built-libraries/tensorflow-built/2.9.1-cuda10.1-py3.10/ios/tensorflow_io_gcs_filesystem/core/python/ops/*.so /Users/llv23/opt/miniconda3/lib/python3.10/site-packages/tensorflow_io/python/ops/

### Python Package

The `tensorflow-io` Python package can be installed with pip directly using:

```sh
$ pip install tensorflow-io
```

People who are a little more adventurous can also try our nightly binaries:

```sh
$ pip install tensorflow-io-nightly
```

To ensure you have a version of TensorFlow that is compatible with TensorFlow-IO,
you can specify the `tensorflow` extra requirement during install:

    pip install tensorflow-io[tensorflow]

Similar extras exist for the `tensorflow-gpu`, `tensorflow-cpu` and `tensorflow-rocm`
packages.

### Docker Images

In addition to the pip packages, the docker images can be used to quickly get started.

For stable builds:

```sh
$ docker pull tfsigio/tfio:latest
$ docker run -it --rm --name tfio-latest tfsigio/tfio:latest
```

For nightly builds:

```sh
$ docker pull tfsigio/tfio:nightly
$ docker run -it --rm --name tfio-nightly tfsigio/tfio:nightly
```

### R Package

Once the `tensorflow-io` Python package has been successfully installed, you
can install the development version of the R package from GitHub via the following:

```r
if (!require("remotes")) install.packages("remotes")
remotes::install_github("tensorflow/io", subdir = "R-package")
```

### TensorFlow Version Compatibility

To ensure compatibility with TensorFlow, it is recommended to install a matching
version of TensorFlow I/O according to the table below. You can find the list
of releases [here](https://github.com/tensorflow/io/releases).

| TensorFlow I/O Version | TensorFlow Compatibility | Release Date |
| ---------------------- | ------------------------ | ------------ |
| 0.26.0                 | 2.9.x                    | May 17, 2022 |
| 0.25.0                 | 2.8.x                    | Apr 19, 2022 |
| 0.24.0                 | 2.8.x                    | Feb 04, 2022 |
| 0.23.1                 | 2.7.x                    | Dec 15, 2021 |
| 0.23.0                 | 2.7.x                    | Dec 14, 2021 |
| 0.22.0                 | 2.7.x                    | Nov 10, 2021 |
| 0.21.0                 | 2.6.x                    | Sep 12, 2021 |
| 0.20.0                 | 2.6.x                    | Aug 11, 2021 |
| 0.19.1                 | 2.5.x                    | Jul 25, 2021 |
| 0.19.0                 | 2.5.x                    | Jun 25, 2021 |
| 0.18.0                 | 2.5.x                    | May 13, 2021 |
| 0.17.1                 | 2.4.x                    | Apr 16, 2021 |
| 0.17.0                 | 2.4.x                    | Dec 14, 2020 |
| 0.16.0                 | 2.3.x                    | Oct 23, 2020 |
| 0.15.0                 | 2.3.x                    | Aug 03, 2020 |
| 0.14.0                 | 2.2.x                    | Jul 08, 2020 |
| 0.13.0                 | 2.2.x                    | May 10, 2020 |
| 0.12.0                 | 2.1.x                    | Feb 28, 2020 |
| 0.11.0                 | 2.1.x                    | Jan 10, 2020 |
| 0.10.0                 | 2.0.x                    | Dec 05, 2019 |
| 0.9.1                  | 2.0.x                    | Nov 15, 2019 |
| 0.9.0                  | 2.0.x                    | Oct 18, 2019 |
| 0.8.1                  | 1.15.x                   | Nov 15, 2019 |
| 0.8.0                  | 1.15.x                   | Oct 17, 2019 |
| 0.7.2                  | 1.14.x                   | Nov 15, 2019 |
| 0.7.1                  | 1.14.x                   | Oct 18, 2019 |
| 0.7.0                  | 1.14.x                   | Jul 14, 2019 |
| 0.6.0                  | 1.13.x                   | May 29, 2019 |
| 0.5.0                  | 1.13.x                   | Apr 12, 2019 |
| 0.4.0                  | 1.13.x                   | Mar 01, 2019 |
| 0.3.0                  | 1.12.0                   | Feb 15, 2019 |
| 0.2.0                  | 1.12.0                   | Jan 29, 2019 |
| 0.1.0                  | 1.12.0                   | Dec 16, 2018 |

## Performance Benchmarking

We use [github-pages](https://tensorflow.github.io/io/dev/bench/) to document the results of API performance benchmarks. The benchmark job is triggered on every commit to `master` branch and
facilitates tracking performance w\.r.t commits.

## Contributing

Tensorflow I/O is a community led open source project. As such, the project
depends on public contributions, bug-fixes, and documentation. Please see:

*   [contribution guidelines](CONTRIBUTING.md) for a guide on how to contribute.

*   [development doc](docs/development.md) for instructions on the development environment setup.

*   [tutorials](docs/tutorials) for a list of tutorial notebooks and instructions on how to write one.

### Build Status and CI

| Build              | Status                                                                                                                                                                                 |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Linux CPU Python 2 | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/io/ubuntu-py2.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/io/ubuntu-py2.html)         |
| Linux CPU Python 3 | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/io/ubuntu-py3.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/io/ubuntu-py3.html)         |
| Linux GPU Python 2 | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/io/ubuntu-gpu-py2.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/io/ubuntu-gpu-py2.html) |
| Linux GPU Python 3 | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/io/ubuntu-gpu-py3.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/io/ubuntu-gpu-py3.html) |

Because of manylinux2010 requirement, TensorFlow I/O is built with
Ubuntu:16.04 + Developer Toolset 7 (GCC 7.3) on Linux. Configuration
with Ubuntu 16.04 with Developer Toolset 7 is not exactly straightforward.
If the system have docker installed, then the following command
will automatically build manylinux2010 compatible whl package:

```sh
#!/usr/bin/env bash

ls dist/*
for f in dist/*.whl; do
  docker run -i --rm -v $PWD:/v -w /v --net=host quay.io/pypa/manylinux2010_x86_64 bash -x -e /v/tools/build/auditwheel repair --plat manylinux2010_x86_64 $f
done
sudo chown -R $(id -nu):$(id -ng) .
ls wheelhouse/*
```

It takes some time to build, but once complete, there will be python
`3.5`, `3.6`, `3.7` compatible whl packages available in `wheelhouse`
directory.

On macOS, the same command could be used. However, the script expects `python` in shell
and will only generate a whl package that matches the version of `python` in shell. If
you want to build a whl package for a specific python then you have to alias this version
of python to `python` in shell. See [.github/workflows/build.yml](.github/workflows/build.yml)
Auditwheel step for instructions how to do that.

Note the above command is also the command we use when releasing packages for Linux and macOS.

TensorFlow I/O uses both GitHub Workflows and Google CI (Kokoro) for continuous integration.
GitHub Workflows is used for macOS build and test. Kokoro is used for Linux build and test.
Again, because of the manylinux2010 requirement, on Linux whl packages are always
built with Ubuntu 16.04 + Developer Toolset 7. Tests are done on a variatiy of systems
with different python3 versions to ensure a good coverage:

| Python | Ubuntu 18.04 | Ubuntu 20.04 | macOS + osx9 | Windows-2019 |
| ------ | ------------ | ------------ | ------------ | ------------ |
| 2.7    | ✔️           | ✔️           | ✔️           | N/A          |
| 3.7    | ✔️           | ✔️           | ✔️           | ✔️           |
| 3.8    | ✔️           | ✔️           | ✔️           | ✔️           |

TensorFlow I/O has integrations with many systems and cloud vendors such as
Prometheus, Apache Kafka, Apache Ignite, Google Cloud PubSub, AWS Kinesis,
Microsoft Azure Storage, Alibaba Cloud OSS etc.

We tried our best to test against those systems in our continuous integration
whenever possible. Some tests such as Prometheus, Kafka, and Ignite
are done with live systems, meaning we install Prometheus/Kafka/Ignite on CI machine before
the test is run. Some tests such as Kinesis, PubSub, and Azure Storage are done
through official or non-official emulators. Offline tests are also performed whenever
possible, though systems covered through offine tests may not have the same
level of coverage as live systems or emulators.

|                              | Live System | Emulator    | CI Integration | Offline |
| ---------------------------- | ----------- | ----------- | -------------- | ------- |
| Apache Kafka                 | ✔️          |             | ✔️             |         |
| Apache Ignite                | ✔️          |             | ✔️             |         |
| Prometheus                   | ✔️          |             | ✔️             |         |
| Google PubSub                |             | ✔️          | ✔️             |         |
| Azure Storage                |             | ✔️          | ✔️             |         |
| AWS Kinesis                  |             | ✔️          | ✔️             |         |
| Alibaba Cloud OSS            |             |             |                | ✔️      |
| Google BigTable/BigQuery     |             | to be added |                |         |
| Elasticsearch (experimental) | ✔️          |             | ✔️             |         |
| MongoDB (experimental)       | ✔️          |             | ✔️             |         |

References for emulators:

*   Official [PubSub Emulator](https://cloud.google.com/sdk/gcloud/reference/beta/emulators/pubsub/) by Google Cloud for Cloud PubSub.

*   Official [Azurite Emulator](https://github.com/Azure/Azurite) by Azure for Azure Storage.

*   None-official [LocalStack emulator](https://github.com/localstack/localstack) by LocalStack for AWS Kinesis.

## Community

*   SIG IO [Google Group](https://groups.google.com/a/tensorflow.org/forum/#!forum/io) and mailing list: [io@tensorflow.org](io@tensorflow.org)

*   SIG IO [Monthly Meeting Notes](https://docs.google.com/document/d/1CB51yJxns5WA4Ylv89D-a5qReiGTC0GYum6DU-9nKGo/edit)

*   Gitter room: [tensorflow/sig-io](https://gitter.im/tensorflow/sig-io)

## Additional Information

*   [Streaming Machine Learning with Tiered Storage and Without a Data Lake](https://www.confluent.io/blog/streaming-machine-learning-with-tiered-storage/) - [Kai Waehner](https://github.com/kaiwaehner)

*   [TensorFlow with Apache Arrow Datasets](https://medium.com/tensorflow/tensorflow-with-apache-arrow-datasets-cdbcfe80a59f) - [Bryan Cutler](https://github.com/BryanCutler)

*   [How to build a custom Dataset for Tensorflow](https://towardsdatascience.com/how-to-build-a-custom-dataset-for-tensorflow-1fe3967544d8) - [Ivelin Ivanov](https://github.com/ivelin)

*   [TensorFlow on Apache Ignite](https://medium.com/tensorflow/tensorflow-on-apache-ignite-99f1fc60efeb) - [Anton Dmitriev](https://github.com/dmitrievanthony)

## License

[Apache License 2.0](LICENSE)
