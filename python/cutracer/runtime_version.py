# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Version identity for the CUTracer runtime in the current Python package."""

from importlib.metadata import PackageNotFoundError, version


def get_runtime_version() -> str:
    try:
        return version("cutracer")
    except PackageNotFoundError:
        pass

    from cutracer.shared_vars import is_fbcode

    if is_fbcode():
        from cutracer.fb.version import get_build_version

        build_version = get_build_version()
        if build_version:
            return build_version
    return "0+unknown"
