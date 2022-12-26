"""Build `units` C-extension module"""
from setuptools import Extension
from setuptools.command.build_py import build_py as _build_py
import numpy as np

# pylint:disable=missing-class-docstring
class build_py(_build_py):
    def run(self):
        self.run_command("build_ext")
        return super().run()

    def initialize_options(self):
        super().initialize_options()
        if self.distribution.ext_modules is None:
            self.distribution.ext_modules = []
        self.distribution.ext_modules.append(
            Extension(
                "microohm.quantitydtype",
                sources=[
                    "microohm/quantitydtype.c",
                    "microohm/casts.c",
                    "microohm/multiply.c",
                    "microohm/divide.c",
                    "microohm/add.c",
                    "microohm/subtract.c",
                    "microohm/negative.c",
                    "microohm/absolute.c",
                    "microohm/greater.c",
                    "microohm/less.c",
                    "microohm/maximum.c",
                    "microohm/minimum.c",
                    "microohm/quantitydtype_module.c",
                ],
                extra_compile_args=[f"-I{np.get_include()}"],
            )
        )
