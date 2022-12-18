"""Build `units` C-extension module"""
from setuptools import Extension
from setuptools.command.build_py import build_py as _build_py

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
            Extension("microohm.units", sources=["microohm/units.c"])
        )
