#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy

module = Extension('exclude',
                    sources = ['exclude.c'],
                    include_dirs=[numpy.get_include()],
                    language = "c")

setup (name="exclude",
       description = "module for excluding atom distances based on intersection with a voronoi sphere",
       ext_modules = [module])
