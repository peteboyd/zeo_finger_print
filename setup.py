#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy

module = Extension('SphereCollision',
                    sources = ['exclude.c'],
                    include_dirs=[numpy.get_include()],
                    language = "c")

setup (name="SphereCollision",
       description = "module for excluding atom distances based on intersection with a voronoi sphere",
       ext_modules = [module])
