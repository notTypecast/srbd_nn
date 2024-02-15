#!/usr/bin/env python
# encoding: utf-8
import sys
import os
import fnmatch
import glob
import copy

sys.path.insert(0, sys.path[0] + "/waf_tools")

VERSION = "1.0.0"
APPNAME = "tiago_pick_place"

srcdir = "."
blddir = "build"

from waflib.Build import BuildContext
from waflib import Logs

# from waflib.Tools import waf_unit_test
import dart
import boost
import eigen
import corrade
import magnum
import magnum_integration
import magnum_plugins
import robot_dart
import tbb


def options(opt):
    opt.load("compiler_cxx")
    opt.load("compiler_c")
    # opt.load("waf_unit_test")
    opt.load("boost")
    opt.load("eigen")
    opt.load("dart")
    opt.load("corrade")
    opt.load("magnum")
    opt.load("magnum_integration")
    opt.load("magnum_plugins")
    opt.load("robot_dart")
    opt.load("tbb")


def configure(conf):
    conf.get_env()["BUILD_GRAPHIC"] = False

    conf.load("compiler_cxx")
    conf.load("compiler_c")
    # conf.load("waf_unit_test")
    conf.load("boost")
    conf.load("eigen")
    conf.load("dart")
    conf.load("avx")
    conf.load("corrade")
    conf.load("magnum")
    conf.load("magnum_integration")
    conf.load("magnum_plugins")
    conf.load("robot_dart")
    conf.load("tbb")

    conf.check_boost(lib="regex system filesystem", min_version="1.58")
    # we need pthread for video saving
    conf.check(features="cxx cxxprogram", lib=["pthread"], uselib_store="PTHREAD")
    conf.check_eigen(required=True, min_version=(3, 2, 92))
    conf.check_dart(required=True)
    conf.check_robot_dart(required=True)
    conf.check_corrade(components="Utility PluginManager", required=False)
    conf.env["magnum_dep_libs"] = (
        "MeshTools Primitives Shaders SceneGraph GlfwApplication Text MagnumFont"
    )
    conf.check_tbb(required=True)
    if conf.env["DEST_OS"] == "darwin":
        conf.env["magnum_dep_libs"] += " WindowlessCglApplication"
    else:
        conf.env["magnum_dep_libs"] += " WindowlessEglApplication"
    if len(conf.env.INCLUDES_Corrade):
        conf.check_magnum(components=conf.env["magnum_dep_libs"], required=False)
    if len(conf.env.INCLUDES_Magnum):
        conf.check_magnum_plugins(
            components="AssimpImporter StbTrueTypeFont", required=False
        )
        conf.check_magnum_integration(components="Dart Eigen", required=False)

    # We require Magnum DartIntegration, EigenIntegration, AssimpImporter, and StbTrueTypeFont
    if (
        len(conf.env.INCLUDES_MagnumIntegration_Dart) > 0
        and len(conf.env.INCLUDES_MagnumIntegration_Eigen) > 0
        and len(conf.env.INCLUDES_MagnumPlugins_AssimpImporter) > 0
        and len(conf.env.INCLUDES_MagnumPlugins_StbTrueTypeFont) > 0
    ):
        conf.get_env()["BUILD_GRAPHIC"] = True
        conf.env["magnum_libs"] = magnum.get_magnum_dependency_libs(
            conf, conf.env["magnum_dep_libs"]
        ) + magnum_integration.get_magnum_integration_dependency_libs(
            conf, "Dart Eigen"
        )

    avx_dart = conf.check_avx(
        lib="dart", required=["dart", "dart-utils", "dart-utils-urdf"]
    )
    avx_robot_dart = conf.check_avx(
        lib="robot_dart",
        required=["RobotDARTSimu", "RobotDARTMagnum"],
        lib_type="static",
    )

    native = ""
    native_icc = ""
    if avx_dart and avx_robot_dart:
        conf.msg("-march=native (AVX support)", "yes", color="GREEN")
        native = " -march=native"
        native_icc = " mtune=native"
    elif avx_dart or avx_robot_dart:
        conf.msg(
            "-march=native (AVX support)",
            "yes, but either DART or robot_dart are not compiled with native flags",
            color="YELLOW",
        )
        native = " -march=native"
        native_icc = " mtune=native"
    else:
        conf.msg("-march=native (AVX support)", "no (optional)", color="YELLOW")

    if conf.env.CXX_NAME in ["icc", "icpc"]:
        common_flags = "-Wall -std=c++17"
        opt_flags = " -O3 -xHost -unroll -g " + native_icc
    elif conf.env.CXX_NAME in ["clang"]:
        common_flags = "-Wall -std=c++17"
        # no-stack-check required for Catalina
        opt_flags = " -O3 -g -faligned-new  -fno-stack-check" + native
    else:
        gcc_version = int(conf.env["CC_VERSION"][0] + conf.env["CC_VERSION"][1])
        common_flags = "-Wall -std=c++17"
        opt_flags = " -O3 -g" + native
        if gcc_version >= 71:
            opt_flags = opt_flags + " -faligned-new"

    all_flags = common_flags + opt_flags
    conf.env["CXXFLAGS"] = conf.env["CXXFLAGS"] + all_flags.split(" ")

    if len(conf.env.CXXFLAGS_DART) > 0:
        if "-std=c++11" in conf.env["CXXFLAGS"]:
            conf.env["CXXFLAGS"].remove("-std=c++11")
        if "-std=c++0x" in conf.env["CXXFLAGS"]:
            conf.env["CXXFLAGS"].remove("-std=c++0x")
        conf.env["CXXFLAGS"] = conf.env["CXXFLAGS"] + conf.env.CXXFLAGS_DART

    # add the prefix
    conf.env["CXXFLAGS"] = conf.env["CXXFLAGS"]
    print(conf.env["CXXFLAGS"])


def summary(bld):
    lst = getattr(bld, "utest_results", [])
    total = 0
    tfail = 0
    if lst:
        total = len(lst)
        tfail = len([x for x in lst if x[1]])
    # waf_unit_test.summary(bld)
    if tfail > 0:
        bld.fatal("Build failed, because some tests failed!")


def build(bld):
    if (
        len(bld.env.INCLUDES_DART) == 0
        or len(bld.env.INCLUDES_EIGEN) == 0
        or len(bld.env.INCLUDES_BOOST) == 0
        or len(bld.env.INCLUDES_ROBOT_DART) == 0
        or len(bld.env.INCLUDES_TBB) == 0
    ):
        bld.fatal("Some libraries were not found! Cannot proceed!")

    #### compilation of experiment
    libs = "BOOST EIGEN DART PTHREAD ROBOT_DART TBB"

    if bld.get_env()["BUILD_GRAPHIC"]:
        bld.program(
            features="c cxx cxxprogram    ",
            install_path=None,
            source=[
                "main.cpp",
                "src/sim/single_rigid_body_dynamics.cpp",
            ],
            includes=["./", "./simple_nn/src", "./algevo/src"],
            uselib="ROBOT_DART_GRAPHIC " + bld.env["magnum_libs"] + libs,
            defines=["GRAPHIC", "USE_TBB", "USE_TBB_ONEAPI"],
            target="main",
        )
