import os
import subprocess
import sys
import structlog

logger = structlog.get_logger()


def compile_library() -> bool:
    """
    Checks for C++ compiler and compiles the raw C++ shared speedups library.
    Selects correct platform file extension (.dll, .so, or .dylib) and compiles
    with aggressive -O3 optimization level.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cpp_source = os.path.join(base_dir, "features", "wavelet", "kalman_speedups.cpp")
    
    # Select shared library name based on current operating system
    if sys.platform.startswith("win"):
        lib_name = "kalman_speedups.dll"
    elif sys.platform.startswith("darwin"):
        lib_name = "kalman_speedups.dylib"
    else:
        lib_name = "kalman_speedups.so"
        
    cpp_dest = os.path.join(base_dir, "features", "wavelet", lib_name)
    
    logger.info(
        "Beginning compilation of C++ acceleration layer",
        source=cpp_source,
        destination=cpp_dest,
        platform=sys.platform
    )
    
    # Try using g++ first
    compiler = "g++"
    compile_cmd = [
        compiler,
        "-O3",
        "-shared",
        "-fPIC",
        cpp_source,
        "-o",
        cpp_dest
    ]
    
    try:
        # Run compilation command
        result = subprocess.run(
            compile_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        logger.info(
            "C++ compilation completed successfully!",
            output=result.stdout,
            destination=cpp_dest
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(
            "C++ compilation failed via subprocess command",
            command=" ".join(compile_cmd),
            stderr=e.stderr,
            stdout=e.stdout
        )
        return False
    except FileNotFoundError:
        logger.warning(
            "Compiler 'g++' not found in system PATH. Cannot compile acceleration layer."
        )
        return False


if __name__ == "__main__":
    success = compile_library()
    sys.exit(0 if success else 1)
