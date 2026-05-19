import os
import subprocess
import sys
import structlog

logger = structlog.get_logger()


def compile_target(cpp_source_rel: str, lib_name_base: str) -> bool:
    """
    Checks for C++ compiler and compiles a specific raw C++ shared speedups library.
    Selects correct platform file extension (.dll, .so, or .dylib) and compiles
    with aggressive -O3 optimization level.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cpp_source = os.path.join(base_dir, *cpp_source_rel.split("/"))
    
    # Select shared library name based on current operating system
    if sys.platform.startswith("win"):
        lib_name = f"{lib_name_base}.dll"
    elif sys.platform.startswith("darwin"):
        lib_name = f"{lib_name_base}.dylib"
    else:
        lib_name = f"{lib_name_base}.so"
        
    cpp_dest = os.path.join(os.path.dirname(cpp_source), lib_name)
    
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
            f"C++ compilation for {lib_name_base} completed successfully!",
            output=result.stdout,
            destination=cpp_dest
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(
            f"C++ compilation for {lib_name_base} failed via subprocess command",
            command=" ".join(compile_cmd),
            stderr=e.stderr,
            stdout=e.stdout
        )
        return False
    except FileNotFoundError:
        logger.warning(
            f"Compiler 'g++' not found in system PATH. Cannot compile acceleration layer for {lib_name_base}."
        )
        return False


def compile_all() -> bool:
    """
    Compiles all C++ extensions.
    """
    targets = [
        ("features/wavelet/kalman_speedups.cpp", "kalman_speedups"),
        ("models/rl_agent/rl_speedups.cpp", "rl_speedups")
    ]
    
    all_success = True
    for src, name in targets:
        success = compile_target(src, name)
        if not success:
            all_success = False
            
    return all_success


if __name__ == "__main__":
    success = compile_all()
    sys.exit(0 if success else 1)

