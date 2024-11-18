import os
import shutil
import subprocess


def generate_lib_from_dll(dll_path, output_lib=None):
    """
    Generates a .lib file from a given .dll file using Visual Studio tools.

    Args:
        dll_path (str): Path to the DLL file.
        output_lib (str): Path for the generated .lib file (optional).

    Returns:
        str: Path to the generated .lib file.
    """
    if not os.path.isfile(dll_path):
        raise FileNotFoundError(f"DLL file not found: {dll_path}")

    # 获取文件名和路径
    base_name = os.path.splitext(os.path.basename(dll_path))[0]
    dll_dir = os.path.dirname(dll_path)
    def_file = os.path.join(dll_dir, f"{base_name}.def")
    output_lib = output_lib or os.path.join(dll_dir, f"{base_name}.lib")

    # 检查工具链是否存在
    if not shutil.which("dumpbin") or not shutil.which("lib"):
        raise EnvironmentError("Visual Studio tools (dumpbin, lib) not found in PATH.")

    try:
        # 使用 dumpbin 生成 .def 文件
        print(f"Generating .def file: {def_file}")
        with open(def_file, "w") as def_output:
            subprocess.run(["dumpbin", "/exports", dll_path], check=True, stdout=def_output)

        # 使用 lib 工具生成 .lib 文件
        print(f"Generating .lib file: {output_lib}")
        subprocess.run(["lib", f"/def:{def_file}", f"/out:{output_lib}", "/machine:x64"], check=True)

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error during Visual Studio toolchain execution: {e}")

    if not os.path.isfile(output_lib):
        raise FileNotFoundError(f"Failed to generate .lib file: {output_lib}")

    print(f".lib file successfully generated: {output_lib}")
    return output_lib


if __name__ == "__main__":
    # DLL 文件路径
    dll_file = r"C:\Users\edith\software\aubio-0.4.6-win64\bin\libaubio-5.dll"
    # 生成的 .lib 文件路径
    output_file = r"C:\Users\edith\software\aubio-0.4.6-win64\lib\libaubio-5.lib"

    try:
        generate_lib_from_dll(dll_file, output_lib=output_file)
    except Exception as e:
        print(f"Error: {e}")
