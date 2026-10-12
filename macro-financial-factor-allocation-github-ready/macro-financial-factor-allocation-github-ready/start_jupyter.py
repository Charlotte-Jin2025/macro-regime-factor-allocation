"""Check this project's environment or start Jupyter at its portable project root.

This script never installs packages. Run with --check for a read-only preflight.
The interpreter executing this file is also the interpreter starting Jupyter.
"""
from pathlib import Path
import argparse
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

BASE = Path(__file__).resolve().parent
RESEARCH_PACKAGES = {
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
    "sklearn": "scikit-learn", "statsmodels": "statsmodels",
    "matplotlib": "matplotlib", "PIL": "pillow",
    "docx": "python-docx", "lxml": "lxml",
}
NOTEBOOK_PACKAGES = {"notebook": "notebook", "ipykernel": "ipykernel",
                     "nbformat": "nbformat", "nbclient": "nbclient"}
KERNEL_NAME = "utsg7394_crossmarket"


def inspect_environment(include_notebook=True):
    """Inspect import availability and tested direct versions without mutations."""
    required = dict(RESEARCH_PACKAGES)
    if include_notebook:
        required.update(NOTEBOOK_PACKAGES)
    pins = {}
    for line in (BASE / "requirements-tested.txt").read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.lstrip().startswith("#"):
            name, version = line.strip().split("==", 1)
            pins[name] = version
    versions, missing, mismatches = {}, [], []
    for module, distribution in required.items():
        if importlib.util.find_spec(module) is None:
            missing.append(distribution)
            continue
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = "metadata unavailable"
        versions[distribution] = version
        if distribution in pins and version != pins[distribution]:
            mismatches.append({"package": distribution, "installed": version,
                               "tested": pins[distribution]})
    return {"python": sys.version.split()[0], "executable": sys.executable,
            "project": str(BASE), "missing": missing,
            "version_differences": mismatches, "versions": versions,
            "python_minimum_satisfied": sys.version_info >= (3, 11),
            "python_minor_tested": sys.version_info[:2] == (3, 13),
            "project_complete": all((BASE / name).is_file() for name in
                ["run_study.py", "specification.json", "data/input_manifest.json"]),
            "output_directory_writable": os.access(BASE, os.W_OK)}


def installation_help():
    requirements = BASE / "requirements-notebook.txt"
    return ("在当前 Notebook 的 Code 单元格运行 / Run in a notebook Code cell:\n"
            "import subprocess, sys\n"
            "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', "
            + repr(str(requirements)) + "])\n"
            "安装完成后重启内核；若采用 Anaconda base，推荐先创建独立环境。\n"
            "Restart the kernel after installation; a separate environment is recommended.")


def temporary_kernel_directory(directory):
    """Create a session-local kernel spec; never alter global or base kernels."""
    root = Path(directory)
    kernel = root / "kernels" / KERNEL_NAME
    kernel.mkdir(parents=True, exist_ok=True)
    spec = {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "UTSG7394 (project Python)", "language": "python",
            "env": {"UTSG7394_PROJECT_DIRECTORY": str(BASE)}}
    (kernel / "kernel.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    return root


def require_environment(include_notebook=True, strict_versions=False):
    details = inspect_environment(include_notebook)
    print("项目路径 / Project:", BASE, flush=True)
    print("Python:", details["python"], flush=True)
    print("解释器 / Interpreter:", details["executable"], flush=True)
    if not details["project_complete"]:
        raise RuntimeError("项目不完整：请解压整个项目，不能只复制 Notebook。 / Extract the complete project.")
    if not details["python_minimum_satisfied"]:
        raise RuntimeError("固定依赖需要 Python >=3.11；请按 environment.yml 使用已测试的 Python 3.13。")
    if not details["output_directory_writable"]:
        raise RuntimeError("项目目录不可写：请将完整项目移动到有写入权限的目录。")
    if details["missing"]:
        raise RuntimeError("缺少依赖 / Missing packages: " + ", ".join(details["missing"])
                           + "\n" + installation_help())
    if not details["python_minor_tested"]:
        print("提示：本研究完整验证使用 Python 3.13；当前 Python 版本尚未验证。", flush=True)
        if strict_versions:
            raise RuntimeError("严格版本预检要求已测试的 Python 3.13。请使用 environment.yml 创建环境。")
    if details["version_differences"]:
        print("当前依赖与测试版本不同；计算仍会进行数值核对：", flush=True)
        for item in details["version_differences"]:
            print(f"  {item['package']}: {item['installed']} (tested {item['tested']})", flush=True)
        if strict_versions:
            raise RuntimeError("依赖版本与 requirements-tested.txt 不一致。\n" + installation_help())
    print("环境预检通过 / Environment preflight passed.", flush=True)
    return details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Inspect the environment without starting a server")
    parser.add_argument("--strict-versions", action="store_true", help="Require tested direct package versions")
    parser.add_argument("--json", action="store_true", help="Print preflight metadata as JSON; does not start Jupyter")
    args = parser.parse_args(argv)
    try:
        if args.json:
            details = inspect_environment()
            print(json.dumps(details, ensure_ascii=False, indent=2))
            passed = (details["project_complete"] and details["python_minimum_satisfied"]
                      and details["output_directory_writable"] and not details["missing"])
            if args.strict_versions:
                passed = passed and not details["version_differences"] and details["python_minor_tested"]
            return 0 if passed else 1
        require_environment(strict_versions=args.strict_versions)
        if args.check:
            return 0
        print("浏览器打开后选择 Run_Study.ipynb；内核使用 UTSG7394 (project Python)。", flush=True)
        print("启动器仅为本次会话建立临时内核配置，不更改全局或 Anaconda base。", flush=True)
        with tempfile.TemporaryDirectory(prefix="utsg7394_kernel_") as directory:
            kernel_data = temporary_kernel_directory(directory)
            env = dict(os.environ)
            existing = env.get("JUPYTER_PATH", "")
            env["JUPYTER_PATH"] = str(kernel_data) + (os.pathsep + existing if existing else "")
            return subprocess.call([sys.executable, "-m", "notebook", "--notebook-dir", str(BASE)],
                                   cwd=BASE, env=env)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
