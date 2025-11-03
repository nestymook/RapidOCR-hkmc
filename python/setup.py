# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path
from typing import List, Union

import setuptools

try:
    from get_pypi_latest_version import GetPyPiLatestVersion
    has_version_tool = True
except ImportError:
    has_version_tool = False


def read_txt(txt_path: Union[Path, str]) -> List[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        data = [v.rstrip("\n") for v in f]
    return data


def get_readme():
    try:
        root_dir = Path(__file__).resolve().parent.parent
        readme_path = str(root_dir / "docs" / "doc_whl_rapidocr.md")
        print(readme_path)
        with open(readme_path, "r", encoding="utf-8") as f:
            readme = f.read()
        return readme
    except Exception:
        return "Awesome OCR Library with NPU/GPU Support"


MODULE_NAME = "rapidocr_hkmc"

if has_version_tool:
    obtainer = GetPyPiLatestVersion()
    try:
        latest_version = obtainer(MODULE_NAME)
    except Exception as e:
        latest_version = "0.0.0"
    VERSION_NUM = obtainer.version_add_one(latest_version, add_patch=True)

    if len(sys.argv) > 2:
        match_str = " ".join(sys.argv[2:])
        matched_versions = obtainer.extract_version(match_str)
        if matched_versions:
            VERSION_NUM = matched_versions
    sys.argv = sys.argv[:2]
else:
    # Default version when get_pypi_latest_version is not available
    VERSION_NUM = "1.0.0"

project_urls = {
    "Documentation": "https://rapidai.github.io/RapidOCRDocs",
    "Changelog": "https://github.com/RapidAI/RapidOCR/releases",
}

setuptools.setup(
    name=MODULE_NAME,
    version=VERSION_NUM,
    platforms="Any",
    description="Awesome OCR Library with NPU/GPU Support",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    author="SWHL",
    author_email="liekkaskono@163.com",
    url="https://github.com/RapidAI/RapidOCR",
    project_urls=project_urls,
    license="Apache-2.0",
    include_package_data=True,
    install_requires=read_txt("requirements.txt"),
    packages=[MODULE_NAME] + [
        f"{MODULE_NAME}.{pkg}"
        for pkg in setuptools.find_packages(where=MODULE_NAME)
    ],
    package_data={"": ["*.onnx", "*.yaml", "*.txt"]},
    keywords=[
        "ocr,text_detection,text_recognition,db,onnxruntime,paddleocr,openvino,rapidocr,npu,gpu"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.6,<4",
    entry_points={
        "console_scripts": [f"{MODULE_NAME}={MODULE_NAME}.main:main"],
    },
)
