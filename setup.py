from setuptools import find_packages, setup
from glob import glob

package_name = "nn_magnetometer"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.yaml")),
        ("share/" + package_name + "/data/paths", glob("data/paths/*.csv")),
        ("share/" + package_name + "/data/models", glob("data/models/*.pth")),
        ("share/" + package_name + "/data/map", glob("data/map/*.csv")),
        
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Aditya Penumarti",
    maintainer_email="apenumarti@ufl.edu",
    description="A package for a neural based magnetometer",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "nn_magnetometer_node = nn_magnetometer.nn_magnetometer:main",
            "path_follower_node = nn_magnetometer.path_follower:main",
        ],
    },
)
