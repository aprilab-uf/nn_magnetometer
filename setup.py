from setuptools import find_packages, setup
from glob import glob

package_name = "magnav_nn_sim"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="basestation",
    maintainer_email="basestation@todo.todo",
    description="TODO: Package description",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["magnav_nn_node = magnav_nn_sim.magnav_nn_node:main"],
    },
)
