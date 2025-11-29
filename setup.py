from setuptools import setup, find_packages

setup(
    name="tti_lab",
    version="0.1.0",
    description="TTI Lab — computational tools for informational simulations",
    packages=find_packages(),
    install_requires=[
        # Keep small here; rely on requirements.txt for full env.
    ],
    include_package_data=True,
)