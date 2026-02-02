from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import sys
import os

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()


class PostInstallCommand(install):
    """Post-installation command to precompile numba functions."""
    
    def run(self):
        # Run the standard installation first
        install.run(self)
        
        # Precompile numba functions after installation
        print("\n" + "=" * 60)
        print("Precompiling numba functions...")
        print("=" * 60)
        
        try:
            # Get the path to the precompile script
            script_path = os.path.join(os.path.dirname(__file__), 'precompile_numba_functions.py')
            
            # Check if script exists
            if not os.path.exists(script_path):
                print(f"\n⚠ Warning: Precompilation script not found at {script_path}")
                print("Functions will be compiled on first use instead.")
                return
            
            # Run the precompilation script
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=os.path.dirname(__file__),
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                print("\n✓ Numba functions successfully precompiled!")
            else:
                print("\n⚠ Warning: Numba precompilation completed with errors.")
                print("Functions will be compiled on first use instead.")
                
        except Exception as e:
            print(f"\n⚠ Warning: Failed to precompile numba functions: {e}")
            print("Functions will be compiled on first use instead.")
            # Don't fail the installation if precompilation fails


setup(
    name="GalaxyGeniusMSA",
    version="0.1.0",
    description="JWST MSA-3D Data Simulation",
    author="Xingchen Zhou et al.",
    author_email="xczhou95@gmail.com",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/xczhou-astro/galaxyGenius-MSA",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=requirements,
    cmdclass={
        'install': PostInstallCommand,
    },
    classifiers=[
        "Programming Language :: Python :: 3.11.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
