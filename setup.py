"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="smartboiler",  # Required
    version="{{VERSION_PLACEHOLDER}}",  # Required
    description="Smart boiling of household",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    url="https://github.com/grinwi/smartboiler",  # Optional
    author="Adam GRUNWALD",  # Optional
    author_email="grunwald.adam24@gmail.com",  # Optional
    classifiers=[  # Optional
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    keywords="energy, management, optimization, hass",  # Optional
    package_dir={"": "src"},  # Optional
    packages=find_packages(where="src"),  # Required
    python_requires=">=3.9, <=3.10",
    install_requires=[
        "geopy",
        "google-api-python-client",
        "google-auth-httplib2",
        "google-auth-oauthlib",
        "haversine",
        "h5py==3.10.0",
        "influxdb==5.3.1",
        "keras==2.15.0",
        "matplotlib==3.5.1",
        "numpy>=1.22.2",
        "pandas>=1.4.1",
        "pytz>=2021.1",
        "requests>=2.25.1",
        "skforecast>=0.10.1",
        "wheel",
    ],
    # Optional
    entry_points={  # Optional
        "console_scripts": [
            "smartboiler=smartboiler.command_line:main",
        ],
    },
    package_data={"smartboiler": ["templates/index.html", "static/style.css"]},
)
