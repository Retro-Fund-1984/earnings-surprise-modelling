"""All minimum dependencies for esurprise."""
import argparse

NUMPY_MIN_VERSION = "1.14.6"
PANDAS_MIN_VERSION = "1.3.5"
SKLEARN_MIN_VERSION = "1.0.0"
MATPLOTLIB_MIN_VERSION = "2.2.3"
XGBOOST_MIN_VERSION = "1.7.5"
MLRESEARCH_MIN_VERSION = "0.4.0"

# The values are (version_spec, comma separated tags)
dependent_packages = {
    "numpy": (NUMPY_MIN_VERSION, "install"),
    "pandas": (PANDAS_MIN_VERSION, "install"),
    "scikit-learn": (SKLEARN_MIN_VERSION, "install"),
    "xgboost": (XGBOOST_MIN_VERSION, "install"),
    "ml-research": (MLRESEARCH_MIN_VERSION, "install"),
    "research-learn": ("0.3.1", "install"),
    "yahooquery": ("2.3.1", "install"),
    "requests": ("2.26.0", "install"),
    "matplotlib": (MATPLOTLIB_MIN_VERSION, "install"),
    "flake8": ("3.8.2", "optional"),
    "black": ("22.3", "optional"),
    "pylint": ("2.12.2", "optional"),
}


# create inverse mapping for setuptools
tag_to_packages: dict = {
    extra: [] for extra in ["install", "optional", "docs", "examples", "tests", "all"]
}
for package, (min_version, extras) in dependent_packages.items():
    for extra in extras.split(", "):
        tag_to_packages[extra].append("{}>={}".format(package, min_version))
    tag_to_packages["all"].append("{}>={}".format(package, min_version))


# Used by CI to get the min dependencies
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get min dependencies for a package")

    parser.add_argument("package", choices=dependent_packages)
    args = parser.parse_args()
    min_version = dependent_packages[args.package][0]
    print(min_version)
