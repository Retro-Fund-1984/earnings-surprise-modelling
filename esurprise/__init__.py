import sys

try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages of sklearn when
    # the binaries are not built
    # mypy error: Cannot determine type of '__SKLEARN_SETUP__'
    __ESURPRISE_SETUP__  # type: ignore
except NameError:
    __ESURPRISE_SETUP__ = False

if __ESURPRISE_SETUP__:
    sys.stderr.write("Partial import of imblearn during the build process.\n")
    # We are not importing the rest of scikit-learn during the build
    # process, as it may not be compiled yet
else:
    from . import data
    from ._version import __version__

    __all__ = ["data", "__version__"]
