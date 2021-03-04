#!/usr/bin/env python
import ast
from pathlib import Path
import pkg_resources
# noinspection PyUnresolvedReferences
from pkg_resources.extern import packaging
import setuptools
from typing import cast, Generator, Iterable, List, Optional, Tuple, Union
import urllib.parse


###############################################################################
# Some helper functions

def parse_version(fpath: Union[str, Path]) -> str:
    """
    Statically parse the "__version__" number string from a python file.

    TODO: Auto-append dev version based on how forward from latest release
          Basically a simpler version of what setuptools_scm does but without
          the added cruft and bringing the ENTIRE git repo in with the dist
          See: https://github.com/pypa/setuptools_scm/blob/master/setuptools_scm/version.py
          Would need to know number of commits ahead from last version tag.
    """
    with open(fpath, 'r') as file_:
        pt = ast.parse(file_.read())

    class VersionVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.version: Optional[str] = None

        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__version__":
                    self.version = cast(ast.Str, node.value).s

    visitor = VersionVisitor()
    visitor.visit(pt)
    if visitor.version is None:
        raise RuntimeError("Failed to find __version__!")
    return visitor.version


def parse_req_strip_version(filepath: Union[str, Path]) -> List[str]:
    """
    Read requirements file and return the list of requirements specified
    therein but with their version aspects striped.

    We strictly strip version requirements here as we are adhering to the
    recommended purpose of requirements files: the listing of pinned
    requirement versions for the purpose of achieving repeatable installations.
    To that extent, we expect any packages that require minimum runtime
    requirement versioning to detail that in the `install_requires` section of
    the below `setup` function.

    See pkg_resources.Requirement docs here:
        https://setuptools.readthedocs.io/en/latest/pkg_resources.html#requirement-objects
    """
    filepath = Path(filepath)
    # Known prefixes of lines that are definitely not requirements
    # specifications.
    skip_prefix_tuple = (
        "#", "--index-url"
    )

    def _filter_req_lines(_filepath: Union[str, Path]) -> Generator[Tuple[str, str], None, None]:
        """ Filter lines from file that are requirements, also splitting out
        environment markers from appropriate lines.

        Environment markers string may be empty if there are none for a
        requirement.
        """
        with open(_filepath, 'r') as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith(skip_prefix_tuple):
                    # Empty or has a skippable prefix.
                    continue
                elif _line.startswith('-r '):
                    # sub-requirements file specification, yield that file's
                    # req lines.
                    # ! Requirements/pip does not seem to support requirement
                    # specifiers on nested requirements file includes.
                    target = _filepath.parent / _line.split(" ")[1]
                    for req, env_markers in _filter_req_lines(target):
                        yield req, env_markers
                elif _line.startswith('-e '):
                    # Indicator for URL-based requirement. Look to the egg
                    # fragment.
                    frag = urllib.parse.urlparse(_line.split(' ')[1]).fragment
                    try:
                        egg = dict(
                            cast(Tuple[str, str], part.split('=', 1))
                            for part in frag.split('&')
                            if part  # handle no fragments
                        )['egg']
                    except KeyError:
                        raise packaging.requirements.InvalidRequirement(
                            f"Failed to parse egg name from the requirements "
                            f"line: '{_line}'"
                        )
                    # requirements/pip does not seem to support requirement
                    # specifiers on URLs.
                    yield egg, ""
                else:
                    # Separate out platform deps if any are present, signified
                    # by a semi-colon
                    req = _line
                    env_markers = ""
                    if ';' in _line:
                        req, env_markers = map(str.strip, _line.rsplit(';', 1))
                    yield req, env_markers

    def _strip_req_specifier(
        req_iter: Iterable[pkg_resources.Requirement]
    ) -> Generator[pkg_resources.Requirement, None, None]:
        """
        Modify requirements objects to null out the specifier component.
        """
        for r in req_iter:
            r.specs = []
            # `specifier` property is defined in extern base-class of the
            # `pkg_resources.Requirement` type.
            # noinspection PyTypeHints
            r.specifier = packaging.specifiers.SpecifierSet()  # type: ignore
            yield r

    reqs, markers = zip(*_filter_req_lines(filepath))
    reqs_stripped = list(_strip_req_specifier(pkg_resources.parse_requirements(reqs)))
    return [
        f"{str(req)}{f'; {marker}' if marker else ''}"
        for req, marker in zip(reqs_stripped, markers)
    ]


def ep_repeat(s: str) -> str:
    """
    Simple helper function that will repeat `s` in the format: f"{s} = {s}".

    This is a simple helper function to shorted the redundant typing commonly
    used in the entry-points section when registering plugins.

    :param s: String to repeat.
    :return: Resultant string.
    """
    return f"{s} = {s}"


################################################################################

PACKAGE_NAME = "smqtk_indexing"
SETUP_DIR = Path(__file__).parent

with open(SETUP_DIR / "README.md") as f:
    LONG_DESCRIPTION = f.read()

VERSION = parse_version(SETUP_DIR / PACKAGE_NAME / "__init__.py")


if __name__ == "__main__":
    setuptools.setup(
        name=PACKAGE_NAME,
        version=VERSION,
        description=(
            "Algorithms, data structures and utilities around computing "
            "descriptor k-nearest-neighbors."
        ),
        long_description=LONG_DESCRIPTION,
        author='Kitware, Inc.',
        author_email='smqtk-developers@kitware.com',
        url='https://github.com/Kitware/SMQTK-Indexing',
        license='BSD 3-Clause',
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: Unix',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
        platforms=[
            'Linux',
            'Mac OS-X',
            'Unix',
            # 'Windows',  # Not tested yet
        ],

        packages=setuptools.find_packages(include=[f'{PACKAGE_NAME}*']),
        package_data={PACKAGE_NAME: ["py.typed"]},
        # Required for mypy to be able to find the installed package.
        # https://mypy.readthedocs.io/en/latest/installed_packages.html#installed-packages
        zip_safe=False,

        install_requires=parse_req_strip_version(SETUP_DIR / "requirements" / "runtime.txt"),
        extras_require={
            'ci': parse_req_strip_version(SETUP_DIR / "requirements" / "ci.txt"),
            'docs': parse_req_strip_version(SETUP_DIR / "requirements" / "docs.txt"),
            'test': parse_req_strip_version(SETUP_DIR / "requirements" / "test.txt"),
        },

        entry_points={
            'smqtk_plugins': {
            }
        }
    )
