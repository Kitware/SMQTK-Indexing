Pending Release Notes
=====================

Updates / New Features
----------------------

* Updated supported Python versions to 3.10 through 3.14.

* Updated CI to use ``actions/setup-python`` on Ubuntu 24.04 instead of job
  containers to avoid container registry pull limits.

* Updated NumPy requirements to match the other SMQTK packages and refreshed
  dependency requirements for supported Python versions and security fixes.

CI

* Added workflow to inherit the smqtk-core publish workflow.

* Updated CI unittests workflow to include codecov reporting.
  Reduced CodeCov report submission by skipping this step on scheduled runs.

Miscellaneous

* Added a wrapper script to pull the versioning/changelog update helper from
  smqtk-core to use here without duplication.

Documentation

* Updated CONTRIBUTING.md to reference smqtk-core's CONTRIBUTING.md file.

Fixes
-----

* Explicitly installed the configured Sphinx theme in Read the Docs builds.

* Replaced local worker lambdas with picklable callables in ITQ, simple random
  projection, and MRPT descriptor collection to support Python 3.14's default
  multiprocessing start method.

* Required Tornado 6.6 or newer on Python 3.11 and newer to address security
  advisories in the documentation and notebook dependencies.

CI

* Modified CI unittests workflow to run for PRs targetting branches that match
  the `release*` glob.

Dependency Versions

* Updated the developer dependency and locked version of ipython to address a
  security vulnerability.

* Removed `jedi = "^0.17.2"` requirement since recent `ipython = "^7.17.3"`
  update appropriately addresses the dependency.

* Update pinned jupyter notebook transitive dependency version due to
  vulnerability warning.

Known Issues
------------

* Python 3.10 retains Pillow 12.3.0 and Tornado 6.5.10 because their patched
  releases require Python 3.11. Newer Pillow and Tornado security advisories
  remain unresolved on Python 3.10; Python 3.11 and newer require Pillow 13
  and Tornado 6.6 with these fixes.
