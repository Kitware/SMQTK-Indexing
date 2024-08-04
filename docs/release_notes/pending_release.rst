Pending Release Notes
=====================

Updates / New Features
----------------------

CI

* Added workflow to inherit the smqtk-core publish workflow.

* Updated CI unittests workflow to include codecov reporting.
  Reduced CodeCov report submission by skipping this step on scheduled runs.

Dependencies

* Update minimum minimum python to 3.9 to reflect currently and soon-to-be
  deprecated versions of python.

Documentation

* Updated CONTRIBUTING.md to reference smqtk-core's CONTRIBUTING.md file.

Miscellaneous

* Added a wrapper script to pull the versioning/changelog update helper from
  smqtk-core to use here without duplication.

Fixes
-----

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
