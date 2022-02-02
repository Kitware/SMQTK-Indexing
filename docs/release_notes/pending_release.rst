Pending Release Notes
=====================

Updates / New Features
----------------------

CI

* Add workflow to inherit the smqtk-core publish workflow.

Miscellaneous

* Added a wrapper script to pull the versioning/changelog update helper from
  smqtk-core to use here without duplication.

Documentation

* Updated CONTRIBUTING.md to reference smqtk-core's CONTRIBUTING.md file.

Fixes
-----

CI

* Also run CI unittests for PRs targetting branches that match the `release*`
  glob.

Dependency Versions

* Update the developer dependency and locked version of ipython to address a
  security vulnerability.
