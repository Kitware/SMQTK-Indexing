# SMQTK - Indexing

This package provides interfaces and implementations around the
k-nearest-neighbor algorithm.

This package defines interfaces and implementations around efficient,
large-scale indexing of descriptor vectors.
The sources of such descriptor vectors may come from a multitude of sources,
such as hours of video archives.
Some provided implementation plugins include [Locality-sensitive Hashing
(LSH)](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) and FAIR's
[FAISS] library.

## Documentation
You can build the sphinx documentation locally for the most up-to-date
reference:
```bash
# Install dependencies
poetry install
# Navigate to the documentation root.
cd docs
# Build the docs.
poetry run make html
# Open in your favorite browser!
firefox _build/html/index.html
```
