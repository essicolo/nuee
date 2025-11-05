1. update pyproject.toml and src/nuee/__init__.py __version__ = "0.1.3" to the new tag
2. run
git commit -am "Release v0.1.3"
git tag -s v0.1.3 -m "Release v0.1.3"
git push
git push origin v0.1.3
