import importlib.metadata

try:
    VERSION = importlib.metadata.version("pyelastica_rigid")
except importlib.metadata.PackageNotFoundError:
    VERSION = "unknown"
