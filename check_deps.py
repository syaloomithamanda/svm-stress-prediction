import sys
import importlib.util
from importlib.metadata import version, PackageNotFoundError

print("Python:", sys.version)
print("Executable:", sys.executable)

spec = importlib.util.find_spec("plotly")
print("Plotly module found:", spec is not None)

try:
    print("Plotly version:", version("plotly"))
    import plotly.express as px
    print("Plotly Express import: OK")
except (PackageNotFoundError, ImportError, Exception) as e:
    print("Plotly import failed:", repr(e))
