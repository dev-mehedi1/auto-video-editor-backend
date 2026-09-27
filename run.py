"""Start the API from inside the Backend folder: `python run.py`.

`Backend` is a package (it has __init__.py) and main.py uses relative imports,
so plain `uvicorn Backend.main:app` only works from the repository root. This
wrapper puts the root on sys.path so the command works from either directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
ROOT = BACKEND.parent

# The reload supervisor spawns a fresh interpreter, so both the working
# directory and sys.path have to be set up before uvicorn imports anything.
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def main() -> None:
    import uvicorn

    uvicorn.run(
        "Backend.main:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=True,
        # Watch only Backend; the default would be the repo root and would
        # crawl Frontend/node_modules on every save.
        reload_dirs=[str(BACKEND)],
    )


if __name__ == "__main__":
    main()
