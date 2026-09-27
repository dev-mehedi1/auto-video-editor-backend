"""Allow `python -m Backend` from the repository root."""
from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    from .main import app  # noqa: F401  (imported so errors surface here)

    uvicorn.run(
        "Backend.main:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=True,
        reload_dirs=[str(os.path.dirname(os.path.abspath(__file__)))],
    )


if __name__ == "__main__":
    main()
