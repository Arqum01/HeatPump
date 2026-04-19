from __future__ import annotations

import os
from typing import Callable


def resolve_main() -> Callable[[], None]:
    """Select the Streamlit app entrypoint for Hugging Face Spaces."""
    mode = os.getenv("SPACE_APP_MODE", "customer").strip().lower()

    if mode == "customer":
        from customer_app import main as app_main

        return app_main

    if mode == "admin":
        from admin_app import main as app_main

        return app_main

    raise ValueError("SPACE_APP_MODE must be 'customer' or 'admin'.")


if __name__ == "__main__":
    resolve_main()()
