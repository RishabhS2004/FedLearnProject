"""
RadioFed Client — Launcher

Starts the FastHTML client UI.
"""

import sys, os, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="RadioFed Client")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--client-id", type=str, default=None)
    args = parser.parse_args()

    from client.app import create_client_app, S, _load_cfg
    import uvicorn

    if args.client_id:
        _load_cfg()
        S.config["client_id"] = args.client_id

    app = create_client_app(port=args.port)
    print(f"RadioFed Client running at http://localhost:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
