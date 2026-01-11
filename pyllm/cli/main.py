"""CLI entry point for PyLLM."""

import argparse
import sys
import logging


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="pyllm",
        description="PyLLM - LLM Inference with Streaming Chat",
    )

    parser.add_argument(
        "--version", "-v",
        action="version",
        version="pyllm 1.8.3",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument(
        "--model", "-m",
        type=str,
        help="Path to model weights",
    )
    serve_parser.add_argument(
        "--host", "-H",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    serve_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    serve_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps", "directml"],
        help="Device to use (default: auto - detects best available)",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    # UI command
    ui_parser = subparsers.add_parser("ui", help="Start the Streamlit UI")
    ui_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8501,
        help="Port for Streamlit (default: 8501)",
    )
    ui_parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="API server URL (default: http://localhost:8000)",
    )

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text from prompt")
    gen_parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to model weights",
    )
    gen_parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Input prompt (or read from stdin)",
    )
    gen_parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )
    gen_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    gen_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps", "directml"],
        help="Device to use (default: auto - detects best available)",
    )

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    chat_parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to model weights",
    )
    chat_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps", "directml"],
        help="Device to use (default: auto - detects best available)",
    )
    chat_parser.add_argument(
        "--system",
        type=str,
        help="System prompt",
    )

    args = parser.parse_args()

    if args.command == "serve":
        run_server(args)
    elif args.command == "ui":
        run_ui(args)
    elif args.command == "generate":
        run_generate(args)
    elif args.command == "chat":
        run_chat(args)
    else:
        parser.print_help()
        sys.exit(1)


def run_server(args):
    """Run the API server."""
    import uvicorn
    from pyllm.core.config import Config, ModelConfig, ServerConfig
    from pyllm.api.routes import create_app

    # Configure
    config = Config()
    config.model.path = args.model
    config.model.device = args.device
    config.server.host = args.host
    config.server.port = args.port

    print(f"""
╔═══════════════════════════════════════════════════════╗
║                    PyLLM Server                        ║
║           LLM Inference with Streaming                 ║
╠═══════════════════════════════════════════════════════╣
║  API:       http://{args.host}:{args.port}                     ║
║  Docs:      http://{args.host}:{args.port}/docs                ║
║  Model:     {(args.model or 'Not loaded')[:40]:<40} ║
║  Device:    {args.device:<40} ║
╚═══════════════════════════════════════════════════════╝
    """)

    app = create_app(config)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def run_ui(args):
    """Run the Streamlit UI."""
    import subprocess
    import os

    ui_path = os.path.join(os.path.dirname(__file__), "..", "ui", "app.py")

    if not os.path.exists(ui_path):
        print("Error: UI not found")
        sys.exit(1)

    print(f"""
╔═══════════════════════════════════════════════════════╗
║                    PyLLM Chat UI                       ║
╠═══════════════════════════════════════════════════════╣
║  URL:      http://localhost:{args.port}                       ║
║  API:      {args.api_url}                       ║
╚═══════════════════════════════════════════════════════╝
    """)

    os.environ["PYLLM_API_URL"] = args.api_url

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        ui_path,
        "--server.port", str(args.port),
        "--server.headless", "true",
    ])


def run_generate(args):
    """Run text generation."""
    from pyllm.core.config import ModelConfig
    from pyllm.inference.engine import InferenceEngine, GenerationConfig

    # Get prompt
    prompt = args.prompt
    if not prompt:
        print("Enter prompt (Ctrl+D to submit):")
        prompt = sys.stdin.read().strip()

    if not prompt:
        print("Error: No prompt provided")
        sys.exit(1)

    # Load model
    print(f"Loading model from {args.model}...")
    config = ModelConfig(path=args.model, device=args.device)
    engine = InferenceEngine(config)
    engine.load()

    # Generate
    print("\nGenerating...\n")
    print("-" * 50)

    gen_config = GenerationConfig(
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
    )

    for token in engine.generate(prompt, gen_config):
        print(token, end="", flush=True)

    print("\n" + "-" * 50)


def run_chat(args):
    """Run interactive chat."""
    from pyllm.core.config import ModelConfig
    from pyllm.inference.engine import InferenceEngine, GenerationConfig, Message
    from pyllm.inference.templates import ChatTemplate

    # Load model
    print(f"Loading model from {args.model}...")
    config = ModelConfig(path=args.model, device=args.device)
    engine = InferenceEngine(config)
    engine.load()

    print("""
╔═══════════════════════════════════════════════════════╗
║                  PyLLM Interactive Chat                ║
║              Type 'quit' or 'exit' to stop             ║
╚═══════════════════════════════════════════════════════╝
    """)

    messages = []
    template = ChatTemplate()

    if args.system:
        messages.append(Message(role="system", content=args.system))
        print(f"System: {args.system}\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            messages.append(Message(role="user", content=user_input))

            print("Assistant: ", end="", flush=True)

            response_tokens = []
            for token in engine.chat(messages):
                print(token, end="", flush=True)
                response_tokens.append(token)

            print("\n")

            response = "".join(response_tokens)
            messages.append(Message(role="assistant", content=response))

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            break


if __name__ == "__main__":
    main()
