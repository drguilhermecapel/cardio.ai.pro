#!/usr/bin/env python3
"""
Build script for creating standalone Windows frontend of CardioAI Pro.
Builds the React application and prepares it for bundling.
"""

import os
import sys
import shutil
import subprocess
import time
import traceback
import logging
import argparse
from pathlib import Path
from contextlib import contextmanager

NODE_CMD = "node"
NPM_CMD = "npm"


class BuildError(Exception):
    """Custom exception with detailed context."""

    def __init__(self, message, phase=None, details=None):
        self.phase = phase
        self.details = details or {}
        super().__init__(message)


@contextmanager
def build_phase(name, progress_callback=None):
    """Context manager for build phases with detailed error tracking."""
    start_time = time.time()
    try:
        logger.info(f"Starting phase: {name}")
        print(f"🔄 {name}...")
        if progress_callback:
            progress_callback(f"Phase: {name}", 0)
        yield
        duration = time.time() - start_time
        logger.info(f"Completed phase: {name} ({duration:.2f}s)")
        print(f"✅ {name} completed ({duration:.2f}s)")
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Failed phase: {name} after {duration:.2f}s")
        logger.error(traceback.format_exc())

        diagnostics = {
            "phase": name,
            "duration": duration,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "working_directory": os.getcwd(),
            "node_version": get_node_version(),
            "npm_version": get_npm_version(),
            "environment_vars": dict(os.environ),
        }

        save_diagnostic_snapshot(diagnostics)

        raise BuildError(
            f"Build failed at phase: {name}", phase=name, details=diagnostics
        )


def save_diagnostic_snapshot(diagnostics):
    """Save diagnostic information to file for debugging."""
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        snapshot_file = f"frontend_build_error_snapshot_{timestamp}.log"

        with open(snapshot_file, "w") as f:
            f.write("=" * 60 + "\n")
            f.write(f"FRONTEND BUILD ERROR DIAGNOSTIC SNAPSHOT\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

            for key, value in diagnostics.items():
                f.write(f"{key.upper()}:\n")
                f.write(f"{value}\n\n")

        print(f"📋 Diagnostic snapshot saved: {snapshot_file}")
        logger.info(f"Diagnostic snapshot saved: {snapshot_file}")
    except Exception as e:
        logger.error(f"Failed to save diagnostic snapshot: {e}")


def setup_logging(debug_mode=False):
    """Configure comprehensive logging."""
    log_level = logging.DEBUG if debug_mode else logging.INFO

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / "build_frontend.log"),
            (
                logging.FileHandler(logs_dir / "build_errors.log", mode="a")
                if not debug_mode
                else logging.NullHandler()
            ),
            logging.StreamHandler() if debug_mode else logging.NullHandler(),
        ],
    )

    return logging.getLogger(__name__)


def get_node_version():
    """Get Node.js version safely."""
    try:
        result = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except:
        return "Not available"


def detect_npm_command():
    """Detect npm command with support for portable installations."""
    portable_node_dir = Path(__file__).parent / "portable_node"
    portable_npm = portable_node_dir / "npm.cmd"

    if portable_npm.exists():
        return str(portable_npm)
    return "npm"


def get_npm_version():
    """Get npm version safely."""
    try:
        npm_cmd = detect_npm_command()
        result = subprocess.run(
            [npm_cmd, "--version"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except:
        return "Not available"


def check_environment():
    """Comprehensive environment validation."""
    checks = []

    try:
        result = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, check=True
        )
        node_version = result.stdout.strip()

        version_num = int(node_version.replace("v", "").split(".")[0])
        if version_num < 16:
            checks.append(
                (
                    "Node.js Version",
                    "FAIL",
                    f"Node.js 16+ required, found {node_version}",
                )
            )
        else:
            checks.append(("Node.js Version", "PASS", f"Node.js {node_version}"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        checks.append(
            ("Node.js", "FAIL", "Node.js not found - install from https://nodejs.org/")
        )

    try:
        npm_cmd = detect_npm_command()
        result = subprocess.run(
            [npm_cmd, "--version"], capture_output=True, text=True, check=True
        )
        npm_version = result.stdout.strip()
        checks.append(("npm", "PASS", f"npm {npm_version}"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        checks.append(("npm", "FAIL", "npm not found - usually comes with Node.js"))

    if not Path("../frontend").exists():
        checks.append(
            (
                "Frontend Directory",
                "FAIL",
                "Frontend directory not found at ../frontend",
            )
        )
    else:
        checks.append(("Frontend Directory", "PASS", "Frontend directory found"))

    frontend_dir = Path("../frontend")
    if frontend_dir.exists() and not (frontend_dir / "package.json").exists():
        checks.append(
            ("package.json", "FAIL", "package.json not found in frontend directory")
        )
    elif frontend_dir.exists():
        checks.append(("package.json", "PASS", "package.json found"))

    try:
        import shutil

        free_space = shutil.disk_usage(".").free / (1024**3)  # GB
        if free_space < 1:
            checks.append(
                (
                    "Disk Space",
                    "FAIL",
                    f"Insufficient disk space: {free_space:.1f}GB (1GB required)",
                )
            )
        else:
            checks.append(("Disk Space", "PASS", f"Available: {free_space:.1f}GB"))
    except Exception as e:
        checks.append(("Disk Space", "WARNING", f"Could not check disk space: {e}"))

    try:
        test_file = Path("test_write_permissions.tmp")
        test_file.write_text("test")
        test_file.unlink()
        checks.append(("Write Permissions", "PASS", "Write permissions verified"))
    except Exception as e:
        checks.append(("Write Permissions", "FAIL", f"No write permissions: {e}"))

    print("\n" + "=" * 50)
    print("FRONTEND ENVIRONMENT VALIDATION REPORT")
    print("=" * 50)

    for name, status, message in checks:
        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_icon} {name}: {message}")

    print("=" * 50 + "\n")

    critical_failures = [check for check in checks if check[1] == "FAIL"]
    if critical_failures:
        print("❌ CRITICAL VALIDATION FAILURES DETECTED:")
        for name, _, message in critical_failures:
            print(f"   • {name}: {message}")
        print("\nPlease resolve these issues before continuing.")
        return False

    return True


def check_node_npm():
    """Check if Node.js and npm are available with detailed validation."""
    with build_phase("Node.js and npm Verification"):
        print("Checking Node.js and npm availability...")

        portable_node_dir = Path(__file__).parent / "portable_node"
        portable_node = portable_node_dir / "node.exe"

        node_cmd = "node"
        if portable_node.exists():
            print(f"✅ Using portable Node.js: {portable_node}")
            node_cmd = str(portable_node)

        npm_cmd = detect_npm_command()

        try:
            result = subprocess.run(
                [node_cmd, "--version"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            node_version = result.stdout.strip()
            print(f"✅ Node.js found: {node_version}")
            logger.info(f"Node.js version: {node_version}")

            version_num = int(node_version.replace("v", "").split(".")[0])
            if version_num < 16:
                print(f"❌ ERROR: Node.js version {node_version} is too old")
                print("SOLUTION: Install Node.js 16 or higher from https://nodejs.org/")
                return False

            result = subprocess.run(
                [npm_cmd, "--version"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            npm_version = result.stdout.strip()
            print(f"✅ npm found: {npm_version}")
            logger.info(f"npm version: {npm_version}")

            global NODE_CMD, NPM_CMD
            NODE_CMD = node_cmd
            NPM_CMD = npm_cmd

            return True

        except subprocess.TimeoutExpired:
            print("❌ ERROR: Node.js/npm check timed out")
            print("SOLUTIONS:")
            print("1. Check if Node.js is properly installed")
            print("2. Restart your terminal/command prompt")
            print("3. Add Node.js to your PATH environment variable")
            return False

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"❌ ERROR: Node.js or npm not found: {e}")
            print("SOLUTIONS:")
            print("1. Install Node.js from https://nodejs.org/")
            print("2. Make sure Node.js is added to your PATH")
            print("3. Restart your terminal after installation")
            print("4. Try running: node --version")
            return False

        except Exception as e:
            print(f"❌ ERROR: Unexpected error checking Node.js/npm: {e}")
            logger.error(f"Node.js/npm check error: {e}\n{traceback.format_exc()}")
            return False


def install_dependencies_with_progress():
    """Install frontend dependencies with detailed progress tracking and error handling."""
    with build_phase("Frontend Dependencies Installation"):
        print("📦 Installing frontend dependencies...")
        print("This may take 5-10 minutes depending on your internet connection...")

        frontend_dir = Path(__file__).parent.parent / "frontend"
        os.chdir(frontend_dir)
        print(f"📁 Working in: {frontend_dir}")

        package_manager = determine_package_manager(frontend_dir)

        try:
            if package_manager == "npm":
                install_with_npm()
            elif package_manager == "yarn":
                install_with_yarn()
            elif package_manager == "pnpm":
                install_with_pnpm()
            else:
                print("Using npm (default)...")
                install_with_npm()

            print("✅ Dependencies installed successfully")

        except subprocess.TimeoutExpired:
            print("❌ ERROR: Dependency installation timed out after 10 minutes")
            print("SOLUTIONS:")
            print("1. Check your internet connection")
            print("2. Clear npm cache: npm cache clean --force")
            print("3. Delete node_modules and try again")
            print(
                "4. Try using a different npm registry: npm config set registry https://registry.npmjs.org/"
            )
            raise BuildError("Dependency installation timeout", phase="dependencies")

        except subprocess.CalledProcessError as e:
            print(
                f"❌ ERROR: Failed to install dependencies (exit code: {e.returncode})"
            )
            print("SOLUTIONS:")
            print("1. Check your internet connection")
            print("2. Clear package manager cache")
            print("3. Delete node_modules and package-lock.json, then try again")
            print("4. Check if package.json is valid")
            print("5. Try: npm install --legacy-peer-deps")
            raise BuildError("Dependency installation failed", phase="dependencies")

        except Exception as e:
            print(f"❌ ERROR: Unexpected error during dependency installation: {e}")
            logger.error(
                f"Dependency installation error: {e}\n{traceback.format_exc()}"
            )
            raise BuildError(
                "Unexpected dependency installation error", phase="dependencies"
            )


def determine_package_manager(frontend_dir):
    """Determine which package manager to use based on lock files."""
    if (frontend_dir / "pnpm-lock.yaml").exists():
        return "pnpm"
    elif (frontend_dir / "yarn.lock").exists():
        return "yarn"
    elif (frontend_dir / "package-lock.json").exists():
        return "npm"
    else:
        return "npm"  # default


def install_with_npm():
    """Install dependencies using npm with progress monitoring."""
    print("Using npm (package-lock.json found or default)...")

    process = subprocess.Popen(
        [NPM_CMD, "install"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    start_time = time.time()
    timeout = 600  # 10 minutes

    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            if any(
                keyword in output.lower()
                for keyword in ["warn", "error", "installing", "added", "updated"]
            ):
                print(f"   {output.strip()}")
            logger.debug(f"npm install output: {output.strip()}")

        if time.time() - start_time > timeout:
            process.terminate()
            raise subprocess.TimeoutExpired("npm install", timeout)

    return_code = process.poll()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, "npm install")


def install_with_yarn():
    """Install dependencies using yarn with fallback to npm."""
    print("Using yarn (yarn.lock found)...")

    try:
        subprocess.run(
            ["yarn", "--version"], check=True, capture_output=True, timeout=10
        )

        process = subprocess.Popen(
            ["yarn", "install"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        start_time = time.time()
        timeout = 600  # 10 minutes

        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(f"   {output.strip()}")
                logger.debug(f"yarn install output: {output.strip()}")

            if time.time() - start_time > timeout:
                process.terminate()
                raise subprocess.TimeoutExpired("yarn install", timeout)

        return_code = process.poll()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, "yarn install")

    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Yarn not found or failed, falling back to npm...")
        install_with_npm()


def install_with_pnpm():
    """Install dependencies using pnpm with fallback to npm."""
    print("Using pnpm (pnpm-lock.yaml found)...")

    try:
        subprocess.run(
            ["pnpm", "--version"], check=True, capture_output=True, timeout=10
        )

        subprocess.run(["pnpm", "install"], check=True, timeout=600)

    except (FileNotFoundError, subprocess.CalledProcessError):
        print("pnpm not found or failed, falling back to npm...")
        install_with_npm()


def install_dependencies():
    """Legacy function - redirects to new implementation."""
    install_dependencies_with_progress()


def create_production_env():
    """Create production environment configuration."""
    print("Creating production environment configuration...")

    frontend_dir = Path(__file__).parent.parent / "frontend"
    env_file = frontend_dir / ".env.production"

    env_content = """# Production environment for standalone Windows build
VITE_API_URL=http://localhost:8000
VITE_APP_TITLE=CardioAI Pro
VITE_APP_VERSION=1.0.0
VITE_ENVIRONMENT=production
"""

    with open(env_file, "w") as f:
        f.write(env_content)

    print(f"Created production environment file: {env_file}")


def build_frontend_with_progress():
    """Build the frontend for production with detailed progress tracking."""
    with build_phase("Frontend Production Build"):
        print("🔨 Building frontend for production...")
        print("This may take 3-5 minutes...")

        frontend_dir = Path(__file__).parent.parent / "frontend"
        os.chdir(frontend_dir)
        print(f"📁 Working in: {frontend_dir}")

        create_production_env()

        package_manager = determine_package_manager(frontend_dir)

        try:
            if package_manager == "npm":
                build_with_npm()
            elif package_manager == "yarn":
                build_with_yarn()
            elif package_manager == "pnpm":
                build_with_pnpm()
            else:
                build_with_npm()

            dist_dir = frontend_dir / "dist"
            if not dist_dir.exists():
                raise BuildError(
                    "Build output directory 'dist' not found",
                    phase="build_verification",
                )

            essential_files = ["index.html"]
            missing_files = []
            for file in essential_files:
                if not (dist_dir / file).exists():
                    missing_files.append(file)

            if missing_files:
                print(f"⚠️ WARNING: Missing essential files in build: {missing_files}")
                logger.warning(f"Missing essential files: {missing_files}")

            build_size = sum(
                f.stat().st_size for f in dist_dir.rglob("*") if f.is_file()
            ) / (
                1024 * 1024
            )  # MB
            print(f"✅ Frontend built successfully ({build_size:.1f} MB)")

        except subprocess.TimeoutExpired:
            print("❌ ERROR: Frontend build timed out after 15 minutes")
            print("SOLUTIONS:")
            print("1. Check available disk space")
            print("2. Close other applications to free memory")
            print("3. Try building with --verbose flag for more details")
            print("4. Check if there are TypeScript errors: npm run type-check")
            raise BuildError("Build timeout", phase="frontend_build")

        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR: Frontend build failed (exit code: {e.returncode})")
            print("SOLUTIONS:")
            print("1. Check the error messages above")
            print("2. Verify all dependencies are installed: npm install")
            print("3. Check for TypeScript errors: npm run type-check")
            print("4. Try: npm run build -- --verbose")
            print("5. Clear build cache and try again")
            raise BuildError("Frontend build failed", phase="frontend_build")

        except Exception as e:
            print(f"❌ ERROR: Unexpected error during frontend build: {e}")
            logger.error(f"Frontend build error: {e}\n{traceback.format_exc()}")
            raise BuildError("Unexpected frontend build error", phase="frontend_build")


def build_with_npm():
    """Build frontend using npm with progress monitoring."""
    print("Building with npm...")

    process = subprocess.Popen(
        [NPM_CMD, "run", "build"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    start_time = time.time()
    timeout = 900  # 15 minutes

    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            if any(
                keyword in output.lower()
                for keyword in ["building", "compiled", "error", "warning", "done"]
            ):
                print(f"   {output.strip()}")
            logger.debug(f"npm build output: {output.strip()}")

        if time.time() - start_time > timeout:
            process.terminate()
            raise subprocess.TimeoutExpired("npm run build", timeout)

    return_code = process.poll()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, "npm run build")


def build_with_yarn():
    """Build frontend using yarn with fallback to npm."""
    print("Building with yarn...")

    try:
        subprocess.run(
            ["yarn", "--version"], check=True, capture_output=True, timeout=10
        )

        process = subprocess.Popen(
            ["yarn", "build"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        start_time = time.time()
        timeout = 900  # 15 minutes

        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(f"   {output.strip()}")
                logger.debug(f"yarn build output: {output.strip()}")

            if time.time() - start_time > timeout:
                process.terminate()
                raise subprocess.TimeoutExpired("yarn build", timeout)

        return_code = process.poll()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, "yarn build")

    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Yarn not found or failed, falling back to npm...")
        build_with_npm()


def build_with_pnpm():
    """Build frontend using pnpm with fallback to npm."""
    print("Building with pnpm...")

    try:
        subprocess.run(
            ["pnpm", "--version"], check=True, capture_output=True, timeout=10
        )

        subprocess.run(["pnpm", "run", "build"], check=True, timeout=900)

    except (FileNotFoundError, subprocess.CalledProcessError):
        print("pnpm not found or failed, falling back to npm...")
        build_with_npm()


def build_frontend():
    """Legacy function - redirects to new implementation."""
    build_frontend_with_progress()


def copy_build_files():
    """Copy built frontend files to installer directory."""
    print("Copying built frontend files...")

    frontend_dir = Path(__file__).parent.parent / "frontend"
    installer_dir = Path(__file__).parent

    dist_dir = frontend_dir / "dist"
    frontend_build_dir = installer_dir / "frontend_build"

    if not dist_dir.exists():
        raise FileNotFoundError(f"Frontend build directory not found: {dist_dir}")

    if frontend_build_dir.exists():
        shutil.rmtree(frontend_build_dir)

    shutil.copytree(dist_dir, frontend_build_dir)
    print(f"Frontend build copied to: {frontend_build_dir}")


def create_frontend_server_script():
    """Create a simple HTTP server script for serving the frontend."""
    print("Creating frontend server script...")

    installer_dir = Path(__file__).parent
    server_script = installer_dir / "serve_frontend.py"

    script_content = '''#!/usr/bin/env python3
"""
Simple HTTP server for serving the CardioAI Pro frontend.
This script serves the built React application.
"""

import os
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
import threading
import time

class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Custom handler to serve React app with proper routing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent / "frontend_build"), **kwargs)
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_GET(self):
        if self.path.startswith('/api/'):
            self.send_error(404, "API endpoint - should be handled by backend")
            return
        
        if '.' not in self.path.split('/')[-1] and self.path != '/':
            self.path = '/index.html'
        
        return super().do_GET()

def open_browser_delayed():
    """Open browser after a short delay."""
    time.sleep(2)
    webbrowser.open('http://localhost:3000')

def main():
    """Start the frontend server."""
    port = 3000
    
    frontend_build = Path(__file__).parent / "frontend_build"
    if not frontend_build.exists():
        print("❌ Frontend build not found!")
        print("Please run build_frontend.py first.")
        sys.exit(1)
    
    print(f"Starting CardioAI Pro frontend server on port {port}...")
    print(f"Serving files from: {frontend_build}")
    print(f"Open your browser to: http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    
    browser_thread = threading.Thread(target=open_browser_delayed)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        server = HTTPServer(('localhost', port), CustomHTTPRequestHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\nShutting down frontend server...")
        server.shutdown()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Build interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
'''

    with open(server_script, "w") as f:
        f.write(script_content)

    print(f"Frontend server script created: {server_script}")


def main():
    """Main build function with comprehensive error handling."""
    parser = argparse.ArgumentParser(
        description="Build CardioAI Pro Frontend for Windows"
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug mode with verbose output",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )
    args = parser.parse_args()

    global logger
    logger = setup_logging(args.debug)

    print("🚀 Building CardioAI Pro Frontend for Windows...")
    print("=" * 50)

    if args.debug:
        print("🐛 DEBUG MODE ENABLED - Verbose output active")
        print(f"🐛 Log level: {args.log_level}")
        print(f"🐛 Node.js version: {get_node_version()}")
        print(f"🐛 npm version: {get_npm_version()}")
        print(f"🐛 Working directory: {os.getcwd()}")
        print("=" * 50)

    try:
        with build_phase("Environment Validation"):
            if not check_environment():
                raise BuildError("Environment validation failed", phase="validation")

        if not check_node_npm():
            raise BuildError("Node.js/npm validation failed", phase="node_validation")

        with build_phase("Production Environment Setup"):
            create_production_env()

        install_dependencies_with_progress()

        build_frontend_with_progress()

        with build_phase("Build Deployment"):
            copy_build_files()

        with build_phase("Server Script Creation"):
            create_frontend_server_script()

        print("\n🎉 Frontend build completed successfully!")
        print("Files created:")
        installer_dir = Path(__file__).parent
        print(f"  - {installer_dir / 'frontend_build'} (directory)")
        print(f"  - {installer_dir / 'serve_frontend.py'}")
        print(f"📋 Build logs available in: logs/")

        if args.debug:
            print(f"🐛 Debug logs written to: logs/build_frontend.log")

    except BuildError as e:
        print(f"\n❌ Build failed at phase: {e.phase}")
        print(f"❌ Error: {e}")
        if e.details:
            print(f"📋 Diagnostic information saved for debugging")
        logger.error(f"Build failed: {e}")
        input("Press Enter to exit...")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Unexpected build failure: {e}")
        logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")

        emergency_diagnostics = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "working_directory": os.getcwd(),
            "node_version": get_node_version(),
            "npm_version": get_npm_version(),
        }
        save_diagnostic_snapshot(emergency_diagnostics)

        input("Press Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Build interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
