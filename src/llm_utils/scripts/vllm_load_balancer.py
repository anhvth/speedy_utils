import argparse
import sys
import os
import time
from datetime import datetime
from collections import defaultdict

import asyncio
import contextlib
import random

import aiohttp
from loguru import logger
from tabulate import tabulate

from speedy_utils import setup_logger

setup_logger(min_interval=5)


# --- CLI Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(
        description="🚀 vLLM Load Balancer - High-Performance Async TCP/HTTP Load Balancer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vllm_load_balancer.py 8001 --ports 8140,8150,8160
  python vllm_load_balancer.py 8080 --ports 8140,8150 --host 192.168.1.100
  python vllm_load_balancer.py 8001 --ports 8140,8150 --status-interval 3
  python vllm_load_balancer.py 8001 --ports 8140,8150 --throttle-ms 10

Features:
  • Real-time dashboard with color-coded status
  • Automatic health checks and failover
  • Least-connections load balancing
  • Request throttling to prevent server overload
  • Professional terminal interface
  • Connection statistics and monitoring
        """,
    )
    parser.add_argument(
        "port",
        type=int,
        help="Port for the load balancer to listen on (e.g., 8001)",
    )
    parser.add_argument(
        "--ports",
        type=str,
        required=True,
        help="Comma-separated list of backend ports to use (e.g., 8140,8150)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Backend host (default: localhost)",
    )
    parser.add_argument(
        "--status-interval",
        type=int,
        default=5,
        help="Status print interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--health-timeout",
        type=int,
        default=2,
        help="Health check timeout in seconds (default: 2)",
    )
    parser.add_argument(
        "--stats-port",
        type=int,
        default=None,
        help="Port for the HTTP stats dashboard (default: proxy port + 1)",
    )
    parser.add_argument(
        "--throttle-ms",
        type=float,
        default=30.0,
        help="Minimum milliseconds between requests to same server (default: 5ms)",
    )
    return parser.parse_args()


# --- Configuration (populated from CLI) ---
LOAD_BALANCER_HOST = "0.0.0.0"
LOAD_BALANCER_PORT = 8008  # Will be overwritten by CLI
STATS_PORT = 8009  # Will be overwritten by CLI
BACKEND_HOST = "localhost"  # Will be overwritten by CLI
BACKEND_PORTS = []  # Will be overwritten by CLI
STATUS_PRINT_INTERVAL = 5
HEALTH_CHECK_TIMEOUT = 2
THROTTLE_MS = 5.0  # Will be overwritten by CLI
BUFFER_SIZE = 4096

# --- Global Shared State ---
available_servers = []
connection_counts = defaultdict(int)
last_send_times = defaultdict(float)  # Track last send time per server
state_lock = asyncio.Lock()
start_time = None
total_connections_served = 0
current_active_connections = 0


# --- Terminal Utilities ---
def clear_terminal():
    """Clear terminal screen with cross-platform support."""
    if os.name == "nt":  # Windows
        os.system("cls")
    else:  # Unix/Linux/MacOS
        os.system("clear")


def get_terminal_size():
    """Get terminal dimensions."""
    try:
        columns, rows = os.get_terminal_size()
        return columns, rows
    except OSError:
        return 80, 24  # Default fallback


def format_uptime(start_time):
    """Format uptime in a human-readable way."""
    if not start_time:
        return "Unknown"

    uptime_seconds = time.time() - start_time
    hours = int(uptime_seconds // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    seconds = int(uptime_seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def print_banner():
    """Print a professional startup banner."""
    columns, _ = get_terminal_size()
    banner_width = min(columns - 4, 80)

    print("=" * banner_width)
    print(f"{'🚀 vLLM Load Balancer':^{banner_width}}")
    print(f"{'High-Performance Async TCP/HTTP Load Balancer':^{banner_width}}")
    print("=" * banner_width)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Load Balancer Port: {LOAD_BALANCER_PORT}")
    print(f"Backend Host: {BACKEND_HOST}")
    print(f"Backend Ports: {', '.join(map(str, BACKEND_PORTS))}")
    print(f"Health Check Interval: 10s (Timeout: {HEALTH_CHECK_TIMEOUT}s)")
    print(f"Status Update Interval: {STATUS_PRINT_INTERVAL}s")
    print(f"Request Throttling: {THROTTLE_MS}ms minimum between requests")
    print("=" * banner_width)
    print()


# --- ANSI Color Codes ---
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


# --- Helper Functions --- (relay_data and safe_close_writer remain the same)
async def relay_data(reader, writer, direction):
    """Reads data from reader and writes to writer until EOF or error."""
    try:
        while True:
            data = await reader.read(BUFFER_SIZE)
            if not data:
                logger.debug(f"EOF received on {direction} stream.")
                break
            writer.write(data)
            await writer.drain()
    except ConnectionResetError:
        logger.warning(f"Connection reset on {direction} stream.")
    except asyncio.CancelledError:
        logger.debug(f"Relay task cancelled for {direction}.")
        raise
    except Exception as e:
        logger.warning(f"Error during data relay ({direction}): {e}")
    finally:
        if not writer.is_closing():
            try:
                writer.close()
                await writer.wait_closed()
                logger.debug(f"Closed writer for {direction}")
            except Exception as close_err:
                logger.debug(
                    f"Error closing writer for {direction} (might be expected): {close_err}"
                )


@contextlib.asynccontextmanager
async def safe_close_writer(writer):
    """Async context manager to safely close an asyncio StreamWriter."""
    try:
        yield writer
    finally:
        if writer and not writer.is_closing():
            try:
                writer.close()
                await writer.wait_closed()
            except Exception as e:
                logger.debug(f"Error closing writer in context manager: {e}")


# --- Health Check for Provided Ports ---
async def check_server_health(session, host, port):
    """Performs an HTTP GET request to the /health endpoint."""
    url = f"http://{host}:{port}/health"
    try:
        async with session.get(url, timeout=HEALTH_CHECK_TIMEOUT) as response:
            if 200 <= response.status < 300:
                logger.debug(
                    f"[{LOAD_BALANCER_PORT=}] Health check success for {url} (Status: {response.status})"
                )
                await response.release()
                return True
            else:
                logger.debug(
                    f"[{LOAD_BALANCER_PORT=}] Health check failed for {url} (Status: {response.status})"
                )
                await response.release()
                return False
    except asyncio.TimeoutError:
        logger.debug(f"Health check HTTP request timeout for {url}")
        return False
    except aiohttp.ClientConnectorError as e:
        logger.debug(f"Health check connection error for {url}: {e}")
        return False
    except aiohttp.ClientError as e:
        logger.warning(f"Health check client error for {url}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected health check error for {url}: {e}")
        return False


async def scan_and_update_servers():
    """Periodically checks the provided backend ports and updates available servers."""
    global available_servers
    logger.debug(
        f"Starting server scan task (HTTP GET /health on ports {BACKEND_PORTS} every 10s)"
    )
    while True:
        try:
            current_scan_results = []
            scan_tasks = []
            ports_to_scan = BACKEND_PORTS

            async with aiohttp.ClientSession() as session:
                for port in ports_to_scan:
                    task = asyncio.create_task(
                        check_server_health(session, BACKEND_HOST, port)
                    )
                    scan_tasks.append((task, port))

                await asyncio.gather(
                    *(task for task, port in scan_tasks), return_exceptions=True
                )

                for task, port in scan_tasks:
                    try:
                        if (
                            task.done()
                            and not task.cancelled()
                            and task.result() is True
                        ):
                            current_scan_results.append((BACKEND_HOST, port))
                    except Exception as e:
                        logger.error(
                            f"Error retrieving health check result for port {port}: {e}"
                        )
            async with state_lock:
                previous_servers = set(available_servers)
                current_set = set(current_scan_results)

                added = current_set - previous_servers
                removed = previous_servers - current_set

                if added:
                    logger.info(
                        f"Servers added (passed /health check): {sorted(list(added))}"
                    )
                if removed:
                    logger.info(
                        f"Servers removed (failed /health check or stopped): {sorted(list(removed))}"
                    )
                    for server in removed:
                        if server in connection_counts:
                            del connection_counts[server]
                            logger.debug(
                                f"Removed connection count entry for unavailable server {server}"
                            )
                        if server in last_send_times:
                            del last_send_times[server]
                            logger.debug(
                                f"Removed throttling timestamp for unavailable server {server}"
                            )

                available_servers = sorted(list(current_set))
                for server in available_servers:
                    if server not in connection_counts:
                        connection_counts[server] = 0

            logger.debug(
                f"[{LOAD_BALANCER_PORT=}]Scan complete. Active servers: {available_servers}"
            )

        except asyncio.CancelledError:
            logger.info("Server scan task cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in scan_and_update_servers loop: {e}")
            await asyncio.sleep(5)  # Avoid tight loop on error

        await asyncio.sleep(10)


# --- Core Load Balancer Logic (handle_client remains the same) ---
async def handle_client(client_reader, client_writer):
    """Handles a single client connection."""
    client_addr = client_writer.get_extra_info("peername")

    backend_server = None
    backend_reader = None
    backend_writer = None
    server_selected = False

    global total_connections_served, current_active_connections
    try:
        # --- Select Backend Server (Least Connections from Available) ---
        selected_server = None
        async with (
            state_lock
        ):  # Lock to safely access available_servers and connection_counts
            if not available_servers:
                logger.warning(
                    f"No backend servers available (failed health checks?) for client {client_addr}. Closing connection."
                )
                async with safe_close_writer(client_writer):
                    pass
                return

            min_connections = float("inf")
            least_used_available_servers = []
            for server in (
                available_servers
            ):  # Iterate only over servers that passed health check
                count = connection_counts.get(server, 0)
                if count < min_connections:
                    min_connections = count
                    least_used_available_servers = [server]
                elif count == min_connections:
                    least_used_available_servers.append(server)

            if least_used_available_servers:
                selected_server = random.choice(least_used_available_servers)
                connection_counts[selected_server] += 1
                backend_server = selected_server
                server_selected = True

                # Update global statistics
                global total_connections_served, current_active_connections
                total_connections_served += 1
                current_active_connections += 1
            else:
                logger.error(
                    f"Logic error: No server chosen despite available servers list not being empty for {client_addr}."
                )
                async with safe_close_writer(client_writer):
                    pass
                return

        # --- Connect to Backend Server ---
        if not backend_server:
            logger.error(
                f"No backend server selected for {client_addr} before connection attempt."
            )
            async with safe_close_writer(client_writer):
                pass
            server_selected = False
            return
        
        # --- Throttling Logic ---
        # Check if we need to throttle requests to avoid overwhelming the backend
        current_time = time.time() * 1000  # Convert to milliseconds
        sleep_time = 0
        async with state_lock:
            last_send_time = last_send_times.get(backend_server, 0)
            time_since_last_send = current_time - last_send_time
            
            if time_since_last_send < THROTTLE_MS:
                sleep_time = (THROTTLE_MS - time_since_last_send) / 1000  # Convert to seconds
                logger.debug(
                    f"Throttling request to {backend_server} for {sleep_time:.3f}s (last send: {time_since_last_send:.1f}ms ago)"
                )
        
        # Sleep outside the lock to avoid blocking other clients
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        
        # Update last send time after throttling
        async with state_lock:
            last_send_times[backend_server] = time.time() * 1000
        
        try:
            logger.debug(
                f"Attempting connection to backend {backend_server} for {client_addr}"
            )
            backend_reader, backend_writer = await asyncio.open_connection(
                backend_server[0], backend_server[1]
            )
            logger.debug(
                f"Successfully connected to backend {backend_server} for {client_addr}"
            )

        # Handle connection failure AFTER selection (server might go down between health check and selection)
        except ConnectionRefusedError:
            logger.error(
                f"Connection refused by selected backend server {backend_server} for {client_addr}"
            )
            async with state_lock:  # Decrement count under lock
                if (
                    backend_server in connection_counts
                    and connection_counts[backend_server] > 0
                ):
                    connection_counts[backend_server] -= 1
            server_selected = False  # Mark failure
            async with safe_close_writer(client_writer):
                pass
            return
        except Exception as e:
            logger.error(
                f"Failed to connect to selected backend server {backend_server} for {client_addr}: {e}"
            )
            async with state_lock:  # Decrement count under lock
                if (
                    backend_server in connection_counts
                    and connection_counts[backend_server] > 0
                ):
                    connection_counts[backend_server] -= 1
            server_selected = False  # Mark failure
            async with safe_close_writer(client_writer):
                pass
            return

        # --- Relay Data Bidirectionally ---
        async with safe_close_writer(backend_writer):  # Ensure backend writer is closed
            client_to_backend = asyncio.create_task(
                relay_data(
                    client_reader, backend_writer, f"{client_addr} -> {backend_server}"
                )
            )
            backend_to_client = asyncio.create_task(
                relay_data(
                    backend_reader, client_writer, f"{backend_server} -> {client_addr}"
                )
            )
            done, pending = await asyncio.wait(
                [client_to_backend, backend_to_client],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in done:
                with contextlib.suppress(asyncio.CancelledError):
                    if task.exception():
                        logger.warning(
                            f"Relay task finished with error: {task.exception()}"
                        )

    except asyncio.CancelledError:
        logger.info(f"Client handler for {client_addr} cancelled.")
    except Exception as e:
        logger.error(f"Error handling client {client_addr}: {e}")
    finally:
        # Decrement connection count only if we successfully selected/incremented
        if backend_server and server_selected:
            async with state_lock:
                if backend_server in connection_counts:
                    if connection_counts[backend_server] > 0:
                        connection_counts[backend_server] -= 1
                        current_active_connections = max(
                            0, current_active_connections - 1
                        )
                    else:
                        logger.warning(
                            f"Attempted to decrement count below zero for {backend_server} on close"
                        )
                        connection_counts[backend_server] = 0


# --- Status Reporting Task ---
async def print_status_periodically():
    """Periodically displays a professional real-time status dashboard."""
    while True:
        await asyncio.sleep(STATUS_PRINT_INTERVAL)
        await display_status_dashboard()


async def display_status_dashboard():
    """Display a professional real-time status dashboard."""
    global current_active_connections, total_connections_served

    async with state_lock:
        current_available = set(available_servers)
        current_counts = connection_counts.copy()

    # Clear terminal for fresh display
    clear_terminal()

    # Get terminal dimensions for responsive layout
    columns, rows = get_terminal_size()
    dash_width = min(columns - 4, 100)

    # Header with title and timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'=' * dash_width}{Colors.RESET}")
    print(
        f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'🚀 vLLM Load Balancer Dashboard':^{dash_width}}{Colors.RESET}"
    )
    print(
        f"{Colors.BRIGHT_CYAN}{'Real-time Status & Monitoring':^{dash_width}}{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'=' * dash_width}{Colors.RESET}")
    print()

    # System Information Section
    uptime = format_uptime(start_time)
    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}📊 System Information{Colors.RESET}")
    print(f"{Colors.BRIGHT_BLACK}{'─' * (dash_width // 2)}{Colors.RESET}")
    print(f"{Colors.YELLOW}🕐 Current Time:{Colors.RESET} {current_time}")
    print(f"{Colors.YELLOW}⏱️  Uptime:{Colors.RESET} {uptime}")
    print(
        f"{Colors.YELLOW}🌐 Load Balancer:{Colors.RESET} {LOAD_BALANCER_HOST}:{LOAD_BALANCER_PORT}"
    )
    print(f"{Colors.YELLOW}🎯 Backend Host:{Colors.RESET} {BACKEND_HOST}")
    print(
        f"{Colors.YELLOW}🔧 Configured Ports:{Colors.RESET} {', '.join(map(str, BACKEND_PORTS))}"
    )
    print(f"{Colors.YELLOW}⚡ Request Throttling:{Colors.RESET} {THROTTLE_MS}ms minimum")
    print()

    # Connection Statistics Section
    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}📈 Connection Statistics{Colors.RESET}")
    print(f"{Colors.BRIGHT_BLACK}{'─' * (dash_width // 2)}{Colors.RESET}")
    print(
        f"{Colors.GREEN}📊 Total Connections Served:{Colors.RESET} {total_connections_served:,}"
    )
    print(
        f"{Colors.GREEN}🔗 Currently Active:{Colors.RESET} {current_active_connections}"
    )
    print(
        f"{Colors.GREEN}⚡ Health Check Timeout:{Colors.RESET} {HEALTH_CHECK_TIMEOUT}s"
    )
    print(
        f"{Colors.GREEN}🔄 Status Update Interval:{Colors.RESET} {STATUS_PRINT_INTERVAL}s"
    )
    print()

    # Backend Servers Status
    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}Backend Servers Status{Colors.RESET}")
    print(f"{Colors.BRIGHT_BLACK}{'─' * (dash_width // 2)}{Colors.RESET}")

    headers = [
        f"{Colors.BOLD}Server{Colors.RESET}",
        f"{Colors.BOLD}Host{Colors.RESET}",
        f"{Colors.BOLD}Port{Colors.RESET}",
        f"{Colors.BOLD}Active Conn.{Colors.RESET}",
        f"{Colors.BOLD}Status{Colors.RESET}",
    ]

    table_data = []
    total_backend_connections = 0

    for port in BACKEND_PORTS:
        server = (BACKEND_HOST, port)
        is_online = server in current_available
        count = current_counts.get(server, 0) if is_online else 0
        total_backend_connections += count

        # Color-code connection count based on load
        if count == 0:
            conn_display = f"{Colors.DIM}0{Colors.RESET}"
        elif count < 5:
            conn_display = f"{Colors.GREEN}{count}{Colors.RESET}"
        elif count < 10:
            conn_display = f"{Colors.YELLOW}{count}{Colors.RESET}"
        else:
            conn_display = f"{Colors.RED}{count}{Colors.RESET}"

        status_display = (
            f"{Colors.BG_GREEN}{Colors.BLACK} ONLINE {Colors.RESET}"
            if is_online
            else f"{Colors.BG_RED}{Colors.WHITE} OFFLINE {Colors.RESET}"
        )

        table_data.append(
            [
                f"{Colors.CYAN}{BACKEND_HOST}:{port}{Colors.RESET}",
                BACKEND_HOST,
                str(port),
                conn_display,
                status_display,
            ]
        )

    try:
        table = tabulate(table_data, headers=headers, tablefmt="fancy_grid")
        print(table)
        print()

        # Summary metrics
        online_count = sum(
            1 for port in BACKEND_PORTS if (BACKEND_HOST, port) in current_available
        )
        avg_connections = (
            total_backend_connections / online_count if online_count else 0
        )
        print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}📋 Summary{Colors.RESET}")
        print(f"{Colors.BRIGHT_BLACK}{'─' * (dash_width // 4)}{Colors.RESET}")
        print(
            f"{Colors.MAGENTA}🟢 Available Servers:{Colors.RESET} {online_count} / {len(BACKEND_PORTS)}"
        )
        print(
            f"{Colors.MAGENTA}📊 Total Backend Connections:{Colors.RESET} {total_backend_connections}"
        )
        print(
            f"{Colors.MAGENTA}📈 Average Load per Online Server:{Colors.RESET} {avg_connections:.1f}"
        )

    except Exception as e:
        logger.error(f"Error displaying status table: {e}")
        print(f"{Colors.RED}Error displaying server table: {e}{Colors.RESET}")

    # Footer with refresh info
    print()
    print(f"{Colors.BRIGHT_BLACK}{'─' * dash_width}{Colors.RESET}")
    print(
        f"{Colors.DIM}🔄 Auto-refresh every {STATUS_PRINT_INTERVAL}s | Press Ctrl+C to stop{Colors.RESET}"
    )
    print(f"{Colors.BRIGHT_BLACK}{'─' * dash_width}{Colors.RESET}")
    print()


# --- HTTP Stats Server ---
from aiohttp import web


async def stats_json(request):
    async with state_lock:
        # Build a list of all configured servers, with status and connections
        all_servers = []
        available_set = set(available_servers)
        for port in BACKEND_PORTS:
            server = (BACKEND_HOST, port)
            is_online = server in available_set
            all_servers.append(
                {
                    "host": BACKEND_HOST,
                    "port": port,
                    "active_connections": connection_counts.get(server, 0)
                    if is_online
                    else 0,
                    "status": "ONLINE" if is_online else "OFFLINE",
                }
            )
        stats = {
            "time": datetime.now().isoformat(),
            "uptime": format_uptime(start_time),
            "load_balancer_host": LOAD_BALANCER_HOST,
            "load_balancer_port": LOAD_BALANCER_PORT,
            "backend_host": BACKEND_HOST,
            "backend_ports": BACKEND_PORTS,
            "total_connections_served": total_connections_served,
            "current_active_connections": current_active_connections,
            "health_check_timeout": HEALTH_CHECK_TIMEOUT,
            "status_update_interval": STATUS_PRINT_INTERVAL,
            "throttle_ms": THROTTLE_MS,
            "servers": all_servers,
        }
    return web.json_response(stats)


async def stats_page(request):
    # High-quality HTML dashboard with auto-refresh and charts
    return web.Response(
        content_type="text/html",
        text="""
<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>vLLM Load Balancer Stats</title>
    <link rel='preconnect' href='https://fonts.googleapis.com'>
    <link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>
    <link href='https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap' rel='stylesheet'>
    <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
    <style>
        body { font-family: 'Roboto', sans-serif; background: #181c20; color: #f3f3f3; margin: 0; }
        .container { max-width: 900px; margin: 32px auto; background: #23272b; border-radius: 12px; box-shadow: 0 2px 16px #0008; padding: 32px; }
        h1 { text-align: center; font-size: 2.2em; margin-bottom: 0.2em; }
        .subtitle { text-align: center; color: #7fd7ff; margin-bottom: 1.5em; }
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 2em; }
        .stat-box { background: #20232a; border-radius: 8px; padding: 18px 24px; text-align: center; }
        .stat-label { color: #7fd7ff; font-size: 1.1em; margin-bottom: 0.2em; }
        .stat-value { font-size: 2em; font-weight: bold; }
        .server-table { width: 100%; border-collapse: collapse; margin-top: 1.5em; }
        .server-table th, .server-table td { padding: 10px 8px; text-align: center; }
        .server-table th { background: #2c313a; color: #7fd7ff; }
        .server-table tr:nth-child(even) { background: #23272b; }
        .server-table tr:nth-child(odd) { background: #1b1e22; }
        .status-online { color: #00e676; font-weight: bold; }
        .status-offline { color: #ff5252; font-weight: bold; }
        .chart-container { background: #20232a; border-radius: 8px; padding: 18px 24px; margin-top: 2em; }
        @media (max-width: 700px) { .stats-grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class='container'>
        <h1>🚀 vLLM Load Balancer</h1>
        <div class='subtitle'>Live Stats Dashboard</div>
        <div class='stats-grid' id='statsGrid'>
            <!-- Stats will be injected here -->
        </div>
        <div class='chart-container'>
            <canvas id='connChart' height='80'></canvas>
        </div>
        <table class='server-table' id='serverTable'>
            <thead>
                <tr>
                    <th>Backend Server</th>
                    <th>Host</th>
                    <th>Port</th>
                    <th>Active Connections</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>
        <div style='text-align:center; margin-top:2em; color:#888;'>
            <span id='lastUpdate'></span> | Auto-refreshing every 1s
        </div>
    </div>
    <script>
        let connChart;
        let connHistory = [];
        let timeHistory = [];
        async function fetchStats() {
            const res = await fetch('/stats.json');
            return await res.json();
        }
        function updateStats(stats) {
            document.getElementById('lastUpdate').textContent = 'Last update: ' + new Date(stats.time).toLocaleTimeString();
            // Stats grid
            document.getElementById('statsGrid').innerHTML = `
                <div class='stat-box'><div class='stat-label'>Uptime</div><div class='stat-value'>${stats.uptime}</div></div>
                <div class='stat-box'><div class='stat-label'>Total Connections</div><div class='stat-value'>${stats.total_connections_served}</div></div>
                <div class='stat-box'><div class='stat-label'>Active Connections</div><div class='stat-value'>${stats.current_active_connections}</div></div>
                <div class='stat-box'><div class='stat-label'>Configured Servers</div><div class='stat-value'>${stats.servers.length}</div></div>
            `;
            // Server table
            let tbody = document.querySelector('#serverTable tbody');
            tbody.innerHTML = '';
            for (const s of stats.servers) {
                tbody.innerHTML += `<tr>
                    <td>${s.host}:${s.port}</td>
                    <td>${s.host}</td>
                    <td>${s.port}</td>
                    <td>${s.active_connections}</td>
                    <td class='${s.status === "ONLINE" ? "status-online" : "status-offline"}'>${s.status}</td>
                </tr>`;
            }
            // Chart (only count online servers for active connections)
            connHistory.push(stats.current_active_connections);
            timeHistory.push(new Date(stats.time).toLocaleTimeString());
            if (connHistory.length > 60) { connHistory.shift(); timeHistory.shift(); }
            if (!connChart) {
                connChart = new Chart(document.getElementById('connChart').getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: timeHistory,
                        datasets: [{
                            label: 'Active Connections',
                            data: connHistory,
                            borderColor: '#7fd7ff',
                            backgroundColor: 'rgba(127,215,255,0.1)',
                            tension: 0.3,
                            fill: true,
                            pointRadius: 0
                        }]
                    },
                    options: {
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { display: false },
                            y: { beginAtZero: true, grid: { color: '#333' }, ticks: { color: '#7fd7ff' } }
                        },
                        animation: false,
                        responsive: true,
                        maintainAspectRatio: false
                    }
                });
            } else {
                connChart.data.labels = timeHistory;
                connChart.data.datasets[0].data = connHistory;
                connChart.update();
            }
        }
        async function refresh() {
            try {
                const stats = await fetchStats();
                updateStats(stats);
            } catch (e) {
                document.getElementById('lastUpdate').textContent = 'Error fetching stats';
            }
            setTimeout(refresh, 1000);
        }
        refresh();
    </script>
</body>
</html>
            """,
    )


async def start_stats_server(loop):
    app = web.Application()
    app.router.add_get("/stats", stats_page)
    app.router.add_get("/stats.json", stats_json)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, LOAD_BALANCER_HOST, STATS_PORT)
    await site.start()
    logger.info(
        f"Stats HTTP server running at http://{LOAD_BALANCER_HOST}:{STATS_PORT}/stats"
    )


async def main():
    global start_time
    start_time = time.time()
    clear_terminal()
    print_banner()

    # Start background tasks
    scan_task = asyncio.create_task(scan_and_update_servers())
    status_task = asyncio.create_task(print_status_periodically())

    # Start HTTP stats server (on STATS_PORT)
    loop = asyncio.get_running_loop()
    await start_stats_server(loop)

    # Start TCP server (on LOAD_BALANCER_PORT)
    server = await asyncio.start_server(
        handle_client, LOAD_BALANCER_HOST, LOAD_BALANCER_PORT
    )

    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    logger.info(f"Load balancer serving on {addrs}")
    logger.info(f"Configured backend ports: {BACKEND_PORTS} on host {BACKEND_HOST}")
    print(f"{Colors.BRIGHT_GREEN}✅ Load balancer started successfully!{Colors.RESET}")
    print(f"{Colors.BRIGHT_GREEN}🌐 Proxy listening on: {addrs}{Colors.RESET}")
    print(
        f"{Colors.BRIGHT_GREEN}📊 Stats dashboard: http://localhost:{STATS_PORT}/stats{Colors.RESET}"
    )
    print(f"{Colors.YELLOW}🔍 Scanning backend servers...{Colors.RESET}")
    print()
    await asyncio.sleep(2)

    async with server:
        try:
            await server.serve_forever()
        except asyncio.CancelledError:
            print(f"\n{Colors.YELLOW}🛑 Shutdown signal received...{Colors.RESET}")
            logger.info("Load balancer server shutting down.")
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}🛑 Shutdown requested by user...{Colors.RESET}")
            logger.info("Shutdown requested by user.")
        finally:
            print(f"{Colors.CYAN}🔄 Cleaning up background tasks...{Colors.RESET}")
            logger.info("Cancelling background tasks...")
            scan_task.cancel()
            status_task.cancel()
            try:
                await asyncio.gather(scan_task, status_task, return_exceptions=True)
            except asyncio.CancelledError:
                pass
            print(f"{Colors.BRIGHT_GREEN}✅ Shutdown complete. Goodbye!{Colors.RESET}")
            logger.info("Background tasks finished.")


def run_load_balancer():
    global \
        LOAD_BALANCER_PORT, \
        BACKEND_PORTS, \
        BACKEND_HOST, \
        STATUS_PRINT_INTERVAL, \
        HEALTH_CHECK_TIMEOUT, \
        THROTTLE_MS, \
        STATS_PORT
    args = parse_args()
    LOAD_BALANCER_PORT = args.port
    BACKEND_HOST = args.host
    BACKEND_PORTS = [int(p.strip()) for p in args.ports.split(",") if p.strip()]
    STATUS_PRINT_INTERVAL = args.status_interval
    HEALTH_CHECK_TIMEOUT = args.health_timeout
    THROTTLE_MS = args.throttle_ms
    if args.stats_port is not None:
        STATS_PORT = args.stats_port
    else:
        STATS_PORT = LOAD_BALANCER_PORT + 1
    if not BACKEND_PORTS:
        print(f"{Colors.BG_RED}{Colors.WHITE} ❌ ERROR {Colors.RESET}")
        print(
            f"{Colors.RED}No backend ports specified. Use --ports 8140,8150 ...{Colors.RESET}"
        )
        logger.critical("No backend ports specified. Use --ports 8140,8150 ...")
        sys.exit(1)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # This is handled in the main() function now
        pass
    except Exception as e:
        print(f"\n{Colors.BG_RED}{Colors.WHITE} ❌ CRITICAL ERROR {Colors.RESET}")
        print(f"{Colors.RED}Critical error in main execution: {e}{Colors.RESET}")
        logger.critical(f"Critical error in main execution: {e}")


if __name__ == "__main__":
    run_load_balancer()
