import asyncio
import random
from collections import defaultdict
import time
from tabulate import tabulate
import logging
import contextlib
import aiohttp # <-- Import aiohttp

# --- Configuration ---
LOAD_BALANCER_HOST = '0.0.0.0'
LOAD_BALANCER_PORT = 8008

SCAN_TARGET_HOST = 'localhost'
SCAN_PORT_START = 8150
SCAN_PORT_END = 8170 # Inclusive
SCAN_INTERVAL = 30
# Timeout applies to the HTTP health check request now
HEALTH_CHECK_TIMEOUT = 2.0 # Increased slightly for HTTP requests

STATUS_PRINT_INTERVAL = 5
BUFFER_SIZE = 4096

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Shared State ---
available_servers = []
connection_counts = defaultdict(int)
state_lock = asyncio.Lock()

# --- Helper Functions --- (relay_data and safe_close_writer remain the same)
async def relay_data(reader, writer, direction):
    """Reads data from reader and writes to writer until EOF or error."""
    try:
        while True:
            data = await reader.read(BUFFER_SIZE)
            if not data:
                logging.debug(f"EOF received on {direction} stream.")
                break
            writer.write(data)
            await writer.drain()
    except ConnectionResetError:
        logging.warning(f"Connection reset on {direction} stream.")
    except asyncio.CancelledError:
        logging.debug(f"Relay task cancelled for {direction}.")
        raise
    except Exception as e:
        logging.warning(f"Error during data relay ({direction}): {e}")
    finally:
        if not writer.is_closing():
            try:
                writer.close()
                await writer.wait_closed()
                logging.debug(f"Closed writer for {direction}")
            except Exception as close_err:
                logging.debug(f"Error closing writer for {direction} (might be expected): {close_err}")

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
                logging.debug(f"Error closing writer in context manager: {e}")

# --- Server Scanning and Health Check (Modified) ---

async def check_server_health(session, host, port):
    """Performs an HTTP GET request to the /health endpoint."""
    url = f"http://{host}:{port}/health"
    try:
        # Use the provided aiohttp session to make the GET request
        async with session.get(url, timeout=HEALTH_CHECK_TIMEOUT) as response:
            # Check for a successful status code (2xx range)
            if 200 <= response.status < 300:
                logging.debug(f"Health check success for {url} (Status: {response.status})")
                # Ensure the connection is released back to the pool
                await response.release()
                return True
            else:
                logging.debug(f"Health check failed for {url} (Status: {response.status})")
                await response.release()
                return False
    except asyncio.TimeoutError:
        logging.debug(f"Health check HTTP request timeout for {url}")
        return False
    except aiohttp.ClientConnectorError as e:
        # Handles connection refused, DNS errors etc. - server likely down
        logging.debug(f"Health check connection error for {url}: {e}")
        return False
    except aiohttp.ClientError as e:
        # Catch other potential client errors (e.g., invalid URL structure, too many redirects)
        logging.warning(f"Health check client error for {url}: {e}")
        return False
    except Exception as e:
        # Catch any other unexpected errors during the check
        logging.error(f"Unexpected health check error for {url}: {e}")
        return False

async def scan_and_update_servers():
    """Periodically scans ports using HTTP /health check and updates available servers."""
    global available_servers
    logging.debug(f"Starting server scan task (HTTP GET /health on Ports {SCAN_PORT_START}-{SCAN_PORT_END} every {SCAN_INTERVAL}s)")
    while True:
        try:
            current_scan_results = []
            scan_tasks = []
            ports_to_scan = range(SCAN_PORT_START, SCAN_PORT_END + 1)

            # Create ONE aiohttp session for all checks within this scan cycle for efficiency
            async with aiohttp.ClientSession() as session:
                # Create health check tasks for all ports, passing the shared session
                for port in ports_to_scan:
                    task = asyncio.create_task(check_server_health(session, SCAN_TARGET_HOST, port))
                    scan_tasks.append((task, port))

                # Wait for all health checks to complete
                # return_exceptions=True prevents gather from stopping if one check fails
                await asyncio.gather(*(task for task, port in scan_tasks), return_exceptions=True)

                # Collect results from completed tasks
                for task, port in scan_tasks:
                    try:
                        # Check if task finished, wasn't cancelled, and returned True
                        if task.done() and not task.cancelled() and task.result() is True:
                            current_scan_results.append((SCAN_TARGET_HOST, port))
                    except Exception as e:
                        # Log errors from the health check task itself if gather didn't catch them
                        logging.error(f"Error retrieving health check result for port {port}: {e}")

            # --- Update Shared State (Locked) ---
            async with state_lock:
                previous_servers = set(available_servers)
                current_set = set(current_scan_results)

                added = current_set - previous_servers
                removed = previous_servers - current_set

                if added:
                    logging.info(f"Servers added (passed /health check): {sorted(list(added))}")
                if removed:
                    logging.info(f"Servers removed (failed /health check or stopped): {sorted(list(removed))}")
                    for server in removed:
                        if server in connection_counts:
                            del connection_counts[server]
                            logging.debug(f"Removed connection count entry for unavailable server {server}")

                available_servers = sorted(list(current_set))
                for server in available_servers:
                    if server not in connection_counts:
                        connection_counts[server] = 0

            logging.debug(f"Scan complete. Active servers: {available_servers}")

        except asyncio.CancelledError:
             logging.info("Server scan task cancelled.")
             break
        except Exception as e:
            logging.error(f"Error in scan_and_update_servers loop: {e}")
            await asyncio.sleep(SCAN_INTERVAL / 2) # Avoid tight loop on error

        await asyncio.sleep(SCAN_INTERVAL)


# --- Core Load Balancer Logic (handle_client remains the same) ---
async def handle_client(client_reader, client_writer):
    """Handles a single client connection."""
    client_addr = client_writer.get_extra_info('peername')
    logging.info(f"Accepted connection from {client_addr}")

    backend_server = None
    backend_reader = None
    backend_writer = None
    server_selected = False

    try:
        # --- Select Backend Server (Least Connections from Available) ---
        selected_server = None
        async with state_lock: # Lock to safely access available_servers and connection_counts
            if not available_servers:
                logging.warning(f"No backend servers available (failed health checks?) for client {client_addr}. Closing connection.")
                async with safe_close_writer(client_writer): pass
                return

            min_connections = float('inf')
            least_used_available_servers = []
            for server in available_servers: # Iterate only over servers that passed health check
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
                logging.info(f"Routing {client_addr} to {backend_server} (Current connections: {connection_counts[backend_server]})")
            else:
                 logging.error(f"Logic error: No server chosen despite available servers list not being empty for {client_addr}.")
                 async with safe_close_writer(client_writer): pass
                 return

        # --- Connect to Backend Server ---
        if not backend_server:
             logging.error(f"No backend server selected for {client_addr} before connection attempt.")
             async with safe_close_writer(client_writer): pass
             server_selected = False
             return

        try:
            logging.debug(f"Attempting connection to backend {backend_server} for {client_addr}")
            backend_reader, backend_writer = await asyncio.open_connection(
                backend_server[0], backend_server[1]
            )
            logging.debug(f"Successfully connected to backend {backend_server} for {client_addr}")

        # Handle connection failure AFTER selection (server might go down between health check and selection)
        except ConnectionRefusedError:
            logging.error(f"Connection refused by selected backend server {backend_server} for {client_addr}")
            async with state_lock: # Decrement count under lock
                 if backend_server in connection_counts and connection_counts[backend_server] > 0: connection_counts[backend_server] -= 1
            server_selected = False # Mark failure
            async with safe_close_writer(client_writer): pass
            return
        except Exception as e:
            logging.error(f"Failed to connect to selected backend server {backend_server} for {client_addr}: {e}")
            async with state_lock: # Decrement count under lock
                 if backend_server in connection_counts and connection_counts[backend_server] > 0: connection_counts[backend_server] -= 1
            server_selected = False # Mark failure
            async with safe_close_writer(client_writer): pass
            return

        # --- Relay Data Bidirectionally ---
        async with safe_close_writer(backend_writer): # Ensure backend writer is closed
            client_to_backend = asyncio.create_task(
                relay_data(client_reader, backend_writer, f"{client_addr} -> {backend_server}")
            )
            backend_to_client = asyncio.create_task(
                relay_data(backend_reader, client_writer, f"{backend_server} -> {client_addr}")
            )
            done, pending = await asyncio.wait(
                [client_to_backend, backend_to_client], return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending: task.cancel()
            for task in done:
                 with contextlib.suppress(asyncio.CancelledError):
                     if task.exception(): logging.warning(f"Relay task finished with error: {task.exception()}")

    except asyncio.CancelledError:
        logging.info(f"Client handler for {client_addr} cancelled.")
    except Exception as e:
        logging.error(f"Error handling client {client_addr}: {e}")
    finally:
        logging.info(f"Closing connection for {client_addr}")
        # Decrement connection count only if we successfully selected/incremented
        if backend_server and server_selected:
            async with state_lock:
                if backend_server in connection_counts:
                    if connection_counts[backend_server] > 0:
                        connection_counts[backend_server] -= 1
                        logging.info(f"Connection closed for {client_addr}. Backend {backend_server} connections: {connection_counts[backend_server]}")
                    else:
                        logging.warning(f"Attempted to decrement count below zero for {backend_server} on close")
                        connection_counts[backend_server] = 0

# --- Status Reporting Task (print_status_periodically remains the same) ---
async def print_status_periodically():
    """Periodically prints the connection status based on available servers."""
    while True:
        await asyncio.sleep(STATUS_PRINT_INTERVAL)
        async with state_lock:
            headers = ["Backend Server", "Host", "Port", "Active Connections", "Status"]
            table_data = []
            total_connections = 0
            current_available = available_servers[:]
            current_counts = connection_counts.copy()

        if not current_available:
            # clear terminal and print status
            print("\033[H\033[J", end="")  # Clear terminal
            print("\n----- Load Balancer Status -----")
            print("No backend servers currently available (failed /health check).")
            print("------------------------------\n")
            continue

        for server in current_available:
            host, port = server
            count = current_counts.get(server, 0)
            table_data.append([f"{host}:{port}", host, port, count, "Available"])
            total_connections += count

        table_data.sort(key=lambda row: (row[1], row[2]))

        try:
            table = tabulate(table_data, headers=headers, tablefmt="grid")
            print("\n----- Load Balancer Status -----")
            print(f"Scanning Ports: {SCAN_PORT_START}-{SCAN_PORT_END} on {SCAN_TARGET_HOST} (using /health endpoint)")
            print(f"Scan Interval: {SCAN_INTERVAL}s | Health Check Timeout: {HEALTH_CHECK_TIMEOUT}s")
            print(table)
            print(f"Total Active Connections (on available servers): {total_connections}")
            print("------------------------------\n")
        except Exception as e:
            logging.error(f"Error printing status table: {e}")


# --- Main Execution (main remains the same) ---
async def main():
    scan_task = asyncio.create_task(scan_and_update_servers())
    status_task = asyncio.create_task(print_status_periodically())

    server = await asyncio.start_server(
        handle_client, LOAD_BALANCER_HOST, LOAD_BALANCER_PORT
    )

    addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
    logging.info(f'Load balancer serving on {addrs}')
    logging.info(f'Dynamically discovering servers via HTTP /health on {SCAN_TARGET_HOST}:{SCAN_PORT_START}-{SCAN_PORT_END}')

    async with server:
        try:
            await server.serve_forever()
        except asyncio.CancelledError:
            logging.info("Load balancer server shutting down.")
        finally:
            logging.info("Cancelling background tasks...")
            scan_task.cancel()
            status_task.cancel()
            try:
                await asyncio.gather(scan_task, status_task, return_exceptions=True)
            except asyncio.CancelledError:
                pass
            logging.info("Background tasks finished.")

def run_load_balancer():
    # Make sure to install aiohttp: pip install aiohttp
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Shutdown requested by user.")
    except Exception as e:
         logging.critical(f"Critical error in main execution: {e}")

if __name__ == "__main__":
    run_load_balancer()