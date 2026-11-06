import argparse
import json
import os
import random
import socket
import statistics
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def worker(args):
    import polars as pl

    case = json.loads(args.case)
    pl.Series([1, 2, 3]).sum()
    start = time.perf_counter()
    result = pl.scan_parquet(
        f"s3://benchmark/{case['file']}",
        glob=False,
        storage_options={"endpoint_url": args.endpoint, "access_key_id": "test",
                         "secret_access_key": "test", "region": "us-east-1",
                         "allow_http": "true"},
    ).select(pl.col(case["columns"]).sum()).collect()
    elapsed = time.perf_counter() - start
    assert list(result.row(0)) == case["expected"], result
    print(json.dumps({"seconds": elapsed, "polars": pl.__version__,
                      "threads": pl.thread_pool_size(), "verified": True}))


def fixtures(root):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    root.mkdir(exist_ok=True)
    cases = {}

    def write(name, table, selected, expected):
        file = root / f"{name}.parquet"
        pq.write_table(table, file, compression=None, use_dictionary=False,
                       row_group_size=table.num_rows)
        rg = pq.read_metadata(file).row_group(0)
        cases[name] = {
            "file": file.name, "columns": selected, "expected": expected,
            "file_bytes": file.stat().st_size,
            "selected_bytes": sum(rg.column(table.column_names.index(c)).total_compressed_size
                                  for c in selected),
        }

    n = 8192
    flags = pa.array(np.arange(n) % 2 == 0)
    integers = pa.array(np.arange(n, dtype=np.int64))
    for name, selected_array, gap_array, expected in [
        ("sparse", flags, pa.Array.from_buffers(pa.binary(64), n,
                                                [None, pa.py_buffer(bytes(64*n))]), n//2),
        ("small_gaps", integers, flags, n*(n-1)//2),
        ("medium_gaps", integers, pa.Array.from_buffers(pa.binary(8), n,
                                                        [None, pa.py_buffer(bytes(8*n))]), n*(n-1)//2),
    ]:
        columns = {}
        selected = []
        for i in range(16):
            selected.append(f"selected{i}")
            columns[f"selected{i}"] = selected_array
            columns[f"gap{i}"] = gap_array
        write(name, pa.table(columns), selected, [expected]*16)

    n = 2_097_152
    values = pa.array(np.arange(n, dtype=np.int64))
    columns = [f"c{i}" for i in range(4)]
    write("dense", pa.table({c: values for c in columns}), columns, [n*(n-1)//2]*4)
    return cases


class Server(ThreadingHTTPServer):
    request_queue_size = 256
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, format, *args):
        pass

    def do_HEAD(self):
        self.respond(False)

    def do_GET(self):
        self.respond(True)

    def respond(self, body):
        server = self.server
        name = self.path.split("?")[0].rsplit("/", 1)[-1]
        data = server.files.get(name)
        if data is None:
            self.send_error(404)
            return
        size = len(data)
        start, end = 0, size - 1
        header = self.headers.get("Range")
        if header:
            assert header.startswith("bytes=") and "," not in header, header
            first, last = header[6:].split("-", 1)
            if first:
                start = int(first)
                end = min(int(last), end) if last else end
            else:
                start = max(size - int(last), 0)
        length = end - start + 1
        with server.lock:
            server.requests.append({"method": self.command, "range": header,
                                    "bytes": length if body else 0})
        if server.latency:
            time.sleep(server.latency)
        self.send_response(206 if header else 200)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", f'"{name}-v1"')
        self.send_header("Last-Modified", "Fri, 06 Nov 2026 12:00:00 GMT")
        if header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if body:
            for offset in range(start, end+1, 64*1024):
                chunk = memoryview(data)[offset:min(offset+64*1024, end+1)]
                if server.bandwidth:
                    with server.lock:
                        server.next_send = max(time.perf_counter(), server.next_send) + len(chunk)/server.bandwidth
                        ready = server.next_send
                    time.sleep(max(0, ready-time.perf_counter()))
                self.wfile.write(chunk)


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    cases = fixtures(root / "fixtures")
    server = Server(("127.0.0.1", 0), Handler)
    server.files = {c["file"]: (root / "fixtures" / c["file"]).read_bytes() for c in cases.values()}
    server.lock = threading.Lock()
    server.requests = []
    server.latency = 0
    server.bandwidth = 0
    server.next_send = 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    variants = {"baseline": args.baseline, "patched": args.patched}
    profiles = {"loopback": (0, 0), "25MiB_10ms": (.010, 25*2**20),
                "100MiB_20ms": (.020, 100*2**20), "1000MiB_80ms": (.080, 1000*2**20)}
    records = []
    rng = random.Random(20261106)
    try:
        with (root / "results.jsonl").open("w") as output:
            for profile, (latency, bandwidth) in profiles.items():
                server.latency, server.bandwidth = latency, bandwidth
                for name, case in cases.items():
                    for repeat in range(-1, args.repeats):
                        order = list(variants)
                        rng.shuffle(order)
                        for variant in order:
                            with server.lock:
                                server.requests.clear()
                                server.next_send = 0
                            env = {**os.environ, "PYTHONPATH": variants[variant], "POLARS_MAX_THREADS": "4"}
                            completed = subprocess.run(
                                [sys.executable, __file__, "--worker", "--case", json.dumps(case),
                                 "--endpoint", endpoint], env=env, check=True, capture_output=True, text=True)
                            result = json.loads(completed.stdout)
                            with server.lock:
                                requests = list(server.requests)
                            row = {"profile": profile, "case": name, "variant": variant,
                                   "repeat": repeat, **result,
                                   "get_count": sum(r["method"] == "GET" for r in requests),
                                   "head_count": sum(r["method"] == "HEAD" for r in requests),
                                   "get_bytes": sum(r["bytes"] for r in requests), "requests": requests}
                            output.write(json.dumps(row)+"\n")
                            output.flush()
                            if repeat >= 0:
                                records.append(row)
                    print(f"Finished {profile} / {name}", flush=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    summaries = []
    for profile in profiles:
        for name in cases:
            item = {"profile": profile, "case": name}
            for variant in variants:
                rows = [r for r in records if (r["profile"], r["case"], r["variant"]) == (profile, name, variant)]
                times = sorted(r["seconds"] for r in rows)
                item[variant] = {"median_ms": statistics.median(times)*1000,
                                 "min_ms": times[0]*1000, "max_ms": times[-1]*1000,
                                 "get_bytes": sorted({r["get_bytes"] for r in rows}),
                                 "get_count": sorted({r["get_count"] for r in rows})}
            item["speedup"] = item["baseline"]["median_ms"] / item["patched"]["median_ms"]
            summaries.append(item)
    report = {"cases": cases, "profiles": profiles, "repeats": args.repeats,
              "python": sys.version, "platform": sys.platform, "summary": summaries}
    (root / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--endpoint")
    parser.add_argument("--baseline", default="")
    parser.add_argument("--patched", default="")
    parser.add_argument("--output", default="results")
    parser.add_argument("--repeats", type=int, default=7)
    options = parser.parse_args()
    worker(options) if options.worker else main(options)
