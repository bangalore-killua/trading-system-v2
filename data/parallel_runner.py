"""
parallel_runner.py — Adaptive Parallelism for Trading Pipeline Stages

Two execution modes, chosen per stage:
  I/O-bound stages (RSS fetch, yfinance, FINRA API):
    asyncio.Semaphore — concurrency limited, no OS threads needed
    Same speed on CX22 (2vCPU) and laptop (8-core) — GIL irrelevant for I/O

  CPU-bound stages (ML inference, RSI/MACD computation, sector_percentile):
    concurrent.futures.ThreadPoolExecutor — actual parallel CPU use
    Workers = min(os.cpu_count(), len(symbols)) — adapts to any machine

  Execution (portfolio mutations):
    Serial — always. Portfolio state is not thread-safe.

Usage:
    from data.parallel_runner import run_io_bound, run_cpu_bound, adaptive_workers

    # I/O-bound: fetch OHLCV for 90 symbols concurrently
    results = await run_io_bound(fetch_ohlcv, symbols, concurrency=10)

    # CPU-bound: run ML inference on 90 feature vectors
    results = run_cpu_bound(model.predict, feature_vectors)

    # How many workers will be used?
    print(adaptive_workers(len(symbols)))  # 2 on CX22, 8 on laptop
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Coroutine, List, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def adaptive_workers(n_tasks: int) -> int:
    """Return optimal worker count for CPU-bound tasks.

    Caps at CPU count (CX22=2, laptop=8) and task count (no idle threads).
    """
    cpu_count = os.cpu_count() or 2
    return max(1, min(cpu_count, n_tasks))


async def run_io_bound(
    coro_fn: Callable[..., Coroutine],
    items: List[Any],
    concurrency: int = 10,
    *args,
    **kwargs,
) -> List[Any]:
    """Run an async coroutine for each item, with concurrency cap.

    Parameters
    ----------
    coro_fn    : async callable(item, *args, **kwargs) → result
    items      : list of items to process
    concurrency: max simultaneous connections (default 10)

    Returns list of results (same order as items); exceptions replaced with None.
    """
    sem = asyncio.Semaphore(concurrency)

    async def bounded(item: Any) -> Any:
        async with sem:
            try:
                return await coro_fn(item, *args, **kwargs)
            except Exception as e:
                logger.debug("run_io_bound: %s failed: %s", item, e)
                return None

    return list(await asyncio.gather(*[bounded(it) for it in items]))


def run_cpu_bound(
    fn: Callable[..., T],
    items: List[Any],
    *args,
    **kwargs,
) -> List[Any]:
    """Run a sync function for each item using adaptive thread pool.

    Parameters
    ----------
    fn    : sync callable(item, *args, **kwargs) → result
    items : list of items to process

    Returns list of results (same order as items); exceptions replaced with None.
    """
    n_workers = adaptive_workers(len(items))
    logger.debug("run_cpu_bound: %d items | %d workers (CPU count=%d)", len(items), n_workers, os.cpu_count() or 2)

    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="cpu-worker") as pool:
        futures = {pool.submit(fn, item, *args, **kwargs): i for i, item in enumerate(items)}
        for future, idx in futures.items():
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.debug("run_cpu_bound item[%d] failed: %s", idx, e)
                results[idx] = None
    return results


async def run_cpu_bound_async(
    fn: Callable[..., T],
    items: List[Any],
    *args,
    **kwargs,
) -> List[Any]:
    """Async wrapper for run_cpu_bound — runs in executor so it doesn't block the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: run_cpu_bound(fn, items, *args, **kwargs),
    )
