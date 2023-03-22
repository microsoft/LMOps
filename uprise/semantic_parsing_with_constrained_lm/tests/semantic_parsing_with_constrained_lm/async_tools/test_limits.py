# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import itertools
import time
from typing import List

import pytest

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.async_tools import limits


@pytest.mark.skip(reason="Flaky due to use of time.time()")
def test_token_bucket_simple():
    now = time.time()
    claimed: List[int] = []
    bucket = limits.TokenBucket(capacity=100, refill_rate=10)

    async def claim_then_record(bucket, tokens):
        await bucket.claim(tokens)
        print(time.time() - now)
        claimed.append(tokens)

    async def inner():
        task1 = asyncio.create_task(claim_then_record(bucket, 1))
        task2 = asyncio.create_task(claim_then_record(bucket, 2))
        task3 = asyncio.create_task(claim_then_record(bucket, 3))
        await asyncio.gather(task1, task2, task3)
        assert abs((time.time() - now) - (1 + 2 + 3) / 10) <= 0.01
        assert claimed == [1, 2, 3]

    asyncio.run(inner())


@pytest.mark.skip(reason="Flaky due to use of time.time()")
def test_token_bucket_big():
    now = time.time()
    claimed: List[int] = []
    bucket = limits.TokenBucket(capacity=10, refill_rate=100)

    async def claim_then_record(bucket, tokens):
        await bucket.claim(tokens)
        print(time.time() - now)
        claimed.append(tokens)

    async def inner():
        tasks = []
        for _ in range(100):
            tasks.append(asyncio.create_task(claim_then_record(bucket, 1)))
        await asyncio.gather(*tasks)

    asyncio.run(inner())


@pytest.mark.skip(reason="Flaky due to use of time.time()")
def test_token_bucket_small_capacity():
    now = time.time()
    claimed: List[int] = []
    bucket = limits.TokenBucket(capacity=3, refill_rate=100)

    async def claim_then_record(bucket, tokens):
        await bucket.claim(tokens)
        print(time.time() - now)
        claimed.append(tokens)

    async def inner():
        task1 = asyncio.create_task(claim_then_record(bucket, 3))
        task2 = asyncio.create_task(claim_then_record(bucket, 2))
        task3 = asyncio.create_task(claim_then_record(bucket, 1))
        await asyncio.gather(task1, task2, task3)
        assert claimed == [3, 2, 1]

    asyncio.run(inner())


@pytest.mark.skip(reason="Flaky due to use of time.time()")
def test_token_bucket_rate_change():
    now = time.time()
    claimed: List[int] = []
    bucket = limits.TokenBucket(capacity=100, refill_rate=10)

    async def claim_then_record(bucket, tokens):
        await bucket.claim(tokens)
        await bucket.change_rate(bucket.refill_rate * 2)

        print(time.time() - now)
        claimed.append(tokens)

    async def inner():
        task1 = asyncio.create_task(claim_then_record(bucket, 3))
        task2 = asyncio.create_task(claim_then_record(bucket, 2))
        task3 = asyncio.create_task(claim_then_record(bucket, 1))
        await asyncio.gather(task1, task2, task3)
        assert abs((time.time() - now) - (3 / 10 + 2 / 20 + 1 / 40)) <= 0.01
        assert claimed == [3, 2, 1]

    asyncio.run(inner())


@pytest.mark.skip(reason="Flaky due to use of time.time()")
def test_adaptive_limiter():
    COUNT = 8
    QPS = 10

    last_called = None
    called_timestamps = []
    success_timestamps = []

    start = time.time()

    async def query():
        nonlocal last_called
        now = time.time()
        elapsed = 100000 if last_called is None else now - last_called
        last_called = now
        since_start = now - start
        called_timestamps.append(since_start)
        if elapsed < 1 / QPS:
            raise limits.RateLimitExceededError()
        success_timestamps.append(since_start)

    async def inner():
        limiter = limits.AdaptiveLimiter(initial_qps=QPS * 0.9)
        await asyncio.gather(*(limiter(query)() for _ in range(COUNT)))

    asyncio.run(inner())
    last_measured_qps = 1 / (called_timestamps[-1] - called_timestamps[-2])
    assert 0.9 * QPS <= last_measured_qps < QPS
    assert len(success_timestamps) == COUNT


def test_map_async_limited():
    count = 0
    max_count_seen = 0

    async def f(elem: int):
        nonlocal count, max_count_seen
        count += 1
        max_count_seen = max(count, max_count_seen)
        await asyncio.sleep(0)
        count -= 1

        return elem * 2

    async def inner():
        nonlocal count, max_count_seen
        for max_concurrency in [2, 4, 6]:
            max_count_seen = 0
            async for x2, x in limits.map_async_limited(
                f, range(5), max_concurrency=max_concurrency
            ):
                assert x2 == x * 2
                assert max_count_seen == min(5, max_concurrency)
            assert count == 0

    asyncio.run(inner(), debug=True)


@pytest.mark.parametrize(
    ("parties", "max_wait", "sleep_between"),
    itertools.product((3, 10), (0.3, 0.5), (0.0, 0.4)),
)
def test_timeout_barrier(parties, max_wait, sleep_between):
    completed = []
    barrier = limits.TimeoutBarrier(parties=parties, max_wait=max_wait)

    async def wait_for_barrier(arg):
        await barrier.arrive_and_wait()
        completed.append(arg)

    async def inner():
        tasks = []
        for i in range(5):
            tasks.append(asyncio.create_task(wait_for_barrier(i)))
            if sleep_between > 0:
                await asyncio.sleep(sleep_between)
        await asyncio.gather(*tasks)

    asyncio.run(inner(), debug=True)
    assert sorted(completed) == [0, 1, 2, 3, 4]


def test_timeout_barrier_with_exception():
    async def raises():
        return 1 / 0

    async def wait_for_barrier(barrier, exceptions):
        try:
            await barrier.arrive_and_wait()
        except Exception as e:  # pylint: disable=broad-except
            exceptions.append(e)

    # Exception is raised in one waiter when the barrier is released due to timeout
    async def inner1():
        exceptions = []
        barrier = limits.TimeoutBarrier(parties=10, max_wait=0.1, callback=raises)
        await asyncio.gather(
            wait_for_barrier(barrier, exceptions), wait_for_barrier(barrier, exceptions)
        )
        assert len(exceptions) == 1 and isinstance(exceptions[0], ZeroDivisionError)

    asyncio.run(inner1(), debug=True)

    # Exception is raised in one waiter when the barrier is released due to enough parties arriving
    async def inner2():
        exceptions = []
        barrier = limits.TimeoutBarrier(parties=2, max_wait=100, callback=raises)
        await asyncio.gather(
            wait_for_barrier(barrier, exceptions), wait_for_barrier(barrier, exceptions)
        )
        assert len(exceptions) == 1 and isinstance(exceptions[0], ZeroDivisionError)

    asyncio.run(inner2(), debug=True)
