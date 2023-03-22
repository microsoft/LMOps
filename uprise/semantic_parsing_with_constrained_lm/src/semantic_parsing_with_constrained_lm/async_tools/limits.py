# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: disable=unsubscriptable-object
import asyncio
import collections
import dataclasses
import functools
import inspect
import time
from asyncio.futures import Future
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Deque,
    Dict,
    Iterable,
    Optional,
    Set,
    Tuple,
    TypeVar,
    cast,
)


@dataclass
class TokenBucket:
    """Used to prevent too much resource consumption during any time period.

    These tokens are not words (like in NLP) and the bucket is not a map.
    Instead the tokens are like coins and the bucket is like a physical bucket.

    See https://en.wikipedia.org/wiki/Token_bucket#Algorithm for more details.
    However, unlike typical implementations of token buckets, this one allows
    the refill rate to change at any time.
    """

    # Maximum number of tokens that can be in the bucket.
    capacity: float
    # Amount of tokens added per second.
    refill_rate: float

    # The current amount of tokens in the bucket.
    level: float = 0

    # Time when `level` was last updated.
    last_updated: float = dataclasses.field(default_factory=time.time)
    # This lock is used when updating self.level. Not sure if it's really necessary but can't hurt...
    _lock: Optional[asyncio.Lock] = None
    # One entry per call to `claim` that's waiting for the bucket to refill.
    _waiting: Deque[Tuple[asyncio.Event, float]] = dataclasses.field(
        default_factory=collections.deque
    )
    # Set by the next `claim` in line so that it can be notified if the refill rate changes.
    _next_token_refill: Optional[asyncio.Event] = None

    async def claim(self, tokens: float) -> None:
        assert tokens <= self.capacity

        # We initialize self._lock here because Locks can only be created in async functions
        # (unless we create an event loop for the thread manually).
        if self._lock is None:
            self._lock = asyncio.Lock()

        event: Optional[asyncio.Event] = None
        try:
            # Check if we have sufficient tokens already
            await self._update_level()
            async with self._lock:
                if self.level >= tokens:
                    self.level -= tokens
                    return

            event = asyncio.Event()
            self._waiting.append((event, tokens))
            while True:
                # Check whether we're first in line
                first_event, _ = self._waiting[0]
                if event is first_event:
                    try:
                        # Wait for the bucket to fill up.
                        self._next_token_refill = event
                        await asyncio.wait_for(
                            event.wait(),
                            timeout=(tokens - self.level) / self.refill_rate,
                        )

                        # If we reach here, before we refilled all the tokens,
                        # the refill rate changed. Recompute how long we should
                        # wait until the tokens are refilled, and try again.
                        event.clear()
                        self._next_token_refill = None
                        continue
                    except asyncio.TimeoutError:
                        # We should have collected enough tokens by now to take our turn
                        await self._update_level()
                        break
                else:
                    # Wait until we get to the start of the line
                    await event.wait()
                    event.clear()

            if self.level < tokens:
                # Not sure why this happens, but it does sometimes...
                await asyncio.sleep((tokens - self.level) / self.refill_rate)
                await self._update_level()
            async with self._lock:
                assert self.level >= tokens
                self.level -= tokens
            self._waiting.popleft()
            if self._waiting:
                # Tell the next in line that they're first now.
                event, _ = self._waiting[0]
                event.set()
            return

        except asyncio.CancelledError:
            pass

    async def change_rate(self, new_rate: float) -> None:
        self.refill_rate = new_rate
        await self._update_level()
        # Tell the task waiting for tokens to refill to wake up so that we can
        # recompute how much longer it should sleep, with the current rate.
        if self._next_token_refill:
            self._next_token_refill.set()

    async def _update_level(self) -> None:
        async with self._lock:  # type: ignore
            now = time.time()
            elapsed = now - self.last_updated
            self.level = min(self.capacity, self.level + self.refill_rate * elapsed)
            self.last_updated = now


class RateLimitExceededError(Exception):
    pass


CallableT = TypeVar("CallableT", bound=Callable[..., Any])


@dataclass
class AdaptiveLimiter:
    """Adaptively rate-limit the wrapped function.

    When a call to the function is successful, we increase the rate additively;
    if it complains that the rate was exceeded, we decrease the rate multiplicatively."""

    initial_qps: float = 10
    max_qps: float = 500
    min_qps: float = 1
    bucket: TokenBucket = dataclasses.field(init=False)

    def __post_init__(self):
        self.bucket = TokenBucket(self.max_qps, self.initial_qps)

    def __call__(self, func: CallableT) -> CallableT:
        @functools.wraps(func)
        async def wrapped(*args, **kwargs):
            while True:
                # Wait our turn
                await self.bucket.claim(1)
                # Try calling the function
                try:
                    result = await func(*args, **kwargs)
                    await self.bucket.change_rate(
                        min(self.bucket.refill_rate + 1, self.max_qps)
                    )
                    return result
                except RateLimitExceededError:
                    await self.bucket.change_rate(
                        max(self.bucket.refill_rate * 0.9, self.min_qps)
                    )

        return cast(CallableT, wrapped)


@dataclass
class MapInnerException(Exception):
    orig_item: Any


T = TypeVar("T")
U = TypeVar("U")


async def map_async_limited(
    f: Callable[[T], Awaitable[U]],
    items: Iterable[T],
    max_concurrency: int,
    wrap_exception: bool = True,
) -> AsyncGenerator[Tuple[U, T], None]:
    """Runs `f` on each argument of `items`, and returns the results as an iterator.

    `f` is an async function. We start up to `max_concurrency` copies of `f`
    at once. The resulting iterator may not have the same order as `items`,
    but we staple the result of `f` with the argument that produced it."""
    assert max_concurrency > 0

    iterator = iter(items)
    exhausted = False
    tasks_to_items: Dict[Future[U], T] = {}
    pending: Set[Future[U]] = set()

    def fill_pending():
        nonlocal exhausted
        while not exhausted and len(pending) < max_concurrency:
            try:
                item = next(iterator)
            except StopIteration:
                exhausted = True
                break
            task = asyncio.create_task(f(item))
            tasks_to_items[task] = item
            pending.add(task)

    # Start off by scheduling the first `max_concurrency` tasks.
    fill_pending()
    if not pending:
        # Nothing to do, so just exit
        return

    while True:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        fill_pending()
        for task in done:
            orig_item = tasks_to_items[task]
            del tasks_to_items[task]
            if wrap_exception:
                exc = task.exception()
                if exc:
                    raise MapInnerException(orig_item) from exc
            yield task.result(), orig_item
        if not pending:
            break


@dataclass
class TimeoutBarrier:
    """Allows concurrent tasks to block until enough are waiting.

    Unlike a standard barrier, TimeoutBarrier also releases the barrier if
    enough time has passed since the last task has started waiting for the
    barrier."""

    # When we have this many `arrive_and_wait`s waiting, then we allow them to pass through.
    parties: int

    # The amount of time to wait after the last `arrive_and_wait` call before we release the barrier.
    max_wait: float

    # Called once when the barrier is released. The return value is ignored.
    callback: Callable[[], Any] = lambda: None

    _event: asyncio.Event = dataclasses.field(init=False, default=None)
    _release_lock: asyncio.Lock = dataclasses.field(init=False, default=None)
    _count: int = dataclasses.field(init=False, default=0)
    _deadline_release: Optional[asyncio.Task] = dataclasses.field(
        init=False, default=None
    )

    def _setup(self):
        # We create these objects here, instead of in the constructor, so that
        # we can invoke the constructor outside of an async function.
        if self._event is None:
            self._event = asyncio.Event()
        if self._release_lock is None:
            self._release_lock = asyncio.Lock()

    async def arrive_and_wait(self):
        """Wait for the barrier to release.

        When `parties` calls to this function are waiting, or `max_wait` has
        elapsed since the last blocking call to this function, then the
        barrier will release."""
        self._setup()

        # If we're in the middle of releasing the barrier, wait until that's finished
        async with self._release_lock:
            pass

        # Automatically release after `max_wait`.
        if self._deadline_release:
            self._deadline_release.cancel()
        self._deadline_release = asyncio.create_task(self._release_after(self.max_wait))

        self._count += 1
        if await self._maybe_release():
            self._decrement()
            return

        await self._event.wait()
        self._check_for_exception_in_deadline()
        self._decrement()

    async def wait(self):
        """Wait for the barrier to release, but don't count towards `parties` or `max_wait`."""

        # If we're in the middle of releasing the barrier, wait until that's finished
        async with self._release_lock:
            pass
        await self._event.wait()
        self._check_for_exception_in_deadline()

    async def _release_after(self, wait: float):
        await asyncio.sleep(wait)
        await self._release(True)

    async def deregister(self):
        self._setup()
        self.parties -= 1
        await self._maybe_release()

    @property
    def empty(self):
        return self.parties == 0

    @property
    def num_waiting(self):
        return self._count

    async def _maybe_release(self):
        if self._count == self.parties:
            await self._release(False)
            return True
        return False

    async def _release(self, timeout: bool):
        await self._release_lock.acquire()
        try:
            callback_result = self.callback()  # type: ignore[misc]
            if inspect.isawaitable(callback_result):
                # We need to await on this so that the function actually runs.
                await callback_result
        finally:
            # If we don't do this, we'll have deadlock
            self._event.set()
            if not timeout and self._deadline_release:
                self._deadline_release.cancel()
                self._deadline_release = None

    def _check_for_exception_in_deadline(self):
        if (
            self._deadline_release
            and self._deadline_release.done()
            and self._deadline_release.exception()
        ):
            # The timeout expired for the parties to arrive, so we called `callback` and set `self._event`.
            # However, `callback` threw an exception, so propagate it further here.
            # Also, clear `self._deadline_release` so that only one waiter will get the exception.
            # This is consistent with the behavior when the barrier is released due to enough parties arriving.
            exc = self._deadline_release.exception()
            self._deadline_release = None
            raise exc

    def _decrement(self):
        self._count -= 1
        if self._count == 0:
            self._event.clear()
            self._release_lock.release()

    @property
    def currently_releasing(self):
        return self._release_lock.locked()
