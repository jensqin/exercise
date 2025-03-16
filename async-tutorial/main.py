import asyncio
import time


async def say_after(delay, what):
    await asyncio.sleep(delay)
    time.sleep(1)
    print(what)


async def aio_slow():
    print(f"slow started at {time.strftime('%X')}")

    await say_after(1, "slow hello")
    await say_after(1, "slow world")

    print(f"slow finished at {time.strftime('%X')}")


async def aio_fast():
    print(f"fast started at {time.strftime('%X')}")

    async with asyncio.TaskGroup() as tg:
        tg.create_task(say_after(1, "fast hello"))
        tg.create_task(say_after(1, "fast world"))

    print(f"fast finished at {time.strftime('%X')}")


async def main():
    # Schedule calls *concurrently*:

    async with asyncio.TaskGroup() as tg:
        tg.create_task(aio_slow())
        tg.create_task(aio_fast())


if __name__ == "__main__":
    asyncio.run(main())
    # asyncio.run(aio_slow())
    # asyncio.run(aio_fast())
