import asyncio
import json
import time
from statistics import mean

from app.moderation.local.moderator import LocalModerator
from app.moderation.cloud.moderator import CloudModerator


async def benchmark():
    with open("tests/benchmark_dataset.json", "r") as f:
        dataset = json.load(f)

    local = LocalModerator()
    cloud = CloudModerator()

    local_times = []
    cloud_times = []

    local_correct = 0
    cloud_correct = 0

    print("\n=== BENCHMARK START ===\n")

    for item in dataset:
        content = item["content"]
        expected = item["expected"]

        # LOCAL
        start = time.perf_counter()
        local_result = await local.moderate(content)
        local_latency = (time.perf_counter() - start) * 1000
        local_times.append(local_latency)

        if local_result.decision.value == expected:
            local_correct += 1

        # CLOUD
        start = time.perf_counter()
        cloud_result = await cloud.moderate(content)
        cloud_latency = (time.perf_counter() - start) * 1000
        cloud_times.append(cloud_latency)

        if cloud_result.decision.value == expected:
            cloud_correct += 1

        print(f"\nINPUT: {content}")
        print(
            f"LOCAL => {local_result.decision.value} "
            f"({local_latency:.2f} ms)"
        )
        print(
            f"CLOUD => {cloud_result.decision.value} "
            f"({cloud_latency:.2f} ms)"
        )

    print("\n=== RESULTS ===\n")

    print("LOCAL:")
    print(f"Accuracy: {local_correct}/{len(dataset)}")
    print(f"Avg latency: {mean(local_times):.2f} ms")

    print("\nCLOUD:")
    print(f"Accuracy: {cloud_correct}/{len(dataset)}")
    print(f"Avg latency: {mean(cloud_times):.2f} ms")


if __name__ == "__main__":
    asyncio.run(benchmark())