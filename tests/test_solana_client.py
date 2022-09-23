import json

from solana.rpc.async_api import AsyncClient

network = "https://api.devnet.solana.com"


async def test_connect():
    async with AsyncClient(network) as client:
        connected = await client.is_connected()
        print(f"Connectivity to {network}:{connected}")

        if connected:
            nodes = await client.get_cluster_nodes()
            print(f"cluster:{json.dumps(nodes, indent=4)}")
        else:
            raise AssertionError(f"failed to connect to Solana on {network}")

    return True
