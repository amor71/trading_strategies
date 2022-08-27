from solana.rpc.async_api import AsyncClient

env = "https://api.devnet.solana.com"


async def test_connect():
    async with AsyncClient(env) as client:
        res = await client.is_connected()
    print(res)  # True
