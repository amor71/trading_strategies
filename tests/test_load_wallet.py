import json

from solana.keypair import Keypair

wallet_filename = "tests/solana-wallet/keypair.json"


async def test_load_wallet_json():
    f = open(wallet_filename)
    data = json.load(f)

    data = bytes(data)
    print(f"data length: {len(data)}, {data}")

    kp = Keypair.from_secret_key(data)

    print(f"public-key:{kp.public_key}")
    print(f"secret:{kp.secret_key}")
    return True
