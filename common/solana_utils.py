import json

from solana.keypair import Keypair


def load_wallet(wallet_filename: str) -> Keypair:
    with open(wallet_filename) as f:
        data = json.load(f)

        return Keypair.from_secret_key(bytes(data))
