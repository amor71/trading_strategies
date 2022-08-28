from base64 import b64decode

from solana.publickey import PublicKey
from solana.rpc.async_api import AsyncClient
from solana.system_program import (CreateAccountWithSeedParams,
                                   create_account_with_seed)
from solana.transaction import AccountMeta, Transaction, TransactionInstruction

from common import solana_utils

network = "https://api.devnet.solana.com"
programId: str = "9n73P7niEy9fgDofrXvk2xaaktKb3XzZ7fGT1nuDfXnJ"
wallet_filename = "tests/solana-wallet/keypair.json"


async def test_program_deployment():
    async with AsyncClient(network) as client:
        res = await client.is_connected()
        print(f"Connectivity to {network}:{res}")

        publicKey = PublicKey(programId)
        print(f"program public key: {publicKey}")

        programInfo = await client.get_account_info(publicKey)

        if not programInfo:
            raise AssertionError(f"failed to locate program {programId}")

    return True


async def test_greeter_program():
    async with AsyncClient(network) as client:
        res = await client.is_connected()
        print(f"Connectivity to {network}:{res}")

        program_publicKey = PublicKey(programId)
        programInfo = await client.get_account_info(program_publicKey)

        if not programInfo:
            raise AssertionError(f"failed to locate program {programId}")

        payer = solana_utils.load_wallet(wallet_filename)

        greetedPubKey = PublicKey.create_with_seed(
            payer.public_key, "hello3", program_publicKey
        )
        print(f"greeted public-key : {greetedPubKey}")

        cost: int = (await client.get_minimum_balance_for_rent_exemption(4))[
            "result"
        ]
        print(f"cost in lamports: {cost}")

        result = await client.request_airdrop(payer.public_key, 2 * cost)
        print(f"funding result: {result}")

        result = await client.get_account_info(greetedPubKey)
        print(f"result getter: {result}")

        print(f"{b64decode(result['result']['value']['data'][0])}")

        instruction = create_account_with_seed(
            CreateAccountWithSeedParams(
                from_pubkey=payer.public_key,
                new_account_pubkey=greetedPubKey,
                base_pubkey=payer.public_key,
                seed={"length": 6, "chars": "hello3"},
                lamports=cost,
                space=4,
                program_id=program_publicKey,
            )
        )
        # trans = Transaction().add(instruction)
        # result = await client.send_transaction(trans, payer)
        # print(result)

        instruction = TransactionInstruction(
            keys=[
                AccountMeta(
                    pubkey=greetedPubKey, is_signer=False, is_writable=True
                )
            ],
            program_id=program_publicKey,
            data="",
        )

        trans = Transaction().add(instruction)
        result = await client.send_transaction(trans, payer)
        print(result)
        result = await client.confirm_transaction(
            result["result"], "confirmed"
        )
        print(result)

    return True
