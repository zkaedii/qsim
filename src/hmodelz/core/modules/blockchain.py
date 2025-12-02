"""
Blockchain Connector

This module provides secure blockchain integration with comprehensive features:
- Block creation and mining
- Merkle tree computation
- Chain verification
- Transaction management
"""

import logging
import time
from typing import Any, Dict, List

try:
    import json
except ImportError:
    json = None  # type: ignore

try:
    import hashlib
except ImportError:
    hashlib = None  # type: ignore

from .security import secure_operation


logger = logging.getLogger(__name__)


class BlockchainConnector:
    """Secure blockchain integration with comprehensive features."""

    def __init__(self):
        self.chain: List[Dict] = []
        self.pending_transactions: List[Dict] = []
        self.mining_difficulty = 4
        self.mining_reward = 10.0
        self.network_id = "hmodel_network"
        self.peer_nodes: set = set()

    def create_genesis_block(self) -> Dict[str, Any]:
        """Create the genesis block."""
        genesis_block = {
            "index": 0,
            "timestamp": time.time(),
            "transactions": [],
            "previous_hash": "0",
            "nonce": 0,
            "merkle_root": "",
            "difficulty": self.mining_difficulty,
        }

        genesis_block["hash"] = self._calculate_hash(genesis_block)
        return genesis_block

    @secure_operation
    def create_block(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new block with comprehensive validation."""
        if not self.chain:
            self.chain.append(self.create_genesis_block())

        new_block = {
            "index": len(self.chain),
            "timestamp": time.time(),
            "data": data,
            "transactions": self.pending_transactions.copy(),
            "previous_hash": self.chain[-1]["hash"] if self.chain else "0",
            "nonce": 0,
            "difficulty": self.mining_difficulty,
            "merkle_root": self._calculate_merkle_root(self.pending_transactions),
        }

        new_block = self._mine_block(new_block)

        self.chain.append(new_block)
        self.pending_transactions = []

        logger.info(f"Block {new_block['index']} created with hash {new_block['hash'][:16]}...")

        return new_block

    def _mine_block(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """Mine block using proof of work."""
        target = "0" * self.mining_difficulty

        while True:
            block["nonce"] += 1
            block_hash = self._calculate_hash(block)

            if block_hash.startswith(target):
                block["hash"] = block_hash
                logger.info(f"Block mined with nonce {block['nonce']}")
                return block

    def _calculate_hash(self, block: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash of a block."""
        if not hashlib or not json:
            return ""
        block_copy = block.copy()
        if "hash" in block_copy:
            del block_copy["hash"]

        block_string = json.dumps(block_copy, sort_keys=True).encode("utf-8")
        return hashlib.sha256(block_string).hexdigest()

    def _calculate_merkle_root(self, transactions: List[Dict]) -> str:
        """Calculate Merkle root of transactions."""
        if not transactions:
            if not hashlib:
                return ""
            return hashlib.sha256("".encode()).hexdigest()

        if not hashlib or not json:
            return ""

        transaction_hashes = [
            hashlib.sha256(json.dumps(tx, sort_keys=True).encode()).hexdigest()
            for tx in transactions
        ]

        while len(transaction_hashes) > 1:
            if len(transaction_hashes) % 2 != 0:
                transaction_hashes.append(transaction_hashes[-1])

            new_level = []
            for i in range(0, len(transaction_hashes), 2):
                combined = transaction_hashes[i] + transaction_hashes[i + 1]
                new_level.append(hashlib.sha256(combined.encode()).hexdigest())

            transaction_hashes = new_level

        return transaction_hashes[0]

    def verify_chain(self) -> bool:
        """Verify blockchain integrity."""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block["hash"] != self._calculate_hash(current_block):
                logger.error(f"Invalid hash at block {i}")
                return False

            if current_block["previous_hash"] != previous_block["hash"]:
                logger.error(f"Invalid previous hash at block {i}")
                return False

            target = "0" * current_block["difficulty"]
            if not current_block["hash"].startswith(target):
                logger.error(f"Invalid proof of work at block {i}")
                return False

        return True

    def add_transaction(self, transaction: Dict[str, Any]) -> str:
        """Add transaction to pending pool."""
        if not hashlib or not json:
            return ""
        transaction_id = hashlib.sha256(
            json.dumps(transaction, sort_keys=True).encode()
        ).hexdigest()

        transaction["id"] = transaction_id
        transaction["timestamp"] = time.time()

        self.pending_transactions.append(transaction)

        logger.info(f"Transaction {transaction_id[:16]}... added to pool")
        return transaction_id

    def get_balance(self, address: str) -> float:
        """Get balance for an address by scanning the blockchain."""
        balance = 0.0
        for block in self.chain:
            for tx in block.get("transactions", []):
                if tx.get("to") == address:
                    balance += tx.get("amount", 0)
                if tx.get("from") == address:
                    balance -= tx.get("amount", 0)
        return balance

    def get_block_by_index(self, index: int) -> Dict[str, Any]:
        """Get a block by its index."""
        if 0 <= index < len(self.chain):
            return self.chain[index]
        raise IndexError(f"Block index {index} out of range")

    def get_latest_block(self) -> Dict[str, Any]:
        """Get the latest block in the chain."""
        if self.chain:
            return self.chain[-1]
        return self.create_genesis_block()
