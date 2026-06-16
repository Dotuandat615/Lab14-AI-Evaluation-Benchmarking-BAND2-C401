import asyncio
from typing import Dict


class LegacyAgentV1:
    """Legacy baseline agent used for regression benchmarking.

    This version intentionally behaves like an older support bot: it does not
    retrieve local documents, gives generic answers, and simulates higher
    latency/token usage so V1 vs V2 deltas are visible in the release gate.
    """

    def __init__(self):
        self.name = "SupportAgent-v1-legacy"
        self.version = "Agent_V1_Legacy"

    async def query(self, question: str) -> Dict:
        await asyncio.sleep(0.8)

        return {
            "answer": (
                "Toi da ghi nhan cau hoi cua ban. Vui long kiem tra handbook "
                "noi bo hoac lien he phong ban phu trach de duoc ho tro them. "
                f"Cau hoi: {question}"
            ),
            "contexts": [],
            "metadata": {
                "agent_version": self.version,
                "model": "legacy-support-template",
                "tokens_used": 320,
                "sources": [],
                "retrieval_used": False,
            },
        }


if __name__ == "__main__":
    async def _demo():
        agent = LegacyAgentV1()
        resp = await agent.query("Lam the nao de reset mat khau?")
        print(resp)

    asyncio.run(_demo())
