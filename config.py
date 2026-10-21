"""Project paths and the ten banking intents we deliberately support."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 42
INTENTS = {
    "card_arrival": "Card has not arrived",
    "lost_or_stolen_card": "Lost or stolen card",
    "passcode_forgotten": "Forgotten app passcode",
    "change_pin": "Change card PIN",
    "cash_withdrawal_charge": "Cash withdrawal fee",
    "declined_card_payment": "Card payment declined",
    "pending_transfer": "Transfer is pending",
    "transfer_not_received_by_recipient": "Recipient has not received transfer",
    "failed_transfer": "Transfer failed",
    "terminate_account": "Close account",
}
EXAMPLES = [
    "My new card has not arrived yet.",
    "Someone stole my bank card. What should I do?",
    "I cannot remember the passcode for my app.",
    "How can I change my card PIN?",
    "Why was I charged a fee at the cash machine?",
    "My card payment was declined at the shop.",
    "The transfer is still pending.",
    "The recipient has not received the money I sent.",
    "My bank transfer failed. What happened?",
    "I want to close my account.",
]
