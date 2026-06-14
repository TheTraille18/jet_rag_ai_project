from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PDF_PATH = PROJECT_ROOT / "doc" / "Tesla_earnings.pdf"
DB_PATH = PROJECT_ROOT / "db" / "Tesla_earnings_db"
COLLECTION_NAME = "document_dq_collection_tesla_earnings"

AWS_REGION = "us-east-1"
MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

DEFAULT_QUERY = "What is the future of autonomous vehicles?"
