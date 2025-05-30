from dotenv import load_dotenv
import os
# Load environment variables
if os.getenv("APP_ENV", "dev").lower() == "dev":
    load_dotenv()