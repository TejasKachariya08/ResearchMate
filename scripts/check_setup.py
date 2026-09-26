"""ResearchMate Diagnostic & Setup Verification Script.

Run this script to verify your environment, Redis connection,
Google Gemini API key, embeddings, and arXiv connectivity.

Usage:
    python scripts/check_setup.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Fix Windows console UTF-8 output encoding if needed
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure src directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from researchmate.config import get_settings
from researchmate.utils import check_redis_connection, check_gemini_connection, check_arxiv_connection
from researchmate.rag import get_embeddings


def main() -> int:
    print("=" * 65)
    print("           RESEARCHMATE SETUP VERIFICATION")
    print("=" * 65)

    settings = get_settings()
    print(f"\nConfiguration:")
    print(f"  * Gemini Chat Model:      {settings.gemini_chat_model}")
    print(f"  * Gemini Embedding Model: {settings.gemini_embedding_model}")
    print(f"  * Embedding Dimensions:   {settings.embedding_dimensions}")
    print(f"  * Redis URL:              {settings.redis_url}")
    print(f"  * Index Basename:         {settings.redis_index_basename}")
    print(f"  * Google API Key set:     {'Yes' if bool(settings.google_api_key) else 'NO (Missing!)'}")

    all_passed = True

    # 1. Test Redis
    print("\n[1/4] Checking Redis Connection...")
    redis_ok, redis_msg = check_redis_connection()
    if redis_ok:
        print(f"  [OK] SUCCESS: {redis_msg}")
    else:
        print(f"  [FAIL] FAILED:  {redis_msg}")
        all_passed = False

    # 2. Test Gemini API
    print("\n[2/4] Checking Google Gemini Chat Model...")
    gemini_ok, gemini_msg = check_gemini_connection()
    if gemini_ok:
        print(f"  [OK] SUCCESS: {gemini_msg}")
    else:
        print(f"  [FAIL] FAILED:  {gemini_msg}")
        all_passed = False

    # 3. Test Gemini Embeddings
    print("\n[3/4] Checking Google Gemini Embeddings...")
    if not settings.google_api_key:
        print("  [FAIL] SKIPPED: Missing GOOGLE_API_KEY")
        all_passed = False
    else:
        try:
            emb = get_embeddings(settings)
            vec = emb.embed_query("ResearchMate test query")
            print(f"  [OK] SUCCESS: Embedding generated! Dimension: {len(vec)} (configured: {settings.embedding_dimensions})")
        except Exception as exc:
            print(f"  [FAIL] FAILED:  {exc}")
            all_passed = False

    # 4. Test arXiv API
    print("\n[4/4] Checking arXiv API...")
    arxiv_ok, arxiv_msg = check_arxiv_connection()
    if arxiv_ok:
        print(f"  [OK] SUCCESS: {arxiv_msg}")
    else:
        print(f"  [FAIL] FAILED:  {arxiv_msg}")
        all_passed = False

    print("\n" + "=" * 65)
    if all_passed:
        print(" [ALL CHECKS PASSED] ResearchMate is ready to use!")
        print(" Launch with: streamlit run app.py")
        print("=" * 65)
        return 0
    else:
        print(" [SOME CHECKS FAILED] Please review the errors above.")
        print("=" * 65)
        return 1


if __name__ == "__main__":
    sys.exit(main())
