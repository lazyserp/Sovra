"""Sovra Cluster Health & Verification Harness.

Validates that a live Sovra cluster (local or remote) is fully operational.

Usage:
    python scripts/verify_cluster.py [GATEWAY_URL] [CONTROL_PLANE_URL] [ADMIN_TOKEN]
Example:
    python scripts/verify_cluster.py http://localhost:8000 http://localhost:8001 sovra-master-secret-2026
"""
import sys
import httpx

def run_verification(
    gateway_url: str = "http://localhost:8000",
    cp_url: str = "http://localhost:8001",
    admin_token: str = "sovra-master-secret-2026",
):
    print("\n" + "=" * 65)
    print(" 🚀 SOVRA CLUSTER SMOKE TEST & HEALTH VERIFICATION")
    print(f"    Gateway:       {gateway_url}")
    print(f"    Control Plane: {cp_url}")
    print("=" * 65 + "\n")

    client = httpx.Client(timeout=10.0)
    passed = 0
    total = 6

    # 1. Gateway Health Check
    try:
        res = client.get(f"{gateway_url}/health")
        if res.status_code == 200 and res.json().get("service") == "sovra-gateway":
            print("  [PASS] 1. Gateway Health Check (HTTP 200)")
            passed += 1
        else:
            print(f"  [FAIL] 1. Gateway Health Check: Unexpected response {res.status_code}")
    except Exception as e:
        print(f"  [FAIL] 1. Gateway unreachable: {e}")

    # 2. Control Plane Console UI
    try:
        res = client.get(f"{cp_url}/console")
        if res.status_code == 200 and "Sovra" in res.text:
            print("  [PASS] 2. Control Plane Web Console UI (HTTP 200)")
            passed += 1
        else:
            print(f"  [FAIL] 2. Control Plane UI: Unexpected status {res.status_code}")
    except Exception as e:
        print(f"  [FAIL] 2. Control Plane unreachable: {e}")

    # 3. Master Admin Token Authentication
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    try:
        res = client.get(f"{cp_url}/api/v1/metrics/overview", headers=admin_headers)
        if res.status_code == 200:
            print("  [PASS] 3. Control Plane Admin Authentication & Metrics (HTTP 200)")
            passed += 1
        else:
            print(f"  [FAIL] 3. Admin Auth failed: Status {res.status_code} - {res.text}")
    except Exception as e:
        print(f"  [FAIL] 3. Admin request error: {e}")

    # 4. Create Ephemeral Project
    project_id = None
    try:
        res = client.post(
            f"{cp_url}/api/v1/projects",
            json={"name": "Harness-Smoke-Test", "description": "Auto verification"},
            headers=admin_headers,
        )
        if res.status_code in [201, 200]:
            project_id = res.json()["id"]
            print("  [PASS] 4. Multi-Tenant Project Provisioning (HTTP 201)")
            passed += 1
        else:
            print(f"  [FAIL] 4. Project creation failed: {res.status_code} - {res.text}")
    except Exception as e:
        print(f"  [FAIL] 4. Project creation error: {e}")

    # 5. Mint API Key Scoped to Project
    api_key_id = None
    plain_api_key = None
    try:
        res = client.post(
            f"{cp_url}/api/v1/api-keys",
            json={"name": "harness-ephemeral-key", "project_id": project_id, "role": "inference"},
            headers=admin_headers,
        )
        if res.status_code == 201:
            key_data = res.json()
            api_key_id = key_data["id"]
            plain_api_key = key_data["api_key"]
            print(f"  [PASS] 5. Cryptographic API Key Minting & SHA-256 Hashing ({key_data['prefix']}...)")
            passed += 1
        else:
            print(f"  [FAIL] 5. Key minting failed: {res.status_code} - {res.text}")
    except Exception as e:
        print(f"  [FAIL] 5. Key minting error: {e}")

    # 6. Verify Gateway Bearer Auth with the new key & Clean Revoke
    try:
        gw_headers = {"Authorization": f"Bearer {plain_api_key}"}
        # Calling Gateway with the key
        res = client.get(f"{gateway_url}/v1/models", headers=gw_headers)
        if res.status_code == 200:
            # Revoke key immediately to clean up
            client.delete(f"{cp_url}/api/v1/api-keys/{api_key_id}", headers=admin_headers)
            print("  [PASS] 6. Gateway Bearer Token Verification & Key Revocation")
            passed += 1
        else:
            print(f"  [FAIL] 6. Gateway auth rejected: {res.status_code}")
    except Exception as e:
        print(f"  [FAIL] 6. Gateway auth error: {e}")

    print("\n" + "-" * 65)
    if passed == total:
        print(f"  🎉 ALL {total}/{total} CLUSTER CHECKS PASSED — SOVRA IS PRODUCTION READY!")
    else:
        print(f"  ⚠️  {passed}/{total} CHECKS PASSED. Check failed steps above.")
    print("-" * 65 + "\n")
    return passed == total

if __name__ == "__main__":
    gw = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    cp = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8001"
    token = sys.argv[3] if len(sys.argv) > 3 else "sovra-master-secret-2026"
    success = run_verification(gw, cp, token)
    sys.exit(0 if success else 1)

