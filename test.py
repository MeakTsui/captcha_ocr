#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 用法:
#   python login.py --base http://localhost:8787
# 依赖:
#   pip install requests eth-account

import argparse
import json
import sys
import time
import requests
from eth_account import Account
from eth_account.messages import encode_defunct

def main():
    parser = argparse.ArgumentParser(description="Wallet login (generate key, sign message, login).")
    parser.add_argument("--base", default="https://app.tsuishiqiang.workers.dev", help="API base url, e.g. https://app.tsuishiqiang.workers.dev")
    args = parser.parse_args()

    base = args.base.rstrip("/")
    # 1) 生成私钥与地址（注意：仅用于测试）
    acct = Account.create()  # 随机新账户
    private_key = acct.key.hex()
    address = acct.address
    print(f"[INFO] Generated address: {address}")
    print(f"[INFO] Private key     : {private_key}  (请妥善保管，仅测试使用)")

    # 2) 获取 nonce 与 message
    nonce_url = f"{base}/api/v1/auth/nonce"
    r = requests.post(nonce_url, json={"wallet_address": address}, timeout=15)
    if r.status_code != 200:
        print("[ERROR] nonce 请求失败:", r.status_code, r.text)
        sys.exit(1)
    data = r.json()
    nonce = data.get("nonce")
    message = data.get("message")
    timestamp = data.get("timestamp")
    print(f"[INFO] Nonce: {nonce}, Timestamp: {timestamp}")
    print(f"[INFO] Message to sign:\n{message}")

    # 3) 使用 EIP-191 对 message 签名
    sign_msg = encode_defunct(text=message)
    signed = Account.sign_message(sign_msg, private_key=private_key)
    signature = signed.signature.hex()
    print(f"[INFO] Signature: {signature}")

    # 4) 登录
    login_url = f"{base}/api/v1/auth/wallet/login"
    payload = {
        "wallet_address": address,
        "signature": signature,
        "message": message,
        "device_info": "python-cli"
    }
    r = requests.post(login_url, json=payload, timeout=15)
    if r.status_code != 200:
        print("[ERROR] 登录失败:", r.status_code, r.text)
        sys.exit(1)
    tokens = r.json()
    print("[INFO] 登录成功，返回：")
    print(json.dumps(tokens, indent=2, ensure_ascii=False))

    access = tokens.get("access_token")
    if not access:
        print("[WARN] 未获取到 access_token")
        sys.exit(0)

    # 5) 调用 /auth/me 验证
    me_url = f"{base}/api/v1/auth/me"
    r = requests.get(me_url, headers={"authorization": f"Bearer {access}"}, timeout=15)
    print(f"[INFO] /auth/me 状态: {r.status_code}")
    try:
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    except Exception:
        print(r.text)

if __name__ == "__main__":
    main()