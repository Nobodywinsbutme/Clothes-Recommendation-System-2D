"use client";
import { useState } from "react";
import { api } from "@/lib/api";

export default function RegisterPage() {
  const [displayName, setDisplayName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [isError, setIsError] = useState(false);

  async function onSubmit() {
    try {
      const r = await api.register(email, password, displayName);
      localStorage.setItem("token", r.token);
      setIsError(false);
      setMsg(`Tạo tài khoản thành công. Xin chào ${r.displayName}!`);
    } catch (e) {
      setIsError(true);
      setMsg((e as Error).message);
    }
  }

  return (
    <main>
      <h1>Tạo tài khoản</h1>
      <form onSubmit={(e) => { e.preventDefault(); onSubmit(); }}>
        <input placeholder="Tên hiển thị" value={displayName} onChange={(e) => setDisplayName(e.target.value)} required />
        <input type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        <input type="password" placeholder="Mật khẩu (tối thiểu 8 ký tự)" minLength={8} value={password} onChange={(e) => setPassword(e.target.value)} required />
        <button type="submit">Tạo tài khoản</button>
      </form>
      {msg && <p className={isError ? "error" : ""}>{msg}</p>}
    </main>
  );
}
