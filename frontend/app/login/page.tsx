"use client";
import { useState } from "react";
import { api } from "@/lib/api";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [isError, setIsError] = useState(false);

  async function onSubmit() {
    try {
      const r = await api.login(email, password);
      localStorage.setItem("token", r.token);
      setIsError(false);
      setMsg(`Xin chào ${r.displayName}!`);
    } catch (e) {
      setIsError(true);
      setMsg((e as Error).message);
    }
  }

  return (
    <main>
      <h1>Đăng nhập</h1>
      <form onSubmit={(e) => { e.preventDefault(); onSubmit(); }}>
        <input type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        <input type="password" placeholder="Mật khẩu" value={password} onChange={(e) => setPassword(e.target.value)} required />
        <button type="submit">Đăng nhập</button>
      </form>
      {msg && <p className={isError ? "error" : ""}>{msg}</p>}
    </main>
  );
}
