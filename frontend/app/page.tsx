"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

export default function Home() {
  const [backend, setBackend] = useState("đang kiểm tra...");
  const [ai, setAi] = useState("đang kiểm tra...");

  useEffect(() => {
    api.health().then((r) => setBackend(r.status)).catch(() => setBackend("không kết nối được"));
    api.pingAi().then((r) => setAi(String(r.aiService))).catch(() => setAi("không kết nối được"));
  }, []);

  return (
    <main>
      <h1>Fashion Suggest</h1>
      <p>Khung dự án đã chạy. Trạng thái các service:</p>
      <div className="status"><span>Backend (Spring Boot)</span><span>{backend}</span></div>
      <div className="status"><span>AI service (FastAPI)</span><span>{ai}</span></div>
      <div className="row">
        <Link className="btn" href="/login">Đăng nhập</Link>
        <Link className="btn" href="/register">Tạo tài khoản</Link>
      </div>
    </main>
  );
}
