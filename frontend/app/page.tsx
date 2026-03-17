"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import axios from "axios";

const API = process.env.NEXT_PUBLIC_API_URL || "/api";

/** 在浏览器内从视频提取单帧为 JPEG，避免上传整段视频，大幅提升预览速度 */
async function extractFrameFromVideo(videoFile: File): Promise<{ blob: Blob; frameIndex: number; totalFrames: number }> {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.preload = "metadata";
    const url = URL.createObjectURL(videoFile);
    video.src = url;

    video.onloadedmetadata = () => {
      const duration = video.duration || 1;
      const fps = 30;
      const totalFrames = Math.max(1, Math.floor(duration * fps));
      const frameIndex = Math.floor(totalFrames / 2);
      const time = Math.max(0.001, Math.min(duration - 0.001, (frameIndex / totalFrames) * duration));
      video.currentTime = time;
    };

    video.onseeked = () => {
      const maxDim = 720;
      let w = video.videoWidth;
      let h = video.videoHeight;
      if (w > maxDim || h > maxDim) {
        const s = maxDim / Math.max(w, h);
        w = Math.round(w * s);
        h = Math.round(h * s);
      }
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        URL.revokeObjectURL(url);
        reject(new Error("Canvas 不可用"));
        return;
      }
      ctx.drawImage(video, 0, 0, w, h);
      const totalFrames = Math.max(1, Math.floor((video.duration || 1) * 30));
      const frameIndex = Math.floor(totalFrames / 2);
      canvas.toBlob(
        (blob) => {
          URL.revokeObjectURL(url);
          if (blob) resolve({ blob, frameIndex, totalFrames });
          else reject(new Error("无法生成预览帧"));
        },
        "image/jpeg",
        0.82
      );
    };

    video.onerror = () => {
      URL.revokeObjectURL(url);
      reject(video.error);
    };
  });
}

type Stage = "idle" | "previewing" | "previewed" | "processing" | "finished" | "error";
type Model = "person_fast" | "person_quality" | "general_object" | "general_object_hq";

const MODELS: { value: Model; label: string; desc: string }[] = [
  { value: "person_fast", label: "人物抠像（快速）", desc: "适合人物视频，速度优先" },
  { value: "person_quality", label: "人物抠像（高质量）", desc: "人物视频，质量更高但更慢" },
  { value: "general_object", label: "通用物品抠像（快速）", desc: "U2-Net 单模型，速度快" },
  { value: "general_object_hq", label: "通用物品抠像（高质量）", desc: "U2-Net + IS-Net 双模型融合，质量最高但更慢" },
];

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [stage, setStage] = useState<Stage>("idle");
  const [model, setModel] = useState<Model>("person_fast");
  const [pct, setPct] = useState(0);
  const [dlUrl, setDlUrl] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [origImg, setOrigImg] = useState<string | null>(null);
  const [prevImg, setPrevImg] = useState<string | null>(null);
  const [frameInfo, setFrameInfo] = useState("");
  const [showOrig, setShowOrig] = useState(false);
  const ref = useRef<HTMLInputElement>(null);

  const reset = useCallback(() => {
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    if (dlUrl) URL.revokeObjectURL(dlUrl);
    setFile(null); setVideoUrl(null); setStage("idle"); setPct(0);
    setDlUrl(null); setErr(null); setOrigImg(null); setPrevImg(null);
    setFrameInfo(""); setShowOrig(false);
  }, [dlUrl, videoUrl]);

  useEffect(() => () => {
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    if (dlUrl) URL.revokeObjectURL(dlUrl);
  }, [dlUrl, videoUrl]);

  useEffect(() => {
    const base = API.startsWith("http") ? API.replace(/\/api\/?$/, "") : (typeof window !== "undefined" ? window.location.origin : "");
    if (base) axios.get(`${base}/warmup`).catch(() => {});
  }, []);

  const pick = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    if (dlUrl) URL.revokeObjectURL(dlUrl);
    setFile(f); setVideoUrl(URL.createObjectURL(f));
    setDlUrl(null); setErr(null); setStage("idle"); setPct(0);
    setOrigImg(null); setPrevImg(null);
  };

  const preview = async () => {
    if (!file) return;
    setStage("previewing"); setErr(null); setOrigImg(null); setPrevImg(null);
    try {
      let frameIndex = 0;
      let totalFrames = 1;
      let r: { data: { original: string; preview: string; frame_index?: number; total_frames?: number } };
      try {
        const extracted = await extractFrameFromVideo(file);
        const fd = new FormData();
        fd.append("file", extracted.blob, "frame.jpg");
        fd.append("model_kind", model);
        r = await axios.post(`${API}/preview-frame`, fd, { timeout: 120000 });
        frameIndex = extracted.frameIndex;
        totalFrames = extracted.totalFrames;
      } catch (_) {
        const fd = new FormData();
        fd.append("file", file);
        fd.append("model_kind", model);
        r = await axios.post(`${API}/preview`, fd, { timeout: 120000 });
        frameIndex = r.data.frame_index ?? 0;
        totalFrames = r.data.total_frames ?? 1;
      }
      setOrigImg(r.data.original);
      setPrevImg(r.data.preview);
      setFrameInfo(`第 ${frameIndex + 1} / ${totalFrames} 帧`);
      setStage("previewed");
      setShowOrig(false);
    } catch (e: any) {
      setStage("error");
      setErr(e?.response?.data?.detail ?? e?.message ?? "预览失败");
    }
  };

  const process = async () => {
    if (!file) return;
    setStage("processing"); setPct(0); setErr(null); setDlUrl(null);
    try {
      const fd = new FormData();
      fd.append("file", file); fd.append("model_kind", model);
      const r = await axios.post(`${API}/upload`, fd, {
        responseType: "blob",
        timeout: 600000,
        onUploadProgress: (e) => {
          if (!e.total) return;
          setPct(Math.round((e.loaded / e.total) * 100));
        },
      });
      const url = URL.createObjectURL(new Blob([r.data], { type: "application/zip" }));
      setDlUrl(url); setStage("finished");
      const a = document.createElement("a");
      a.href = url; a.download = "output_sequence.zip";
      document.body.appendChild(a); a.click(); a.remove();
    } catch (e: any) {
      setStage("error");
      if (e?.response?.data instanceof Blob) {
        const t = await e.response.data.text();
        try { setErr(JSON.parse(t).detail ?? "失败"); } catch { setErr(t || "失败"); }
      } else {
        setErr(e?.message ?? "处理失败");
      }
    }
  };

  const busy = stage === "previewing" || stage === "processing";

  return (
    <div className="flex flex-col gap-6">
      {/* 标题 */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Jing&apos;s Video Matting Studio</h1>
        <p className="mt-1 text-sm text-gray-500">
          上传视频，选择模型，先预览单帧效果，满意后处理全部帧并下载透明 PNG 序列帧 ZIP。
        </p>
      </div>

      <div className="grid gap-5 lg:grid-cols-[1fr_1.3fr]">
        {/* ===== 左栏 ===== */}
        <div className="flex flex-col gap-4">
          {/* 上传区 */}
          <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
            <label className="group flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-gray-300 bg-gray-50 px-6 py-8 transition hover:border-primary hover:bg-indigo-50/50">
              <input ref={ref} type="file" accept="video/*" className="hidden" onChange={pick} />
              <svg xmlns="http://www.w3.org/2000/svg" className="mb-2 h-8 w-8 text-gray-400 transition group-hover:text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
              </svg>
              <span className="text-sm font-medium text-gray-700">
                Drag / Paste / Click
              </span>
              <span className="mt-1 text-xs text-gray-400">Upload</span>
            </label>

            {file && (
              <div className="mt-3 flex items-center justify-between rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-xs text-gray-600">
                <span className="truncate font-medium">{file.name}</span>
                <div className="flex items-center gap-2">
                  <span className="text-gray-400">({(file.size / 1048576).toFixed(1)} MB)</span>
                  <button onClick={() => { reset(); if (ref.current) ref.current.value = ""; }}
                    className="text-gray-400 hover:text-red-500">✕</button>
                </div>
              </div>
            )}

            {videoUrl && (
              <video src={videoUrl} controls muted className="mt-3 max-h-36 w-full rounded-lg border border-gray-200 object-contain" />
            )}
          </div>

          {/* 模型选择 */}
          <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
            <label className="mb-2 block text-xs font-semibold text-gray-700">选择抠像模型</label>
            <div className="flex flex-col gap-2">
              {MODELS.map((m) => (
                <label
                  key={m.value}
                  className={`flex cursor-pointer items-start gap-3 rounded-xl border px-4 py-3 transition ${
                    model === m.value
                      ? "border-primary bg-indigo-50 ring-1 ring-primary/30"
                      : "border-gray-200 bg-white hover:border-gray-300"
                  } ${busy ? "pointer-events-none opacity-60" : ""}`}
                >
                  <input
                    type="radio" name="model" value={m.value}
                    checked={model === m.value}
                    onChange={() => { setModel(m.value); setOrigImg(null); setPrevImg(null); if (stage === "previewed") setStage("idle"); }}
                    className="mt-0.5 accent-primary"
                  />
                  <div>
                    <span className="text-sm font-medium text-gray-800">{m.label}</span>
                    <p className="text-xs text-gray-500">{m.desc}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* 按钮 */}
          <div className="flex gap-3">
            <button onClick={preview} disabled={!file || busy}
              className="flex-1 rounded-xl border-2 border-primary bg-white px-4 py-2.5 text-sm font-semibold text-primary transition hover:bg-indigo-50 disabled:cursor-not-allowed disabled:border-gray-300 disabled:text-gray-400">
              {stage === "previewing" ? "预览中…" : "预览效果"}
            </button>
            <button onClick={process} disabled={!file || busy}
              className="flex-1 rounded-xl bg-primary px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-primary-dark disabled:cursor-not-allowed disabled:bg-gray-300 disabled:text-gray-500">
              {stage === "processing" ? "处理中…" : "处理全部帧"}
            </button>
          </div>

          {/* 进度 */}
          {(stage === "previewing" || stage === "processing") && (
            <div className="rounded-xl border border-gray-200 bg-white px-4 py-3 shadow-sm">
              <div className="flex items-center justify-between text-xs text-gray-500">
                <span>{stage === "previewing" ? "正在生成预览，首次约 10–30 秒…" : "Processing..."}</span>
                <span>{stage === "processing" && pct >= 100 ? "—" : `${pct}%`}</span>
              </div>
              <div className="mt-1.5 h-2 overflow-hidden rounded-full bg-gray-100">
                {stage === "processing" && pct >= 100 ? (
                  <div className="h-2 w-full animate-pulse rounded-full bg-primary/40" />
                ) : (
                  <div className="h-2 rounded-full bg-primary transition-all" style={{ width: `${pct}%` }} />
                )}
              </div>
            </div>
          )}

          {err && (
            <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-xs text-red-600">{err}</div>
          )}

          {stage === "finished" && (
            <div className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-xs text-emerald-700">
              处理完成！ZIP 已自动下载。
            </div>
          )}

          {dlUrl && (
            <a href={dlUrl} download="output_sequence.zip"
              className="flex items-center justify-center gap-2 rounded-xl border-2 border-primary bg-white px-4 py-2.5 text-sm font-semibold text-primary transition hover:bg-indigo-50">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2M7 10l5 5m0 0 5-5m-5 5V3" />
              </svg>
              下载 output_sequence.zip
            </a>
          )}

          {stage === "finished" && (
            <button onClick={() => { reset(); if (ref.current) ref.current.value = ""; }}
              className="rounded-xl border border-gray-300 px-4 py-2 text-xs text-gray-500 transition hover:border-gray-400 hover:text-gray-700">
              处理下一个视频
            </button>
          )}
        </div>

        {/* ===== 右栏：预览 ===== */}
        <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-bold text-gray-800">Preview</h2>
            {prevImg && <span className="text-xs text-gray-400">{frameInfo}</span>}
          </div>

          {!prevImg ? (
            <div className="flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-gray-200 bg-gray-50 py-28 text-center">
              <svg xmlns="http://www.w3.org/2000/svg" className="mb-3 h-10 w-10 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                <path strokeLinecap="round" strokeLinejoin="round" d="m2.25 15.75 5.159-5.159a2.25 2.25 0 0 1 3.182 0l5.159 5.159m-1.5-1.5 1.409-1.409a2.25 2.25 0 0 1 3.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0 0 22.5 18.75V5.25A2.25 2.25 0 0 0 20.25 3H3.75A2.25 2.25 0 0 0 1.5 5.25v13.5A2.25 2.25 0 0 0 3.75 21Z" />
              </svg>
              <p className="text-sm text-gray-400">选择视频后点击「预览效果」</p>
              <p className="mt-1 text-xs text-gray-300">先看单帧效果，满意再处理全部</p>
            </div>
          ) : (
            <div className="flex flex-col gap-3">
              {/* 切换标签 */}
              <div className="flex rounded-lg bg-gray-100 p-1">
                <button onClick={() => setShowOrig(true)}
                  className={`flex-1 rounded-md px-4 py-2 text-xs font-semibold transition ${showOrig ? "bg-white text-gray-800 shadow-sm" : "text-gray-500 hover:text-gray-700"}`}>
                  Original
                </button>
                <button onClick={() => setShowOrig(false)}
                  className={`flex-1 rounded-md px-4 py-2 text-xs font-semibold transition ${!showOrig ? "bg-white text-gray-800 shadow-sm" : "text-gray-500 hover:text-gray-700"}`}>
                  Transparent
                </button>
              </div>

              {/* 图片 */}
              <div className="relative overflow-hidden rounded-xl border border-gray-200 bg-gray-50">
                <img
                  src={showOrig ? origImg! : prevImg!}
                  alt={showOrig ? "Original" : "Preview"}
                  className="w-full object-contain"
                  style={{ maxHeight: "460px" }}
                />
                <span className="absolute left-3 top-3 rounded-md bg-black/50 px-2 py-1 text-[10px] font-semibold text-white backdrop-blur">
                  {showOrig ? "Original" : "Transparent"}
                </span>
              </div>

              <p className="text-center text-xs text-gray-400">
                {showOrig ? "原始视频帧" : "棋盘格 = 透明区域 · 满意后点击「处理全部帧」"}
              </p>
            </div>
          )}

          {/* 流程说明 */}
          <div className="mt-4 rounded-xl bg-gray-50 p-4">
            <h3 className="mb-2 text-xs font-bold text-gray-600">How to remove a background from a video</h3>
            <div className="grid grid-cols-3 gap-3">
              <div className="text-center">
                <span className="mb-1 inline-block rounded-full bg-primary/10 px-2.5 py-0.5 text-[10px] font-bold text-primary">Step 1</span>
                <p className="text-[11px] text-gray-500">上传视频文件</p>
              </div>
              <div className="text-center">
                <span className="mb-1 inline-block rounded-full bg-primary/10 px-2.5 py-0.5 text-[10px] font-bold text-primary">Step 2</span>
                <p className="text-[11px] text-gray-500">选择模型并预览</p>
              </div>
              <div className="text-center">
                <span className="mb-1 inline-block rounded-full bg-primary/10 px-2.5 py-0.5 text-[10px] font-bold text-primary">Step 3</span>
                <p className="text-[11px] text-gray-500">处理全部并下载</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
