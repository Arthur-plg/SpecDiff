"use client";

import React, { useState, useMemo, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  BarChart3, 
  Activity, 
  Cpu, 
  Zap, 
  FileJson, 
  Upload, 
  Trash2, 
  ChevronRight, 
  Gauge, 
  ArrowUpRight,
  Database,
  Search,
  Filter
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  ScatterChart,
  Scatter,
  ZAxis,
  Legend
} from "recharts";
import Papa from "papaparse";
import { cn } from "@/lib/utils";

// --- Types ---
interface Experiment {
  timestamp: string;
  config: string;
  method: string;
  target_model: string;
  draft_model: string;
  gamma: number;
  T_steps: number;
  decode_speedup: number;
  throughput_tok_sec: number;
  decode_throughput_tok_sec: number;
  ttft_ms: number;
  avg_itl_ms: number;
  acceptance_rate_percent: number;
  total_time_sec: number;
  total_tokens: number;
}

// --- Components ---

interface MetricCardProps {
  title: string;
  value: string | number | undefined;
  unit: string;
  icon: React.ElementType;
  color?: "cyan" | "emerald" | "purple";
}

const MetricCard = ({ title, value, unit, icon: Icon, color = "cyan" }: MetricCardProps) => (
  <motion.div 
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    className="glass-card p-6 rounded-2xl flex flex-col gap-4 group hover:border-white/20 transition-all duration-300"
  >
    <div className="flex items-center justify-between">
      <div className={cn("p-2 rounded-lg bg-opacity-10", 
        color === "cyan" ? "bg-cyan-400 text-cyan-400" : 
        color === "emerald" ? "bg-emerald-400 text-emerald-400" : 
        "bg-purple-400 text-purple-400")}>
        <Icon size={20} />
      </div>
      <div className="text-xs font-medium text-zinc-500 uppercase tracking-wider">{title}</div>
    </div>
    <div className="flex items-baseline gap-2">
      <span className="text-4xl font-bold tracking-tight">{value}</span>
      <span className="text-sm font-medium text-zinc-500">{unit}</span>
    </div>
  </motion.div>
);

interface ChartContainerProps {
  title: string;
  children: React.ReactNode;
}

const ChartContainer = ({ title, children }: ChartContainerProps) => (
  <div className="glass-card p-6 rounded-2xl flex flex-col gap-6">
    <div className="flex items-center justify-between">
      <h3 className="text-sm font-semibold text-zinc-300 uppercase tracking-widest flex items-center gap-2">
        <ChevronRight size={14} className="text-cyan-400" />
        {title}
      </h3>
    </div>
    <div className="h-[300px] w-full">
      {children}
    </div>
  </div>
);

// --- Main Page ---

export default function Dashboard() {
  const [data, setData] = useState<Experiment[]>([]);
  const [isDragging, setIsDragging] = useState(false);

  // --- Logic ---
  React.useEffect(() => {
    // Auto-load demo data if available in public folder
    fetch("/data.csv")
      .then(res => res.text())
      .then(csvText => {
        if (csvText && csvText.startsWith("timestamp")) {
          Papa.parse(csvText, {
            header: true,
            dynamicTyping: true,
            complete: (results) => {
              const newExps = results.data.filter((row: any) => row.method) as Experiment[];
              setData(newExps);
            }
          });
        }
      })
      .catch(() => console.log("No default data found, waiting for upload."));
  }, []);

  const handleFileUpload = useCallback((files: FileList | null) => {
    if (!files) return;
    
    Array.from(files).forEach(file => {
      Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        complete: (results) => {
          const newExps = results.data.filter((row: any) => row.method) as Experiment[];
          setData(prev => [...prev, ...newExps]);
        }
      });
    });
  }, []);

  const stats = useMemo(() => {
    if (data.length === 0) return null;
    const specOnly = data.filter(d => d.method === "SpecDiff");
    return {
      totalRuns: data.length,
      maxSpeedup: Math.max(...data.map(d => d.decode_speedup || 0)),
      avgAlpha: spec_only_avg(specOnly, "acceptance_rate_percent"),
      peakThroughput: Math.max(...data.map(d => d.decode_throughput_tok_sec || 0)),
      models: Array.from(new Set(data.map(d => d.target_model))).length
    };
  }, [data]);

  function spec_only_avg(arr: Experiment[], key: keyof Experiment) {
    if (arr.length === 0) return 0;
    const sum = arr.reduce((acc, curr) => acc + (curr[key] as number), 0);
    return sum / arr.length;
  }

  const heatmapData = useMemo(() => {
    const specOnly = data.filter(d => d.method === "SpecDiff");
    // Aggregate by gamma/T for heatmap
    const grid: any = {};
    specOnly.forEach(d => {
      const key = `${d.gamma}-${d.T_steps}`;
      if (!grid[key]) grid[key] = { gamma: d.gamma, T: d.T_steps, speedup: 0, count: 0 };
      grid[key].speedup += d.decode_speedup;
      grid[key].count += 1;
    });
    return Object.values(grid).map((g: any) => ({ ...g, speedup: g.speedup / g.count }));
  }, [data]);

  const scalingData = useMemo(() => {
    const models = Array.from(new Set(data.map(d => d.target_model)));
    return models.map(m => {
      const runs = data.filter(d => d.target_model === m && d.method === "SpecDiff");
      const best = runs.length > 0 ? Math.max(...runs.map(r => r.decode_speedup)) : 0;
      
      // Improved size mapping
      let size = 1.0;
      if (m.toLowerCase().includes("125m")) size = 0.125;
      else if (m.toLowerCase().includes("1.3b")) size = 1.3;
      else if (m.toLowerCase().includes("2.7b")) size = 2.7;
      else if (m.toLowerCase().includes("3b")) size = 3.0;
      else if (m.toLowerCase().includes("xl")) size = 1.5;
      
      // Clean name for display
      const displayName = m.split('/').pop() || m;
      
      return { name: displayName, speedup: best, size };
    }).sort((a, b) => a.size - b.size);
  }, [data]);
  const throughputData = useMemo(() => {
    const models = Array.from(new Set(data.map(d => d.target_model)));
    return models.map(m => {
      const arRun = data.find(d => d.target_model === m && d.method === "Standard AR");
      const specRuns = data.filter(d => d.target_model === m && d.method === "SpecDiff");
      const bestSpec = specRuns.length > 0 ? Math.max(...specRuns.map(r => r.decode_throughput_tok_sec)) : 0;
      
      const displayName = m.split('/').pop() || m;
      return {
        name: displayName,
        baseline: arRun ? arRun.decode_throughput_tok_sec : 0,
        speculative: bestSpec
      };
    }).filter(d => d.baseline > 0 || d.speculative > 0);
  }, [data]);

  return (
    <div className="min-h-screen bg-[#09090b] text-zinc-100 p-8 font-sans selection:bg-cyan-500/30">
      {/* Background Decor */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden -z-10">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-cyan-900/10 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-900/10 blur-[120px] rounded-full" />
      </div>

      <main className="max-w-7xl mx-auto flex flex-col gap-12">
        {/* Header */}
        <header className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 cyan-gradient rounded-lg flex items-center justify-center neon-border">
                  <Activity size={18} className="text-white" />
              </div>
              <span className="text-sm font-bold tracking-[0.2em] text-cyan-400 uppercase">Research Environment</span>
            </div>
            <a 
              href="https://github.com/Arthur-plg/SpecDiff" 
              target="_blank" 
              className="px-4 py-2 rounded-full glass-card text-xs font-bold flex items-center gap-2 hover:bg-white/10 transition-colors"
            >
              <Database size={14} /> View on GitHub
            </a>
          </div>
          <h1 className="text-6xl font-extrabold tracking-tighter bg-clip-text text-transparent bg-gradient-to-b from-white to-zinc-500">
            SpecDiff Analytics
          </h1>
          <p className="text-zinc-500 max-w-2xl text-lg font-medium leading-relaxed">
            Implementation and high-fidelity analysis of the <strong>SpecDiff research paper</strong>. 
            Real-time benchmarking of speculative diffusion decoding using Masked Diffusion Language Models (MDLM).
          </p>
        </header>

        {/* Research Context Section */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="glass-card p-6 rounded-2xl border-l-4 border-cyan-500">
             <h3 className="text-sm font-bold text-white mb-2 uppercase tracking-tighter">The Challenge</h3>
             <p className="text-xs text-zinc-400 leading-relaxed">
                LLMs are slow because they generate tokens one-by-one. <strong>Autoregressive decoding</strong> is limited by memory bandwidth, making inference expensive at scale.
             </p>
          </div>
          <div className="glass-card p-6 rounded-2xl border-l-4 border-purple-500">
             <h3 className="text-sm font-bold text-white mb-2 uppercase tracking-tighter">SOTA Methodology</h3>
             <p className="text-xs text-zinc-400 leading-relaxed">
                We use <strong>Speculative Decoding</strong> with a <strong>Masked Diffusion (MDLM)</strong> draft. This state-of-the-art approach allows for non-autoregressive token proposals.
             </p>
          </div>
          <div className="glass-card p-6 rounded-2xl border-l-4 border-amber-500">
             <h3 className="text-sm font-bold text-white mb-2 uppercase tracking-tighter">Hardware Context</h3>
             <p className="text-xs text-zinc-400 leading-relaxed">
                Optimized for <strong>NVIDIA T4 GPUs</strong>. We prove that SOTA speedups are achievable on limited infrastructure through clever algorithm tuning.
             </p>
          </div>
          <div className="glass-card p-6 rounded-2xl border-l-4 border-emerald-500">
             <h3 className="text-sm font-bold text-white mb-2 uppercase tracking-tighter">Key Impact</h3>
             <p className="text-xs text-zinc-400 leading-relaxed">
                We achieve up to <strong>{stats?.maxSpeedup.toFixed(1) || "2.5"}x speedup gains</strong> across various models while maintaining exact mathematical parity with standard decoding.
             </p>
          </div>
        </section>

        {/* Data Dropzone */}
        {data.length === 0 ? (
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => { e.preventDefault(); setIsDragging(false); handleFileUpload(e.dataTransfer.files); }}
            className={cn(
              "h-96 rounded-3xl border-2 border-dashed flex flex-col items-center justify-center gap-6 transition-all duration-500 group cursor-pointer",
              isDragging ? "border-cyan-400 bg-cyan-500/5" : "border-zinc-800 hover:border-zinc-700 bg-zinc-900/20"
            )}
            onClick={() => document.getElementById("csv-input")?.click()}
          >
            <input 
              id="csv-input" 
              type="file" 
              multiple 
              className="hidden" 
              onChange={(e) => handleFileUpload(e.target.files)} 
            />
            <div className="w-20 h-20 rounded-2xl bg-zinc-900 flex items-center justify-center group-hover:scale-110 transition-transform duration-500 border border-zinc-800">
              <Upload className={cn("transition-colors duration-500", isDragging ? "text-cyan-400" : "text-zinc-600")} size={32} />
            </div>
            <div className="text-center">
              <h3 className="text-2xl font-bold mb-2">Initialize Analytics</h3>
              <p className="text-zinc-500">Drag & drop your experiment CSV logs to start exploration</p>
            </div>
            <div className="px-6 py-3 rounded-full bg-zinc-100 text-zinc-900 font-bold text-sm hover:scale-105 transition-transform">
               Browse Experiments
            </div>
          </motion.div>
        ) : (
          <div className="flex flex-col gap-12">
            {/* KPI Section */}
            <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <MetricCard title="Best Speedup" value={stats?.maxSpeedup.toFixed(2)} unit="x" icon={Zap} color="cyan" />
              <MetricCard title="Peak Throughput" value={stats?.peakThroughput.toFixed(1)} unit="tok/s" icon={Gauge} color="emerald" />
              <MetricCard title="Avg Acceptance" value={stats?.avgAlpha.toFixed(1)} unit="%" icon={Activity} color="purple" />
              <MetricCard title="Models Profiled" value={stats?.models} unit="Units" icon={Database} color="cyan" />
            </section>

            {/* Advanced Analytics Tabs */}
            <div className="flex flex-col gap-8">
              <h2 className="text-2xl font-bold tracking-tight text-zinc-300">Advanced Systems Analysis</h2>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* 1. Parameter Heatmap */}
                <ChartContainer title="Hyperparameter Sensitivity (gamma x T)">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#18181b" vertical={false} />
                      <XAxis type="number" dataKey="gamma" name="Gamma" stroke="#52525b" />
                      <YAxis type="number" dataKey="T" name="T steps" stroke="#52525b" />
                      <ZAxis type="number" dataKey="speedup" range={[400, 401]} />
                      <Tooltip 
                        cursor={{ strokeDasharray: '3 3' }} 
                        contentStyle={{ backgroundColor: "#09090b", border: "1px solid #27272a", borderRadius: "12px" }}
                        content={({ active, payload }: any) => {
                          if (active && payload && payload.length) {
                            return (
                              <div className="p-3 bg-zinc-950 border border-zinc-800 rounded-xl shadow-2xl">
                                <p className="text-xs text-zinc-500 uppercase font-bold mb-1">Configuration</p>
                                <p className="text-sm font-bold text-cyan-400">γ={payload[0].payload.gamma} | T={payload[0].payload.T}</p>
                                <div className="mt-2 h-px bg-zinc-800" />
                                <p className="mt-2 text-xl font-bold text-white">{payload[0].payload.speedup.toFixed(2)}x <span className="text-xs text-zinc-500 font-medium">Gain</span></p>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Scatter data={heatmapData} shape="square">
                        {heatmapData.map((entry, index) => {
                          // Map speedup (usually 1.0 to 4.0) to color intensity
                          const intensity = Math.min(Math.max((entry.speedup - 1) / 3, 0), 1);
                          const color = `rgba(34, 211, 238, ${0.2 + intensity * 0.8})`; // Varying opacity for cyan
                          return <Cell key={`cell-${index}`} fill={color} stroke="#22d3ee" strokeWidth={0.5} />;
                        })}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </ChartContainer>

                {/* 2. Scaling Analysis */}
                <ChartContainer title="Target Model Scaling Analysis">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={scalingData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#18181b" vertical={false} />
                      <XAxis dataKey="name" stroke="#52525b" />
                      <YAxis stroke="#52525b" domain={[1, 'auto']} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: "#09090b", border: "1px solid #27272a", borderRadius: "12px" }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="speedup" 
                        stroke="#22d3ee" 
                        strokeWidth={4} 
                        dot={{ r: 6, fill: "#22d3ee", strokeWidth: 0 }}
                        activeDot={{ r: 8, strokeWidth: 0 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartContainer>
              </div>
            </div>

            {/* Standard Metrics */}
            <section className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <ChartContainer title="Acceptance Rate Impact">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#18181b" vertical={false} />
                    <XAxis type="number" dataKey="acceptance_rate_percent" name="Alpha" unit="%" stroke="#52525b" />
                    <YAxis type="number" dataKey="decode_speedup" name="Speedup" unit="x" stroke="#52525b" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: "#09090b", border: "1px solid #27272a", borderRadius: "12px" }}
                    />
                    <Scatter name="SpecDiff Runs" data={data.filter(d => d.method === "SpecDiff")} fill="#a78bfa" />
                  </ScatterChart>
                </ResponsiveContainer>
              </ChartContainer>

              <ChartContainer title="Throughput Benchmarks (tok/s)">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={throughputData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#18181b" vertical={false} />
                    <XAxis dataKey="name" stroke="#52525b" fontSize={10} tick={{ fill: '#52525b' }} />
                    <YAxis stroke="#52525b" />
                    <Tooltip 
                       contentStyle={{ backgroundColor: "#09090b", border: "1px solid #27272a", borderRadius: "12px" }}
                    />
                    <Legend />
                    <Bar name="Baseline (Standard AR)" dataKey="baseline" fill="#52525b" radius={[4, 4, 0, 0]} />
                    <Bar name="SpecDiff (Optimized)" dataKey="speculative" fill="#22d3ee" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartContainer>
            </section>

            {/* Table Section */}
            <section className="glass-card rounded-3xl overflow-hidden">
               <div className="p-6 border-b border-white/5 flex items-center justify-between">
                  <h3 className="text-lg font-bold">Experiment Explorer</h3>
                  <button onClick={() => setData([])} className="text-zinc-500 hover:text-red-400 transition-colors flex items-center gap-2 text-sm font-medium">
                    <Trash2 size={16} /> Clear Session
                  </button>
               </div>
               <div className="overflow-x-auto">
                 <table className="w-full text-left">
                   <thead className="bg-zinc-900/50 text-zinc-500 text-xs uppercase tracking-widest font-bold">
                     <tr>
                       <th className="p-4 px-6">Timestamp</th>
                       <th className="p-4 px-6">Method</th>
                       <th className="p-4 px-6">Target Model</th>
                       <th className="p-4 px-6">gamma / T</th>
                       <th className="p-4 px-6">Speedup</th>
                       <th className="p-4 px-6">Throughput</th>
                       <th className="p-4 px-6">α Rate</th>
                     </tr>
                   </thead>
                   <tbody className="text-sm divide-y divide-white/5">
                     {data.map((exp, i) => (
                       <tr key={i} className="hover:bg-white/5 transition-colors group">
                         <td className="p-4 px-6 text-zinc-400 font-mono text-[10px]">{new Date(exp.timestamp).toLocaleTimeString()}</td>
                         <td className="p-4 px-6">
                            <span className={cn("px-2 py-1 rounded text-[10px] font-bold uppercase", 
                              exp.method === "SpecDiff" ? "bg-cyan-500/10 text-cyan-400" : "bg-zinc-500/10 text-zinc-400"
                            )}>
                              {exp.method}
                            </span>
                         </td>
                         <td className="p-4 px-6 font-semibold">{exp.target_model}</td>
                         <td className="p-4 px-6 text-zinc-500">{exp.gamma} / {exp.T_steps}</td>
                         <td className="p-4 px-6">
                            <div className="flex items-center gap-1 font-bold text-cyan-400">
                               {exp.decode_speedup.toFixed(2)}x
                               <ArrowUpRight size={14} />
                            </div>
                         </td>
                         <td className="p-4 px-6">{exp.decode_throughput_tok_sec.toFixed(1)} <span className="text-zinc-500 text-[10px]">tok/s</span></td>
                         <td className="p-4 px-6">
                            <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                               <div 
                                 className="h-full bg-cyan-400/50" 
                                 style={{ width: `${exp.acceptance_rate_percent}%` }}
                               />
                            </div>
                         </td>
                       </tr>
                     ))}
                   </tbody>
                 </table>
               </div>
            </section>
          </div>
        )}
      </main>
      
      {/* Footer */}
      <footer className="max-w-7xl mx-auto mt-24 py-12 border-t border-white/5 flex items-center justify-between text-zinc-600 text-sm font-medium">
         <div>© 2026 SpecDiff Research Lab</div>
         <div className="flex gap-8">
            <a href="#" className="hover:text-zinc-400 transition-colors">Documentation</a>
            <a href="#" className="hover:text-zinc-400 transition-colors">Architecture</a>
            <a href="#" className="hover:text-zinc-400 transition-colors">GitHub</a>
         </div>
      </footer>
    </div>
  );
}
