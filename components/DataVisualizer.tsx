
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import { MLClass } from '../types';
import { X, Activity, RefreshCw, Box as BoxIcon, Circle, AlertTriangle, ArrowRight, Lightbulb } from 'lucide-react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Stars } from '@react-three/drei';

interface DataVisualizerProps {
  classifier: knnClassifier.KNNClassifier | null;
  classes: MLClass[];
  onClose: () => void;
  onTrainClass: (classId: string) => void;
}

interface DataPoint {
  x: number;
  y: number;
  z: number;
  classId: string;
  color: string;
}

interface Insight {
    type: 'warning' | 'critical';
    message: string;
    classId: string;
    relatedClassId?: string;
}

const DataVisualizer: React.FC<DataVisualizerProps> = ({ classifier, classes, onClose, onTrainClass }) => {
  const [points, setPoints] = useState<DataPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [is3DMode, setIs3DMode] = useState(false);
  const [insights, setInsights] = useState<Insight[]>([]);

  const calculatePCA = async () => {
    if (!classifier) {
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);
    setInsights([]);

    try {
      const dataset = classifier.getClassifierDataset();
      const classIds = Object.keys(dataset);
      
      if (classIds.length === 0) {
        setPoints([]);
        setLoading(false);
        return;
      }

      // 1. Collect all tensors
      let allTensors: tf.Tensor[] = [];
      const pointClassIds: string[] = [];

      classIds.forEach(id => {
        const tensor = dataset[id];
        if (tensor) {
          const numExamples = tensor.shape[0];
          for(let i=0; i<numExamples; i++) {
            pointClassIds.push(id);
          }
          allTensors.push(tensor);
        }
      });

      if (allTensors.length === 0) {
        setPoints([]);
        setLoading(false);
        return;
      }

      const bigTensor = tf.concat(allTensors, 0);
      const numPoints = bigTensor.shape[0];
      const numFeatures = bigTensor.shape[1];

      if (numPoints < 3) {
         setError("Cần ít nhất 3 mẫu dữ liệu để phân tích 3D.");
         setLoading(false);
         bigTensor.dispose();
         return;
      }

      // Check if svd is available to prevent crash
      // @ts-ignore
      if (!tf.linalg || typeof tf.linalg.svd !== 'function') {
          console.warn("tf.linalg.svd is not available.");
          // Fallback: Just use the first 3 dimensions as projection (not true PCA but safe)
          const fallbackProjection = bigTensor.slice([0, 0], [numPoints, 3]);
          await processProjectedData(fallbackProjection, pointClassIds);
          bigTensor.dispose();
          fallbackProjection.dispose();
          setLoading(false);
          return;
      }

      // 2. PCA Calculation (Always 3D)
      const mean = bigTensor.mean(0);
      const centered = bigTensor.sub(mean);
      
      try {
          const svd = tf.linalg.svd(centered);
          const v = svd.v; 
          
          // Take top 3 components for X, Y, Z
          const projectionMatrix = v.slice([0, 0], [numFeatures, 3]);
          const projected = centered.matMul(projectionMatrix);
          
          await processProjectedData(projected, pointClassIds);
          
          if(svd.u) svd.u.dispose();
          if(svd.v) svd.v.dispose();
          if(svd.s) svd.s.dispose();
          projectionMatrix.dispose();
          projected.dispose();
      } catch (svdError) {
          console.error("SVD Error, falling back to basic projection:", svdError);
          // Fallback if SVD fails despite check
          const fallbackProjection = bigTensor.slice([0, 0], [numPoints, 3]);
          await processProjectedData(fallbackProjection, pointClassIds);
          fallbackProjection.dispose();
      }
      
      // Cleanup
      bigTensor.dispose();
      mean.dispose();
      centered.dispose();
      
      setLoading(false);

    } catch (e) {
      console.error("PCA Analysis Error:", e);
      setError("Lỗi khi phân tích dữ liệu (Có thể do trình duyệt không hỗ trợ WebGL/SVD).");
      setLoading(false);
    }
  };

  const processProjectedData = async (projected: tf.Tensor, pointClassIds: string[]) => {
      const projectedArray = await projected.array() as number[][];

      // 3. Normalize Logic
      let minX = Infinity, maxX = -Infinity;
      let minY = Infinity, maxY = -Infinity;
      let minZ = Infinity, maxZ = -Infinity;

      projectedArray.forEach(([x, y, z]) => {
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
        if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
      });

      const rangeX = maxX - minX || 1;
      const rangeY = maxY - minY || 1;
      const rangeZ = maxZ - minZ || 1;

      const newPoints: DataPoint[] = projectedArray.map((p, index) => {
        const clsId = pointClassIds[index];
        const cls = classes.find(c => c.id === clsId);
        return {
          x: (p[0] - minX) / rangeX,
          y: (p[1] - minY) / rangeY,
          z: (p[2] - minZ) / rangeZ, 
          classId: clsId,
          color: cls ? cls.color : '#ffffff'
        };
      });

      setPoints(newPoints);
      analyzeClusters(newPoints, classes);
  }

  const analyzeClusters = (points: DataPoint[], classes: MLClass[]) => {
      const clusterMap: Record<string, { xSum: number, ySum: number, zSum: number, count: number, points: DataPoint[] }> = {};
      const newInsights: Insight[] = [];

      // 1. Group points and check quantity
      points.forEach(p => {
          if (!clusterMap[p.classId]) {
              clusterMap[p.classId] = { xSum: 0, ySum: 0, zSum: 0, count: 0, points: [] };
          }
          const c = clusterMap[p.classId];
          c.xSum += p.x;
          c.ySum += p.y;
          c.zSum += p.z;
          c.count++;
          c.points.push(p);
      });

      const activeClassIds = Object.keys(clusterMap);

      // 2. Check Quantity
      activeClassIds.forEach(id => {
          const count = clusterMap[id].count;
          const clsName = classes.find(c => c.id === id)?.name || id;
          
          if (count < 15) {
              newInsights.push({
                  type: 'critical',
                  message: `Hành động "${clsName}" có quá ít dữ liệu (${count} mẫu). AI sẽ học không chính xác.`,
                  classId: id
              });
          } else if (count < 30) {
              newInsights.push({
                  type: 'warning',
                  message: `Hành động "${clsName}" hơi ít dữ liệu (${count} mẫu). Nên thêm khoảng 20 mẫu nữa.`,
                  classId: id
              });
          }
      });

      // 3. Calculate Centroids & Radii for Separation check
      const clusters: Record<string, { x: number, y: number, z: number, radius: number }> = {};
      
      activeClassIds.forEach(id => {
          const c = clusterMap[id];
          const centerX = c.xSum / c.count;
          const centerY = c.ySum / c.count;
          const centerZ = c.zSum / c.count;

          // Calculate avg distance from center (radius)
          let totalDist = 0;
          c.points.forEach(p => {
              totalDist += Math.sqrt(
                  Math.pow(p.x - centerX, 2) + 
                  Math.pow(p.y - centerY, 2) + 
                  Math.pow(p.z - centerZ, 2)
              );
          });
          
          clusters[id] = {
              x: centerX, 
              y: centerY, 
              z: centerZ, 
              radius: totalDist / c.count
          };
      });

      // 4. Check Overlaps (Conflict)
      // Simple Euclidean check: Distance(A, B) < (RadiusA + RadiusB) * threshold
      const SEPARATION_THRESHOLD = 1.1; // Multiplier to make it slightly strict

      for (let i = 0; i < activeClassIds.length; i++) {
          for (let j = i + 1; j < activeClassIds.length; j++) {
              const idA = activeClassIds[i];
              const idB = activeClassIds[j];
              const cA = clusters[idA];
              const cB = clusters[idB];

              if(!cA || !cB) continue;

              const dist = Math.sqrt(
                  Math.pow(cA.x - cB.x, 2) + 
                  Math.pow(cA.y - cB.y, 2) + 
                  Math.pow(cA.z - cB.z, 2)
              );

              const combinedRadius = cA.radius + cB.radius;

              if (dist < combinedRadius * SEPARATION_THRESHOLD) {
                   const nameA = classes.find(c => c.id === idA)?.name;
                   const nameB = classes.find(c => c.id === idB)?.name;
                   newInsights.push({
                       type: 'critical',
                       message: `"${nameA}" đang bị nhầm lẫn với "${nameB}". Cần dạy lại ở góc độ khác nhau.`,
                       classId: idA,
                       relatedClassId: idB
                   });
              }
          }
      }

      setInsights(newInsights);
  };

  useEffect(() => {
    const t = setTimeout(calculatePCA, 100);
    return () => clearTimeout(t);
  }, [classifier]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 backdrop-blur-md animate-in fade-in duration-200">
      <div className="w-full max-w-6xl bg-[#151515] border border-white/10 rounded-3xl shadow-2xl flex flex-col max-h-[90vh] h-[800px]">
        
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-white/5 bg-[#111]">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-full bg-purple-500/20 flex items-center justify-center text-purple-400">
                <Activity size={20} />
            </div>
            <div>
                <h2 className="text-xl font-bold text-white">Phân tích Dữ liệu AI</h2>
                <p className="text-xs text-gray-400">Trực quan hóa PCA & Đề xuất tối ưu</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
             <div className="flex bg-black/50 rounded-lg p-1 border border-white/10">
                <button 
                    onClick={() => setIs3DMode(false)}
                    className={`px-3 py-1.5 rounded-md text-xs font-bold flex items-center gap-2 transition-all ${!is3DMode ? 'bg-white text-black shadow-lg' : 'text-gray-400 hover:text-white'}`}
                >
                    <Circle size={12} /> 2D
                </button>
                <button 
                    onClick={() => setIs3DMode(true)}
                    className={`px-3 py-1.5 rounded-md text-xs font-bold flex items-center gap-2 transition-all ${is3DMode ? 'bg-purple-600 text-white shadow-lg' : 'text-gray-400 hover:text-white'}`}
                >
                    <BoxIcon size={12} /> 3D
                </button>
             </div>

             <div className="w-px h-6 bg-white/10 mx-1"></div>

             <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors p-2 rounded-full hover:bg-white/5">
                <X size={24} />
             </button>
          </div>
        </div>

        {/* Content Grid */}
        <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
            
            {/* Left: Visualizer Chart (70%) */}
            <div className="flex-1 relative bg-[#050505] border-r border-white/5">
                {loading ? (
                    <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 z-10">
                        <RefreshCw className="animate-spin text-purple-500" size={32} />
                        <span className="text-gray-400 text-sm animate-pulse">Đang tính toán PCA...</span>
                    </div>
                ) : error ? (
                    <div className="absolute inset-0 flex items-center justify-center text-gray-500 text-sm z-10">
                        {error}
                    </div>
                ) : points.length === 0 ? (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-500 z-10">
                        <Activity size={48} className="opacity-20 mb-4" />
                        <p>Chưa có dữ liệu mẫu.</p>
                    </div>
                ) : (
                   <>
                     {is3DMode ? (
                        // 3D VIEW
                        <div className="w-full h-full">
                            <Canvas camera={{ position: [4, 3, 5], fov: 50 }}>
                                <ambientLight intensity={0.5} />
                                <pointLight position={[10, 10, 10]} intensity={1} />
                                <pointLight position={[-10, -10, -10]} intensity={0.5} />
                                
                                <OrbitControls makeDefault autoRotate autoRotateSpeed={0.5} />
                                <Grid infiniteGrid fadeDistance={30} sectionColor="#444" cellColor="#222" />
                                <axesHelper args={[2]} />
                                
                                <group>
                                    {points.map((p, i) => (
                                        <mesh 
                                            key={i} 
                                            position={[
                                                (p.x - 0.5) * 6, 
                                                (p.y - 0.5) * 6,
                                                (p.z - 0.5) * 6
                                            ]}
                                        >
                                            <sphereGeometry args={[0.08, 16, 16]} />
                                            <meshStandardMaterial color={p.color} roughness={0.3} metalness={0.8} />
                                        </mesh>
                                    ))}
                                </group>
                                
                                <Stars radius={50} depth={50} count={1000} factor={4} saturation={0} fade speed={1} />
                            </Canvas>
                            
                            <div className="absolute bottom-4 left-4 text-[10px] text-gray-500 font-mono pointer-events-none">
                                Left Click: Rotate | Right Click: Pan | Scroll: Zoom
                            </div>
                        </div>
                     ) : (
                        // 2D VIEW
                        <div className="w-full h-full relative p-8">
                             <div className="w-full h-full border border-white/5 rounded-xl bg-[#080808] relative overflow-hidden">
                                <div className="absolute inset-0 grid grid-cols-6 grid-rows-6 pointer-events-none opacity-20">
                                    {Array.from({length: 36}).map((_,i) => (
                                        <div key={i} className="border border-white/10"></div>
                                    ))}
                                </div>

                                <svg width="100%" height="100%" className="absolute inset-0 overflow-visible">
                                    {points.map((p, i) => (
                                        <circle 
                                            key={i}
                                            cx={`${p.x * 100}%`}
                                            cy={`${p.y * 100}%`}
                                            r="5"
                                            fill={p.color}
                                            fillOpacity="0.8"
                                            stroke="#fff"
                                            strokeWidth="1"
                                            strokeOpacity="0.3"
                                            className="transition-all duration-300 hover:r-8 hover:fill-opacity-100 cursor-pointer"
                                        >
                                            <title>{classes.find(c => c.id === p.classId)?.name}</title>
                                        </circle>
                                    ))}
                                </svg>
                                
                                <div className="absolute bottom-2 left-1/2 -translate-x-1/2 text-[10px] text-gray-600 font-mono">Principal Component 1</div>
                                <div className="absolute left-2 top-1/2 -translate-y-1/2 -rotate-90 text-[10px] text-gray-600 font-mono">Principal Component 2</div>
                             </div>
                        </div>
                     )}
                   </>
                )}
            </div>

            {/* Right: Insights Panel (30%) */}
            <div className="w-full md:w-80 bg-[#111] flex flex-col border-l border-white/5">
                <div className="p-4 border-b border-white/5">
                    <h3 className="text-sm font-bold text-gray-300 flex items-center gap-2">
                        <Lightbulb size={16} className="text-yellow-400" />
                        Trợ lý Huấn luyện
                    </h3>
                </div>
                
                <div className="flex-1 overflow-y-auto p-4 space-y-3">
                    {loading ? (
                        <div className="space-y-3 opacity-50">
                            <div className="h-20 bg-white/5 rounded-lg animate-pulse"></div>
                            <div className="h-20 bg-white/5 rounded-lg animate-pulse"></div>
                        </div>
                    ) : insights.length > 0 ? (
                        insights.map((insight, idx) => (
                            <div key={idx} className={`p-3 rounded-xl border ${
                                insight.type === 'critical' 
                                ? 'bg-red-900/10 border-red-500/30' 
                                : 'bg-yellow-900/10 border-yellow-500/30'
                            }`}>
                                <div className="flex items-start gap-3 mb-2">
                                    <AlertTriangle size={16} className={insight.type === 'critical' ? 'text-red-400' : 'text-yellow-400'} />
                                    <p className={`text-xs leading-relaxed ${insight.type === 'critical' ? 'text-red-200' : 'text-yellow-200'}`}>
                                        {insight.message}
                                    </p>
                                </div>
                                <button 
                                    onClick={() => onTrainClass(insight.classId)}
                                    className="w-full py-1.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-xs font-medium text-white flex items-center justify-center gap-2 transition-colors"
                                >
                                    Sửa lỗi ngay <ArrowRight size={12} />
                                </button>
                            </div>
                        ))
                    ) : (
                        <div className="text-center py-8">
                             <div className="w-12 h-12 bg-green-500/10 rounded-full flex items-center justify-center mx-auto mb-3">
                                 <Activity className="text-green-500" size={24} />
                             </div>
                             <p className="text-sm text-gray-400">Dữ liệu của bạn đang rất tốt!</p>
                             <p className="text-xs text-gray-600 mt-1">Không tìm thấy vấn đề chồng chéo hay thiếu mẫu nào.</p>
                        </div>
                    )}
                </div>

                {/* Legend */}
                <div className="p-4 border-t border-white/5 bg-[#0a0a0a]">
                    <h4 className="text-[10px] font-bold text-gray-500 uppercase mb-2">Chú thích màu sắc</h4>
                    <div className="flex flex-wrap gap-2 max-h-32 overflow-y-auto scroller">
                        {classes.filter(c => c.exampleCount > 0).map(c => (
                            <div key={c.id} className="flex items-center gap-1.5 text-[10px] bg-[#1a1a1a] px-2 py-1 rounded-full border border-white/10">
                                <div className="w-2 h-2 rounded-full" style={{backgroundColor: c.color}}></div>
                                <span className="text-gray-400 truncate max-w-[80px]">{c.name}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>

      </div>
    </div>
  );
};

export default DataVisualizer;
