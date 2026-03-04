
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useRef, useState, useEffect, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { useMediaPipe, createHandLandmarkerInstance } from './hooks/useMediaPipe';
import WebcamPreview from './components/WebcamPreview';
import VisualizerScene from './components/GameScene';
import DataVisualizer from './components/DataVisualizer';
import { MLClass, AppSettings } from './types';
import { DEFAULT_CLASSES } from './constants';
import { BrainCircuit, Plus, RotateCcw, Mic, Volume2, Keyboard, StopCircle, Check, Save, Upload, Settings, X, Loader2, Trash2, RefreshCw, ListFilter, Pause, Play, Hand, Activity, Gauge, Layers, MousePointer2, MoveHorizontal, Scan, Rotate3D, Grid, Cloud, CloudLightning, Lock, Unlock, User, ShieldCheck, DownloadCloud, Clock, Camera, Video, MessageSquare, Search, Repeat, FastForward, Image as ImageIcon, AlertTriangle, FileVideo, CheckCircle2 } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import { loadFromFirestore, saveToFirestore } from './services/firebase';
import { saveToIndexedDB, loadFromIndexedDB } from './services/storage';
import { HandLandmarkerResult, HandLandmarker } from '@mediapipe/tasks-vision';

// Changed key to v7 for Extended Temporal Sequence Data schema
const STORAGE_KEY = 'sign_language_ml_data_v7';
const FEATURES_PER_HAND = 63; // 21 landmarks * 3 coords
// v7 Update: Increased sequence length to 12 frames (approx 0.7s - 0.8s) to capture longer trajectories
// This is crucial for distinguishing moves like "Eat" (repetitive) vs "Thank you" (directional path)
const SEQUENCE_LENGTH = 12; 
const FRAME_INTERVAL = 60; // ms between frames captured for the sequence

const App: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);
  
  // Settings State
  const [showSettings, setShowSettings] = useState(false);
  const [showVisualizer, setShowVisualizer] = useState(false);
  const [showTranslator, setShowTranslator] = useState(false); // Translator UI
  const [appSettings, setAppSettings] = useState<AppSettings>({
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    predictionThreshold: 0.8 
  });
  const [tempSettings, setTempSettings] = useState<AppSettings>(appSettings);

  // App State
  const [isFrozen, setIsFrozen] = useState(false);
  const [lastSaved, setLastSaved] = useState<Date | null>(null);
  const [activeTab, setActiveTab] = useState<'all' | 'left' | 'right' | 'both'>('all');

  // Admin Mode State
  const [isAdmin, setIsAdmin] = useState(false);
  const [isCloudLoading, setIsCloudLoading] = useState(false);
  const [uploadTargetClassId, setUploadTargetClassId] = useState<string | null>(null);
  
  // Video Processing State
  const [videoProcessing, setVideoProcessing] = useState({ processing: false, progress: 0, totalFrames: 0 });
  // Hand Selection Modal State
  const [showHandSelector, setShowHandSelector] = useState(false);
  const [targetClassForHandSelect, setTargetClassForHandSelect] = useState<string | null>(null);

  const { isCameraReady, isModelLoading, lastResultsRef, handPositionsRef, detectImage, error: cameraError } = useMediaPipe(videoRef, appSettings, isFrozen);
  
  // Machine Learning State
  const classifierRef = useRef<knnClassifier.KNNClassifier | null>(null);
  const samplesCollectedRef = useRef<number>(0); 
  const lastPredictedIdRef = useRef<string | null>(null); 
  const [classes, setClasses] = useState<MLClass[]>(DEFAULT_CLASSES);
  const [trainingClassId, setTrainingClassId] = useState<string | null>(null);
  const [predictedClass, setPredictedClass] = useState<MLClass | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [speechProgress, setSpeechProgress] = useState(0);
  
  // Temporal Sequence Buffer
  const sequenceBufferRef = useRef<number[][]>([]);
  
  // Velocity Indicator State
  const [handSpeed, setHandSpeed] = useState(0);
  const [showSpeed, setShowSpeed] = useState(false);

  // UI State
  const [isAddingClass, setIsAddingClass] = useState(false);
  const [newClassName, setNewClassName] = useState("");
  const [newClassHandType, setNewClassHandType] = useState<'left' | 'right' | 'both'>('right');
  const [newClassType, setNewClassType] = useState<'static' | 'sequence'>('static');

  // Translator / Playback State
  const [translateInput, setTranslateInput] = useState("");
  const [playbackData, setPlaybackData] = useState<number[][] | null>(null); // Array of frames for playback
  const [isPlaybackActive, setIsPlaybackActive] = useState(false);
  const [playbackStatus, setPlaybackStatus] = useState<string>("");
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [isLooping, setIsLooping] = useState(false);

  // Initialization
  useEffect(() => {
    const initTF = async () => {
      await tf.ready();
      classifierRef.current = knnClassifier.create();
      console.log("TensorFlow KNN Initialized (v7 Extended Temporal)");
      await loadData(true); 
    };
    initTF();
  }, []);

  // Helper: Normalize Landmarks (Centering & Scaling)
  const normalizeLandmarks = (landmarks: {x: number, y: number, z: number}[]) => {
      if (landmarks.length === 0) return [];

      // 1. Centering: Make Wrist (index 0) the origin (0,0,0)
      const wrist = landmarks[0];
      const centered = landmarks.map(lm => ({
          x: lm.x - wrist.x,
          y: lm.y - wrist.y,
          z: lm.z - wrist.z
      }));

      // 2. Scaling: Scale so the max distance from wrist is 1.0
      let maxDist = 0;
      centered.forEach(lm => {
          const dist = Math.sqrt(lm.x*lm.x + lm.y*lm.y + lm.z*lm.z);
          if (dist > maxDist) maxDist = dist;
      });
      
      if (maxDist < 0.0001) maxDist = 1;

      // 3. Return flattened normalized coordinates
      return centered.map(lm => [
          lm.x / maxDist,
          lm.y / maxDist,
          lm.z / maxDist
      ]).flat();
  };

  const extractFeaturesFromLandmarks = (results: HandLandmarkerResult, targetHandType: 'left' | 'right' | 'both' = 'both') => {
      const leftHandFeatures = new Array(FEATURES_PER_HAND).fill(0);
      const rightHandFeatures = new Array(FEATURES_PER_HAND).fill(0);
      let hasLeft = false;
      let hasRight = false;
      
      if (results.landmarks.length > 0) {
          for (let i = 0; i < results.landmarks.length; i++) {
              const landmarks = results.landmarks[i];
              const handedness = results.handedness[i][0].categoryName;
              
              const normalizedPoints = normalizeLandmarks(landmarks);
              
              if (handedness === 'Right') {
                  for(let j=0; j<normalizedPoints.length; j++) rightHandFeatures[j] = normalizedPoints[j];
                  hasRight = true;
              } else {
                  for(let j=0; j<normalizedPoints.length; j++) leftHandFeatures[j] = normalizedPoints[j];
                  hasLeft = true;
              }
          }
      }

      // STRICT MODE LOGIC:
      // If we are training for a specific hand, we MUST return null if that hand is not detected.
      // This prevents training noise (e.g., training a "Right Hand" word with only a "Left Hand" visible).
      
      if (targetHandType === 'left') {
          if (!hasLeft) return null; // Strict requirement
          rightHandFeatures.fill(0); // Mask other hand
      } else if (targetHandType === 'right') {
          if (!hasRight) return null; // Strict requirement
          leftHandFeatures.fill(0); // Mask other hand
      } else if (targetHandType === 'both') {
          // For 'both', typically at least one hand should be present, ideally both depending on the sign.
          // For now, we allow either, but generally 'both' implies interaction.
          if (!hasLeft && !hasRight) return null;
      }

      return [...leftHandFeatures, ...rightHandFeatures];
  };

  // Clear buffer when changing classes or starting/stopping
  const clearSequenceBuffer = () => {
      sequenceBufferRef.current = [];
  };

  // Main Loop: Training & Prediction
  useEffect(() => {
      if (!isCameraReady || isModelLoading) return;
      
      const interval = setInterval(async () => {
          // Calculate Velocity for UI
          if (handPositionsRef.current) {
              const vLeft = handPositionsRef.current.leftVelocity.length();
              const vRight = handPositionsRef.current.rightVelocity.length();
              const speed = Math.max(vLeft, vRight);
              setHandSpeed(speed);
          }

          // Stop prediction if frozen, training, playing back, OR PROCESSING VIDEO
          if (isFrozen || trainingClassId || isPlaybackActive || videoProcessing.processing) return; 

          const results = lastResultsRef.current;
          
          if (classifierRef.current && results) {
              
              // --- 1. EXTRACT FEATURES FROM CURRENT FRAME ---
              let currentFrameFeatures: number[] = [];
              const extracted = extractFeaturesFromLandmarks(results); // Prediction uses 'both' usually to find matches
              
              if (extracted) {
                  currentFrameFeatures = extracted;
              } else {
                  // No hands detected
                  if (!isFrozen) { 
                    setPredictedClass(null);
                    setClasses(prev => prev.map(c => ({ ...c, confidence: 0 })));
                    sequenceBufferRef.current = []; 
                    return;
                  }
              }

              // --- 2. UPDATE TEMPORAL BUFFER ---
              if (currentFrameFeatures.length > 0) {
                  sequenceBufferRef.current.push(currentFrameFeatures);
                  if (sequenceBufferRef.current.length > SEQUENCE_LENGTH) {
                      sequenceBufferRef.current.shift();
                  }
              }

              if (sequenceBufferRef.current.length < SEQUENCE_LENGTH) {
                  return;
              }

              // --- 3. FLATTEN ---
              const temporalFeatures = sequenceBufferRef.current.flat();
              const tensor = tf.tensor(temporalFeatures);

              // --- 4. TRAINING ---
              // Training handled in separate effect

              // --- 5. PREDICTION ---
              if (!trainingClassId) {
                const exampleCounts = classifierRef.current.getClassExampleCount();
                if (Object.keys(exampleCounts).length > 0) {
                    try {
                        const k = 20; 
                        const result = await classifierRef.current.predictClass(tensor, k);
                        
                        if (result.label) {
                            const conf = result.confidences[result.label];
                            
                            setClasses(prev => prev.map(c => ({
                                ...c,
                                confidence: result.confidences[c.id] || 0
                            })));

                            const foundClass = classes.find(c => c.id === result.label);
                            
                            if (foundClass && conf >= appSettings.predictionThreshold) { 
                                setPredictedClass(foundClass);

                                if (lastPredictedIdRef.current !== foundClass.id) {
                                    if (foundClass.id !== '0' && typeof navigator !== 'undefined' && navigator.vibrate) {
                                        navigator.vibrate(50); 
                                    }
                                    lastPredictedIdRef.current = foundClass.id;
                                }
                            } else {
                                setPredictedClass(null);
                                lastPredictedIdRef.current = null;
                            }
                        }
                    } catch (e) {
                        console.warn("Prediction error:", e);
                    }
                }
              }
              
              tensor.dispose();
          }
      }, FRAME_INTERVAL);

      return () => clearInterval(interval);
  }, [isCameraReady, isModelLoading, trainingClassId, classes, appSettings.predictionThreshold, isFrozen, isPlaybackActive, videoProcessing.processing]);


  // Separate Training Loop to ensure we capture data correctly
  useEffect(() => {
    if (!trainingClassId || isFrozen || !isCameraReady || videoProcessing.processing) return;

    const trainInterval = setInterval(() => {
        const results = lastResultsRef.current;
        const currentTrainingClass = classes.find(c => c.id === trainingClassId);

        if (classifierRef.current && results && results.landmarks.length > 0 && currentTrainingClass) {
             
             const extracted = extractFeaturesFromLandmarks(results, currentTrainingClass.handType);
             if (!extracted) return;

             sequenceBufferRef.current.push(extracted);
             if (sequenceBufferRef.current.length > SEQUENCE_LENGTH) sequenceBufferRef.current.shift();

             if (sequenceBufferRef.current.length === SEQUENCE_LENGTH) {
                  const temporalFeatures = sequenceBufferRef.current.flat();
                  const tensor = tf.tensor(temporalFeatures);
                  
                  classifierRef.current.addExample(tensor, trainingClassId);
                  samplesCollectedRef.current += 1;
                  
                  setClasses(prev => prev.map(c => 
                      c.id === trainingClassId 
                      ? { ...c, exampleCount: (classifierRef.current?.getClassExampleCount()[trainingClassId] || 0) }
                      : c
                  ));

                  if (samplesCollectedRef.current >= 50) {
                      setTrainingClassId(null);
                      samplesCollectedRef.current = 0;
                      clearSequenceBuffer();
                      alert("Đã học xong! Bạn có thể thử nghiệm ngay.");
                  }
                  tensor.dispose();
             }
        }
    }, FRAME_INTERVAL);

    return () => clearInterval(trainInterval);
  }, [trainingClassId, isFrozen, isCameraReady, classes, videoProcessing.processing]);


  // --- IMAGE UPLOAD TRAINING LOGIC ---

  const triggerImageUpload = (classId: string) => {
      setUploadTargetClassId(classId);
      if (fileInputRef.current) {
          fileInputRef.current.click();
      }
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file || !uploadTargetClassId) return;
      
      const reader = new FileReader();
      reader.onload = async (e) => {
          const src = e.target?.result as string;
          const img = new Image();
          img.src = src;
          img.onload = async () => {
              // Detect landmarks from image
              const results = await detectImage(img);
              
              if (results && results.landmarks.length > 0 && classifierRef.current) {
                  const currentClass = classes.find(c => c.id === uploadTargetClassId);
                  if (!currentClass) return;

                  const extracted = extractFeaturesFromLandmarks(results, currentClass.handType);
                  
                  if (extracted) {
                      // For image upload, we assume it's a static pose or a keyframe.
                      const sequence = Array(SEQUENCE_LENGTH).fill(extracted);
                      const flatSequence = sequence.flat();
                      const tensor = tf.tensor(flatSequence);
                      
                      classifierRef.current.addExample(tensor, uploadTargetClassId);
                      
                      setClasses(prev => prev.map(c => 
                        c.id === uploadTargetClassId 
                        ? { ...c, exampleCount: (classifierRef.current?.getClassExampleCount()[uploadTargetClassId] || 0) }
                        : c
                      ));
                      
                      tensor.dispose();
                      alert(`Đã học xong ảnh cho "${currentClass.name}"!`);
                  } else {
                      alert("Không tìm thấy bàn tay phù hợp trong ảnh (Kiểm tra tay trái/phải).");
                  }
              } else {
                  alert("Không tìm thấy bàn tay nào trong ảnh.");
              }
              
              if (fileInputRef.current) fileInputRef.current.value = "";
              setUploadTargetClassId(null);
          };
      };
      reader.readAsDataURL(file);
  };

  // --- VIDEO UPLOAD TRAINING LOGIC ---

  // Step 1: Trigger the Modal instead of the file input directly
  const triggerVideoUpload = (classId: string) => {
      setTargetClassForHandSelect(classId);
      setShowHandSelector(true);
  };

  // Step 2: Handle Hand Selection from Modal
  const handleHandSelect = (handType: 'left' | 'right' | 'both') => {
      if (!targetClassForHandSelect) return;

      // Update the class configuration to enforce the selected hand type
      const updatedClasses = classes.map(c => 
          c.id === targetClassForHandSelect 
          ? { ...c, handType: handType } 
          : c
      );
      setClasses(updatedClasses);
      saveData(true, updatedClasses); // Persist the change

      // Proceed to open file dialog
      setUploadTargetClassId(targetClassForHandSelect);
      setShowHandSelector(false);
      setTargetClassForHandSelect(null);

      // Small delay to allow UI to update before opening native picker
      setTimeout(() => {
          if (videoInputRef.current) {
              videoInputRef.current.click();
          }
      }, 100);
  };

  // Step 3: Handle the file processing (Strict Mode enforced by Updated Class config)
  const handleVideoFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file || !uploadTargetClassId || !classifierRef.current) return;

      const currentClass = classes.find(c => c.id === uploadTargetClassId);
      if (!currentClass) return;

      // Pause everything else
      setIsFrozen(true);
      setVideoProcessing({ processing: true, progress: 0, totalFrames: 0 });

      let videoLandmarker: HandLandmarker | null = null;

      try {
          // Initialize a dedicated instance for this video processing session
          videoLandmarker = await createHandLandmarkerInstance(appSettings);

          // Create hidden video element
          const video = document.createElement('video');
          video.src = URL.createObjectURL(file);
          video.muted = true;
          video.playsInline = true;
          
          await new Promise<void>((resolve) => {
              video.onloadedmetadata = () => resolve();
          });

          const duration = video.duration;
          
          // ENHANCED SAMPLING: Scan at 3x the resolution (every 20ms instead of 60ms)
          // Then split into 3 interleaved streams. This generates 3x the training data.
          const OVERSAMPLE_FACTOR = 3;
          const RAW_STEP = FRAME_INTERVAL / 1000 / OVERSAMPLE_FACTOR; // 20ms step
          const totalSteps = Math.ceil(duration / RAW_STEP);
          setVideoProcessing(prev => ({ ...prev, totalFrames: totalSteps }));

          // Collect ALL raw frames first
          const rawFeaturesBuffer: number[][] = [];

          // Seek-and-Capture Loop (High Resolution)
          for (let currentTime = 0; currentTime < duration; currentTime += RAW_STEP) {
              video.currentTime = currentTime;
              await new Promise<void>((resolve) => {
                 const onSeeked = () => {
                     video.removeEventListener('seeked', onSeeked);
                     resolve();
                 };
                 video.addEventListener('seeked', onSeeked);
              });
              
              // Update Progress
              setVideoProcessing(prev => ({ ...prev, progress: Math.round((currentTime / duration) * 100) }));

              const results = videoLandmarker.detectForVideo(video, currentTime * 1000);

              if (results && results.landmarks.length > 0) {
                  const extracted = extractFeaturesFromLandmarks(results, currentClass.handType);
                  if (extracted) {
                      rawFeaturesBuffer.push(extracted);
                  } else {
                      // Push null or empty to maintain temporal integrity? 
                      // For simplicity, if hand is lost, we skip this frame in the raw buffer.
                      // This acts as a natural cut in the sequence.
                      // Ideally we might want to insert a marker, but skipping is safer for KNN.
                  }
              } 
              
              // Small delay to prevent UI freeze
              await new Promise(r => setTimeout(r, 2));
          }

          // PROCESS RAW BUFFER INTO MULTIPLE STREAMS
          // We have frames at: 0ms, 20ms, 40ms, 60ms, 80ms, 100ms...
          // Stream 1 (standard): 0, 60, 120... (Indices 0, 3, 6...)
          // Stream 2 (offset 1): 20, 80, 140... (Indices 1, 4, 7...)
          // Stream 3 (offset 2): 40, 100, 160... (Indices 2, 5, 8...)

          let addedCount = 0;

          for (let offset = 0; offset < OVERSAMPLE_FACTOR; offset++) {
              const sequence: number[][] = [];
              
              // Construct interleaved sequence
              for (let i = offset; i < rawFeaturesBuffer.length; i += OVERSAMPLE_FACTOR) {
                  sequence.push(rawFeaturesBuffer[i]);
                  
                  if (sequence.length > SEQUENCE_LENGTH) {
                      sequence.shift();
                  }

                  if (sequence.length === SEQUENCE_LENGTH) {
                      const tensor = tf.tensor(sequence.flat());
                      classifierRef.current.addExample(tensor, uploadTargetClassId);
                      tensor.dispose();
                      addedCount++;
                  }
              }
          }

          setClasses(prev => prev.map(c => 
            c.id === uploadTargetClassId 
            ? { ...c, exampleCount: (classifierRef.current?.getClassExampleCount()[uploadTargetClassId] || 0) }
            : c
          ));

          alert(`Đã xử lý xong video! Với kỹ thuật Multi-Pass Sampling, hệ thống đã trích xuất được ${addedCount} mẫu dữ liệu chất lượng cao (gấp 3 lần bình thường) từ video ngắn này.`);

      } catch (e) {
          console.error("Video processing failed", e);
          alert("Lỗi khi xử lý video. Vui lòng thử file khác.");
      } finally {
          if (videoLandmarker) {
              videoLandmarker.close();
          }
          setVideoProcessing({ processing: false, progress: 0, totalFrames: 0 });
          setIsFrozen(false);
          if (videoInputRef.current) videoInputRef.current.value = "";
          setUploadTargetClassId(null);
      }
  };


  // Text-to-Speech Trigger Logic
  useEffect(() => {
      let interval: ReturnType<typeof setInterval>;
      const targetId = predictedClass?.id;
      const isTraining = trainingClassId !== null;
      
      if (targetId && targetId !== '0' && !isSpeaking && !isTraining && !isFrozen && !isPlaybackActive && !videoProcessing.processing) { 
          setSpeechProgress(0); 

          interval = setInterval(() => {
              setSpeechProgress(prev => {
                  if (prev >= 100) {
                      clearInterval(interval);
                      if (predictedClass) speak(predictedClass.name);
                      return 100;
                  }
                  return prev + 2; 
              });
          }, 20);
      } else {
          setSpeechProgress(0);
      }
      
      return () => clearInterval(interval);
  }, [predictedClass?.id, isSpeaking, trainingClassId, isFrozen, isPlaybackActive, videoProcessing.processing]);

  useEffect(() => {
    if (predictedClass && predictedClass.id !== '0') {
        setShowSpeed(true);
        const timer = setTimeout(() => setShowSpeed(false), 4000);
        return () => clearTimeout(timer);
    }
  }, [predictedClass?.id]);

  const speak = (text: string) => {
      if ('speechSynthesis' in window) {
          window.speechSynthesis.cancel(); 
          setIsSpeaking(true);
          const utterance = new SpeechSynthesisUtterance(text);
          utterance.lang = 'vi-VN'; 
          utterance.rate = 0.9;
          utterance.onend = () => {
              setIsSpeaking(false);
              setSpeechProgress(0);
          };
          window.speechSynthesis.speak(utterance);
      }
  };

  // --- TRANSLATOR / PLAYBACK ENGINE ---

  const handleTranslate = async () => {
      if (!translateInput.trim() || !classifierRef.current) return;
      
      setPlaybackStatus("Đang phân tích...");
      setPlaybackData(null);

      const words = translateInput.toLowerCase().split(' ').map(w => w.trim()).filter(w => w);
      const sequence: MLClass[] = [];
      for (const word of words) {
          const match = classes.find(c => c.name.toLowerCase().includes(word));
          if (match) {
              sequence.push(match);
          }
      }

      if (sequence.length === 0) {
          setPlaybackStatus("Không tìm thấy dữ liệu cho câu này.");
          setIsPlaybackActive(true);
          setTimeout(() => {
              setIsPlaybackActive(false);
              setPlaybackStatus("");
          }, 2000);
          return;
      }

      // Build the full animation timeline
      const dataset = classifierRef.current.getClassifierDataset();
      let compiledFrames: number[][] = [];

      const gapFrames = 10; 
      const gapFrameData = new Array(FEATURES_PER_HAND * 2).fill(0);

      for (let i = 0; i < sequence.length; i++) {
          const cls = sequence[i];
          const tensor = dataset[cls.id];
          if (!tensor) continue;

          const data = await tensor.array() as number[][];
          
          const steps = data.length;
          const maxSteps = Math.min(steps, 120); 
          const stride = 1; 

          for (let j = 0; j < maxSteps; j += stride) {
              const rawRow = data[j];
              const singleFrame = rawRow.slice(rawRow.length - (FEATURES_PER_HAND * 2));
              compiledFrames.push(singleFrame);
          }

          if (i < sequence.length - 1) {
              for(let g=0; g<gapFrames; g++) compiledFrames.push(gapFrameData);
          }
      }

      if (compiledFrames.length === 0) {
           setPlaybackStatus("Dữ liệu chưa được học.");
           setIsPlaybackActive(true);
           setTimeout(() => {
               setIsPlaybackActive(false);
               setPlaybackStatus("");
           }, 2000);
           return;
      }

      setIsFrozen(true); // Stop camera
      setIsPlaybackActive(true);
      setPlaybackData(compiledFrames);
      setPlaybackStatus(`Đang phát: ${translateInput}`);
  };

  const stopPlayback = () => {
      setIsPlaybackActive(false);
      setIsFrozen(false);
      setPlaybackData(null);
      setPlaybackStatus("");
      setIsLooping(false);
      setPlaybackSpeed(1.0);
  };

  // --- STORAGE LOGIC (Updated for IndexedDB) ---

  // Updated saveData to optionally accept classes state to ensure immediate consistency
  const saveData = async (silent = false, classesOverride?: MLClass[]) => {
      if (!classifierRef.current) return;
      const dataset = classifierRef.current.getClassifierDataset();
      const datasetObj: {[key: string]: number[][]} = {};
      Object.keys(dataset).forEach((key) => {
          const data = dataset[key].arraySync();
          // @ts-ignore 
          datasetObj[key] = data; 
      });
      const now = new Date();

      // Explicitly map only serializable properties to ensure no React Symbols/Components (like icons) are passed to IndexedDB
      const classesToSave = (classesOverride || classes).map(c => ({
          id: c.id,
          name: c.name,
          exampleCount: c.exampleCount,
          confidence: 0, // Reset confidence for storage
          color: c.color,
          handType: c.handType,
          type: c.type
      }));

      const payload = {
          dataset: datasetObj,
          classes: classesToSave,
          timestamp: now.toISOString(),
          version: 7
      };
      try {
          // Force wait for IndexedDB save
          await saveToIndexedDB(STORAGE_KEY, payload);
          setLastSaved(now);
          if (!silent) console.log("Auto-saved to IndexedDB.");
      } catch (e) {
          console.error("Save failed", e);
          if (!silent) alert("Lỗi: Không thể lưu dữ liệu vào bộ nhớ máy (IndexedDB Error).");
      }
  };

  const loadData = async (silent = false) => {
      let parsed: any = null;
      
      // 1. Try IndexedDB first
      try {
          parsed = await loadFromIndexedDB(STORAGE_KEY);
      } catch (e) {
          console.warn("Failed to load from IDB", e);
      }

      // 2. Migration Fallback: Check LocalStorage
      if (!parsed) {
        const json = localStorage.getItem(STORAGE_KEY);
        if (json) {
            try {
                parsed = JSON.parse(json);
                if (!silent) console.log("Migrating data from LocalStorage to IndexedDB...");
                await saveToIndexedDB(STORAGE_KEY, parsed);
            } catch (e) {
                console.error("Migration error", e);
            }
        }
      }

      if (!parsed) {
          if(!silent) alert("Không tìm thấy dữ liệu tương thích (v7 - Extended Temporal) trong máy.");
          return;
      }

      try {
          const { dataset, classes: savedClasses, timestamp } = parsed;
          if (classifierRef.current) {
              const tensorObj: {[key: string]: tf.Tensor} = {};
              Object.keys(dataset).forEach((key) => {
                  const data = dataset[key];
                  tensorObj[key] = tf.tensor(data, [data.length, data[0].length]);
              });
              classifierRef.current.setClassifierDataset(tensorObj);
              const restoredClasses = savedClasses.map((c: MLClass) => {
                const defaultClass = DEFAULT_CLASSES.find(dc => dc.id === c.id);
                return {
                    ...c,
                    icon: defaultClass ? defaultClass.icon : Hand,
                    handType: c.handType || (defaultClass ? defaultClass.handType : 'right'),
                    type: c.type || (defaultClass ? defaultClass.type : 'static')
                };
              });
              setClasses(restoredClasses);
              if (timestamp) setLastSaved(new Date(timestamp));
              if(!silent) alert("Đã tải dữ liệu từ máy thành công!");
          }
      } catch (e) {
          console.error("Load failed", e);
          if(!silent) alert("Lỗi khi đọc dữ liệu.");
      }
  };

  // --- CLOUD FIRESTORE LOGIC ---

  const handleCloudSave = async () => {
      if (!classifierRef.current) return;
      if (!isAdmin) {
          alert("Chỉ Admin mới có quyền cập nhật dữ liệu gốc.");
          return;
      }
      setIsCloudLoading(true);
      try {
          await saveToFirestore(classifierRef.current, classes);
          alert("Đã lưu dữ liệu lên Cloud thành công! Mọi người dùng khác sẽ thấy dữ liệu này.");
      } catch (error) {
          alert("Lỗi khi lưu lên Cloud. Kiểm tra console để xem chi tiết.");
          console.error(error);
      } finally {
          setIsCloudLoading(false);
      }
  };

  const handleCloudLoad = async () => {
      if (!classifierRef.current) return;
      setIsCloudLoading(true);
      try {
          const success = await loadFromFirestore(
              classifierRef.current, 
              (newClasses) => {
                  const hydratedClasses = newClasses.map(c => {
                       const defaultClass = DEFAULT_CLASSES.find(dc => dc.id === c.id);
                       return { 
                           ...c, 
                           icon: defaultClass?.icon || Hand,
                           type: c.type || defaultClass?.type || 'static'
                       };
                  });
                  setClasses(hydratedClasses);
              }, 
              setLastSaved
          );
          if (success) alert("Đã tải bộ dữ liệu chuẩn từ Cloud thành công!");
          else alert("Chưa có dữ liệu trên Cloud.");
      } catch (error) {
          alert("Lỗi tải dữ liệu Cloud.");
      } finally {
          setIsCloudLoading(false);
      }
  };

  const saveDataRef = useRef(saveData);
  useEffect(() => { saveDataRef.current = saveData; });
  useEffect(() => {
    const interval = setInterval(() => {
        if (saveDataRef.current && isAdmin) saveDataRef.current(true); 
    }, 60000);
    return () => clearInterval(interval);
  }, [isAdmin]);

  const clearAllExamples = () => {
      if (confirm("Bạn có chắc muốn xóa toàn bộ dữ liệu đã học của tất cả các lớp không?")) {
          if (classifierRef.current) {
              classifierRef.current.clearAllClasses();
              const newClasses = classes.map(c => ({ ...c, exampleCount: 0, confidence: 0 }));
              setClasses(newClasses);
              setPredictedClass(null);
              setSpeechProgress(0);
              setTrainingClassId(null);
              setLastSaved(null);
              clearSequenceBuffer();
              saveData(true, newClasses); // Save immediately
          }
      }
  };

  const resetClass = (id: string, e: React.MouseEvent) => {
      e.stopPropagation();
      if (confirm("Bạn muốn xóa hết dữ liệu mẫu của hành động này để dạy lại?")) {
          if (classifierRef.current) classifierRef.current.clearClass(id);
          // Important: Clear sequence buffer to prevent old sequences from triggering ghostly predictions or mixing with new training
          clearSequenceBuffer();
          
          const newClasses = classes.map(c => c.id === id ? { ...c, exampleCount: 0, confidence: 0 } : c);
          setClasses(newClasses);
          
          // Force save to prevent data reappearing on refresh
          saveData(true, newClasses);
      }
  };

  const deleteClass = (id: string, e: React.MouseEvent) => {
      e.stopPropagation();
      if (confirm("Bạn chắc chắn muốn xóa hoàn toàn hành động này khỏi danh sách?")) {
          if (classifierRef.current) classifierRef.current.clearClass(id);
          // Clear buffer
          clearSequenceBuffer();

          const newClasses = classes.filter(c => c.id !== id);
          setClasses(newClasses);

          if (trainingClassId === id) setTrainingClassId(null);
          if (predictedClass?.id === id) setPredictedClass(null);

          // Force save to prevent data reappearing on refresh
          saveData(true, newClasses);
      }
  };

  const getRandomColor = () => {
      const hue = Math.floor(Math.random() * 360);
      return `hsl(${hue}, 70%, 60%)`;
  };

  const handleAddClass = () => {
      if (!newClassName.trim()) return;
      const newId = `custom_${Date.now()}`;
      const newClass: MLClass = {
          id: newId,
          name: newClassName,
          exampleCount: 0,
          confidence: 0,
          color: getRandomColor(),
          icon: Hand,
          handType: newClassHandType,
          type: newClassType
      };
      const updatedClasses = [...classes, newClass];
      setClasses(updatedClasses);
      setNewClassName("");
      setIsAddingClass(false);
      if (newClassHandType !== activeTab && activeTab !== 'all') setActiveTab(newClassHandType);
      saveData(true, updatedClasses);
  };

  const toggleTraining = (id: string) => {
      if (trainingClassId === id) {
          setTrainingClassId(null);
      } else {
          clearSequenceBuffer(); 
          samplesCollectedRef.current = 0; 
          setTrainingClassId(id);
      }
  };

  const handleQuickTrain = (classId: string) => {
      setShowVisualizer(false);
      if (!isAdmin) alert("Lưu ý: Bạn đang ở chế độ Người dùng. Bạn có thể dạy thử để trải nghiệm, nhưng không thể lưu thay đổi lên Cloud.");
      const cls = classes.find(c => c.id === classId);
      if (cls) {
          setActiveTab(cls.handType);
          setTimeout(() => {
              const element = document.getElementById(`class-item-${classId}`);
              if (element) {
                  element.scrollIntoView({ behavior: 'smooth', block: 'center' });
                  element.classList.add('bg-white/10', 'ring-2', 'ring-purple-500');
                  setTimeout(() => element.classList.remove('bg-white/10', 'ring-2', 'ring-purple-500'), 2000);
              }
          }, 150);
      }
  };

  const openSettings = () => {
      setTempSettings(appSettings);
      setShowSettings(true);
  };

  const applySettings = () => {
      setAppSettings(tempSettings);
      setShowSettings(false);
  };

  const visibleClasses = classes.filter(c => {
      if (activeTab === 'all') return true;
      return c.handType === activeTab;
  });

  const currentMaxConfidence = useMemo(() => {
    if (classes.length === 0) return 0;
    return Math.max(...classes.map(c => c.confidence));
  }, [classes]);

  // --- TRAINING GUIDE LOGIC ---
  const trainingPhase = useMemo(() => {
      const count = samplesCollectedRef.current;
      const currentClass = classes.find(c => c.id === trainingClassId);
      
      let phaseText = "";
      if (currentClass) {
          if (currentClass.handType === 'left') phaseText = " (Tay Trái)";
          if (currentClass.handType === 'right') phaseText = " (Tay Phải)";
          if (currentClass.handType === 'both') phaseText = " (2 Tay)";
      }

      if (currentClass?.type === 'sequence') {
          if (count <= 15) return { id: 1, text: "Làm chậm & Chuẩn" + phaseText, icon: Hand, color: "text-green-400", border: "border-green-500/60" };
          if (count <= 35) return { id: 2, text: "Thực hiện hành động dứt khoát", icon: Scan, color: "text-yellow-400", border: "border-yellow-500/60" };
          return { id: 3, text: "Thử đổi góc độ / hướng", icon: Rotate3D, color: "text-purple-400", border: "border-purple-500/60" };
      } 
      else {
          if (count <= 15) return { id: 1, text: "Giữ yên tay & Đặt ở giữa" + phaseText, icon: Hand, color: "text-green-400", border: "border-green-500/60" };
          if (count <= 35) return { id: 2, text: "Di chuyển xa / gần camera", icon: MoveHorizontal, color: "text-yellow-400", border: "border-yellow-500/60" };
          return { id: 3, text: "Xoay nhẹ cổ tay các hướng", icon: Rotate3D, color: "text-purple-400", border: "border-purple-500/60" };
      }

  }, [samplesCollectedRef.current, classes, trainingClassId]);

  return (
    <div className="relative w-full h-screen bg-black overflow-hidden font-sans flex flex-col md:flex-row">
      <video 
        ref={videoRef} 
        className="absolute opacity-0 pointer-events-none"
        playsInline
        muted
        autoPlay
        style={{ width: '640px', height: '480px' }}
      />
      <input 
        ref={fileInputRef} 
        type="file" 
        accept="image/*" 
        className="hidden" 
        onChange={handleFileChange} 
      />
      <input 
        ref={videoInputRef} 
        type="file" 
        accept="video/*" 
        className="hidden" 
        onChange={handleVideoFileChange} 
      />

      <div className="relative w-full h-[60vh] md:h-full md:flex-1 order-1 md:order-2 transition-all duration-300">
          {/* Layer 1: Canvas for Visualizer (Background/Overlay) */}
          <div className="absolute inset-0 z-0">
             <Canvas>
                <VisualizerScene 
                  activeClass={predictedClass} 
                  handPositions={handPositionsRef}
                  playbackData={playbackData}
                  playbackSpeed={playbackSpeed}
                  isLooping={isLooping}
                  onPlaybackComplete={isLooping ? undefined : stopPlayback}
                />
             </Canvas>
          </div>

          {/* Layer 2: Webcam Container & Overlays */}
          <div className={`absolute inset-0 z-0 flex items-center justify-center pointer-events-none transition-colors duration-500 ${isPlaybackActive ? 'bg-transparent' : 'bg-black'}`}>
                 {!isPlaybackActive && (
                    <div className="relative w-full h-full md:w-auto md:h-auto md:max-w-[90%] md:max-h-[90%] md:aspect-[4/3] md:rounded-2xl overflow-hidden shadow-2xl border border-white/10 bg-black">
                        <WebcamPreview 
                            videoRef={videoRef} 
                            resultsRef={lastResultsRef} 
                            isCameraReady={isCameraReady}
                            isFrozen={isFrozen}
                        />
                    </div>
                 )}

                 {/* Video Processing Overlay */}
                 {videoProcessing.processing && (
                     <div className="absolute inset-0 bg-black/90 flex flex-col items-center justify-center z-50 backdrop-blur-md animate-in fade-in duration-300 pointer-events-auto">
                         <div className="w-24 h-24 relative flex items-center justify-center mb-6">
                             <svg className="w-full h-full -rotate-90">
                                 <circle cx="50%" cy="50%" r="40" className="stroke-white/10 fill-none" strokeWidth="8" />
                                 <circle 
                                    cx="50%" cy="50%" r="40" 
                                    className="stroke-purple-500 fill-none transition-all duration-75 ease-linear" 
                                    strokeWidth="8" 
                                    strokeDasharray={2 * Math.PI * 40}
                                    strokeDashoffset={2 * Math.PI * 40 * ((100 - videoProcessing.progress) / 100)}
                                    strokeLinecap="round"
                                 />
                             </svg>
                             <FileVideo className="absolute text-white animate-pulse" size={32} />
                         </div>
                         <h2 className="text-white font-bold text-2xl mb-2">Đang phân tích Video...</h2>
                         <p className="text-gray-400 text-sm font-mono mb-4">{videoProcessing.progress}% Hoàn thành</p>
                         <p className="text-gray-500 text-xs max-w-sm text-center">
                             Hệ thống đang quét từng khung hình với kỹ thuật <strong>Multi-Pass</strong> (x3 dữ liệu) để đảm bảo độ chính xác cao nhất cho video ngắn.
                         </p>
                     </div>
                 )}

                 {cameraError && (
                    <div className="absolute inset-0 bg-black/90 flex flex-col items-center justify-center z-50 backdrop-blur-md p-8 text-center">
                        <div className="w-20 h-20 bg-red-500/20 rounded-full flex items-center justify-center mb-6">
                            <AlertTriangle className="text-red-500" size={40} />
                        </div>
                        <h2 className="text-red-400 font-bold text-2xl mb-2">Lỗi Camera</h2>
                        <p className="text-gray-300 text-base max-w-md">{cameraError}</p>
                        <p className="text-gray-500 text-xs mt-6 border-t border-white/10 pt-4">
                            Vui lòng kiểm tra quyền truy cập camera trên trình duyệt và tải lại trang.
                        </p>
                    </div>
                 )}

                 {/* Playback Controls Overlay */}
                 {isPlaybackActive && (
                     <>
                        <div className="absolute top-16 left-0 right-0 flex justify-center z-50 pointer-events-none">
                             <div className="bg-black/80 backdrop-blur-md px-6 py-3 rounded-full border border-white/20 flex items-center gap-3 shadow-2xl animate-in slide-in-from-top-5">
                                 <Video className="text-purple-400 animate-pulse" size={20} />
                                 <span className="text-white font-bold font-mono">{playbackStatus}</span>
                             </div>
                        </div>

                        <div className="absolute bottom-8 left-0 right-0 flex flex-col items-center gap-4 z-50 pointer-events-auto">
                             <div className="bg-black/80 backdrop-blur-md px-6 py-3 rounded-2xl border border-white/20 flex items-center gap-6 shadow-2xl animate-in slide-in-from-bottom-5 delay-100">
                                  <button 
                                    onClick={() => setIsLooping(!isLooping)}
                                    className={`p-2 rounded-full transition-all ${isLooping ? 'bg-purple-600 text-white' : 'bg-white/5 text-gray-400 hover:text-white'}`}
                                    title="Lặp lại"
                                  >
                                      <Repeat size={20} />
                                  </button>
                                  
                                  <div className="flex items-center gap-3">
                                      <FastForward size={16} className="text-gray-400" />
                                      <input 
                                        type="range" 
                                        min="0.2" max="3.0" step="0.1" 
                                        value={playbackSpeed}
                                        onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
                                        className="w-24 accent-purple-500 h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                                      />
                                      <span className="text-xs font-mono text-white w-8">{playbackSpeed.toFixed(1)}x</span>
                                  </div>

                                  <button 
                                    onClick={stopPlayback}
                                    className="px-4 py-1.5 bg-red-500/20 hover:bg-red-500/40 text-red-200 border border-red-500/30 rounded-full text-xs font-bold transition-colors"
                                  >
                                      Dừng
                                  </button>
                             </div>
                         </div>
                     </>
                 )}

                 {trainingClassId && (
                     <div className={`absolute top-4 left-4 backdrop-blur-md px-4 py-2 rounded-lg border flex flex-col items-start shadow-lg animate-pulse z-50 ${
                         classes.find(c => c.id === trainingClassId)?.type === 'sequence' 
                         ? 'bg-red-600/90 border-red-400/30' 
                         : 'bg-blue-600/90 border-blue-400/30' 
                     }`}>
                         <span className="text-[10px] text-white/80 font-bold uppercase flex items-center gap-1">
                            {classes.find(c => c.id === trainingClassId)?.type === 'sequence' ? <Video size={10}/> : <Camera size={10}/>}
                            ĐANG DẠY ({classes.find(c => c.id === trainingClassId)?.type === 'sequence' ? 'CHUỖI HÀNH ĐỘNG' : 'DÁNG TAY TĨNH'}):
                         </span>
                         <span className="text-white font-bold text-sm flex items-center gap-2">
                             {classes.find(c => c.id === trainingClassId)?.name}
                         </span>
                     </div>
                 )}

                 {!trainingClassId && !isFrozen && isCameraReady && currentMaxConfidence > 0.4 && currentMaxConfidence < appSettings.predictionThreshold && !isPlaybackActive && !videoProcessing.processing && (
                    <div className="absolute inset-0 border-4 border-dashed border-yellow-400/30 z-20 pointer-events-none rounded-2xl transition-all duration-300 animate-pulse">
                        <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-yellow-500/90 text-black text-[10px] font-bold px-3 py-1 rounded-full backdrop-blur-md shadow-lg flex items-center gap-2">
                           <span className="w-1.5 h-1.5 rounded-full bg-black animate-ping"></span>
                           Độ tin cậy thấp: {(currentMaxConfidence * 100).toFixed(0)}%
                        </div>
                    </div>
                 )}

                 {isModelLoading && !cameraError && (
                     <div className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center z-50 backdrop-blur-sm">
                        <Loader2 className="text-purple-500 animate-spin mb-3" size={48} />
                        <p className="text-white font-mono text-sm animate-pulse">Đang khởi tạo AI (v7)...</p>
                     </div>
                 )}
                 
                 {isFrozen && !isModelLoading && !isPlaybackActive && !cameraError && !videoProcessing.processing && (
                     <div className="absolute top-4 right-4 bg-yellow-500 text-black px-3 py-1 rounded-full text-xs font-bold flex items-center gap-2 shadow-lg z-50 animate-pulse">
                         <Pause size={12} fill="currentColor" />
                         TẠM DỪNG
                     </div>
                 )}

                 <div className="absolute top-4 right-4 flex gap-2 z-40 pointer-events-auto">
                    {isCloudLoading && (
                        <div className="bg-black/60 backdrop-blur px-3 py-1 rounded-full text-xs font-bold text-white flex items-center gap-2 border border-white/20">
                            <Loader2 size={12} className="animate-spin text-blue-400" />
                            Cloud Sync...
                        </div>
                    )}
                    
                    <div className={`px-3 py-1 rounded-full text-[10px] font-bold uppercase flex items-center gap-1.5 backdrop-blur-md border shadow-lg ${isAdmin ? 'bg-red-500/20 text-red-200 border-red-500/40' : 'bg-blue-500/20 text-blue-200 border-blue-500/40'}`}>
                         {isAdmin ? <ShieldCheck size={12} /> : <User size={12} />}
                         {isAdmin ? 'Admin Mode' : 'User Mode'}
                    </div>
                 </div>
                 
                 {trainingClassId && !isFrozen && (
                     <>
                        <div className={`absolute inset-0 pointer-events-none z-30 shadow-[inset_0_0_50px_rgba(0,0,0,0.5)] rounded-2xl border-2 animate-pulse ${
                            trainingPhase.id === 1 ? 'border-green-500/30' : 
                            trainingPhase.id === 2 ? 'border-yellow-500/30' : 
                            'border-purple-500/30'
                        }`}></div>

                        <div className="absolute bottom-12 left-1/2 -translate-x-1/2 flex flex-col items-center gap-3 z-50 w-full px-4 pointer-events-none">
                            <div className={`flex items-center gap-4 bg-black/90 backdrop-blur-xl border pl-5 pr-6 py-3 rounded-full shadow-2xl animate-in slide-in-from-bottom-10 fade-in duration-300 ${trainingPhase.border}`}>
                                <div className="relative flex items-center justify-center w-4 h-4">
                                    <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-100 duration-1000 ${trainingPhase.id === 1 ? 'bg-green-500' : trainingPhase.id === 2 ? 'bg-yellow-500' : 'bg-purple-500'}`}></span>
                                    <span className={`relative inline-flex rounded-full h-3 w-3 ${trainingPhase.id === 1 ? 'bg-green-600' : trainingPhase.id === 2 ? 'bg-yellow-600' : 'bg-purple-600'}`}></span>
                                </div>
                                
                                <div className="flex items-center gap-3">
                                    <trainingPhase.icon size={18} className={trainingPhase.color} />
                                    <span className={`font-bold text-sm tracking-wider uppercase drop-shadow-md ${trainingPhase.color}`}>
                                        {trainingPhase.text}
                                    </span>
                                    <div className="w-px h-4 bg-white/30"></div>
                                    <span className="text-white font-mono text-lg font-bold">{samplesCollectedRef.current}/50</span>
                                </div>
                            </div>
                        </div>

                        <div className="absolute bottom-0 left-0 w-full h-2 bg-white/10 z-40 flex">
                             <div className="flex-1 relative border-r border-black/50 bg-white/5">
                                <div 
                                    className="h-full bg-green-500 transition-all duration-100 ease-linear" 
                                    style={{width: `${Math.min(Math.max(0, samplesCollectedRef.current), 15) / 15 * 100}%`}}
                                ></div>
                             </div>
                             <div className="flex-[1.3] relative border-r border-black/50 bg-white/5">
                                <div 
                                    className="h-full bg-yellow-500 transition-all duration-100 ease-linear" 
                                    style={{width: `${Math.min(Math.max(0, samplesCollectedRef.current - 15), 20) / 20 * 100}%`}}
                                ></div>
                             </div>
                             <div className="flex-1 relative bg-white/5">
                                <div 
                                    className="h-full bg-purple-500 transition-all duration-100 ease-linear" 
                                    style={{width: `${Math.min(Math.max(0, samplesCollectedRef.current - 35), 15) / 15 * 100}%`}}
                                ></div>
                             </div>
                        </div>
                     </>
                 )}
          </div>

          {showSpeed && (
            <div className="absolute top-24 left-1/2 -translate-x-1/2 z-10 animate-in fade-in slide-in-from-top-2 duration-500 pointer-events-none">
                <div className="bg-black/50 backdrop-blur-sm px-4 py-1.5 rounded-full border border-white/10 flex items-center gap-3 shadow-sm">
                    <Gauge size={14} className="text-cyan-400" />
                    <div className="flex flex-col gap-1 w-24">
                        <div className="flex justify-between text-[9px] text-cyan-100/70 font-mono font-bold uppercase leading-none">
                            <span>TỐC ĐỘ</span>
                            <span>{handSpeed.toFixed(1)}</span>
                        </div>
                        <div className="h-1 w-full bg-white/10 rounded-full overflow-hidden">
                            <div 
                                className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-300 ease-out"
                                style={{ width: `${Math.min(handSpeed * 20, 100)}%` }}
                            />
                        </div>
                    </div>
                </div>
            </div>
          )}

          <div className="absolute top-4 left-0 right-0 flex justify-center z-10 pointer-events-none">
              {predictedClass && predictedClass.id !== '0' && !trainingClassId && !isPlaybackActive && !videoProcessing.processing ? (
                  <div className="bg-black/80 backdrop-blur-xl px-6 py-3 rounded-full border border-white/10 shadow-2xl flex items-center gap-4 animate-in slide-in-from-top-5 fade-in duration-300">
                      <div className="relative w-10 h-10 flex items-center justify-center">
                          <svg className="absolute w-full h-full transform -rotate-90">
                              <circle cx="50%" cy="50%" r="18" stroke="rgba(255,255,255,0.1)" strokeWidth="3" fill="none" />
                              <circle
                                  cx="50%" cy="50%" r="18"
                                  stroke={predictedClass.color} strokeWidth="3" fill="none"
                                  strokeDasharray={2 * Math.PI * 18} strokeDashoffset={2 * Math.PI * 18 - (speechProgress / 100) * (2 * Math.PI * 18)}
                                  strokeLinecap="round" className="transition-all duration-75 ease-linear"
                              />
                          </svg>
                          <div className="z-10">
                            {isSpeaking ? <Volume2 className="animate-bounce text-white" size={14} /> : <Mic className={speechProgress > 0 ? "animate-pulse text-white" : "text-white/50"} size={14} />}
                          </div>
                      </div>
                      
                      <div className="flex flex-col">
                          <h2 className="text-xl font-black text-white leading-none" style={{color: predictedClass.color}}>
                              {predictedClass.name}
                          </h2>
                          <span className="text-[10px] text-gray-400 font-mono leading-none mt-1 flex items-center gap-1.5">
                             <span className={`w-1.5 h-1.5 rounded-full ${predictedClass.type === 'sequence' ? 'bg-red-400' : 'bg-blue-400'}`}></span>
                             {predictedClass.type === 'sequence' ? 'CHUỖI HÀNH ĐỘNG' : 'DÁNG TAY TĨNH'}
                             <span className="mx-1">•</span>
                             {(predictedClass.confidence * 100).toFixed(0)}%
                          </span>
                      </div>
                  </div>
              ) : null}
          </div>
      </div>

      <div className="relative w-full h-[40vh] md:h-full md:w-96 bg-[#111] md:bg-[#0a0a0a] rounded-t-3xl md:rounded-none border-t border-white/10 md:border-t-0 md:border-l flex flex-col z-20 shadow-[0_-10px_40px_rgba(0,0,0,0.5)] order-2 md:order-1 overflow-hidden">
          
          <div className="w-full flex justify-center pt-3 pb-1 md:hidden bg-[#111]">
              <div className="w-12 h-1.5 rounded-full bg-white/20"></div>
          </div>

          <div className="px-4 py-3 border-b border-white/5 flex flex-col gap-3 bg-[#111]">
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                    <BrainCircuit className="text-purple-500" size={20} />
                    <h1 className="text-lg font-bold text-white">Hỗ Trợ Khiếm Thính</h1>
                </div>
                
                <div className="flex items-center gap-2">
                    <button 
                        onClick={() => setShowTranslator(true)}
                        className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-full bg-white/5 hover:bg-white/10 text-gray-300 border border-white/10 transition-colors text-xs font-medium"
                        title="Dịch văn bản sang thủ ngữ"
                    >
                        <MessageSquare size={14} />
                    </button>

                    <button 
                        onClick={() => setIsFrozen(!isFrozen)}
                        className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-full border transition-colors text-xs font-medium ${
                            isFrozen 
                            ? 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30' 
                            : 'bg-white/5 hover:bg-white/10 text-gray-300 border-white/10'
                        }`}
                        title={isFrozen ? "Tiếp tục" : "Dừng hình"}
                    >
                        {isFrozen ? <Play size={14} /> : <Pause size={14} />}
                    </button>
                    
                    <button 
                        onClick={() => setShowVisualizer(true)}
                        className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-full bg-white/5 hover:bg-white/10 text-gray-300 border border-white/10 transition-colors text-xs font-medium"
                        title="Phân tích dữ liệu"
                    >
                        <Activity size={14} />
                    </button>

                    <button 
                        onClick={openSettings}
                        className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-full bg-white/5 hover:bg-white/10 text-gray-300 border border-white/10 transition-colors text-xs font-medium"
                        title="Cài đặt"
                    >
                        <Settings size={14} />
                    </button>
                </div>
              </div>

              <div className="flex bg-black/40 p-1 rounded-xl border border-white/5">
                  {(['all', 'left', 'right', 'both'] as const).map((tab) => (
                      <button
                        key={tab}
                        onClick={() => setActiveTab(tab)}
                        className={`flex-1 py-1.5 text-[10px] font-bold uppercase tracking-wide rounded-lg transition-all flex items-center justify-center gap-1.5 ${
                            activeTab === tab 
                            ? 'bg-white/10 text-white shadow-sm border border-white/5' 
                            : 'text-gray-500 hover:text-gray-300'
                        }`}
                      >
                          {tab === 'all' && <Grid size={12} />}
                          {tab === 'left' && <Hand size={12} className="scale-x-[-1]" />}
                          {tab === 'right' && <Hand size={12} />}
                          {tab === 'both' && <Layers size={12} />}
                          
                          {tab === 'all' ? 'Tất cả' : tab === 'left' ? 'Trái' : tab === 'right' ? 'Phải' : '2 Tay'}
                      </button>
                  ))}
              </div>
          </div>

          <div className="flex-1 overflow-y-auto scroller p-4 space-y-3 pb-20 bg-[#111]">
              
              <div className="px-1 pb-1 flex justify-between items-end">
                  <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">Danh sách cử chỉ</h3>
                  <span className="text-[10px] font-mono text-gray-400 bg-white/5 px-2 py-0.5 rounded-full">
                      Tổng: {visibleClasses.length}
                  </span>
              </div>

              {visibleClasses.map((cls, index) => {
                  const isTraining = trainingClassId === cls.id;
                  const isPredicted = predictedClass?.id === cls.id;
                  const hasExamples = cls.exampleCount > 0;
                  const Icon = cls.icon || Hand;
                  const isSequence = cls.type === 'sequence';

                  return (
                    <div 
                        key={cls.id}
                        id={`class-item-${cls.id}`}
                        className={`relative p-1 rounded-2xl transition-all duration-200 group ${
                            isPredicted 
                            ? 'bg-gradient-to-r from-white/10 to-transparent border border-white/20' 
                            : 'bg-[#1a1a1a] border border-white/5'
                        }`}
                    >
                        <div className="flex items-center p-3 gap-3">
                             <div className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 shadow-inner relative" style={{backgroundColor: `${cls.color}20`}}>
                                <Icon size={20} color={cls.color} />
                                <div className="absolute -bottom-1 -right-1 bg-[#222] rounded-full p-0.5 border border-white/10">
                                    {cls.handType === 'left' && <Hand size={10} className="text-red-400 scale-x-[-1]" />}
                                    {cls.handType === 'right' && <Hand size={10} className="text-blue-400" />}
                                    {cls.handType === 'both' && <Layers size={10} className="text-purple-400" />}
                                </div>
                                <div className="absolute -top-1 -right-1 bg-[#222] rounded-full p-0.5 border border-white/10">
                                    {isSequence ? (
                                        <Video size={10} className="text-yellow-400" />
                                    ) : (
                                        <Camera size={10} className="text-gray-400" />
                                    )}
                                </div>
                            </div>
                            
                            <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 mb-1">
                                    <span className="text-gray-500 font-mono text-xs">#{index + 1}</span>
                                    <span className="font-bold text-white text-sm truncate">{cls.name}</span>
                                </div>
                                <div className="flex items-center gap-3 text-xs text-gray-500">
                                    <span className="font-mono bg-black/30 px-1.5 rounded text-gray-400">
                                        {cls.exampleCount} mẫu
                                    </span>
                                    
                                    <div className="h-1 flex-1 bg-gray-800 rounded-full overflow-hidden max-w-[60px]">
                                        <div 
                                            className="h-full transition-all duration-300"
                                            style={{ width: `${cls.confidence * 100}%`, backgroundColor: cls.color }}
                                        />
                                    </div>
                                </div>
                            </div>
                            
                            <button
                                onClick={() => toggleTraining(cls.id)}
                                disabled={!isAdmin || (trainingClassId !== null && trainingClassId !== cls.id) || isFrozen || videoProcessing.processing}
                                className={`h-10 px-3 rounded-xl font-bold text-sm transition-all flex items-center justify-center gap-2 shadow-lg flex-shrink-0 order-3 ${
                                    isTraining 
                                    ? 'bg-red-500 text-white shadow-red-500/30 min-w-[80px]' 
                                    : isAdmin 
                                        ? 'bg-white/5 hover:bg-white/10 text-white border border-white/10 min-w-[70px]'
                                        : 'bg-white/5 opacity-50 cursor-not-allowed border border-white/10 min-w-[70px]'
                                } ${(trainingClassId !== null && !isTraining) || isFrozen || videoProcessing.processing ? 'opacity-20 blur-[1px]' : ''}`}
                            >
                                {isTraining ? (
                                    <StopCircle size={18} className="animate-pulse" />
                                ) : (
                                    <>
                                        {hasExamples ? (
                                            <span className="text-white text-xs">{isAdmin ? "Thêm" : "Sẵn sàng"}</span>
                                        ) : (
                                            <span className="text-purple-400 text-xs">{isAdmin ? "Dạy" : "Trống"}</span>
                                        )}
                                    </>
                                )}
                            </button>

                            {isAdmin && (
                                <div className="flex items-center gap-1 mr-1 order-4">
                                    {!isTraining && !isFrozen && !videoProcessing.processing && (
                                        <>
                                            <button
                                                onClick={() => triggerVideoUpload(cls.id)}
                                                className="p-2 rounded-full text-gray-500 hover:text-purple-400 hover:bg-white/5 transition-colors"
                                                title="Tải video (MP4) để dạy"
                                            >
                                                <FileVideo size={14} />
                                            </button>
                                            <button
                                                onClick={() => triggerImageUpload(cls.id)}
                                                className="p-2 rounded-full text-gray-500 hover:text-green-400 hover:bg-white/5 transition-colors"
                                                title="Tải ảnh lên để dạy"
                                            >
                                                <ImageIcon size={14} />
                                            </button>
                                        </>
                                    )}
                                    {hasExamples && !isTraining && !isFrozen && !videoProcessing.processing && (
                                        <button 
                                            onClick={(e) => resetClass(cls.id, e)}
                                            className="p-2 rounded-full text-gray-500 hover:text-yellow-400 hover:bg-white/5 transition-colors"
                                            title="Dạy lại"
                                        >
                                            <RefreshCw size={14} />
                                        </button>
                                    )}
                                    {!isTraining && !isFrozen && !videoProcessing.processing && (
                                        <button 
                                            onClick={(e) => deleteClass(cls.id, e)}
                                            className="p-2 rounded-full text-gray-500 hover:text-red-500 hover:bg-white/5 transition-colors"
                                            title="Xóa"
                                        >
                                            <Trash2 size={14} />
                                        </button>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                  );
              })}

              {visibleClasses.length === 0 && (
                  <div className="text-center py-12 px-4 opacity-50">
                      <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-white/5 mb-3">
                          <ListFilter size={20} className="text-gray-400" />
                      </div>
                      <p className="text-gray-400 text-sm">Chưa có hành động nào trong mục này.</p>
                  </div>
              )}

                {isAdmin && (
                    <div className="pt-2">
                        {!isAddingClass ? (
                            <button 
                                onClick={() => setIsAddingClass(true)}
                                className="w-full py-4 border-2 border-dashed border-white/10 rounded-2xl text-gray-500 hover:text-white hover:border-white/30 hover:bg-white/5 text-sm font-bold transition-all flex items-center justify-center gap-2"
                            >
                                <Plus size={18} /> Thêm hành động mới
                            </button>
                        ) : (
                            <div className="p-4 bg-[#1a1a1a] rounded-2xl border border-purple-500/30 animate-in slide-in-from-bottom-2 fade-in duration-200 shadow-xl">
                                <h3 className="text-white text-sm font-bold mb-3 flex items-center gap-2">
                                    <Keyboard size={14} className="text-purple-500"/>
                                    Tạo hành động mới
                                </h3>
                                
                                <input
                                    autoFocus
                                    type="text"
                                    value={newClassName}
                                    onChange={(e) => setNewClassName(e.target.value)}
                                    placeholder="Tên hành động (VD: Xin chào)"
                                    className="w-full bg-black/50 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500 mb-4 transition-all text-sm"
                                    onKeyDown={(e) => e.key === 'Enter' && handleAddClass()}
                                />

                                <div className="mb-4">
                                    <label className="text-[10px] font-bold text-gray-500 uppercase mb-2 block">Dạng Cử chỉ</label>
                                    <div className="flex bg-black/30 p-1 rounded-lg mb-3">
                                        <button
                                            onClick={() => setNewClassType('static')}
                                            className={`flex-1 py-2 rounded-md text-xs font-medium transition-all flex items-center justify-center gap-2 ${
                                                newClassType === 'static' 
                                                ? 'bg-blue-600 text-white shadow-lg' 
                                                : 'text-gray-400 hover:bg-white/5'
                                            }`}
                                        >
                                            <Camera size={12} />
                                            Dáng Tĩnh (Pose)
                                        </button>
                                        <button
                                            onClick={() => setNewClassType('sequence')}
                                            className={`flex-1 py-2 rounded-md text-xs font-medium transition-all flex items-center justify-center gap-2 ${
                                                newClassType === 'sequence' 
                                                ? 'bg-red-600 text-white shadow-lg' 
                                                : 'text-gray-400 hover:bg-white/5'
                                            }`}
                                        >
                                            <Video size={12} />
                                            Chuyển động (Action)
                                        </button>
                                    </div>

                                    <label className="text-[10px] font-bold text-gray-500 uppercase mb-2 block">Sử dụng tay</label>
                                    <div className="flex bg-black/30 p-1 rounded-lg">
                                        {(['left', 'right', 'both'] as const).map(type => (
                                            <button
                                                key={type}
                                                onClick={() => setNewClassHandType(type)}
                                                className={`flex-1 py-2 rounded-md text-xs font-medium transition-all flex items-center justify-center gap-2 ${
                                                    newClassHandType === type 
                                                    ? 'bg-purple-600 text-white shadow-lg' 
                                                    : 'text-gray-400 hover:bg-white/5'
                                                }`}
                                            >
                                                {type === 'left' && <Hand size={12} className="scale-x-[-1]" />}
                                                {type === 'right' && <Hand size={12} />}
                                                {type === 'both' && <Layers size={12} />}
                                                {type === 'left' ? 'Trái' : type === 'right' ? 'Phải' : '2 Tay'}
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                <div className="flex gap-3">
                                    <button 
                                        onClick={() => setIsAddingClass(false)} 
                                        className="flex-1 py-3 bg-white/5 hover:bg-white/10 text-gray-300 text-sm rounded-xl font-medium transition-colors"
                                    >
                                        Hủy
                                    </button>
                                    <button 
                                        onClick={handleAddClass} 
                                        disabled={!newClassName.trim()} 
                                        className="flex-1 py-3 bg-purple-600 hover:bg-purple-500 disabled:opacity-50 disabled:hover:bg-purple-600 text-white text-sm rounded-xl font-bold shadow-lg shadow-purple-900/20 transition-all flex items-center justify-center gap-2"
                                    >
                                        <Check size={16} />
                                        Tạo
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {lastSaved && (
                    <div className="w-full flex justify-center mt-4 pb-6">
                        <div className="flex items-center gap-2 px-3 py-1.5 bg-white/5 rounded-full border border-white/5 transition-all duration-500 hover:bg-white/10">
                             <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></div>
                             <span className="text-[10px] text-gray-400 font-mono">
                                 Đã lưu: {lastSaved.toLocaleTimeString()}
                             </span>
                             {isAdmin && (
                                 <button onClick={clearAllExamples} className="ml-2 text-red-400 hover:text-red-300" title="Xóa tất cả">
                                     <RotateCcw size={10} />
                                 </button>
                             )}
                        </div>
                    </div>
                )}
          </div>
      </div>

      {/* HAND SELECTION MODAL FOR VIDEO UPLOAD */}
      {showHandSelector && targetClassForHandSelect && (
          <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/90 backdrop-blur-md animate-in fade-in duration-200">
              <div className="w-full max-w-md bg-[#151515] border border-white/10 rounded-3xl shadow-2xl p-6">
                  <div className="text-center mb-6">
                      <div className="w-16 h-16 bg-purple-500/20 rounded-full flex items-center justify-center mx-auto mb-4 border border-purple-500/30">
                          <CheckCircle2 className="text-purple-400" size={32} />
                      </div>
                      <h2 className="text-xl font-bold text-white mb-2">Xác nhận Chế độ Tay</h2>
                      <p className="text-gray-400 text-sm">
                          Để AI học chính xác từ video, vui lòng chọn tay mà người trong video sẽ sử dụng. Hệ thống sẽ <strong>bỏ qua</strong> tay còn lại.
                      </p>
                      <div className="mt-3 px-4 py-2 bg-white/5 rounded-lg text-sm text-white font-bold border border-white/10 inline-block">
                          {classes.find(c => c.id === targetClassForHandSelect)?.name}
                      </div>
                  </div>

                  <div className="grid grid-cols-1 gap-3 mb-6">
                      <button 
                          onClick={() => handleHandSelect('right')}
                          className="flex items-center justify-between p-4 bg-[#1a1a1a] hover:bg-blue-500/20 border border-white/10 hover:border-blue-500 rounded-xl transition-all group"
                      >
                          <div className="flex items-center gap-3">
                              <Hand className="text-blue-400" />
                              <div className="text-left">
                                  <div className="text-white font-bold text-sm">Tay Phải</div>
                                  <div className="text-xs text-gray-500 group-hover:text-blue-200">Chỉ lấy dữ liệu tay phải</div>
                              </div>
                          </div>
                          <div className="w-4 h-4 rounded-full border-2 border-gray-600 group-hover:border-blue-500"></div>
                      </button>

                      <button 
                          onClick={() => handleHandSelect('left')}
                          className="flex items-center justify-between p-4 bg-[#1a1a1a] hover:bg-red-500/20 border border-white/10 hover:border-red-500 rounded-xl transition-all group"
                      >
                          <div className="flex items-center gap-3">
                              <Hand className="text-red-400 scale-x-[-1]" />
                              <div className="text-left">
                                  <div className="text-white font-bold text-sm">Tay Trái</div>
                                  <div className="text-xs text-gray-500 group-hover:text-red-200">Chỉ lấy dữ liệu tay trái</div>
                              </div>
                          </div>
                          <div className="w-4 h-4 rounded-full border-2 border-gray-600 group-hover:border-red-500"></div>
                      </button>

                      <button 
                          onClick={() => handleHandSelect('both')}
                          className="flex items-center justify-between p-4 bg-[#1a1a1a] hover:bg-purple-500/20 border border-white/10 hover:border-purple-500 rounded-xl transition-all group"
                      >
                          <div className="flex items-center gap-3">
                              <Layers className="text-purple-400" />
                              <div className="text-left">
                                  <div className="text-white font-bold text-sm">Cả 2 Tay</div>
                                  <div className="text-xs text-gray-500 group-hover:text-purple-200">Lấy dữ liệu cả hai tay</div>
                              </div>
                          </div>
                          <div className="w-4 h-4 rounded-full border-2 border-gray-600 group-hover:border-purple-500"></div>
                      </button>
                  </div>

                  <button 
                      onClick={() => { setShowHandSelector(false); setTargetClassForHandSelect(null); }}
                      className="w-full py-3 bg-white/5 hover:bg-white/10 text-gray-300 text-sm rounded-xl font-medium transition-colors"
                  >
                      Hủy bỏ
                  </button>
              </div>
          </div>
      )}

      {/* TRANSLATOR MODAL */}
      {showTranslator && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="w-full max-w-md bg-[#151515] border border-white/10 rounded-3xl shadow-2xl overflow-hidden">
                <div className="flex items-center justify-between p-6 border-b border-white/5">
                    <h2 className="text-xl font-bold text-white flex items-center gap-2">
                        <MessageSquare className="text-purple-500" />
                        Dịch Văn bản sang Thủ ngữ
                    </h2>
                    <button onClick={() => setShowTranslator(false)} className="text-gray-400 hover:text-white transition-colors">
                        <X size={24} />
                    </button>
                </div>
                
                <div className="p-6 space-y-4">
                    <p className="text-xs text-gray-400">
                        Nhập câu văn bên dưới (VD: "Con muốn ăn"). AI sẽ tìm kiếm trong kho dữ liệu đã học để tái tạo lại chuyển động tay 3D.
                    </p>
                    
                    <div className="relative">
                        <input 
                            type="text" 
                            value={translateInput}
                            onChange={(e) => setTranslateInput(e.target.value)}
                            placeholder="Nhập văn bản..."
                            className="w-full bg-black/50 border border-white/10 rounded-xl px-4 py-3 pl-10 text-white focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500 transition-all"
                            onKeyDown={(e) => e.key === 'Enter' && handleTranslate()}
                        />
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={18} />
                    </div>

                    <div className="flex flex-wrap gap-2 mt-2">
                        <span className="text-[10px] text-gray-500 font-bold uppercase">Gợi ý:</span>
                        {classes.filter(c => c.id !== '0').slice(0, 5).map(c => (
                            <button 
                                key={c.id} 
                                onClick={() => setTranslateInput(prev => prev ? `${prev} ${c.name}` : c.name)}
                                className="text-[10px] px-2 py-1 bg-white/5 rounded hover:bg-white/10 text-gray-300 border border-white/5 transition-colors"
                            >
                                {c.name}
                            </button>
                        ))}
                    </div>
                </div>

                <div className="p-6 border-t border-white/5 flex gap-3 bg-[#111]">
                    <button 
                        onClick={() => setShowTranslator(false)}
                        className="flex-1 py-3 bg-white/5 hover:bg-white/10 text-white rounded-xl font-bold text-sm transition-colors"
                    >
                        Đóng
                    </button>
                    <button 
                        onClick={() => { setShowTranslator(false); handleTranslate(); }}
                        className="flex-1 py-3 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 text-white rounded-xl font-bold text-sm shadow-lg transition-colors flex items-center justify-center gap-2"
                    >
                        <Play size={16} fill="currentColor"/>
                        Dịch & Phát
                    </button>
                </div>
            </div>
        </div>
      )}

      {showSettings && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="w-full max-w-md bg-[#151515] border border-white/10 rounded-3xl shadow-2xl overflow-hidden">
                <div className="flex items-center justify-between p-6 border-b border-white/5">
                    <h2 className="text-xl font-bold text-white flex items-center gap-2">
                        <Settings className="text-purple-500" />
                        Cấu hình Hệ thống
                    </h2>
                    <button onClick={() => setShowSettings(false)} className="text-gray-400 hover:text-white transition-colors">
                        <X size={24} />
                    </button>
                </div>
                
                <div className="p-6 space-y-6 overflow-y-auto max-h-[60vh]">
                    
                    <div className={`p-4 rounded-2xl border transition-all ${isAdmin ? 'bg-red-900/10 border-red-500/30' : 'bg-blue-900/10 border-blue-500/30'}`}>
                        <div className="flex items-center justify-between mb-2">
                            <h3 className={`font-bold flex items-center gap-2 ${isAdmin ? 'text-red-300' : 'text-blue-300'}`}>
                                {isAdmin ? <ShieldCheck size={18} /> : <User size={18} />}
                                {isAdmin ? 'Chế độ Quản trị viên' : 'Chế độ Người dùng'}
                            </h3>
                            <button 
                                onClick={() => setIsAdmin(!isAdmin)}
                                className={`px-3 py-1 rounded-full text-xs font-bold transition-colors ${isAdmin ? 'bg-red-500 text-white' : 'bg-blue-500 text-white'}`}
                            >
                                {isAdmin ? 'Tắt Admin' : 'Bật Admin'}
                            </button>
                        </div>
                        <p className="text-xs text-gray-400">
                            {isAdmin 
                                ? "Bạn có toàn quyền thêm, xóa cử chỉ và LƯU thay đổi lên Cloud cho mọi người dùng." 
                                : "Bạn chỉ có thể sử dụng dữ liệu có sẵn. Để dạy máy, hãy bật chế độ Admin."}
                        </p>
                    </div>

                    <div className="bg-white/5 rounded-2xl p-4 border border-white/5">
                        <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
                            <Cloud className="text-cyan-400" size={16} />
                            Đồng bộ Dữ liệu (Cloud)
                        </h3>
                        
                        <div className="grid grid-cols-2 gap-3">
                            <button
                                onClick={handleCloudLoad}
                                disabled={isCloudLoading}
                                className="flex flex-col items-center justify-center gap-2 p-3 bg-black/30 hover:bg-black/50 border border-white/10 rounded-xl transition-all text-center group"
                            >
                                <DownloadCloud className="text-blue-400 group-hover:scale-110 transition-transform" size={24} />
                                <div>
                                    <div className="text-xs font-bold text-gray-300">Tải Dữ Liệu</div>
                                    <div className="text-[9px] text-gray-500">Lấy bản chuẩn từ Cloud</div>
                                </div>
                            </button>

                            <button
                                onClick={handleCloudSave}
                                disabled={!isAdmin || isCloudLoading}
                                className={`flex flex-col items-center justify-center gap-2 p-3 border rounded-xl transition-all text-center group ${
                                    !isAdmin 
                                    ? 'bg-white/5 border-white/5 opacity-30 cursor-not-allowed' 
                                    : 'bg-black/30 hover:bg-black/50 border-red-500/30'
                                }`}
                            >
                                <Upload className={`${isAdmin ? 'text-red-400 group-hover:scale-110' : 'text-gray-500'} transition-transform`} size={24} />
                                <div>
                                    <div className={`text-xs font-bold ${isAdmin ? 'text-red-200' : 'text-gray-500'}`}>Lưu lên Cloud</div>
                                    <div className="text-[9px] text-gray-500">Cập nhật bản gốc</div>
                                </div>
                            </button>
                        </div>
                    </div>

                    <div className="h-px bg-white/5"></div>

                    <div className="space-y-3">
                        <div className="flex justify-between items-center">
                            <label className="text-sm font-bold text-white">Ngưỡng dự đoán</label>
                            <span className="text-xs font-mono bg-purple-500/20 text-purple-300 px-2 py-0.5 rounded">
                                {Math.round(tempSettings.predictionThreshold * 100)}%
                            </span>
                        </div>
                        <input 
                            type="range" min="0.5" max="0.99" step="0.01"
                            value={tempSettings.predictionThreshold}
                            onChange={(e) => setTempSettings({...tempSettings, predictionThreshold: parseFloat(e.target.value)})}
                            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                        />
                    </div>

                    <div className="space-y-3 opacity-50 hover:opacity-100 transition-opacity">
                         <label className="text-xs font-bold text-gray-500 uppercase">Nâng cao</label>
                        <div className="flex justify-between items-center">
                            <label className="text-xs font-medium text-gray-300">Độ nhạy phát hiện tay</label>
                            <span className="text-[10px] font-mono text-gray-400">{tempSettings.minHandDetectionConfidence.toFixed(2)}</span>
                        </div>
                        <input 
                            type="range" min="0.2" max="0.9" step="0.05"
                            value={tempSettings.minHandDetectionConfidence}
                            onChange={(e) => setTempSettings({...tempSettings, minHandDetectionConfidence: parseFloat(e.target.value)})}
                            className="w-full h-1.5 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                        />
                    </div>
                </div>

                <div className="p-6 border-t border-white/5 flex gap-3 bg-[#111]">
                    <button 
                        onClick={() => setShowSettings(false)}
                        className="flex-1 py-3 bg-white/5 hover:bg-white/10 text-white rounded-xl font-bold text-sm transition-colors"
                    >
                        Hủy
                    </button>
                    <button 
                        onClick={applySettings}
                        className="flex-1 py-3 bg-purple-600 hover:bg-purple-500 text-white rounded-xl font-bold text-sm shadow-lg shadow-purple-900/20 transition-colors"
                    >
                        Áp dụng
                    </button>
                </div>
            </div>
        </div>
      )}

      {showVisualizer && (
        <DataVisualizer 
            classifier={classifierRef.current}
            classes={classes}
            onClose={() => setShowVisualizer(false)}
            onTrainClass={handleQuickTrain}
        />
      )}

    </div>
  );
};

export default App;
