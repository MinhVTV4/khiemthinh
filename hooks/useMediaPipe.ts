
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useEffect, useRef, useState } from 'react';
import { HandLandmarker, FilesetResolver, HandLandmarkerResult } from '@mediapipe/tasks-vision';
import * as THREE from 'three';
import { AppSettings } from '../types';

// Mapping 2D normalized coordinates to 3D game world.
const mapHandToWorld = (x: number, y: number): THREE.Vector3 => {
  const GAME_X_RANGE = 5; 
  const GAME_Y_RANGE = 3.5;
  const Y_OFFSET = 0.8;

  const worldX = (0.5 - x) * GAME_X_RANGE; 
  const worldY = (1.0 - y) * GAME_Y_RANGE - (GAME_Y_RANGE / 2) + Y_OFFSET;

  const worldZ = -Math.max(0, worldY * 0.2);

  return new THREE.Vector3(worldX, Math.max(0.1, worldY), worldZ);
};

// Export helper to create an independent instance
export const createHandLandmarkerInstance = async (settings: AppSettings): Promise<HandLandmarker> => {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm"
    );
    
    return await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: settings.minHandDetectionConfidence,
      minHandPresenceConfidence: settings.minHandPresenceConfidence,
      minTrackingConfidence: settings.minTrackingConfidence
    });
};

export const useMediaPipe = (
  videoRef: React.RefObject<HTMLVideoElement | null>,
  settings: AppSettings,
  isPaused: boolean
) => {
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const isPausedRef = useRef(isPaused);
  useEffect(() => {
      isPausedRef.current = isPaused;
  }, [isPaused]);

  const handPositionsRef = useRef<{
    left: THREE.Vector3 | null;
    right: THREE.Vector3 | null;
    lastLeft: THREE.Vector3 | null;
    lastRight: THREE.Vector3 | null;
    leftVelocity: THREE.Vector3;
    rightVelocity: THREE.Vector3;
    lastTimestamp: number;
  }>({
    left: null,
    right: null,
    lastLeft: null,
    lastRight: null,
    leftVelocity: new THREE.Vector3(0,0,0),
    rightVelocity: new THREE.Vector3(0,0,0),
    lastTimestamp: 0
  });

  // To expose raw results for UI preview
  const lastResultsRef = useRef<HandLandmarkerResult | null>(null);

  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const requestRef = useRef<number>(0);

  // Initialize and reload MediaPipe when settings change
  useEffect(() => {
    let isActive = true;
    setIsModelLoading(true);

    const setupMediaPipe = async () => {
      try {
        if (landmarkerRef.current) {
          landmarkerRef.current.close();
          landmarkerRef.current = null;
        }

        // Use the exported helper for consistency
        const landmarker = await createHandLandmarkerInstance(settings);

        if (!isActive) {
             landmarker.close();
             return;
        }

        landmarkerRef.current = landmarker;
        setIsModelLoading(false);
        
        if (!videoRef.current?.srcObject) {
            startCamera();
        }
      } catch (err: any) {
        console.error("Error initializing MediaPipe:", err);
        setError(`Failed to load hand tracking: ${err.message}`);
        setIsModelLoading(false);
      }
    };

    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'user',
            width: { ideal: 640 },
            height: { ideal: 480 }
          }
        });

        if (videoRef.current && isActive) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadeddata = () => {
             if (isActive) {
                 setIsCameraReady(true);
                 predictWebcam();
             }
          };
        }
      } catch (err) {
        console.error("Camera Error:", err);
        setError("Could not access camera. Please allow camera permissions.");
      }
    };

    setupMediaPipe();

    return () => {
      isActive = false;
    };
  }, [settings.minHandDetectionConfidence, settings.minHandPresenceConfidence, settings.minTrackingConfidence]);

  const predictWebcam = () => {
      if (!videoRef.current || !landmarkerRef.current) {
          requestRef.current = requestAnimationFrame(predictWebcam);
          return;
      }

      if (isPausedRef.current) {
          requestRef.current = requestAnimationFrame(predictWebcam);
          return;
      }

      const video = videoRef.current;
      if (video.videoWidth > 0 && video.videoHeight > 0) {
           let startTimeMs = performance.now();
           try {
               const results = landmarkerRef.current.detectForVideo(video, startTimeMs);
               lastResultsRef.current = results;
               processResults(results);
           } catch (e) {
               // console.warn("Detection failed this frame", e);
           }
      }

      requestRef.current = requestAnimationFrame(predictWebcam);
  };

  // Helper to process an image (using the shared landmarker instance with current time)
  const detectImage = async (image: HTMLImageElement): Promise<HandLandmarkerResult | null> => {
      if (!landmarkerRef.current) return null;
      try {
          // Using detectForVideo allows us to use the same instance initialized in VIDEO mode.
          // We provide a timestamp to satisfy the API.
          return landmarkerRef.current.detectForVideo(image, performance.now());
      } catch (e) {
          console.error("Image detection failed", e);
          return null;
      }
  };

  useEffect(() => {
     if (isCameraReady && !isModelLoading) {
         cancelAnimationFrame(requestRef.current);
         predictWebcam();
     }
     return () => cancelAnimationFrame(requestRef.current);
  }, [isCameraReady, isModelLoading]);


  const processResults = (results: HandLandmarkerResult) => {
        const now = performance.now();
        const deltaTime = (now - handPositionsRef.current.lastTimestamp) / 1000;
        handPositionsRef.current.lastTimestamp = now;

        let newLeft: THREE.Vector3 | null = null;
        let newRight: THREE.Vector3 | null = null;

        if (results.landmarks) {
          for (let i = 0; i < results.landmarks.length; i++) {
            const landmarks = results.landmarks[i];
            const classification = results.handedness[i][0];
            const isRight = classification.categoryName === 'Right'; 
            
            const tip = landmarks[8];
            const worldPos = mapHandToWorld(tip.x, tip.y);

            if (isRight) {
                 newRight = worldPos; 
            } else {
                 newLeft = worldPos;
            }
          }
        }

        const s = handPositionsRef.current;
        const LERP_POS = 0.6; 
        const LERP_VEL = 0.2;

        if (newLeft) {
            if (s.left) {
                newLeft.lerpVectors(s.left, newLeft, LERP_POS);
                if (deltaTime > 0.001) { 
                     const instantVel = new THREE.Vector3().subVectors(newLeft, s.left).divideScalar(deltaTime);
                     s.leftVelocity.lerp(instantVel, LERP_VEL);
                }
            }
            s.lastLeft = s.left ? s.left.clone() : newLeft.clone();
            s.left = newLeft;
        } else {
            s.left = null;
            s.leftVelocity.lerp(new THREE.Vector3(0,0,0), 0.1);
        }

        if (newRight) {
             if (s.right) {
                 newRight.lerpVectors(s.right, newRight, LERP_POS);
                 if (deltaTime > 0.001) {
                      const instantVel = new THREE.Vector3().subVectors(newRight, s.right).divideScalar(deltaTime);
                      s.rightVelocity.lerp(instantVel, LERP_VEL);
                 }
             }
             s.lastRight = s.right ? s.right.clone() : newRight.clone();
             s.right = newRight;
        } else {
            s.right = null;
            s.rightVelocity.lerp(new THREE.Vector3(0,0,0), 0.1);
        }
  };

  return { isCameraReady, isModelLoading, handPositionsRef, lastResultsRef, error, detectImage };
};
