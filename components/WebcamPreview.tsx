
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/


import React, { useEffect, useRef } from 'react';
import { HandLandmarkerResult } from '@mediapipe/tasks-vision';
import { COLORS } from '../types';

interface WebcamPreviewProps {
    videoRef: React.RefObject<HTMLVideoElement | null>;
    resultsRef: React.MutableRefObject<HandLandmarkerResult | null>;
    isCameraReady: boolean;
    isFrozen?: boolean;
}

const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8], // Index
    [0, 9], [9, 10], [10, 11], [11, 12], // Middle
    [0, 13], [13, 14], [14, 15], [15, 16], // Ring
    [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
    [5, 9], [9, 13], [13, 17], [0, 5], [0, 17] // Palm
];

const WebcamPreview: React.FC<WebcamPreviewProps> = ({ videoRef, resultsRef, isCameraReady, isFrozen = false }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const isFrozenRef = useRef(isFrozen);

    useEffect(() => {
        isFrozenRef.current = isFrozen;
    }, [isFrozen]);

    useEffect(() => {
        if (!isCameraReady) return;
        let animationFrameId: number;

        const render = () => {
            // If frozen, we stop drawing new frames so the last one persists.
            // We keep the loop alive to be ready to resume immediately.
            if (isFrozenRef.current) {
                animationFrameId = requestAnimationFrame(render);
                return;
            }

            const canvas = canvasRef.current;
            const video = videoRef.current;

            if (canvas && video && video.readyState >= 2) { 
                const ctx = canvas.getContext('2d');
                if (ctx) {
                    if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
                    if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;

                    ctx.clearRect(0, 0, canvas.width, canvas.height);

                    // 1. Draw Video Feed (Mirrored)
                    ctx.save();
                    ctx.scale(-1, 1);
                    ctx.translate(-canvas.width, 0);
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    ctx.restore();

                    // 2. Draw Landmarks Overlay
                    if (resultsRef.current && resultsRef.current.landmarks) {
                        for (let i = 0; i < resultsRef.current.landmarks.length; i++) {
                            const landmarks = resultsRef.current.landmarks[i];
                            const handInfo = resultsRef.current.handedness[i];
                            const isRight = handInfo && handInfo[0].categoryName === 'Right';
                            
                            // NOTE: 'Right' in MediaPipe when mirrored is actually the user's LEFT visual hand if using standard mirror logic,
                            // but generally we color code based on detection.
                            const color = isRight ? COLORS.right : COLORS.left;
                            const label = isRight ? "PHẢI" : "TRÁI";

                            ctx.lineWidth = 4;
                            ctx.lineCap = 'round';
                            ctx.lineJoin = 'round';

                            // Draw connections
                            ctx.strokeStyle = color;
                            ctx.beginPath();
                            for (const [start, end] of HAND_CONNECTIONS) {
                                const p1 = landmarks[start];
                                const p2 = landmarks[end];
                                ctx.moveTo((1 - p1.x) * canvas.width, p1.y * canvas.height);
                                ctx.lineTo((1 - p2.x) * canvas.width, p2.y * canvas.height);
                            }
                            ctx.stroke();

                            // Draw joints
                            ctx.fillStyle = 'white';
                            for (const lm of landmarks) {
                                ctx.beginPath();
                                ctx.arc((1 - lm.x) * canvas.width, lm.y * canvas.height, 3, 0, 2 * Math.PI);
                                ctx.fill();
                            }

                            // Draw Label near Wrist (point 0)
                            const wrist = landmarks[0];
                            const wristX = (1 - wrist.x) * canvas.width;
                            const wristY = wrist.y * canvas.height;

                            ctx.font = "bold 16px monospace";
                            ctx.fillStyle = color;
                            ctx.shadowColor = "black";
                            ctx.shadowBlur = 4;
                            ctx.fillText(label, wristX + 10, wristY + 10);
                            ctx.shadowBlur = 0;
                        }
                    }
                }
            }
            animationFrameId = requestAnimationFrame(render);
        };
        render();

        return () => {
            if (animationFrameId) cancelAnimationFrame(animationFrameId);
        };
    }, [isCameraReady, videoRef, resultsRef]); // Remove isFrozen from dep array to avoid loop restart

    return (
        <div className="absolute inset-0 z-0">
            <canvas ref={canvasRef} className="w-full h-full object-cover opacity-60" />
        </div>
    );
};

export default WebcamPreview;
