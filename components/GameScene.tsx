
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useRef, useMemo, useState } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { Stars, Float, Grid } from '@react-three/drei';
import * as THREE from 'three';
import { MLClass } from '../types';
import HandModel from './HandModel';

interface VisualizerProps {
  activeClass: MLClass | null;
  handPositions?: React.MutableRefObject<{
    leftVelocity: THREE.Vector3;
    rightVelocity: THREE.Vector3;
  }>;
  playbackData?: number[][] | null; // Sequence of frames for playback
  playbackSpeed?: number;
  isLooping?: boolean;
  onPlaybackComplete?: () => void;
}

const VisualizerScene: React.FC<VisualizerProps> = ({ 
    activeClass, 
    handPositions, 
    playbackData,
    playbackSpeed = 1.0,
    isLooping = false,
    onPlaybackComplete
}) => {
  const particlesRef = useRef<THREE.Points>(null);
  const { camera } = useThree();
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const frameProgress = useRef(0);
  
  // Adjust camera when playback starts to ensure visibility
  React.useEffect(() => {
      if (playbackData) {
          // Move camera back to see the full LARGE hand scale (6x scale)
          camera.position.set(0, 2, 25);
          camera.lookAt(0, 0, 0);
          frameProgress.current = 0;
          setCurrentFrameIndex(0);
      } else {
          // Default camera for particles
          camera.position.set(0, 0, 5);
          camera.lookAt(0, 0, 0);
      }
  }, [playbackData, camera]);

  const particlesCount = 1200;
  const positions = useMemo(() => {
    const pos = new Float32Array(particlesCount * 3);
    for(let i=0; i<particlesCount; i++) {
      pos[i*3] = (Math.random() - 0.5) * 25; // x
      pos[i*3+1] = (Math.random() - 0.5) * 25; // y
      pos[i*3+2] = (Math.random() - 0.5) * 15; // z
    }
    return pos;
  }, []);

  useFrame((state, delta) => {
    if (playbackData) {
        // Playback Logic
        if (playbackData.length > 0) {
            // 30fps base speed * speed multiplier
            const fps = 30 * playbackSpeed;
            frameProgress.current += delta * fps;
            
            let nextIndex = Math.floor(frameProgress.current);
            
            if (nextIndex >= playbackData.length) {
                if (isLooping) {
                    frameProgress.current = 0;
                    nextIndex = 0;
                } else {
                    nextIndex = playbackData.length - 1;
                    if (onPlaybackComplete) onPlaybackComplete();
                }
            }
            setCurrentFrameIndex(nextIndex);
        }
        return; 
    }

    const time = state.clock.getElapsedTime();

    // Calculate total velocity magnitude from both hands
    let totalSpeed = 0;
    if (handPositions && handPositions.current) {
        const leftSpeed = handPositions.current.leftVelocity.length();
        const rightSpeed = handPositions.current.rightVelocity.length();
        totalSpeed = (leftSpeed + rightSpeed) * 0.5; // Average speed
    }

    // Smooth clamping for visual effects
    const speedIntensity = Math.min(totalSpeed * 0.5, 5.0);

    if (particlesRef.current) {
      // Base rotation + Speed rotation
      // When hand moves fast, the universe spins faster
      particlesRef.current.rotation.y += 0.002 + (speedIntensity * 0.02);
      particlesRef.current.rotation.z = Math.sin(time * 0.1) * 0.1 + (speedIntensity * 0.05);

      // Scale Effect: Expansion when moving fast
      const targetScale = activeClass && activeClass.id !== '0' ? 1.5 : 1.0;
      const speedScale = 1.0 + (speedIntensity * 0.3); // Expands up to 1.5x more with speed
      
      const currentScale = particlesRef.current.scale.x;
      
      // Smooth transition
      const combinedTargetScale = targetScale * speedScale;
      const newScale = THREE.MathUtils.lerp(currentScale, combinedTargetScale, 0.05);
      
      // Beat effect if active
      const pulse = activeClass && activeClass.id !== '0' ? Math.sin(time * 10) * 0.1 : 0;
      
      particlesRef.current.scale.setScalar(newScale + pulse);
    }
  });

  // Determine color
  const baseColor = new THREE.Color(activeClass ? activeClass.color : '#ffffff');
  
  return (
    <>
      <color attach="background" args={['#050505']} />
      {/* Standard lights to ensure materials are visible */}
      <ambientLight intensity={0.8} />
      <pointLight position={[10, 10, 10]} intensity={1.0} />
      <pointLight position={[-10, -10, 10]} intensity={0.5} />
      
      {playbackData ? (
        // Playback Mode: Show 3D Skeleton Hand
        // Scale increased to 6x for larger visuals as requested
        <group position={[0, -5, 0]} scale={[6, 6, 6]}>
             {playbackData[currentFrameIndex] && (
                 <HandModel landmarks={playbackData[currentFrameIndex]} opacity={1.0} />
             )}
             {/* Expanded Grid for wider feeling */}
             <Grid args={[60, 60]} cellColor="#222" sectionColor="#333" fadeDistance={40} position={[0, -2, 0]} />
        </group>
      ) : (
        // Normal Mode: Abstract Particles
        <Float speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
            <points ref={particlesRef}>
            <bufferGeometry>
                <bufferAttribute
                attach="attributes-position"
                count={particlesCount}
                array={positions}
                itemSize={3}
                />
            </bufferGeometry>
            <pointsMaterial 
                size={0.2} 
                color={baseColor}
                transparent 
                opacity={0.7} 
                sizeAttenuation 
                blending={THREE.AdditiveBlending}
            />
            </points>
        </Float>
      )}

      <Stars radius={100} count={5000} factor={4} fade speed={0.5} />
    </>
  );
};

export default VisualizerScene;
