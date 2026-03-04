
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useMemo } from 'react';
import { Sphere, Cylinder } from '@react-three/drei';
import * as THREE from 'three';

interface HandModelProps {
  landmarks: number[]; // Flat array of coordinates [x, y, z, x, y, z...]
  color?: string;
  opacity?: number;
}

// Defines the skeletal structure
// Metacarpals connected to wrist (0) are included.
// Palm plate lines [5,9], [9,13], etc. are removed as requested.
const JOINTS = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20]
];

const Bone: React.FC<{ start: THREE.Vector3; end: THREE.Vector3; color: string; opacity: number }> = ({ start, end, color, opacity }) => {
    const distance = start.distanceTo(end);
    const position = start.clone().add(end).multiplyScalar(0.5);
    
    const direction = end.clone().sub(start).normalize();
    const up = new THREE.Vector3(0, 1, 0);
    const quaternion = new THREE.Quaternion().setFromUnitVectors(up, direction);

    return (
        <group position={position} quaternion={quaternion}>
            <Cylinder args={[0.14, 0.14, distance, 16]}>
                <meshPhysicalMaterial 
                    color={color} 
                    transparent 
                    opacity={opacity} 
                    roughness={0.2} 
                    metalness={0.9} 
                    clearcoat={1.0} 
                    clearcoatRoughness={0.1} 
                />
            </Cylinder>
        </group>
    );
};

const SingleHand: React.FC<{ points: THREE.Vector3[], color: string, opacity: number }> = ({ points, color, opacity }) => {
    return (
        <group>
            {/* Joints - Hyper Glossy Spheres */}
            {points.map((p, i) => (
                <mesh key={i} position={p}>
                    {/* Increased radius for strong joint look */}
                    <sphereGeometry args={[0.24, 32, 32]} />
                    <meshPhysicalMaterial 
                        color={color} 
                        transparent 
                        opacity={opacity} 
                        roughness={0.1} 
                        metalness={0.8} 
                        clearcoat={1.0}
                    />
                </mesh>
            ))}
            
            {/* Bones - Cylinders */}
            {JOINTS.map(([start, end], i) => {
                if (start >= points.length || end >= points.length) return null;
                return (
                    <Bone 
                        key={`bone-${i}`} 
                        start={points[start]} 
                        end={points[end]} 
                        color={color} 
                        opacity={opacity} 
                    />
                );
            })}
        </group>
    );
};

const HandModel: React.FC<HandModelProps> = ({ landmarks, color = "#00f0ff", opacity = 1 }) => {
    const { leftPoints, rightPoints } = useMemo(() => {
        const lPoints: THREE.Vector3[] = [];
        const rPoints: THREE.Vector3[] = [];
        
        const numPointsPerHand = 21;
        const stride = 3;
        const totalPerHand = numPointsPerHand * stride;

        // Left Hand Data
        let hasLeft = false;
        for(let i=0; i<totalPerHand; i+=3) {
            if (landmarks[i] !== 0 || landmarks[i+1] !== 0 || landmarks[i+2] !== 0) {
                hasLeft = true;
            }
        }

        if (hasLeft) {
            for(let i=0; i<numPointsPerHand; i++) {
                const idx = i * 3;
                // Scaling & Positioning for the 3D Scene
                lPoints.push(new THREE.Vector3(
                    -landmarks[idx] * 5 - 0.8, 
                    -landmarks[idx+1] * 5, 
                    landmarks[idx+2] * 5
                ));
            }
        }

        // Right Hand Data
        let hasRight = false;
        const offset = totalPerHand;
        if (landmarks.length >= offset + totalPerHand) {
             for(let i=0; i<totalPerHand; i+=3) {
                if (landmarks[offset + i] !== 0 || landmarks[offset + i+1] !== 0 || landmarks[offset + i+2] !== 0) {
                    hasRight = true;
                }
            }
        }

        if (hasRight) {
             for(let i=0; i<numPointsPerHand; i++) {
                const idx = offset + (i * 3);
                rPoints.push(new THREE.Vector3(
                    landmarks[idx] * 5 + 0.8, 
                    -landmarks[idx+1] * 5, 
                    landmarks[idx+2] * 5
                ));
            }
        }

        return { leftPoints: lPoints, rightPoints: rPoints };
    }, [landmarks]);

    return (
        <group>
            {leftPoints.length > 0 && <SingleHand points={leftPoints} color="#ff4444" opacity={opacity} />}
            {rightPoints.length > 0 && <SingleHand points={rightPoints} color="#4444ff" opacity={opacity} />}
        </group>
    );
};

export default HandModel;
