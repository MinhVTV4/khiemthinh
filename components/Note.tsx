
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useRef } from 'react';
import { Box, Sphere, Octahedron } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { GameItem, COLORS } from '../types';
import { OBSTACLE_SIZE } from '../constants';

interface NoteProps {
  data: GameItem;
}

const Item: React.FC<NoteProps> = ({ data }) => {
  const meshRef = useRef<THREE.Group>(null);
  const innerRef = useRef<THREE.Mesh>(null);

  useFrame((state, delta) => {
    if (innerRef.current) {
        // Rotate coins/obstacles for effect
        innerRef.current.rotation.y += delta * 2;
        innerRef.current.rotation.z += delta;
    }
  });

  if (!data.active) return null;

  const isObstacle = data.type === 'obstacle';
  const color = isObstacle ? COLORS.obstacle : COLORS.coin;

  return (
    <group position={[data.x, isObstacle ? 0.4 : 0.5, data.z]} ref={meshRef}>
      {isObstacle ? (
          // OBSTACLE: Spiky Box
          <group>
            <Box args={[OBSTACLE_SIZE, OBSTACLE_SIZE, OBSTACLE_SIZE]} ref={innerRef}>
                <meshStandardMaterial 
                    color={color} 
                    emissive={color} 
                    emissiveIntensity={0.5}
                    roughness={0.2}
                    metalness={0.8}
                />
            </Box>
            {/* Wireframe outline */}
            <Box args={[OBSTACLE_SIZE * 1.1, OBSTACLE_SIZE * 1.1, OBSTACLE_SIZE * 1.1]}>
                <meshBasicMaterial color="white" wireframe transparent opacity={0.1} />
            </Box>
          </group>
      ) : (
          // COIN: Floating Crystal
          <group>
              <Octahedron args={[0.4]} ref={innerRef}>
                  <meshStandardMaterial 
                      color={color} 
                      emissive={color} 
                      emissiveIntensity={0.8}
                      roughness={0}
                      metalness={1}
                  />
              </Octahedron>
              {/* Glow */}
              <pointLight color={color} distance={2} intensity={1} />
          </group>
      )}
    </group>
  );
};

export default React.memo(Item, (prev, next) => {
    return prev.data.z === next.data.z && prev.data.active === next.data.active;
});
