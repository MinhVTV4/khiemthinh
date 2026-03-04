
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import * as THREE from 'three';

export interface MLClass {
  id: string;
  name: string;
  exampleCount: number;
  confidence: number; // 0 to 1
  color: string;
  icon?: any;
  handType: 'left' | 'right' | 'both';
  type: 'static' | 'sequence'; // New field: 'static' for poses, 'sequence' for actions
}

export interface HandPositions {
  left: THREE.Vector3 | null;
  right: THREE.Vector3 | null;
  leftVelocity: THREE.Vector3;
  rightVelocity: THREE.Vector3;
}

export interface GameItem {
  id: string;
  x: number;
  z: number;
  type: 'obstacle' | 'coin';
  active: boolean;
}

export interface AppSettings {
  minHandDetectionConfidence: number;
  minHandPresenceConfidence: number;
  minTrackingConfidence: number;
  predictionThreshold: number;
}

export const COLORS = {
  skeleton: '#00f0ff',
  left: '#ff0000', // Strict Red
  right: '#0000ff', // Strict Blue
  text: '#ffffff',
  bg: '#050505',
  active: '#4ade80',
  inactive: '#374151',
  obstacle: '#ef4444',
  coin: '#eab308',
};
