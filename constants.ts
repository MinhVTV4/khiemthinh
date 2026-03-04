
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import { MLClass } from './types';
import { 
  Hand, 
  Heart, 
  ThumbsUp, 
  ThumbsDown, 
  Smile, 
  Frown, 
  Utensils, 
  Users, 
  Droplets, 
  Star, 
  HelpCircle, 
  XCircle, 
  LifeBuoy,
  Sparkles,
  LogOut
} from 'lucide-react';

export const DEFAULT_CLASSES: MLClass[] = [
  { id: '0', name: 'Không cử chỉ', exampleCount: 0, confidence: 0, color: '#9ca3af', icon: XCircle, handType: 'both', type: 'static' },
  { id: '1', name: 'Xin Chào (Hello)', exampleCount: 0, confidence: 0, color: '#4ade80', icon: Hand, handType: 'right', type: 'static' },
  { id: '2', name: 'Cảm Ơn (Thanks)', exampleCount: 0, confidence: 0, color: '#facc15', icon: Sparkles, handType: 'both', type: 'sequence' }, // Action
  { id: '3', name: 'Tôi Yêu Bạn (ILY)', exampleCount: 0, confidence: 0, color: '#ec4899', icon: Heart, handType: 'right', type: 'static' },
  { id: '4', name: 'Giúp Đỡ (Help)', exampleCount: 0, confidence: 0, color: '#60a5fa', icon: LifeBuoy, handType: 'both', type: 'sequence' }, // Action
  { id: '5', name: 'Đồng Ý (Yes)', exampleCount: 0, confidence: 0, color: '#34d399', icon: ThumbsUp, handType: 'right', type: 'static' },
  { id: '6', name: 'Không (No)', exampleCount: 0, confidence: 0, color: '#f87171', icon: ThumbsDown, handType: 'right', type: 'static' },
  { id: '7', name: 'Vui Vẻ (Happy)', exampleCount: 0, confidence: 0, color: '#fb923c', icon: Smile, handType: 'both', type: 'sequence' }, // Action
  { id: '8', name: 'Xin Lỗi (Sorry)', exampleCount: 0, confidence: 0, color: '#a78bfa', icon: Frown, handType: 'right', type: 'sequence' }, // Action
  { id: '9', name: 'Ăn Uống (Eat)', exampleCount: 0, confidence: 0, color: '#22d3ee', icon: Utensils, handType: 'right', type: 'sequence' }, // Action
  { id: '10', name: 'Gia Đình (Family)', exampleCount: 0, confidence: 0, color: '#e879f9', icon: Users, handType: 'both', type: 'sequence' }, // Action
  { id: '11', name: 'Nước (Water)', exampleCount: 0, confidence: 0, color: '#0ea5e9', icon: Droplets, handType: 'right', type: 'sequence' }, // Action
  { id: '12', name: 'Tốt (Good)', exampleCount: 0, confidence: 0, color: '#84cc16', icon: Star, handType: 'right', type: 'static' },
  { id: '13', name: 'Tạm Biệt (Bye)', exampleCount: 0, confidence: 0, color: '#f43f5e', icon: LogOut, handType: 'right', type: 'sequence' }, // Action
  { id: '14', name: 'Tại Sao (Why)', exampleCount: 0, confidence: 0, color: '#64748b', icon: HelpCircle, handType: 'right', type: 'static' },
];

export const OBSTACLE_SIZE = 0.8;

// Helper to calculate distance between two landmarks
export const distance = (p1: {x: number, y: number}, p2: {x: number, y: number}) => {
    return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
};
