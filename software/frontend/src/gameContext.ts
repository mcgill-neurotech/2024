import { createContext } from 'react';
import { useGameSocketReturn } from './useSocket';

export const GameContext = createContext<useGameSocketReturn|null>(null);
