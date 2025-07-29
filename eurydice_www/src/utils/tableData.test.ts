import { describe, it, expect } from 'vitest';
import {
  getAllUniqueOutcomes,
  computeTableData,
  computeDistributionStatistics,
  calculateBracketingProbabilities,
} from './tableData';
import { DisplayMode } from './chartData';
import { Distribution } from '../util';

describe('tableData', () => {
  const testDistributions: [string, Distribution][] = [
    [
      'output 1',
      {
        probabilities: [
          [2, 0.111111111111],
          [3, 0.222222222222],
          [4, 0.333333333333],
          [5, 0.222222222222],
          [6, 0.111111111111]
        ]
      }
    ],
    [
      'output 2',
      {
        probabilities: [
          [3, 0.037037037037],
          [4, 0.111111111111],
          [5, 0.222222222222],
          [6, 0.259259259259],
          [7, 0.222222222222],
          [8, 0.111111111111],
          [9, 0.037037037037]
        ]
      }
    ]
  ];

  describe('getAllUniqueOutcomes', () => {
    it('should return all unique outcomes sorted in ascending order', () => {
      const result = getAllUniqueOutcomes(testDistributions);
      expect(result).toEqual([2, 3, 4, 5, 6, 7, 8, 9]);
    });

    it('should handle empty distributions', () => {
      const result = getAllUniqueOutcomes([]);
      expect(result).toEqual([]);
    });

    it('should handle single distribution', () => {
      const singleDistribution: [string, Distribution][] = [
        ['test', { probabilities: [[1, 0.5], [3, 0.5]] }]
      ];
      const result = getAllUniqueOutcomes(singleDistribution);
      expect(result).toEqual([1, 3]);
    });

    it('should handle duplicate outcomes across distributions', () => {
      const duplicateDistributions: [string, Distribution][] = [
        ['dist1', { probabilities: [[1, 0.5], [2, 0.5]] }],
        ['dist2', { probabilities: [[2, 0.3], [3, 0.7]] }]
      ];
      const result = getAllUniqueOutcomes(duplicateDistributions);
      expect(result).toEqual([1, 2, 3]);
    });

    it('should sort outcomes numerically, not lexicographically', () => {
      const unsortedDistributions: [string, Distribution][] = [
        ['test', { probabilities: [[10, 0.3], [2, 0.4], [20, 0.3]] }]
      ];
      const result = getAllUniqueOutcomes(unsortedDistributions);
      expect(result).toEqual([2, 10, 20]);
    });
  });

  describe('computeTableData', () => {
    it('should compute table data for Distribution mode', () => {
      const outcomes = getAllUniqueOutcomes(testDistributions);
      const result = computeTableData(testDistributions, DisplayMode.Distribution, outcomes);

      expect(result).toHaveLength(8); // 8 unique outcomes
      
      // Check outcome 2 (exists only in output 1)
      const outcome2Row = result.find(row => row.outcome === 2);
      expect(outcome2Row?.values).toEqual(['11.11%', '-']);
      
      // Check outcome 3 (exists in both distributions)
      const outcome3Row = result.find(row => row.outcome === 3);
      expect(outcome3Row?.values).toEqual(['22.22%', '3.70%']);
      
      // Check outcome 9 (exists only in output 2)
      const outcome9Row = result.find(row => row.outcome === 9);
      expect(outcome9Row?.values).toEqual(['-', '3.70%']);
    });

    it('should compute table data for AtMost mode', () => {
      const outcomes = getAllUniqueOutcomes(testDistributions);
      const result = computeTableData(testDistributions, DisplayMode.AtMost, outcomes);

      // Check that cumulative probabilities are calculated correctly
      const outcome4Row = result.find(row => row.outcome === 4);
      // For output 1: outcomes 2,3,4 = 11.11% + 22.22% + 33.33% = 66.66%
      expect(outcome4Row?.values[0]).toBe('66.67%');
    });

    it('should compute table data for AtLeast mode', () => {
      const outcomes = getAllUniqueOutcomes(testDistributions);
      const result = computeTableData(testDistributions, DisplayMode.AtLeast, outcomes);

      // Check that reverse cumulative probabilities are calculated correctly
      const outcome5Row = result.find(row => row.outcome === 5);
      // For output 1: outcomes 5,6 = 22.22% + 11.11% = 33.33%
      expect(outcome5Row?.values[0]).toBe('33.33%');
    });

    it('should handle empty outcomes array', () => {
      const result = computeTableData(testDistributions, DisplayMode.Distribution, []);
      expect(result).toEqual([]);
    });

    it('should handle empty distributions', () => {
      const result = computeTableData([], DisplayMode.Distribution, [1, 2, 3]);
      expect(result).toHaveLength(3);
      result.forEach(row => {
        expect(row.values).toEqual([]);
      });
    });

    it('should format percentages to 2 decimal places', () => {
      const precisionDistributions: [string, Distribution][] = [
        ['test', { probabilities: [[1, 0.123456789]] }]
      ];
      const result = computeTableData(precisionDistributions, DisplayMode.Distribution, [1]);
      expect(result[0].values[0]).toBe('12.35%');
    });
  });

  describe('computeDistributionStatistics', () => {
    it('should compute correct statistics for each distribution', () => {
      const result = computeDistributionStatistics(testDistributions);

      expect(result).toHaveLength(2);
      
      // Check output 1 statistics
      const stats1 = result[0];
      expect(stats1.min).toBe('2');
      expect(stats1.max).toBe('6');
      // Mean: 2*0.111 + 3*0.222 + 4*0.333 + 5*0.222 + 6*0.111 = 4.0
      expect(parseFloat(stats1.mean)).toBeCloseTo(4.0, 1);
      // Standard deviation calculation
      expect(parseFloat(stats1.stdDev)).toBeGreaterThan(0);
      
      // Check output 2 statistics
      const stats2 = result[1];
      expect(stats2.min).toBe('3');
      expect(stats2.max).toBe('9');
      expect(parseFloat(stats2.mean)).toBeGreaterThan(5);
      expect(parseFloat(stats2.mean)).toBeLessThan(7);
    });

    it('should handle single outcome distribution', () => {
      const singleOutcome: [string, Distribution][] = [
        ['constant', { probabilities: [[5, 1.0]] }]
      ];
      const result = computeDistributionStatistics(singleOutcome);

      expect(result[0].mean).toBe('5.00');
      expect(result[0].stdDev).toBe('0.00');
      expect(result[0].min).toBe('5');
      expect(result[0].max).toBe('5');
    });

    it('should handle empty distributions', () => {
      const result = computeDistributionStatistics([]);
      expect(result).toEqual([]);
    });

    it('should format numbers to 2 decimal places', () => {
      const precisionDistribution: [string, Distribution][] = [
        ['test', { probabilities: [[1, 0.123456], [2, 0.876544]] }]
      ];
      const result = computeDistributionStatistics(precisionDistribution);

      // All statistics should be formatted to 2 decimal places
      expect(result[0].mean).toMatch(/^\d+\.\d{2}$/);
      expect(result[0].stdDev).toMatch(/^\d+\.\d{2}$/);
    });

    it('should calculate correct mean and variance', () => {
      // Simple distribution: [1, 2] with equal probabilities
      const simpleDistribution: [string, Distribution][] = [
        ['simple', { probabilities: [[1, 0.5], [2, 0.5]] }]
      ];
      const result = computeDistributionStatistics(simpleDistribution);

      // Mean should be 1.5
      expect(parseFloat(result[0].mean)).toBeCloseTo(1.5, 2);
      // Standard deviation should be 0.5
      expect(parseFloat(result[0].stdDev)).toBeCloseTo(0.5, 2);
    });
  });

  describe('calculateBracketingProbabilities', () => {
    it('should calculate correct bracketing probabilities', () => {
      const distribution = testDistributions[0][1]; // output 1
      const result = calculateBracketingProbabilities(distribution, 3, 5);

      // P(X < 3): outcome 2 = 11.11%
      expect(result.pLower).toBeCloseTo(11.11, 2);
      
      // P(3 <= X <= 5): outcomes 3,4,5 = 22.22% + 33.33% + 22.22% = 77.77%
      expect(result.pBetween).toBeCloseTo(77.77, 1);
      
      // P(X > 5): outcome 6 = 11.11%
      expect(result.pUpper).toBeCloseTo(11.11, 2);
    });

    it('should handle bounds that include all outcomes', () => {
      const distribution = testDistributions[0][1]; // output 1
      const result = calculateBracketingProbabilities(distribution, 1, 10);

      expect(result.pLower).toBe(0);
      expect(result.pBetween).toBeCloseTo(100, 2);
      expect(result.pUpper).toBe(0);
    });

    it('should handle bounds that exclude all outcomes', () => {
      const distribution = testDistributions[0][1]; // output 1
      const result = calculateBracketingProbabilities(distribution, -5, -1);

      expect(result.pLower).toBe(0);
      expect(result.pBetween).toBe(0);
      expect(result.pUpper).toBeCloseTo(100, 2);
    });

    it('should handle equal lower and upper bounds', () => {
      const distribution = testDistributions[0][1]; // output 1
      const result = calculateBracketingProbabilities(distribution, 4, 4);

      // P(X < 4): outcomes 2,3 = 11.11% + 22.22% = 33.33%
      expect(result.pLower).toBeCloseTo(33.33, 2);
      
      // P(4 <= X <= 4): outcome 4 = 33.33%
      expect(result.pBetween).toBeCloseTo(33.33, 2);
      
      // P(X > 4): outcomes 5,6 = 22.22% + 11.11% = 33.33%
      expect(result.pUpper).toBeCloseTo(33.33, 2);
    });

    it('should handle bounds between existing outcomes', () => {
      const distribution = testDistributions[0][1]; // output 1
      const result = calculateBracketingProbabilities(distribution, 2.5, 4.5);

      // P(X < 2.5): outcome 2 = 11.11%
      expect(result.pLower).toBeCloseTo(11.11, 2);
      
      // P(2.5 <= X <= 4.5): outcomes 3,4 = 22.22% + 33.33% = 55.55%
      expect(result.pBetween).toBeCloseTo(55.55, 1);
      
      // P(X > 4.5): outcomes 5,6 = 22.22% + 11.11% = 33.33%
      expect(result.pUpper).toBeCloseTo(33.33, 2);
    });

    it('should handle reversed bounds (lower > upper)', () => {
      const distribution = testDistributions[0][1]; // output 1
      const result = calculateBracketingProbabilities(distribution, 5, 3);

      // Should treat as if bounds were swapped: [3, 5]
      // But actually, the function treats this literally, so nothing falls in [5,3]
      expect(result.pBetween).toBe(0);
      expect(result.pLower + result.pUpper).toBeCloseTo(100, 2);
    });

    it('should return percentages (multiply by 100)', () => {
      const distribution: Distribution = {
        probabilities: [[1, 0.5], [2, 0.5]]
      };
      const result = calculateBracketingProbabilities(distribution, 1, 1);

      // Should return 50%, not 0.5
      expect(result.pBetween).toBe(50);
      expect(result.pLower).toBe(0);
      expect(result.pUpper).toBe(50);
    });
  });
});