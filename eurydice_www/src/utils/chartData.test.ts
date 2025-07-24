import { describe, it, expect } from 'vitest';
import {
  DisplayMode,
  prepareChartData,
  ColorGenerator,
  partialSums,
} from './chartData';
import { Distribution } from '../util';

describe('chartData', () => {
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

  describe('partialSums', () => {
    it('should calculate cumulative sums forward', () => {
      const input = [10, 20, 30, 40];
      const result = partialSums(input, false);
      expect(result).toEqual([10, 30, 60, 100]);
    });

    it('should calculate cumulative sums backward', () => {
      const input = [10, 20, 30, 40];
      const result = partialSums(input, true);
      expect(result).toEqual([100, 90, 70, 40]);
    });

    it('should cap values at 100 to handle floating point errors', () => {
      const input = [50, 50, 0.0000001];
      const result = partialSums(input, false);
      expect(result).toEqual([50, 100, 100]);
    });

    it('should handle empty array', () => {
      const result = partialSums([], false);
      expect(result).toEqual([]);
    });
  });

  describe('ColorGenerator', () => {
    it('should generate valid rgba colors', () => {
      const generator = new ColorGenerator();
      const color = generator.nextColor();
      expect(color).toMatch(/^rgba\(\d+, \d+, \d+, 1\.0\)$/);
    });

    it('should generate different colors on subsequent calls', () => {
      const generator = new ColorGenerator();
      const color1 = generator.nextColor();
      const color2 = generator.nextColor();
      expect(color1).not.toBe(color2);
    });

    it('should generate colors with values between 0 and 255', () => {
      const generator = new ColorGenerator();
      const color = generator.nextColor();
      const matches = color.match(/rgba\((\d+), (\d+), (\d+), 1\.0\)/);
      expect(matches).toBeTruthy();
      if (matches) {
        const [, r, g, b] = matches;
        expect(parseInt(r)).toBeGreaterThanOrEqual(0);
        expect(parseInt(r)).toBeLessThanOrEqual(255);
        expect(parseInt(g)).toBeGreaterThanOrEqual(0);
        expect(parseInt(g)).toBeLessThanOrEqual(255);
        expect(parseInt(b)).toBeGreaterThanOrEqual(0);
        expect(parseInt(b)).toBeLessThanOrEqual(255);
      }
    });
  });

  describe('prepareChartData', () => {
    it('should create chart data for Distribution mode', () => {
      const result = prepareChartData(testDistributions, DisplayMode.Distribution);
      
      expect(result.labels).toEqual(['2', '3', '4', '5', '6', '7', '8', '9']);
      expect(result.datasets).toHaveLength(2);
      expect(result.datasets[0].label).toBe('output 1');
      expect(result.datasets[1].label).toBe('output 2');
    });

    it('should convert probabilities to percentages', () => {
      const result = prepareChartData(testDistributions, DisplayMode.Distribution);
      
      // Check first distribution's data
      const firstDataset = result.datasets[0];
      expect(firstDataset.data[0]).toBeCloseTo(11.1111111111, 5); // outcome 2
      expect(firstDataset.data[1]).toBeCloseTo(22.2222222222, 5); // outcome 3
      expect(firstDataset.data[2]).toBeCloseTo(33.3333333333, 5); // outcome 4
    });

    it('should handle AtMost mode with cumulative probabilities', () => {
      const result = prepareChartData(testDistributions, DisplayMode.AtMost);
      
      const firstDataset = result.datasets[0];
      // Should be cumulative sums
      expect(firstDataset.data[0]).toBeCloseTo(11.1111111111, 5); // 11.11%
      expect(firstDataset.data[1]).toBeCloseTo(33.3333333333, 5); // 11.11% + 22.22%
      expect(firstDataset.data[2]).toBeCloseTo(66.6666666666, 5); // 11.11% + 22.22% + 33.33%
    });

    it('should handle AtLeast mode with reverse cumulative probabilities', () => {
      const result = prepareChartData(testDistributions, DisplayMode.AtLeast);
      
      const firstDataset = result.datasets[0];
      // Should be reverse cumulative sums
      expect(firstDataset.data[4]).toBeCloseTo(11.1111111111, 5); // Last value (outcome 6)
      expect(firstDataset.data[3]).toBeCloseTo(33.3333333333, 5); // 11.11% + 22.22% (outcomes 5-6)
    });

    it('should handle empty distributions', () => {
      const result = prepareChartData([], DisplayMode.Distribution);
      
      expect(result.labels).toEqual([]);
      expect(result.datasets).toEqual([]);
    });

    it('should fill gaps with zero probabilities', () => {
      const sparseDistributions: [string, Distribution][] = [
        [
          'sparse',
          {
            probabilities: [
              [1, 0.5],
              [5, 0.5]
            ]
          }
        ]
      ];
      
      const result = prepareChartData(sparseDistributions, DisplayMode.Distribution);
      
      expect(result.labels).toEqual(['1', '2', '3', '4', '5']);
      expect(result.datasets[0].data).toEqual([50, 0, 0, 0, 50]);
    });
  });

  describe('prepareChartData (transposed)', () => {
    it('should create transposed chart data with distributions as x-axis', () => {
      const result = prepareChartData(testDistributions, DisplayMode.Transposed);
      
      expect(result.labels).toEqual(['output 1', 'output 2']);
      expect(result.datasets).toHaveLength(8); // 8 unique outcomes (2,3,4,5,6,7,8,9)
    });

    it('should create one dataset per unique outcome', () => {
      const result = prepareChartData(testDistributions, DisplayMode.Transposed);
      
      const outcomeLabels = result.datasets.map(d => d.label);
      expect(outcomeLabels).toEqual(['2', '3', '4', '5', '6', '7', '8', '9']);
    });

    it('should handle outcomes that exist in some distributions but not others', () => {
      const result = prepareChartData(testDistributions, DisplayMode.Transposed);
      
      // Find outcome 2 dataset (exists in output 1, not in output 2)
      const outcome2Dataset = result.datasets.find(d => d.label === '2');
      expect(outcome2Dataset?.data).toEqual([11.1111111111, 0]);
      
      // Find outcome 9 dataset (doesn't exist in output 1, exists in output 2)
      const outcome9Dataset = result.datasets.find(d => d.label === '9');
      expect(outcome9Dataset?.data).toEqual([0, 3.7037037037]);
    });

    it('should handle empty distributions', () => {
      const result = prepareChartData([], DisplayMode.Transposed);
      
      expect(result.labels).toEqual([]);
      expect(result.datasets).toEqual([]);
    });

    it('should sort outcomes numerically', () => {
      const unorderedDistributions: [string, Distribution][] = [
        [
          'test',
          {
            probabilities: [
              [10, 0.3],
              [2, 0.4],
              [5, 0.3]
            ]
          }
        ]
      ];
      
      const result = prepareChartData(unorderedDistributions, DisplayMode.Transposed);
      
      const outcomeLabels = result.datasets.map(d => d.label);
      expect(outcomeLabels).toEqual(['2', '5', '10']);
    });

    it('should convert probabilities to percentages', () => {
      const result = prepareChartData(testDistributions, DisplayMode.Transposed);
      
      // Find outcome 3 dataset (exists in both distributions)
      const outcome3Dataset = result.datasets.find(d => d.label === '3');
      expect(outcome3Dataset?.data[0]).toBeCloseTo(22.2222222222, 5); // output 1
      expect(outcome3Dataset?.data[1]).toBeCloseTo(3.7037037037, 5);  // output 2
    });

    it('should include borderColor and backgroundColor for each dataset', () => {
      const result = prepareChartData(testDistributions, DisplayMode.Transposed);
      
      result.datasets.forEach(dataset => {
        expect(dataset.borderColor).toBeDefined();
        expect(dataset.backgroundColor).toBeDefined();
        expect(typeof dataset.borderColor).toBe('string');
        expect(typeof dataset.backgroundColor).toBe('string');
      });
    });
  });

  describe('DisplayMode enum', () => {
    it('should have correct values', () => {
      expect(DisplayMode.Distribution).toBe(0);
      expect(DisplayMode.AtMost).toBe(1);
      expect(DisplayMode.AtLeast).toBe(2);
      expect(DisplayMode.Transposed).toBe(3);
    });
  });
});