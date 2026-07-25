import { describe, it, expect } from 'vitest';
import { generateSpreadsheetCSV, generateAnyDiceFormatCSV, escapeCSVField } from './csvExport';
import { NamedDistribution } from '../util';
import { scalarDistribution } from './testData';

describe('csvExport', () => {
  const testDistributions: NamedDistribution[] = [
    [
      'output 1',
      scalarDistribution([
          [2, 0.111111111111],
          [3, 0.222222222222],
          [4, 0.333333333333],
          [5, 0.222222222222],
          [6, 0.111111111111]
      ])
    ],
    [
      'output 2',
      scalarDistribution([
          [3, 0.037037037037],
          [4, 0.111111111111],
          [5, 0.222222222222],
          [6, 0.259259259259],
          [7, 0.222222222222],
          [8, 0.111111111111],
          [9, 0.037037037037]
      ])
    ]
  ];

  describe('generateSpreadsheetCSV', () => {
    it('should generate a numeric section with outcome header and distribution names', () => {
      const result = generateSpreadsheetCSV(testDistributions);
      const lines = result.split('\n');

      expect(lines[0]).toBe('Numeric outcomes');
      expect(lines[1]).toBe('Outcome,output 1,output 2');
    });

    it('should include all unique outcomes sorted', () => {
      const result = generateSpreadsheetCSV(testDistributions);
      const lines = result.split('\n');

      expect(lines[2].split(',')[0]).toBe('2');
      expect(lines[3].split(',')[0]).toBe('3');
      expect(lines[4].split(',')[0]).toBe('4');
      expect(lines[5].split(',')[0]).toBe('5');
      expect(lines[6].split(',')[0]).toBe('6');
      expect(lines[7].split(',')[0]).toBe('7');
      expect(lines[8].split(',')[0]).toBe('8');
      expect(lines[9].split(',')[0]).toBe('9');
    });

    it('should include probabilities for each distribution', () => {
      const result = generateSpreadsheetCSV(testDistributions);
      const lines = result.split('\n');

      // Check outcome 2 (exists in dist 1, not in dist 2)
      expect(lines[2]).toBe('2,0.111111111111,0');

      // Check outcome 3 (exists in both)
      expect(lines[3]).toBe('3,0.222222222222,0.037037037037');

      // Check outcome 9 (doesn't exist in dist 1, exists in dist 2)
      expect(lines[9]).toBe('9,0,0.037037037037');
    });

    it('separates incompatible output types into ordered blocks', () => {
      const mixed: NamedDistribution[] = [
        ['attack', scalarDistribution(
          [[0, 0.25], [1, 0.75]],
          { kind: 'enum', enumName: 'RESULT', labels: ['MISS', 'HIT'] },
        )],
        testDistributions[0],
        ['defend', scalarDistribution(
          [[1, 1]],
          { kind: 'enum', enumName: 'RESULT', labels: ['MISS', 'HIT'] },
        )],
        ['weather', scalarDistribution(
          [[0, 1]],
          { kind: 'enum', enumName: 'WEATHER', labels: ['SUN', 'RAIN'] },
        )],
      ];

      expect(generateSpreadsheetCSV(mixed)).toBe([
        'RESULT',
        'Outcome,attack,defend',
        'MISS,0.25,0',
        'HIT,0.75,1',
        '',
        'Numeric outcomes',
        'Outcome,output 1',
        '2,0.111111111111',
        '3,0.222222222222',
        '4,0.333333333333',
        '5,0.222222222222',
        '6,0.111111111111',
        '',
        'WEATHER',
        'Outcome,weather',
        'SUN,1',
        'RAIN,0',
      ].join('\n'));
    });

    it('uses explicit tuple field names as CSV columns', () => {
      const result = generateSpreadsheetCSV([['round', {
          fields: [{ kind: 'int' }, { kind: 'int' }],
          fieldNames: ['Attacker losses', 'Defender losses'],
          probabilities: [[[1, 2], 1]],
      }]]);

      expect(result).toBe([
        'round',
        'Attacker losses,Defender losses,Probability',
        '1,2,1',
      ].join('\n'));
    });
  });

  describe('generateAnyDiceFormatCSV', () => {
    it('should generate correct header for first distribution', () => {
      const result = generateAnyDiceFormatCSV(testDistributions);
      const lines = result.split('\n');
      
      // First line should have name, mean, stddev, min, max
      expect(lines[0]).toMatch(/^output 1,\d+\.?\d*,\d+\.?\d*,2,6$/);
      expect(lines[1]).toBe('#,%');
    });

    it('should calculate statistics correctly', () => {
      const result = generateAnyDiceFormatCSV([testDistributions[0]]);
      const lines = result.split('\n');
      
      const headerParts = lines[0].split(',');
      const mean = parseFloat(headerParts[1]);
      const min = parseInt(headerParts[3]);
      const max = parseInt(headerParts[4]);
      
      expect(mean).toBeCloseTo(3.999999999996, 5);
      expect(min).toBe(2);
      expect(max).toBe(6);
    });

    it('should include probabilities as percentages', () => {
      const result = generateAnyDiceFormatCSV([testDistributions[0]]);
      const lines = result.split('\n');
      
      expect(lines[2]).toBe('2,11.1111111111');
      expect(lines[3]).toBe('3,22.2222222222');
      expect(lines[4]).toBe('4,33.3333333333');
      expect(lines[5]).toBe('5,22.2222222222');
      expect(lines[6]).toBe('6,11.1111111111');
    });

    it('should separate distributions with blank lines', () => {
      const result = generateAnyDiceFormatCSV(testDistributions);
      const lines = result.split('\n');
      
      // Find the blank line between distributions
      let blankLineIndex = -1;
      for (let i = 0; i < lines.length; i++) {
        if (lines[i] === '') {
          blankLineIndex = i;
          break;
        }
      }
      
      expect(blankLineIndex).toBeGreaterThan(-1);
      expect(lines[blankLineIndex + 1]).toMatch(/^output 2/);
    });

    it('should handle multiple distributions correctly', () => {
      const result = generateAnyDiceFormatCSV(testDistributions);
      const lines = result.split('\n');
      
      // Should have both distribution headers
      expect(lines[0]).toMatch(/^output 1/);
      expect(lines.find(line => line.match(/^output 2/))).toBeDefined();
      
      // Should have #,% headers for both
      expect(lines.filter(line => line === '#,%').length).toBe(2);
    });

    it('omits categorical outputs', () => {
      const categorical: NamedDistribution = ['attack', scalarDistribution(
        [[0, 0.25], [1, 0.75]],
        { kind: 'enum', enumName: 'RESULT', labels: ['MISS', 'HIT'] },
      )];
      const result = generateAnyDiceFormatCSV([
        categorical,
        testDistributions[0],
      ]);

      expect(result).toMatch(/^output 1/);
      expect(result).not.toContain('attack');
      expect(result).not.toContain('RESULT');
    });
  });

  describe('escapeCSVField', () => {
    it('should not quote simple strings', () => {
      expect(escapeCSVField('simple')).toBe('simple');
      expect(escapeCSVField('output1')).toBe('output1');
    });

    it('should quote strings containing commas', () => {
      expect(escapeCSVField('output, test')).toBe('"output, test"');
    });

    it('should quote strings containing newlines', () => {
      expect(escapeCSVField('output\ntest')).toBe('"output\ntest"');
      expect(escapeCSVField('output\r\ntest')).toBe('"output\r\ntest"');
    });

    it('should escape double quotes by doubling them', () => {
      expect(escapeCSVField('output"test')).toBe('"output""test"');
      expect(escapeCSVField('a"b"c')).toBe('"a""b""c"');
    });

    it('should handle complex cases with multiple special characters', () => {
      expect(escapeCSVField('output "test", value')).toBe('"output ""test"", value"');
    });
  });

  describe('CSV generation with special characters in names', () => {
    const specialDistributions: NamedDistribution[] = [
      ['output "with quotes"', scalarDistribution([[1, 0.5], [2, 0.5]])],
      ['output, with comma', scalarDistribution([[1, 0.3], [2, 0.7]])],
    ];

    it('should handle special characters in generateSpreadsheetCSV', () => {
      const result = generateSpreadsheetCSV(specialDistributions);
      const lines = result.split('\n');

      expect(lines[1]).toBe('Outcome,"output ""with quotes""","output, with comma"');
    });

    it('should handle special characters in generateAnyDiceFormatCSV', () => {
      const result = generateAnyDiceFormatCSV(specialDistributions);
      const lines = result.split('\n');
      
      expect(lines[0]).toMatch(/^"output ""with quotes""",\d+\.?\d*,\d+\.?\d*,1,2$/);
      
      // Find the second distribution header
      const secondDistLine = lines.find(line => line.includes('output, with comma'));
      expect(secondDistLine).toMatch(/^"output, with comma",\d+\.?\d*,\d+\.?\d*,1,2$/);
    });
  });

  describe('tuple export', () => {
    const tuples: NamedDistribution[] = [
      ['joint', {
          fields: [
            { kind: 'int' as const },
            { kind: 'enum' as const, enumName: 'R', labels: ['MISS', 'HIT'] },
          ],
          probabilities: [
            [[1, 1], 0.4],
            [[1, 0], 0.1],
            [[2, 0], 0.5],
          ] as [number[], number][],
      }],
    ];

    it('appends a field-per-column block for each tuple', () => {
      const result = generateSpreadsheetCSV(tuples);
      const lines = result.split('\n');

      expect(lines[0]).toBe('joint');
      expect(lines[1]).toBe('Field 1,R,Probability');
      // Rows are emitted in lexicographic outcome order with enum labels.
      expect(lines[2]).toBe('1,MISS,0.1');
      expect(lines[3]).toBe('1,HIT,0.4');
      expect(lines[4]).toBe('2,MISS,0.5');
    });

    it('places tuple blocks after scalar blocks', () => {
      const result = generateSpreadsheetCSV([...testDistributions, ...tuples]);
      expect(result.indexOf('Numeric outcomes')).toBeLessThan(result.indexOf('joint'));
    });
  });
});
