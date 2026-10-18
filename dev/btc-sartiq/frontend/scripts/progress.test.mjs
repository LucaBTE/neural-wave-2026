import assert from 'node:assert/strict';
import {test} from 'node:test';
import {estimateGenerationProgress} from '../src/lib/generationProgress.ts';

test('distributes 0–90% uniformly across the first 40 seconds', () => {
  for (const [seconds, percent] of [[0,0],[10,22.5],[20,45],[30,67.5],[40,90]]) {
    for (const status of ['queued','generating','finishing']) {
      assert.equal(estimateGenerationProgress(seconds,status,seconds),percent);
    }
  }
});
test('calibrates the final portion to the remaining stages, below 100%', () => {
  const stages = ['generating','pressing','finishing','checking_quality','complete'];
  const values = stages.map(status=>estimateGenerationProgress(50,status,10));
  assert.deepEqual(values,[...values].sort((a,b)=>a-b));
  for (const status of ['queued',...stages]) {
    let previous=90;
    for (const seconds of [40,45,60,120,600]) {
      const value=estimateGenerationProgress(seconds,status,seconds-40);
      assert.ok(value>=previous && value<100); previous=value;
    }
  }
});
