<script lang="ts">
  import type { KnowledgeSynthesis } from '$lib/types/search';
  import { formatConfidence, formatScoreBar } from '$lib/utils/formatting';

  export let synthesis: KnowledgeSynthesis;

  const evidenceQualityColors = {
    strong: 'bg-green-600',
    moderate: 'bg-yellow-600',
    weak: 'bg-orange-600',
    insufficient: 'bg-red-600',
    conflicting: 'bg-purple-600'
  };

  const confidenceLevelColors = {
    high: 'bg-green-600',
    moderate: 'bg-yellow-600',
    low: 'bg-orange-600'
  };

  const severityColors = {
    critical: 'bg-red-600',
    moderate: 'bg-yellow-600',
    minor: 'bg-green-600'
  };
</script>

<div class="bg-gradient-to-r from-purple-900 to-purple-800 rounded-lg p-6 space-y-4">
  <h3 class="text-xl font-semibold text-white mb-4 flex items-center">
    🧠 Knowledge Synthesis
    {#if synthesis.overall_confidence}
      <span class="ml-2 text-sm text-gray-300">
        ({Math.round(synthesis.overall_confidence * 100)}% confidence)
      </span>
    {/if}
  </h3>
  
  <!-- Executive Summary -->
  <div class="bg-black bg-opacity-20 rounded-lg p-4">
    <h4 class="font-semibold text-purple-200 mb-3">Executive Summary</h4>
    <p class="text-gray-200">{synthesis.summary}</p>
  </div>

  <!-- Key Concepts -->
  {#if synthesis.key_concepts && synthesis.key_concepts.length > 0}
    <div class="concepts-section">
      <h4 class="font-semibold text-purple-200 mb-3">🔑 Key Concepts ({synthesis.key_concepts.length})</h4>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
        {#each synthesis.key_concepts.slice(0, 6) as concept}
          <div class="bg-black bg-opacity-20 rounded-lg p-3">
            <div class="flex items-start justify-between mb-2">
              <h5 class="font-medium text-white">{concept.concept}</h5>
              {#if concept.evidence_quality}
                <span class="px-2 py-1 text-xs rounded text-white {evidenceQualityColors[concept.evidence_quality]}">
                  {concept.evidence_quality}
                </span>
              {/if}
            </div>
            <p class="text-gray-300 text-sm mb-2">{concept.explanation}</p>
            {#if concept.importance}
              <p class="text-purple-200 text-xs">💡 {concept.importance}</p>
            {/if}
            {#if concept.confidence_score}
              <div class="mt-2 flex items-center space-x-2 text-xs">
                <span class="text-gray-400">Confidence:</span>
                <span class="font-mono text-green-400">{formatScoreBar(concept.confidence_score)}</span>
                <span class="text-gray-400">{Math.round(concept.confidence_score * 100)}%</span>
              </div>
            {/if}
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Synthesis Insights -->
  {#if synthesis.synthesis_insights && synthesis.synthesis_insights.length > 0}
    <div class="insights-section">
      <h4 class="font-semibold text-purple-200 mb-3">💡 Novel Insights ({synthesis.synthesis_insights.length})</h4>
      <div class="space-y-3">
        {#each synthesis.synthesis_insights.slice(0, 3) as insight}
          <div class="bg-black bg-opacity-20 rounded-lg p-3">
            <div class="flex items-start justify-between">
              <p class="text-white font-medium text-sm">{insight.insight}</p>
              {#if insight.confidence_level}
                <span class="px-2 py-1 text-xs rounded text-white {confidenceLevelColors[insight.confidence_level]}">
                  {insight.confidence_level}
                </span>
              {/if}
            </div>
            {#if insight.implications}
              <p class="text-purple-200 text-xs mt-1">📈 {insight.implications}</p>
            {/if}
            {#if insight.supporting_evidence}
              <p class="text-gray-400 text-xs mt-1">
                📚 {insight.supporting_evidence.length} supporting evidence points
              </p>
            {/if}
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Research Gaps -->
  {#if synthesis.research_gaps && synthesis.research_gaps.length > 0}
    <div class="gaps-section">
      <h4 class="font-semibold text-purple-200 mb-3">🔍 Research Gaps ({synthesis.research_gaps.length})</h4>
      <div class="space-y-3">
        {#each synthesis.research_gaps.slice(0, 3) as gap}
          <div class="bg-black bg-opacity-20 rounded-lg p-3">
            <div class="flex items-start justify-between">
              <p class="text-white font-medium text-sm">{gap.gap_description}</p>
              {#if gap.severity}
                <span class="px-2 py-1 text-xs rounded text-white {severityColors[gap.severity]}">
                  {gap.severity}
                </span>
              {/if}
            </div>
            {#if gap.suggested_investigation}
              <p class="text-purple-200 text-xs mt-1">🎯 {gap.suggested_investigation}</p>
            {/if}
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Topics -->
  {#if synthesis.topics && synthesis.topics.length > 0}
    <div class="topics-section">
      <h4 class="font-semibold text-purple-200 mb-3">📚 Key Topics</h4>
      <div class="flex flex-wrap gap-2">
        {#each synthesis.topics.slice(0, 8) as topic}
          <span class="px-3 py-1 bg-purple-700 text-purple-100 rounded-full text-sm">{topic}</span>
        {/each}
      </div>
    </div>
  {/if}
</div>

 