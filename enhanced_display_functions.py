"""
Enhanced display functions for the Interactive RAG Shell.
Provides much better formatting and readability for synthesized knowledge output.

Add these functions to your interactive_rag.py or replace the existing display logic.
"""

def display_enhanced_synthesis(synthesis):
    """Display synthesized knowledge with enhanced formatting."""
    if not synthesis:
        print("⚠️  Knowledge synthesis not available")
        return
    
    print(f"\n🧠 Enhanced Knowledge Synthesis:")
    print("=" * 80)
    
    # Analysis quality overview
    quality = synthesis.get_analysis_quality_summary()
    print(f"📊 Analysis Quality: {quality['depth']} | Confidence: {quality['confidence']} | "
          f"Sources: {quality['num_sources']} | Concepts: {quality['num_concepts']}")
    
    if synthesis.overall_confidence:
        confidence_bar = "█" * int(synthesis.overall_confidence * 10) + "░" * (10 - int(synthesis.overall_confidence * 10))
        print(f"🎯 Confidence: [{confidence_bar}] {synthesis.overall_confidence:.1%}")
    
    print("-" * 80)
    
    # Summary
    print(f"📋 Executive Summary:")
    print(f"{synthesis.summary}")
    
    if synthesis.key_concepts:
        print(f"\n🔑 Key Concepts ({len(synthesis.key_concepts)}):")
        for i, concept in enumerate(synthesis.key_concepts[:5], 1):  # Show top 5
            evidence_icon = {
                "strong": "🟢",
                "moderate": "🟡", 
                "weak": "🟠",
                "insufficient": "🔴",
                "conflicting": "🟣"
            }.get(concept.evidence_quality.value if concept.evidence_quality else "insufficient", "⚪")
            
            confidence_text = f" ({concept.confidence_score:.1%})" if concept.confidence_score else ""
            
            print(f"\n  {i}. {evidence_icon} {concept.concept}{confidence_text}")
            print(f"     💡 {concept.explanation}")
            print(f"     🎯 {concept.importance}")
            
            if concept.controversies:
                print(f"     ⚠️  Controversies: {concept.controversies}")
            
            if concept.related_concepts:
                print(f"     🔗 Related: {', '.join(concept.related_concepts[:3])}")
    
    if synthesis.synthesis_insights:
        print(f"\n💡 Novel Synthesis Insights ({len(synthesis.synthesis_insights)}):")
        for i, insight in enumerate(synthesis.synthesis_insights[:3], 1):  # Show top 3
            confidence_icon = {"high": "🟢", "moderate": "🟡", "low": "🟠"}.get(
                insight.confidence_level, "⚪"
            )
            
            print(f"\n  {i}. {confidence_icon} {insight.insight}")
            if insight.implications:
                print(f"     📈 Implications: {insight.implications}")
            if insight.supporting_evidence:
                print(f"     📚 Evidence: {len(insight.supporting_evidence)} supporting points")
    
    if synthesis.research_gaps:
        print(f"\n🔍 Research Gaps Identified ({len(synthesis.research_gaps)}):")
        for i, gap in enumerate(synthesis.research_gaps[:3], 1):
            severity_icon = {"critical": "🔴", "moderate": "🟡", "minor": "🟢"}.get(
                gap.severity, "⚪"
            )
            
            print(f"\n  {i}. {severity_icon} {gap.gap_description}")
            if gap.suggested_investigation:
                print(f"     🎯 Suggested: {gap.suggested_investigation}")
    
    if synthesis.topics:
        print(f"\n📚 Key Topics ({len(synthesis.topics)}):")
        for topic in synthesis.topics[:6]:  # Show up to 6 topics
            print(f"  • {topic}")
    
    if synthesis.methodological_observations:
        print(f"\n🔬 Methodological Observations:")
        print(f"{synthesis.methodological_observations}")
    
    if synthesis.theoretical_implications:
        print(f"\n🏛️  Theoretical Implications:")
        print(f"{synthesis.theoretical_implications}")
    
    # Technical details (can be toggled)
    if hasattr(synthesis, 'synthesis_quality_indicators') and synthesis.synthesis_quality_indicators:
        print(f"\n📈 Technical Analysis Metrics:")
        indicators = synthesis.synthesis_quality_indicators
        for key, value in indicators.items():
            print(f"  • {key.replace('_', ' ').title()}: {value}")


def display_contexts_enhanced(contexts):
    """Display retrieved contexts with enhanced formatting."""
    if not contexts:
        print("📊 No contexts retrieved")
        return
    
    print(f"📊 Retrieved {len(contexts)} contexts")
    print("-" * 60)
    
    for i, ctx in enumerate(contexts[:5], 1):  # Show top 5
        initial_score = ctx.get('initial_score', 0)
        rerank_score = ctx.get('rerank_score')
        
        # Score display
        if rerank_score is not None:
            score_bar = "█" * int(rerank_score * 10) + "░" * (10 - int(rerank_score * 10))
            print(f"\n{i}. [{score_bar}] Rerank: {rerank_score:.3f} (Initial: {initial_score:.3f})")
        else:
            score_bar = "█" * int(initial_score * 10) + "░" * (10 - int(initial_score * 10))
            print(f"\n{i}. [{score_bar}] Score: {initial_score:.3f}")
        
        # Document info
        doc_id = ctx.get('metadata', {}).get('document_id', 'Unknown')
        source = ctx.get('metadata', {}).get('source', 'Unknown')
        print(f"   📄 Document: {doc_id}")
        if source != 'Unknown':
            print(f"   📁 Source: {source}")
        
        # Text preview
        text = ctx['text']
        if len(text) > 200:
            print(f"   📝 {text[:200]}...")
        else:
            print(f"   📝 {text}")


# Enhanced command methods - replace the existing ones in interactive_rag.py

def do_query_enhanced(self, query_text: str):
    """Enhanced query command with better display."""
    if not query_text.strip():
        print("Usage: query <text>")
        return
    
    try:
        print(f"\n🔍 Enhanced RAG Analysis for: '{query_text}'")
        print("=" * 80)
        
        # Run full RAG pipeline
        results = self.kb_api.search(
            query=query_text,
            filters=self.current_filters,
            synthesize=True
        )
        
        # Display contexts with enhanced formatting
        display_contexts_enhanced(results['contexts'])
        
        # Display enhanced synthesis
        if results.get('synthesis'):
            display_enhanced_synthesis(results['synthesis'])
        else:
            print("\n⚠️  Knowledge synthesis not available (Google API key required)")
        
        # Store results for further analysis
        self.last_results = {
            "query": query_text,
            "results": results
        }
        
        print("\n" + "=" * 80)
        print("✅ Analysis Complete! Use 'last' command to see full details.")
        
    except Exception as e:
        print(f"❌ Enhanced query failed: {str(e)}")


def do_analyze_enhanced(self, query_text: str):
    """Enhanced analyze command focusing on synthesis."""
    if not query_text.strip():
        print("Usage: analyze <text>")
        return
    
    try:
        print(f"\n🧠 Deep Knowledge Analysis for: '{query_text}'")
        print("=" * 80)
        
        # Search with synthesis focus
        results = self.kb_api.search(
            query=query_text,
            filters=self.current_filters,
            synthesize=True
        )
        
        print(f"📊 Analyzed {results['num_results']} contexts")
        
        if results.get('synthesis'):
            display_enhanced_synthesis(results['synthesis'])
            
            # Show additional technical details
            synthesis = results['synthesis']
            if hasattr(synthesis, 'analysis_timestamp'):
                print(f"\n⏰ Analysis completed at: {synthesis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if hasattr(synthesis, 'analysis_model'):
                print(f"🤖 Analysis model: {synthesis.analysis_model}")
        else:
            print("⚠️  Knowledge synthesis not available (Google API key required)")
        
    except Exception as e:
        print(f"❌ Enhanced analysis failed: {str(e)}")


"""
INTEGRATION INSTRUCTIONS:

1. Add the display functions above to your interactive_rag.py file

2. Replace your existing do_query method with this enhanced version:
   
   def do_query(self, query_text: str):
       return do_query_enhanced(self, query_text)

3. Replace your existing do_analyze method with this enhanced version:
   
   def do_analyze(self, query_text: str):
       return do_analyze_enhanced(self, query_text)

4. Update the synthesis display in other methods by replacing:
   
   # Old display code
   if results.get('synthesis'):
       synthesis = results['synthesis']
       print(f"Summary: {synthesis['summary'][:300]}...")
   
   # With enhanced display
   if results.get('synthesis'):
       display_enhanced_synthesis(results['synthesis'])

This will give you much more readable and informative output with:
- Progress bars for confidence scores
- Color-coded evidence quality indicators
- Structured display of insights and research gaps
- Better formatting and organization
- Technical metrics and analysis depth indicators
"""
