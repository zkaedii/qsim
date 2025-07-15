#!/usr/bin/env python3
"""
📊 H_MODEL_Z STREAMLIT PERFORMANCE DASHBOARD 📊
Interactive visualization of parallel processing performance
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
import numpy as np
from datetime import datetime
import altair as alt

class PerformanceDashboard:
    """Interactive Streamlit dashboard for H_MODEL_Z performance visualization"""
    
    def __init__(self):
        self.data = self.load_performance_data()
        
    def load_performance_data(self):
        """Load all performance data for visualization"""
        data = {}
        
        try:
            with open('complete_performance_comparison.json', 'r') as f:
                data['performance_comparison'] = json.load(f)
        except FileNotFoundError:
            st.warning("Performance comparison data not found")
        
        try:
            with open('ultimate_speed_optimization_results.json', 'r') as f:
                data['ultimate_speed'] = json.load(f)
        except FileNotFoundError:
            st.warning("Ultimate speed data not found")
        
        try:
            with open('parallel_scaling_performance_results.json', 'r') as f:
                data['parallel_scaling'] = json.load(f)
        except FileNotFoundError:
            st.warning("Parallel scaling data not found")
        
        try:
            with open('claude_comprehensive_analysis.json', 'r') as f:
                data['claude_analysis'] = json.load(f)
        except FileNotFoundError:
            st.info("Claude analysis data not found - run claude_analysis_agent.py first")
        
        return data
    
    def create_performance_evolution_chart(self):
        """Create performance evolution visualization"""
        
        if 'performance_comparison' not in self.data:
            st.error("Performance comparison data not available")
            return
        
        perf_data = self.data['performance_comparison']['performance_evolution']
        
        # Prepare data for visualization
        methods = list(perf_data.keys())
        rps_values = list(perf_data.values())
        
        # Create DataFrame
        df = pd.DataFrame({
            'Method': methods,
            'RPS': rps_values,
            'Log_RPS': np.log10(rps_values)
        })
        
        # Sort by performance
        df = df.sort_values('RPS', ascending=True)
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=df['Method'],
            x=df['RPS'],
            orientation='h',
            marker=dict(
                color=df['RPS'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Tasks/Second")
            ),
            text=[f'{val:,.0f}' for val in df['RPS']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="🚀 H_MODEL_Z Performance Evolution",
            xaxis_title="Tasks per Second (RPS)",
            yaxis_title="Optimization Method",
            height=600,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        return df
    
    def create_latency_comparison_chart(self):
        """Create latency comparison visualization"""
        
        if 'ultimate_speed' not in self.data:
            st.error("Ultimate speed data not available")
            return
        
        bench_data = self.data['ultimate_speed']['benchmark_results']
        
        # Prepare latency data
        methods = []
        latencies = []
        
        for method, results in bench_data.items():
            methods.append(method.replace('_', ' ').title())
            latencies.append(results['processing_time_per_task_ns'])
        
        df = pd.DataFrame({
            'Method': methods,
            'Latency_ns': latencies
        })
        
        df = df.sort_values('Latency_ns')
        
        # Create scatter plot with log scale
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['Method'],
            y=df['Latency_ns'],
            mode='markers+lines',
            marker=dict(
                size=15,
                color=df['Latency_ns'],
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="Latency (ns)")
            ),
            line=dict(width=3),
            text=[f'{val:.0f} ns' for val in df['Latency_ns']],
            textposition='top center'
        ))
        
        fig.update_layout(
            title="⚡ Processing Latency Comparison",
            xaxis_title="Optimization Method",
            yaxis_title="Latency (nanoseconds)",
            yaxis_type="log",
            height=500
        )
        
        fig.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_scalability_analysis(self):
        """Create scalability analysis visualization"""
        
        if 'parallel_scaling' not in self.data:
            st.error("Parallel scaling data not available")
            return
        
        bench_data = self.data['parallel_scaling']['benchmark_results']
        
        # Extract scaling data
        thread_data = []
        process_data = []
        async_data = []
        
        for key, value in bench_data.items():
            if 'thread' in key and 'tasks_per_second' in value:
                task_count = int(key.split('_')[0])
                thread_data.append({
                    'Task_Count': task_count,
                    'RPS': value['tasks_per_second'],
                    'Method': 'ThreadPool'
                })
            elif 'process' in key and 'tasks_per_second' in value:
                task_count = int(key.split('_')[0])
                process_data.append({
                    'Task_Count': task_count,
                    'RPS': value['tasks_per_second'],
                    'Method': 'ProcessPool'
                })
            elif 'async' in key and 'tasks_per_second' in value:
                task_count = int(key.split('_')[0])
                async_data.append({
                    'Task_Count': task_count,
                    'RPS': value['tasks_per_second'],
                    'Method': 'AsyncAwait'
                })
        
        # Combine data
        scaling_df = pd.DataFrame(thread_data + process_data + async_data)
        
        if not scaling_df.empty:
            fig = px.line(
                scaling_df,
                x='Task_Count',
                y='RPS',
                color='Method',
                markers=True,
                title="📈 Scalability Analysis: Performance vs Task Count",
                log_y=True
            )
            
            fig.update_layout(
                xaxis_title="Task Count",
                yaxis_title="Tasks per Second (RPS)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def create_performance_radar_chart(self):
        """Create radar chart for performance characteristics"""
        
        if 'ultimate_speed' not in self.data:
            st.error("Ultimate speed data not available")
            return
        
        bench_data = self.data['ultimate_speed']['benchmark_results']
        
        # Normalize metrics for radar chart
        methods = []
        throughput_scores = []
        latency_scores = []
        efficiency_scores = []
        
        max_rps = max(results['tasks_per_second'] for results in bench_data.values())
        min_latency = min(results['processing_time_per_task_ns'] for results in bench_data.values())
        
        for method, results in bench_data.items():
            methods.append(method.replace('_', ' ').title())
            
            # Normalize scores (0-100)
            throughput_score = (results['tasks_per_second'] / max_rps) * 100
            latency_score = (min_latency / results['processing_time_per_task_ns']) * 100
            efficiency_score = (throughput_score + latency_score) / 2
            
            throughput_scores.append(throughput_score)
            latency_scores.append(latency_score)
            efficiency_scores.append(efficiency_score)
        
        # Create radar chart
        fig = go.Figure()
        
        for i, method in enumerate(methods):
            fig.add_trace(go.Scatterpolar(
                r=[throughput_scores[i], latency_scores[i], efficiency_scores[i], throughput_scores[i]],
                theta=['Throughput', 'Low Latency', 'Efficiency', 'Throughput'],
                fill='toself',
                name=method
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="🎯 Performance Characteristics Radar",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_system_metrics_dashboard(self):
        """Create system metrics visualization"""
        
        if 'performance_comparison' not in self.data:
            return
        
        system_specs = self.data['performance_comparison']['system_specifications']
        
        # Create system info cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🖥️ CPU Cores",
                value=f"{system_specs['cpu_cores']}"
            )
        
        with col2:
            st.metric(
                label="⚡ CPU Frequency",
                value=f"{system_specs['cpu_frequency_mhz']:.0f} MHz"
            )
        
        with col3:
            st.metric(
                label="💾 Memory",
                value=f"{system_specs['total_memory_gb']:.1f} GB"
            )
        
        with col4:
            if 'max_threads' in system_specs:
                st.metric(
                    label="🧵 Max Threads",
                    value=f"{system_specs['max_threads']}"
                )
    
    def create_claude_insights_section(self):
        """Create Claude insights visualization"""
        
        if 'claude_analysis' not in self.data:
            st.info("🧠 Run `python claude_analysis_agent.py` to generate Claude insights")
            return
        
        claude_data = self.data['claude_analysis']
        
        st.subheader("🧠 Claude Intelligence Insights")
        
        # Executive summary
        if 'executive_summary' in claude_data:
            summary = claude_data['executive_summary']
            
            st.success(f"**Key Achievement:** {summary['key_achievement']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Optimization Confidence:** {summary['optimization_confidence']}")
            with col2:
                st.info(f"**Enterprise Status:** {summary['enterprise_status']}")
        
        # Claude insights
        if 'claude_insights' in claude_data:
            insights = claude_data['claude_insights']
            
            # Key discoveries
            if 'key_discoveries' in insights:
                st.subheader("🔍 Key Discoveries")
                for discovery in insights['key_discoveries']:
                    st.write(f"• {discovery}")
            
            # Optimization recommendations
            if 'optimization_recommendations' in insights:
                st.subheader("🚀 Optimization Recommendations")
                for rec in insights['optimization_recommendations']:
                    st.write(f"• {rec}")
    
    def run_dashboard(self):
        """Run the complete Streamlit dashboard"""
        
        st.set_page_config(
            page_title="H_MODEL_Z Performance Dashboard",
            page_icon="🚀",
            layout="wide"
        )
        
        st.title("🚀 H_MODEL_Z Ultimate Performance Dashboard")
        st.markdown("### Interactive visualization of parallel processing and optimization results")
        
        # Sidebar navigation
        st.sidebar.title("📊 Navigation")
        sections = [
            "🏠 Overview",
            "📈 Performance Evolution", 
            "⚡ Latency Analysis",
            "📊 Scalability",
            "🎯 Performance Radar",
            "🧠 Claude Insights"
        ]
        
        selected_section = st.sidebar.selectbox("Select Section", sections)
        
        # System metrics at top
        st.subheader("💻 System Specifications")
        self.create_system_metrics_dashboard()
        
        st.divider()
        
        # Main content based on selection
        if selected_section == "🏠 Overview":
            st.subheader("📊 Performance Overview")
            
            if 'performance_comparison' in self.data:
                summary = self.data['performance_comparison']['optimization_achievements']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="🌟 Peak Performance",
                        value=f"{summary['peak_performance_rps']:,.0f}",
                        delta="RPS"
                    )
                
                with col2:
                    st.metric(
                        label="⚡ Fastest Processing",
                        value=f"{summary['fastest_per_task_ns']:.0f}",
                        delta="nanoseconds"
                    )
                
                with col3:
                    st.metric(
                        label="📈 Speed Improvement",
                        value=f"{summary['speed_improvement_factor']:.0f}x",
                        delta="vs baseline"
                    )
            
            st.info("💡 Navigate through different sections using the sidebar to explore detailed performance analysis")
        
        elif selected_section == "📈 Performance Evolution":
            st.subheader("🚀 Performance Evolution Analysis")
            df = self.create_performance_evolution_chart()
            
            if df is not None:
                st.subheader("📊 Performance Statistics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Top 3 Performers:**")
                    top_3 = df.nlargest(3, 'RPS')
                    for idx, row in top_3.iterrows():
                        st.write(f"• {row['Method']}: {row['RPS']:,.0f} RPS")
                
                with col2:
                    st.write("**Performance Range:**")
                    st.write(f"• Highest: {df['RPS'].max():,.0f} RPS")
                    st.write(f"• Lowest: {df['RPS'].min():,.0f} RPS")
                    st.write(f"• Improvement Factor: {df['RPS'].max() / df['RPS'].min():.1f}x")
        
        elif selected_section == "⚡ Latency Analysis":
            st.subheader("⚡ Processing Latency Analysis")
            self.create_latency_comparison_chart()
            
            st.info("💡 Lower latency values indicate faster per-task processing. Note the logarithmic scale.")
        
        elif selected_section == "📊 Scalability":
            st.subheader("📈 Scalability Analysis")
            self.create_scalability_analysis()
            
            st.info("💡 This chart shows how different methods scale with increasing task counts.")
        
        elif selected_section == "🎯 Performance Radar":
            st.subheader("🎯 Performance Characteristics")
            self.create_performance_radar_chart()
            
            st.info("💡 The radar chart compares normalized performance metrics across different optimization methods.")
        
        elif selected_section == "🧠 Claude Insights":
            self.create_claude_insights_section()
        
        # Footer
        st.divider()
        st.markdown("### 🎊 H_MODEL_Z Ultimate Performance Framework")
        st.markdown("**Enterprise-grade parallel processing with Claude-powered optimization**")
        
        # Performance summary
        if 'performance_comparison' in self.data:
            st.success("✅ All performance data loaded and analyzed successfully!")
        else:
            st.warning("⚠️ Some performance data missing - run benchmark scripts first")

def main():
    """Launch the Streamlit dashboard"""
    dashboard = PerformanceDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
