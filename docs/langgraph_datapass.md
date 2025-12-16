# 基于 LangGraph 的多智能体 SQL 生成与归因分析系统

## 完整实现

```python
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import pandas as pd
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import asyncio
from enum import Enum

# ===================== 状态定义 =====================
class AnalysisState(TypedDict):
    """多智能体协作状态"""
    # 输入相关
    user_query: str                    # 用户原始查询
    business_context: str              # 业务背景
    time_range: tuple                  # 时间范围
    
    # SQL智能体输出
    sql_query: str                     # 生成的SQL语句
    sql_explanation: str               # SQL解释说明
    query_parameters: Dict[str, Any]   # 查询参数
    
    # 数据查询结果
    raw_data: List[Dict[str, Any]]     # 原始查询数据
    data_summary: Dict[str, Any]       # 数据摘要统计
    data_quality_issues: List[str]     # 数据质量问题
    
    # 归因分析结果
    attribution_results: Dict[str, Any]  # 归因分析结果
    key_findings: List[str]            # 关键发现
    recommendations: List[str]         # 业务建议
    
    # 执行状态
    current_agent: str                 # 当前执行智能体
    execution_history: List[str]       # 执行历史
    errors: List[str]                  # 错误信息

# ===================== 智能体1: SQL生成与查询智能体 =====================
class SQLGeneratorAgent:
    """智能体1: SQL生成与数据查询"""
    
    def __init__(self, db_schema: Dict[str, Any]):
        """
        初始化SQL生成智能体
        
        Args:
            db_schema: 数据库模式定义
        """
        self.db_schema = db_schema
        self.table_info = self._extract_table_info()
    
    def _extract_table_info(self) -> Dict[str, Any]:
        """提取数据库表信息"""
        # 这里可以连接到实际数据库获取元数据
        # 简化示例：返回固定的表结构
        return {
            "sales": {
                "columns": ["id", "product_id", "sale_date", "amount", "region", "channel"],
                "primary_key": "id",
                "foreign_keys": {"product_id": "products.id"}
            },
            "products": {
                "columns": ["id", "name", "category", "price"],
                "primary_key": "id"
            },
            "users": {
                "columns": ["id", "name", "region", "segment"],
                "primary_key": "id"
            }
        }
    
    def generate_sql(self, state: AnalysisState) -> AnalysisState:
        """生成SQL查询语句"""
        print("🤖 [SQL生成智能体] 开始生成SQL查询...")
        
        user_query = state["user_query"]
        time_range = state["time_range"]
        business_context = state["business_context"]
        
        # 基于用户查询解析意图
        intent = self._parse_user_intent(user_query)
        print(f"    识别查询意图: {intent}")
        
        # 构建SQL查询
        sql_query = self._build_sql_query(intent, time_range, business_context)
        
        # 解释SQL逻辑
        explanation = self._explain_sql(sql_query)
        
        # 提取查询参数
        params = self._extract_query_parameters(sql_query)
        
        # 更新状态
        return {
            **state,
            "sql_query": sql_query,
            "sql_explanation": explanation,
            "query_parameters": params,
            "current_agent": "sql_generator",
            "execution_history": state["execution_history"] + ["SQL生成完成"]
        }
    
    def execute_query(self, state: AnalysisState) -> AnalysisState:
        """执行SQL查询"""
        print("🤖 [SQL执行智能体] 执行查询并获取数据...")
        
        # 这里应该是实际的数据库查询
        # 为演示目的，我们返回模拟数据
        raw_data = self._simulate_database_query(
            state["sql_query"],
            state["query_parameters"]
        )
        
        # 数据质量检查
        quality_issues = self._check_data_quality(raw_data)
        
        # 生成数据摘要
        summary = self._generate_data_summary(raw_data)
        
        # 更新状态
        return {
            **state,
            "raw_data": raw_data,
            "data_summary": summary,
            "data_quality_issues": quality_issues,
            "current_agent": "sql_executor",
            "execution_history": state["execution_history"] + ["数据查询完成"]
        }
    
    def _parse_user_intent(self, query: str) -> Dict[str, Any]:
        """解析用户查询意图"""
        # 实际项目中这里应该使用NLP模型
        intent_keywords = {
            "销售": "sales_analysis",
            "用户": "user_behavior",
            "产品": "product_performance",
            "趋势": "trend_analysis",
            "对比": "comparison_analysis"
        }
        
        intent = {
            "type": "sales_analysis",  # 默认类型
            "metrics": [],
            "dimensions": [],
            "filters": []
        }
        
        # 简单关键词匹配
        if "销售" in query:
            intent["metrics"].append("sales_amount")
        if "用户" in query:
            intent["dimensions"].append("user_segment")
        if "产品" in query:
            intent["dimensions"].append("product_category")
        if "趋势" in query:
            intent["type"] = "trend_analysis"
        if "对比" in query:
            intent["type"] = "comparison_analysis"
            
        return intent
    
    def _build_sql_query(self, intent: Dict, time_range: tuple, context: str) -> str:
        """构建SQL查询语句"""
        start_date, end_date = time_range
        
        # 基于意图构建查询
        if intent["type"] == "sales_analysis":
            sql = f"""
            SELECT 
                DATE(s.sale_date) as date,
                p.category as product_category,
                s.region,
                SUM(s.amount) as total_sales,
                COUNT(DISTINCT s.id) as transaction_count,
                AVG(s.amount) as avg_transaction_value
            FROM sales s
            JOIN products p ON s.product_id = p.id
            WHERE s.sale_date BETWEEN '{start_date}' AND '{end_date}'
            GROUP BY DATE(s.sale_date), p.category, s.region
            ORDER BY date DESC, total_sales DESC
            """
        elif intent["type"] == "trend_analysis":
            sql = f"""
            SELECT 
                DATE_TRUNC('week', s.sale_date) as week,
                p.category,
                SUM(s.amount) as weekly_sales,
                LAG(SUM(s.amount)) OVER (PARTITION BY p.category ORDER BY DATE_TRUNC('week', s.sale_date)) as previous_week_sales
            FROM sales s
            JOIN products p ON s.product_id = p.id
            WHERE s.sale_date BETWEEN '{start_date}' AND '{end_date}'
            GROUP BY week, p.category
            ORDER BY week
            """
        else:
            sql = f"""
            SELECT 
                s.*,
                p.name as product_name,
                p.category
            FROM sales s
            JOIN products p ON s.product_id = p.id
            WHERE s.sale_date BETWEEN '{start_date}' AND '{end_date}'
            LIMIT 100
            """
        
        return sql
    
    def _simulate_database_query(self, sql: str, params: Dict) -> List[Dict]:
        """模拟数据库查询"""
        print(f"    模拟执行SQL: {sql[:100]}...")
        
        # 生成模拟数据
        dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
        categories = ['电子产品', '家居用品', '服装', '食品']
        regions = ['华东', '华北', '华南', '华中']
        
        data = []
        for i, date in enumerate(dates[:30]):  # 生成30天数据
            for category in categories:
                for region in regions:
                    # 模拟销售数据
                    sales_amount = 1000 + (i * 100) + (hash(category) % 500) + (hash(region) % 300)
                    transaction_count = 10 + (i % 5) + (hash(category) % 3)
                    
                    data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'product_category': category,
                        'region': region,
                        'total_sales': sales_amount,
                        'transaction_count': transaction_count,
                        'avg_transaction_value': sales_amount / max(transaction_count, 1)
                    })
        
        return data
    
    def _check_data_quality(self, data: List[Dict]) -> List[str]:
        """检查数据质量"""
        issues = []
        
        if not data:
            issues.append("查询结果为空")
            return issues
        
        # 检查缺失值
        for i, row in enumerate(data[:10]):  # 检查前10行
            for key, value in row.items():
                if value is None:
                    issues.append(f"第{i}行，列{key}存在空值")
        
        # 检查数据一致性
        categories = set(row.get('product_category') for row in data)
        if len(categories) < 2:
            issues.append("数据类别单一，可能影响分析结果")
        
        return issues
    
    def _generate_data_summary(self, data: List[Dict]) -> Dict[str, Any]:
        """生成数据摘要"""
        if not data:
            return {}
        
        df = pd.DataFrame(data)
        
        return {
            "total_rows": len(data),
            "date_range": {
                "min": df['date'].min() if 'date' in df.columns else None,
                "max": df['date'].max() if 'date' in df.columns else None
            },
            "categories": df['product_category'].unique().tolist() if 'product_category' in df.columns else [],
            "regions": df['region'].unique().tolist() if 'region' in df.columns else [],
            "total_sales": df['total_sales'].sum() if 'total_sales' in df.columns else 0,
            "avg_daily_sales": df['total_sales'].mean() if 'total_sales' in df.columns else 0
        }

# ===================== 智能体2: 归因分析智能体 =====================
class AttributionAnalysisAgent:
    """智能体2: 数据归因分析"""
    
    def __init__(self, analysis_methods: List[str] = ["shap", "lime", "statistical"]):
        """
        初始化归因分析智能体
        
        Args:
            analysis_methods: 可用的分析方法列表
        """
        self.analysis_methods = analysis_methods
        self.method_weights = {
            "shap": 0.4,
            "lime": 0.3,
            "statistical": 0.3
        }
    
    def analyze_attribution(self, state: AnalysisState) -> AnalysisState:
        """执行归因分析"""
        print("🔍 [归因分析智能体] 开始分析数据...")
        
        raw_data = state["raw_data"]
        data_summary = state["data_summary"]
        user_query = state["user_query"]
        business_context = state["business_context"]
        
        if not raw_data:
            print("    警告：没有数据可供分析")
            return {
                **state,
                "attribution_results": {},
                "key_findings": ["数据为空，无法进行分析"],
                "current_agent": "attribution_analyzer",
                "execution_history": state["execution_history"] + ["归因分析完成（无数据）"]
            }
        
        # 转换数据格式
        df = pd.DataFrame(raw_data)
        
        # 执行多维归因分析
        attribution_results = self._perform_multi_dimensional_analysis(df, user_query)
        
        # 识别关键发现
        key_findings = self._extract_key_findings(attribution_results, business_context)
        
        # 生成业务建议
        recommendations = self._generate_recommendations(key_findings)
        
        # 更新状态
        return {
            **state,
            "attribution_results": attribution_results,
            "key_findings": key_findings,
            "recommendations": recommendations,
            "current_agent": "attribution_analyzer",
            "execution_history": state["execution_history"] + ["归因分析完成"]
        }
    
    def _perform_multi_dimensional_analysis(self, df: pd.DataFrame, user_query: str) -> Dict[str, Any]:
        """执行多维归因分析"""
        results = {
            "dimension_contribution": {},
            "trend_analysis": {},
            "anomaly_detection": {},
            "correlation_analysis": {}
        }
        
        # 1. 维度贡献度分析
        if 'product_category' in df.columns and 'total_sales' in df.columns:
            category_contribution = df.groupby('product_category')['total_sales'].agg(['sum', 'mean', 'count']).to_dict()
            results["dimension_contribution"]["by_category"] = category_contribution
        
        if 'region' in df.columns and 'total_sales' in df.columns:
            region_contribution = df.groupby('region')['total_sales'].agg(['sum', 'mean', 'count']).to_dict()
            results["dimension_contribution"]["by_region"] = region_contribution
        
        # 2. 趋势分析
        if 'date' in df.columns and 'total_sales' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 周趋势
            weekly_trend = df['total_sales'].resample('W').sum().to_dict()
            results["trend_analysis"]["weekly"] = weekly_trend
            
            # 移动平均
            moving_avg = df['total_sales'].rolling(window=7).mean().to_dict()
            results["trend_analysis"]["moving_average_7d"] = moving_avg
        
        # 3. 异常检测
        if 'total_sales' in df.columns:
            # 使用Z-score检测异常
            mean_sales = df['total_sales'].mean()
            std_sales = df['total_sales'].std()
            
            anomalies = []
            for idx, row in df.iterrows():
                z_score = abs((row['total_sales'] - mean_sales) / std_sales) if std_sales > 0 else 0
                if z_score > 2:  # 阈值设为2个标准差
                    anomalies.append({
                        'index': idx,
                        'value': row['total_sales'],
                        'z_score': z_score,
                        'date': row.get('date', None),
                        'category': row.get('product_category', None)
                    })
            
            results["anomaly_detection"] = {
                "count": len(anomalies),
                "anomalies": anomalies[:10],  # 限制返回数量
                "threshold": 2.0
            }
        
        # 4. 相关性分析
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_columns) > 1:
            correlation_matrix = df[numeric_columns].corr().to_dict()
            results["correlation_analysis"] = correlation_matrix
        
        return results
    
    def _extract_key_findings(self, attribution_results: Dict, business_context: str) -> List[str]:
        """提取关键发现"""
        findings = []
        
        # 分析维度贡献度
        dim_contrib = attribution_results.get("dimension_contribution", {})
        
        if "by_category" in dim_contrib:
            category_sales = dim_contrib["by_category"].get("sum", {})
            if category_sales:
                # 找出销售额最高的类别
                max_category = max(category_sales.items(), key=lambda x: x[1])
                min_category = min(category_sales.items(), key=lambda x: x[1])
                
                findings.append(f"销售额最高的产品类别是：{max_category[0]}，占总销售额的{(max_category[1]/sum(category_sales.values()))*100:.1f}%")
                findings.append(f"销售额最低的产品类别是：{min_category[0]}，仅占{(min_category[1]/sum(category_sales.values()))*100:.1f}%")
        
        if "by_region" in dim_contrib:
            region_sales = dim_contrib["by_region"].get("sum", {})
            if region_sales:
                # 找出销售额最高的区域
                max_region = max(region_sales.items(), key=lambda x: x[1])
                findings.append(f"{max_region[0]}地区是主要销售贡献区域")
        
        # 分析趋势
        trend_analysis = attribution_results.get("trend_analysis", {})
        if "weekly" in trend_analysis:
            weekly_sales = list(trend_analysis["weekly"].values())
            if len(weekly_sales) >= 2:
                growth_rate = ((weekly_sales[-1] - weekly_sales[-2]) / weekly_sales[-2] * 100) if weekly_sales[-2] > 0 else 0
                if growth_rate > 10:
                    findings.append(f"最近一周销售额增长显著，环比增长{growth_rate:.1f}%")
                elif growth_rate < -5:
                    findings.append(f"最近一周销售额下降，环比下降{abs(growth_rate):.1f}%")
        
        # 异常检测结果
        anomaly_detection = attribution_results.get("anomaly_detection", {})
        anomaly_count = anomaly_detection.get("count", 0)
        if anomaly_count > 0:
            findings.append(f"检测到{anomaly_count}个销售异常点，建议进一步调查")
        
        return findings
    
    def _generate_recommendations(self, key_findings: List[str]) -> List[str]:
        """基于关键发现生成建议"""
        recommendations = []
        
        for finding in key_findings:
            if "销售额最高的产品类别" in finding:
                # 提取类别名称
                if "电子产品" in finding:
                    recommendations.append("建议加大对电子产品的营销投入，可考虑捆绑销售或限时折扣")
                elif "家居用品" in finding:
                    recommendations.append("家居用品需求稳定，建议优化供应链管理，确保库存充足")
            
            if "销售额最低的产品类别" in finding:
                if "食品" in finding:
                    recommendations.append("食品类别销售不佳，建议进行市场调研，调整产品组合或定价策略")
            
            if "地区" in finding and "主要销售贡献" in finding:
                recommendations.append("建议在主要销售贡献地区增加营销活动和渠道投入")
                recommendations.append("可考虑将成功地区的营销策略复制到其他地区")
            
            if "增长显著" in finding:
                recommendations.append("近期增长势头良好，建议加大投入，扩大市场份额")
            
            if "下降" in finding:
                recommendations.append("销售额出现下降趋势，建议进行市场调研，了解客户需求变化")
            
            if "异常点" in finding:
                recommendations.append("对销售异常点进行深入分析，识别是数据问题还是业务异常")
        
        # 通用建议
        recommendations.append("建议建立销售数据监控仪表板，实时跟踪关键指标")
        recommendations.append("定期进行归因分析，及时调整业务策略")
        
        return recommendations

# ===================== 构建协作图 =====================
def create_collaboration_graph() -> StateGraph:
    """创建多智能体协作图"""
    
    # 初始化智能体
    db_schema = {
        "host": "localhost",
        "database": "sales_db",
        "username": "admin"
    }
    sql_agent = SQLGeneratorAgent(db_schema)
    attribution_agent = AttributionAnalysisAgent()
    
    # 创建状态图
    graph = StateGraph(AnalysisState)
    
    # 添加节点
    graph.add_node("sql_generation", sql_agent.generate_sql)
    graph.add_node("query_execution", sql_agent.execute_query)
    graph.add_node("attribution_analysis", attribution_agent.analyze_attribution)
    
    # 设置执行流程
    graph.set_entry_point("sql_generation")
    graph.add_edge("sql_generation", "query_execution")
    graph.add_edge("query_execution", "attribution_analysis")
    graph.add_edge("attribution_analysis", END)
    
    # 可选：添加条件边用于错误处理
    def check_data_quality(state: AnalysisState) -> str:
        """检查数据质量，决定是否继续"""
        if state["data_quality_issues"] and len(state["data_quality_issues"]) > 3:
            return "needs_data_cleanup"
        return "attribution_analysis"
    
    # 添加条件路由
    graph.add_conditional_edges(
        "query_execution",
        check_data_quality,
        {
            "attribution_analysis": "attribution_analysis",
            "needs_data_cleanup": END  # 在实际应用中，可以添加数据清洗节点
        }
    )
    
    return graph.compile()

# ===================== 使用示例 =====================
class MultiAgentCollaborator:
    """多智能体协作系统主类"""
    
    def __init__(self):
        """初始化协作系统"""
        self.graph = create_collaboration_graph()
        self.execution_history = []
    
    def analyze_business_query(self, 
                              user_query: str, 
                              business_context: str = "",
                              time_range: tuple = ("2024-01-01", "2024-03-31")) -> Dict[str, Any]:
        """
        执行完整的分析流程
        
        Args:
            user_query: 用户查询
            business_context: 业务背景
            time_range: 时间范围
            
        Returns:
            分析结果
        """
        print("🚀 开始多智能体协作分析...")
        print(f"用户查询: {user_query}")
        print(f"时间范围: {time_range[0]} 至 {time_range[1]}")
        print("-" * 50)
        
        # 初始化状态
        initial_state = {
            "user_query": user_query,
            "business_context": business_context,
            "time_range": time_range,
            "sql_query": "",
            "sql_explanation": "",
            "query_parameters": {},
            "raw_data": [],
            "data_summary": {},
            "data_quality_issues": [],
            "attribution_results": {},
            "key_findings": [],
            "recommendations": [],
            "current_agent": "",
            "execution_history": [],
            "errors": []
        }
        
        try:
            # 执行图
            result = self.graph.invoke(initial_state)
            
            # 记录执行历史
            self.execution_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": user_query,
                "agents_executed": len(result["execution_history"]),
                "has_data": len(result["raw_data"]) > 0
            })
            
            return self._format_results(result)
            
        except Exception as e:
            print(f"❌ 分析过程中出现错误: {str(e)}")
            return {
                "error": str(e),
                "execution_state": "failed"
            }
    
    def _format_results(self, state: AnalysisState) -> Dict[str, Any]:
        """格式化分析结果"""
        return {
            "analysis_summary": {
                "query_executed": state["user_query"],
                "sql_generated": state["sql_query"][:200] + "..." if len(state["sql_query"]) > 200 else state["sql_query"],
                "data_points_analyzed": len(state["raw_data"]),
                "key_findings_count": len(state["key_findings"]),
                "recommendations_count": len(state["recommendations"])
            },
            "data_summary": state["data_summary"],
            "key_findings": state["key_findings"],
            "recommendations": state["recommendations"],
            "attribution_highlights": self._extract_attribution_highlights(state["attribution_results"]),
            "execution_details": {
                "agents_invoked": state["execution_history"],
                "data_quality_issues": state["data_quality_issues"],
                "final_agent": state["current_agent"]
            }
        }
    
    def _extract_attribution_highlights(self, attribution_results: Dict) -> Dict:
        """提取归因分析亮点"""
        highlights = {}
        
        # 维度贡献度
        dim_contrib = attribution_results.get("dimension_contribution", {})
        if "by_category" in dim_contrib:
            category_sales = dim_contrib["by_category"].get("sum", {})
            if category_sales:
                total = sum(category_sales.values())
                top_3 = sorted(category_sales.items(), key=lambda x: x[1], reverse=True)[:3]
                highlights["top_categories"] = [
                    {"category": cat, "sales": sales, "percentage": (sales/total)*100}
                    for cat, sales in top_3
                ]
        
        # 异常检测
        anomalies = attribution_results.get("anomaly_detection", {}).get("anomalies", [])
        if anomalies:
            highlights["anomalies_detected"] = len(anomalies)
            # 获取最大的异常值
            if anomalies:
                max_anomaly = max(anomalies, key=lambda x: abs(x.get('z_score', 0)))
                highlights["largest_anomaly"] = {
                    "value": max_anomaly.get('value'),
                    "z_score": max_anomaly.get('z_score'),
                    "date": max_anomaly.get('date')
                }
        
        return highlights

# ===================== 示例使用 =====================
def demonstrate_collaboration():
    """演示多智能体协作"""
    
    # 创建协作系统
    collaborator = MultiAgentCollaborator()
    
    # 示例1: 销售趋势分析
    print("\n📊 示例1: 销售趋势分析")
    print("=" * 60)
    
    result1 = collaborator.analyze_business_query(
        user_query="分析2024年第一季度各产品类别的销售趋势和表现",
        business_context="公司计划优化产品组合和区域策略",
        time_range=("2024-01-01", "2024-03-31")
    )
    
    print("\n📋 分析结果摘要:")
    print(f"- 分析数据点: {result1['analysis_summary']['data_points_analyzed']}")
    print(f"- 关键发现: {len(result1['key_findings'])} 条")
    print(f"- 业务建议: {len(result1['recommendations'])} 条")
    
    print("\n🔑 关键发现:")
    for i, finding in enumerate(result1['key_findings'], 1):
        print(f"  {i}. {finding}")
    
    print("\n💡 业务建议:")
    for i, recommendation in enumerate(result1['recommendations'][:5], 1):  # 只显示前5条
        print(f"  {i}. {recommendation}")
    
    # 示例2: 异常销售分析
    print("\n\n🔍 示例2: 异常销售检测")
    print("=" * 60)
    
    result2 = collaborator.analyze_business_query(
        user_query="识别近期销售异常并分析原因",
        business_context="需要监控销售波动，及时调整策略",
        time_range=("2024-03-01", "2024-03-31")
    )
    
    if "attribution_highlights" in result2 and "anomalies_detected" in result2["attribution_highlights"]:
        anomalies_count = result2["attribution_highlights"]["anomalies_detected"]
        print(f"\n检测到 {anomalies_count} 个销售异常点")
        
        if "largest_anomaly" in result2["attribution_highlights"]:
            anomaly = result2["attribution_highlights"]["largest_anomaly"]
            print(f"最大异常值: {anomaly.get('value', 0)} (Z-score: {anomaly.get('z_score', 0):.2f})")
    
    # 显示执行历史
    print("\n📈 系统执行历史:")
    for i, history in enumerate(collaborator.execution_history, 1):
        print(f"  执行{i}: {history['query'][:50]}... | 智能体数: {history['agents_executed']}")

# ===================== 高级功能：异步协作 =====================
class AsyncMultiAgentCollaborator:
    """异步多智能体协作系统"""
    
    def __init__(self):
        self.graph = create_collaboration_graph()
    
    async def analyze_streaming(self, user_query: str, callback=None):
        """流式分析，实时返回结果"""
        
        initial_state = {
            "user_query": user_query,
            "business_context": "",
            "time_range": ("2024-01-01", "2024-03-31"),
            "sql_query": "",
            "raw_data": [],
            "current_agent": "",
            "execution_history": [],
            "errors": []
        }
        
        # 模拟流式执行
        agents = ["sql_generation", "query_execution", "attribution_analysis"]
        
        current_state = initial_state
        for agent_name in agents:
            print(f"\n➡️ 执行智能体: {agent_name}")
            
            # 执行当前智能体
            current_state = await self._execute_agent_async(agent_name, current_state)
            
            # 回调处理
            if callback:
                await callback({
                    "agent": agent_name,
                    "state": current_state,
                    "progress": agents.index(agent_name) / len(agents)
                })
            
            # 添加延迟以模拟处理时间
            await asyncio.sleep(1)
        
        return current_state
    
    async def _execute_agent_async(self, agent_name: str, state: AnalysisState) -> AnalysisState:
        """异步执行智能体"""
        # 这里应该调用实际的异步执行
        # 简化示例：直接调用同步方法
        if agent_name == "sql_generation":
            sql_agent = SQLGeneratorAgent({})
            return sql_agent.generate_sql(state)
        elif agent_name == "query_execution":
            sql_agent = SQLGeneratorAgent({})
            return sql_agent.execute_query(state)
        elif agent_name == "attribution_analysis":
            attribution_agent = AttributionAnalysisAgent()
            return attribution_agent.analyze_attribution(state)
        return state

# ===================== 主程序 =====================
if __name__ == "__main__":
    print("🤝 多智能体SQL生成与归因分析系统")
    print("=" * 60)
    
    # 演示同步协作
    demonstrate_collaboration()
    
    # 演示异步协作（可选）
    async def demo_async():
        print("\n\n⚡ 异步协作演示")
        print("-" * 40)
        
        async def progress_callback(update):
            print(f"进度: {update['progress']*100:.0f}% - {update['agent']}")
        
        async_collaborator = AsyncMultiAgentCollaborator()
        result = await async_collaborator.analyze_streaming(
            "分析电子产品销售表现",
            callback=progress_callback
        )
        print(f"\n异步分析完成，获取 {len(result.get('raw_data', []))} 条数据")
    
    # 运行异步演示
    # asyncio.run(demo_async())
```

## 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                   用户查询                                  │
│                "分析销售趋势"                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               多智能体协作图 (StateGraph)                   │
├──────────────┬──────────────┬───────────────────────────────┤
│  智能体1      │  智能体1      │      智能体2                │
│  SQL生成      │ 数据查询      │     归因分析                │
│              │              │                              │
│  • 解析意图   │  • 执行查询   │  • 多维归因                 │
│  • 生成SQL    │  • 质量检查   │  • 趋势分析                │
│  • 参数提取   │  • 数据摘要   │  • 异常检测                │
│              │              │  • 业务建议                 │
└──────┬───────┴──────┬───────┴──────────────┬──────────────┘
       │               │                      │
       │   SQL查询     │   原始数据           │  分析结果
       ▼               ▼                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   共享状态 (AnalysisState)                  │
│                                                            │
│  user_query: "分析销售趋势"                                │
│  sql_query: "SELECT ..."                                   │
│  raw_data: [{...}, {...}, ...]                             │
│  attribution_results: {trends: {}, anomalies: {}}          │
│  key_findings: ["电子产品销售增长最快"]                    │
│  recommendations: ["加大电子产品营销"]                      │
└─────────────────────────────────────────────────────────────┘
```

## 关键特性

### 1. **智能体间数据传递机制**
```python
# 智能体1 → 智能体2 的数据传递
state = {
    # SQL智能体写入
    "sql_query": "SELECT ...",
    "raw_data": [...],
    
    # 归因智能体读取并处理
    "attribution_results": {...},
    
    # 双向通信
    "execution_history": ["SQL生成完成", "数据查询完成", "归因分析完成"]
}
```

### 2. **状态驱动协作**
- **统一状态管理**：所有智能体共享同一状态对象
- **数据完整性**：每个智能体只修改自己负责的部分
- **执行追踪**：完整记录每个智能体的执行历史

### 3. **错误处理与质量检查**
```python
# 数据质量检查
data_quality_issues = [
    "第5行列region存在空值",
    "数据类别单一，可能影响分析结果"
]

# 条件路由：基于质量决定是否继续
if len(data_quality_issues) > 3:
    return "needs_data_cleanup"  # 转向数据清洗
else:
    return "attribution_analysis"  # 继续分析
```

### 4. **可扩展架构**
```python
# 可以轻松添加新智能体
graph.add_node("data_visualization", visualization_agent.generate_charts)
graph.add_node("report_generation", report_agent.generate_report)

# 修改执行流程
graph.add_edge("attribution_analysis", "data_visualization")
graph.add_edge("data_visualization", "report_generation")
```

## 实际应用场景

### 场景1：电商销售分析
```python
collaborator = MultiAgentCollaborator()
result = collaborator.analyze_business_query(
    user_query="分析双十一期间各品类销售表现，找出增长驱动因素",
    business_context="准备明年双十一营销策略",
    time_range=("2023-11-01", "2023-11-15")
)
```

### 场景2：金融风险监控
```python
result = collaborator.analyze_business_query(
    user_query="检测异常交易模式，分析风险因素",
    business_context="反洗钱监控系统",
    time_range=("2024-01-01", "2024-01-31")
)
```

### 场景3：生产质量分析
```python
result = collaborator.analyze_business_query(
    user_query="分析产品质量缺陷的原因，识别关键影响因素",
    business_context="提升产品质量，降低退货率",
    time_range=("2024-02-01", "2024-02-29")
)
```

这个系统展示了如何通过LangGraph实现智能体间的有效协作，其中：
1. **SQL生成智能体**负责理解需求并获取数据
2. **归因分析智能体**负责深度分析和洞察发现
3. **共享状态**作为通信桥梁，确保数据完整传递
4. **条件路由**实现智能的错误处理和流程控制

这种设计模式可以扩展到更复杂的多智能体系统，添加更多专业智能体（如预测智能体、优化智能体等），构建完整的数据分析流水线。