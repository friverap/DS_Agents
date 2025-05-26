import dspy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import sys
import traceback
import re
from typing import Dict, Any, List, Optional
from contextlib import redirect_stdout, redirect_stderr
import warnings
warnings.filterwarnings('ignore')

class CodeFormatterAgent(dspy.Signature):
    """Agent that formats and fixes Python code for data analysis"""
    
    raw_code = dspy.InputField(desc="Raw Python code that may have formatting issues")
    formatted_code = dspy.OutputField(desc="Clean, properly formatted Python code with correct syntax and line breaks")

class CodeExecutorAgent(dspy.Signature):
    """Agent that executes Python code and generates insights"""
    
    code = dspy.InputField(desc="Python code to execute")
    data_context = dspy.InputField(desc="Context about the data being analyzed")
    execution_results = dspy.OutputField(desc="Detailed insights and explanations based on code execution results")

class CodeFormatterModule(dspy.Module):
    """Module for formatting Python code"""
    
    def __init__(self):
        super().__init__()
        try:
            self.formatter = dspy.ChainOfThought(CodeFormatterAgent)
            self.use_ai = True
        except Exception as e:
            print(f"Warning: DSPy not configured for code formatting, using fallback: {e}")
            self.use_ai = False
    
    def forward(self, raw_code: str) -> str:
        """Format the given Python code"""
        if self.use_ai:
            try:
                result = self.formatter(raw_code=raw_code)
                return result.formatted_code
            except Exception as e:
                print(f"AI formatting failed, using fallback: {e}")
                return self._basic_format(raw_code)
        else:
            return self._basic_format(raw_code)
    
    def _basic_format(self, code: str) -> str:
        """Basic code formatting as fallback"""
        lines = code.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Fix common issues
                line = line.replace('=  =  =', '===')
                
                # Fix broken comments that lost their # symbol
                if line and not line.startswith('#') and not '=' in line and not '(' in line:
                    # Check if this looks like a comment
                    comment_patterns = [
                        r'^(Configuración|Crear|Calcular|Distribución|Análisis|Seleccionar|Gráfico|Box plot|Resumen)',
                        r'^\d+\.\s*[A-Za-z]',  # Numbered comments
                        r'^[A-Z][a-z]+\s+(de|del|por|para|con)',  # Spanish comment patterns
                    ]
                    
                    for pattern in comment_patterns:
                        if re.match(pattern, line):
                            line = '# ' + line
                            break
                
                # Fix concatenated print statements
                if 'print(' in line and line.count('print(') > 1:
                    # Split multiple print statements
                    parts = line.split('print(')
                    if len(parts) > 1:
                        formatted_lines.append(parts[0].strip())
                        for i, part in enumerate(parts[1:], 1):
                            if part:
                                formatted_lines.append(f'print({part}')
                    continue
                
                # Fix concatenated statements
                if ')' in line and not line.startswith('#'):
                    # Look for patterns like )statement
                    import re
                    pattern = r'(\))([a-zA-Z_][a-zA-Z0-9_]*\s*[=\(])'
                    if re.search(pattern, line):
                        parts = re.split(pattern, line)
                        current_line = ""
                        for j, part in enumerate(parts):
                            if j % 3 == 0:  # Main content
                                current_line += part
                            elif j % 3 == 1:  # Closing parenthesis
                                current_line += part
                                if current_line.strip():
                                    formatted_lines.append(current_line.strip())
                                current_line = ""
                            elif j % 3 == 2:  # Next statement
                                current_line = part
                        if current_line.strip():
                            formatted_lines.append(current_line.strip())
                        continue
                
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)

class CodeExecutorModule(dspy.Module):
    """Module for executing Python code and generating insights"""
    
    def __init__(self):
        super().__init__()
        try:
            self.executor = dspy.ChainOfThought(CodeExecutorAgent)
            self.use_ai_insights = True
        except Exception as e:
            print(f"Warning: DSPy not configured for insights generation, using fallback: {e}")
            self.use_ai_insights = False
        
        self.formatter = CodeFormatterModule()
    
    def execute_code(self, code: str, data_context: str = "") -> Dict[str, Any]:
        """Execute Python code safely and capture results"""
        
        # First, format the code
        formatted_code = self.formatter.forward(code)
        
        # Prepare execution environment
        execution_globals = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            '__builtins__': __builtins__
        }
        
        # Capture output
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        plots = []
        
        execution_result = {
            'success': False,
            'output': '',
            'error': None,
            'plots': [],
            'formatted_code': formatted_code,
            'variables_created': [],
            'insights': ''
        }
        
        try:
            # Redirect output
            with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                # Execute the code
                exec(formatted_code, execution_globals)
                
                # Check for matplotlib figures
                if plt.get_fignums():
                    for fig_num in plt.get_fignums():
                        plt.figure(fig_num)
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                        buf.seek(0)
                        img_str = base64.b64encode(buf.read()).decode('utf-8')
                        plots.append(img_str)
                        plt.close()
            
            # Get captured output
            execution_result['output'] = output_buffer.getvalue()
            execution_result['plots'] = plots
            execution_result['success'] = True
            
            # Get variables that were created
            data_vars = []
            for var_name, var_value in execution_globals.items():
                if var_name not in ['pd', 'np', 'plt', 'sns', '__builtins__']:
                    if isinstance(var_value, pd.DataFrame):
                        data_vars.append(f"DataFrame '{var_name}' with shape {var_value.shape}")
                    elif isinstance(var_value, (list, tuple, np.ndarray)):
                        data_vars.append(f"Array '{var_name}' with {len(var_value)} elements")
                    elif isinstance(var_value, (int, float, str)):
                        data_vars.append(f"Variable '{var_name}' = {var_value}")
            
            execution_result['variables_created'] = data_vars
            
        except Exception as e:
            execution_result['error'] = str(e)
            execution_result['output'] = output_buffer.getvalue()
            error_output = error_buffer.getvalue()
            if error_output:
                execution_result['error'] += f"\nStderr: {error_output}"
        
        return execution_result
    
    def generate_insights(self, execution_result: Dict[str, Any], data_context: str = "") -> str:
        """Generate insights based on execution results"""
        
        if not execution_result['success']:
            return f"**Error en la ejecución del código:**\n\n{execution_result['error']}\n\nPor favor revisa el código y vuelve a intentar."
        
        # Prepare context for the AI
        context_info = []
        
        if execution_result['output']:
            context_info.append(f"**Salida del código:**\n{execution_result['output']}")
        
        if execution_result['variables_created']:
            context_info.append(f"**Variables creadas:**\n" + "\n".join(execution_result['variables_created']))
        
        if execution_result['plots']:
            context_info.append(f"**Gráficos generados:** {len(execution_result['plots'])} visualizaciones")
        
        context_text = "\n\n".join(context_info)
        
        if self.use_ai_insights:
            try:
                # Use AI to generate insights
                result = self.executor(
                    code=execution_result['formatted_code'],
                    data_context=f"{data_context}\n\nResultados de ejecución:\n{context_text}"
                )
                return result.execution_results
            except Exception as e:
                print(f"AI insights generation failed, using fallback: {e}")
                return self._generate_enhanced_insights(execution_result, data_context)
        else:
            return self._generate_enhanced_insights(execution_result, data_context)
    
    def _generate_enhanced_insights(self, execution_result: Dict[str, Any], data_context: str) -> str:
        """Generate comprehensive and detailed insights as fallback"""
        insights = ["## 🎯 Análisis de Datos Completado\n"]
        
        # Analyze the output for specific patterns
        output = execution_result.get('output', '')
        code = execution_result.get('formatted_code', '')
        
        # Extract key information from output
        dataset_info = self._extract_dataset_info(output)
        statistical_info = self._extract_statistical_info(output)
        missing_data_info = self._extract_missing_data_info(output)
        
        # Analyze code to understand what was done
        analysis_performed = self._analyze_code_operations(code)
        
        # Generate comprehensive insights
        if dataset_info:
            insights.append("### 📊 Información del Dataset")
            insights.extend(dataset_info)
            insights.append("")
        
        if analysis_performed:
            insights.append("### 🔍 Análisis Realizados")
            insights.extend(analysis_performed)
            insights.append("")
        
        if statistical_info:
            insights.append("### 📈 Hallazgos Estadísticos Clave")
            insights.extend(statistical_info)
            insights.append("")
        
        if missing_data_info:
            insights.append("### ⚠️ Calidad de los Datos")
            insights.extend(missing_data_info)
            insights.append("")
        
        # Generate visualization insights
        if execution_result['plots']:
            viz_insights = self._generate_visualization_insights(code, len(execution_result['plots']))
            insights.append("### 📊 Insights de Visualizaciones")
            insights.extend(viz_insights)
            insights.append("")
        
        # Generate business insights and recommendations
        business_insights = self._generate_business_insights(output, code, dataset_info)
        if business_insights:
            insights.append("### 💼 Implicaciones de Negocio")
            insights.extend(business_insights)
            insights.append("")
        
        # Generate specific recommendations
        recommendations = self._generate_specific_recommendations(output, code, dataset_info)
        insights.append("### 💡 Recomendaciones Específicas")
        insights.extend(recommendations)
        insights.append("")
        
        # Generate next steps
        next_steps = self._generate_next_steps(output, code, analysis_performed)
        insights.append("### 🎯 Próximos Pasos Recomendados")
        insights.extend(next_steps)
        
        return "\n".join(insights)
    
    def _extract_dataset_info(self, output: str) -> list:
        """Extract dataset information from output"""
        info = []
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            if 'Shape:' in line:
                shape_info = line.split('Shape:')[1].strip()
                rows, cols = shape_info.split(',')
                rows_num = rows.strip().split()[0]
                cols_num = cols.strip().split()[0]
                info.append(f"- **Tamaño del dataset:** {rows_num} filas y {cols_num} columnas")
                
                # Provide context about dataset size
                rows_int = int(rows_num)
                if rows_int < 1000:
                    info.append(f"  - Dataset pequeño ({rows_int} registros) - ideal para análisis exploratorio rápido")
                elif rows_int < 10000:
                    info.append(f"  - Dataset mediano ({rows_int} registros) - buen tamaño para análisis detallado")
                else:
                    info.append(f"  - Dataset grande ({rows_int} registros) - considera técnicas de muestreo para análisis iniciales")
                    
            elif 'columns' in line.lower() and 'numeric' in line.lower():
                info.append(f"- **Tipos de variables:** {line}")
            elif 'Memory usage' in line:
                info.append(f"- **Uso de memoria:** {line.split('Memory usage:')[1].strip()}")
        
        return info
    
    def _extract_statistical_info(self, output: str) -> list:
        """Extract statistical insights from output"""
        insights = []
        lines = output.split('\n')
        
        # Look for statistical patterns
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Look for correlation insights
            if 'correlation' in line.lower() and any(op in line for op in ['>', '<', '0.', '-0.']):
                insights.append(f"- **Correlación detectada:** {line}")
            
            # Look for distribution insights
            if any(stat in line.lower() for stat in ['mean', 'std', 'min', 'max', 'median']):
                if '.' in line and any(char.isdigit() for char in line):
                    insights.append(f"- **Estadística clave:** {line}")
            
            # Look for outlier indicators
            if 'outlier' in line.lower() or 'extreme' in line.lower():
                insights.append(f"- **Valores atípicos:** {line}")
        
        # Add interpretation if we found statistical data
        if insights:
            insights.insert(0, "**Análisis estadístico revela:**")
        
        return insights
    
    def _extract_missing_data_info(self, output: str) -> list:
        """Extract missing data insights"""
        insights = []
        lines = output.split('\n')
        
        missing_found = False
        for line in lines:
            if 'missing' in line.lower() or 'null' in line.lower() or 'nan' in line.lower():
                if any(char.isdigit() for char in line):
                    missing_found = True
                    if '%' in line:
                        percentage = line.split('%')[0].split()[-1]
                        try:
                            pct = float(percentage)
                            if pct > 50:
                                insights.append(f"- ⚠️ **Crítico:** {line} - Requiere atención inmediata")
                            elif pct > 20:
                                insights.append(f"- ⚠️ **Alto:** {line} - Considerar estrategias de imputación")
                            elif pct > 5:
                                insights.append(f"- ⚠️ **Moderado:** {line} - Evaluar impacto en análisis")
                            else:
                                insights.append(f"- ✅ **Bajo:** {line} - Impacto mínimo")
                        except:
                            insights.append(f"- **Datos faltantes:** {line}")
                    else:
                        insights.append(f"- **Datos faltantes:** {line}")
        
        if not missing_found:
            insights.append("- ✅ **Excelente:** No se detectaron valores faltantes significativos")
        
        return insights
    
    def _analyze_code_operations(self, code: str) -> list:
        """Analyze what operations were performed in the code"""
        operations = []
        
        if 'describe()' in code:
            operations.append("- 📊 **Análisis estadístico descriptivo** - Medidas de tendencia central y dispersión")
        if 'corr()' in code:
            operations.append("- 🔗 **Análisis de correlación** - Relaciones entre variables numéricas")
        if 'value_counts()' in code:
            operations.append("- 📋 **Análisis de frecuencias** - Distribución de valores categóricos")
        if 'isnull()' in code or 'isna()' in code:
            operations.append("- 🔍 **Análisis de calidad** - Detección de valores faltantes")
        if 'hist(' in code:
            operations.append("- 📈 **Histogramas** - Distribución de variables numéricas")
        if 'boxplot(' in code:
            operations.append("- 📦 **Diagramas de caja** - Detección de outliers y cuartiles")
        if 'heatmap(' in code:
            operations.append("- 🌡️ **Mapa de calor** - Visualización de correlaciones")
        if 'scatter(' in code:
            operations.append("- 🎯 **Gráficos de dispersión** - Relaciones bivariadas")
        if 'bar(' in code or 'barplot(' in code:
            operations.append("- 📊 **Gráficos de barras** - Comparación de categorías")
        
        return operations
    
    def _generate_visualization_insights(self, code: str, num_plots: int) -> list:
        """Generate insights about visualizations created"""
        insights = [f"- **Total de gráficos generados:** {num_plots} visualizaciones"]
        
        if 'hist(' in code:
            insights.append("- **Histogramas:** Permiten identificar la forma de la distribución (normal, sesgada, bimodal)")
            insights.append("  - Busca patrones como asimetría, múltiples picos o valores extremos")
        
        if 'boxplot(' in code:
            insights.append("- **Diagramas de caja:** Ideales para detectar outliers y comparar distribuciones")
            insights.append("  - Los puntos fuera de los 'bigotes' son posibles valores atípicos")
        
        if 'heatmap(' in code:
            insights.append("- **Mapa de calor de correlaciones:** Muestra relaciones lineales entre variables")
            insights.append("  - Valores cercanos a 1 o -1 indican correlaciones fuertes")
            insights.append("  - Colores intensos señalan relaciones importantes a investigar")
        
        if 'scatter(' in code:
            insights.append("- **Gráficos de dispersión:** Revelan relaciones no lineales y patrones complejos")
            insights.append("  - Busca tendencias, agrupaciones o patrones no obvios")
        
        return insights
    
    def _generate_business_insights(self, output: str, code: str, dataset_info: list) -> list:
        """Generate business-relevant insights"""
        insights = []
        
        # Analyze dataset size for business implications
        if dataset_info:
            for info in dataset_info:
                if 'filas' in info and 'columnas' in info:
                    if 'grande' in info:
                        insights.append("- **Volumen de datos:** El gran volumen de datos sugiere operaciones maduras con potencial para análisis avanzados")
                    elif 'pequeño' in info:
                        insights.append("- **Volumen de datos:** Dataset compacto permite análisis detallado y experimentación rápida")
        
        # Business insights based on analysis type
        if 'correlation' in code.lower():
            insights.append("- **Relaciones de negocio:** Las correlaciones identificadas pueden revelar drivers clave del negocio")
            insights.append("- **Oportunidades:** Variables altamente correlacionadas pueden indicar redundancias o sinergias")
        
        if 'missing' in output.lower():
            insights.append("- **Calidad operacional:** Los datos faltantes pueden indicar problemas en procesos de captura")
            insights.append("- **Riesgo:** La calidad de datos impacta directamente la confiabilidad de decisiones")
        
        if 'describe()' in code:
            insights.append("- **Benchmarking:** Las estadísticas descriptivas permiten comparar con estándares de la industria")
        
        return insights
    
    def _generate_specific_recommendations(self, output: str, code: str, dataset_info: list) -> list:
        """Generate specific, actionable recommendations"""
        recommendations = []
        
        # Data quality recommendations
        if 'missing' in output.lower() or 'null' in output.lower():
            recommendations.append("- **Acción inmediata:** Investigar las causas de los datos faltantes")
            recommendations.append("- **Estrategia:** Definir reglas de imputación basadas en el contexto del negocio")
            recommendations.append("- **Prevención:** Implementar validaciones en el punto de captura de datos")
        
        # Analysis depth recommendations
        if 'correlation' in code.lower():
            recommendations.append("- **Profundización:** Investigar las correlaciones más fuertes (>0.7) con análisis de causalidad")
            recommendations.append("- **Validación:** Confirmar correlaciones con conocimiento del dominio")
        
        if 'hist(' in code or 'boxplot(' in code:
            recommendations.append("- **Outliers:** Investigar valores atípicos - pueden ser errores o insights valiosos")
            recommendations.append("- **Segmentación:** Considerar análisis por subgrupos si hay patrones evidentes")
        
        # Technical recommendations
        recommendations.append("- **Documentación:** Registrar todos los hallazgos y decisiones tomadas")
        recommendations.append("- **Validación:** Contrastar resultados con stakeholders del negocio")
        recommendations.append("- **Automatización:** Considerar automatizar este análisis para datos futuros")
        
        return recommendations
    
    def _generate_next_steps(self, output: str, code: str, analysis_performed: list) -> list:
        """Generate specific next steps based on analysis"""
        steps = []
        
        # Based on what was analyzed
        if any('correlación' in analysis for analysis in analysis_performed):
            steps.append("- **Análisis de causalidad:** Investigar si las correlaciones implican relaciones causales")
            steps.append("- **Feature engineering:** Crear variables derivadas basadas en correlaciones fuertes")
        
        if any('frecuencias' in analysis for analysis in analysis_performed):
            steps.append("- **Análisis de segmentos:** Profundizar en las categorías más relevantes")
            steps.append("- **Balanceo:** Evaluar si hay desbalance en variables categóricas importantes")
        
        if any('outliers' in analysis for analysis in analysis_performed):
            steps.append("- **Investigación de outliers:** Determinar si son errores o casos especiales valiosos")
            steps.append("- **Tratamiento:** Decidir estrategia para valores atípicos (eliminar, transformar, mantener)")
        
        # General next steps
        steps.append("- **Modelado predictivo:** Si hay variables objetivo, considerar modelos de machine learning")
        steps.append("- **Dashboard:** Crear visualizaciones interactivas para stakeholders")
        steps.append("- **Monitoreo:** Establecer métricas para seguimiento continuo de la calidad de datos")
        steps.append("- **Validación externa:** Comparar hallazgos con fuentes externas o benchmarks de industria")
        
        return steps
    
    def forward(self, code: str, data_context: str = "") -> Dict[str, Any]:
        """Main method to execute code and generate insights"""
        
        # Execute the code
        execution_result = self.execute_code(code, data_context)
        
        # Generate insights
        insights = self.generate_insights(execution_result, data_context)
        execution_result['insights'] = insights
        
        return execution_result

# Global instance
code_executor = CodeExecutorModule()

def get_code_executor():
    """Get the global code executor instance"""
    return code_executor

def execute_and_analyze_code(code: str, data_context: str = "") -> Dict[str, Any]:
    """
    Execute Python code and return detailed analysis with insights.
    
    Args:
        code: Python code to execute
        data_context: Context about the data being analyzed
        
    Returns:
        Dictionary with execution results and insights
    """
    executor = get_code_executor()
    return executor.forward(code, data_context) 